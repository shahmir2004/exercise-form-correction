"""WebSocket API for real-time exercise form correction."""

import json
import asyncio
from typing import Optional
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from state_machine.manager import FormManager, SystemState
from exercises.classifier import ExerciseType


router = APIRouter()


class LandmarkData(BaseModel):
    """Incoming landmark data from client."""
    landmarks: list[dict]
    timestamp: float


class FormCorrectionResponse(BaseModel):
    """Response sent back to client."""
    state: str
    current_exercise: Optional[str]
    exercise_display: str
    rep_count: int
    rep_phase: str
    is_rep_valid: bool
    violations: list[str]
    corrections: list[str]
    correction_message: str
    joint_colors: dict[str, str]
    confidence: float
    timestamp: float


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.form_managers: dict[str, FormManager] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.form_managers[client_id] = FormManager()
    
    def disconnect(self, client_id: str) -> None:
        """Remove a disconnected client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.form_managers:
            del self.form_managers[client_id]
    
    def get_manager(self, client_id: str) -> Optional[FormManager]:
        """Get the FormManager for a client."""
        return self.form_managers.get(client_id)
    
    async def send_response(self, client_id: str, response: FormCorrectionResponse) -> None:
        """Send a response to a specific client."""
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_json(response.model_dump())


manager = ConnectionManager()


@router.websocket("/ws/pose/{client_id}")
async def pose_websocket(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time pose analysis.
    
    Client sends: { landmarks: [...], timestamp: float }
    Server responds: FormCorrectionResponse
    """
    await manager.connect(websocket, client_id)
    form_manager = manager.get_manager(client_id)
    
    try:
        while True:
            # Receive landmark data
            data = await websocket.receive_json()
            
            if not data.get("landmarks"):
                continue
            
            landmarks = data["landmarks"]
            timestamp = data.get("timestamp", 0)
            
            # Process frame
            state = form_manager.process_frame(landmarks)
            
            # Build response
            response = FormCorrectionResponse(
                state=state.system_state.value,
                current_exercise=state.current_exercise.value if state.current_exercise else None,
                exercise_display=form_manager.get_state_display(),
                rep_count=form_manager.rep_count,
                rep_phase=state.exercise_result.rep_phase if state.exercise_result else "idle",
                is_rep_valid=state.exercise_result.is_valid if state.exercise_result else True,
                violations=state.exercise_result.violations if state.exercise_result else [],
                corrections=state.exercise_result.corrections if state.exercise_result else [],
                correction_message=_build_correction_message(state.exercise_result),
                joint_colors=state.exercise_result.joint_colors if state.exercise_result else {},
                confidence=state.motion_analysis.confidence if state.motion_analysis else 0.0,
                timestamp=timestamp
            )
            
            await manager.send_response(client_id, response)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


def _build_correction_message(result) -> str:
    """Build a user-friendly correction message."""
    if not result:
        return ""
    
    if not result.corrections:
        if result.is_valid:
            return "Great form! Keep it up!"
        return ""
    
    # Return the most important correction
    return result.corrections[0] if result.corrections else ""


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "connections": len(manager.active_connections)}


@router.post("/reset/{client_id}")
async def reset_session(client_id: str):
    """Reset the form manager for a client."""
    form_manager = manager.get_manager(client_id)
    if form_manager:
        form_manager.reset()
        return {"status": "reset", "client_id": client_id}
    return {"status": "not_found", "client_id": client_id}
