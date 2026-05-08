"""WebSocket API for real-time exercise form correction."""

import logging
import re
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from config.settings import settings
from exercises.registry import supported_exercises_payload
from state_machine.manager import FormManager, SystemState


router = APIRouter()
logger = logging.getLogger(__name__)
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


class FormCorrectionResponse(BaseModel):
    """Response sent back to client."""
    state: str  # "idle" | "stationary" | "scanning" | "active"
    current_exercise: Optional[str]
    exercise_display: str
    rep_count: int
    rep_phase: str  # "idle" | "setup" | "eccentric" | "concentric" | "hold"
    phase_display: str  # Human-readable per-exercise (e.g. "Lowering down")
    is_rep_valid: bool
    violations: list[str]
    corrections: list[str]
    correction_message: str
    joint_colors: dict[str, str]
    confidence: float  # Legacy alias of form_confidence
    is_stationary: bool
    timestamp: float
    exercise_confidence: float = 0.0
    form_confidence: float = 0.0
    signal_quality: str = "good"
    exercise_variant: Optional[str] = None
    exercise_source: str = "hmm"


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.form_managers: dict[str, FormManager] = {}
        self._last_frame_times: dict[str, float] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.form_managers[client_id] = FormManager()
        self._last_frame_times[client_id] = 0.0

    def disconnect(self, client_id: str) -> None:
        self.active_connections.pop(client_id, None)
        self.form_managers.pop(client_id, None)
        self._last_frame_times.pop(client_id, None)

    def get_manager(self, client_id: str) -> Optional[FormManager]:
        return self.form_managers.get(client_id)

    def should_rate_limit(self, client_id: str, max_fps: Optional[int] = None) -> bool:
        """Return True if this frame should be dropped (rate limit exceeded)."""
        max_fps = max_fps or settings.MAX_FRAMES_PER_SECOND
        now = time.time()
        last = self._last_frame_times.get(client_id, 0.0)
        if (now - last) < (1.0 / max_fps):
            return True
        self._last_frame_times[client_id] = now
        return False

    async def send_response(self, client_id: str, response: FormCorrectionResponse) -> None:
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_json(response.model_dump())


manager = ConnectionManager()


@router.websocket("/ws/pose/{client_id}")
async def pose_websocket(websocket: WebSocket, client_id: str):
    if (
        len(client_id) > settings.MAX_CLIENT_ID_LENGTH
        or not _CLIENT_ID_RE.fullmatch(client_id)
    ):
        await websocket.close(code=1008, reason="Invalid client_id")
        return

    await manager.connect(websocket, client_id)
    form_manager = manager.get_manager(client_id)

    try:
        while True:
            data = await websocket.receive_json()

            if not data.get("landmarks"):
                continue

            # Rate limit: drop excess frames silently
            if manager.should_rate_limit(client_id):
                continue

            landmarks = data["landmarks"]
            timestamp = data.get("timestamp", 0)

            state = form_manager.process_frame(landmarks)

            result = state.exercise_result
            response = FormCorrectionResponse(
                state=state.system_state.value,
                current_exercise=state.current_exercise.value if state.current_exercise else None,
                exercise_display=form_manager.get_state_display(),
                rep_count=form_manager.rep_count,
                rep_phase=result.rep_phase if result else "idle",
                phase_display=result.phase_display if result else "",
                is_rep_valid=result.is_valid if result else True,
                violations=result.violations if result else [],
                corrections=result.corrections if result else [],
                correction_message=_build_correction_message(result),
                joint_colors=result.joint_colors if result else {},
                confidence=state.form_confidence,
                is_stationary=state.is_stationary,
                timestamp=timestamp,
                exercise_confidence=state.exercise_confidence,
                form_confidence=state.form_confidence,
                signal_quality=state.signal_quality,
                exercise_variant=state.exercise_variant,
                exercise_source=state.exercise_source,
            )

            await manager.send_response(client_id, response)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception:
        logger.exception("WebSocket error for client_id=%s", client_id)
        manager.disconnect(client_id)


def _build_correction_message(result) -> str:
    if not result:
        return ""
    if not result.corrections:
        if result.is_valid:
            return "Great form! Keep it up!"
        return ""
    return result.corrections[0] if result.corrections else ""


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "connections": len(manager.active_connections),
        "supported_exercises": supported_exercises_payload(),
    }


@router.get("/exercises")
async def list_supported_exercises():
    """List exercise labels currently supported by the detection pipeline."""
    return {"exercises": supported_exercises_payload()}


@router.post("/reset/{client_id}")
async def reset_session(client_id: str):
    form_manager = manager.get_manager(client_id)
    if form_manager:
        form_manager.reset()
        return {"status": "reset", "client_id": client_id}
    return {"status": "not_found", "client_id": client_id}
