"""Pose library export endpoints."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, conlist

from config.settings import settings
from exercises.registry import POSE_LIBRARY_LABELS
from pipeline.kalman import KalmanPoseTracker
from pipeline.features import FeatureExtractor
from pipeline.pose_embedder import embed_pose


router = APIRouter(prefix="/pose-library", tags=["pose-library"])


_ALLOWED_EXERCISES = POSE_LIBRARY_LABELS

_KEY_VISIBILITY = [11, 12, 23, 24]


class PoseLandmark(BaseModel):
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0


class PoseFrame(BaseModel):
    landmarks: conlist(PoseLandmark, min_length=33, max_length=33)
    timestamp: Optional[float] = None


class PoseLibraryRecordRequest(BaseModel):
    exercise: str = Field(..., description="Exercise label")
    frames: list[PoseFrame]
    max_frames: int = Field(180, ge=1, le=2000)
    min_visibility: float = Field(0.3, ge=0.0, le=1.0)
    append: bool = False


class PoseLibraryRecordResponse(BaseModel):
    status: str
    exercise: str
    frame_count: int
    embedding_count: int
    file_path: str
    download_url: str


class PoseLibraryListResponse(BaseModel):
    files: list[dict]


def _pose_library_dir() -> Path:
    directory = Path(settings.POSE_LIBRARY_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@router.get("", response_model=PoseLibraryListResponse)
async def list_pose_library():
    """List available pose library files."""
    library_dir = _pose_library_dir()
    files = []
    for path in sorted(library_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            embeddings = data.get("embeddings", [])
            files.append({
                "exercise": data.get("exercise", path.stem),
                "file": path.name,
                "embedding_count": len(embeddings),
                "updated_at": path.stat().st_mtime,
            })
        except (OSError, json.JSONDecodeError):
            continue

    return PoseLibraryListResponse(files=files)


@router.get("/{exercise}")
async def download_pose_library(exercise: str):
    """Download a pose library JSON file."""
    exercise = exercise.strip().lower()
    if exercise not in _ALLOWED_EXERCISES:
        raise HTTPException(status_code=400, detail="Unknown exercise label")

    path = _pose_library_dir() / f"{exercise}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pose library not found")

    return FileResponse(path, media_type="application/json", filename=path.name)


@router.delete("/{exercise}")
async def delete_pose_library(exercise: str):
    """Delete a pose library JSON file."""
    exercise = exercise.strip().lower()
    if exercise not in _ALLOWED_EXERCISES:
        raise HTTPException(status_code=400, detail="Unknown exercise label")

    path = _pose_library_dir() / f"{exercise}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pose library not found")

    path.unlink()
    return {"status": "deleted", "exercise": exercise}


@router.post("/record", response_model=PoseLibraryRecordResponse)
async def record_pose_library(request: PoseLibraryRecordRequest):
    """Convert raw pose frames into embeddings and save to disk."""
    exercise = request.exercise.strip().lower()
    if exercise not in _ALLOWED_EXERCISES:
        raise HTTPException(status_code=400, detail="Unknown exercise label")

    if not request.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    frames = request.frames
    if request.max_frames and len(frames) > request.max_frames:
        step = max(1, len(frames) // request.max_frames)
        frames = frames[::step]

    kalman = KalmanPoseTracker()
    extractor = FeatureExtractor()

    embeddings: list[list[float]] = []
    for frame in frames:
        arr = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in frame.landmarks],
            dtype=np.float32,
        )

        if not np.isfinite(arr).all():
            continue

        key_vis = float(np.mean(arr[_KEY_VISIBILITY, 3]))
        if key_vis < request.min_visibility:
            continue

        smoothed_xyz, uncertainty = kalman.update(arr)
        body_frame = extractor.extract(smoothed_xyz, uncertainty, arr[:, 3])
        embedding = embed_pose(body_frame.coords, body_frame.torso_length)
        embeddings.append(embedding.astype(np.float32).tolist())

    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid frames after filtering")

    library_dir = _pose_library_dir()
    output_path = library_dir / f"{exercise}.json"

    if request.append and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            existing_embeddings = existing.get("embeddings", [])
        except (OSError, json.JSONDecodeError):
            existing_embeddings = []
        embeddings = existing_embeddings + embeddings

    payload = {
        "exercise": exercise,
        "embeddings": embeddings,
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    return PoseLibraryRecordResponse(
        status="saved",
        exercise=exercise,
        frame_count=len(frames),
        embedding_count=len(embeddings),
        file_path=str(output_path),
        download_url=f"/api/pose-library/{exercise}",
    )
