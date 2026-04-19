"""File upload API with chunked upload support."""

import json
import hashlib
import shutil
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.settings import settings


router = APIRouter(prefix="/upload", tags=["upload"])

# Create upload directories
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
CHUNK_DIR = Path(settings.CHUNK_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


class InitUploadRequest(BaseModel):
    """Request to initialize a chunked upload."""
    filename: str
    total_size: int
    total_chunks: int
    file_hash: Optional[str] = None


class InitUploadResponse(BaseModel):
    """Response from upload initialization."""
    upload_id: str
    chunk_size: int


class ChunkUploadResponse(BaseModel):
    """Response from chunk upload."""
    chunk_index: int
    uploaded_chunks: int
    total_chunks: int
    progress: float


class UploadStatusResponse(BaseModel):
    """Response for upload status check."""
    upload_id: str
    uploaded_chunks: list[int]
    total_chunks: int
    progress: float
    status: str


class CompleteUploadResponse(BaseModel):
    """Response from upload completion."""
    status: str
    filename: str
    file_path: str
    size: int


@router.post("/init", response_model=InitUploadResponse)
async def init_upload(request: InitUploadRequest):
    """
    Initialize a chunked upload session.
    
    Creates a unique upload ID and session directory for chunk storage.
    """
    # Generate upload ID from filename and size
    upload_id = hashlib.md5(
        f"{request.filename}{request.total_size}{hash(request.file_hash or '')}".encode()
    ).hexdigest()
    
    # Create session directory
    session_dir = CHUNK_DIR / upload_id
    session_dir.mkdir(exist_ok=True)
    
    # Store session metadata
    metadata = {
        "upload_id": upload_id,
        "filename": request.filename,
        "total_size": request.total_size,
        "total_chunks": request.total_chunks,
        "uploaded_chunks": [],
        "file_hash": request.file_hash,
        "status": "in_progress"
    }
    
    async with aiofiles.open(session_dir / "metadata.json", "w") as f:
        await f.write(json.dumps(metadata))
    
    return InitUploadResponse(
        upload_id=upload_id,
        chunk_size=settings.CHUNK_SIZE
    )


@router.post("/chunk/{upload_id}", response_model=ChunkUploadResponse)
async def upload_chunk(
    upload_id: str,
    chunk_index: int,
    chunk: UploadFile = File(...)
):
    """
    Upload a single chunk of the file.
    
    Chunks are stored temporarily and assembled on completion.
    """
    session_dir = CHUNK_DIR / upload_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    # Save chunk
    chunk_path = session_dir / f"chunk_{chunk_index:06d}"
    async with aiofiles.open(chunk_path, "wb") as f:
        content = await chunk.read()
        await f.write(content)
    
    # Update metadata
    metadata_path = session_dir / "metadata.json"
    async with aiofiles.open(metadata_path, "r") as f:
        metadata = json.loads(await f.read())
    
    if chunk_index not in metadata["uploaded_chunks"]:
        metadata["uploaded_chunks"].append(chunk_index)
        metadata["uploaded_chunks"].sort()
    
    async with aiofiles.open(metadata_path, "w") as f:
        await f.write(json.dumps(metadata))
    
    progress = len(metadata["uploaded_chunks"]) / metadata["total_chunks"] * 100
    
    return ChunkUploadResponse(
        chunk_index=chunk_index,
        uploaded_chunks=len(metadata["uploaded_chunks"]),
        total_chunks=metadata["total_chunks"],
        progress=round(progress, 2)
    )


@router.get("/status/{upload_id}", response_model=UploadStatusResponse)
async def get_upload_status(upload_id: str):
    """
    Get the current status of an upload.
    
    Useful for resuming interrupted uploads.
    """
    session_dir = CHUNK_DIR / upload_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    async with aiofiles.open(session_dir / "metadata.json", "r") as f:
        metadata = json.loads(await f.read())
    
    progress = len(metadata["uploaded_chunks"]) / metadata["total_chunks"] * 100
    
    return UploadStatusResponse(
        upload_id=upload_id,
        uploaded_chunks=metadata["uploaded_chunks"],
        total_chunks=metadata["total_chunks"],
        progress=round(progress, 2),
        status=metadata["status"]
    )


@router.post("/complete/{upload_id}", response_model=CompleteUploadResponse)
async def complete_upload(upload_id: str, background_tasks: BackgroundTasks):
    """
    Complete the upload by assembling all chunks.
    
    Verifies all chunks are present, assembles the final file,
    and schedules cleanup of chunk files.
    """
    session_dir = CHUNK_DIR / upload_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    async with aiofiles.open(session_dir / "metadata.json", "r") as f:
        metadata = json.loads(await f.read())
    
    # Verify all chunks present
    if len(metadata["uploaded_chunks"]) != metadata["total_chunks"]:
        missing = set(range(metadata["total_chunks"])) - set(metadata["uploaded_chunks"])
        raise HTTPException(
            status_code=400, 
            detail=f"Missing chunks: {sorted(missing)}"
        )
    
    # Assemble file
    final_path = UPLOAD_DIR / metadata["filename"]
    
    # Handle duplicate filenames
    if final_path.exists():
        base = final_path.stem
        ext = final_path.suffix
        counter = 1
        while final_path.exists():
            final_path = UPLOAD_DIR / f"{base}_{counter}{ext}"
            counter += 1
    
    async with aiofiles.open(final_path, "wb") as final_file:
        for i in range(metadata["total_chunks"]):
            chunk_path = session_dir / f"chunk_{i:06d}"
            async with aiofiles.open(chunk_path, "rb") as chunk_file:
                await final_file.write(await chunk_file.read())
    
    # Update metadata
    metadata["status"] = "complete"
    metadata["final_path"] = str(final_path)
    async with aiofiles.open(session_dir / "metadata.json", "w") as f:
        await f.write(json.dumps(metadata))
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_chunks, session_dir)
    
    return CompleteUploadResponse(
        status="complete",
        filename=final_path.name,
        file_path=str(final_path),
        size=metadata["total_size"]
    )


@router.delete("/{upload_id}")
async def cancel_upload(upload_id: str):
    """Cancel an upload and clean up chunks."""
    session_dir = CHUNK_DIR / upload_id
    
    if session_dir.exists():
        shutil.rmtree(session_dir)
        return {"status": "cancelled", "upload_id": upload_id}
    
    return {"status": "not_found", "upload_id": upload_id}


async def cleanup_chunks(session_dir: Path):
    """Remove chunk files after assembly."""
    try:
        shutil.rmtree(session_dir)
    except Exception as e:
        print(f"Error cleaning up chunks: {e}")


@router.get("/files")
async def list_uploaded_files():
    """List all uploaded video files."""
    files = []
    for path in UPLOAD_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in [".mp4", ".mov", ".avi", ".webm", ".mkv"]:
            files.append({
                "name": path.name,
                "path": str(path),
                "size": path.stat().st_size
            })
    return {"files": files}
