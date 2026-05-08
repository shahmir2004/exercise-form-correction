import pytest
from fastapi import HTTPException

from api.routes import list_supported_exercises
from api.upload import InitUploadRequest, _safe_filename, _session_dir, _validate_init_request
from config.settings import settings
from exercises.registry import SUPPORTED_EXERCISE_LABELS


@pytest.mark.asyncio
async def test_supported_exercises_endpoint_lists_current_detection_contract():
    payload = await list_supported_exercises()
    labels = {item["label"] for item in payload["exercises"]}

    assert labels == SUPPORTED_EXERCISE_LABELS
    assert "alternate_bicep_curl" in labels


def test_upload_filename_is_sanitized_to_basename_video_file():
    assert _safe_filename("../unsafe/my video.mp4") == "my video.mp4"
    assert _safe_filename("weird?name.webm") == "weird_name.webm"


def test_upload_filename_rejects_non_video_extension():
    with pytest.raises(HTTPException):
        _safe_filename("payload.exe")


def test_upload_init_rejects_chunk_count_mismatch():
    request = InitUploadRequest(
        filename="sample.mp4",
        total_size=settings.CHUNK_SIZE + 1,
        total_chunks=1,
        file_hash=None,
    )

    with pytest.raises(HTTPException):
        _validate_init_request(request)


def test_upload_session_id_rejects_path_traversal():
    with pytest.raises(HTTPException):
        _session_dir("../outside")
