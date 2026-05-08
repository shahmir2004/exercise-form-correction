"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from config.settings import settings
from api.routes import router as api_router
from api.upload import router as upload_router


# Configure detection logging.
# - 'detect' logger emits INFO events (state transitions, exercise switches,
#   reps, sessions) always — these surface in Render's runtime logs.
# - DEBUG per-frame lines only when DETECTION_DEBUG_LOG=true.
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
))
_detect_logger = logging.getLogger("detect")
_detect_logger.handlers = [_handler]
_detect_logger.setLevel(
    logging.DEBUG if settings.DETECTION_DEBUG_LOG else logging.INFO
)
_detect_logger.propagate = False


# Create FastAPI app
app = FastAPI(
    title="Exercise Form Correction API",
    description="Real-time exercise detection and form correction using MediaPipe pose estimation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=settings.EFFECTIVE_CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

# Mount uploads directory for serving videos
uploads_path = Path(settings.UPLOAD_DIR)
uploads_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Exercise Form Correction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
