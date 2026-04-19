from .routes import router as api_router
from .upload import router as upload_router

__all__ = ["api_router", "upload_router"]
