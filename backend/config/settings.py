"""Application settings with environment variable support."""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS settings
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000", 
        "http://localhost:5173",
        "https://exercise-form-correction.vercel.app",
        "https://*.vercel.app",
    ]
    
    # Upload settings
    UPLOAD_DIR: str = "./uploads"
    CHUNK_DIR: str = "./uploads/chunks"
    MAX_FILE_SIZE: int = 5 * 1024 * 1024 * 1024  # 5GB
    CHUNK_SIZE: int = 5 * 1024 * 1024  # 5MB
    
    # Exercise detection settings
    MOTION_BUFFER_SIZE: int = 60  # frames
    CONFIDENCE_THRESHOLD: float = 0.80  # 80%
    EXERCISE_SWITCH_DELAY: float = 2.0  # seconds
    
    # Supabase settings (disabled for MVP)
    SUPABASE_ENABLED: bool = False
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
