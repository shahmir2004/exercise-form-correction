"""Application settings with environment variable support."""

import os
from typing import Any, ClassVar, Dict, Optional, List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_cors_origins() -> List[str]:
    """Parse CORS_ORIGINS from environment or return defaults."""
    env_value = os.environ.get('CORS_ORIGINS', '')
    if env_value:
        # Special case: allow all origins
        if env_value.strip() == '*':
            return ["*"]
        return [origin.strip() for origin in env_value.split(',') if origin.strip()]
    return [
        "http://localhost:3000", 
        "http://localhost:5173",
        "https://exercise-form-correction.vercel.app",
    ]


class Settings(BaseSettings):
    """Application configuration settings."""
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Upload settings
    UPLOAD_DIR: str = "./uploads"
    CHUNK_DIR: str = "./uploads/chunks"
    MAX_FILE_SIZE: int = 5 * 1024 * 1024 * 1024  # 5GB
    CHUNK_SIZE: int = 5 * 1024 * 1024  # 5MB

    # Exercise detection settings
    MOTION_BUFFER_SIZE: int = 60  # frames
    CONFIDENCE_THRESHOLD: float = 0.80  # 80%
    EXERCISE_SWITCH_DELAY: float = 2.0  # seconds
    # Exercise switching hysteresis (time-based + frame-based)
    EXERCISE_SWITCH_MIN_SECONDS: float = 0.5
    EXERCISE_SWITCH_MIN_FRAMES: int = 6
    EXERCISE_SWITCH_CONFIDENCE: float = 0.6
    SQUAT_RULE_GATE_CONFIDENCE: float = 0.72
    PUSHUP_HORIZONTAL_MIN_CONFIDENCE: float = 0.78
    BLOCK_SWITCH_ON_UNRELIABLE: bool = True

    # Stationary detection (motion variance over rolling window)
    STATIONARY_WINDOW_FRAMES: int = 30
    STATIONARY_THRESHOLD: float = 0.015  # Normalized image-space delta

    # Detailed per-frame detection logging. Set DETECTION_DEBUG_LOG=true to
    # emit one DEBUG line per frame (HMM posterior, EMAs, candidate, state).
    # Transition events (state change, exercise switch, rep completed) are
    # always logged at INFO regardless of this flag.
    DETECTION_DEBUG_LOG: bool = False
    # Confidence floor below which an active exercise module is dropped.
    # Above this floor we keep showing the detected exercise even during
    # symmetric crossover frames in alternating-arm exercises so the UI
    # doesn't flicker back to "Detecting…".
    DETECTION_STICKY_FLOOR: float = 0.3
    
    # Supabase settings (disabled for MVP)
    SUPABASE_ENABLED: bool = False
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None

    # Pipeline — Kalman filter
    KALMAN_PROCESS_NOISE: float = 1e-3
    KALMAN_MIN_VISIBILITY: float = 0.05

    # Pipeline — HMM
    HMM_TRANSITION_SELF_LOOP: float = 0.95
    HMM_TRANSITION_TO_IDLE: float = 0.02
    HMM_OBSERVATION_VARIANCE_SCALE: float = 1.0
    # Minimum HMM exercise_confidence to start running the active exercise
    # module (rep counter + form check). Below this, the system stays in
    # passive scanning. Lower values let reps count earlier; the rep counter's
    # own hysteresis filters spurious frames.
    MIN_CONFIDENCE_FOR_REPS: float = 0.4

    # Pipeline — ViolationAggregator
    VIOLATION_AGG_M: int = 4
    VIOLATION_AGG_N: int = 6
    VIOLATION_COOLDOWN_FRAMES: int = 15

    # Pipeline — ConfidenceComposer
    JOINT_IMPORTANCE_SQUAT: str = ""   # JSON dict override (empty = use defaults)
    JOINT_IMPORTANCE_PUSHUP: str = ""
    JOINT_IMPORTANCE_CURL: str = ""

    # Pipeline — Rate limiting
    MAX_FRAMES_PER_SECOND: int = 60
    MAX_CLIENT_ID_LENGTH: int = 80
    CORS_ALLOW_CREDENTIALS: bool = False
    CORS_ORIGIN_REGEX: Optional[str] = None

    # Pipeline — Rep detection (Savitzky-Golay + find_peaks).
    # Per-exercise overrides for the HysteresisRepCounter internals.
    # smooth_window must be odd and > polyorder. prominence is in degrees;
    # min_rep_frames is the minimum peak-to-peak distance (≈ 0.5s @ 20fps).
    # ClassVar so pydantic-settings doesn't treat it as an env-loaded field.
    REP_DETECTION: ClassVar[Dict[str, Dict[str, Any]]] = {
        "squat":               {"prominence": 22.0, "min_rep_frames": 8, "smooth_window": 7},
        "pushup":              {"prominence": 22.0, "min_rep_frames": 8, "smooth_window": 7},
        "bicep_curl":          {"prominence": 25.0, "min_rep_frames": 8,  "smooth_window": 7},
        "alternate_bicep_curl":{"prominence": 18.0, "min_rep_frames": 6,  "smooth_window": 7},
    }

    # Mid-video exercise switching. Allows handing off from one exercise to
    # another in the same session without resetting. The switch is allowed
    # when EITHER (a) the current rep counter has been idle/setup for
    # EXERCISE_SWITCH_IDLE_SECONDS, OR (b) the current exercise's signal has
    # collapsed below EXERCISE_DROP_THRESHOLD for EXERCISE_DROP_SECONDS while
    # the candidate exercise's confidence is rising.
    EXERCISE_SWITCH_IDLE_SECONDS: float = 0.75
    EXERCISE_DROP_THRESHOLD: float = 0.35
    EXERCISE_DROP_SECONDS: float = 1.0
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Get CORS origins from environment."""
        return get_cors_origins()

    @property
    def EFFECTIVE_CORS_ALLOW_CREDENTIALS(self) -> bool:
        """Wildcard CORS cannot be safely combined with credentials."""
        return self.CORS_ALLOW_CREDENTIALS and "*" not in self.CORS_ORIGINS
    
    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        """Accept deployment env labels commonly used for non-debug builds."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"debug", "development", "dev"}:
                return True
        return value


settings = Settings()
