"""Input validation for WebSocket landmark payloads."""

from dataclasses import dataclass, field
import numpy as np


class ValidationError(Exception):
    pass


@dataclass
class QualityFlags:
    off_screen_count: int = 0
    low_visibility_count: int = 0
    partial_body: bool = False


@dataclass
class ValidatedFrame:
    landmarks: np.ndarray  # shape (33, 4): x, y, z, visibility
    timestamp: float
    quality_flags: QualityFlags = field(default_factory=QualityFlags)


_KEY_JOINT_INDICES = [11, 12, 23, 24]  # shoulders + hips
_VIS_THRESHOLD = 0.3
_LOW_VIS_THRESHOLD = 0.5


class InputValidator:
    def __init__(self):
        self._last_timestamp: float = -1.0

    def validate(self, payload: dict) -> ValidatedFrame:
        if not isinstance(payload, dict):
            raise ValidationError("Payload must be a dict")

        landmarks_raw = payload.get("landmarks")
        if landmarks_raw is None:
            raise ValidationError("Missing 'landmarks' key")
        if not isinstance(landmarks_raw, list):
            raise ValidationError("'landmarks' must be a list")
        if len(landmarks_raw) != 33:
            raise ValidationError(f"Expected 33 landmarks, got {len(landmarks_raw)}")

        timestamp = payload.get("timestamp", 0.0)
        if not isinstance(timestamp, (int, float)):
            raise ValidationError("'timestamp' must be numeric")
        timestamp = float(timestamp)

        # Warn on non-monotonic timestamp (don't reject)
        if timestamp < self._last_timestamp:
            pass  # MediaPipe occasionally emits slightly out-of-order frames
        self._last_timestamp = timestamp

        arr = np.zeros((33, 4), dtype=np.float32)
        for i, lm in enumerate(landmarks_raw):
            if not isinstance(lm, dict):
                raise ValidationError(f"Landmark {i} must be a dict")
            try:
                x = float(lm["x"])
                y = float(lm["y"])
                z = float(lm.get("z", 0.0))
                vis = float(lm.get("visibility", 0.0))
            except (KeyError, TypeError, ValueError) as e:
                raise ValidationError(f"Landmark {i} invalid: {e}")

            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and np.isfinite(vis)):
                raise ValidationError(f"Landmark {i} contains NaN/Inf")
            if not (0.0 <= x <= 1.5 and 0.0 <= y <= 1.5):
                pass  # x/y can slightly exceed [0,1] at edges — don't reject
            vis = float(np.clip(vis, 0.0, 1.0))
            arr[i] = [x, y, z, vis]

        flags = QualityFlags()
        flags.off_screen_count = int(np.sum(arr[:, 3] < _VIS_THRESHOLD))
        flags.low_visibility_count = int(np.sum(arr[:, 3] < _LOW_VIS_THRESHOLD))
        key_vis = [arr[i, 3] for i in _KEY_JOINT_INDICES]
        flags.partial_body = sum(1 for v in key_vis if v < _LOW_VIS_THRESHOLD) >= 2

        return ValidatedFrame(landmarks=arr, timestamp=timestamp, quality_flags=flags)

    def reset(self):
        self._last_timestamp = -1.0
