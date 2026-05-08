"""Stationary detection — emits a stationary flag when the user is motionless.

Tracks normalized x/y positions of 12 key landmarks across a rolling window;
declares stationary when the per-landmark range over the window is below a
configurable threshold and overall visibility is healthy. The signal is used
purely for UX (telling the user "hold still" vs "we see motion") and does
not affect exercise classification or rep counting.
"""

from collections import deque
from typing import Optional
import numpy as np

from config.settings import settings


# MediaPipe indices for the 12 key landmarks used for motion sensing.
# Shoulders, elbows, wrists, hips, knees, ankles.
_KEY_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class MotionDetector:
    """Rolling-window motion detector. O(window_size) per update."""

    def __init__(
        self,
        window_frames: Optional[int] = None,
        threshold: Optional[float] = None,
        min_mean_visibility: float = 0.5,
    ):
        self._window = window_frames or settings.STATIONARY_WINDOW_FRAMES
        self._threshold = threshold or settings.STATIONARY_THRESHOLD
        self._min_mean_visibility = min_mean_visibility
        # Buffer of (12, 2) arrays — x/y of each key landmark per frame.
        self._positions: deque[np.ndarray] = deque(maxlen=self._window)
        self._visibilities: deque[float] = deque(maxlen=self._window)

    def update(self, smoothed_xyz: np.ndarray, visibility: np.ndarray) -> bool:
        """
        Append a frame and return whether the body is stationary now.

        Args:
            smoothed_xyz: shape (33, 3) Kalman-smoothed coordinates.
            visibility: shape (33,) visibility scores.

        Returns:
            True if the rolling window declares the body stationary.
        """
        key_xy = smoothed_xyz[_KEY_LANDMARK_INDICES, :2]
        self._positions.append(key_xy.copy())
        self._visibilities.append(float(visibility[_KEY_LANDMARK_INDICES].mean()))

        if len(self._positions) < self._window:
            return False

        if np.mean(self._visibilities) < self._min_mean_visibility:
            return False

        stacked = np.stack(self._positions, axis=0)  # (window, 12, 2)
        # Per-landmark range across the window, then take the maximum.
        ranges = stacked.max(axis=0) - stacked.min(axis=0)  # (12, 2)
        max_range = float(ranges.max())
        return max_range < self._threshold

    def reset(self) -> None:
        self._positions.clear()
        self._visibilities.clear()
