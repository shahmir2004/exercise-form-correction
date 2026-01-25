"""
Pose smoothing utilities for stable detection.
Based on Stack Overflow best practices for MediaPipe pose estimation.

Implements:
- Exponential Moving Average (EMA) smoothing
- Kalman-like filtering for landmarks
- Velocity-based outlier rejection
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SmoothingConfig:
    """Configuration for smoothing parameters."""
    alpha: float = 0.4  # EMA smoothing factor (0.0-1.0, higher = less smoothing)
    velocity_threshold: float = 0.3  # Max velocity before considering outlier
    min_visibility: float = 0.5  # Minimum visibility to accept landmark
    history_size: int = 5  # Number of frames to keep for averaging


class ValueSmoother:
    """Smooths a single value over time using EMA."""
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.value: Optional[float] = None
        self.velocity: float = 0.0
        self._last_value: Optional[float] = None
    
    def update(self, new_value: float) -> float:
        """Update with new value and return smoothed result."""
        if self.value is None:
            self.value = new_value
            self._last_value = new_value
            return new_value
        
        # Calculate velocity
        self.velocity = new_value - self._last_value
        self._last_value = self.value
        
        # Apply EMA
        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        
        return self.value
    
    def reset(self):
        """Reset smoother state."""
        self.value = None
        self.velocity = 0.0
        self._last_value = None


class LandmarkSmoother:
    """
    Smooths all MediaPipe landmarks for stable pose detection.
    
    Features:
    - Per-landmark EMA smoothing
    - Velocity-based outlier rejection
    - Visibility-weighted updates
    """
    
    def __init__(self, config: Optional[SmoothingConfig] = None):
        self.config = config or SmoothingConfig()
        self._smoothers: dict[int, dict[str, ValueSmoother]] = {}
        self._frame_count = 0
    
    def _get_smoother(self, idx: int, coord: str) -> ValueSmoother:
        """Get or create smoother for a landmark coordinate."""
        if idx not in self._smoothers:
            self._smoothers[idx] = {}
        if coord not in self._smoothers[idx]:
            self._smoothers[idx][coord] = ValueSmoother(alpha=self.config.alpha)
        return self._smoothers[idx][coord]
    
    def smooth(self, landmarks: list[dict]) -> list[dict]:
        """
        Smooth all landmarks in a frame.
        
        Args:
            landmarks: List of landmark dicts with x, y, z, visibility
            
        Returns:
            Smoothed landmarks list
        """
        self._frame_count += 1
        smoothed = []
        
        for idx, lm in enumerate(landmarks):
            visibility = lm.get("visibility", 0)
            
            # Skip low-visibility landmarks (use last known good value)
            if visibility < self.config.min_visibility:
                # Use last smoothed value if available
                if idx in self._smoothers and "x" in self._smoothers[idx]:
                    smoothed.append({
                        "x": self._smoothers[idx]["x"].value or lm.get("x", 0),
                        "y": self._smoothers[idx]["y"].value or lm.get("y", 0),
                        "z": self._smoothers[idx]["z"].value or lm.get("z", 0),
                        "visibility": visibility
                    })
                else:
                    smoothed.append(lm)
                continue
            
            x = lm.get("x", 0)
            y = lm.get("y", 0)
            z = lm.get("z", 0)
            
            # Check for outliers based on velocity
            x_smoother = self._get_smoother(idx, "x")
            y_smoother = self._get_smoother(idx, "y")
            z_smoother = self._get_smoother(idx, "z")
            
            # Apply visibility-weighted alpha (more visible = trust more)
            effective_alpha = self.config.alpha * visibility
            x_smoother.alpha = effective_alpha
            y_smoother.alpha = effective_alpha
            z_smoother.alpha = effective_alpha
            
            # Outlier rejection: if velocity is too high, reduce alpha
            if x_smoother.value is not None:
                velocity = np.sqrt(
                    (x - x_smoother.value) ** 2 + 
                    (y - y_smoother.value) ** 2
                )
                if velocity > self.config.velocity_threshold:
                    # Likely an outlier, trust historical value more
                    x_smoother.alpha = min(0.1, effective_alpha * 0.3)
                    y_smoother.alpha = min(0.1, effective_alpha * 0.3)
                    z_smoother.alpha = min(0.1, effective_alpha * 0.3)
            
            smoothed.append({
                "x": x_smoother.update(x),
                "y": y_smoother.update(y),
                "z": z_smoother.update(z),
                "visibility": visibility
            })
        
        return smoothed
    
    def reset(self):
        """Reset all smoothers."""
        self._smoothers.clear()
        self._frame_count = 0


class AngleSmoother:
    """
    Smooths joint angles for stable exercise detection.
    
    Uses a combination of:
    - EMA for general smoothing
    - Moving window for outlier detection
    - Dead zone to prevent small fluctuations
    """
    
    def __init__(
        self, 
        alpha: float = 0.3, 
        window_size: int = 5,
        dead_zone: float = 2.0  # Degrees
    ):
        self.alpha = alpha
        self.dead_zone = dead_zone
        self._history: deque[float] = deque(maxlen=window_size)
        self._ema_value: Optional[float] = None
        self._last_output: Optional[float] = None
    
    def smooth(self, angle: float) -> float:
        """
        Smooth an angle value.
        
        Args:
            angle: Raw angle in degrees
            
        Returns:
            Smoothed angle
        """
        self._history.append(angle)
        
        # Calculate EMA
        if self._ema_value is None:
            self._ema_value = angle
        else:
            self._ema_value = self.alpha * angle + (1 - self.alpha) * self._ema_value
        
        # Apply dead zone (prevent small fluctuations from changing output)
        if self._last_output is not None:
            if abs(self._ema_value - self._last_output) < self.dead_zone:
                return self._last_output
        
        self._last_output = self._ema_value
        return self._ema_value
    
    def get_velocity(self) -> float:
        """Get rate of change of angle."""
        if len(self._history) < 2:
            return 0.0
        return self._history[-1] - self._history[-2]
    
    def get_range(self) -> float:
        """Get range of recent angles."""
        if len(self._history) < 2:
            return 0.0
        return max(self._history) - min(self._history)
    
    def reset(self):
        """Reset smoother state."""
        self._history.clear()
        self._ema_value = None
        self._last_output = None


class MultiAngleSmoother:
    """Manages smoothing for multiple joint angles."""
    
    def __init__(self, alpha: float = 0.3, dead_zone: float = 2.0):
        self.alpha = alpha
        self.dead_zone = dead_zone
        self._smoothers: dict[str, AngleSmoother] = {}
    
    def smooth(self, angles: dict[str, float]) -> dict[str, float]:
        """
        Smooth multiple angles.
        
        Args:
            angles: Dict of angle_name -> angle_value
            
        Returns:
            Dict of smoothed angles
        """
        result = {}
        for name, angle in angles.items():
            if name not in self._smoothers:
                self._smoothers[name] = AngleSmoother(
                    alpha=self.alpha, 
                    dead_zone=self.dead_zone
                )
            result[name] = self._smoothers[name].smooth(angle)
        return result
    
    def get_velocities(self) -> dict[str, float]:
        """Get velocities for all tracked angles."""
        return {name: s.get_velocity() for name, s in self._smoothers.items()}
    
    def get_ranges(self) -> dict[str, float]:
        """Get ranges for all tracked angles."""
        return {name: s.get_range() for name, s in self._smoothers.items()}
    
    def reset(self):
        """Reset all smoothers."""
        for smoother in self._smoothers.values():
            smoother.reset()
