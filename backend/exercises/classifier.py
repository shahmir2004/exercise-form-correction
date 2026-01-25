"""Exercise classifier using motion buffer analysis."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import numpy as np

from config.settings import settings
from .base import JointName, Landmark, landmarks_to_dict, calculate_angle


class ExerciseType(str, Enum):
    """Recognized exercise types."""
    UNKNOWN = "unknown"
    SQUAT = "squat"
    PUSHUP = "pushup"
    BICEP_CURL = "bicep_curl"


@dataclass
class MotionAnalysis:
    """Result of motion buffer analysis."""
    exercise_type: ExerciseType
    confidence: float
    elbow_displacement: float = 0.0
    knee_displacement: float = 0.0
    hip_displacement: float = 0.0
    shoulder_displacement: float = 0.0
    is_horizontal: bool = False
    has_full_rep: bool = False


@dataclass
class MotionBuffer:
    """Rolling buffer of landmark frames for motion analysis."""
    
    max_size: int = field(default_factory=lambda: settings.MOTION_BUFFER_SIZE)
    
    def __post_init__(self):
        self._frames: deque[list[dict]] = deque(maxlen=self.max_size)
        self._angles_history: deque[dict] = deque(maxlen=self.max_size)
    
    def add_frame(self, landmarks: list[dict]) -> None:
        """Add a frame of landmarks to the buffer."""
        self._frames.append(landmarks)
        
        # Calculate and store key angles
        landmark_dict = landmarks_to_dict(landmarks)
        angles = self._calculate_key_angles(landmark_dict)
        self._angles_history.append(angles)
    
    def _calculate_key_angles(self, landmarks: dict[JointName, Landmark]) -> dict:
        """Calculate key angles for exercise detection."""
        angles = {}
        
        try:
            # Elbow angles
            angles["left_elbow"] = calculate_angle(
                landmarks[JointName.LEFT_SHOULDER],
                landmarks[JointName.LEFT_ELBOW],
                landmarks[JointName.LEFT_WRIST]
            )
            angles["right_elbow"] = calculate_angle(
                landmarks[JointName.RIGHT_SHOULDER],
                landmarks[JointName.RIGHT_ELBOW],
                landmarks[JointName.RIGHT_WRIST]
            )
            
            # Knee angles
            angles["left_knee"] = calculate_angle(
                landmarks[JointName.LEFT_HIP],
                landmarks[JointName.LEFT_KNEE],
                landmarks[JointName.LEFT_ANKLE]
            )
            angles["right_knee"] = calculate_angle(
                landmarks[JointName.RIGHT_HIP],
                landmarks[JointName.RIGHT_KNEE],
                landmarks[JointName.RIGHT_ANKLE]
            )
            
            # Hip angles
            angles["left_hip"] = calculate_angle(
                landmarks[JointName.LEFT_SHOULDER],
                landmarks[JointName.LEFT_HIP],
                landmarks[JointName.LEFT_KNEE]
            )
            angles["right_hip"] = calculate_angle(
                landmarks[JointName.RIGHT_SHOULDER],
                landmarks[JointName.RIGHT_HIP],
                landmarks[JointName.RIGHT_KNEE]
            )
            
            # Body orientation
            mid_shoulder_y = (landmarks[JointName.LEFT_SHOULDER].y + 
                             landmarks[JointName.RIGHT_SHOULDER].y) / 2
            mid_hip_y = (landmarks[JointName.LEFT_HIP].y + 
                        landmarks[JointName.RIGHT_HIP].y) / 2
            mid_ankle_y = (landmarks[JointName.LEFT_ANKLE].y + 
                          landmarks[JointName.RIGHT_ANKLE].y) / 2
            
            # Check if body is horizontal (push-up position)
            angles["is_horizontal"] = abs(mid_shoulder_y - mid_ankle_y) < 0.15
            angles["shoulder_y"] = mid_shoulder_y
            angles["hip_y"] = mid_hip_y
            
        except (KeyError, TypeError):
            # Missing landmarks
            pass
        
        return angles
    
    def get_displacement(self, angle_key: str) -> float:
        """Calculate range of motion for a specific angle over the buffer."""
        if len(self._angles_history) < 10:
            return 0.0
        
        values = [a.get(angle_key, 0) for a in self._angles_history if angle_key in a]
        if not values:
            return 0.0
        
        return max(values) - min(values)
    
    def is_horizontal(self) -> bool:
        """Check if body has been horizontal recently."""
        if len(self._angles_history) < 5:
            return False
        
        recent = list(self._angles_history)[-10:]
        horizontal_count = sum(1 for a in recent if a.get("is_horizontal", False))
        return horizontal_count > len(recent) * 0.6
    
    def has_completed_rep(self, angle_key: str, min_angle: float, max_angle: float) -> bool:
        """Check if a full rep has been completed for the given angle."""
        if len(self._angles_history) < 20:
            return False
        
        values = [a.get(angle_key, 0) for a in self._angles_history if angle_key in a]
        if len(values) < 20:
            return False
        
        # Look for pattern: high -> low -> high (or low -> high -> low)
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        
        if range_val < 30:  # Not enough movement
            return False
        
        # Check for at least one full cycle
        crossed_low = False
        crossed_high = False
        returned = False
        
        threshold_low = min_val + range_val * 0.3
        threshold_high = max_val - range_val * 0.3
        
        for v in values:
            if v < threshold_low:
                crossed_low = True
            if crossed_low and v > threshold_high:
                crossed_high = True
            if crossed_high and v < threshold_low:
                returned = True
                break
        
        return crossed_low and crossed_high
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._frames.clear()
        self._angles_history.clear()
    
    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self._frames)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._frames) >= self.max_size


class ExerciseClassifier:
    """Classifies exercises based on motion buffer analysis."""
    
    def __init__(self):
        self.motion_buffer = MotionBuffer()
        self._locked_exercise: Optional[ExerciseType] = None
        self._confidence_history: deque[float] = deque(maxlen=30)
        self._exercise_history: deque[ExerciseType] = deque(maxlen=30)
    
    def add_frame(self, landmarks: list[dict]) -> None:
        """Add a frame to the motion buffer."""
        self.motion_buffer.add_frame(landmarks)
    
    def identify_exercise(self) -> MotionAnalysis:
        """
        Analyze the motion buffer to identify the exercise.
        
        Returns:
            MotionAnalysis with exercise type and confidence
        """
        if self.motion_buffer.size < 15:
            return MotionAnalysis(
                exercise_type=ExerciseType.UNKNOWN,
                confidence=0.0
            )
        
        # Calculate displacements for key joints
        left_elbow_disp = self.motion_buffer.get_displacement("left_elbow")
        right_elbow_disp = self.motion_buffer.get_displacement("right_elbow")
        elbow_displacement = max(left_elbow_disp, right_elbow_disp)
        
        left_knee_disp = self.motion_buffer.get_displacement("left_knee")
        right_knee_disp = self.motion_buffer.get_displacement("right_knee")
        knee_displacement = max(left_knee_disp, right_knee_disp)
        
        left_hip_disp = self.motion_buffer.get_displacement("left_hip")
        right_hip_disp = self.motion_buffer.get_displacement("right_hip")
        hip_displacement = max(left_hip_disp, right_hip_disp)
        
        is_horizontal = self.motion_buffer.is_horizontal()
        
        # Classification logic based on greatest displacement
        exercise_type = ExerciseType.UNKNOWN
        confidence = 0.0
        has_full_rep = False
        
        # Push-up: body horizontal + elbow/shoulder movement
        if is_horizontal and elbow_displacement > 30:
            exercise_type = ExerciseType.PUSHUP
            confidence = min(0.95, 0.5 + elbow_displacement / 100)
            has_full_rep = self.motion_buffer.has_completed_rep("left_elbow", 70, 160)
        
        # Squat: knee/hip displacement is greatest (body vertical)
        elif knee_displacement > elbow_displacement and knee_displacement > 30:
            exercise_type = ExerciseType.SQUAT
            confidence = min(0.95, 0.5 + knee_displacement / 100)
            has_full_rep = self.motion_buffer.has_completed_rep("left_knee", 70, 160)
        
        # Bicep curl: elbow displacement is greatest (body vertical)
        elif elbow_displacement > knee_displacement and elbow_displacement > 40:
            exercise_type = ExerciseType.BICEP_CURL
            confidence = min(0.95, 0.5 + elbow_displacement / 120)
            has_full_rep = self.motion_buffer.has_completed_rep("left_elbow", 30, 160)
        
        # Update history
        self._exercise_history.append(exercise_type)
        self._confidence_history.append(confidence)
        
        # Check for lock-in
        if has_full_rep and confidence >= settings.CONFIDENCE_THRESHOLD:
            if not self._locked_exercise:
                self._locked_exercise = exercise_type
        
        return MotionAnalysis(
            exercise_type=self._locked_exercise or exercise_type,
            confidence=confidence,
            elbow_displacement=elbow_displacement,
            knee_displacement=knee_displacement,
            hip_displacement=hip_displacement,
            is_horizontal=is_horizontal,
            has_full_rep=has_full_rep
        )
    
    def should_switch_exercise(self, new_exercise: ExerciseType, duration_seconds: float) -> bool:
        """
        Check if we should switch to a different exercise.
        
        Args:
            new_exercise: The newly detected exercise type
            duration_seconds: How long this exercise has been detected
            
        Returns:
            True if we should switch exercises
        """
        if not self._locked_exercise:
            return True
        
        if new_exercise == self._locked_exercise:
            return False
        
        # Count how many recent frames show the new exercise
        recent_count = sum(1 for e in list(self._exercise_history)[-30:] if e == new_exercise)
        
        # Switch if new exercise detected consistently for threshold duration
        if recent_count > 20 and duration_seconds >= settings.EXERCISE_SWITCH_DELAY:
            return True
        
        return False
    
    def lock_exercise(self, exercise_type: ExerciseType) -> None:
        """Lock in a specific exercise type."""
        self._locked_exercise = exercise_type
    
    def unlock_exercise(self) -> None:
        """Unlock the current exercise to allow re-detection."""
        self._locked_exercise = None
        self._exercise_history.clear()
        self._confidence_history.clear()
    
    def reset(self) -> None:
        """Reset all classifier state."""
        self.motion_buffer.clear()
        self._locked_exercise = None
        self._exercise_history.clear()
        self._confidence_history.clear()
    
    @property
    def locked_exercise(self) -> Optional[ExerciseType]:
        """Currently locked exercise type."""
        return self._locked_exercise
