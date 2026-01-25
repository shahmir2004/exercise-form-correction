"""Exercise classifier using motion buffer analysis."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import numpy as np

from config.settings import settings
from .base import JointName, Landmark, landmarks_to_dict, calculate_angle
from utils.smoothing import LandmarkSmoother, MultiAngleSmoother, SmoothingConfig


class ExerciseType(str, Enum):
    """Recognized exercise types."""
    UNKNOWN = "unknown"
    SQUAT = "squat"
    PUSHUP = "pushup"
    BICEP_CURL = "bicep_curl"
    ALTERNATE_BICEP_CURL = "alternate_bicep_curl"


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
    classification_scores: dict = field(default_factory=dict)  # Detailed scoring


@dataclass
class ClassificationScore:
    """Weighted score for exercise classification."""
    score: float = 0.0
    factors: dict = field(default_factory=dict)
    
    def add_factor(self, name: str, value: float, weight: float = 1.0):
        """Add a scoring factor."""
        self.factors[name] = {"value": value, "weight": weight, "contribution": value * weight}
        self.score += value * weight


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
            
            # Wrist Y position (for tracking vertical hand movement)
            angles["left_wrist_y"] = landmarks[JointName.LEFT_WRIST].y
            angles["right_wrist_y"] = landmarks[JointName.RIGHT_WRIST].y
            
            # Elbow Y position 
            angles["left_elbow_y"] = landmarks[JointName.LEFT_ELBOW].y
            angles["right_elbow_y"] = landmarks[JointName.RIGHT_ELBOW].y
            
            # Hip Y position (for squat depth tracking)
            angles["left_hip_y"] = landmarks[JointName.LEFT_HIP].y
            angles["right_hip_y"] = landmarks[JointName.RIGHT_HIP].y
            
            # Shoulder Y position
            angles["left_shoulder_y"] = landmarks[JointName.LEFT_SHOULDER].y
            angles["right_shoulder_y"] = landmarks[JointName.RIGHT_SHOULDER].y
            
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
    """
    Classifies exercises based on motion buffer analysis.
    
    Features:
    - Landmark smoothing for noise reduction
    - Multi-factor weighted classification
    - Confidence scoring with history
    - Exercise lock-in with switch detection
    """
    
    def __init__(self):
        self.motion_buffer = MotionBuffer()
        self._locked_exercise: Optional[ExerciseType] = None
        self._confidence_history: deque[float] = deque(maxlen=30)
        self._exercise_history: deque[ExerciseType] = deque(maxlen=30)
        
        # Smoothing for pose accuracy
        self._landmark_smoother = LandmarkSmoother(
            config=SmoothingConfig(alpha=0.4, velocity_threshold=0.25)
        )
        self._angle_smoother = MultiAngleSmoother(alpha=0.35, dead_zone=2.0)
        
        # Classification weights
        self._weights = {
            "pushup": {"horizontal": 3.0, "elbow_disp": 2.0, "wrist_stable": 1.0},
            "bicep_curl": {"elbow_disp": 2.5, "wrist_y_move": 2.0, "hip_stable": 1.5, "shoulder_stable": 1.0},
            "squat": {"knee_disp": 2.5, "hip_disp": 2.0, "hip_y_move": 1.5, "upper_stable": 1.0}
        }
    
    def add_frame(self, landmarks: list[dict]) -> list[dict]:
        """
        Add a frame to the motion buffer with smoothing.
        
        Returns:
            Smoothed landmarks
        """
        # Apply landmark smoothing for pose accuracy
        smoothed_landmarks = self._landmark_smoother.smooth(landmarks)
        self.motion_buffer.add_frame(smoothed_landmarks)
        return smoothed_landmarks
    
    def _calculate_exercise_scores(self) -> dict[ExerciseType, ClassificationScore]:
        """
        Calculate weighted classification scores for each exercise.
        
        Returns:
            Dict mapping exercise type to its score
        """
        scores = {
            ExerciseType.PUSHUP: ClassificationScore(),
            ExerciseType.BICEP_CURL: ClassificationScore(),
            ExerciseType.ALTERNATE_BICEP_CURL: ClassificationScore(),
            ExerciseType.SQUAT: ClassificationScore(),
        }
        
        # Get raw displacements
        left_elbow_disp = self.motion_buffer.get_displacement("left_elbow")
        right_elbow_disp = self.motion_buffer.get_displacement("right_elbow")
        elbow_displacement = max(left_elbow_disp, right_elbow_disp)
        
        left_knee_disp = self.motion_buffer.get_displacement("left_knee")
        right_knee_disp = self.motion_buffer.get_displacement("right_knee")
        knee_displacement = max(left_knee_disp, right_knee_disp)
        
        left_hip_disp = self.motion_buffer.get_displacement("left_hip")
        right_hip_disp = self.motion_buffer.get_displacement("right_hip")
        hip_displacement = max(left_hip_disp, right_hip_disp)
        
        # Positional displacements
        wrist_y_disp = max(
            self.motion_buffer.get_displacement("left_wrist_y"),
            self.motion_buffer.get_displacement("right_wrist_y")
        )
        hip_y_disp = max(
            self.motion_buffer.get_displacement("left_hip_y"),
            self.motion_buffer.get_displacement("right_hip_y")
        )
        shoulder_y_disp = max(
            self.motion_buffer.get_displacement("left_shoulder_y"),
            self.motion_buffer.get_displacement("right_shoulder_y")
        )
        
        is_horizontal = self.motion_buffer.is_horizontal()
        
        # ============ PUSH-UP SCORING ============
        pushup_score = scores[ExerciseType.PUSHUP]
        w = self._weights["pushup"]
        
        # Horizontal body position (critical for pushup)
        pushup_score.add_factor("horizontal", 1.0 if is_horizontal else 0.0, w["horizontal"])
        
        # Elbow angle change
        elbow_factor = min(1.0, elbow_displacement / 60) if is_horizontal else 0.0
        pushup_score.add_factor("elbow_displacement", elbow_factor, w["elbow_disp"])
        
        # Wrists should be stable (not moving up/down like curls)
        wrist_stable = 1.0 - min(1.0, wrist_y_disp / 0.15)
        pushup_score.add_factor("wrist_stable", wrist_stable if is_horizontal else 0.0, w["wrist_stable"])
        
        # ============ BICEP CURL SCORING ============
        curl_score = scores[ExerciseType.BICEP_CURL]
        w = self._weights["bicep_curl"]
        
        # Elbow angle change (primary indicator)
        elbow_factor = min(1.0, elbow_displacement / 70)
        curl_score.add_factor("elbow_displacement", elbow_factor, w["elbow_disp"])
        
        # Wrist vertical movement
        wrist_move = min(1.0, wrist_y_disp / 0.15)
        curl_score.add_factor("wrist_y_movement", wrist_move, w["wrist_y_move"])
        
        # Hips should be stable (not squatting)
        hip_stable = 1.0 - min(1.0, hip_y_disp / 0.10)
        curl_score.add_factor("hip_stable", hip_stable, w["hip_stable"])
        
        # Shoulders should be stable (not pressing overhead)
        shoulder_stable = 1.0 - min(1.0, shoulder_y_disp / 0.08)
        curl_score.add_factor("shoulder_stable", shoulder_stable, w["shoulder_stable"])
        
        # Penalty if horizontal (push-up position)
        if is_horizontal:
            curl_score.score *= 0.2
        
        # ============ SQUAT SCORING ============
        squat_score = scores[ExerciseType.SQUAT]
        w = self._weights["squat"]
        
        # Knee angle change (primary indicator)
        knee_factor = min(1.0, knee_displacement / 60)
        squat_score.add_factor("knee_displacement", knee_factor, w["knee_disp"])
        
        # Hip angle change
        hip_factor = min(1.0, hip_displacement / 40)
        squat_score.add_factor("hip_displacement", hip_factor, w["hip_disp"])
        
        # Hip vertical movement
        hip_y_factor = min(1.0, hip_y_disp / 0.12)
        squat_score.add_factor("hip_y_movement", hip_y_factor, w["hip_y_move"])
        
        # Upper body should be relatively stable
        upper_stable = 1.0 - min(1.0, elbow_displacement / 80)
        squat_score.add_factor("upper_body_stable", upper_stable, w["upper_stable"])
        
        # Penalty if horizontal
        if is_horizontal:
            squat_score.score *= 0.1
        
        # ============ ALTERNATE BICEP CURL SCORING ============
        # Key difference from regular curl: arms move one at a time
        alt_curl_score = scores[ExerciseType.ALTERNATE_BICEP_CURL]
        
        # Check for alternating pattern: one arm moves while other stays stable
        left_wrist_y_disp = self.motion_buffer.get_displacement("left_wrist_y")
        right_wrist_y_disp = self.motion_buffer.get_displacement("right_wrist_y")
        
        # Arms should both have significant movement (just not simultaneously)
        both_arms_active = left_elbow_disp > 25 and right_elbow_disp > 25
        
        # Check for asymmetric movement (one arm much more active at a time)
        elbow_diff = abs(left_elbow_disp - right_elbow_disp)
        wrist_diff = abs(left_wrist_y_disp - right_wrist_y_disp)
        
        # Alternate curls show asymmetric instantaneous movement
        asymmetric_movement = elbow_diff > 15 or wrist_diff > 0.03
        
        if both_arms_active and asymmetric_movement:
            # This looks like alternate curls
            alt_curl_score.add_factor("both_arms_active", 1.0, 2.0)
            alt_curl_score.add_factor("asymmetric_pattern", min(1.0, elbow_diff / 30), 2.5)
            alt_curl_score.add_factor("hip_stable", hip_stable, 1.5)
            alt_curl_score.add_factor("shoulder_stable", shoulder_stable, 1.0)
        else:
            # Doesn't look like alternate curls
            alt_curl_score.add_factor("both_arms_active", 0.0, 2.0)
        
        # Penalty if horizontal
        if is_horizontal:
            alt_curl_score.score *= 0.2
        
        return scores
    
    def identify_exercise(self) -> MotionAnalysis:
        """
        Analyze the motion buffer to identify the exercise.
        Uses weighted multi-factor scoring for better accuracy.
        
        Returns:
            MotionAnalysis with exercise type and confidence
        """
        if self.motion_buffer.size < 15:
            return MotionAnalysis(
                exercise_type=ExerciseType.UNKNOWN,
                confidence=0.0
            )
        
        # Calculate weighted scores for each exercise
        scores = self._calculate_exercise_scores()
        
        # Get raw displacement values for the result
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
        
        # Find best matching exercise
        best_exercise = ExerciseType.UNKNOWN
        best_score = 0.0
        classification_scores = {}
        
        min_score_threshold = 2.5  # Minimum score to consider valid
        
        for exercise_type, score_data in scores.items():
            classification_scores[exercise_type.value] = {
                "score": round(score_data.score, 2),
                "factors": score_data.factors
            }
            
            if score_data.score > best_score and score_data.score >= min_score_threshold:
                best_score = score_data.score
                best_exercise = exercise_type
        
        # Calculate confidence (normalize score to 0-1)
        max_possible_score = 7.0  # Approximate max weighted score
        confidence = min(0.95, best_score / max_possible_score) if best_exercise != ExerciseType.UNKNOWN else 0.0
        
        # Check for full rep
        has_full_rep = False
        if best_exercise == ExerciseType.PUSHUP:
            has_full_rep = self.motion_buffer.has_completed_rep("left_elbow", 70, 160)
        elif best_exercise == ExerciseType.BICEP_CURL:
            has_full_rep = (
                self.motion_buffer.has_completed_rep("left_elbow", 40, 150) or
                self.motion_buffer.has_completed_rep("right_elbow", 40, 150)
            )
        elif best_exercise == ExerciseType.ALTERNATE_BICEP_CURL:
            # For alternate curls, check either arm
            has_full_rep = (
                self.motion_buffer.has_completed_rep("left_elbow", 40, 150) or
                self.motion_buffer.has_completed_rep("right_elbow", 40, 150)
            )
        elif best_exercise == ExerciseType.SQUAT:
            has_full_rep = (
                self.motion_buffer.has_completed_rep("left_knee", 70, 160) or
                self.motion_buffer.has_completed_rep("right_knee", 70, 160)
            )
        
        # Update history
        self._exercise_history.append(best_exercise)
        self._confidence_history.append(confidence)
        
        # Lock-in logic: require consistent detection + full rep
        if has_full_rep and confidence >= settings.CONFIDENCE_THRESHOLD:
            if not self._locked_exercise:
                # Check consistency in recent history
                recent = list(self._exercise_history)[-15:]
                if recent.count(best_exercise) >= 10:
                    self._locked_exercise = best_exercise
        
        return MotionAnalysis(
            exercise_type=self._locked_exercise or best_exercise,
            confidence=confidence,
            elbow_displacement=elbow_displacement,
            knee_displacement=knee_displacement,
            hip_displacement=hip_displacement,
            is_horizontal=is_horizontal,
            has_full_rep=has_full_rep,
            classification_scores=classification_scores
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
        self._landmark_smoother.reset()
        self._angle_smoother.reset()
    
    @property
    def locked_exercise(self) -> Optional[ExerciseType]:
        """Currently locked exercise type."""
        return self._locked_exercise
    
    def get_smoothed_angles(self, angles: dict[str, float]) -> dict[str, float]:
        """Get smoothed angles for display/analysis."""
        return self._angle_smoother.smooth(angles)
