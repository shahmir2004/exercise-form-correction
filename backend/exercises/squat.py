"""Squat exercise module with form correction."""

from typing import Optional
from .base import (
    BaseExercise,
    ExerciseResult,
    JointAngles,
    JointName,
    Landmark,
    calculate_angle,
)


class SquatModule(BaseExercise):
    """Squat exercise detection and form correction."""
    
    # Form thresholds
    MIN_KNEE_ANGLE = 70  # Minimum knee bend for valid squat depth
    MAX_KNEE_ANGLE = 160  # Standing position
    KNEE_VALGUS_THRESHOLD = 0.03  # Normalized X difference threshold
    BACK_ANGLE_THRESHOLD = 45  # Maximum forward lean angle
    
    # Hysteresis thresholds
    ANGLE_HYSTERESIS = 12  # Degrees of buffer to prevent flickering
    
    def __init__(self):
        super().__init__()
        self._lowest_knee_angle = 180.0
        self._in_squat = False
        
        # Create hysteresis-based rep counter for stable counting
        self._create_rep_counter(
            upper_threshold=self.MAX_KNEE_ANGLE - 10,  # ~150 degrees (standing)
            lower_threshold=self.MIN_KNEE_ANGLE + 20   # ~90 degrees (parallel)
        )
    
    @property
    def name(self) -> str:
        return "Squat"
    
    @property
    def required_joints(self) -> list[JointName]:
        return [
            JointName.LEFT_HIP,
            JointName.RIGHT_HIP,
            JointName.LEFT_KNEE,
            JointName.RIGHT_KNEE,
            JointName.LEFT_ANKLE,
            JointName.RIGHT_ANKLE,
            JointName.LEFT_SHOULDER,
            JointName.RIGHT_SHOULDER,
        ]
    
    def _calculate_angles(self, landmarks: dict[JointName, Landmark]) -> JointAngles:
        """Calculate all relevant joint angles for squat analysis."""
        angles = JointAngles()
        
        # Left knee angle (hip-knee-ankle)
        angles.left_knee = calculate_angle(
            landmarks[JointName.LEFT_HIP],
            landmarks[JointName.LEFT_KNEE],
            landmarks[JointName.LEFT_ANKLE]
        )
        
        # Right knee angle
        angles.right_knee = calculate_angle(
            landmarks[JointName.RIGHT_HIP],
            landmarks[JointName.RIGHT_KNEE],
            landmarks[JointName.RIGHT_ANKLE]
        )
        
        # Left hip angle (shoulder-hip-knee)
        angles.left_hip = calculate_angle(
            landmarks[JointName.LEFT_SHOULDER],
            landmarks[JointName.LEFT_HIP],
            landmarks[JointName.LEFT_KNEE]
        )
        
        # Right hip angle
        angles.right_hip = calculate_angle(
            landmarks[JointName.RIGHT_SHOULDER],
            landmarks[JointName.RIGHT_HIP],
            landmarks[JointName.RIGHT_KNEE]
        )
        
        # Torso angle (deviation from vertical)
        mid_shoulder_y = (landmarks[JointName.LEFT_SHOULDER].y + 
                         landmarks[JointName.RIGHT_SHOULDER].y) / 2
        mid_hip_y = (landmarks[JointName.LEFT_HIP].y + 
                    landmarks[JointName.RIGHT_HIP].y) / 2
        mid_shoulder_x = (landmarks[JointName.LEFT_SHOULDER].x + 
                         landmarks[JointName.RIGHT_SHOULDER].x) / 2
        mid_hip_x = (landmarks[JointName.LEFT_HIP].x + 
                    landmarks[JointName.RIGHT_HIP].x) / 2
        
        import numpy as np
        torso_vector = np.array([mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y])
        vertical = np.array([0, -1])  # Up direction in normalized coords
        
        if np.linalg.norm(torso_vector) > 0:
            cos_angle = np.dot(torso_vector, vertical) / np.linalg.norm(torso_vector)
            angles.torso_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        self._last_angles = angles
        return angles
    
    def _check_knee_valgus(self, landmarks: dict[JointName, Landmark]) -> tuple[bool, bool]:
        """
        Check for knee valgus (knees caving inward).
        Returns (left_valgus, right_valgus).
        """
        # Compare knee X position to ankle X position
        # Knees should track over toes, not cave inward
        
        left_knee_x = landmarks[JointName.LEFT_KNEE].x
        left_ankle_x = landmarks[JointName.LEFT_ANKLE].x
        right_knee_x = landmarks[JointName.RIGHT_KNEE].x
        right_ankle_x = landmarks[JointName.RIGHT_ANKLE].x
        
        # For left leg: knee should not be significantly to the right of ankle
        left_valgus = (left_knee_x - left_ankle_x) > self.KNEE_VALGUS_THRESHOLD
        
        # For right leg: knee should not be significantly to the left of ankle
        right_valgus = (right_ankle_x - right_knee_x) > self.KNEE_VALGUS_THRESHOLD
        
        return left_valgus, right_valgus
    
    def detect_rep_phase(self, landmarks: dict[JointName, Landmark]) -> str:
        """Detect current phase of squat repetition with hysteresis rep counter."""
        angles = self._calculate_angles(landmarks)
        avg_knee_angle = (angles.left_knee + angles.right_knee) / 2
        
        # Use hysteresis-based rep counter for stable phase detection
        if self._rep_counter:
            phase_str, rep_completed = self.update_rep_counter(
                angle=avg_knee_angle,
                left_angle=angles.left_knee,
                right_angle=angles.right_knee
            )
            
            # Track lowest angle for depth validation
            self._lowest_knee_angle = min(self._lowest_knee_angle, avg_knee_angle)
            
            # Update internal state to match counter
            if phase_str in ["up", "down", "hold"]:
                self._in_squat = phase_str in ["down", "hold"]
            
            if rep_completed:
                # Reset tracking for next rep
                self._lowest_knee_angle = 180.0
            
            return phase_str
        
        # Fallback to legacy detection if counter not available
        if avg_knee_angle > self.MAX_KNEE_ANGLE - 10:
            # Standing position
            if self._in_squat:
                self._in_squat = False
                self._lowest_knee_angle = 180.0
                return "up"
            return "idle"
        elif avg_knee_angle < self.MIN_KNEE_ANGLE + 20:
            # Bottom of squat
            self._in_squat = True
            self._lowest_knee_angle = min(self._lowest_knee_angle, avg_knee_angle)
            return "hold"
        else:
            # Transitioning
            if self._in_squat:
                if avg_knee_angle > self._lowest_knee_angle + 15:
                    return "up"
                return "down"
            else:
                self._in_squat = True
                return "down"
    
    def check_form(self, landmarks: dict[JointName, Landmark]) -> ExerciseResult:
        """Analyze squat form and return corrections."""
        violations = []
        corrections = []
        joint_colors = {}
        
        # Calculate angles
        angles = self._calculate_angles(landmarks)
        avg_knee_angle = (angles.left_knee + angles.right_knee) / 2
        
        # Initialize all tracked joints as green
        for joint in self.required_joints:
            joint_colors[joint.value] = "green"
        
        is_valid = True
        
        # Check depth (only relevant when in squat position)
        if self._in_squat and self._lowest_knee_angle > self.MIN_KNEE_ANGLE + 10:
            violations.append("Insufficient squat depth")
            corrections.append("Lower your hips until thighs are parallel to ground")
            # Don't mark as invalid yet - they might go deeper
        
        # Check knee valgus
        left_valgus, right_valgus = self._check_knee_valgus(landmarks)
        if left_valgus:
            violations.append("Left knee caving inward")
            corrections.append("Push your left knee outward over your toes")
            joint_colors[JointName.LEFT_KNEE.value] = "red"
            is_valid = False
        
        if right_valgus:
            violations.append("Right knee caving inward")
            corrections.append("Push your right knee outward over your toes")
            joint_colors[JointName.RIGHT_KNEE.value] = "red"
            is_valid = False
        
        # Check back angle (forward lean)
        if angles.torso_angle > self.BACK_ANGLE_THRESHOLD:
            violations.append("Excessive forward lean")
            corrections.append("Keep your chest up and back straighter")
            joint_colors[JointName.LEFT_SHOULDER.value] = "yellow"
            joint_colors[JointName.RIGHT_SHOULDER.value] = "yellow"
            # Yellow warning, not invalid
        
        # Check knee angle symmetry
        knee_asymmetry = abs(angles.left_knee - angles.right_knee)
        if knee_asymmetry > 15:
            violations.append("Uneven knee bend")
            corrections.append("Distribute weight evenly on both legs")
            if angles.left_knee < angles.right_knee:
                joint_colors[JointName.RIGHT_KNEE.value] = "yellow"
            else:
                joint_colors[JointName.LEFT_KNEE.value] = "yellow"
        
        # Calculate confidence based on joint visibility
        visibility_sum = sum(landmarks[j].visibility for j in self.required_joints)
        confidence = visibility_sum / len(self.required_joints)
        
        return ExerciseResult(
            is_valid=is_valid,
            rep_count=self.rep_count,
            rep_phase=self.current_phase,
            violations=violations,
            corrections=corrections,
            joint_colors=joint_colors,
            confidence=confidence,
            angles=angles
        )
    
    def _is_rep_complete(self, last_phase: str, current_phase: str) -> bool:
        """Rep complete when returning to standing from squat."""
        return last_phase == "up" and current_phase == "idle"
