"""Push-up exercise module with form correction."""

from typing import Optional
import numpy as np
from .base import (
    BaseExercise,
    ExerciseResult,
    JointAngles,
    JointName,
    Landmark,
    calculate_angle,
)


class PushupModule(BaseExercise):
    """Push-up exercise detection and form correction."""
    
    # Form thresholds
    MIN_ELBOW_ANGLE = 70  # Bottom of push-up
    MAX_ELBOW_ANGLE = 160  # Top of push-up (arms extended)
    ELBOW_FLARE_THRESHOLD = 75  # Max angle between upper arm and torso
    HIP_SAG_THRESHOLD = 0.05  # Normalized Y difference for hip sag
    HIP_PIKE_THRESHOLD = 0.05  # Normalized Y difference for hip pike
    HORIZONTAL_THRESHOLD = 0.15  # How horizontal body should be
    
    def __init__(self):
        super().__init__()
        self._lowest_elbow_angle = 180.0
        self._in_pushup = False
    
    @property
    def name(self) -> str:
        return "Push-up"
    
    @property
    def required_joints(self) -> list[JointName]:
        return [
            JointName.LEFT_SHOULDER,
            JointName.RIGHT_SHOULDER,
            JointName.LEFT_ELBOW,
            JointName.RIGHT_ELBOW,
            JointName.LEFT_WRIST,
            JointName.RIGHT_WRIST,
            JointName.LEFT_HIP,
            JointName.RIGHT_HIP,
            JointName.LEFT_ANKLE,
            JointName.RIGHT_ANKLE,
        ]
    
    def _calculate_angles(self, landmarks: dict[JointName, Landmark]) -> JointAngles:
        """Calculate all relevant joint angles for push-up analysis."""
        angles = JointAngles()
        
        # Left elbow angle (shoulder-elbow-wrist)
        angles.left_elbow = calculate_angle(
            landmarks[JointName.LEFT_SHOULDER],
            landmarks[JointName.LEFT_ELBOW],
            landmarks[JointName.LEFT_WRIST]
        )
        
        # Right elbow angle
        angles.right_elbow = calculate_angle(
            landmarks[JointName.RIGHT_SHOULDER],
            landmarks[JointName.RIGHT_ELBOW],
            landmarks[JointName.RIGHT_WRIST]
        )
        
        # Left shoulder angle (elbow-shoulder-hip) for flare detection
        angles.left_shoulder = calculate_angle(
            landmarks[JointName.LEFT_ELBOW],
            landmarks[JointName.LEFT_SHOULDER],
            landmarks[JointName.LEFT_HIP]
        )
        
        # Right shoulder angle
        angles.right_shoulder = calculate_angle(
            landmarks[JointName.RIGHT_ELBOW],
            landmarks[JointName.RIGHT_SHOULDER],
            landmarks[JointName.RIGHT_HIP]
        )
        
        # Torso angle (body alignment from shoulders to ankles)
        mid_shoulder = np.array([
            (landmarks[JointName.LEFT_SHOULDER].x + landmarks[JointName.RIGHT_SHOULDER].x) / 2,
            (landmarks[JointName.LEFT_SHOULDER].y + landmarks[JointName.RIGHT_SHOULDER].y) / 2
        ])
        mid_hip = np.array([
            (landmarks[JointName.LEFT_HIP].x + landmarks[JointName.RIGHT_HIP].x) / 2,
            (landmarks[JointName.LEFT_HIP].y + landmarks[JointName.RIGHT_HIP].y) / 2
        ])
        mid_ankle = np.array([
            (landmarks[JointName.LEFT_ANKLE].x + landmarks[JointName.RIGHT_ANKLE].x) / 2,
            (landmarks[JointName.LEFT_ANKLE].y + landmarks[JointName.RIGHT_ANKLE].y) / 2
        ])
        
        # Calculate deviation of hips from shoulder-ankle line
        shoulder_ankle = mid_ankle - mid_shoulder
        shoulder_hip = mid_hip - mid_shoulder
        
        if np.linalg.norm(shoulder_ankle) > 0:
            # Project hip onto shoulder-ankle line
            t = np.dot(shoulder_hip, shoulder_ankle) / np.dot(shoulder_ankle, shoulder_ankle)
            projected = mid_shoulder + t * shoulder_ankle
            deviation = np.linalg.norm(mid_hip - projected)
            angles.torso_angle = deviation  # Store as deviation magnitude
        
        self._last_angles = angles
        return angles
    
    def _check_body_alignment(self, landmarks: dict[JointName, Landmark]) -> tuple[bool, bool, bool]:
        """
        Check body alignment for hip sag, pike, and horizontal position.
        Returns (hip_sag, hip_pike, is_horizontal).
        """
        # Get key points
        mid_shoulder_y = (landmarks[JointName.LEFT_SHOULDER].y + 
                         landmarks[JointName.RIGHT_SHOULDER].y) / 2
        mid_hip_y = (landmarks[JointName.LEFT_HIP].y + 
                    landmarks[JointName.RIGHT_HIP].y) / 2
        mid_ankle_y = (landmarks[JointName.LEFT_ANKLE].y + 
                      landmarks[JointName.RIGHT_ANKLE].y) / 2
        
        # Expected hip Y position (linear interpolation between shoulder and ankle)
        expected_hip_y = (mid_shoulder_y + mid_ankle_y) / 2
        
        # Hip sag: hips below the expected line
        hip_sag = (mid_hip_y - expected_hip_y) > self.HIP_SAG_THRESHOLD
        
        # Hip pike: hips above the expected line
        hip_pike = (expected_hip_y - mid_hip_y) > self.HIP_PIKE_THRESHOLD
        
        # Check if body is horizontal enough (shoulders and ankles at similar Y)
        y_difference = abs(mid_shoulder_y - mid_ankle_y)
        is_horizontal = y_difference < self.HORIZONTAL_THRESHOLD
        
        return hip_sag, hip_pike, is_horizontal
    
    def _check_elbow_flare(self, landmarks: dict[JointName, Landmark]) -> tuple[bool, bool]:
        """
        Check for excessive elbow flare.
        Returns (left_flare, right_flare).
        """
        angles = self._last_angles or self._calculate_angles(landmarks)
        
        # Elbow flare is when shoulder angle (elbow-shoulder-hip) is too large
        # Ideally, elbows should be at ~45 degrees from body, not 90
        left_flare = angles.left_shoulder > self.ELBOW_FLARE_THRESHOLD
        right_flare = angles.right_shoulder > self.ELBOW_FLARE_THRESHOLD
        
        return left_flare, right_flare
    
    def detect_rep_phase(self, landmarks: dict[JointName, Landmark]) -> str:
        """Detect current phase of push-up repetition."""
        angles = self._calculate_angles(landmarks)
        avg_elbow_angle = (angles.left_elbow + angles.right_elbow) / 2
        
        # Check if body is in push-up position (horizontal)
        _, _, is_horizontal = self._check_body_alignment(landmarks)
        if not is_horizontal:
            return "idle"
        
        if avg_elbow_angle > self.MAX_ELBOW_ANGLE - 10:
            # Top position (arms extended)
            if self._in_pushup:
                self._in_pushup = False
                self._lowest_elbow_angle = 180.0
                return "up"
            return "idle"
        elif avg_elbow_angle < self.MIN_ELBOW_ANGLE + 20:
            # Bottom of push-up
            self._in_pushup = True
            self._lowest_elbow_angle = min(self._lowest_elbow_angle, avg_elbow_angle)
            return "hold"
        else:
            # Transitioning
            if self._in_pushup:
                if avg_elbow_angle > self._lowest_elbow_angle + 15:
                    return "up"
                return "down"
            else:
                self._in_pushup = True
                return "down"
    
    def check_form(self, landmarks: dict[JointName, Landmark]) -> ExerciseResult:
        """Analyze push-up form and return corrections."""
        violations = []
        corrections = []
        joint_colors = {}
        
        # Calculate angles
        angles = self._calculate_angles(landmarks)
        
        # Initialize all tracked joints as green
        for joint in self.required_joints:
            joint_colors[joint.value] = "green"
        
        is_valid = True
        
        # Check body alignment
        hip_sag, hip_pike, is_horizontal = self._check_body_alignment(landmarks)
        
        if hip_sag:
            violations.append("Hips sagging")
            corrections.append("Engage your core and lift your hips in line with shoulders")
            joint_colors[JointName.LEFT_HIP.value] = "red"
            joint_colors[JointName.RIGHT_HIP.value] = "red"
            is_valid = False
        
        if hip_pike:
            violations.append("Hips too high")
            corrections.append("Lower your hips to form a straight line from head to heels")
            joint_colors[JointName.LEFT_HIP.value] = "yellow"
            joint_colors[JointName.RIGHT_HIP.value] = "yellow"
        
        # Check elbow flare
        left_flare, right_flare = self._check_elbow_flare(landmarks)
        
        if left_flare:
            violations.append("Left elbow flaring out")
            corrections.append("Tuck your left elbow closer to your body (45° angle)")
            joint_colors[JointName.LEFT_ELBOW.value] = "red"
            is_valid = False
        
        if right_flare:
            violations.append("Right elbow flaring out")
            corrections.append("Tuck your right elbow closer to your body (45° angle)")
            joint_colors[JointName.RIGHT_ELBOW.value] = "red"
            is_valid = False
        
        # Check depth
        avg_elbow_angle = (angles.left_elbow + angles.right_elbow) / 2
        if self._in_pushup and self._lowest_elbow_angle > self.MIN_ELBOW_ANGLE + 10:
            violations.append("Insufficient push-up depth")
            corrections.append("Lower your chest closer to the ground")
        
        # Check elbow angle symmetry
        elbow_asymmetry = abs(angles.left_elbow - angles.right_elbow)
        if elbow_asymmetry > 20:
            violations.append("Uneven arm bend")
            corrections.append("Distribute weight evenly on both arms")
            if angles.left_elbow < angles.right_elbow:
                joint_colors[JointName.RIGHT_ELBOW.value] = "yellow"
            else:
                joint_colors[JointName.LEFT_ELBOW.value] = "yellow"
        
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
        """Rep complete when returning to top position from bottom."""
        return last_phase == "up" and (current_phase == "idle" or current_phase == "down")
