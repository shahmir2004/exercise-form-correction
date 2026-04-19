"""Base exercise module with abstract interface for form correction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

from utils.rep_counter import HysteresisRepCounter, RepPhase, RepQuality


class JointName(str, Enum):
    """MediaPipe pose landmark names."""
    NOSE = "nose"
    LEFT_EYE_INNER = "left_eye_inner"
    LEFT_EYE = "left_eye"
    LEFT_EYE_OUTER = "left_eye_outer"
    RIGHT_EYE_INNER = "right_eye_inner"
    RIGHT_EYE = "right_eye"
    RIGHT_EYE_OUTER = "right_eye_outer"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    MOUTH_LEFT = "mouth_left"
    MOUTH_RIGHT = "mouth_right"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_PINKY = "left_pinky"
    RIGHT_PINKY = "right_pinky"
    LEFT_INDEX = "left_index"
    RIGHT_INDEX = "right_index"
    LEFT_THUMB = "left_thumb"
    RIGHT_THUMB = "right_thumb"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"
    LEFT_FOOT_INDEX = "left_foot_index"
    RIGHT_FOOT_INDEX = "right_foot_index"


# MediaPipe landmark index mapping
LANDMARK_INDICES = {
    JointName.NOSE: 0,
    JointName.LEFT_EYE_INNER: 1,
    JointName.LEFT_EYE: 2,
    JointName.LEFT_EYE_OUTER: 3,
    JointName.RIGHT_EYE_INNER: 4,
    JointName.RIGHT_EYE: 5,
    JointName.RIGHT_EYE_OUTER: 6,
    JointName.LEFT_EAR: 7,
    JointName.RIGHT_EAR: 8,
    JointName.MOUTH_LEFT: 9,
    JointName.MOUTH_RIGHT: 10,
    JointName.LEFT_SHOULDER: 11,
    JointName.RIGHT_SHOULDER: 12,
    JointName.LEFT_ELBOW: 13,
    JointName.RIGHT_ELBOW: 14,
    JointName.LEFT_WRIST: 15,
    JointName.RIGHT_WRIST: 16,
    JointName.LEFT_PINKY: 17,
    JointName.RIGHT_PINKY: 18,
    JointName.LEFT_INDEX: 19,
    JointName.RIGHT_INDEX: 20,
    JointName.LEFT_THUMB: 21,
    JointName.RIGHT_THUMB: 22,
    JointName.LEFT_HIP: 23,
    JointName.RIGHT_HIP: 24,
    JointName.LEFT_KNEE: 25,
    JointName.RIGHT_KNEE: 26,
    JointName.LEFT_ANKLE: 27,
    JointName.RIGHT_ANKLE: 28,
    JointName.LEFT_HEEL: 29,
    JointName.RIGHT_HEEL: 30,
    JointName.LEFT_FOOT_INDEX: 31,
    JointName.RIGHT_FOOT_INDEX: 32,
}


@dataclass
class Landmark:
    """Single pose landmark with normalized coordinates."""
    x: float  # 0-1 normalized
    y: float  # 0-1 normalized
    z: float  # Depth relative to hips
    visibility: float  # 0-1 confidence


@dataclass
class JointAngles:
    """Computed joint angles for exercise analysis."""
    left_elbow: float = 0.0
    right_elbow: float = 0.0
    left_shoulder: float = 0.0
    right_shoulder: float = 0.0
    left_hip: float = 0.0
    right_hip: float = 0.0
    left_knee: float = 0.0
    right_knee: float = 0.0
    torso_angle: float = 0.0  # Angle from vertical
    

@dataclass
class ExerciseResult:
    """Result from exercise form analysis."""
    is_valid: bool
    rep_count: int
    rep_phase: str  # "up", "down", "hold", "transition"
    violations: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    joint_colors: dict[str, str] = field(default_factory=dict)  # joint_name -> "green" | "red" | "yellow"
    confidence: float = 0.0
    angles: Optional[JointAngles] = None
    rep_quality: Optional[float] = None  # 0-1 quality score for last rep
    partial_reps: int = 0  # Count of incomplete reps


def calculate_angle(p1: Landmark, p2: Landmark, p3: Landmark) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3.
    Returns angle in degrees (0-180).
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Calculate angle
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def landmarks_to_dict(landmarks: list[dict]) -> dict[JointName, Landmark]:
    """Convert landmark list to dictionary by joint name."""
    result = {}
    for joint_name, idx in LANDMARK_INDICES.items():
        if idx < len(landmarks):
            lm = landmarks[idx]
            result[joint_name] = Landmark(
                x=lm.get("x", 0),
                y=lm.get("y", 0),
                z=lm.get("z", 0),
                visibility=lm.get("visibility", 0)
            )
    return result


class BaseExercise(ABC):
    """Abstract base class for exercise correction modules."""
    
    def __init__(self):
        self.rep_count = 0
        self.current_phase = "idle"
        self._phase_history: list[str] = []
        self._last_angles: Optional[JointAngles] = None
        self._rep_counter: Optional[HysteresisRepCounter] = None
    
    def _create_rep_counter(self, upper_threshold: float, lower_threshold: float) -> HysteresisRepCounter:
        """
        Create a hysteresis-based rep counter.
        
        Args:
            upper_threshold: Angle at extended/starting position
            lower_threshold: Angle at contracted/bottom position
        """
        self._rep_counter = HysteresisRepCounter(
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            min_rep_duration=0.5,
            max_rep_duration=10.0
        )
        return self._rep_counter
    
    def update_rep_counter(
        self, 
        angle: float, 
        left_angle: Optional[float] = None,
        right_angle: Optional[float] = None,
        form_violations: Optional[list[str]] = None
    ) -> tuple[str, bool]:
        """
        Update rep counter with current angle.
        
        Returns:
            Tuple of (phase_string, rep_just_completed)
        """
        if not self._rep_counter:
            return "idle", False
        
        phase, rep_completed = self._rep_counter.update(
            angle, left_angle, right_angle, form_violations
        )
        
        if rep_completed:
            self.rep_count = self._rep_counter.rep_count
        
        return phase.value, rep_completed
    
    @property
    def rep_quality(self) -> Optional[float]:
        """Get average rep quality score."""
        if self._rep_counter:
            return self._rep_counter.average_quality
        return None
    
    @property
    def partial_reps(self) -> int:
        """Get count of partial/invalid reps."""
        if self._rep_counter:
            return self._rep_counter.partial_reps
        return 0
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Exercise name for display."""
        pass
    
    @property
    @abstractmethod
    def required_joints(self) -> list[JointName]:
        """List of joints required for this exercise."""
        pass
    
    @abstractmethod
    def check_form(self, landmarks: dict[JointName, Landmark]) -> ExerciseResult:
        """
        Analyze landmarks and return form assessment.
        
        Args:
            landmarks: Dictionary mapping joint names to landmark data
            
        Returns:
            ExerciseResult with form validity, violations, and corrections
        """
        pass
    
    @abstractmethod
    def detect_rep_phase(self, landmarks: dict[JointName, Landmark]) -> str:
        """
        Detect current phase of the repetition.
        
        Returns:
            Phase string: "up", "down", "hold", "transition"
        """
        pass
    
    def process_frame(self, landmarks: list[dict]) -> ExerciseResult:
        """
        Process a single frame of landmark data.
        
        Args:
            landmarks: Raw landmark list from MediaPipe (33 landmarks)
            
        Returns:
            ExerciseResult with full analysis
        """
        # Convert to landmark dictionary
        landmark_dict = landmarks_to_dict(landmarks)
        
        # Check if required joints are visible
        visibility_threshold = 0.5
        for joint in self.required_joints:
            if joint not in landmark_dict:
                return ExerciseResult(
                    is_valid=False,
                    rep_count=self.rep_count,
                    rep_phase="unknown",
                    violations=["Required joints not visible"],
                    corrections=["Please ensure your full body is in frame"]
                )
            if landmark_dict[joint].visibility < visibility_threshold:
                return ExerciseResult(
                    is_valid=False,
                    rep_count=self.rep_count,
                    rep_phase="unknown",
                    violations=[f"{joint.value} not clearly visible"],
                    corrections=["Adjust camera angle or lighting"]
                )
        
        # Detect rep phase
        current_phase = self.detect_rep_phase(landmark_dict)
        
        # Count reps on phase transitions
        if self._phase_history:
            last_phase = self._phase_history[-1]
            if self._is_rep_complete(last_phase, current_phase):
                self.rep_count += 1
        
        self._phase_history.append(current_phase)
        if len(self._phase_history) > 30:
            self._phase_history.pop(0)
        
        # Check form
        result = self.check_form(landmark_dict)
        result.rep_count = self.rep_count
        result.rep_phase = current_phase
        result.rep_quality = self.rep_quality
        result.partial_reps = self.partial_reps
        
        return result
    
    def _is_rep_complete(self, last_phase: str, current_phase: str) -> bool:
        """Check if a rep was completed based on phase transition."""
        # Rep complete when transitioning from "up" back to starting position
        return last_phase == "up" and current_phase == "down"
    
    def reset(self) -> None:
        """Reset rep counter and phase history."""
        self.rep_count = 0
        self.current_phase = "idle"
        self._phase_history = []
        self._last_angles = None
        if self._rep_counter:
            self._rep_counter.reset()
