"""Bicep curl exercise module with form correction."""

from typing import Optional
from collections import deque
import numpy as np
from .base import (
    BaseExercise,
    ExerciseResult,
    JointAngles,
    JointName,
    Landmark,
    calculate_angle,
)


class BicepCurlModule(BaseExercise):
    """Bicep curl exercise detection and form correction.
    
    Supports both standing and seated bicep curls.
    """
    
    # Form thresholds - IMPROVED for better detection
    MIN_ELBOW_ANGLE = 50   # Was 30 - top of curl (more forgiving)
    MAX_ELBOW_ANGLE = 145  # Was 160 - bottom of curl (more forgiving)
    
    # Hysteresis to prevent flickering between states
    ANGLE_HYSTERESIS = 15  # Degrees of buffer
    
    # Form thresholds - slightly relaxed for real-world conditions
    ELBOW_DRIFT_THRESHOLD = 0.06  # Was 0.03 - normalized X difference
    BODY_SWING_THRESHOLD = 0.04   # Was 0.02 - allowed shoulder movement
    WRIST_CURL_THRESHOLD = 20     # Wrist angle deviation
    
    # Detection settings
    MIN_VISIBILITY = 0.3   # Lowered from implicit 0.5
    SMOOTHING_WINDOW = 5   # Number of frames for angle smoothing
    MIN_FRAMES_TO_CONFIRM = 3  # Frames needed to confirm state change
    
    # Seated detection
    SEATED_HIP_ANGLE_MIN = 70   # Hip angle when seated (bent at hips)
    SEATED_HIP_ANGLE_MAX = 120  # 
    SEATED_KNEE_ANGLE_MIN = 60  # Knee angle when seated
    SEATED_KNEE_ANGLE_MAX = 110 #
    
    def __init__(self):
        super().__init__()
        self._lowest_elbow_angle = 180.0
        self._highest_elbow_angle = 0.0
        self._in_curl = False
        self._initial_shoulder_x: Optional[float] = None
        self._initial_shoulder_y: Optional[float] = None
        self._active_arm: Optional[str] = None  # "left", "right", or "both"
        self._is_seated: bool = False  # Track if user is seated
        
        # Smoothing buffers for angle readings
        self._left_elbow_buffer: deque = deque(maxlen=self.SMOOTHING_WINDOW)
        self._right_elbow_buffer: deque = deque(maxlen=self.SMOOTHING_WINDOW)
        
        # State confirmation counters
        self._curl_confirm_count = 0
        self._extend_confirm_count = 0
        self._pending_state_change: Optional[str] = None
        
        # Seated detection buffer
        self._seated_frame_count = 0
        
        # Create hysteresis-based rep counter for stable counting
        self._create_rep_counter(
            upper_threshold=self.MAX_ELBOW_ANGLE - 10,  # ~135 degrees
            lower_threshold=self.MIN_ELBOW_ANGLE + 20    # ~70 degrees
        )
    
    @property
    def name(self) -> str:
        return "Bicep Curl"
    
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
            JointName.LEFT_KNEE,
            JointName.RIGHT_KNEE,
        ]
    
    def _detect_seated_position(self, landmarks: dict[JointName, Landmark]) -> bool:
        """Detect if user is in a seated position.
        
        Seated indicators:
        - Hip angle is bent (not straight like standing)
        - Knee angle is around 90 degrees
        - Hip Y position is lower relative to knee (sitting)
        """
        try:
            # Calculate hip angle (shoulder-hip-knee)
            left_hip_angle = calculate_angle(
                landmarks[JointName.LEFT_SHOULDER],
                landmarks[JointName.LEFT_HIP],
                landmarks[JointName.LEFT_KNEE]
            )
            right_hip_angle = calculate_angle(
                landmarks[JointName.RIGHT_SHOULDER],
                landmarks[JointName.RIGHT_HIP],
                landmarks[JointName.RIGHT_KNEE]
            )
            avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
            
            # Calculate knee angle
            left_knee_angle = calculate_angle(
                landmarks[JointName.LEFT_HIP],
                landmarks[JointName.LEFT_KNEE],
                landmarks[JointName.LEFT_ANKLE] if JointName.LEFT_ANKLE in landmarks else landmarks[JointName.LEFT_KNEE]
            )
            right_knee_angle = calculate_angle(
                landmarks[JointName.RIGHT_HIP],
                landmarks[JointName.RIGHT_KNEE],
                landmarks[JointName.RIGHT_ANKLE] if JointName.RIGHT_ANKLE in landmarks else landmarks[JointName.RIGHT_KNEE]
            )
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            
            # Check seated criteria
            hip_bent = self.SEATED_HIP_ANGLE_MIN < avg_hip_angle < self.SEATED_HIP_ANGLE_MAX
            knee_bent = self.SEATED_KNEE_ANGLE_MIN < avg_knee_angle < self.SEATED_KNEE_ANGLE_MAX
            
            # Also check relative positions - when seated, hips are at similar height to knees
            hip_y = (landmarks[JointName.LEFT_HIP].y + landmarks[JointName.RIGHT_HIP].y) / 2
            knee_y = (landmarks[JointName.LEFT_KNEE].y + landmarks[JointName.RIGHT_KNEE].y) / 2
            hips_near_knees = abs(hip_y - knee_y) < 0.15  # Close vertical position
            
            is_seated = (hip_bent or knee_bent) and hips_near_knees
            
            # Use frame counter for stability
            if is_seated:
                self._seated_frame_count = min(self._seated_frame_count + 1, 10)
            else:
                self._seated_frame_count = max(self._seated_frame_count - 1, 0)
            
            return self._seated_frame_count >= 5
            
        except (KeyError, TypeError):
            return False
    
    def _calculate_angles(self, landmarks: dict[JointName, Landmark]) -> JointAngles:
        """Calculate all relevant joint angles for bicep curl analysis with smoothing."""
        angles = JointAngles()
        
        # Check visibility before calculating
        left_visible = self._check_arm_visibility(landmarks, "left")
        right_visible = self._check_arm_visibility(landmarks, "right")
        
        # Left elbow angle (shoulder-elbow-wrist) with smoothing
        if left_visible:
            raw_left = calculate_angle(
                landmarks[JointName.LEFT_SHOULDER],
                landmarks[JointName.LEFT_ELBOW],
                landmarks[JointName.LEFT_WRIST]
            )
            self._left_elbow_buffer.append(raw_left)
            angles.left_elbow = sum(self._left_elbow_buffer) / len(self._left_elbow_buffer)
        else:
            angles.left_elbow = 180.0  # Default extended
        
        # Right elbow angle with smoothing
        if right_visible:
            raw_right = calculate_angle(
                landmarks[JointName.RIGHT_SHOULDER],
                landmarks[JointName.RIGHT_ELBOW],
                landmarks[JointName.RIGHT_WRIST]
            )
            self._right_elbow_buffer.append(raw_right)
            angles.right_elbow = sum(self._right_elbow_buffer) / len(self._right_elbow_buffer)
        else:
            angles.right_elbow = 180.0  # Default extended
        
        # Shoulder angles for detecting arm raise (shoulder-hip-knee equivalent)
        angles.left_shoulder = calculate_angle(
            landmarks[JointName.LEFT_ELBOW],
            landmarks[JointName.LEFT_SHOULDER],
            landmarks[JointName.LEFT_HIP]
        )
        
        angles.right_shoulder = calculate_angle(
            landmarks[JointName.RIGHT_ELBOW],
            landmarks[JointName.RIGHT_SHOULDER],
            landmarks[JointName.RIGHT_HIP]
        )
        
        self._last_angles = angles
        return angles
    
    def _check_arm_visibility(self, landmarks: dict[JointName, Landmark], side: str) -> bool:
        """Check if arm landmarks are visible enough for detection."""
        if side == "left":
            joints = [JointName.LEFT_SHOULDER, JointName.LEFT_ELBOW, JointName.LEFT_WRIST]
        else:
            joints = [JointName.RIGHT_SHOULDER, JointName.RIGHT_ELBOW, JointName.RIGHT_WRIST]
        
        return all(landmarks[j].visibility > self.MIN_VISIBILITY for j in joints)
    
    def _detect_active_arm(self, landmarks: dict[JointName, Landmark]) -> str:
        """Detect which arm(s) are performing the curl."""
        angles = self._last_angles or self._calculate_angles(landmarks)
        
        left_visible = self._check_arm_visibility(landmarks, "left")
        right_visible = self._check_arm_visibility(landmarks, "right")
        
        # If only one arm is visible, use that one
        if left_visible and not right_visible:
            return "left"
        if right_visible and not left_visible:
            return "right"
        if not left_visible and not right_visible:
            return "both"  # Fallback
        
        # Both visible - check which is moving more
        left_moving = angles.left_elbow < self.MAX_ELBOW_ANGLE - 15
        right_moving = angles.right_elbow < self.MAX_ELBOW_ANGLE - 15
        
        if left_moving and right_moving:
            return "both"
        elif left_moving:
            return "left"
        elif right_moving:
            return "right"
        
        # Neither clearly moving yet - check which has more curl
        left_curl_amount = self.MAX_ELBOW_ANGLE - angles.left_elbow
        right_curl_amount = self.MAX_ELBOW_ANGLE - angles.right_elbow
        
        if abs(left_curl_amount - right_curl_amount) < 10:
            return "both"
        return "left" if left_curl_amount > right_curl_amount else "right"
    
    def _check_elbow_drift(self, landmarks: dict[JointName, Landmark]) -> tuple[bool, bool]:
        """
        Check if elbows are drifting away from the body.
        Returns (left_drift, right_drift).
        """
        left_drift = False
        right_drift = False
        
        # Only check if arm is visible
        if self._check_arm_visibility(landmarks, "left"):
            left_elbow_x = landmarks[JointName.LEFT_ELBOW].x
            left_shoulder_x = landmarks[JointName.LEFT_SHOULDER].x
            left_drift = abs(left_elbow_x - left_shoulder_x) > self.ELBOW_DRIFT_THRESHOLD
        
        if self._check_arm_visibility(landmarks, "right"):
            right_elbow_x = landmarks[JointName.RIGHT_ELBOW].x
            right_shoulder_x = landmarks[JointName.RIGHT_SHOULDER].x
            right_drift = abs(right_elbow_x - right_shoulder_x) > self.ELBOW_DRIFT_THRESHOLD
        
        return left_drift, right_drift
    
    def _check_body_swing(self, landmarks: dict[JointName, Landmark]) -> bool:
        """Check if body is swinging to generate momentum."""
        mid_shoulder_x = (landmarks[JointName.LEFT_SHOULDER].x + 
                         landmarks[JointName.RIGHT_SHOULDER].x) / 2
        mid_shoulder_y = (landmarks[JointName.LEFT_SHOULDER].y + 
                         landmarks[JointName.RIGHT_SHOULDER].y) / 2
        
        # Initialize reference position
        if self._initial_shoulder_x is None:
            self._initial_shoulder_x = mid_shoulder_x
            self._initial_shoulder_y = mid_shoulder_y
            return False
        
        # Check for excessive movement
        x_movement = abs(mid_shoulder_x - self._initial_shoulder_x)
        y_movement = abs(mid_shoulder_y - self._initial_shoulder_y)
        
        return x_movement > self.BODY_SWING_THRESHOLD or y_movement > self.BODY_SWING_THRESHOLD
    
    def _check_full_rom(self, angles: JointAngles) -> tuple[bool, bool]:
        """
        Check if full range of motion is being achieved.
        Returns (full_extension, full_contraction).
        """
        # Use the active arm's angle
        if self._active_arm == "left":
            elbow_angle = angles.left_elbow
        elif self._active_arm == "right":
            elbow_angle = angles.right_elbow
        else:
            elbow_angle = (angles.left_elbow + angles.right_elbow) / 2
        
        # Track min/max through the rep
        self._lowest_elbow_angle = min(self._lowest_elbow_angle, elbow_angle)
        self._highest_elbow_angle = max(self._highest_elbow_angle, elbow_angle)
        
        full_extension = self._highest_elbow_angle > self.MAX_ELBOW_ANGLE - 20
        full_contraction = self._lowest_elbow_angle < self.MIN_ELBOW_ANGLE + 20
        
        return full_extension, full_contraction
    
    def detect_rep_phase(self, landmarks: dict[JointName, Landmark]) -> str:
        """Detect current phase of bicep curl repetition with hysteresis rep counter."""
        angles = self._calculate_angles(landmarks)
        
        # Detect which arm is active
        self._active_arm = self._detect_active_arm(landmarks)
        
        # Use active arm's angle
        if self._active_arm == "left":
            elbow_angle = angles.left_elbow
            left_angle = angles.left_elbow
            right_angle = None
        elif self._active_arm == "right":
            elbow_angle = angles.right_elbow
            left_angle = None
            right_angle = angles.right_elbow
        else:
            elbow_angle = min(angles.left_elbow, angles.right_elbow)  # Use most curled
            left_angle = angles.left_elbow
            right_angle = angles.right_elbow
        
        # Use hysteresis-based rep counter for stable phase detection
        if self._rep_counter:
            phase_str, rep_completed = self.update_rep_counter(
                angle=elbow_angle,
                left_angle=left_angle,
                right_angle=right_angle
            )
            
            # Track min/max for ROM validation
            self._lowest_elbow_angle = min(self._lowest_elbow_angle, elbow_angle)
            self._highest_elbow_angle = max(self._highest_elbow_angle, elbow_angle)
            
            # Update internal state to match counter
            if phase_str in ["up", "down", "hold"]:
                self._in_curl = phase_str in ["up", "hold"]
            
            if rep_completed:
                # Reset tracking for next rep
                self._lowest_elbow_angle = 180.0
                self._highest_elbow_angle = 0.0
            
            return phase_str
        
        # Fallback to legacy detection if counter not available
        # Apply hysteresis for state transitions
        if self._in_curl:
            # Currently in curl - need to extend past threshold + hysteresis to exit
            extension_threshold = self.MAX_ELBOW_ANGLE - self.ANGLE_HYSTERESIS
            contraction_threshold = self.MIN_ELBOW_ANGLE + self.ANGLE_HYSTERESIS
        else:
            # Currently extended - need to curl past threshold - hysteresis to enter
            extension_threshold = self.MAX_ELBOW_ANGLE
            contraction_threshold = self.MIN_ELBOW_ANGLE + (self.ANGLE_HYSTERESIS * 2)
        
        if elbow_angle > extension_threshold:
            # Arms extended (bottom position)
            if self._in_curl:
                # Confirm with multiple frames
                self._extend_confirm_count += 1
                if self._extend_confirm_count >= self.MIN_FRAMES_TO_CONFIRM:
                    self._in_curl = False
                    self._extend_confirm_count = 0
                    self._curl_confirm_count = 0
                    # Reset tracking for next rep
                    self._lowest_elbow_angle = 180.0
                    self._highest_elbow_angle = 0.0
                    return "down"
                return "down"  # Transitioning down
            return "idle"
        elif elbow_angle < contraction_threshold:
            # Top of curl (contracted)
            self._curl_confirm_count += 1
            if self._curl_confirm_count >= self.MIN_FRAMES_TO_CONFIRM:
                self._in_curl = True
                self._curl_confirm_count = 0
                self._extend_confirm_count = 0
            return "hold"
        else:
            # Transitioning - reset confirm counters
            self._curl_confirm_count = 0
            self._extend_confirm_count = 0
            
            if self._in_curl:
                if elbow_angle > self._lowest_elbow_angle + 15:
                    return "down"
                return "up"
            else:
                self._in_curl = True
                return "up"
    
    def check_form(self, landmarks: dict[JointName, Landmark]) -> ExerciseResult:
        """Analyze bicep curl form and return corrections.
        
        Supports both standing and seated bicep curls.
        """
        violations = []
        corrections = []
        joint_colors = {}
        
        # Calculate angles
        angles = self._calculate_angles(landmarks)
        
        # Detect if user is seated
        self._is_seated = self._detect_seated_position(landmarks)
        
        # Initialize all tracked joints as green
        for joint in self.required_joints:
            joint_colors[joint.value] = "green"
        
        is_valid = True
        
        # Check visibility - provide feedback if arms not visible
        left_visible = self._check_arm_visibility(landmarks, "left")
        right_visible = self._check_arm_visibility(landmarks, "right")
        
        if not left_visible and not right_visible:
            return ExerciseResult(
                is_valid=False,
                rep_count=self.rep_count,
                rep_phase=self.current_phase,
                violations=["Arms not visible"],
                corrections=["Adjust camera to see your arms clearly"],
                joint_colors=joint_colors,
                confidence=0.3,
                angles=angles
            )
        
        # Check elbow drift (only for visible arms)
        left_drift, right_drift = self._check_elbow_drift(landmarks)
        
        if left_drift and self._active_arm in ["left", "both"] and left_visible:
            violations.append("Left elbow drifting forward")
            corrections.append("Keep your left elbow pinned to your side")
            joint_colors[JointName.LEFT_ELBOW.value] = "red"
            is_valid = False
        
        if right_drift and self._active_arm in ["right", "both"] and right_visible:
            violations.append("Right elbow drifting forward")
            corrections.append("Keep your right elbow pinned to your side")
            joint_colors[JointName.RIGHT_ELBOW.value] = "red"
            is_valid = False
        
        # Check body swing - less strict for seated (upper body can move slightly)
        swing_threshold = self.BODY_SWING_THRESHOLD * 1.5 if self._is_seated else self.BODY_SWING_THRESHOLD
        if self._check_body_swing(landmarks):
            if not self._is_seated:  # Only warn for standing
                violations.append("Body swinging for momentum")
                corrections.append("Keep your torso still - use only your biceps")
                joint_colors[JointName.LEFT_SHOULDER.value] = "yellow"
                joint_colors[JointName.RIGHT_SHOULDER.value] = "yellow"
            # Update reference position
            self._initial_shoulder_x = None
        
        # Check range of motion - only warn if clearly incomplete
        full_extension, full_contraction = self._check_full_rom(angles)
        
        # Only check ROM if we're in an active curl (reduce false positives)
        if self._in_curl and self.current_phase == "hold" and not full_contraction:
            # Be lenient - only flag if significantly incomplete
            active_angle = angles.left_elbow if self._active_arm == "left" else angles.right_elbow
            if self._active_arm == "both":
                active_angle = min(angles.left_elbow, angles.right_elbow)
            
            if active_angle > self.MIN_ELBOW_ANGLE + 30:  # More than 30 degrees off
                violations.append("Incomplete curl at top")
                corrections.append("Squeeze your biceps fully at the top")
                joint_colors[JointName.LEFT_WRIST.value] = "yellow"
                joint_colors[JointName.RIGHT_WRIST.value] = "yellow"
        
        # Check symmetry for alternating or simultaneous curls (more lenient)
        if self._active_arm == "both" and left_visible and right_visible:
            elbow_asymmetry = abs(angles.left_elbow - angles.right_elbow)
            if elbow_asymmetry > 35:  # Was 25 - more forgiving
                violations.append("Uneven arm movement")
                corrections.append("Curl both arms at the same pace")
                if angles.left_elbow > angles.right_elbow:
                    joint_colors[JointName.LEFT_ELBOW.value] = "yellow"
                else:
                    joint_colors[JointName.RIGHT_ELBOW.value] = "yellow"
        
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
        """Rep complete when returning to bottom from top."""
        return last_phase == "down" and current_phase == "idle"
    
    def reset(self) -> None:
        """Reset all tracking state."""
        super().reset()
        self._lowest_elbow_angle = 180.0
        self._highest_elbow_angle = 0.0
        self._in_curl = False
        self._initial_shoulder_x = None
        self._initial_shoulder_y = None
        self._active_arm = None
        self._is_seated = False
        self._left_elbow_buffer.clear()
        self._right_elbow_buffer.clear()
        self._curl_confirm_count = 0
        self._extend_confirm_count = 0
        self._pending_state_change = None
        self._seated_frame_count = 0


class AlternateBicepCurlModule(BicepCurlModule):
    """Alternate bicep curl - alternating between left and right arms.
    
    Each arm performs a complete rep before switching to the other arm.
    Counts each arm's rep separately but displays total reps.
    
    Supports:
    - Standing alternate curls
    - Seated alternate curls
    - Tracking left/right arm balance
    """
    
    def __init__(self):
        super().__init__()
        
        # Alternate curl specific state
        self._current_arm: str = "left"  # Which arm should curl next
        self._left_rep_count: int = 0
        self._right_rep_count: int = 0
        self._last_curling_arm: Optional[str] = None
        
        # Per-arm state tracking
        self._left_in_curl: bool = False
        self._right_in_curl: bool = False
        self._left_lowest_angle: float = 180.0
        self._right_lowest_angle: float = 180.0
        
        # Arm switch detection
        self._arm_switch_pending: bool = False
        self._frames_since_last_rep: int = 0
        self._max_frames_between_switch: int = 60  # ~2 seconds at 30fps
        
        # Create separate rep counters for each arm
        from utils.rep_counter import HysteresisRepCounter
        self._left_rep_counter = HysteresisRepCounter(
            upper_threshold=self.MAX_ELBOW_ANGLE - 10,
            lower_threshold=self.MIN_ELBOW_ANGLE + 20,
            min_rep_duration=0.5,
            max_rep_duration=8.0
        )
        self._right_rep_counter = HysteresisRepCounter(
            upper_threshold=self.MAX_ELBOW_ANGLE - 10,
            lower_threshold=self.MIN_ELBOW_ANGLE + 20,
            min_rep_duration=0.5,
            max_rep_duration=8.0
        )
    
    @property
    def name(self) -> str:
        return "Alternate Bicep Curl"
    
    @property
    def current_arm(self) -> str:
        """Get the arm that is currently curling or should curl next."""
        return self._current_arm
    
    @property
    def left_reps(self) -> int:
        """Get left arm rep count."""
        return self._left_rep_count
    
    @property
    def right_reps(self) -> int:
        """Get right arm rep count."""
        return self._right_rep_count
    
    @property
    def arm_balance(self) -> float:
        """Get arm balance ratio (1.0 = perfectly balanced)."""
        total = self._left_rep_count + self._right_rep_count
        if total == 0:
            return 1.0
        min_reps = min(self._left_rep_count, self._right_rep_count)
        max_reps = max(self._left_rep_count, self._right_rep_count)
        return min_reps / max_reps if max_reps > 0 else 1.0
    
    def _detect_curling_arm(self, landmarks: dict[JointName, Landmark]) -> Optional[str]:
        """Detect which arm is currently performing a curl.
        
        For alternate curls, only one arm should be curling at a time.
        Returns 'left', 'right', or None if neither is clearly curling.
        """
        angles = self._last_angles or self._calculate_angles(landmarks)
        
        left_visible = self._check_arm_visibility(landmarks, "left")
        right_visible = self._check_arm_visibility(landmarks, "right")
        
        # Check which arm is in a curled position
        left_curling = (left_visible and 
                       angles.left_elbow < self.MAX_ELBOW_ANGLE - 20 and
                       angles.left_elbow < angles.right_elbow - 15)
        
        right_curling = (right_visible and 
                        angles.right_elbow < self.MAX_ELBOW_ANGLE - 20 and
                        angles.right_elbow < angles.left_elbow - 15)
        
        if left_curling and not right_curling:
            return "left"
        elif right_curling and not left_curling:
            return "right"
        elif left_curling and right_curling:
            # Both curling - this shouldn't happen in alternate curls
            # Return the one that's more curled
            return "left" if angles.left_elbow < angles.right_elbow else "right"
        
        return None
    
    def detect_rep_phase(self, landmarks: dict[JointName, Landmark]) -> str:
        """Detect current phase for alternate bicep curls.
        
        Tracks each arm independently and switches focus when one arm completes.
        """
        angles = self._calculate_angles(landmarks)
        self._frames_since_last_rep += 1
        
        # Detect which arm is actively curling
        curling_arm = self._detect_curling_arm(landmarks)
        
        # Update both arm states
        left_phase, left_completed = self._left_rep_counter.update(
            angles.left_elbow,
            left_angle=angles.left_elbow
        )
        right_phase, right_completed = self._right_rep_counter.update(
            angles.right_elbow,
            right_angle=angles.right_elbow
        )
        
        # Track lowest angles for each arm
        self._left_lowest_angle = min(self._left_lowest_angle, angles.left_elbow)
        self._right_lowest_angle = min(self._right_lowest_angle, angles.right_elbow)
        
        # Handle rep completion for each arm
        if left_completed:
            self._left_rep_count += 1
            self.rep_count = self._left_rep_count + self._right_rep_count
            self._last_curling_arm = "left"
            self._current_arm = "right"  # Switch to other arm
            self._frames_since_last_rep = 0
            self._left_lowest_angle = 180.0
            self._left_in_curl = False
        
        if right_completed:
            self._right_rep_count += 1
            self.rep_count = self._left_rep_count + self._right_rep_count
            self._last_curling_arm = "right"
            self._current_arm = "left"  # Switch to other arm
            self._frames_since_last_rep = 0
            self._right_lowest_angle = 180.0
            self._right_in_curl = False
        
        # Update curl state based on current position
        self._left_in_curl = angles.left_elbow < self.MAX_ELBOW_ANGLE - 20
        self._right_in_curl = angles.right_elbow < self.MAX_ELBOW_ANGLE - 20
        
        # Determine overall phase based on which arm is active
        if curling_arm == "left":
            return left_phase.value if hasattr(left_phase, 'value') else str(left_phase)
        elif curling_arm == "right":
            return right_phase.value if hasattr(right_phase, 'value') else str(right_phase)
        else:
            # Neither curling - use the expected arm's phase
            if self._current_arm == "left":
                return left_phase.value if hasattr(left_phase, 'value') else str(left_phase)
            else:
                return right_phase.value if hasattr(right_phase, 'value') else str(right_phase)
    
    def check_form(self, landmarks: dict[JointName, Landmark]) -> ExerciseResult:
        """Analyze alternate bicep curl form.
        
        Additional checks for alternate curls:
        - One arm should stay extended while other curls
        - Arms should alternate (not both curl together)
        - Balance between left and right reps
        """
        # Get base form check
        result = super().check_form(landmarks)
        
        angles = self._last_angles or self._calculate_angles(landmarks)
        
        # Check for proper alternation (one arm extended, one curling)
        left_extended = angles.left_elbow > self.MAX_ELBOW_ANGLE - 25
        right_extended = angles.right_elbow > self.MAX_ELBOW_ANGLE - 25
        left_curled = angles.left_elbow < self.MIN_ELBOW_ANGLE + 30
        right_curled = angles.right_elbow < self.MIN_ELBOW_ANGLE + 30
        
        # Warn if both arms are curling at the same time
        if left_curled and right_curled:
            result.violations.append("Both arms curling together")
            result.corrections.append("Keep one arm extended while curling the other")
            result.joint_colors[JointName.LEFT_ELBOW.value] = "yellow"
            result.joint_colors[JointName.RIGHT_ELBOW.value] = "yellow"
        
        # Check arm balance (after a few reps)
        total_reps = self._left_rep_count + self._right_rep_count
        if total_reps >= 4:
            balance = self.arm_balance
            if balance < 0.7:
                weaker_arm = "left" if self._left_rep_count < self._right_rep_count else "right"
                result.violations.append(f"Arm imbalance detected")
                result.corrections.append(f"Focus on {weaker_arm} arm - it has fewer reps")
        
        # Check that non-curling arm stays extended
        curling_arm = self._detect_curling_arm(landmarks)
        if curling_arm == "left" and not right_extended:
            if angles.right_elbow < self.MAX_ELBOW_ANGLE - 35:
                result.violations.append("Right arm not staying extended")
                result.corrections.append("Keep right arm straight while curling left")
                result.joint_colors[JointName.RIGHT_ELBOW.value] = "yellow"
        elif curling_arm == "right" and not left_extended:
            if angles.left_elbow < self.MAX_ELBOW_ANGLE - 35:
                result.violations.append("Left arm not staying extended")
                result.corrections.append("Keep left arm straight while curling right")
                result.joint_colors[JointName.LEFT_ELBOW.value] = "yellow"
        
        return result
    
    def get_arm_stats(self) -> dict:
        """Get detailed statistics for each arm."""
        return {
            "left": {
                "rep_count": self._left_rep_count,
                "quality": self._left_rep_counter.average_quality,
                "partial_reps": self._left_rep_counter.partial_reps,
                "in_curl": self._left_in_curl,
            },
            "right": {
                "rep_count": self._right_rep_count,
                "quality": self._right_rep_counter.average_quality,
                "partial_reps": self._right_rep_counter.partial_reps,
                "in_curl": self._right_in_curl,
            },
            "balance": self.arm_balance,
            "total_reps": self.rep_count,
            "current_arm": self._current_arm,
        }
    
    def reset(self) -> None:
        """Reset all tracking state for alternate curls."""
        super().reset()
        self._current_arm = "left"
        self._left_rep_count = 0
        self._right_rep_count = 0
        self._last_curling_arm = None
        self._left_in_curl = False
        self._right_in_curl = False
        self._left_lowest_angle = 180.0
        self._right_lowest_angle = 180.0
        self._arm_switch_pending = False
        self._frames_since_last_rep = 0
        self._left_rep_counter.reset()
        self._right_rep_counter.reset()
