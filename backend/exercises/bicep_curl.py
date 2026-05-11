"""Bicep curl exercise module with form correction."""

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


class BicepCurlModule(BaseExercise):
    """Bicep curl exercise detection and form correction.

    Supports both standing and seated bicep curls.
    """

    # Mechanical→semantic mapping is SWAPPED for bicep curl: when the elbow
    # angle decreases (mechanical eccentric), the user is curling up — that
    # is the *concentric* phase of the lift. When the angle increases
    # (mechanical concentric), they are lowering — the lift's *eccentric*.
    SEMANTIC_PHASE_MAP = {
        "idle": "idle",
        "setup": "setup",
        "eccentric": "concentric",
        "concentric": "eccentric",
        "hold": "hold",
    }

    PHASE_DISPLAY = {
        "idle": "",
        "setup": "Arms extended",
        "eccentric": "Lowering",
        "concentric": "Curling up",
        "hold": "At top",
    }

    # Form thresholds
    MIN_ELBOW_ANGLE = 50    # top of curl (fully contracted)
    MAX_ELBOW_ANGLE = 160   # bottom of curl (fully extended)

    ANGLE_HYSTERESIS = 15

    ELBOW_DRIFT_THRESHOLD = 0.06
    BODY_SWING_THRESHOLD = 0.04
    WRIST_CURL_THRESHOLD = 20

    MIN_VISIBILITY = 0.3
    MIN_FRAMES_TO_CONFIRM = 3
    # Rep counter ROM gate. The previous 140°/130° (10° band) accepted
    # micro-movements as reps; the briefly-tried 160°/60° required near-
    # perfect ROM. 150°/80° = 70° band — achievable for normal curls and
    # passes find_peaks prominence (25°) cleanly. The new rep counter
    # smooths internally via Savitzky-Golay, so no per-module pre-smoother
    # is needed (used to add ~250ms phase-transition lag).
    REP_EXTENDED_THRESHOLD = 150
    REP_CONTRACTED_THRESHOLD = 80
    
    # Seated detection
    SEATED_HIP_ANGLE_MIN = 70   # Hip angle when seated (bent at hips)
    SEATED_HIP_ANGLE_MAX = 120  # 
    SEATED_KNEE_ANGLE_MIN = 60  # Knee angle when seated
    SEATED_KNEE_ANGLE_MAX = 110 #
    
    def __init__(self):
        super().__init__()
        self._lowest_elbow_angle = 180.0
        self._highest_elbow_angle = 0.0
        self._initial_shoulder_x: Optional[float] = None
        self._initial_shoulder_y: Optional[float] = None
        self._active_arm: Optional[str] = None  # "left", "right", or "both"
        self._is_seated: bool = False  # Track if user is seated

        # State confirmation counters
        self._curl_confirm_count = 0
        self._extend_confirm_count = 0
        self._pending_state_change: Optional[str] = None

        # Seated detection buffer
        self._seated_frame_count = 0

        # Rep counter: Savitzky-Golay smoothing + find_peaks lives inside
        # HysteresisRepCounter, so no per-module pre-smoothing is needed.
        self._create_rep_counter(
            upper_threshold=self.REP_EXTENDED_THRESHOLD,
            lower_threshold=self.REP_CONTRACTED_THRESHOLD,
            exercise_key="bicep_curl",
            min_rep_duration=0.3,
            max_rep_duration=8.0,
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
        ]

    def _lower_body_visible(self, landmarks: dict[JointName, Landmark]) -> bool:
        joints = [
            JointName.LEFT_HIP,
            JointName.RIGHT_HIP,
            JointName.LEFT_KNEE,
            JointName.RIGHT_KNEE,
        ]
        return all(landmarks[j].visibility > self.MIN_VISIBILITY for j in joints)
    
    def _detect_seated_position(self, landmarks: dict[JointName, Landmark]) -> bool:
        """Detect if user is in a seated position.
        
        Seated indicators:
        - Hip angle is bent (not straight like standing)
        - Knee angle is around 90 degrees
        - Hip Y position is lower relative to knee (sitting)
        """
        try:
            if not self._lower_body_visible(landmarks):
                return False
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
        """Calculate all relevant joint angles for bicep curl analysis.

        Raw angles are fed straight to the rep counter; HysteresisRepCounter's
        Savitzky-Golay window handles smoothing and computes velocity for
        phase output. Pre-smoothing here on top of that used to delay phase
        transitions by ~250ms.
        """
        angles = JointAngles()

        left_visible = self._check_arm_visibility(landmarks, "left")
        right_visible = self._check_arm_visibility(landmarks, "right")

        if left_visible:
            angles.left_elbow = calculate_angle(
                landmarks[JointName.LEFT_SHOULDER],
                landmarks[JointName.LEFT_ELBOW],
                landmarks[JointName.LEFT_WRIST]
            )
        else:
            angles.left_elbow = 180.0  # Default extended

        if right_visible:
            angles.right_elbow = calculate_angle(
                landmarks[JointName.RIGHT_SHOULDER],
                landmarks[JointName.RIGHT_ELBOW],
                landmarks[JointName.RIGHT_WRIST]
            )
        else:
            angles.right_elbow = 180.0  # Default extended
        
        # Shoulder angles need hips. Half-body curl framing is valid, so skip
        # this form cue when lower body is not visible.
        if self._lower_body_visible(landmarks):
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
        else:
            angles.left_shoulder = 0.0
            angles.right_shoulder = 0.0
        
        self._last_angles = angles
        return angles
    
    def _check_arm_visibility(self, landmarks: dict[JointName, Landmark], side: str) -> bool:
        """Check if arm landmarks are visible enough for detection."""
        if side == "left":
            joints = [JointName.LEFT_SHOULDER, JointName.LEFT_ELBOW, JointName.LEFT_WRIST]
        else:
            joints = [JointName.RIGHT_SHOULDER, JointName.RIGHT_ELBOW, JointName.RIGHT_WRIST]

        return all(landmarks[j].visibility > self.MIN_VISIBILITY for j in joints)

    def _arm_visibility(self, landmarks: dict[JointName, Landmark], side: str) -> float:
        if side == "left":
            joints = [JointName.LEFT_SHOULDER, JointName.LEFT_ELBOW, JointName.LEFT_WRIST]
        else:
            joints = [JointName.RIGHT_SHOULDER, JointName.RIGHT_ELBOW, JointName.RIGHT_WRIST]
        return min(landmarks[j].visibility for j in joints)

    def _active_arm_visibility(self, landmarks: dict[JointName, Landmark]) -> float:
        """Return the worst visibility across the active arm's joints."""
        if self._active_arm == "left":
            return self._arm_visibility(landmarks, "left")
        if self._active_arm == "right":
            return self._arm_visibility(landmarks, "right")
        return min(self._arm_visibility(landmarks, "left"), self._arm_visibility(landmarks, "right"))
    
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
        if not self._lower_body_visible(landmarks):
            return False
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
        """Detect current phase of bicep curl repetition using hysteresis rep counter."""
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
            # Both arms - use the most curled angle
            elbow_angle = min(angles.left_elbow, angles.right_elbow)
            left_angle = angles.left_elbow
            right_angle = angles.right_elbow
        
        # Use hysteresis-based rep counter for stable phase detection
        if self._rep_counter:
            # Visibility = the active arm's worst-visible required joint.
            arm_vis = self._active_arm_visibility(landmarks)
            phase_str, rep_completed = self.update_rep_counter(
                angle=elbow_angle,
                left_angle=left_angle,
                right_angle=right_angle,
                visibility=arm_vis,
            )
            
            # Track min/max for ROM validation and depth checking
            self._lowest_elbow_angle = min(self._lowest_elbow_angle, elbow_angle)
            self._highest_elbow_angle = max(self._highest_elbow_angle, elbow_angle)
            
            if rep_completed:
                # Reset tracking for next rep
                self._lowest_elbow_angle = 180.0
                self._highest_elbow_angle = 0.0
            
            return phase_str
        
        # Fallback: Should not reach here if rep counter is properly initialized
        return "idle"
    
    def check_form(self, landmarks: dict[JointName, Landmark]) -> ExerciseResult:
        """Analyze bicep curl form and return corrections.
        
        Supports both standing and seated bicep curls.
        """
        violations = []
        corrections = []
        joint_colors = {}
        
        # Reuse the angles calculated by detect_rep_phase for this frame.
        # Recomputing here advances the smoothing buffers twice and can make
        # the rep counter reject valid reps as partial.
        angles = self._last_angles or self._calculate_angles(landmarks)
        
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
        if self.current_phase in ("concentric", "hold") and not full_contraction:
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
        """Rep complete when returning to extension from curl."""
        return last_phase == "eccentric" and current_phase in ("setup", "idle")
    
    def reset(self) -> None:
        """Reset all tracking state."""
        super().reset()
        self._lowest_elbow_angle = 180.0
        self._highest_elbow_angle = 0.0
        self._initial_shoulder_x = None
        self._initial_shoulder_y = None
        self._active_arm = None
        self._is_seated = False
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
        
        # Per-arm rep counters. Each runs its own Savitzky-Golay smoother
        # internally, so per-arm pre-smoothing is unnecessary.
        from pipeline.rep_counter import HysteresisRepCounter, _params_for
        params = _params_for("alternate_bicep_curl")
        self._left_rep_counter = HysteresisRepCounter(
            upper_threshold=self.REP_EXTENDED_THRESHOLD,
            lower_threshold=self.REP_CONTRACTED_THRESHOLD,
            min_rep_duration=0.2,
            max_rep_duration=8.0,
            **params,
        )
        self._right_rep_counter = HysteresisRepCounter(
            upper_threshold=self.REP_EXTENDED_THRESHOLD,
            lower_threshold=self.REP_CONTRACTED_THRESHOLD,
            min_rep_duration=0.2,
            max_rep_duration=8.0,
            **params,
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
        """Detect current phase for alternate bicep curls using per-arm rep counters.
        
        Tracks each arm independently with separate rep counters and switches focus 
        when one arm completes a full rep.
        """
        angles = self._calculate_angles(landmarks)
        self._frames_since_last_rep += 1
        
        # Detect which arm is actively curling
        curling_arm = self._detect_curling_arm(landmarks)
        
        # Update both arm counters
        left_vis = self._arm_visibility(landmarks, "left")
        right_vis = self._arm_visibility(landmarks, "right")
        left_phase, left_completed = self._left_rep_counter.update(
            angles.left_elbow,
            left_angle=angles.left_elbow,
            visibility=left_vis,
        )
        right_phase, right_completed = self._right_rep_counter.update(
            angles.right_elbow,
            right_angle=angles.right_elbow,
            visibility=right_vis,
        )
        
        # Track lowest angles for each arm (for depth validation)
        self._left_lowest_angle = min(self._left_lowest_angle, angles.left_elbow)
        self._right_lowest_angle = min(self._right_lowest_angle, angles.right_elbow)
        
        # Handle rep completion for each arm.
        # A "rep" in alternate curl = one full cycle of BOTH arms, so the
        # surfaced rep_count is min(left_reps, right_reps). Doing 3 lefts
        # then 3 rights still ends at 3 (not 6), and a single arm alone
        # cannot increment the count.
        if left_completed:
            self._left_rep_count += 1
            self._last_curling_arm = "left"
            self._current_arm = "right"
            self._frames_since_last_rep = 0
            self._left_lowest_angle = 180.0
            self._left_in_curl = False

        if right_completed:
            self._right_rep_count += 1
            self._last_curling_arm = "right"
            self._current_arm = "left"
            self._frames_since_last_rep = 0
            self._right_lowest_angle = 180.0
            self._right_in_curl = False

        if left_completed or right_completed:
            self.rep_count = min(self._left_rep_count, self._right_rep_count)
        
        # Update curl state based on current position
        self._left_in_curl = angles.left_elbow < self.MAX_ELBOW_ANGLE - 20
        self._right_in_curl = angles.right_elbow < self.MAX_ELBOW_ANGLE - 20
        
        # Determine overall phase based on which arm is active and map to client-friendly phase
        if curling_arm == "left":
            return self._to_semantic(left_phase.value)
        elif curling_arm == "right":
            return self._to_semantic(right_phase.value)
        else:
            # Neither curling - use the expected arm's phase
            if self._current_arm == "left":
                return self._to_semantic(left_phase.value)
            else:
                return self._to_semantic(right_phase.value)
    
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
        
        # Warn if both arms are curling at the same time, but only after the
        # user has actually completed at least one alternating rep. Otherwise
        # this fires immediately when the classifier mis-locks regular bicep
        # curls into the alternate module — the user is doing both arms
        # together because that IS the exercise they want, not a form fault.
        if left_curled and right_curled and self.rep_count >= 1:
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
