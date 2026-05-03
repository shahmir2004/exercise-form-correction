"""Form evaluation with temporal violation aggregation."""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import numpy as np

from .features import BodyFrame


@dataclass
class Violation:
    code: str
    severity: str  # "yellow" | "red"
    joints: list = field(default_factory=list)
    message: str = ""
    correction: str = ""


class ViolationAggregator:
    """
    Sliding window per violation code.
    Emits violation only if it persists for M of last N frames.
    Cool-down suppresses duplicate emission for K frames.
    """

    def __init__(self, m: int = 4, n: int = 6, cooldown: int = 15):
        self.m = m
        self.n = n
        self.cooldown = cooldown
        # Per violation code: deque of bool (present/absent per frame)
        self._windows: dict[str, deque] = {}
        # Per violation code: frames remaining in cooldown
        self._cooldowns: dict[str, int] = {}

    def update(self, violations: list[Violation]) -> list[Violation]:
        """
        Feed current-frame violations; return stable (aggregated) violations.
        """
        current_codes = {v.code for v in violations}
        viol_by_code = {v.code: v for v in violations}

        # Tick all windows
        all_codes = set(self._windows.keys()) | current_codes
        for code in all_codes:
            if code not in self._windows:
                self._windows[code] = deque(maxlen=self.n)
            self._windows[code].append(code in current_codes)

        # Tick cooldowns
        for code in list(self._cooldowns.keys()):
            self._cooldowns[code] -= 1
            if self._cooldowns[code] <= 0:
                del self._cooldowns[code]

        # Emit violations that pass the M-of-N gate and aren't in cooldown
        emitted = []
        for code in current_codes:
            window = self._windows[code]
            if sum(window) >= self.m and code not in self._cooldowns:
                emitted.append(viol_by_code[code])
                self._cooldowns[code] = self.cooldown

        return emitted

    def reset(self):
        self._windows.clear()
        self._cooldowns.clear()


class FormEvaluator:
    """
    Runs per-exercise form check predicates on BodyFrame,
    feeds raw violations through ViolationAggregator for stability.
    """

    def __init__(self, aggregator: Optional[ViolationAggregator] = None):
        self._aggregator = aggregator or ViolationAggregator()

    def evaluate(self, frame: BodyFrame, exercise_name: Optional[str]) -> list[Violation]:
        """
        Evaluate form for the given exercise and return stable violations.
        exercise_name: "squat" | "pushup" | "bicep_curl" | "alternate_bicep_curl" | None
        """
        if exercise_name is None:
            return []

        raw = self._check_form(frame, exercise_name)
        return self._aggregator.update(raw)

    def _check_form(self, frame: BodyFrame, exercise_name: str) -> list[Violation]:
        """Dispatch to per-exercise form checks."""
        if exercise_name == "squat":
            return self._check_squat(frame)
        elif exercise_name == "pushup":
            return self._check_pushup(frame)
        elif exercise_name in ("bicep_curl", "alternate_bicep_curl"):
            return self._check_curl(frame, alternate=(exercise_name == "alternate_bicep_curl"))
        return []

    def _check_squat(self, frame: BodyFrame) -> list[Violation]:
        violations = []
        angles = frame.angles
        coords = frame.coords

        left_knee = angles.get("left_knee", 180.0)
        right_knee = angles.get("right_knee", 180.0)
        torso = angles.get("torso_angle", 0.0)

        # Knee valgus: knee X closer to midline than ankle X
        lk = coords[25]; la = coords[27]  # left knee, left ankle (hip-relative)
        rk = coords[26]; ra = coords[28]  # right knee, right ankle
        # In hip-relative coords, left side is negative x, right positive x
        # Valgus: knee drifts toward center relative to ankle
        left_valgus = bool((lk[0] - la[0]) > 0.05)
        right_valgus = bool((ra[0] - rk[0]) > 0.05)

        if left_valgus:
            violations.append(Violation(
                code="left_knee_valgus", severity="red",
                joints=["left_knee"],
                message="Left knee caving inward",
                correction="Push your left knee outward over your toes"
            ))
        if right_valgus:
            violations.append(Violation(
                code="right_knee_valgus", severity="red",
                joints=["right_knee"],
                message="Right knee caving inward",
                correction="Push your right knee outward over your toes"
            ))

        # Excessive forward lean — skip when profile view (z-axis not reliable front-on)
        from .features import ViewEstimate
        if frame.view_estimate not in (ViewEstimate.FRONTAL,) and torso > 45.0:
            violations.append(Violation(
                code="forward_lean", severity="yellow",
                joints=["left_shoulder", "right_shoulder"],
                message="Excessive forward lean",
                correction="Keep your chest up and back straighter"
            ))

        # Insufficient depth (only when actively squatting — knees bent)
        avg_knee = (left_knee + right_knee) / 2
        if avg_knee < 130 and avg_knee > 80:
            # In mid-squat, check depth
            pass  # depth check requires rep phase context — skip here

        # Knee asymmetry
        if abs(left_knee - right_knee) > 15:
            violations.append(Violation(
                code="knee_asymmetry", severity="yellow",
                joints=["left_knee", "right_knee"],
                message="Uneven knee bend",
                correction="Distribute weight evenly on both legs"
            ))

        return violations

    def _check_pushup(self, frame: BodyFrame) -> list[Violation]:
        violations = []
        coords = frame.coords
        angles = frame.angles

        # Hip sag / pike — compare hip Y to shoulder-ankle midpoint
        # In hip-relative normalized coords: hip origin ≈ 0
        # shoulder is near top (negative y), ankle near bottom (positive y)
        ls_y = float(coords[11][1]); rs_y = float(coords[12][1])
        la_y = float(coords[27][1]); ra_y = float(coords[28][1])
        mid_shoulder_y = (ls_y + rs_y) / 2.0
        mid_ankle_y = (la_y + ra_y) / 2.0
        expected_hip_y = (mid_shoulder_y + mid_ankle_y) / 2.0
        # hip_y in hip-relative coords ≈ 0 by definition; deviation from 0 is the sag
        actual_hip_y = 0.0  # origin
        sag = actual_hip_y - expected_hip_y
        if sag > 0.1:
            violations.append(Violation(
                code="hip_sag", severity="red",
                joints=["left_hip", "right_hip"],
                message="Hips sagging",
                correction="Engage your core and lift your hips in line with shoulders"
            ))
        elif sag < -0.1:
            violations.append(Violation(
                code="hip_pike", severity="yellow",
                joints=["left_hip", "right_hip"],
                message="Hips too high",
                correction="Lower your hips to form a straight line"
            ))

        # Elbow flare: shoulder angle (elbow-shoulder-hip) > 75°
        left_shoulder_ang = angles.get("left_shoulder", 45.0)
        right_shoulder_ang = angles.get("right_shoulder", 45.0)
        if left_shoulder_ang > 75.0:
            violations.append(Violation(
                code="left_elbow_flare", severity="red",
                joints=["left_elbow"],
                message="Left elbow flaring out",
                correction="Tuck your left elbow closer to your body (45° angle)"
            ))
        if right_shoulder_ang > 75.0:
            violations.append(Violation(
                code="right_elbow_flare", severity="red",
                joints=["right_elbow"],
                message="Right elbow flaring out",
                correction="Tuck your right elbow closer to your body (45° angle)"
            ))

        # Elbow asymmetry
        left_elbow = angles.get("left_elbow", 180.0)
        right_elbow = angles.get("right_elbow", 180.0)
        if abs(left_elbow - right_elbow) > 20:
            violations.append(Violation(
                code="elbow_asymmetry", severity="yellow",
                joints=["left_elbow", "right_elbow"],
                message="Uneven arm bend",
                correction="Distribute weight evenly on both arms"
            ))

        return violations

    def _check_curl(self, frame: BodyFrame, alternate: bool = False) -> list[Violation]:
        violations = []
        coords = frame.coords
        angles = frame.angles

        left_elbow = angles.get("left_elbow", 180.0)
        right_elbow = angles.get("right_elbow", 180.0)

        # Elbow drift: elbow X far from shoulder X
        # In hip-relative coords
        le = coords[13]; ls = coords[11]  # left elbow, left shoulder
        re = coords[14]; rs = coords[12]  # right elbow, right shoulder
        left_drift = bool(abs(float(le[0]) - float(ls[0])) > 0.12)
        right_drift = bool(abs(float(re[0]) - float(rs[0])) > 0.12)

        if left_drift:
            violations.append(Violation(
                code="left_elbow_drift", severity="red",
                joints=["left_elbow"],
                message="Left elbow drifting forward",
                correction="Keep your left elbow pinned to your side"
            ))
        if right_drift:
            violations.append(Violation(
                code="right_elbow_drift", severity="red",
                joints=["right_elbow"],
                message="Right elbow drifting forward",
                correction="Keep your right elbow pinned to your side"
            ))

        # Elbow asymmetry (both curls)
        if abs(left_elbow - right_elbow) > 35 and not alternate:
            violations.append(Violation(
                code="curl_asymmetry", severity="yellow",
                joints=["left_elbow", "right_elbow"],
                message="Uneven arm movement",
                correction="Curl both arms at the same pace"
            ))

        # Alternate curl specific: both arms curling simultaneously
        if alternate:
            left_curled = left_elbow < 110.0
            right_curled = right_elbow < 110.0
            if left_curled and right_curled:
                violations.append(Violation(
                    code="both_arms_curling", severity="yellow",
                    joints=["left_elbow", "right_elbow"],
                    message="Both arms curling together",
                    correction="Keep one arm extended while curling the other"
                ))

        return violations

    def reset(self):
        self._aggregator.reset()
