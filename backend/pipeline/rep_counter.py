"""
Advanced rep counting with hysteresis and state machine.
Moved to pipeline package; utils/rep_counter.py re-exports from here.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import time


class RepPhase(str, Enum):
    """Mechanical rep counter phase, semantic to angle direction.

    ECCENTRIC = angle moving toward lower threshold (joint flexing).
    CONCENTRIC = angle moving toward upper threshold (joint extending).

    Each exercise module maps these mechanical phases to user-facing
    eccentric/concentric labels — for squat/pushup the mapping is identity;
    for bicep curl the mapping is swapped (flexion = concentric of the lift).
    """
    IDLE = "idle"
    READY = "setup"
    ECCENTRIC = "eccentric"
    CONCENTRIC = "concentric"
    HOLD = "hold"
    TRANSITION = "setup"


@dataclass
class RepQuality:
    form_score: float = 1.0
    depth_score: float = 1.0
    tempo_score: float = 1.0
    symmetry_score: float = 1.0

    @property
    def overall_score(self) -> float:
        return (
            self.form_score * 0.4 +
            self.depth_score * 0.3 +
            self.tempo_score * 0.15 +
            self.symmetry_score * 0.15
        )


@dataclass
class RepData:
    rep_number: int
    quality: RepQuality
    duration_seconds: float
    min_angle: float
    max_angle: float
    timestamp: float = field(default_factory=time.time)


class HysteresisRepCounter:
    def __init__(
        self,
        upper_threshold: float,
        lower_threshold: float,
        min_rep_duration: float = 0.5,
        max_rep_duration: float = 10.0,
        require_full_extension: bool = True
    ):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.min_rep_duration = min_rep_duration
        self.max_rep_duration = max_rep_duration
        self.require_full_extension = require_full_extension

        self._phase = RepPhase.IDLE
        self._rep_count = 0
        self._partial_rep_count = 0
        self._rep_start_time: Optional[float] = None
        self._min_angle_in_rep: float = 180.0
        self._max_angle_in_rep: float = 0.0
        self._angle_history: deque[float] = deque(maxlen=30)
        self._reps: list[RepData] = []
        self._form_violations: list[str] = []
        self._left_angle: float = 0.0
        self._right_angle: float = 0.0
        self._has_seen_upper: bool = False
        self._min_visibility_in_rep: float = 1.0
        self._low_visibility_streak: int = 0

    def update(
        self,
        angle: float,
        left_angle: Optional[float] = None,
        right_angle: Optional[float] = None,
        form_violations: Optional[list[str]] = None,
        visibility: Optional[float] = None,
    ) -> tuple[RepPhase, bool]:
        self._angle_history.append(angle)
        self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
        self._max_angle_in_rep = max(self._max_angle_in_rep, angle)

        if left_angle is not None:
            self._left_angle = left_angle
        if right_angle is not None:
            self._right_angle = right_angle
        if form_violations:
            self._form_violations.extend(form_violations)

        # Visibility gating: if required joints drop out of frame, abort the
        # in-flight rep. Prevents counting partial reps when the user steps
        # out and back into view mid-set.
        if visibility is not None:
            self._min_visibility_in_rep = min(self._min_visibility_in_rep, visibility)
            if visibility < 0.3:
                self._low_visibility_streak += 1
                if self._low_visibility_streak >= 3 and self._phase in (
                    RepPhase.ECCENTRIC,
                    RepPhase.CONCENTRIC,
                ):
                    self._phase = RepPhase.READY
                    self._min_angle_in_rep = 180.0
                    self._max_angle_in_rep = 0.0
                    self._min_visibility_in_rep = 1.0
                    self._rep_start_time = time.time()
                    self._form_violations = []
                    return self._phase, False
            else:
                self._low_visibility_streak = 0

        current_time = time.time()
        rep_completed = False
        if angle >= self.upper_threshold:
            self._has_seen_upper = True

        if self._phase == RepPhase.IDLE:
            # Start in READY - wait for user to reach extended position before counting down
            self._phase = RepPhase.READY
            self._rep_start_time = current_time
            self._min_angle_in_rep = angle
            self._max_angle_in_rep = angle

        elif self._phase == RepPhase.READY:
            # READY: Waiting for the motion to start (angle must go down first)
            # Only transition to ECCENTRIC if angle goes DOWN from upper threshold
            if angle < self.upper_threshold and (
                self._has_seen_upper or angle > self.lower_threshold
            ):
                self._phase = RepPhase.ECCENTRIC
                self._rep_start_time = current_time
                self._min_angle_in_rep = angle
                self._max_angle_in_rep = angle
                self._form_violations = []

        elif self._phase == RepPhase.ECCENTRIC:
            # ECCENTRIC (down): Going from extended to contracted
            # Transition to CONCENTRIC when reaching lower threshold
            if angle < self.lower_threshold:
                self._phase = RepPhase.CONCENTRIC
            # If we somehow go back up without reaching lower, go back to READY
            elif angle > self.upper_threshold + 5:
                self._phase = RepPhase.READY

        elif self._phase == RepPhase.CONCENTRIC:
            # CONCENTRIC (up): Going from contracted back to extended
            # Complete rep when returning to upper threshold
            if angle > self.upper_threshold:
                rep_duration = current_time - (self._rep_start_time or current_time)
                if self._is_valid_rep(rep_duration):
                    self._rep_count += 1
                    rep_completed = True
                    quality = self._calculate_rep_quality(rep_duration)
                    self._reps.append(RepData(
                        rep_number=self._rep_count,
                        quality=quality,
                        duration_seconds=rep_duration,
                        min_angle=self._min_angle_in_rep,
                        max_angle=self._max_angle_in_rep
                    ))
                else:
                    self._partial_rep_count += 1
                self._phase = RepPhase.READY
                self._min_angle_in_rep = 180.0
                self._max_angle_in_rep = 0.0
                self._min_visibility_in_rep = 1.0
                self._rep_start_time = current_time
                self._form_violations = []

        # Timeout: If rep is taking too long, reset to READY
        if self._rep_start_time and (current_time - self._rep_start_time) > self.max_rep_duration:
            if self._phase in [RepPhase.ECCENTRIC, RepPhase.CONCENTRIC]:
                self._phase = RepPhase.READY
                self._rep_start_time = current_time

        return self._phase, rep_completed

    def record_violations(self, violations: list[str]) -> None:
        """Append form violations for the in-progress rep (called after form check)."""
        self._form_violations.extend(violations)

    def _is_valid_rep(self, duration: float) -> bool:
        if duration < self.min_rep_duration or duration > self.max_rep_duration:
            return False
        if self.require_full_extension:
            if self._max_angle_in_rep < self.upper_threshold:
                return False
            if self._min_angle_in_rep > self.lower_threshold:
                return False
        # NB: do NOT reject on per-rep minimum visibility. MediaPipe routinely
        # reports 0.4–0.6 visibility for partly-occluded but clearly-visible
        # joints; a single low-visibility frame would poison the entire rep.
        # The 3-frame "vis < 0.3" streak abort in update() already handles
        # the real failure mode (user steps out of frame).
        return True

    def _calculate_rep_quality(self, duration: float) -> RepQuality:
        form_score = max(0.0, 1.0 - len(self._form_violations) * 0.15)
        expected_range = self.upper_threshold - self.lower_threshold
        actual_range = self._max_angle_in_rep - self._min_angle_in_rep
        depth_score = min(1.0, actual_range / max(expected_range, 1))
        if 2.0 <= duration <= 4.0:
            tempo_score = 1.0
        elif duration < 1.0:
            tempo_score = 0.5
        elif duration > 6.0:
            tempo_score = 0.7
        else:
            tempo_score = 0.85
        if self._left_angle > 0 and self._right_angle > 0:
            symmetry_score = max(0.0, 1.0 - abs(self._left_angle - self._right_angle) / 30)
        else:
            symmetry_score = 1.0
        return RepQuality(
            form_score=form_score, depth_score=depth_score,
            tempo_score=tempo_score, symmetry_score=symmetry_score
        )

    @property
    def rep_count(self) -> int:
        return self._rep_count

    @property
    def phase(self) -> RepPhase:
        return self._phase

    @property
    def phase_str(self) -> str:
        return self._phase.value

    @property
    def partial_reps(self) -> int:
        return self._partial_rep_count

    @property
    def average_quality(self) -> float:
        if not self._reps:
            return 0.0
        return sum(r.quality.overall_score for r in self._reps) / len(self._reps)

    @property
    def last_rep(self) -> Optional[RepData]:
        return self._reps[-1] if self._reps else None

    def reset(self):
        self._phase = RepPhase.IDLE
        self._rep_count = 0
        self._partial_rep_count = 0
        self._rep_start_time = None
        self._min_angle_in_rep = 180.0
        self._max_angle_in_rep = 0.0
        self._angle_history.clear()
        self._reps.clear()
        self._form_violations.clear()
        self._has_seen_upper = False
        self._min_visibility_in_rep = 1.0
        self._low_visibility_streak = 0


class ExerciseRepCounter:
    @staticmethod
    def create_bicep_curl_counter() -> HysteresisRepCounter:
        return HysteresisRepCounter(upper_threshold=150, lower_threshold=50,
                                    min_rep_duration=0.8, max_rep_duration=8.0)

    @staticmethod
    def create_squat_counter() -> HysteresisRepCounter:
        return HysteresisRepCounter(upper_threshold=160, lower_threshold=90,
                                    min_rep_duration=1.0, max_rep_duration=10.0)

    @staticmethod
    def create_pushup_counter() -> HysteresisRepCounter:
        return HysteresisRepCounter(upper_threshold=155, lower_threshold=90,
                                    min_rep_duration=0.8, max_rep_duration=8.0)
