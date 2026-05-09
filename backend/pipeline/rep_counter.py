"""
Rep counting via signal processing — Savitzky-Golay smoothing + scipy.find_peaks.

Replaces the previous Schmitt-trigger threshold logic with a denoise-then-find-peaks
pipeline. Public API is preserved, so callers (`exercises/*.py`,
`state_machine/manager.py`) continue to work unchanged.

Phase output is derived from the smoothed angle's first derivative; a single
noisy frame near a threshold no longer flips the phase.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque
import time

import numpy as np
from scipy.signal import savgol_filter, find_peaks


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


# Defaults; per-exercise overrides come from settings.REP_DETECTION via the
# exercise modules' constructor flow (see exercises/base.py:_create_rep_counter).
_DEFAULT_SMOOTH_WINDOW = 9
_DEFAULT_POLYORDER = 3
_DEFAULT_PROMINENCE = 25.0          # degrees
_DEFAULT_MIN_REP_FRAMES = 8         # peak-to-peak minimum distance

# Velocity thresholds for phase output (degrees / second).
_VELOCITY_HOLD_THRESHOLD = 8.0
_VELOCITY_DIR_THRESHOLD = 4.0
_PHASE_STILL_RANGE_DEGREES = 15.0
_PHASE_ENDPOINT_MARGIN_DEGREES = 8.0

# Visibility-based abort. The previous "3 frames < 0.3" rule killed reps on
# transient occlusion; this version requires a longer streak at a stricter
# threshold AND a per-rep minimum, so single bad frames never poison a rep.
_VISIBILITY_ABORT_THRESHOLD = 0.2
_VISIBILITY_ABORT_FRAMES = 4
_VISIBILITY_REP_MIN = 0.25


class HysteresisRepCounter:
    """Drop-in rep counter — same public surface as before, smarter internals.

    Public surface preserved:
      - constructor: upper_threshold, lower_threshold, min_rep_duration,
        max_rep_duration, require_full_extension
      - .update(angle, left_angle, right_angle, form_violations, visibility)
        -> (RepPhase, rep_completed: bool)
      - .rep_count, .partial_reps, .phase, .phase_str, .last_rep,
        .average_quality, .reset(), .record_violations()

    New optional kwargs (with sensible defaults so existing call sites keep
    working): smooth_window, polyorder, prominence, min_rep_frames.
    """

    def __init__(
        self,
        upper_threshold: float,
        lower_threshold: float,
        min_rep_duration: float = 0.5,
        max_rep_duration: float = 10.0,
        require_full_extension: bool = True,
        *,
        smooth_window: int = _DEFAULT_SMOOTH_WINDOW,
        polyorder: int = _DEFAULT_POLYORDER,
        prominence: float = _DEFAULT_PROMINENCE,
        min_rep_frames: int = _DEFAULT_MIN_REP_FRAMES,
    ):
        self.upper_threshold = float(upper_threshold)
        self.lower_threshold = float(lower_threshold)
        self.min_rep_duration = float(min_rep_duration)
        self.max_rep_duration = float(max_rep_duration)
        self.require_full_extension = bool(require_full_extension)

        # savgol_filter requires an odd window strictly greater than polyorder.
        sw = int(smooth_window)
        if sw % 2 == 0:
            sw += 1
        po = int(polyorder)
        if po >= sw:
            po = sw - 1
        self.smooth_window = sw
        self.polyorder = po
        self.prominence = float(prominence)
        self.min_rep_frames = max(2, int(min_rep_frames))

        # Rolling buffer ~3s @ 20fps so find_peaks has room to register a full
        # cycle even after smoothing-edge effects.
        self._buffer_size = max(60, self.smooth_window * 6)
        self._angles: deque = deque(maxlen=self._buffer_size)
        self._timestamps: deque = deque(maxlen=self._buffer_size)
        self._visibilities: deque = deque(maxlen=self._buffer_size)

        # Monotonic global frame index — survives buffer rotation so we can
        # tell whether a peak/trough returned by find_peaks is one we've
        # already counted.
        self._frame_idx: int = 0
        self._buffer_start_idx: int = 0  # global idx of self._angles[0]

        # Pending-rep bookkeeping.
        self._last_trough_idx: int = -1
        self._last_peak_idx: int = -1
        self._pending_trough_idx: Optional[int] = None
        self._pending_trough_value: float = 180.0
        self._pending_trough_min_vis: float = 1.0
        self._last_peak_value: Optional[float] = None
        self._last_peak_time: Optional[float] = None
        self._threshold_in_rep: bool = False
        self._threshold_start_time: Optional[float] = None
        self._threshold_min_angle: float = 180.0
        self._threshold_min_vis: float = 1.0

        # Output state.
        self._phase: RepPhase = RepPhase.IDLE
        self._rep_count: int = 0
        self._partial_rep_count: int = 0
        self._reps: list[RepData] = []
        self._form_violations: list[str] = []
        self._left_angle: float = 0.0
        self._right_angle: float = 0.0
        self._low_visibility_streak: int = 0

        # Cached for callers that want to inspect the smoothed signal.
        self._smoothed_angle: float = 0.0
        self._velocity_dps: float = 0.0

    def update(
        self,
        angle: float,
        left_angle: Optional[float] = None,
        right_angle: Optional[float] = None,
        form_violations: Optional[list[str]] = None,
        visibility: Optional[float] = None,
    ) -> tuple[RepPhase, bool]:
        now = time.time()
        self._frame_idx += 1
        # Track which global frame index now corresponds to buffer[0].
        if len(self._angles) == self._buffer_size:
            self._buffer_start_idx += 1
        elif len(self._angles) == 0:
            self._buffer_start_idx = self._frame_idx
        self._angles.append(float(angle))
        self._timestamps.append(now)
        self._visibilities.append(float(visibility) if visibility is not None else 1.0)
        if left_angle is not None:
            self._left_angle = float(left_angle)
        if right_angle is not None:
            self._right_angle = float(right_angle)
        if form_violations:
            self._form_violations.extend(form_violations)

        # Soft visibility-abort: only triggers on a sustained streak at a
        # stricter threshold than the old logic, so transient occlusion is
        # tolerated and only true frame-exits kill a pending rep.
        if visibility is not None:
            if visibility < _VISIBILITY_ABORT_THRESHOLD:
                self._low_visibility_streak += 1
                if self._low_visibility_streak >= _VISIBILITY_ABORT_FRAMES:
                    self._pending_trough_idx = None
                    self._threshold_in_rep = False
                    self._threshold_start_time = None
                    self._phase = RepPhase.READY
            else:
                self._low_visibility_streak = 0

        # Until the smoothing window is full, fall back to a simple "engaged"
        # output so the upstream UI sees a phase string. No reps can be
        # counted until we have at least one full smoothing window of data.
        if len(self._angles) < self.smooth_window:
            self._smoothed_angle = float(angle)
            self._velocity_dps = 0.0
            if self._phase == RepPhase.IDLE:
                self._phase = RepPhase.READY
            return self._phase, self._scan_threshold_rep(float(angle), now)

        arr = np.fromiter(self._angles, dtype=float)
        ts_arr = np.fromiter(self._timestamps, dtype=float)

        smoothed = savgol_filter(
            arr, window_length=self.smooth_window, polyorder=self.polyorder
        )
        velocity_per_sample = savgol_filter(
            arr,
            window_length=self.smooth_window,
            polyorder=self.polyorder,
            deriv=1,
        )
        # Convert deg/sample to deg/second using the local mean dt.
        win = min(self.smooth_window, len(ts_arr))
        dts = np.diff(ts_arr[-win:])
        mean_dt = float(np.mean(dts)) if dts.size > 0 and float(np.mean(dts)) > 0 else 1.0 / 20.0
        self._smoothed_angle = float(smoothed[-1])
        self._velocity_dps = float(velocity_per_sample[-1]) / max(mean_dt, 1e-3)

        # Phase from velocity sign, with endpoint stillness checked first.
        # At top/bottom, tiny landmark jitter can create alternating velocity
        # signs even though the body is effectively holding position.
        v = self._velocity_dps
        phase_win = min(self.smooth_window, len(smoothed))
        recent_range = float(np.ptp(smoothed[-phase_win:])) if phase_win > 1 else 0.0
        near_top = self._smoothed_angle >= self.upper_threshold - _PHASE_ENDPOINT_MARGIN_DEGREES
        near_bottom = self._smoothed_angle <= self.lower_threshold + _PHASE_ENDPOINT_MARGIN_DEGREES
        raw_near_top = float(angle) >= self.upper_threshold - _PHASE_ENDPOINT_MARGIN_DEGREES
        raw_near_bottom = float(angle) <= self.lower_threshold + _PHASE_ENDPOINT_MARGIN_DEGREES
        is_latched_top_hold = raw_near_top and near_top and self._phase in {
            RepPhase.CONCENTRIC,
            RepPhase.HOLD,
        }
        is_latched_bottom_hold = raw_near_bottom and near_bottom and self._phase in {
            RepPhase.ECCENTRIC,
            RepPhase.HOLD,
        }
        if is_latched_top_hold or is_latched_bottom_hold:
            new_phase = RepPhase.HOLD
        elif recent_range <= _PHASE_STILL_RANGE_DEGREES:
            new_phase = RepPhase.HOLD if self._phase != RepPhase.IDLE else RepPhase.READY
        elif abs(v) < _VELOCITY_HOLD_THRESHOLD:
            new_phase = RepPhase.HOLD if self._phase != RepPhase.IDLE else RepPhase.READY
        elif v < -_VELOCITY_DIR_THRESHOLD:
            new_phase = RepPhase.ECCENTRIC
        elif v > _VELOCITY_DIR_THRESHOLD:
            new_phase = RepPhase.CONCENTRIC
        else:
            new_phase = self._phase if self._phase != RepPhase.IDLE else RepPhase.READY
        self._phase = new_phase

        rep_completed = self._scan_threshold_rep(float(angle), now)
        if rep_completed:
            return self._phase, True

        rep_completed = self._scan_for_completed_rep(smoothed, ts_arr)
        # Plateau finalizer: if a rep ended at a sustained "hold at top" the
        # smoothed signal rises into a plateau without a descent — find_peaks
        # cannot register a peak there. Detect rising→plateau as a rep end
        # so the last rep of a set isn't dropped.
        if not rep_completed and self._pending_trough_idx is not None:
            rep_completed = self._maybe_finalize_plateau_rep(smoothed, ts_arr)
        return self._phase, rep_completed

    def _scan_threshold_rep(self, angle: float, now: float) -> bool:
        """Fast Schmitt-style fallback for short, clean rep cycles.

        The peak detector is still useful for noisy real motion, but it can
        miss exhibition-speed cycles when there are too few samples around the
        peak/trough. This fallback counts a clear lower-threshold crossing
        followed by a full return to the upper threshold.
        """
        current_vis = float(self._visibilities[-1]) if self._visibilities else 1.0

        if not self._threshold_in_rep and angle <= self.lower_threshold:
            self._threshold_in_rep = True
            self._threshold_start_time = now
            self._threshold_min_angle = angle
            self._threshold_min_vis = current_vis
            return False

        if not self._threshold_in_rep:
            return False

        self._threshold_min_angle = min(self._threshold_min_angle, angle)
        self._threshold_min_vis = min(self._threshold_min_vis, current_vis)

        if angle < self.upper_threshold:
            return False

        start_time = self._threshold_start_time or now
        duration = now - start_time
        trough_value = self._threshold_min_angle
        peak_value = angle

        self._threshold_in_rep = False
        self._threshold_start_time = None

        if not self._is_valid_rep(
            duration=duration,
            peak_value=peak_value,
            trough_value=trough_value,
            min_vis=self._threshold_min_vis,
        ):
            self._partial_rep_count += 1
            return False

        self._rep_count += 1
        self._reps.append(
            RepData(
                rep_number=self._rep_count,
                quality=self._calculate_rep_quality(
                    duration, peak_value, trough_value
                ),
                duration_seconds=duration,
                min_angle=trough_value,
                max_angle=peak_value,
            )
        )
        self._last_peak_idx = self._frame_idx
        self._last_peak_value = peak_value
        self._last_peak_time = now
        self._pending_trough_idx = None
        self._form_violations = []
        return True

    def _maybe_finalize_plateau_rep(
        self, smoothed: np.ndarray, ts_arr: np.ndarray
    ) -> bool:
        """Close out a rep when velocity decays from positive to ~0 with the
        smoothed angle past upper_threshold.

        Triggered when the user holds at the top of the rep (e.g. end of a set
        or brief pause between reps where they don't fully reset).
        """
        if abs(self._velocity_dps) > _VELOCITY_HOLD_THRESHOLD * 0.6:
            return False
        if self._smoothed_angle < self.upper_threshold:
            return False
        # Confirm there was a clear rising trend in the recent window.
        win = min(self.smooth_window, len(smoothed) - 1)
        if win < 3:
            return False
        recent_delta = float(smoothed[-1] - smoothed[-win - 1])
        if recent_delta < self.prominence * 0.5:
            return False

        # Use the current frame's smoothed angle / time as the peak.
        peak_value = float(self._smoothed_angle)
        peak_time = float(ts_arr[-1])
        peak_global = self._buffer_start_idx + (len(smoothed) - 1)
        if peak_global <= self._last_peak_idx:
            return False

        start_time = (
            self._last_peak_time
            if self._last_peak_time is not None
            else float(ts_arr[0])
        )
        rep_duration = peak_time - start_time

        if self._is_valid_rep(
            duration=rep_duration,
            peak_value=peak_value,
            trough_value=self._pending_trough_value,
            min_vis=self._pending_trough_min_vis,
        ):
            self._rep_count += 1
            self._reps.append(
                RepData(
                    rep_number=self._rep_count,
                    quality=self._calculate_rep_quality(
                        rep_duration, peak_value, self._pending_trough_value
                    ),
                    duration_seconds=rep_duration,
                    min_angle=self._pending_trough_value,
                    max_angle=peak_value,
                )
            )
            self._last_peak_idx = peak_global
            self._last_peak_value = peak_value
            self._last_peak_time = peak_time
            self._pending_trough_idx = None
            self._form_violations = []
            return True
        return False

    def _scan_for_completed_rep(
        self, smoothed: np.ndarray, ts_arr: np.ndarray
    ) -> bool:
        """Find a newly-completed rep using find_peaks on the smoothed signal.

        A rep = trough (joint flexed) followed by peak (joint extended). We
        re-run find_peaks every frame; on a 60-sample buffer this is sub-ms.
        Global frame indices ensure we never count the same trough/peak twice
        even as the rolling buffer slides.
        """
        peaks_local, _ = find_peaks(
            smoothed,
            distance=self.min_rep_frames,
            prominence=self.prominence,
        )
        troughs_local, _ = find_peaks(
            -smoothed,
            distance=self.min_rep_frames,
            prominence=self.prominence,
        )

        # Step 1: register any new trough beyond the last consumed one.
        for t_local in troughs_local:
            t_global = self._buffer_start_idx + int(t_local)
            if t_global <= self._last_trough_idx:
                continue
            # Trough must come after the last completed peak — preserves the
            # alternating peak/trough sequence.
            if self._last_peak_idx >= 0 and t_global <= self._last_peak_idx:
                continue
            self._pending_trough_idx = t_global
            self._pending_trough_value = float(smoothed[t_local])
            lo = max(0, int(t_local) - self.smooth_window)
            hi = min(len(self._visibilities), int(t_local) + self.smooth_window + 1)
            window_vis = list(self._visibilities)[lo:hi]
            self._pending_trough_min_vis = (
                float(min(window_vis)) if window_vis else 1.0
            )
            self._last_trough_idx = t_global

        # Step 2: if a trough is pending, look for a subsequent peak.
        rep_completed = False
        if self._pending_trough_idx is not None:
            for p_local in peaks_local:
                p_global = self._buffer_start_idx + int(p_local)
                if p_global <= self._last_peak_idx:
                    continue
                if p_global <= self._pending_trough_idx:
                    continue
                peak_value = float(smoothed[p_local])
                peak_time = float(ts_arr[p_local])
                start_time = (
                    self._last_peak_time
                    if self._last_peak_time is not None
                    else float(ts_arr[0])
                )
                rep_duration = peak_time - start_time

                if self._is_valid_rep(
                    duration=rep_duration,
                    peak_value=peak_value,
                    trough_value=self._pending_trough_value,
                    min_vis=self._pending_trough_min_vis,
                ):
                    self._rep_count += 1
                    rep_completed = True
                    quality = self._calculate_rep_quality(
                        rep_duration, peak_value, self._pending_trough_value
                    )
                    self._reps.append(
                        RepData(
                            rep_number=self._rep_count,
                            quality=quality,
                            duration_seconds=rep_duration,
                            min_angle=self._pending_trough_value,
                            max_angle=peak_value,
                        )
                    )
                else:
                    self._partial_rep_count += 1

                self._last_peak_idx = p_global
                self._last_peak_value = peak_value
                self._last_peak_time = peak_time
                self._pending_trough_idx = None
                self._form_violations = []
                # One rep per update() — multiple new peaks in one frame
                # would be physically impossible at 20fps anyway.
                break

        return rep_completed

    def record_violations(self, violations: list[str]) -> None:
        self._form_violations.extend(violations)

    def _is_valid_rep(
        self,
        *,
        duration: float,
        peak_value: float,
        trough_value: float,
        min_vis: float,
    ) -> bool:
        if duration < self.min_rep_duration or duration > self.max_rep_duration:
            return False
        if self.require_full_extension:
            if peak_value < self.upper_threshold:
                return False
            if trough_value > self.lower_threshold:
                return False
        if min_vis < _VISIBILITY_REP_MIN:
            return False
        return True

    def _calculate_rep_quality(
        self, duration: float, peak_value: float, trough_value: float
    ) -> RepQuality:
        form_score = max(0.0, 1.0 - len(self._form_violations) * 0.15)
        expected_range = self.upper_threshold - self.lower_threshold
        actual_range = peak_value - trough_value
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
            form_score=form_score,
            depth_score=depth_score,
            tempo_score=tempo_score,
            symmetry_score=symmetry_score,
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

    @property
    def smoothed_angle(self) -> float:
        return self._smoothed_angle

    @property
    def velocity_dps(self) -> float:
        return self._velocity_dps

    def reset(self) -> None:
        self._phase = RepPhase.IDLE
        self._rep_count = 0
        self._partial_rep_count = 0
        self._angles.clear()
        self._timestamps.clear()
        self._visibilities.clear()
        self._frame_idx = 0
        self._buffer_start_idx = 0
        self._last_trough_idx = -1
        self._last_peak_idx = -1
        self._pending_trough_idx = None
        self._pending_trough_value = 180.0
        self._pending_trough_min_vis = 1.0
        self._last_peak_value = None
        self._last_peak_time = None
        self._threshold_in_rep = False
        self._threshold_start_time = None
        self._threshold_min_angle = 180.0
        self._threshold_min_vis = 1.0
        self._reps.clear()
        self._form_violations.clear()
        self._low_visibility_streak = 0
        self._smoothed_angle = 0.0
        self._velocity_dps = 0.0


def _params_for(exercise_key: str) -> dict[str, Any]:
    """Look up REP_DETECTION overrides for an exercise key."""
    try:
        from config.settings import settings
    except Exception:
        return {}
    cfg = getattr(settings, "REP_DETECTION", {}) or {}
    return dict(cfg.get(exercise_key, {}))


class ExerciseRepCounter:
    """Optional convenience factories — modules use exercises.base directly."""

    @staticmethod
    def create_bicep_curl_counter() -> HysteresisRepCounter:
        return HysteresisRepCounter(
            upper_threshold=150,
            lower_threshold=50,
            min_rep_duration=0.8,
            max_rep_duration=8.0,
            **_params_for("bicep_curl"),
        )

    @staticmethod
    def create_squat_counter() -> HysteresisRepCounter:
        return HysteresisRepCounter(
            upper_threshold=160,
            lower_threshold=90,
            min_rep_duration=1.0,
            max_rep_duration=10.0,
            **_params_for("squat"),
        )

    @staticmethod
    def create_pushup_counter() -> HysteresisRepCounter:
        return HysteresisRepCounter(
            upper_threshold=155,
            lower_threshold=90,
            min_rep_duration=0.8,
            max_rep_duration=8.0,
            **_params_for("pushup"),
        )
