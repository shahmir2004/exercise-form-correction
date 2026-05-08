"""Form Manager — thin orchestrator over HMM + rule-gated pipeline."""

import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Type

from config.settings import settings
from exercises.base import BaseExercise, ExerciseResult, ExerciseType
from exercises.squat import SquatModule
from exercises.pushup import PushupModule
from exercises.bicep_curl import BicepCurlModule, AlternateBicepCurlModule

from pipeline.validator import InputValidator, ValidationError
from pipeline.kalman import KalmanPoseTracker
from pipeline.features import FeatureExtractor
from pipeline.hmm import ExerciseHMM, ExState
from pipeline.motion_detector import MotionDetector
from pipeline.form_evaluator import FormEvaluator, Violation
from pipeline.confidence import ConfidenceComposer


class SystemState(str, Enum):
    IDLE = "idle"
    STATIONARY = "stationary"
    SCANNING = "scanning"
    ACTIVE = "active"


_EX_TYPE_TO_NAME: dict[ExerciseType, str] = {
    ExerciseType.SQUAT: "squat",
    ExerciseType.PUSHUP: "pushup",
    ExerciseType.BICEP_CURL: "bicep_curl",
    ExerciseType.ALTERNATE_BICEP_CURL: "alternate_bicep_curl",
}

_EX_STATE_TO_TYPE: dict[ExState, Optional[ExerciseType]] = {
    ExState.IDLE: None,
    ExState.SQUAT: ExerciseType.SQUAT,
    ExState.PUSHUP: ExerciseType.PUSHUP,
    ExState.CURL: ExerciseType.BICEP_CURL,
    ExState.ALT_CURL: ExerciseType.ALTERNATE_BICEP_CURL,
}

_VARIANT_DISPLAY: dict[str, str] = {
    "squat": "Squat",
    "pushup": "Push-up",
    "bicep_curl": "Bicep Curl",
    "alternate_bicep_curl": "Alternate Bicep Curl",
}

_EXERCISE_MODULES: dict[ExerciseType, Type[BaseExercise]] = {
    ExerciseType.SQUAT: SquatModule,
    ExerciseType.PUSHUP: PushupModule,
    ExerciseType.BICEP_CURL: BicepCurlModule,
    ExerciseType.ALTERNATE_BICEP_CURL: AlternateBicepCurlModule,
}


@dataclass
class FormManagerState:
    system_state: SystemState
    current_exercise: Optional[ExerciseType]
    exercise_result: Optional[ExerciseResult]
    exercise_confidence: float = 0.0
    form_confidence: float = 0.0
    signal_quality: str = "good"
    stable_violations: list = field(default_factory=list)
    exercise_variant: Optional[str] = None
    exercise_source: str = "hmm"
    is_stationary: bool = False
    time_in_state: float = 0.0
    frames_processed: int = 0


class FormManager:
    """
    Orchestrates:
      InputValidator → KalmanPoseTracker → FeatureExtractor → ExerciseHMM
      → rule-based safety gate → ExerciseModule.process_frame (rep counting)
      → FormEvaluator → ConfidenceComposer

    Single classifier (HMM) with rule-based angle-threshold gates as a
    safety net. Stationary detection runs in parallel for UX feedback.
    """

    def __init__(self):
        self._validator = InputValidator()
        self._kalman = KalmanPoseTracker()
        self._feature_extractor = FeatureExtractor()
        self._hmm = ExerciseHMM()
        self._motion_detector = MotionDetector()
        self._form_evaluator = FormEvaluator()
        self._confidence_composer = ConfidenceComposer()

        self._state = SystemState.IDLE
        self._current_exercise: Optional[ExerciseType] = None
        self._active_module: Optional[BaseExercise] = None
        self._state_start_time = time.time()
        self._frames_processed = 0
        self._last_result: Optional[ExerciseResult] = None
        self._last_rep_phase: str = "idle"
        self._current_variant: Optional[str] = None
        self._exercise_source: str = "hmm"
        self._pending_exercise: Optional[ExerciseType] = None
        self._pending_frames: int = 0
        self._pending_since: Optional[float] = None
        self._is_stationary: bool = False

    def process_frame(self, landmarks: list[dict]) -> FormManagerState:
        """Process a single frame."""
        self._frames_processed += 1
        t_start = time.perf_counter()
        now = time.time()

        # 1. Validate
        try:
            payload = {"landmarks": landmarks, "timestamp": now * 1000}
            validated = self._validator.validate(payload)
        except ValidationError:
            return self._create_state(0.0, 0.0, "unreliable", [])

        # 2. Kalman filter
        smoothed_xyz, uncertainty = self._kalman.update(validated.landmarks)

        # 3. Feature extraction
        vis = validated.landmarks[:, 3]
        frame = self._feature_extractor.extract(smoothed_xyz, uncertainty, vis)

        # 4. HMM classification (single classifier)
        hmm_result = self._hmm.update(frame)
        hmm_exercise = _EX_STATE_TO_TYPE.get(hmm_result.most_likely_state)
        hmm_conf = float(hmm_result.exercise_confidence)

        candidate_exercise = hmm_exercise
        candidate_conf = hmm_conf
        candidate_source = "hmm"

        # 4b. Rule-based safety gate.
        # Angle-threshold heuristics catch HMM uncertainty (e.g. user clearly
        # in pushup plank but HMM still ramping up). Acts as a confidence
        # booster, not a competitor — only overrides when stronger.
        candidate_exercise, candidate_conf, candidate_source = (
            self._apply_rule_gate(
                frame, candidate_exercise, candidate_conf, candidate_source
            )
        )

        # 5. Stationary detection (independent of HMM)
        self._is_stationary = self._motion_detector.update(smoothed_xyz, vis)

        # 6. Map to system state.
        idle_posterior = float(hmm_result.posterior[ExState.IDLE])
        max_non_idle = float(hmm_result.posterior[1:].max())

        if self._is_stationary and idle_posterior > 0.4:
            new_sys_state = SystemState.STATIONARY
        elif idle_posterior > 0.7 or max_non_idle < 0.3:
            new_sys_state = SystemState.IDLE
        elif max_non_idle < 0.7:
            new_sys_state = SystemState.SCANNING
        else:
            new_sys_state = SystemState.ACTIVE

        if candidate_exercise is not None:
            if candidate_conf >= 0.7 and not self._is_stationary:
                new_sys_state = SystemState.ACTIVE
            elif candidate_conf >= 0.3 and new_sys_state == SystemState.IDLE:
                new_sys_state = SystemState.SCANNING

        if new_sys_state != self._state:
            self._state = new_sys_state
            self._state_start_time = time.time()
            if new_sys_state == SystemState.IDLE:
                self._form_evaluator.reset()

        # 7. Confidence composition (before switching, for quality gating)
        ex_name = _EX_TYPE_TO_NAME.get(self._current_exercise)
        conf_result = self._confidence_composer.compose(
            candidate_conf, frame, validated.quality_flags, ex_name
        )

        # 8. Exercise switching with hysteresis + rep-phase gating
        self._maybe_switch_exercise(
            candidate_exercise,
            candidate_conf,
            candidate_source,
            new_sys_state,
            conf_result.signal_quality,
        )

        # 9. Run active module (rep counting + form check)
        if self._active_module is not None:
            self._last_result = self._active_module.process_frame(landmarks)
            if self._last_result is not None:
                self._last_rep_phase = self._last_result.rep_phase

        # 10. Form evaluation (pipeline-based, temporally stable)
        ex_name = _EX_TYPE_TO_NAME.get(self._current_exercise)
        stable_violations = self._form_evaluator.evaluate(frame, ex_name)

        if self._last_result is not None and stable_violations is not None:
            self._last_result.violations = [v.message for v in stable_violations]
            self._last_result.corrections = [v.correction for v in stable_violations if v.correction]
            for v in stable_violations:
                for joint in v.joints:
                    self._last_result.joint_colors[joint] = v.severity

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000
        if t_elapsed_ms > 25:
            import logging
            logging.getLogger(__name__).warning(
                f"Frame processing took {t_elapsed_ms:.1f}ms (>25ms budget)"
            )

        return self._create_state(
            conf_result.exercise_confidence,
            conf_result.form_confidence,
            conf_result.signal_quality,
            stable_violations,
        )

    def _activate_module(
        self,
        exercise_type: ExerciseType,
        source: str,
    ) -> None:
        if self._active_module and self._current_exercise == exercise_type:
            self._exercise_source = source
            return
        module_class = _EXERCISE_MODULES.get(exercise_type)
        if module_class:
            self._active_module = module_class()
            self._current_exercise = exercise_type
            self._current_variant = _EX_TYPE_TO_NAME.get(exercise_type)
            self._exercise_source = source

    def _reset_pending(self) -> None:
        self._pending_exercise = None
        self._pending_frames = 0
        self._pending_since = None

    def _is_safe_to_switch(self) -> bool:
        # Only swap exercise modules when the rep counter is between reps.
        # Mid-rep swapping causes phantom reps in the new module.
        return self._last_rep_phase in ("idle", "setup")

    def _apply_rule_gate(
        self,
        frame,
        candidate_exercise: Optional[ExerciseType],
        candidate_conf: float,
        candidate_source: str,
    ) -> tuple[Optional[ExerciseType], float, str]:
        rule_exercise, rule_conf = self._rule_based_exercise(frame)
        if rule_exercise is None:
            return candidate_exercise, candidate_conf, candidate_source

        strong_gate = (
            (rule_exercise == ExerciseType.PUSHUP and
             rule_conf >= settings.PUSHUP_HORIZONTAL_MIN_CONFIDENCE)
            or (rule_exercise == ExerciseType.SQUAT and
                rule_conf >= settings.SQUAT_RULE_GATE_CONFIDENCE)
            or (rule_exercise in (
                ExerciseType.BICEP_CURL,
                ExerciseType.ALTERNATE_BICEP_CURL,
            ) and rule_conf >= settings.MIN_CONFIDENCE_FOR_REPS)
        )

        if candidate_exercise is None:
            return rule_exercise, rule_conf, "rule_gate"

        if candidate_exercise == rule_exercise:
            return (
                candidate_exercise,
                max(candidate_conf, rule_conf),
                candidate_source,
            )

        # Disagreement: trust the rule gate when it's strongly confident
        # AND the HMM is below the rule confidence (or rule says pushup/squat,
        # which have unambiguous body-orientation signals).
        if strong_gate and (
            candidate_conf < rule_conf
            or rule_exercise in (ExerciseType.PUSHUP, ExerciseType.SQUAT)
        ):
            return rule_exercise, rule_conf, "rule_gate"

        return candidate_exercise, candidate_conf, candidate_source

    def _rule_based_exercise(self, frame) -> tuple[Optional[ExerciseType], float]:
        angles = frame.angles
        left_knee = float(angles.get("left_knee", 180.0))
        right_knee = float(angles.get("right_knee", 180.0))
        left_elbow = float(angles.get("left_elbow", 180.0))
        right_elbow = float(angles.get("right_elbow", 180.0))
        torso = float(angles.get("torso_angle", 0.0))

        avg_knee = (left_knee + right_knee) / 2.0
        min_elbow = min(left_elbow, right_elbow)
        elbow_asym = abs(left_elbow - right_elbow)

        if frame.is_horizontal:
            elbow_signal = 1.0 if min_elbow < 150.0 else 0.55
            return ExerciseType.PUSHUP, min(0.98, 0.78 + 0.12 * elbow_signal)

        squat_like = avg_knee < 135.0 and min_elbow > 135.0 and frame.hip_y > 0.55
        if squat_like:
            depth_score = min(1.0, max(0.0, (135.0 - avg_knee) / 55.0))
            hip_score = min(1.0, max(0.0, (frame.hip_y - 0.55) / 0.20))
            return ExerciseType.SQUAT, min(0.96, 0.70 + 0.16 * depth_score + 0.08 * hip_score)

        curl_like = min_elbow < 145.0 and torso < 55.0
        if curl_like:
            if elbow_asym > 28.0 or frame.arm_phase_diff < -0.25:
                return ExerciseType.ALTERNATE_BICEP_CURL, 0.74
            return ExerciseType.BICEP_CURL, 0.70

        return None, 0.0

    def _maybe_switch_exercise(
        self,
        candidate_exercise: Optional[ExerciseType],
        candidate_conf: float,
        candidate_source: str,
        new_sys_state: SystemState,
        signal_quality: str,
    ) -> None:
        if (
            candidate_exercise is None
            or new_sys_state in (SystemState.IDLE, SystemState.STATIONARY)
        ):
            self._reset_pending()
            return

        if settings.BLOCK_SWITCH_ON_UNRELIABLE and signal_quality == "unreliable":
            self._reset_pending()
            return

        required_conf = (
            settings.MIN_CONFIDENCE_FOR_REPS
            if self._current_exercise is None
            else settings.EXERCISE_SWITCH_CONFIDENCE
        )
        if candidate_conf < required_conf:
            self._reset_pending()
            return

        if candidate_exercise == self._current_exercise:
            self._reset_pending()
            return

        now = time.time()
        if candidate_exercise != self._pending_exercise:
            self._pending_exercise = candidate_exercise
            self._pending_frames = 1
            self._pending_since = now
            return

        self._pending_frames += 1
        # Mid-rep gate: don't swap modules while a rep is in flight.
        if not self._is_safe_to_switch():
            return

        if self._pending_since is None:
            self._pending_since = now

        if (
            self._pending_frames >= settings.EXERCISE_SWITCH_MIN_FRAMES
            and (now - self._pending_since) >= settings.EXERCISE_SWITCH_MIN_SECONDS
        ):
            self._activate_module(candidate_exercise, candidate_source)
            self._reset_pending()

    def _create_state(
        self,
        ex_conf: float,
        form_conf: float,
        signal_quality: str,
        stable_violations: list,
    ) -> FormManagerState:
        return FormManagerState(
            system_state=self._state,
            current_exercise=self._current_exercise,
            exercise_result=self._last_result,
            exercise_confidence=ex_conf,
            form_confidence=form_conf,
            signal_quality=signal_quality,
            stable_violations=stable_violations,
            exercise_variant=self._current_variant,
            exercise_source=self._exercise_source,
            is_stationary=self._is_stationary,
            time_in_state=time.time() - self._state_start_time,
            frames_processed=self._frames_processed,
        )

    def get_exercise_name(self) -> str:
        if self._current_variant:
            return _VARIANT_DISPLAY.get(
                self._current_variant,
                self._current_variant.replace("_", " ").replace("-", " ").title(),
            )
        if self._active_module:
            return self._active_module.name
        if self._current_exercise:
            return self._current_exercise.value.replace("_", " ").title()
        return "Scanning..."

    def get_state_display(self) -> str:
        if self._state == SystemState.IDLE:
            return "Waiting for person..."
        if self._state == SystemState.STATIONARY:
            return "Hold still — start your reps when ready"
        if self._state == SystemState.SCANNING:
            return "Detecting exercise..."
        return f"Activity: {self.get_exercise_name()}"

    def reset(self) -> None:
        self._state = SystemState.IDLE
        self._current_exercise = None
        self._active_module = None
        self._current_variant = None
        self._exercise_source = "hmm"
        self._pending_exercise = None
        self._pending_frames = 0
        self._pending_since = None
        self._state_start_time = time.time()
        self._frames_processed = 0
        self._last_result = None
        self._last_rep_phase = "idle"
        self._is_stationary = False
        self._validator.reset()
        self._kalman.reset()
        self._feature_extractor.reset()
        self._hmm.reset()
        self._motion_detector.reset()
        self._form_evaluator.reset()

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def current_exercise(self) -> Optional[ExerciseType]:
        return self._current_exercise

    @property
    def active_module(self) -> Optional[BaseExercise]:
        return self._active_module

    @property
    def rep_count(self) -> int:
        return self._active_module.rep_count if self._active_module else 0

    @property
    def is_stationary(self) -> bool:
        return self._is_stationary
