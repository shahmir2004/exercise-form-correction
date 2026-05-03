"""Form Manager — thin orchestrator over probabilistic pipeline."""

import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Type

from config.settings import settings
from exercises.base import BaseExercise, ExerciseResult
from exercises.classifier import ExerciseType
from exercises.squat import SquatModule
from exercises.pushup import PushupModule
from exercises.bicep_curl import BicepCurlModule, AlternateBicepCurlModule

from pipeline.validator import InputValidator, ValidationError
from pipeline.kalman import KalmanPoseTracker
from pipeline.features import FeatureExtractor
from pipeline.hmm import ExerciseHMM, ExState
from pipeline.form_evaluator import FormEvaluator, Violation
from pipeline.confidence import ConfidenceComposer


class SystemState(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    ACTIVE = "active"


_EX_STATE_TO_TYPE: dict[ExState, Optional[ExerciseType]] = {
    ExState.IDLE: None,
    ExState.SQUAT: ExerciseType.SQUAT,
    ExState.PUSHUP: ExerciseType.PUSHUP,
    ExState.CURL: ExerciseType.BICEP_CURL,
    ExState.ALT_CURL: ExerciseType.ALTERNATE_BICEP_CURL,
}

_EX_TYPE_TO_NAME: dict[ExerciseType, str] = {
    ExerciseType.SQUAT: "squat",
    ExerciseType.PUSHUP: "pushup",
    ExerciseType.BICEP_CURL: "bicep_curl",
    ExerciseType.ALTERNATE_BICEP_CURL: "alternate_bicep_curl",
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
    # New probabilistic fields
    exercise_confidence: float = 0.0
    form_confidence: float = 0.0
    signal_quality: str = "good"
    stable_violations: list = field(default_factory=list)
    # Kept for backwards compat (was MotionAnalysis)
    motion_analysis: Optional[object] = None
    time_in_state: float = 0.0
    frames_processed: int = 0


class FormManager:
    """
    Orchestrates:
      InputValidator → KalmanPoseTracker → FeatureExtractor → ExerciseHMM
      → ExerciseModule.process_frame (rep counting) → FormEvaluator → ConfidenceComposer

    IDLE/SCANNING/ACTIVE states are derived from HMM posterior.
    """

    def __init__(self):
        # Pipeline components
        self._validator = InputValidator()
        self._kalman = KalmanPoseTracker()
        self._feature_extractor = FeatureExtractor()
        self._hmm = ExerciseHMM()
        self._form_evaluator = FormEvaluator()
        self._confidence_composer = ConfidenceComposer()

        # State
        self._state = SystemState.IDLE
        self._current_exercise: Optional[ExerciseType] = None
        self._active_module: Optional[BaseExercise] = None
        self._state_start_time = time.time()
        self._frames_processed = 0
        self._last_result: Optional[ExerciseResult] = None
        self._last_frame_time: Optional[float] = None

        # Per-client rate limiting
        self._max_fps: int = 60
        self._frame_interval: float = 1.0 / self._max_fps

    def process_frame(self, landmarks: list[dict]) -> FormManagerState:
        """
        Process a single frame.

        Args:
            landmarks: Raw landmark list from MediaPipe (33 landmarks)
        Returns:
            FormManagerState
        """
        self._frames_processed += 1
        t_start = time.perf_counter()

        # Rate limiting
        now = time.time()
        if self._last_frame_time is not None:
            elapsed = now - self._last_frame_time
            if elapsed < self._frame_interval:
                # Drop frame silently
                return self._create_state(0.0, 0.0, "good", [])
        self._last_frame_time = now

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

        # 4. HMM classification
        hmm_result = self._hmm.update(frame)
        ex_conf = hmm_result.exercise_confidence

        # 5. Map HMM state to system state
        idle_posterior = float(hmm_result.posterior[ExState.IDLE])
        max_non_idle = float(hmm_result.posterior[1:].max())

        if idle_posterior > 0.7 or max_non_idle < 0.3:
            new_sys_state = SystemState.IDLE
        elif max_non_idle < 0.7:
            new_sys_state = SystemState.SCANNING
        else:
            new_sys_state = SystemState.ACTIVE

        if new_sys_state != self._state:
            self._state = new_sys_state
            self._state_start_time = time.time()
            if new_sys_state == SystemState.IDLE:
                self._current_exercise = None
                self._active_module = None
                self._last_result = None
                self._form_evaluator.reset()

        # 6. Map HMM to exercise type + activate module
        detected_ex = _EX_STATE_TO_TYPE.get(hmm_result.most_likely_state)
        if new_sys_state == SystemState.ACTIVE and detected_ex is not None:
            if detected_ex != self._current_exercise:
                self._activate_module(detected_ex)
            # Run exercise module for rep counting
            if self._active_module:
                self._last_result = self._active_module.process_frame(landmarks)

        # 7. Form evaluation (pipeline-based, temporally stable)
        ex_name = _EX_TYPE_TO_NAME.get(self._current_exercise)
        stable_violations = self._form_evaluator.evaluate(frame, ex_name)

        # Override violations in exercise result with stable ones
        if self._last_result is not None and stable_violations is not None:
            self._last_result.violations = [v.message for v in stable_violations]
            self._last_result.corrections = [v.correction for v in stable_violations if v.correction]
            # Update joint colors from violation severity
            for v in stable_violations:
                for joint in v.joints:
                    self._last_result.joint_colors[joint] = v.severity

        # 8. Confidence composition
        conf_result = self._confidence_composer.compose(
            ex_conf, frame, validated.quality_flags, ex_name
        )

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

    def _activate_module(self, exercise_type: ExerciseType) -> None:
        module_class = _EXERCISE_MODULES.get(exercise_type)
        if module_class:
            self._active_module = module_class()
            self._current_exercise = exercise_type

    def _create_state(
        self,
        ex_conf: float,
        form_conf: float,
        signal_quality: str,
        stable_violations: list,
    ) -> FormManagerState:
        # Build a minimal MotionAnalysis-like object for backwards compat
        # (routes.py reads motion_analysis.confidence)
        class _FakeAnalysis:
            def __init__(self, c):
                self.confidence = c
        return FormManagerState(
            system_state=self._state,
            current_exercise=self._current_exercise,
            exercise_result=self._last_result,
            exercise_confidence=ex_conf,
            form_confidence=form_conf,
            signal_quality=signal_quality,
            stable_violations=stable_violations,
            motion_analysis=_FakeAnalysis(form_conf),  # backwards compat: confidence = form_confidence
            time_in_state=time.time() - self._state_start_time,
            frames_processed=self._frames_processed,
        )

    def get_exercise_name(self) -> str:
        if self._active_module:
            return self._active_module.name
        if self._current_exercise:
            return self._current_exercise.value.replace("_", " ").title()
        return "Scanning..."

    def get_state_display(self) -> str:
        if self._state == SystemState.IDLE:
            return "Waiting for person..."
        elif self._state == SystemState.SCANNING:
            return "Detecting exercise..."
        return f"Activity: {self.get_exercise_name()}"

    def reset(self) -> None:
        self._state = SystemState.IDLE
        self._current_exercise = None
        self._active_module = None
        self._state_start_time = time.time()
        self._frames_processed = 0
        self._last_result = None
        self._last_frame_time = None
        self._validator.reset()
        self._kalman.reset()
        self._hmm.reset()
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
