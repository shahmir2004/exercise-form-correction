"""Form Manager — thin orchestrator over probabilistic pipeline."""

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
from pipeline.pose_embedder import embed_pose
from pipeline.knn_classifier import PoseKNNClassifier
from pipeline.fusion import ClassifierFusion
from pipeline.form_evaluator import FormEvaluator, Violation
from pipeline.confidence import ConfidenceComposer


class SystemState(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    ACTIVE = "active"


_EX_TYPE_TO_NAME: dict[ExerciseType, str] = {
    ExerciseType.SQUAT: "squat",
    ExerciseType.PUSHUP: "pushup",
    ExerciseType.BICEP_CURL: "bicep_curl",
    ExerciseType.ALTERNATE_BICEP_CURL: "alternate_bicep_curl",
}

_VARIANT_LABEL_TO_EXERCISE: dict[str, ExerciseType] = {
    "curl-stand": ExerciseType.BICEP_CURL,
    "curl-seat": ExerciseType.BICEP_CURL,
    "alt-stand": ExerciseType.ALTERNATE_BICEP_CURL,
    "alt-seat": ExerciseType.ALTERNATE_BICEP_CURL,
    "bicep_curl": ExerciseType.BICEP_CURL,
    "alternate_bicep_curl": ExerciseType.ALTERNATE_BICEP_CURL,
    "squat": ExerciseType.SQUAT,
    "pushup": ExerciseType.PUSHUP,
}

_VARIANT_DISPLAY: dict[str, str] = {
    "curl-stand": "Bicep Curl (Standing)",
    "curl-seat": "Bicep Curl (Seated)",
    "alt-stand": "Alt Bicep Curl (Standing)",
    "alt-seat": "Alt Bicep Curl (Seated)",
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
    exercise_variant: Optional[str] = None
    exercise_source: str = "pipeline"
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
        self._embedder = embed_pose
        self._knn = PoseKNNClassifier(
            min_embeddings_per_class=settings.MIN_CLASS_LIBRARY_EMBEDDINGS
        )
        self._fusion = ClassifierFusion()
        self._form_evaluator = FormEvaluator()
        self._confidence_composer = ConfidenceComposer()

        # State
        self._state = SystemState.IDLE
        self._current_exercise: Optional[ExerciseType] = None
        self._active_module: Optional[BaseExercise] = None
        self._state_start_time = time.time()
        self._frames_processed = 0
        self._last_result: Optional[ExerciseResult] = None
        self._last_rep_phase: str = "idle"
        self._current_variant: Optional[str] = None
        self._exercise_source: str = "pipeline"
        self._pending_exercise: Optional[ExerciseType] = None
        self._pending_variant: Optional[str] = None
        self._pending_frames: int = 0
        self._pending_since: Optional[float] = None

    def process_frame(
        self,
        landmarks: list[dict],
        client_probs: Optional[dict] = None,
    ) -> FormManagerState:
        """
        Process a single frame.

        Args:
            landmarks: Raw landmark list from MediaPipe (33 landmarks)
        Returns:
            FormManagerState
        """
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

        # 4. HMM classification
        hmm_result = self._hmm.update(frame)

        # 4b. k-NN classification + fusion
        embedding = self._embedder(frame.coords, frame.torso_length)
        knn_ex, knn_conf = self._knn.classify(embedding)
        fusion_result = self._fusion.fuse(hmm_result, knn_ex, knn_conf)
        candidate_exercise = fusion_result.exercise
        candidate_conf = fusion_result.confidence
        candidate_variant = (
            _EX_TYPE_TO_NAME.get(candidate_exercise) if candidate_exercise else None
        )
        candidate_source = fusion_result.source

        # 4c. Exercise-first guard rules. These protect broad exercise
        # detection from a sparse k-NN library or a curl-only external model.
        candidate_exercise, candidate_variant, candidate_conf, candidate_source = (
            self._apply_rule_gate(
                frame,
                candidate_exercise,
                candidate_variant,
                candidate_conf,
                candidate_source,
            )
        )

        # 4d. External classifier (ST-GCN) refines curl variants only.
        ext = self._parse_external_probs(client_probs)
        candidate_exercise, candidate_variant, candidate_conf, candidate_source = (
            self._apply_external_variant(
                ext,
                candidate_exercise,
                candidate_variant,
                candidate_conf,
                candidate_source,
                frame,
            )
        )

        # 5. Map HMM state to system state
        idle_posterior = float(hmm_result.posterior[ExState.IDLE])
        max_non_idle = float(hmm_result.posterior[1:].max())

        if idle_posterior > 0.7 or max_non_idle < 0.3:
            new_sys_state = SystemState.IDLE
        elif max_non_idle < 0.7:
            new_sys_state = SystemState.SCANNING
        else:
            new_sys_state = SystemState.ACTIVE

        if candidate_exercise is not None:
            if candidate_conf >= 0.7:
                new_sys_state = SystemState.ACTIVE
            elif candidate_conf >= 0.3 and new_sys_state == SystemState.IDLE:
                new_sys_state = SystemState.SCANNING

        if new_sys_state != self._state:
            self._state = new_sys_state
            self._state_start_time = time.time()
            # NB: do NOT reset _active_module on IDLE transitions. A user
            # standing tall at the top of a squat rep momentarily looks like
            # IDLE — wiping the rep counter there resets the count every rep.
            # The module persists; only the FormEvaluator history (which is
            # exercise-specific stable violations) needs clearing on idle.
            if new_sys_state == SystemState.IDLE:
                self._form_evaluator.reset()

        # 6. Map HMM to exercise type + activate module.
        # Two cases:
        #   (a) HMM is confident enough → (re)activate the matching module.
        #   (b) HMM lost confidence (e.g. user momentarily standing tall at
        #       the top of a rep) but a module is already active → keep
        #       running it so the rep counter sees the angle returning to
        #       extension. Letting the HMM gate every frame's rep update
        #       resets reps mid-set.
        detected_ex = candidate_exercise

        # 6. Confidence composition (before switching, for quality gating)
        ex_name = _EX_TYPE_TO_NAME.get(self._current_exercise)
        conf_result = self._confidence_composer.compose(
            candidate_conf, frame, validated.quality_flags, ex_name
        )

        # 6b. Exercise switching with hysteresis + rep-phase gating
        self._maybe_switch_exercise(
            detected_ex,
            candidate_variant,
            candidate_conf,
            candidate_source,
            new_sys_state,
            conf_result.signal_quality,
        )

        if self._active_module is not None:
            self._last_result = self._active_module.process_frame(landmarks)
            if self._last_result is not None:
                self._last_rep_phase = self._last_result.rep_phase

        # 7. Form evaluation (pipeline-based, temporally stable)
        ex_name = _EX_TYPE_TO_NAME.get(self._current_exercise)
        stable_violations = self._form_evaluator.evaluate(frame, ex_name)

        # Overlay stable violations on the module's result. We only mutate
        # violation/correction/joint_color fields — rep_phase, rep_count,
        # is_valid, and angles are owned by the exercise module and must
        # not be overwritten here.
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
        variant: Optional[str],
        source: str,
    ) -> None:
        if self._active_module and self._current_exercise == exercise_type:
            # Same base exercise; update variant label without resetting reps
            self._current_variant = variant or self._current_variant
            self._exercise_source = source
            return
        module_class = _EXERCISE_MODULES.get(exercise_type)
        if module_class:
            self._active_module = module_class()
            self._current_exercise = exercise_type
            self._current_variant = variant or _EX_TYPE_TO_NAME.get(exercise_type)
            self._exercise_source = source

    def _reset_pending(self) -> None:
        self._pending_exercise = None
        self._pending_variant = None
        self._pending_frames = 0
        self._pending_since = None

    def _is_safe_to_switch(self) -> bool:
        return self._last_rep_phase in ("idle", "ready")

    def _apply_rule_gate(
        self,
        frame,
        candidate_exercise: Optional[ExerciseType],
        candidate_variant: Optional[str],
        candidate_conf: float,
        candidate_source: str,
    ) -> tuple[Optional[ExerciseType], Optional[str], float, str]:
        rule_exercise, rule_conf = self._rule_based_exercise(frame)
        if rule_exercise is None:
            return candidate_exercise, candidate_variant, candidate_conf, candidate_source

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
            return rule_exercise, _EX_TYPE_TO_NAME.get(rule_exercise), rule_conf, "rule_gate"

        if candidate_exercise == rule_exercise:
            return (
                candidate_exercise,
                candidate_variant or _EX_TYPE_TO_NAME.get(candidate_exercise),
                max(candidate_conf, rule_conf),
                candidate_source,
            )

        if strong_gate and (
            candidate_conf < rule_conf
            or candidate_source in {"fusion_low_conf", "fusion_high_gap"}
            or rule_exercise in (ExerciseType.PUSHUP, ExerciseType.SQUAT)
        ):
            return rule_exercise, _EX_TYPE_TO_NAME.get(rule_exercise), rule_conf, "rule_gate"

        return candidate_exercise, candidate_variant, candidate_conf, candidate_source

    def _rule_based_exercise(self, frame) -> tuple[Optional[ExerciseType], float]:
        angles = frame.angles
        left_knee = float(angles.get("left_knee", 180.0))
        right_knee = float(angles.get("right_knee", 180.0))
        left_elbow = float(angles.get("left_elbow", 180.0))
        right_elbow = float(angles.get("right_elbow", 180.0))
        torso = float(angles.get("torso_angle", 0.0))

        min_knee = min(left_knee, right_knee)
        avg_knee = (left_knee + right_knee) / 2.0
        min_elbow = min(left_elbow, right_elbow)
        avg_elbow = (left_elbow + right_elbow) / 2.0
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

    def _apply_external_variant(
        self,
        ext: Optional[tuple[ExerciseType, str, float]],
        candidate_exercise: Optional[ExerciseType],
        candidate_variant: Optional[str],
        candidate_conf: float,
        candidate_source: str,
        frame,
    ) -> tuple[Optional[ExerciseType], Optional[str], float, str]:
        if ext is None:
            return candidate_exercise, candidate_variant, candidate_conf, candidate_source

        ext_exercise, ext_variant, ext_conf = ext
        if ext_conf < settings.CURL_VARIANT_OVERRIDE_CONFIDENCE:
            return candidate_exercise, candidate_variant, candidate_conf, candidate_source

        if ext_exercise not in (ExerciseType.BICEP_CURL, ExerciseType.ALTERNATE_BICEP_CURL):
            return candidate_exercise, candidate_variant, candidate_conf, candidate_source

        rule_exercise, _ = self._rule_based_exercise(frame)
        candidate_is_curl = candidate_exercise in (
            ExerciseType.BICEP_CURL,
            ExerciseType.ALTERNATE_BICEP_CURL,
            None,
        )
        rule_allows_curl = rule_exercise in (
            ExerciseType.BICEP_CURL,
            ExerciseType.ALTERNATE_BICEP_CURL,
            None,
        )
        if not (candidate_is_curl and rule_allows_curl):
            return candidate_exercise, candidate_variant, candidate_conf, candidate_source

        return ext_exercise, ext_variant, max(candidate_conf, ext_conf), "external_variant"

    def _maybe_switch_exercise(
        self,
        candidate_exercise: Optional[ExerciseType],
        candidate_variant: Optional[str],
        candidate_conf: float,
        candidate_source: str,
        new_sys_state: SystemState,
        signal_quality: str,
    ) -> None:
        if candidate_exercise is None or new_sys_state == SystemState.IDLE:
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

        if (
            candidate_exercise == self._current_exercise
            and candidate_variant == self._current_variant
        ):
            self._reset_pending()
            return

        now = time.time()
        if (
            candidate_exercise != self._pending_exercise
            or candidate_variant != self._pending_variant
        ):
            self._pending_exercise = candidate_exercise
            self._pending_variant = candidate_variant
            self._pending_frames = 1
            self._pending_since = now
            return

        self._pending_frames += 1
        if not self._is_safe_to_switch():
            return

        if self._pending_since is None:
            self._pending_since = now

        if (
            self._pending_frames >= settings.EXERCISE_SWITCH_MIN_FRAMES
            and (now - self._pending_since) >= settings.EXERCISE_SWITCH_MIN_SECONDS
        ):
            self._activate_module(candidate_exercise, candidate_variant, candidate_source)
            self._reset_pending()

    def _parse_external_probs(
        self,
        client_probs: Optional[dict],
    ) -> Optional[tuple[ExerciseType, str, float]]:
        if not isinstance(client_probs, dict) or not client_probs:
            return None

        best_label = None
        best_score = -1.0
        for label, score in client_probs.items():
            if not isinstance(label, str):
                continue
            try:
                score_val = float(score)
            except (TypeError, ValueError):
                continue
            if score_val > best_score:
                best_label = label
                best_score = score_val

        if best_label is None:
            return None

        exercise = _VARIANT_LABEL_TO_EXERCISE.get(best_label)
        if exercise is None:
            return None

        return exercise, best_label, float(best_score)

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
        elif self._state == SystemState.SCANNING:
            return "Detecting exercise..."
        return f"Activity: {self.get_exercise_name()}"

    def reset(self) -> None:
        self._state = SystemState.IDLE
        self._current_exercise = None
        self._active_module = None
        self._current_variant = None
        self._exercise_source = "pipeline"
        self._pending_exercise = None
        self._pending_variant = None
        self._pending_frames = 0
        self._pending_since = None
        self._state_start_time = time.time()
        self._frames_processed = 0
        self._last_result = None
        self._last_rep_phase = "idle"
        self._validator.reset()
        self._kalman.reset()
        self._feature_extractor.reset()
        self._hmm.reset()
        self._fusion.reset()
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
