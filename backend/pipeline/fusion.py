"""Classifier fusion: HMM + k-NN with hysteresis."""

from dataclasses import dataclass
from typing import Optional

from exercises.base import ExerciseType

from .hmm import ExState, HMMResult


_EX_STATE_TO_TYPE: dict[ExState, Optional[ExerciseType]] = {
    ExState.IDLE: None,
    ExState.SQUAT: ExerciseType.SQUAT,
    ExState.PUSHUP: ExerciseType.PUSHUP,
    ExState.CURL: ExerciseType.BICEP_CURL,
    ExState.ALT_CURL: ExerciseType.ALTERNATE_BICEP_CURL,
}


@dataclass
class ClassifierFusionResult:
    exercise: Optional[ExerciseType]
    confidence: float
    source: str


class ClassifierFusion:
    """Combine HMM and k-NN with preference for HMM continuity."""

    def __init__(self, hmm_weight: float = 0.6, knn_weight: float = 0.4):
        self.hmm_weight = hmm_weight
        self.knn_weight = knn_weight
        self._prev_exercise: Optional[ExerciseType] = None

    def fuse(
        self,
        hmm_result: HMMResult,
        knn_exercise: Optional[ExerciseType],
        knn_confidence: float,
    ) -> ClassifierFusionResult:
        """
        Fuse HMM and k-NN results.

        Rules:
        1. If both agree -> use HMM (temporal smoothing)
        2. If they disagree and both have confidence > 0.7 -> prefer HMM
        3. If one is much more confident, use it
        4. Otherwise -> idle
        """
        hmm_exercise = _EX_STATE_TO_TYPE.get(hmm_result.most_likely_state)
        hmm_confidence = float(hmm_result.exercise_confidence)

        if hmm_exercise == knn_exercise:
            confidence = (
                self.hmm_weight * hmm_confidence + self.knn_weight * knn_confidence
            )
            self._prev_exercise = hmm_exercise
            return ClassifierFusionResult(
                exercise=hmm_exercise,
                confidence=confidence,
                source="fusion_agree",
            )

        if hmm_confidence > 0.7 and knn_confidence > 0.7:
            self._prev_exercise = hmm_exercise
            return ClassifierFusionResult(
                exercise=hmm_exercise,
                confidence=hmm_confidence,
                source="fusion_hmm_bias",
            )

        confidence_gap = abs(hmm_confidence - knn_confidence)
        if confidence_gap > 0.3:
            winner = hmm_exercise if hmm_confidence >= knn_confidence else knn_exercise
            confidence = max(hmm_confidence, knn_confidence)
            self._prev_exercise = winner
            return ClassifierFusionResult(
                exercise=winner,
                confidence=confidence,
                source="fusion_high_gap",
            )

        if hmm_confidence > 0.6:
            self._prev_exercise = hmm_exercise
            return ClassifierFusionResult(
                exercise=hmm_exercise,
                confidence=hmm_confidence,
                source="fusion_hmm_bias",
            )

        return ClassifierFusionResult(
            exercise=None,
            confidence=0.0,
            source="fusion_low_conf",
        )

    def reset(self) -> None:
        self._prev_exercise = None
