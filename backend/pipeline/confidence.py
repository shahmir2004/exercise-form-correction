"""Confidence composition from HMM posteriors and Kalman uncertainty."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .features import BodyFrame
from .validator import QualityFlags


# Joint importance weights per exercise (higher = more critical for form assessment)
JOINT_WEIGHTS: dict[str, dict[int, float]] = {
    "squat": {25: 1.0, 26: 1.0, 23: 0.8, 24: 0.8, 27: 0.6, 28: 0.6,
              11: 0.4, 12: 0.4},  # knees > hips > ankles > shoulders
    "pushup": {11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 23: 0.8, 24: 0.8,
               15: 0.5, 16: 0.5},
    "bicep_curl": {13: 1.0, 14: 1.0, 11: 0.8, 12: 0.8, 15: 0.5, 16: 0.5},
    "alternate_bicep_curl": {13: 1.0, 14: 1.0, 11: 0.8, 12: 0.8,
                              15: 0.5, 16: 0.5},
}


@dataclass
class ConfidenceResult:
    exercise_confidence: float   # HMM posterior of dominant state
    form_confidence: float       # visibility + uncertainty weighted
    signal_quality: str          # "good" | "degraded" | "unreliable"


class ConfidenceComposer:
    """
    Composes exercise_confidence (from HMM) and form_confidence
    (from Kalman uncertainty + visibility + quality flags).
    """

    def compose(
        self,
        exercise_confidence: float,
        frame: BodyFrame,
        quality_flags: QualityFlags,
        exercise_name: Optional[str] = None,
    ) -> ConfidenceResult:
        form_conf = self._compute_form_confidence(
            frame, quality_flags, exercise_name
        )

        if form_conf >= 0.75:
            quality = "good"
        elif form_conf >= 0.45:
            quality = "degraded"
        else:
            quality = "unreliable"

        return ConfidenceResult(
            exercise_confidence=float(np.clip(exercise_confidence, 0.0, 1.0)),
            form_confidence=float(np.clip(form_conf, 0.0, 1.0)),
            signal_quality=quality,
        )

    def _compute_form_confidence(
        self,
        frame: BodyFrame,
        flags: QualityFlags,
        exercise_name: Optional[str],
    ) -> float:
        weights = JOINT_WEIGHTS.get(exercise_name or "", {})
        if not weights:
            # Fallback: uniform over all 33 joints
            weights = {i: 1.0 for i in range(33)}

        total_w = 0.0
        weighted_conf = 0.0
        for idx, w in weights.items():
            unc = float(frame.uncertainty[idx])
            # Convert Kalman trace uncertainty to a 0-1 confidence
            # uncertainty of 0 → conf 1.0; uncertainty of 0.1 → conf ~0.5
            joint_conf = float(np.exp(-unc * 20.0))
            weighted_conf += joint_conf * w
            total_w += w

        if total_w < 1e-9:
            return 0.0

        base_conf = weighted_conf / total_w

        # Penalize for partial body
        if flags.partial_body:
            base_conf *= 0.6

        # Penalize for off-screen joints
        off_screen_penalty = min(0.4, flags.off_screen_count * 0.03)
        base_conf *= (1.0 - off_screen_penalty)

        return float(np.clip(base_conf, 0.0, 1.0))
