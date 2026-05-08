"""Tests for the rule-gate safety net layered on top of HMM classification."""

import numpy as np
import pytest

from exercises.base import ExerciseType
from pipeline.features import BodyFrame, ViewEstimate
from state_machine.manager import FormManager


def _frame(angles, *, is_horizontal=False, hip_y=0.6, arm_phase_diff=0.0):
    return BodyFrame(
        coords=np.zeros((33, 3)),
        angles=angles,
        uncertainty=np.zeros(33),
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=is_horizontal,
        torso_length=1.0,
        hip_y=hip_y,
        shoulder_y=0.3,
        arm_phase_diff=arm_phase_diff,
    )


def test_pushup_rule_gate_overrides_bad_curl_candidate():
    manager = FormManager()
    pushup_frame = _frame(
        {
            "left_knee": 170,
            "right_knee": 170,
            "left_elbow": 95,
            "right_elbow": 100,
            "torso_angle": 85,
        },
        is_horizontal=True,
        hip_y=0.5,
    )

    exercise, confidence, source = manager._apply_rule_gate(
        pushup_frame,
        ExerciseType.BICEP_CURL,
        0.62,
        "hmm",
    )

    assert exercise == ExerciseType.PUSHUP
    assert confidence >= 0.78
    assert source == "rule_gate"


def test_squat_rule_gate_overrides_missing_library_curl_bias():
    manager = FormManager()
    squat_frame = _frame(
        {
            "left_knee": 88,
            "right_knee": 92,
            "left_elbow": 170,
            "right_elbow": 168,
            "torso_angle": 20,
        },
        is_horizontal=False,
        hip_y=0.72,
    )

    exercise, confidence, source = manager._apply_rule_gate(
        squat_frame,
        ExerciseType.BICEP_CURL,
        0.55,
        "hmm",
    )

    assert exercise == ExerciseType.SQUAT
    assert confidence >= 0.72
    assert source == "rule_gate"


def test_rule_gate_agrees_with_hmm_boosts_confidence():
    manager = FormManager()
    squat_frame = _frame(
        {
            "left_knee": 90,
            "right_knee": 92,
            "left_elbow": 170,
            "right_elbow": 170,
            "torso_angle": 18,
        },
        hip_y=0.7,
    )

    exercise, confidence, source = manager._apply_rule_gate(
        squat_frame,
        ExerciseType.SQUAT,
        0.5,
        "hmm",
    )

    assert exercise == ExerciseType.SQUAT
    assert confidence >= 0.72  # Boosted by rule gate's higher score
    assert source == "hmm"


def test_removed_classifiers_are_not_importable():
    """Regression: k-NN, fusion, and pose embedder must stay deleted."""
    with pytest.raises(ModuleNotFoundError):
        import pipeline.knn_classifier  # noqa: F401

    with pytest.raises(ModuleNotFoundError):
        import pipeline.fusion  # noqa: F401

    with pytest.raises(ModuleNotFoundError):
        import pipeline.pose_embedder  # noqa: F401
