import numpy as np

from exercises.base import ExerciseType
from pipeline.features import BodyFrame, ViewEstimate
from pipeline.knn_classifier import PoseKNNClassifier, REQUIRED_EXERCISE_LABELS
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

    exercise, variant, confidence, source = manager._apply_rule_gate(
        pushup_frame,
        ExerciseType.BICEP_CURL,
        "bicep_curl",
        0.62,
        "fusion_high_gap",
    )

    assert exercise == ExerciseType.PUSHUP
    assert variant == "pushup"
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

    exercise, variant, confidence, source = manager._apply_rule_gate(
        squat_frame,
        ExerciseType.BICEP_CURL,
        "bicep_curl",
        0.55,
        "fusion_high_gap",
    )

    assert exercise == ExerciseType.SQUAT
    assert variant == "squat"
    assert confidence >= 0.72
    assert source == "rule_gate"


def test_external_stgcn_refines_curl_variant_but_not_pushup():
    manager = FormManager()
    pushup_frame = _frame(
        {"left_elbow": 90, "right_elbow": 92, "left_knee": 170, "right_knee": 170},
        is_horizontal=True,
    )

    result = manager._apply_external_variant(
        (ExerciseType.BICEP_CURL, "curl-stand", 0.95),
        ExerciseType.PUSHUP,
        "pushup",
        0.8,
        "rule_gate",
        pushup_frame,
    )

    assert result == (ExerciseType.PUSHUP, "pushup", 0.8, "rule_gate")


def test_external_stgcn_can_choose_alternate_curl_variant_when_curl_like():
    manager = FormManager()
    curl_frame = _frame(
        {
            "left_elbow": 70,
            "right_elbow": 140,
            "left_knee": 170,
            "right_knee": 170,
            "torso_angle": 10,
        },
        arm_phase_diff=-1.0,
    )

    exercise, variant, confidence, source = manager._apply_external_variant(
        (ExerciseType.ALTERNATE_BICEP_CURL, "alt-stand", 0.91),
        ExerciseType.BICEP_CURL,
        "bicep_curl",
        0.63,
        "fusion_agree",
        curl_frame,
    )

    assert exercise == ExerciseType.ALTERNATE_BICEP_CURL
    assert variant == "alt-stand"
    assert confidence == 0.91
    assert source == "external_variant"


def test_pose_library_has_all_required_classes():
    classifier = PoseKNNClassifier()
    assert classifier.missing_required_libraries(REQUIRED_EXERCISE_LABELS, min_embeddings=1) == set()
