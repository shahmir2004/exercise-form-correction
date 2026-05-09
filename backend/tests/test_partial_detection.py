import numpy as np

from exercises.base import JointName, LANDMARK_INDICES
from exercises.bicep_curl import BicepCurlModule
from pipeline.confidence import ConfidenceComposer
from pipeline.features import BodyFrame, ViewEstimate
from pipeline.validator import QualityFlags


def _landmarks_with_upper_body_only():
    landmarks = [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.0}
        for _ in range(33)
    ]
    points = {
        JointName.LEFT_SHOULDER: (0.40, 0.30),
        JointName.RIGHT_SHOULDER: (0.60, 0.30),
        JointName.LEFT_ELBOW: (0.36, 0.46),
        JointName.RIGHT_ELBOW: (0.64, 0.46),
        JointName.LEFT_WRIST: (0.34, 0.62),
        JointName.RIGHT_WRIST: (0.66, 0.62),
    }
    for joint, (x, y) in points.items():
        idx = LANDMARK_INDICES[joint]
        landmarks[idx] = {"x": x, "y": y, "z": 0.0, "visibility": 0.92}
    return landmarks


def test_bicep_curl_accepts_half_body_upper_frame():
    module = BicepCurlModule()
    result = module.process_frame(_landmarks_with_upper_body_only())

    assert result.rep_phase in {"setup", "idle", "hold", "eccentric", "concentric"}
    assert "left_hip not clearly visible" not in result.violations
    assert "right_hip not clearly visible" not in result.violations


def test_curl_confidence_uses_upper_body_not_missing_legs():
    visibility = np.zeros(33)
    for idx in (11, 12, 13, 14, 15, 16):
        visibility[idx] = 0.9

    frame = BodyFrame(
        coords=np.zeros((33, 3)),
        angles={"left_elbow": 90, "right_elbow": 95, "torso_angle": 0},
        uncertainty=np.zeros(33),
        visibility=visibility,
        view_estimate=ViewEstimate.UNKNOWN,
    )
    result = ConfidenceComposer().compose(
        0.74,
        frame,
        QualityFlags(off_screen_count=27, low_visibility_count=27, partial_body=True),
        "bicep_curl",
    )

    assert result.form_confidence >= 0.75
    assert result.signal_quality == "good"
