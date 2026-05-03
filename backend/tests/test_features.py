import numpy as np
import pytest
from pipeline.features import FeatureExtractor, BodyFrame, ViewEstimate, calculate_angle, calculate_3d_angle


def _make_xyz(pattern="standing"):
    """Create a synthetic 33-landmark xyz array."""
    xyz = np.zeros((33, 4), dtype=np.float32)
    # Default: upright standing, rough proportions
    # Indices: left_shoulder=11, right_shoulder=12, left_hip=23, right_hip=24, etc.
    xyz[:, 3] = 0.9  # visibility
    # Standing: hips at y=0.6, shoulders at y=0.3
    xyz[11] = [0.4, 0.3, 0.0, 0.9]  # left shoulder
    xyz[12] = [0.6, 0.3, 0.0, 0.9]  # right shoulder
    xyz[13] = [0.35, 0.5, 0.0, 0.9]  # left elbow
    xyz[14] = [0.65, 0.5, 0.0, 0.9]  # right elbow
    xyz[15] = [0.35, 0.7, 0.0, 0.9]  # left wrist
    xyz[16] = [0.65, 0.7, 0.0, 0.9]  # right wrist
    xyz[23] = [0.45, 0.6, 0.0, 0.9]  # left hip
    xyz[24] = [0.55, 0.6, 0.0, 0.9]  # right hip
    xyz[25] = [0.45, 0.8, 0.0, 0.9]  # left knee
    xyz[26] = [0.55, 0.8, 0.0, 0.9]  # right knee
    xyz[27] = [0.45, 1.0, 0.0, 0.9]  # left ankle
    xyz[28] = [0.55, 1.0, 0.0, 0.9]  # right ankle
    xyz[29] = [0.44, 1.02, 0.0, 0.8]  # left heel
    xyz[30] = [0.54, 1.02, 0.0, 0.8]  # right heel
    return xyz


def test_extract_returns_body_frame():
    xyz = _make_xyz()
    extractor = FeatureExtractor()
    frame = extractor.extract(xyz[:, :3], np.zeros(33), xyz[:, 3])
    assert isinstance(frame, BodyFrame)
    assert frame.coords.shape == (33, 3)


def test_hip_relative_origin():
    xyz = _make_xyz()
    extractor = FeatureExtractor()
    frame = extractor.extract(xyz[:, :3], np.zeros(33), xyz[:, 3])
    # Hip-relative coords: mean(left_hip, right_hip) should be near 0
    hip_mean = (frame.coords[23] + frame.coords[24]) / 2.0
    assert np.allclose(hip_mean, 0.0, atol=0.01)


def test_angle_keys_present():
    xyz = _make_xyz()
    extractor = FeatureExtractor()
    frame = extractor.extract(xyz[:, :3], np.zeros(33), xyz[:, 3])
    for key in ["left_knee", "right_knee", "left_elbow", "right_elbow"]:
        assert key in frame.angles, f"Missing angle: {key}"


def test_horizontal_detection_standing():
    xyz = _make_xyz()
    extractor = FeatureExtractor()
    frame = extractor.extract(xyz[:, :3], np.zeros(33), xyz[:, 3])
    assert frame.is_horizontal is False


def test_horizontal_detection_prone():
    xyz = _make_xyz()
    # Make body horizontal: shoulders and ankles at same Y
    xyz[11, 1] = 0.5; xyz[12, 1] = 0.5  # shoulders
    xyz[27, 1] = 0.5; xyz[28, 1] = 0.5  # ankles
    extractor = FeatureExtractor()
    frame = extractor.extract(xyz[:, :3], np.zeros(33), xyz[:, 3])
    assert frame.is_horizontal is True


def test_calculate_angle_right_angle():
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.0, 1.0, 0.0])
    angle = calculate_angle(p1, p2, p3)
    assert abs(angle - 90.0) < 0.01


def test_calculate_angle_straight():
    p1 = np.array([-1.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    angle = calculate_angle(p1, p2, p3)
    assert abs(angle - 180.0) < 0.01


def test_uncertainty_propagated():
    xyz = _make_xyz()
    unc = np.linspace(0.1, 0.5, 33)
    extractor = FeatureExtractor()
    frame = extractor.extract(xyz[:, :3], unc, xyz[:, 3])
    assert np.allclose(frame.uncertainty, unc)
