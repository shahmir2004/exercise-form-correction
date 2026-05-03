"""Hip-relative, scale-normalized pose features for exercise classification and form checking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# MediaPipe landmark indices
IDX = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
}


class ViewEstimate(str, Enum):
    FRONTAL = "frontal"
    PROFILE_LEFT = "profile_left"
    PROFILE_RIGHT = "profile_right"
    THREE_QUARTER = "three_quarter"
    UNKNOWN = "unknown"


@dataclass
class BodyFrame:
    """Hip-relative, scale-normalized pose features for one frame."""
    # Hip-relative 3D coords, shape (33, 3), torso-length normalized
    coords: np.ndarray
    # All joint angles (degrees); keys are e.g. "left_knee", "right_elbow"
    angles: dict = field(default_factory=dict)
    # Per-landmark uncertainty from Kalman (trace of position covariance)
    uncertainty: np.ndarray = field(default_factory=lambda: np.zeros(33))
    # View classification
    view_estimate: ViewEstimate = ViewEstimate.UNKNOWN
    # Torso length in original units (for reference)
    torso_length: float = 1.0
    # Is body approximately horizontal (push-up check)?
    is_horizontal: bool = False
    # Hip Y position (normalized)
    hip_y: float = 0.5
    # Shoulder Y position (normalized)
    shoulder_y: float = 0.3


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at p2 formed by p1-p2-p3 using 3D vectors. Returns degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def calculate_3d_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Alias for calculate_angle — explicit 3D name."""
    return calculate_angle(p1, p2, p3)


def _estimate_view(xyz: np.ndarray) -> ViewEstimate:
    """
    Estimate camera view angle from shoulder–hip triangle aspect ratio.
    When viewed from the front, both shoulders are visible at similar depth.
    From profile, one shoulder is much closer (z-axis) than the other.
    """
    ls = xyz[IDX["left_shoulder"]]
    rs = xyz[IDX["right_shoulder"]]
    lh = xyz[IDX["left_hip"]]
    rh = xyz[IDX["right_hip"]]

    # Lateral spread vs depth difference of shoulders
    shoulder_x_spread = abs(float(ls[0]) - float(rs[0]))
    shoulder_z_diff = abs(float(ls[2]) - float(rs[2]))
    hip_x_spread = abs(float(lh[0]) - float(rh[0]))

    if shoulder_x_spread < 0.05 and shoulder_z_diff > 0.08:
        # Shoulders nearly same X, different depth → profile
        if float(ls[2]) < float(rs[2]):
            return ViewEstimate.PROFILE_LEFT
        return ViewEstimate.PROFILE_RIGHT

    if shoulder_x_spread > 0.12 and hip_x_spread > 0.08:
        if shoulder_z_diff < 0.04:
            return ViewEstimate.FRONTAL
        return ViewEstimate.THREE_QUARTER

    return ViewEstimate.UNKNOWN


class FeatureExtractor:
    """Converts smoothed Kalman output into BodyFrame features."""

    def extract(self, smoothed_xyz: np.ndarray, uncertainty: np.ndarray,
                original_vis: np.ndarray) -> BodyFrame:
        """
        Args:
            smoothed_xyz: shape (33, 3)
            uncertainty: shape (33,)
            original_vis: shape (33,) — visibility from original landmarks
        Returns:
            BodyFrame
        """
        # Hip origin
        hip_origin = (smoothed_xyz[IDX["left_hip"]] + smoothed_xyz[IDX["right_hip"]]) / 2.0

        # Torso length: mean-hip to mean-shoulder
        shoulder_mid = (smoothed_xyz[IDX["left_shoulder"]] + smoothed_xyz[IDX["right_shoulder"]]) / 2.0
        torso_vec = shoulder_mid - hip_origin
        torso_length = float(np.linalg.norm(torso_vec))
        if torso_length < 1e-6:
            torso_length = 1.0

        # Hip-relative, torso-normalized coords
        coords = (smoothed_xyz - hip_origin) / torso_length

        # Compute joint angles using hip-relative 3D coords
        def pt(name: str) -> np.ndarray:
            return coords[IDX[name]]

        angles: dict[str, float] = {}

        def ang(name, a, b, c):
            angles[name] = calculate_angle(pt(a), pt(b), pt(c))

        ang("left_elbow", "left_shoulder", "left_elbow", "left_wrist")
        ang("right_elbow", "right_shoulder", "right_elbow", "right_wrist")
        ang("left_shoulder", "left_elbow", "left_shoulder", "left_hip")
        ang("right_shoulder", "right_elbow", "right_shoulder", "right_hip")
        ang("left_hip", "left_shoulder", "left_hip", "left_knee")
        ang("right_hip", "right_shoulder", "right_hip", "right_knee")
        ang("left_knee", "left_hip", "left_knee", "left_ankle")
        ang("right_knee", "right_hip", "right_knee", "right_ankle")

        # Body horizontal check (push-up)
        ls_y = float(smoothed_xyz[IDX["left_shoulder"]][1])
        rs_y = float(smoothed_xyz[IDX["right_shoulder"]][1])
        la_y = float(smoothed_xyz[IDX["left_ankle"]][1])
        ra_y = float(smoothed_xyz[IDX["right_ankle"]][1])
        mid_shoulder_y = (ls_y + rs_y) / 2.0
        mid_ankle_y = (la_y + ra_y) / 2.0
        is_horizontal = abs(mid_shoulder_y - mid_ankle_y) < 0.15

        view = _estimate_view(smoothed_xyz)

        # Torso angle from vertical (degrees)
        if torso_length > 1e-6:
            up = np.array([0.0, -1.0, 0.0])
            torso_unit = torso_vec / torso_length
            angles["torso_angle"] = float(np.degrees(np.arccos(
                np.clip(np.dot(torso_unit, up), -1.0, 1.0)
            )))

        hip_y = float(hip_origin[1])
        shoulder_y = float(shoulder_mid[1])

        return BodyFrame(
            coords=coords,
            angles=angles,
            uncertainty=uncertainty.copy(),
            view_estimate=view,
            torso_length=torso_length,
            is_horizontal=is_horizontal,
            hip_y=hip_y,
            shoulder_y=shoulder_y,
        )
