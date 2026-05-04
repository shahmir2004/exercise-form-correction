"""Pose embedder using MediaPipe pairwise-distance approach."""

import numpy as np

from .features import IDX


POSE_LANDMARKS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def embed_pose(coords: np.ndarray, torso_length: float) -> np.ndarray:
    """
    Compute pairwise-distance embedding (MediaPipe recipe).

    Args:
        coords: Hip-relative, torso-normalized shape (33, 3).
        torso_length: Original torso length (already normalized into coords).

    Returns:
        np.ndarray of shape (66,) with all pairwise distances.
    """
    landmarks_3d = np.array(
        [coords[IDX[name]] for name in POSE_LANDMARKS], dtype=np.float32
    )

    distances = []
    for i in range(len(landmarks_3d)):
        for j in range(i + 1, len(landmarks_3d)):
            distances.append(np.linalg.norm(landmarks_3d[i] - landmarks_3d[j]))

    return np.array(distances, dtype=np.float32)
