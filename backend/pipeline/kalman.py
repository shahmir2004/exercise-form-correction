"""Per-landmark constant-velocity Kalman tracker for 33 MediaPipe landmarks."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class KalmanConfig:
    process_noise_pos: float = 1e-3
    process_noise_vel: float = 1e-2
    base_measurement_noise: float = 1e-2
    min_visibility: float = 0.05


class KalmanPoseTracker:
    """
    33 independent 6-state (x,y,z,vx,vy,vz) constant-velocity Kalman filters.
    Measurement noise is scaled by 1/visibility so low-confidence observations
    contribute less to the update.
    """

    def __init__(self, config: Optional[KalmanConfig] = None):
        self.config = config or KalmanConfig()
        self._initialized = False
        # State: (33, 6) — [x, y, z, vx, vy, vz] per landmark
        self._x = np.zeros((33, 6), dtype=np.float64)
        # Covariance: (33, 6, 6)
        self._P = np.stack([np.eye(6) for _ in range(33)])
        self._build_matrices()

    def _build_matrices(self):
        cfg = self.config
        dt = 1.0  # one frame step
        # Transition matrix F (6x6)
        self._F = np.eye(6)
        self._F[0, 3] = dt
        self._F[1, 4] = dt
        self._F[2, 5] = dt

        # Measurement matrix H (3x6) — observe [x,y,z]
        self._H = np.zeros((3, 6))
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0

        # Process noise Q (6x6)
        qp = cfg.process_noise_pos
        qv = cfg.process_noise_vel
        self._Q = np.diag([qp, qp, qp, qv, qv, qv])

        # Base measurement noise R (3x3) — scaled per-frame by visibility
        self._R_base = np.eye(3) * cfg.base_measurement_noise

        # Identity for covariance update
        self._I = np.eye(6)

    def update(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Update all 33 filters with new landmark observations.

        Args:
            landmarks: shape (33, 4) — columns [x, y, z, visibility]

        Returns:
            smoothed_xyz: shape (33, 3) — filtered [x, y, z]
            uncertainty: shape (33,) — trace of position covariance per landmark
        """
        if not self._initialized:
            self._x[:, :3] = landmarks[:, :3]
            self._initialized = True

        F = self._F
        H = self._H
        Q = self._Q
        I = self._I

        for i in range(33):
            vis = float(landmarks[i, 3])
            vis_clamped = max(vis, self.config.min_visibility)

            # Predict
            x_pred = F @ self._x[i]
            P_pred = F @ self._P[i] @ F.T + Q

            # Measurement noise scaled by 1/visibility
            R = self._R_base / vis_clamped

            # Update
            z = landmarks[i, :3].astype(np.float64)
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            innovation = z - H @ x_pred
            self._x[i] = x_pred + K @ innovation
            self._P[i] = (I - K @ H) @ P_pred

        smoothed_xyz = self._x[:, :3].copy().astype(np.float32)
        # Uncertainty = trace of position block (top-left 3x3 of covariance)
        uncertainty = np.array([
            np.trace(self._P[i, :3, :3]) for i in range(33)
        ], dtype=np.float32)

        return smoothed_xyz, uncertainty

    def reset(self):
        self._initialized = False
        self._x = np.zeros((33, 6), dtype=np.float64)
        self._P = np.stack([np.eye(6) for _ in range(33)])
