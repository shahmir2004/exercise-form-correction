import numpy as np
import pytest
from pipeline.kalman import KalmanPoseTracker, KalmanConfig


def _make_landmarks(x=0.5, y=0.5, z=0.0, vis=0.9):
    """Make 33 identical landmarks."""
    lms = np.zeros((33, 4), dtype=np.float32)
    lms[:, 0] = x
    lms[:, 1] = y
    lms[:, 2] = z
    lms[:, 3] = vis
    return lms


def test_output_shape():
    tracker = KalmanPoseTracker()
    lms = _make_landmarks()
    xyz, unc = tracker.update(lms)
    assert xyz.shape == (33, 3)
    assert unc.shape == (33,)


def test_clean_signal_stays_close():
    """With clean input, output should be very close to input."""
    tracker = KalmanPoseTracker()
    lms = _make_landmarks(x=0.3, y=0.7, z=0.1, vis=0.95)
    for _ in range(20):
        xyz, _ = tracker.update(lms)
    # After convergence, output should be close to input
    assert abs(float(xyz[0, 0]) - 0.3) < 0.05
    assert abs(float(xyz[0, 1]) - 0.7) < 0.05


def test_spike_rejection():
    """A low-visibility spike should be attenuated more than a high-vis signal."""
    tracker = KalmanPoseTracker()
    lms = _make_landmarks(x=0.5, y=0.5, vis=0.9)
    # Warm up
    for _ in range(15):
        tracker.update(lms)
    # Inject spike with LOW visibility — Kalman should trust it less
    spike = _make_landmarks(x=0.9, y=0.9, vis=0.1)
    xyz_spike, _ = tracker.update(spike)
    # With low-vis, measurement noise R is 10x higher → output stays near 0.5
    assert abs(float(xyz_spike[0, 0]) - 0.5) < 0.2, \
        f"Low-vis spike not attenuated: x={float(xyz_spike[0, 0]):.3f}"


def test_low_visibility_reduces_update():
    """Low visibility should result in prediction dominating over measurement."""
    cfg = KalmanConfig(base_measurement_noise=1e-2)
    t_high = KalmanPoseTracker(config=cfg)
    t_low = KalmanPoseTracker(config=cfg)

    # Warm up both at x=0.5
    lms_base = _make_landmarks(x=0.5, vis=0.9)
    for _ in range(20):
        t_high.update(lms_base)
        t_low.update(lms_base)

    # Now feed new measurement at x=0.8 — one high-vis, one low-vis
    lms_high = _make_landmarks(x=0.8, vis=0.9)
    lms_low = _make_landmarks(x=0.8, vis=0.1)
    xyz_high, _ = t_high.update(lms_high)
    xyz_low, _ = t_low.update(lms_low)

    # High-vis tracker should move further toward 0.8
    assert float(xyz_high[0, 0]) > float(xyz_low[0, 0])


def test_zero_visibility_prediction_continues():
    """With visibility=0 frames, prediction should continue without NaN/Inf."""
    tracker = KalmanPoseTracker()
    lms = _make_landmarks(x=0.5, y=0.5, vis=0.9)
    for _ in range(10):
        tracker.update(lms)
    lms_zero = _make_landmarks(x=0.5, y=0.5, vis=0.0)
    for _ in range(5):
        xyz, unc = tracker.update(lms_zero)
    assert np.all(np.isfinite(xyz))
    assert np.all(np.isfinite(unc))


def test_uncertainty_increases_low_visibility():
    """Uncertainty (trace of covariance) should be higher for low-vis landmarks."""
    tracker = KalmanPoseTracker()
    lms_high = _make_landmarks(vis=0.95)
    lms_low = lms_high.copy()
    lms_low[0, 3] = 0.1  # landmark 0 has low visibility

    for _ in range(10):
        tracker.update(lms_high)

    _, unc_high = tracker.update(lms_high)
    tracker.reset()
    for _ in range(10):
        tracker.update(lms_low)
    _, unc_low = tracker.update(lms_low)

    # Landmark 0 uncertainty should be higher in low-vis case
    assert unc_low[0] > unc_high[0]


def test_reset_clears_state():
    tracker = KalmanPoseTracker()
    lms = _make_landmarks(x=0.3)
    for _ in range(10):
        tracker.update(lms)
    tracker.reset()
    lms2 = _make_landmarks(x=0.8)
    xyz, _ = tracker.update(lms2)
    assert abs(float(xyz[0, 0]) - 0.8) < 0.1
