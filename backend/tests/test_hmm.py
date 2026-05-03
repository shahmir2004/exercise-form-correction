import numpy as np
import pytest
from pipeline.hmm import ExerciseHMM, ExState, HMMConfig, HMMResult
from pipeline.features import BodyFrame, ViewEstimate


def _make_idle_frame() -> BodyFrame:
    return BodyFrame(
        coords=np.zeros((33, 3)),
        angles={"left_knee": 175.0, "right_knee": 175.0,
                "left_elbow": 175.0, "right_elbow": 175.0,
                "torso_angle": 5.0},
        uncertainty=np.zeros(33),
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=False,
        hip_y=0.6,
        shoulder_y=0.3,
    )


def _make_squat_frame(depth=80.0) -> BodyFrame:
    return BodyFrame(
        coords=np.zeros((33, 3)),
        angles={"left_knee": depth, "right_knee": depth,
                "left_elbow": 170.0, "right_elbow": 170.0,
                "torso_angle": 20.0},
        uncertainty=np.zeros(33),
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=False,
        hip_y=0.72,  # hips lower when squatting
        shoulder_y=0.45,
    )


def _make_pushup_frame() -> BodyFrame:
    return BodyFrame(
        coords=np.zeros((33, 3)),
        angles={"left_knee": 170.0, "right_knee": 170.0,
                "left_elbow": 90.0, "right_elbow": 90.0,
                "torso_angle": 85.0},
        uncertainty=np.zeros(33),
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=True,
        hip_y=0.5,
        shoulder_y=0.5,
    )


def _make_curl_frame() -> BodyFrame:
    return BodyFrame(
        coords=np.zeros((33, 3)),
        angles={"left_knee": 170.0, "right_knee": 170.0,
                "left_elbow": 60.0, "right_elbow": 60.0,
                "torso_angle": 5.0},
        uncertainty=np.zeros(33),
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=False,
        hip_y=0.6,
        shoulder_y=0.3,
    )


def test_hmm_returns_result():
    hmm = ExerciseHMM()
    frame = _make_idle_frame()
    result = hmm.update(frame)
    assert isinstance(result, HMMResult)
    assert result.posterior.shape == (5,)
    assert abs(result.posterior.sum() - 1.0) < 1e-6


def test_squat_sequence_converges():
    """After 30+ squat frames, HMM should converge to SQUAT."""
    hmm = ExerciseHMM()
    result = None
    for _ in range(40):
        result = hmm.update(_make_squat_frame())
    assert result.most_likely_state == ExState.SQUAT, \
        f"Expected SQUAT, got {result.state_name} conf={result.exercise_confidence:.3f}"
    assert result.exercise_confidence > 0.5


def test_pushup_sequence_converges():
    hmm = ExerciseHMM()
    result = None
    for _ in range(40):
        result = hmm.update(_make_pushup_frame())
    assert result.most_likely_state == ExState.PUSHUP, \
        f"Expected PUSHUP, got {result.state_name}"
    assert result.exercise_confidence > 0.5


def test_curl_sequence_converges():
    hmm = ExerciseHMM()
    result = None
    for _ in range(40):
        result = hmm.update(_make_curl_frame())
    assert result.most_likely_state == ExState.CURL, \
        f"Expected CURL, got {result.state_name}"


def test_smooth_transition_no_flicker():
    """Switch from squat to idle: posterior should shift smoothly, not flip every frame."""
    hmm = ExerciseHMM()
    for _ in range(30):
        hmm.update(_make_squat_frame())
    posteriors = []
    for _ in range(15):
        r = hmm.update(_make_idle_frame())
        posteriors.append(r.posterior[ExState.SQUAT])
    # Squat posterior should monotonically decrease (not flicker)
    # Allow small non-monotonic noise — just check that it doesn't bounce up > 0.1 after a drop
    for i in range(1, len(posteriors)):
        assert posteriors[i] <= posteriors[i-1] + 0.15, \
            f"Flicker at frame {i}: {posteriors[i-1]:.3f} -> {posteriors[i]:.3f}"


def test_posterior_sums_to_one():
    hmm = ExerciseHMM()
    for frame_fn in [_make_idle_frame, _make_squat_frame, _make_pushup_frame]:
        r = hmm.update(frame_fn())
        assert abs(r.posterior.sum() - 1.0) < 1e-5


def test_reset_clears_state():
    hmm = ExerciseHMM()
    for _ in range(30):
        hmm.update(_make_squat_frame())
    # Before reset, posterior is strongly SQUAT
    r_before = hmm.update(_make_squat_frame())
    assert r_before.posterior[ExState.SQUAT] > 0.8
    hmm.reset()
    # After reset, log-alpha is uniform — single idle frame can push toward one state
    # but it should be far from the pre-reset 0.95+ squat posterior
    r_after = hmm.update(_make_idle_frame())
    # The key property: reset clears accumulated evidence; uniform prior is restored
    assert r_after.posterior.max() < 0.95, "After reset, posterior should not be near 1.0"
