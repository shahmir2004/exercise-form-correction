import numpy as np
import pytest
from pipeline.form_evaluator import ViolationAggregator, FormEvaluator, Violation
from pipeline.features import BodyFrame, ViewEstimate


def _make_frame(angles=None, is_horizontal=False) -> BodyFrame:
    coords = np.zeros((33, 3))
    if angles is None:
        angles = {}
    return BodyFrame(
        coords=coords,
        angles=angles,
        uncertainty=np.zeros(33),
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=is_horizontal,
        torso_length=1.0,
        hip_y=0.6,
        shoulder_y=0.3,
    )


def test_aggregator_no_emit_alternating():
    agg = ViolationAggregator(m=4, n=6, cooldown=10)
    v = Violation(code="test_v", severity="red", message="test")
    result = []
    for i in range(12):
        result = agg.update([v] if i % 2 == 0 else [])
    assert len(result) == 0


def test_aggregator_emits_after_consecutive():
    agg = ViolationAggregator(m=4, n=6, cooldown=0)
    v = Violation(code="knee_valgus", severity="red")
    emitted_codes = []
    for _ in range(6):
        result = agg.update([v])
        emitted_codes.extend(vv.code for vv in result)
    assert "knee_valgus" in emitted_codes


def test_aggregator_cooldown_suppresses_repeat():
    agg = ViolationAggregator(m=4, n=6, cooldown=20)
    v = Violation(code="hip_sag", severity="red")
    for _ in range(6):
        agg.update([v])
    for _ in range(10):
        result = agg.update([v])
    assert len(result) == 0


def test_aggregator_reset_clears_state():
    agg = ViolationAggregator(m=4, n=6, cooldown=10)
    v = Violation(code="test", severity="yellow")
    for _ in range(6):
        agg.update([v])
    agg.reset()
    result = agg.update([v])
    assert len(result) == 0


def test_form_evaluator_no_exercise():
    fe = FormEvaluator()
    frame = _make_frame()
    result = fe.evaluate(frame, None)
    assert result == []


def test_squat_knee_valgus_detected():
    fe = FormEvaluator(ViolationAggregator(m=1, n=1, cooldown=0))
    coords = np.zeros((33, 3))
    coords[25] = [0.10, 0.3, 0.0]
    coords[27] = [-0.08, 0.5, 0.0]
    frame = BodyFrame(
        coords=coords,
        angles={"left_knee": 90.0, "right_knee": 90.0, "torso_angle": 20.0},
        uncertainty=np.zeros(33),
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=False,
        torso_length=1.0,
        hip_y=0.6,
        shoulder_y=0.3,
    )
    result = fe.evaluate(frame, "squat")
    codes = [v.code for v in result]
    assert "left_knee_valgus" in codes


def test_pushup_hip_sag_detected():
    fe = FormEvaluator(ViolationAggregator(m=1, n=1, cooldown=0))
    coords = np.zeros((33, 3))
    coords[11] = [-0.4, -0.3, 0.0]
    coords[12] = [-0.2, -0.3, 0.0]
    coords[27] = [0.2, -0.3, 0.0]
    coords[28] = [0.4, -0.3, 0.0]
    frame = BodyFrame(
        coords=coords,
        angles={"left_elbow": 90.0, "right_elbow": 90.0,
                "left_shoulder": 45.0, "right_shoulder": 45.0},
        uncertainty=np.zeros(33),
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=True,
        torso_length=1.0,
        hip_y=0.5,
        shoulder_y=0.5,
    )
    result = fe.evaluate(frame, "pushup")
    codes = [v.code for v in result]
    assert "hip_sag" in codes


def test_curl_elbow_drift_detected():
    fe = FormEvaluator(ViolationAggregator(m=1, n=1, cooldown=0))
    coords = np.zeros((33, 3))
    coords[11] = [0.0, -0.5, 0.0]
    coords[13] = [0.3, 0.0, 0.0]
    frame = BodyFrame(
        coords=coords,
        angles={"left_elbow": 90.0, "right_elbow": 170.0},
        uncertainty=np.zeros(33),
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=False,
        torso_length=1.0,
        hip_y=0.6,
        shoulder_y=0.3,
    )
    result = fe.evaluate(frame, "bicep_curl")
    codes = [v.code for v in result]
    assert "left_elbow_drift" in codes
