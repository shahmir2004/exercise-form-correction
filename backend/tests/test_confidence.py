import numpy as np
import pytest
from pipeline.confidence import ConfidenceComposer
from pipeline.features import BodyFrame, ViewEstimate
from pipeline.validator import QualityFlags


def _make_frame(uncertainty=None) -> BodyFrame:
    if uncertainty is None:
        uncertainty = np.zeros(33)
    return BodyFrame(
        coords=np.zeros((33, 3)),
        angles={},
        uncertainty=uncertainty,
        view_estimate=ViewEstimate.FRONTAL,
        is_horizontal=False,
        torso_length=1.0,
        hip_y=0.6,
        shoulder_y=0.3,
    )


def test_high_confidence_good_signal():
    composer = ConfidenceComposer()
    frame = _make_frame(uncertainty=np.zeros(33))
    flags = QualityFlags(off_screen_count=0, low_visibility_count=0, partial_body=False)
    result = composer.compose(0.9, frame, flags, "squat")
    assert result.form_confidence > 0.7
    assert result.signal_quality == "good"


def test_partial_body_degrades_confidence():
    composer = ConfidenceComposer()
    frame = _make_frame(uncertainty=np.zeros(33))
    flags = QualityFlags(off_screen_count=5, low_visibility_count=8, partial_body=True)
    result = composer.compose(0.85, frame, flags, "squat")
    assert result.form_confidence < 0.75


def test_high_uncertainty_degrades_confidence():
    composer = ConfidenceComposer()
    # High uncertainty on critical knee joints (25, 26)
    unc = np.zeros(33)
    unc[25] = 0.5; unc[26] = 0.5
    frame = _make_frame(uncertainty=unc)
    flags = QualityFlags()
    result = composer.compose(0.9, frame, flags, "squat")

    # Compare to zero uncertainty
    frame_clean = _make_frame(uncertainty=np.zeros(33))
    result_clean = composer.compose(0.9, frame_clean, flags, "squat")
    assert result.form_confidence < result_clean.form_confidence


def test_signal_quality_levels():
    composer = ConfidenceComposer()
    flags_good = QualityFlags()
    flags_bad = QualityFlags(off_screen_count=15, partial_body=True)
    frame = _make_frame(uncertainty=np.zeros(33))
    r_good = composer.compose(0.9, frame, flags_good, "squat")
    r_bad = composer.compose(0.9, frame, flags_bad, "squat")
    assert r_good.signal_quality in ("good", "degraded")
    assert r_bad.signal_quality in ("degraded", "unreliable")


def test_exercise_confidence_clamped():
    composer = ConfidenceComposer()
    frame = _make_frame()
    flags = QualityFlags()
    result = composer.compose(1.5, frame, flags)  # Input > 1.0
    assert result.exercise_confidence <= 1.0


def test_no_exercise_fallback():
    composer = ConfidenceComposer()
    frame = _make_frame(uncertainty=np.zeros(33))
    flags = QualityFlags()
    result = composer.compose(0.8, frame, flags, exercise_name=None)
    assert 0.0 <= result.form_confidence <= 1.0
