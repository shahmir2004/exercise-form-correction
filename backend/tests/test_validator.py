import pytest
import numpy as np
from pipeline.validator import InputValidator, ValidationError, ValidatedFrame


def _make_lm(x=0.5, y=0.5, z=0.0, vis=0.9):
    return {"x": x, "y": y, "z": z, "visibility": vis}


def _make_33(vis=0.9):
    return [_make_lm(vis=vis) for _ in range(33)]


def test_valid_frame():
    v = InputValidator()
    frame = v.validate({"landmarks": _make_33(), "timestamp": 100.0})
    assert isinstance(frame, ValidatedFrame)
    assert frame.landmarks.shape == (33, 4)
    assert frame.timestamp == 100.0


def test_missing_landmarks_key():
    v = InputValidator()
    with pytest.raises(ValidationError):
        v.validate({"timestamp": 1.0})


def test_wrong_length():
    v = InputValidator()
    with pytest.raises(ValidationError):
        v.validate({"landmarks": [_make_lm()] * 10, "timestamp": 1.0})


def test_missing_x():
    v = InputValidator()
    lms = _make_33()
    del lms[5]["x"]
    with pytest.raises(ValidationError):
        v.validate({"landmarks": lms, "timestamp": 1.0})


def test_nan_coordinate():
    v = InputValidator()
    lms = _make_33()
    lms[0]["x"] = float("nan")
    with pytest.raises(ValidationError):
        v.validate({"landmarks": lms, "timestamp": 1.0})


def test_inf_coordinate():
    v = InputValidator()
    lms = _make_33()
    lms[0]["y"] = float("inf")
    with pytest.raises(ValidationError):
        v.validate({"landmarks": lms, "timestamp": 1.0})


def test_partial_body_flag():
    v = InputValidator()
    lms = _make_33()
    # Make shoulders (11,12) low visibility
    lms[11]["visibility"] = 0.1
    lms[12]["visibility"] = 0.1
    frame = v.validate({"landmarks": lms, "timestamp": 1.0})
    assert frame.quality_flags.partial_body is True


def test_full_body_no_partial_flag():
    v = InputValidator()
    lms = _make_33(vis=0.95)
    frame = v.validate({"landmarks": lms, "timestamp": 1.0})
    assert frame.quality_flags.partial_body is False


def test_off_screen_count():
    v = InputValidator()
    lms = _make_33()
    for i in range(5):
        lms[i]["visibility"] = 0.1
    frame = v.validate({"landmarks": lms, "timestamp": 1.0})
    assert frame.quality_flags.off_screen_count == 5


def test_non_monotonic_timestamp_does_not_raise():
    v = InputValidator()
    v.validate({"landmarks": _make_33(), "timestamp": 100.0})
    # Should not raise even if timestamp goes backward
    v.validate({"landmarks": _make_33(), "timestamp": 99.0})


def test_non_dict_payload():
    v = InputValidator()
    with pytest.raises(ValidationError):
        v.validate("not a dict")


def test_non_list_landmarks():
    v = InputValidator()
    with pytest.raises(ValidationError):
        v.validate({"landmarks": "bad", "timestamp": 1.0})
