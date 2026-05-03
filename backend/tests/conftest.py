import json, os, pytest
import numpy as np

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def _make_landmark(x, y, z, vis):
    return {"x": float(x), "y": float(y), "z": float(z), "visibility": float(vis)}

def _make_pose_33(pattern="standing"):
    """Generate a synthetic 33-landmark pose."""
    lms = []
    for i in range(33):
        lms.append(_make_landmark(0.5, float(i)/33.0, 0.0, 0.9))
    if pattern == "standing":
        # shoulders at idx 11,12 visible; hips at 23,24 visible
        for idx in [11,12,23,24]:
            lms[idx]["visibility"] = 0.95
    elif pattern == "partial":
        # left arm joints 13,15 low visibility
        for idx in [13,15]:
            lms[idx]["visibility"] = 0.2
    return lms

@pytest.fixture
def good_landmarks():
    return _make_pose_33("standing")

@pytest.fixture
def partial_landmarks():
    return _make_pose_33("partial")

@pytest.fixture
def malformed_landmarks():
    return [{"x": 0.5, "y": 0.5} for _ in range(10)]  # only 10, missing z/visibility

@pytest.fixture
def good_frame():
    lms = _make_pose_33("standing")
    return {"landmarks": lms, "timestamp": 1000.0}

@pytest.fixture
def fixture_sequences():
    """Load JSON fixture sequences if they exist."""
    sequences = {}
    if os.path.exists(FIXTURES_DIR):
        for fname in os.listdir(FIXTURES_DIR):
            if fname.endswith(".json"):
                with open(os.path.join(FIXTURES_DIR, fname)) as f:
                    sequences[fname[:-5]] = json.load(f)
    return sequences
