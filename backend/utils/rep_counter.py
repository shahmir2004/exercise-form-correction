"""
Advanced rep counting with hysteresis and state machine.

Note: Implementation has moved to pipeline/rep_counter.py.
This module re-exports everything from there for backwards compatibility.
"""

# Re-export all public symbols from the canonical location
from pipeline.rep_counter import (  # noqa: F401
    RepPhase,
    RepQuality,
    RepData,
    HysteresisRepCounter,
    ExerciseRepCounter,
)

__all__ = [
    "RepPhase",
    "RepQuality",
    "RepData",
    "HysteresisRepCounter",
    "ExerciseRepCounter",
]
