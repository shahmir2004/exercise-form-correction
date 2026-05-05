from .base import BaseExercise, ExerciseResult, ExerciseType, JointAngles
from .squat import SquatModule
from .pushup import PushupModule
from .bicep_curl import BicepCurlModule, AlternateBicepCurlModule

__all__ = [
    "BaseExercise",
    "ExerciseResult",
    "ExerciseType",
    "JointAngles",
    "SquatModule",
    "PushupModule",
    "BicepCurlModule",
    "AlternateBicepCurlModule",
]
