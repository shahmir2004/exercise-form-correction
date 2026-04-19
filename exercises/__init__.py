from .base import BaseExercise, ExerciseResult, JointAngles
from .classifier import ExerciseClassifier, ExerciseType
from .squat import SquatModule
from .pushup import PushupModule
from .bicep_curl import BicepCurlModule, AlternateBicepCurlModule

__all__ = [
    "BaseExercise",
    "ExerciseResult", 
    "JointAngles",
    "ExerciseClassifier",
    "ExerciseType",
    "SquatModule",
    "PushupModule",
    "BicepCurlModule",
    "AlternateBicepCurlModule",
]
