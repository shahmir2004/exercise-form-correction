"""k-NN classifier over pose embeddings."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from exercises.base import ExerciseType


REQUIRED_EXERCISE_LABELS = {
    "squat",
    "pushup",
    "bicep_curl",
    "alternate_bicep_curl",
    "idle",
}


class PoseKNNClassifier:
    """k-NN over pre-recorded pose library."""

    def __init__(
        self,
        library_dir: Optional[Path] = None,
        k: int = 10,
        min_embeddings_per_class: int = 1,
    ):
        self.k = k
        self.min_embeddings_per_class = min_embeddings_per_class
        self.library_dir = library_dir or (
            Path(__file__).resolve().parent.parent / "data" / "pose_library"
        )
        self.embeddings: dict[str, np.ndarray] = {}
        self._load_library()

    def _load_library(self) -> None:
        """Load all pre-recorded embeddings."""
        if not self.library_dir.exists():
            return

        for json_file in self.library_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                exercise = data["exercise"]
                embeddings = np.array(data["embeddings"], dtype=np.float32)
                if embeddings.ndim != 2 or embeddings.size == 0:
                    continue
                self.embeddings[exercise] = embeddings
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    def library_counts(self) -> dict[str, int]:
        """Return embedding counts for every loaded exercise label."""
        return {
            exercise: int(embeddings.shape[0])
            for exercise, embeddings in self.embeddings.items()
        }

    def missing_required_libraries(
        self,
        required: Optional[set[str]] = None,
        min_embeddings: Optional[int] = None,
    ) -> set[str]:
        """Return required labels that are absent or under-populated."""
        required_labels = required or REQUIRED_EXERCISE_LABELS
        minimum = self.min_embeddings_per_class if min_embeddings is None else min_embeddings
        counts = self.library_counts()
        return {
            label for label in required_labels
            if counts.get(label, 0) < minimum
        }

    def has_library(self, exercise: str, min_embeddings: Optional[int] = None) -> bool:
        minimum = self.min_embeddings_per_class if min_embeddings is None else min_embeddings
        return self.library_counts().get(exercise, 0) >= minimum

    def classify(self, embedding: np.ndarray) -> Tuple[Optional[ExerciseType], float]:
        """
        Classify a single frame embedding via k-NN.

        Args:
            embedding: shape (66,)

        Returns:
            (exercise_type, confidence) where confidence in [0, 1]
        """
        valid_embeddings = {
            exercise: values for exercise, values in self.embeddings.items()
            if values.shape[0] >= self.min_embeddings_per_class
        }
        if not valid_embeddings:
            return None, 0.0

        neighbors: list[tuple[str, float]] = []
        for exercise, lib_embeddings in valid_embeddings.items():
            if lib_embeddings.size == 0:
                continue
            distances = np.linalg.norm(lib_embeddings - embedding, axis=1)
            min_dist = float(np.min(distances))
            neighbors.append((exercise, min_dist))

        if not neighbors:
            return None, 0.0

        neighbors.sort(key=lambda x: x[1])
        top_k = neighbors[: min(self.k, len(neighbors))]

        best_exercise = top_k[0][0]
        mean_dist = float(np.mean([d for _, d in top_k]))

        max_possible_dist = np.sqrt(12 * 12)
        confidence = max(0.0, 1.0 - mean_dist / max_possible_dist)

        exercise_map: dict[str, Optional[ExerciseType]] = {
            "squat": ExerciseType.SQUAT,
            "pushup": ExerciseType.PUSHUP,
            "bicep_curl": ExerciseType.BICEP_CURL,
            "alternate_bicep_curl": ExerciseType.ALTERNATE_BICEP_CURL,
            "idle": None,
        }

        return exercise_map.get(best_exercise), float(confidence)
