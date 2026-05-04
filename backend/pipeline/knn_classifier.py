"""k-NN classifier over pose embeddings."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from exercises.classifier import ExerciseType


class PoseKNNClassifier:
    """k-NN over pre-recorded pose library."""

    def __init__(self, library_dir: Optional[Path] = None, k: int = 10):
        self.k = k
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

    def classify(self, embedding: np.ndarray) -> Tuple[Optional[ExerciseType], float]:
        """
        Classify a single frame embedding via k-NN.

        Args:
            embedding: shape (66,)

        Returns:
            (exercise_type, confidence) where confidence in [0, 1]
        """
        if not self.embeddings:
            return None, 0.0

        neighbors: list[tuple[str, float]] = []
        for exercise, lib_embeddings in self.embeddings.items():
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
