"""Canonical supported exercise metadata.

Keep integration-facing labels in one place so API routes, classifiers, and
clients do not drift as more exercise modules are added.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExerciseDefinition:
    label: str
    display_name: str
    variants: tuple[str, ...] = ()


SUPPORTED_EXERCISES: tuple[ExerciseDefinition, ...] = (
    ExerciseDefinition("squat", "Squat"),
    ExerciseDefinition("pushup", "Push-up"),
    ExerciseDefinition(
        "bicep_curl",
        "Bicep Curl",
        ("curl-stand", "curl-seat"),
    ),
    ExerciseDefinition(
        "alternate_bicep_curl",
        "Alternate Bicep Curl",
        ("alt-stand", "alt-seat"),
    ),
)

IDLE_LABEL = "idle"

SUPPORTED_EXERCISE_LABELS = frozenset(defn.label for defn in SUPPORTED_EXERCISES)
VARIANT_LABELS = frozenset(
    variant
    for definition in SUPPORTED_EXERCISES
    for variant in definition.variants
)


def supported_exercises_payload() -> list[dict]:
    """Return a JSON-serializable exercise list for health/integration endpoints."""
    return [
        {
            "label": definition.label,
            "display_name": definition.display_name,
            "variants": list(definition.variants),
        }
        for definition in SUPPORTED_EXERCISES
    ]
