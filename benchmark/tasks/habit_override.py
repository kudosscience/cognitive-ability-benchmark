from __future__ import annotations

from typing import Any

from benchmark.tasks.base_task import ExactMatchTask
from benchmark.utils.generator import (
    DEFAULT_DATASET_SEED,
    HabitOverrideSample,
    generate_habit_override_dataset,
)


class HabitOverrideTask(ExactMatchTask):
    """Executive function task targeting inhibitory control."""

    name = "habit_override"
    category = "executive_functions"
    description = (
        "Evaluate whether a model can suppress a familiar next-token trajectory "
        "when a late rule override changes the transition behavior."
    )

    @staticmethod
    def generate_samples(
        num_samples: int,
        seed: int = DEFAULT_DATASET_SEED,
    ) -> list[HabitOverrideSample]:
        return generate_habit_override_dataset(num_samples=num_samples, seed=seed)

    @classmethod
    def generate_records(
        cls,
        num_samples: int,
        seed: int = DEFAULT_DATASET_SEED,
    ) -> list[dict[str, Any]]:
        samples = cls.generate_samples(num_samples=num_samples, seed=seed)
        return [cls.to_record(sample) for sample in samples]
