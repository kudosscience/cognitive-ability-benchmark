from __future__ import annotations

from typing import Any, Iterable

from benchmark.tasks.base_task import ExactMatchTask, normalize_csv_sequence
from benchmark.utils.generator import (
    DEFAULT_DATASET_SEED,
    ConflictPlanningSample,
    generate_conflict_planning_dataset,
    simulate_conflict_plan,
)


class ConflictPlanningTask(ExactMatchTask):
    """Executive function task targeting complex planning under conflicting actions."""

    name = "conflict_planning"
    category = "executive_functions"
    description = (
        "Evaluate multi-step planning when tempting actions consume critical resources "
        "and only one shortest plan reaches the goal."
    )

    @staticmethod
    def normalize_output(output: str) -> str:
        return normalize_csv_sequence(output)

    @staticmethod
    def generate_samples(
        num_samples: int,
        seed: int = DEFAULT_DATASET_SEED,
    ) -> list[ConflictPlanningSample]:
        return generate_conflict_planning_dataset(num_samples=num_samples, seed=seed)

    @classmethod
    def generate_records(
        cls,
        num_samples: int,
        seed: int = DEFAULT_DATASET_SEED,
    ) -> list[dict[str, Any]]:
        samples = cls.generate_samples(num_samples=num_samples, seed=seed)
        return [cls.to_record(sample) for sample in samples]

    @staticmethod
    def validate_plan(
        sample: ConflictPlanningSample,
        action_sequence: str | Iterable[str],
    ) -> bool:
        return simulate_conflict_plan(sample=sample, action_sequence=action_sequence)
