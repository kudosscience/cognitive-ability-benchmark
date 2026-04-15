from __future__ import annotations

from dataclasses import asdict
from typing import Any

from benchmark.tasks.conflict_planning import ConflictPlanningTask
from benchmark.tasks.habit_override import HabitOverrideTask
from benchmark.tasks.rule_shift import RuleShiftTask

TASK_REGISTRY = {
    HabitOverrideTask.name: HabitOverrideTask,
    RuleShiftTask.name: RuleShiftTask,
    ConflictPlanningTask.name: ConflictPlanningTask,
}


class CogniFlexSuite:
    """Convenience wrapper for generating and scoring all CogniFlex tasks."""

    task_order = (
        HabitOverrideTask,
        RuleShiftTask,
        ConflictPlanningTask,
    )

    @classmethod
    def generate_samples(
        cls,
        num_samples_per_task: int,
        seed: int,
    ) -> dict[str, list[Any]]:
        datasets: dict[str, list[Any]] = {}
        for task_index, task_type in enumerate(cls.task_order):
            task_seed_stride = 1000
            task_seed = seed + ((task_index + 1) * task_seed_stride)
            datasets[task_type.name] = task_type.generate_samples(
                num_samples=num_samples_per_task,
                seed=task_seed,
            )
        return datasets

    @classmethod
    def generate_records(
        cls,
        num_samples_per_task: int,
        seed: int,
    ) -> dict[str, list[dict[str, Any]]]:
        sample_sets = cls.generate_samples(
            num_samples_per_task=num_samples_per_task,
            seed=seed,
        )

        records: dict[str, list[dict[str, Any]]] = {}
        for task_name, samples in sample_sets.items():
            records[task_name] = [asdict(sample) for sample in samples]
        return records

    @classmethod
    def score_predictions(
        cls,
        task_name: str,
        model_outputs: list[str],
        samples: list[Any],
    ) -> list[float]:
        task_type = TASK_REGISTRY.get(task_name)
        if task_type is None:
            supported = ", ".join(TASK_REGISTRY)
            raise ValueError(f"Unknown task '{task_name}'. Supported tasks: {supported}")

        if len(model_outputs) != len(samples):
            raise ValueError("model_outputs and samples must have the same length.")

        return [
            task_type.score_sample(output, sample)
            for output, sample in zip(model_outputs, samples)
        ]
