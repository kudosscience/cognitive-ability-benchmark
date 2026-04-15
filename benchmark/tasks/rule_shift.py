from __future__ import annotations

import re
from typing import Any

from benchmark.tasks.base_task import ExactMatchTask
from benchmark.utils.generator import (
    DEFAULT_DATASET_SEED,
    RuleShiftSample,
    generate_rule_shift_dataset,
)

INTEGER_PATTERN = re.compile(r"-?\d+")


class RuleShiftTask(ExactMatchTask):
    """Executive function task targeting cognitive flexibility."""

    name = "rule_shift"
    category = "executive_functions"
    description = (
        "Evaluate whether a model can adapt when operation semantics shift "
        "mid-sequence and continue the computation with the new rules."
    )

    @staticmethod
    def normalize_output(output: str) -> str:
        match = INTEGER_PATTERN.search(output.strip())
        return match.group(0) if match else ""

    @staticmethod
    def generate_samples(
        num_samples: int,
        seed: int = DEFAULT_DATASET_SEED,
    ) -> list[RuleShiftSample]:
        return generate_rule_shift_dataset(num_samples=num_samples, seed=seed)

    @classmethod
    def generate_records(
        cls,
        num_samples: int,
        seed: int = DEFAULT_DATASET_SEED,
    ) -> list[dict[str, Any]]:
        samples = cls.generate_samples(num_samples=num_samples, seed=seed)
        return [cls.to_record(sample) for sample in samples]
