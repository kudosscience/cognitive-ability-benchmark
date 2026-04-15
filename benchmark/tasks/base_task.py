from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_csv_sequence(text: str) -> str:
    items = [segment.strip().upper() for segment in text.split(",") if segment.strip()]
    return ",".join(items)


class ExactMatchTask:
    """Minimal exact-match task helper for deterministic benchmark scoring."""

    name = "exact_match"
    category = "executive_functions"
    description = "Exact-match scoring task."

    @staticmethod
    def normalize_output(output: str) -> str:
        return normalize_text(output)

    @classmethod
    def score_output(cls, model_output: str, expected_output: str) -> float:
        if model_output is None:
            return 0.0

        normalized_model_output = cls.normalize_output(str(model_output))
        normalized_expected_output = cls.normalize_output(str(expected_output))
        return float(normalized_model_output == normalized_expected_output)

    @classmethod
    def evaluate_output(cls, model_output: str, expected_output: str) -> dict[str, Any]:
        score = cls.score_output(model_output=model_output, expected_output=expected_output)
        return {
            "score": score,
            "is_correct": bool(score),
            "expected_output": expected_output,
            "model_output": model_output,
        }

    @staticmethod
    def to_record(sample: Any) -> dict[str, Any]:
        if is_dataclass(sample):
            return asdict(sample)
        if isinstance(sample, dict):
            return dict(sample)
        raise TypeError("Sample must be a dataclass instance or dictionary.")

    @classmethod
    def score_sample(cls, model_output: str, sample: Any) -> float:
        if isinstance(sample, dict):
            expected_output = sample.get("expected_output")
        else:
            expected_output = getattr(sample, "expected_output", None)

        if expected_output is None:
            raise ValueError("Sample must include an expected_output field.")

        return cls.score_output(model_output=model_output, expected_output=str(expected_output))

    @classmethod
    def evaluate_sample(cls, model_output: str, sample: Any) -> dict[str, Any]:
        if isinstance(sample, dict):
            expected_output = sample.get("expected_output")
        else:
            expected_output = getattr(sample, "expected_output", None)

        if expected_output is None:
            raise ValueError("Sample must include an expected_output field.")

        return cls.evaluate_output(
            model_output=model_output,
            expected_output=str(expected_output),
        )
