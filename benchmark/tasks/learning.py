"""Learning tasks: few-shot learning, in-context learning, knowledge transfer."""

import difflib
from typing import Any

from benchmark.base import BenchmarkTask


class FewShotLearningTask(BenchmarkTask):
    """Evaluate whether a model can learn from k examples in the prompt."""

    def __init__(self) -> None:
        super().__init__(
            name="few_shot_learning",
            category="learning",
            description=(
                "Evaluates whether a model can learn from a small number of "
                "examples provided in the prompt."
            ),
        )

    # ------------------------------------------------------------------
    def generate_prompt(self, sample: dict) -> str:
        examples: list[dict] = sample.get("examples", [])
        query: str = sample.get("query", "")
        lines = ["Learn from the following examples and answer the query.\n"]
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Input: {ex.get('input', '')}")
            lines.append(f"  Output: {ex.get('output', '')}")
        lines.append(f"\nQuery: {query}")
        lines.append("Answer:")
        return "\n".join(lines)

    def score(self, model_output: str, ground_truth: dict) -> float:
        samples: list[dict] = ground_truth.get("samples", [])
        outputs: list[str] = ground_truth.get("outputs", [model_output])
        if not samples:
            expected = ground_truth.get("expected", "")
            return 1.0 if model_output.strip().lower() == expected.strip().lower() else 0.0
        correct = sum(
            1
            for out, s in zip(outputs, samples)
            if out.strip().lower() == s.get("expected", "").strip().lower()
        )
        return correct / len(samples) if samples else 0.0

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        samples: list[dict] = ground_truth.get("samples", [])
        outputs: list[str] = ground_truth.get("outputs", [model_output])
        if not samples:
            expected = ground_truth.get("expected", "")
            correct = 1 if model_output.strip().lower() == expected.strip().lower() else 0
            return {"accuracy": float(correct), "num_correct": correct, "num_total": 1}
        num_correct = sum(
            1
            for out, s in zip(outputs, samples)
            if out.strip().lower() == s.get("expected", "").strip().lower()
        )
        accuracy = num_correct / len(samples)
        return {
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": len(samples),
        }


class InContextLearningTask(BenchmarkTask):
    """Evaluate adaptation to novel information provided in context."""

    def __init__(self) -> None:
        super().__init__(
            name="in_context_learning",
            category="learning",
            description=(
                "Evaluates adaptation to new information provided in context; "
                "tests if the model uses novel facts to answer questions."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        context: str = sample.get("context", "")
        question: str = sample.get("question", "")
        return (
            f"Context:\n{context}\n\n"
            f"Based on the context above, answer the following question:\n"
            f"Question: {question}\nAnswer:"
        )

    def score(self, model_output: str, ground_truth: dict) -> float:
        expected: str = ground_truth.get("expected", "")
        ratio = difflib.SequenceMatcher(
            None, model_output.strip().lower(), expected.strip().lower()
        ).ratio()
        return float(ratio)

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        return {
            "similarity_score": score,
            "model_output": model_output,
            "expected": ground_truth.get("expected", ""),
        }


class KnowledgeTransferTask(BenchmarkTask):
    """Evaluate applying knowledge from one domain to analogous problems."""

    def __init__(self) -> None:
        super().__init__(
            name="knowledge_transfer",
            category="learning",
            description=(
                "Evaluates the ability to apply knowledge from one domain to "
                "analogous problems in another domain."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        source_domain: str = sample.get("source_domain", "")
        source_example: str = sample.get("source_example", "")
        target_domain: str = sample.get("target_domain", "")
        target_problem: str = sample.get("target_problem", "")
        return (
            f"In {source_domain}, we know that: {source_example}\n\n"
            f"Now consider the following analogous problem in {target_domain}:\n"
            f"{target_problem}\n\nApply the same reasoning and provide your answer:"
        )

    def score(self, model_output: str, ground_truth: dict) -> float:
        expected: str = ground_truth.get("expected", "")
        return 1.0 if expected.strip().lower() in model_output.strip().lower() else 0.0

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        return {
            "correct": bool(score),
            "score": score,
            "model_output": model_output,
            "expected": ground_truth.get("expected", ""),
        }
