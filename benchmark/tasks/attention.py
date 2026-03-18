"""Attention tasks: selective attention, sustained attention, divided attention."""

from typing import Any

from benchmark.base import BenchmarkTask


class SelectiveAttentionTask(BenchmarkTask):
    """Test ability to focus on relevant info while ignoring distractors."""

    def __init__(self) -> None:
        super().__init__(
            name="selective_attention",
            category="attention",
            description=(
                "Tests ability to focus on relevant information while ignoring "
                "distractors embedded in a long context."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        context: str = sample.get("context", "")
        question: str = sample.get("question", "")
        return (
            f"Read the following text carefully and answer the question.\n\n"
            f"Text:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

    def score(self, model_output: str, ground_truth: dict) -> float:
        key_facts: list[str] = ground_truth.get("key_facts", [])
        if not key_facts:
            expected = ground_truth.get("expected", "")
            return 1.0 if expected.lower() in model_output.lower() else 0.0
        found = sum(1 for fact in key_facts if fact.lower() in model_output.lower())
        return found / len(key_facts)

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        key_facts: list[str] = ground_truth.get("key_facts", [])
        found_facts = [f for f in key_facts if f.lower() in model_output.lower()]
        return {
            "recall": score,
            "found_facts": found_facts,
            "total_key_facts": len(key_facts),
        }


class SustainedAttentionTask(BenchmarkTask):
    """Test consistency over a long sequence of items."""

    def __init__(self) -> None:
        super().__init__(
            name="sustained_attention",
            category="attention",
            description=(
                "Tests consistency of performance over a long sequence of items "
                "with a penalty for inconsistency."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        items: list[str] = sample.get("items", [])
        task_instruction: str = sample.get(
            "instruction", "Classify each item as positive or negative."
        )
        formatted_items = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        return (
            f"{task_instruction}\n\nItems:\n{formatted_items}\n\n"
            f"Provide your answers as a numbered list."
        )

    def score(self, model_output: str, ground_truth: dict) -> float:
        expected: list[str] = ground_truth.get("expected", [])
        if not expected:
            return 0.5
        lines = [
            line.strip()
            for line in model_output.strip().split("\n")
            if line.strip()
        ]
        predictions = []
        for line in lines:
            # Strip leading numbering like "1.", "2)", etc.
            import re
            cleaned = re.sub(r"^\d+[.)]\s*", "", line).strip().lower()
            if cleaned:
                predictions.append(cleaned)

        n = min(len(predictions), len(expected))
        if n == 0:
            return 0.0

        window = max(1, n // 5)
        rolling_accs = []
        for i in range(0, n, window):
            chunk_preds = predictions[i : i + window]
            chunk_exp = expected[i : i + window]
            correct = sum(
                1 for p, e in zip(chunk_preds, chunk_exp) if p == e.lower()
            )
            rolling_accs.append(correct / len(chunk_preds))

        overall_correct = sum(
            1 for p, e in zip(predictions[:n], expected[:n]) if p == e.lower()
        )
        accuracy = overall_correct / n

        if len(rolling_accs) > 1:
            mean_acc = sum(rolling_accs) / len(rolling_accs)
            variance = sum((a - mean_acc) ** 2 for a in rolling_accs) / len(rolling_accs)
            std_dev = variance ** 0.5
            penalty = min(0.3, std_dev)
        else:
            penalty = 0.0

        return float(max(0.0, accuracy - penalty))

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        return {
            "score": score,
            "model_output_length": len(model_output),
        }


class DividedAttentionTask(BenchmarkTask):
    """Test ability to track multiple streams of information simultaneously."""

    def __init__(self) -> None:
        super().__init__(
            name="divided_attention",
            category="attention",
            description=(
                "Tests ability to track multiple streams of information simultaneously "
                "by processing interleaved sequences from different topics."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        streams: list[dict] = sample.get("streams", [])
        intro = "Track the following interleaved streams and answer questions about each.\n\n"
        body_parts = []
        for stream in streams:
            body_parts.append(
                f"Stream '{stream.get('name', 'unknown')}':\n"
                + "\n".join(f"  - {item}" for item in stream.get("items", []))
            )
        questions_part = "\n\nQuestions:\n" + "\n".join(
            f"- About '{s.get('name', 'unknown')}': {s.get('question', '')}"
            for s in streams
            if s.get("question")
        )
        return intro + "\n\n".join(body_parts) + questions_part

    def score(self, model_output: str, ground_truth: dict) -> float:
        streams: list[dict] = ground_truth.get("streams", [])
        if not streams:
            expected = ground_truth.get("expected", "")
            return 1.0 if expected.lower() in model_output.lower() else 0.0

        total_weight = sum(s.get("weight", 1.0) for s in streams)
        weighted_score = 0.0
        for stream in streams:
            expected = stream.get("expected", "")
            weight = stream.get("weight", 1.0)
            correct = 1.0 if expected.lower() in model_output.lower() else 0.0
            weighted_score += weight * correct
        return weighted_score / total_weight if total_weight > 0 else 0.0

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        streams: list[dict] = ground_truth.get("streams", [])
        stream_scores = {}
        for stream in streams:
            name = stream.get("name", "unknown")
            expected = stream.get("expected", "")
            stream_scores[name] = 1.0 if expected.lower() in model_output.lower() else 0.0
        return {
            "weighted_accuracy": score,
            "stream_scores": stream_scores,
        }
