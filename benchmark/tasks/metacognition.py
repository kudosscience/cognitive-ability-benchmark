"""Metacognition tasks: confidence calibration, knowledge boundary, self-assessment."""

import re
from typing import Any

from benchmark.base import BenchmarkTask
from benchmark.evaluation.metrics import compute_ece


class ConfidenceCalibrationTask(BenchmarkTask):
    """Evaluate calibration between stated confidence and actual accuracy."""

    def __init__(self) -> None:
        super().__init__(
            name="confidence_calibration",
            category="metacognition",
            description=(
                "Asks the model to provide a confidence score (0-100) with each "
                "answer and evaluates calibration using Expected Calibration Error."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        question: str = sample.get("question", "")
        return (
            f"Answer the following question and provide your confidence (0-100).\n\n"
            f"Question: {question}\n\n"
            f"Respond in the format:\nAnswer: <your answer>\nConfidence: <0-100>"
        )

    def _parse_output(self, model_output: str) -> tuple[str, float]:
        answer_match = re.search(r"Answer:\s*(.+?)(?:\n|$)", model_output, re.IGNORECASE)
        conf_match = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)", model_output, re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else model_output.strip()
        confidence = float(conf_match.group(1)) / 100.0 if conf_match else 0.5
        confidence = max(0.0, min(1.0, confidence))
        return answer, confidence

    def score(self, model_output: str, ground_truth: dict) -> float:
        samples: list[dict] = ground_truth.get("samples", [])
        outputs: list[str] = ground_truth.get("outputs", [model_output])
        if not samples:
            _, confidence = self._parse_output(model_output)
            expected = ground_truth.get("expected", "")
            is_correct = model_output.lower().find(expected.lower()) >= 0 if expected else False
            ece = compute_ece([confidence], [1.0 if is_correct else 0.0])
            return max(0.0, 1.0 - ece)

        confidences, accuracies = [], []
        for out, s in zip(outputs, samples):
            _, conf = self._parse_output(out)
            expected = s.get("expected", "")
            correct = 1.0 if out.lower().find(expected.lower()) >= 0 and expected else 0.0
            confidences.append(conf)
            accuracies.append(correct)
        ece = compute_ece(confidences, accuracies)
        return max(0.0, 1.0 - ece)

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        samples: list[dict] = ground_truth.get("samples", [])
        outputs: list[str] = ground_truth.get("outputs", [model_output])
        if not samples:
            samples = [ground_truth]
            outputs = [model_output]
        confidences, accuracies, calibration_data = [], [], []
        for out, s in zip(outputs, samples):
            _, conf = self._parse_output(out)
            expected = s.get("expected", "")
            correct = 1.0 if out.lower().find(expected.lower()) >= 0 and expected else 0.0
            confidences.append(conf)
            accuracies.append(correct)
            calibration_data.append({"confidence": conf, "correct": bool(correct)})
        ece = compute_ece(confidences, accuracies)
        return {
            "ece": ece,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "calibration_data": calibration_data,
        }


class KnowledgeBoundaryTask(BenchmarkTask):
    """Test if the model correctly identifies what it knows vs. doesn't know."""

    UNKNOWN_PHRASES = [
        "i don't know",
        "i do not know",
        "unknown",
        "i'm not sure",
        "i am not sure",
        "cannot answer",
        "not enough information",
        "i cannot",
    ]

    def __init__(self) -> None:
        super().__init__(
            name="knowledge_boundary",
            category="metacognition",
            description=(
                "Tests if the model correctly identifies what it knows versus "
                "what it does not know."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        question: str = sample.get("question", "")
        return (
            f"Answer the following question if you know the answer. "
            f"If you don't know, say 'I don't know'.\n\n"
            f"Question: {question}\nAnswer:"
        )

    def _is_unknown(self, text: str) -> bool:
        lower = text.lower()
        return any(phrase in lower for phrase in self.UNKNOWN_PHRASES)

    def score(self, model_output: str, ground_truth: dict) -> float:
        samples: list[dict] = ground_truth.get("samples", [])
        outputs: list[str] = ground_truth.get("outputs", [model_output])
        if not samples:
            is_answerable: bool = ground_truth.get("is_answerable", True)
            pred_unknown = self._is_unknown(model_output)
            correct = (is_answerable and not pred_unknown) or (
                not is_answerable and pred_unknown
            )
            return 1.0 if correct else 0.0

        tp = fp = fn = tn = 0
        for out, s in zip(outputs, samples):
            is_answerable = s.get("is_answerable", True)
            pred_unknown = self._is_unknown(out)
            if not is_answerable and pred_unknown:
                tp += 1
            elif is_answerable and pred_unknown:
                fp += 1
            elif not is_answerable and not pred_unknown:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / len(samples)
        return (f1 + accuracy) / 2.0

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        return {"score": score, "model_output": model_output}


class SelfAssessmentTask(BenchmarkTask):
    """Measure correlation between self-assessment and actual correctness."""

    def __init__(self) -> None:
        super().__init__(
            name="self_assessment",
            category="metacognition",
            description=(
                "Asks the model to self-assess the correctness of its answers "
                "and measures the correlation with actual correctness."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        question: str = sample.get("question", "")
        model_answer: str = sample.get("model_answer", "")
        return (
            f"Question: {question}\n"
            f"Answer given: {model_answer}\n\n"
            f"Is the above answer correct? Respond with 'Yes' or 'No' and a brief explanation."
        )

    def _parse_self_assessment(self, text: str) -> float:
        lower = text.lower()
        if lower.startswith("yes") or "\nyes" in lower or "is correct" in lower:
            return 1.0
        if lower.startswith("no") or "\nno" in lower or "is incorrect" in lower:
            return 0.0
        return 0.5

    def score(self, model_output: str, ground_truth: dict) -> float:
        samples: list[dict] = ground_truth.get("samples", [])
        outputs: list[str] = ground_truth.get("outputs", [model_output])
        if not samples:
            actual_correct: float = float(ground_truth.get("actual_correct", 0.5))
            pred = self._parse_self_assessment(model_output)
            return 1.0 - abs(pred - actual_correct)

        assessments = [self._parse_self_assessment(o) for o in outputs]
        actuals = [float(s.get("actual_correct", 0.5)) for s in samples]
        if len(assessments) < 2:
            return 1.0 - abs(assessments[0] - actuals[0]) if assessments else 0.5
        mean_a = sum(assessments) / len(assessments)
        mean_b = sum(actuals) / len(actuals)
        num = sum((a - mean_a) * (b - mean_b) for a, b in zip(assessments, actuals))
        den_a = sum((a - mean_a) ** 2 for a in assessments) ** 0.5
        den_b = sum((b - mean_b) ** 2 for b in actuals) ** 0.5
        if den_a == 0 or den_b == 0:
            return 0.5
        corr = num / (den_a * den_b)
        return float(max(0.0, (corr + 1.0) / 2.0))

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        return {
            "correlation_score": score,
            "model_output": model_output,
        }
