"""Social cognition tasks: theory of mind, emotion recognition, perspective taking."""

from typing import Any

from benchmark.base import BenchmarkTask


class TheoryOfMindTask(BenchmarkTask):
    """Test understanding of others' mental states."""

    def __init__(self) -> None:
        super().__init__(
            name="theory_of_mind",
            category="social_cognition",
            description=(
                "Tests understanding of others' mental states (beliefs, desires, "
                "intentions) via false-belief tasks and perspective-taking."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        scenario: str = sample.get("scenario", "")
        character: str = sample.get("character", "the character")
        question: str = sample.get("question", "")
        return (
            f"Read the following scenario carefully.\n\n"
            f"Scenario:\n{scenario}\n\n"
            f"Based on this scenario, answer the following question about "
            f"{character}'s mental state:\n{question}\nAnswer:"
        )

    def score(self, model_output: str, ground_truth: dict) -> float:
        samples: list[dict] = ground_truth.get("samples", [])
        outputs: list[str] = ground_truth.get("outputs", [model_output])
        if not samples:
            expected: str = ground_truth.get("expected", "")
            return 1.0 if expected.lower() in model_output.lower() else 0.0
        correct = sum(
            1
            for out, s in zip(outputs, samples)
            if s.get("expected", "").lower() in out.lower()
        )
        return correct / len(samples)

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        return {
            "belief_attribution_accuracy": score,
            "model_output": model_output,
            "expected": ground_truth.get("expected", ""),
        }


class EmotionRecognitionTask(BenchmarkTask):
    """Test ability to recognize emotional states from descriptions."""

    EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

    def __init__(self) -> None:
        super().__init__(
            name="emotion_recognition",
            category="social_cognition",
            description=(
                "Tests ability to recognize emotional states from textual "
                "descriptions using weighted F1 scoring."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        description: str = sample.get("description", "")
        emotions_list = ", ".join(self.EMOTIONS)
        return (
            f"Identify the primary emotion expressed in the following description.\n\n"
            f"Description: {description}\n\n"
            f"Choose one of: {emotions_list}\nEmotion:"
        )

    def _extract_emotion(self, text: str) -> str:
        lower = text.lower()
        for emotion in self.EMOTIONS:
            if emotion in lower:
                return emotion
        return "neutral"

    def score(self, model_output: str, ground_truth: dict) -> float:
        samples: list[dict] = ground_truth.get("samples", [])
        outputs: list[str] = ground_truth.get("outputs", [model_output])
        if not samples:
            expected = ground_truth.get("expected", "neutral")
            predicted = self._extract_emotion(model_output)
            return 1.0 if predicted == expected.lower() else 0.0

        # Weighted F1 across emotion categories
        emotion_stats: dict[str, dict[str, int]] = {
            e: {"tp": 0, "fp": 0, "fn": 0} for e in self.EMOTIONS
        }
        for out, s in zip(outputs, samples):
            predicted = self._extract_emotion(out)
            expected = s.get("expected", "neutral").lower()
            if predicted == expected:
                emotion_stats[predicted]["tp"] += 1
            else:
                emotion_stats[predicted]["fp"] += 1
                emotion_stats[expected]["fn"] += 1

        f1_scores, weights = [], []
        for emotion in self.EMOTIONS:
            tp = emotion_stats[emotion]["tp"]
            fp = emotion_stats[emotion]["fp"]
            fn = emotion_stats[emotion]["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            support = tp + fn
            f1_scores.append(f1)
            weights.append(support)

        total_support = sum(weights)
        if total_support == 0:
            return 0.0
        weighted_f1 = sum(f * w for f, w in zip(f1_scores, weights)) / total_support
        return float(weighted_f1)

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        predicted = self._extract_emotion(model_output)
        return {
            "weighted_f1": score,
            "predicted_emotion": predicted,
            "expected_emotion": ground_truth.get("expected", "neutral"),
        }


class PerspectiveTakingTask(BenchmarkTask):
    """Test ability to understand situations from different viewpoints."""

    def __init__(self) -> None:
        super().__init__(
            name="perspective_taking",
            category="social_cognition",
            description=(
                "Tests ability to understand situations from different viewpoints "
                "in multi-character scenario analysis."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        scenario: str = sample.get("scenario", "")
        characters: list[str] = sample.get("characters", [])
        question: str = sample.get("question", "")
        char_list = ", ".join(characters) if characters else "the characters"
        return (
            f"Consider the following scenario from the perspectives of all involved parties.\n\n"
            f"Scenario:\n{scenario}\n\n"
            f"Characters: {char_list}\n\n"
            f"Question: {question}\nAnswer:"
        )

    def score(self, model_output: str, ground_truth: dict) -> float:
        perspectives: list[dict] = ground_truth.get("perspectives", [])
        if not perspectives:
            expected = ground_truth.get("expected", "")
            return 1.0 if expected.lower() in model_output.lower() else 0.0

        total_weight = 0.0
        weighted_correct = 0.0
        for perspective in perspectives:
            expected = perspective.get("expected", "")
            complexity = float(perspective.get("complexity", 1.0))
            correct = 1.0 if expected.lower() in model_output.lower() else 0.0
            weighted_correct += complexity * correct
            total_weight += complexity
        return weighted_correct / total_weight if total_weight > 0 else 0.0

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        perspectives: list[dict] = ground_truth.get("perspectives", [])
        perspective_scores = {
            p.get("character", f"char_{i}"): (
                1.0 if p.get("expected", "").lower() in model_output.lower() else 0.0
            )
            for i, p in enumerate(perspectives)
        }
        return {
            "weighted_accuracy": score,
            "perspective_scores": perspective_scores,
            "model_output": model_output,
        }
