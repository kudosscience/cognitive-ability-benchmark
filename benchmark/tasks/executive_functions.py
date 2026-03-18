"""Executive function tasks: planning, cognitive flexibility, working memory."""

import re
from typing import Any

from benchmark.base import BenchmarkTask


class PlanningTask(BenchmarkTask):
    """Evaluate goal-directed planning given a goal and available actions."""

    def __init__(self) -> None:
        super().__init__(
            name="planning",
            category="executive_functions",
            description=(
                "Gives a goal and available actions, asks model to produce a valid "
                "plan. Evaluates plan validity and efficiency."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        goal: str = sample.get("goal", "")
        actions: list[str] = sample.get("actions", [])
        constraints: str = sample.get("constraints", "")
        action_list = "\n".join(f"  - {a}" for a in actions)
        prompt = (
            f"Goal: {goal}\n\n"
            f"Available actions:\n{action_list}\n\n"
        )
        if constraints:
            prompt += f"Constraints: {constraints}\n\n"
        prompt += (
            "Produce a step-by-step plan to achieve the goal using only the "
            "available actions. List each step on a new line."
        )
        return prompt

    def _extract_steps(self, text: str) -> list[str]:
        steps = []
        for line in text.split("\n"):
            cleaned = re.sub(r"^\s*[\d]+[.)]\s*", "", line).strip()
            if cleaned:
                steps.append(cleaned.lower())
        return steps

    def score(self, model_output: str, ground_truth: dict) -> float:
        valid_actions: list[str] = [
            a.lower() for a in ground_truth.get("valid_actions", [])
        ]
        optimal_length: int = ground_truth.get("optimal_length", 1)
        steps = self._extract_steps(model_output)
        if not steps:
            return 0.0
        if valid_actions:
            valid_count = sum(
                1
                for step in steps
                if any(va in step for va in valid_actions)
            )
            validity = valid_count / len(steps)
        else:
            validity = 1.0
        efficiency = optimal_length / max(len(steps), optimal_length)
        return float(0.7 * validity + 0.3 * efficiency)

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        steps = self._extract_steps(model_output)
        valid_actions: list[str] = [
            a.lower() for a in ground_truth.get("valid_actions", [])
        ]
        invalid_steps = (
            [s for s in steps if not any(va in s for va in valid_actions)]
            if valid_actions
            else []
        )
        return {
            "score": score,
            "num_steps": len(steps),
            "optimal_length": ground_truth.get("optimal_length", 1),
            "invalid_steps": invalid_steps,
        }


class CognitiveFlexibilityTask(BenchmarkTask):
    """Test ability to switch between different rules or strategies."""

    def __init__(self) -> None:
        super().__init__(
            name="cognitive_flexibility",
            category="executive_functions",
            description=(
                "Tests ability to switch between different rules or strategies "
                "in rule-switching scenarios."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        scenario: str = sample.get("scenario", "")
        rule: str = sample.get("current_rule", "")
        items: list[str] = sample.get("items", [])
        formatted_items = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        return (
            f"Scenario: {scenario}\n"
            f"Current rule: {rule}\n\n"
            f"Apply the rule to each item:\n{formatted_items}\n\n"
            f"Provide your answers as a numbered list."
        )

    def score(self, model_output: str, ground_truth: dict) -> float:
        switch_trials: list[dict] = ground_truth.get("switch_trials", [])
        stay_trials: list[dict] = ground_truth.get("stay_trials", [])

        def _trial_accuracy(trials: list[dict]) -> float:
            if not trials:
                return 1.0
            correct = 0
            for trial in trials:
                expected = trial.get("expected", "").lower()
                if expected in model_output.lower():
                    correct += 1
            return correct / len(trials)

        switch_acc = _trial_accuracy(switch_trials)
        stay_acc = _trial_accuracy(stay_trials)
        return float((switch_acc + stay_acc) / 2.0)

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        return {
            "flexibility_score": score,
            "model_output": model_output,
        }


class WorkingMemoryTask(BenchmarkTask):
    """Test ability to maintain and manipulate information over a sequence."""

    def __init__(self) -> None:
        super().__init__(
            name="working_memory",
            category="executive_functions",
            description=(
                "Tests ability to maintain and manipulate information. "
                "Includes serial recall and N-back type tasks."
            ),
        )

    def generate_prompt(self, sample: dict) -> str:
        sequence: list[str] = sample.get("sequence", [])
        task_type: str = sample.get("task_type", "recall")
        if task_type == "n_back":
            n: int = sample.get("n", 2)
            return (
                f"You will see a sequence of items. For each item, indicate whether "
                f"it matches the item {n} positions back (Yes/No).\n\n"
                f"Sequence: {', '.join(sequence)}\n\nProvide Yes/No for each item starting from position {n+1}."
            )
        return (
            f"Memorize the following sequence and recall it in order.\n\n"
            f"Sequence: {', '.join(sequence)}\n\nNow recall the sequence:"
        )

    def score(self, model_output: str, ground_truth: dict) -> float:
        expected_sequence: list[str] = ground_truth.get("expected_sequence", [])
        if not expected_sequence:
            return 0.5
        n = len(expected_sequence)
        output_lower = model_output.lower()
        correct = 0
        for i, item in enumerate(expected_sequence):
            if item.lower() in output_lower:
                weight = (i + 1) / n
                correct += weight
        max_score = sum((i + 1) / n for i in range(n))
        return float(correct / max_score) if max_score > 0 else 0.0

    def evaluate(self, model_output: str, ground_truth: dict) -> dict[str, Any]:
        score = self.score(model_output, ground_truth)
        return {
            "weighted_recall": score,
            "model_output": model_output,
            "expected_sequence": ground_truth.get("expected_sequence", []),
        }
