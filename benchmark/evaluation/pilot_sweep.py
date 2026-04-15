from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any

from benchmark.cogniflex_suite import CogniFlexSuite

MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99
TASK_NAMES = ("habit_override", "rule_shift", "conflict_planning")
TASK_COUNT = len(TASK_NAMES)

BASELINE_TASK_SKILLS = {
    "habit_override": 0.5,
    "rule_shift": 0.5,
    "conflict_planning": 0.5,
}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    task_skill: dict[str, float]
    difficulty_sensitivity: float


@dataclass(frozen=True)
class PilotSweepConfig:
    seed: int
    num_samples_per_task: int
    model_profiles: list[ModelProfile]


def _clamp_probability(value: float) -> float:
    if value < MIN_PROBABILITY:
        return MIN_PROBABILITY
    if value > MAX_PROBABILITY:
        return MAX_PROBABILITY
    return value


def _get_task_skill(profile: ModelProfile, task_name: str) -> float:
    return profile.task_skill.get(task_name, BASELINE_TASK_SKILLS[task_name])


def _probability_correct(profile: ModelProfile, task_name: str, difficulty: int) -> float:
    task_skill = _get_task_skill(profile=profile, task_name=task_name)
    difficulty_adjustment = profile.difficulty_sensitivity * (difficulty - 1)
    return _clamp_probability(task_skill - difficulty_adjustment)


def _generate_incorrect_output(task_name: str, sample: Any) -> str:
    if task_name == "habit_override":
        visited_letters = sample.metadata["visited_letters"]
        if len(visited_letters) < 2:
            return sample.expected_output
        truncated_letters = visited_letters[:-1]
        return " > ".join(truncated_letters)

    if task_name == "rule_shift":
        final_state = int(sample.expected_output)
        return str(final_state + 1)

    if task_name == "conflict_planning":
        trap_action_id = sample.metadata["trap_action_ids"][0]
        return f"{trap_action_id},{sample.expected_output}"

    raise ValueError(f"Unsupported task: {task_name}")


def _simulate_prediction(
    profile: ModelProfile,
    task_name: str,
    sample: Any,
    rng: random.Random,
) -> str:
    difficulty = int(sample.difficulty)
    probability = _probability_correct(
        profile=profile,
        task_name=task_name,
        difficulty=difficulty,
    )

    if rng.random() <= probability:
        return sample.expected_output

    return _generate_incorrect_output(task_name=task_name, sample=sample)


def run_pilot_sweep(config: PilotSweepConfig) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    sample_sets = CogniFlexSuite.generate_samples(
        num_samples_per_task=config.num_samples_per_task,
        seed=config.seed,
    )

    results: list[dict[str, Any]] = []
    for profile in config.model_profiles:
        for task_name in TASK_NAMES:
            samples = sample_sets[task_name]
            for sample_index, sample in enumerate(samples, start=1):
                prediction = _simulate_prediction(
                    profile=profile,
                    task_name=task_name,
                    sample=sample,
                    rng=rng,
                )
                task_type = CogniFlexSuite.task_order[TASK_NAMES.index(task_name)]
                score = task_type.score_sample(prediction, sample)

                results.append(
                    {
                        "model_name": profile.name,
                        "task_name": task_name,
                        "sample_index": sample_index,
                        "difficulty": sample.difficulty,
                        "score": float(score),
                    }
                )

    return results


def summarize_pilot_sweep(
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_model_task: dict[tuple[str, str], list[float]] = {}
    by_model_task_difficulty: dict[tuple[str, str, int], list[float]] = {}
    by_model: dict[str, list[float]] = {}

    for row in results:
        model_name = str(row["model_name"])
        task_name = str(row["task_name"])
        difficulty = int(row["difficulty"])
        score = float(row["score"])

        by_model_task.setdefault((model_name, task_name), []).append(score)
        by_model_task_difficulty.setdefault((model_name, task_name, difficulty), []).append(score)
        by_model.setdefault(model_name, []).append(score)

    task_summary = [
        {
            "model_name": model_name,
            "task_name": task_name,
            "mean_score": float(mean(scores)),
            "num_samples": len(scores),
        }
        for (model_name, task_name), scores in sorted(by_model_task.items())
    ]

    difficulty_summary = [
        {
            "model_name": model_name,
            "task_name": task_name,
            "difficulty": difficulty,
            "mean_score": float(mean(scores)),
            "num_samples": len(scores),
        }
        for (model_name, task_name, difficulty), scores in sorted(by_model_task_difficulty.items())
    ]

    overall_summary = [
        {
            "model_name": model_name,
            "overall_score": float(mean(scores)),
            "num_samples": len(scores),
        }
        for model_name, scores in sorted(by_model.items())
    ]

    return {
        "task_summary": task_summary,
        "difficulty_summary": difficulty_summary,
        "overall_summary": overall_summary,
    }


def default_profiles() -> list[ModelProfile]:
    return [
        ModelProfile(
            name="pattern-matcher-small",
            task_skill={
                "habit_override": 0.62,
                "rule_shift": 0.48,
                "conflict_planning": 0.4,
            },
            difficulty_sensitivity=0.14,
        ),
        ModelProfile(
            name="rule-aware-medium",
            task_skill={
                "habit_override": 0.78,
                "rule_shift": 0.71,
                "conflict_planning": 0.58,
            },
            difficulty_sensitivity=0.11,
        ),
        ModelProfile(
            name="planner-large",
            task_skill={
                "habit_override": 0.89,
                "rule_shift": 0.85,
                "conflict_planning": 0.8,
            },
            difficulty_sensitivity=0.08,
        ),
        ModelProfile(
            name="oracle-upper-bound",
            task_skill={
                "habit_override": 0.995,
                "rule_shift": 0.995,
                "conflict_planning": 0.995,
            },
            difficulty_sensitivity=0.0,
        ),
    ]


def config_from_dict(config_dict: dict[str, Any]) -> PilotSweepConfig:
    profiles_raw = config_dict.get("model_profiles", [])
    profiles = [
        ModelProfile(
            name=str(profile["name"]),
            task_skill={
                "habit_override": float(profile["task_skill"]["habit_override"]),
                "rule_shift": float(profile["task_skill"]["rule_shift"]),
                "conflict_planning": float(profile["task_skill"]["conflict_planning"]),
            },
            difficulty_sensitivity=float(profile["difficulty_sensitivity"]),
        )
        for profile in profiles_raw
    ]

    return PilotSweepConfig(
        seed=int(config_dict["seed"]),
        num_samples_per_task=int(config_dict["num_samples_per_task"]),
        model_profiles=profiles,
    )


def load_config(config_path: str | Path) -> PilotSweepConfig:
    config_file = Path(config_path)
    config_data = json.loads(config_file.read_text(encoding="utf-8"))
    return config_from_dict(config_data)
