"""Data utilities: loading, saving, and generating sample datasets."""

from __future__ import annotations

import json
import os
from datetime import datetime

from benchmark.base import BenchmarkResult


def load_task_data(task_name: str, data_dir: str) -> list[dict]:
    """Load task samples from a JSON file in *data_dir*."""
    path = os.path.join(data_dir, f"{task_name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def save_results(results: list[BenchmarkResult], output_path: str) -> None:
    """Serialize *results* to JSON at *output_path*."""
    serialized = []
    for r in results:
        d = r.model_dump()
        d["timestamp"] = d["timestamp"].isoformat() if isinstance(d["timestamp"], datetime) else str(d["timestamp"])
        serialized.append(d)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(serialized, fh, indent=2)


def create_sample_dataset(task_name: str, n_samples: int = 10) -> list[dict]:
    """Generate synthetic sample data for any supported task type."""
    generators = {
        "few_shot_learning": _few_shot_samples,
        "in_context_learning": _in_context_samples,
        "knowledge_transfer": _knowledge_transfer_samples,
        "confidence_calibration": _confidence_calibration_samples,
        "knowledge_boundary": _knowledge_boundary_samples,
        "self_assessment": _self_assessment_samples,
        "selective_attention": _selective_attention_samples,
        "sustained_attention": _sustained_attention_samples,
        "divided_attention": _divided_attention_samples,
        "planning": _planning_samples,
        "cognitive_flexibility": _cognitive_flexibility_samples,
        "working_memory": _working_memory_samples,
        "theory_of_mind": _theory_of_mind_samples,
        "emotion_recognition": _emotion_recognition_samples,
        "perspective_taking": _perspective_taking_samples,
    }
    generator = generators.get(task_name)
    if generator is None:
        return [{"query": f"Sample {i}", "expected": "answer"} for i in range(n_samples)]
    return [generator(i) for i in range(n_samples)]


# ---------------------------------------------------------------------------
# Per-task sample generators
# ---------------------------------------------------------------------------

def _few_shot_samples(i: int) -> dict:
    return {
        "examples": [
            {"input": "apple", "output": "fruit"},
            {"input": "carrot", "output": "vegetable"},
        ],
        "query": "banana",
        "expected": "fruit",
    }


def _in_context_samples(i: int) -> dict:
    return {
        "context": f"The Zorbax protocol (sample {i}) requires three authentication steps.",
        "question": "How many authentication steps does the Zorbax protocol require?",
        "expected": "three",
    }


def _knowledge_transfer_samples(i: int) -> dict:
    return {
        "source_domain": "music",
        "source_example": "A chord is a combination of three or more notes played together.",
        "target_domain": "color theory",
        "target_problem": "What do you call a combination of three or more colors used together in a design?",
        "expected": "color scheme",
    }


def _confidence_calibration_samples(i: int) -> dict:
    return {
        "question": f"What is the capital of France? (sample {i})",
        "expected": "paris",
    }


def _knowledge_boundary_samples(i: int) -> dict:
    if i % 2 == 0:
        return {
            "question": "What is the speed of light in a vacuum?",
            "is_answerable": True,
            "expected": "299,792,458",
        }
    return {
        "question": f"What was the exact weather in London on March 5, 1842, at noon? (sample {i})",
        "is_answerable": False,
        "expected": "",
    }


def _self_assessment_samples(i: int) -> dict:
    return {
        "question": "What is 2 + 2?",
        "model_answer": "4" if i % 2 == 0 else "5",
        "actual_correct": 1.0 if i % 2 == 0 else 0.0,
    }


def _selective_attention_samples(i: int) -> dict:
    key_fact = f"The secret code is ALPHA-{i}"
    distractor = " ".join(["The weather today is sunny."] * 5)
    return {
        "context": f"{distractor} {key_fact}. {distractor}",
        "question": "What is the secret code?",
        "key_facts": [f"ALPHA-{i}"],
        "expected": f"ALPHA-{i}",
    }


def _sustained_attention_samples(i: int) -> dict:
    items = [f"item_{j}" for j in range(10)]
    labels = ["positive" if j % 2 == 0 else "negative" for j in range(10)]
    return {
        "items": items,
        "instruction": "Label each item as positive (even index) or negative (odd index).",
        "expected": labels,
    }


def _divided_attention_samples(i: int) -> dict:
    return {
        "streams": [
            {
                "name": "colors",
                "items": ["red", "blue", "green"],
                "question": "Which color appeared first?",
                "expected": "red",
                "weight": 1.0,
            },
            {
                "name": "numbers",
                "items": ["7", "3", "9"],
                "question": "Which number appeared first?",
                "expected": "7",
                "weight": 1.0,
            },
        ]
    }


def _planning_samples(i: int) -> dict:
    return {
        "goal": f"Make a cup of tea (sample {i})",
        "actions": ["boil water", "add tea bag", "pour water", "steep", "remove bag"],
        "constraints": "Water must be boiled before pouring.",
        "valid_actions": ["boil water", "add tea bag", "pour water", "steep", "remove bag"],
        "optimal_length": 5,
    }


def _cognitive_flexibility_samples(i: int) -> dict:
    return {
        "scenario": f"Sorting task sample {i}",
        "current_rule": "Sort by color",
        "items": ["red circle", "blue square", "red triangle"],
        "switch_trials": [{"expected": "blue"}],
        "stay_trials": [{"expected": "red"}],
    }


def _working_memory_samples(i: int) -> dict:
    sequence = [str(j + i) for j in range(5)]
    return {
        "sequence": sequence,
        "task_type": "recall",
        "expected_sequence": sequence,
    }


def _theory_of_mind_samples(i: int) -> dict:
    return {
        "scenario": (
            "Alice puts her book on the table and leaves the room. "
            "Bob moves the book to the shelf while Alice is away."
        ),
        "character": "Alice",
        "question": "Where does Alice think her book is?",
        "expected": "table",
    }


def _emotion_recognition_samples(i: int) -> dict:
    emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
    emotion = emotions[i % len(emotions)]
    descriptions = {
        "joy": "She laughed and clapped her hands with delight.",
        "sadness": "He wept quietly, unable to find comfort.",
        "anger": "She slammed the door and stormed out of the room.",
        "fear": "He trembled and backed away from the strange noise.",
        "surprise": "Her eyes went wide as she saw the unexpected gift.",
        "disgust": "He turned away, unable to look at the spoiled food.",
        "neutral": "She walked to the store and bought some groceries.",
    }
    return {
        "description": descriptions[emotion],
        "expected": emotion,
    }


def _perspective_taking_samples(i: int) -> dict:
    return {
        "scenario": (
            "A manager asks an employee to work overtime without extra pay. "
            "The employee has family obligations."
        ),
        "characters": ["manager", "employee"],
        "question": "How might each party feel about this situation?",
        "perspectives": [
            {"character": "manager", "expected": "productivity", "complexity": 1.0},
            {"character": "employee", "expected": "family", "complexity": 1.5},
        ],
    }
