"""Comprehensive tests for the Cognitive Ability Benchmark."""

import json
import os
import tempfile

import pytest

from benchmark.base import BenchmarkResult
from benchmark.tasks import (
    FewShotLearningTask,
    InContextLearningTask,
    KnowledgeTransferTask,
    ConfidenceCalibrationTask,
    KnowledgeBoundaryTask,
    SelfAssessmentTask,
    SelectiveAttentionTask,
    SustainedAttentionTask,
    DividedAttentionTask,
    PlanningTask,
    CognitiveFlexibilityTask,
    WorkingMemoryTask,
    TheoryOfMindTask,
    EmotionRecognitionTask,
    PerspectiveTakingTask,
)
from benchmark.evaluation.metrics import (
    compute_ece,
    compute_composite_score,
    compute_category_score,
    generate_report,
)
from benchmark.utils.data_utils import create_sample_dataset, save_results, load_task_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_result():
    return BenchmarkResult(
        task_name="few_shot_learning",
        category="learning",
        score=0.8,
        metrics={"accuracy": 0.8},
        num_samples=10,
    )


@pytest.fixture
def all_results():
    categories = {
        "learning": ["few_shot_learning", "in_context_learning", "knowledge_transfer"],
        "metacognition": ["confidence_calibration", "knowledge_boundary", "self_assessment"],
        "attention": ["selective_attention", "sustained_attention", "divided_attention"],
        "executive_functions": ["planning", "cognitive_flexibility", "working_memory"],
        "social_cognition": ["theory_of_mind", "emotion_recognition", "perspective_taking"],
    }
    results = []
    for cat, tasks in categories.items():
        for task in tasks:
            results.append(
                BenchmarkResult(
                    task_name=task,
                    category=cat,
                    score=0.75,
                    metrics={},
                    num_samples=5,
                )
            )
    return results


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "TaskClass",
    [
        FewShotLearningTask,
        InContextLearningTask,
        KnowledgeTransferTask,
        ConfidenceCalibrationTask,
        KnowledgeBoundaryTask,
        SelfAssessmentTask,
        SelectiveAttentionTask,
        SustainedAttentionTask,
        DividedAttentionTask,
        PlanningTask,
        CognitiveFlexibilityTask,
        WorkingMemoryTask,
        TheoryOfMindTask,
        EmotionRecognitionTask,
        PerspectiveTakingTask,
    ],
)
def test_task_instantiation(TaskClass):
    task = TaskClass()
    assert task.name
    assert task.category
    assert task.description


# ---------------------------------------------------------------------------
# generate_prompt tests
# ---------------------------------------------------------------------------

def test_few_shot_generate_prompt():
    task = FewShotLearningTask()
    sample = create_sample_dataset("few_shot_learning", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_in_context_generate_prompt():
    task = InContextLearningTask()
    sample = create_sample_dataset("in_context_learning", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and "Context" in prompt


def test_knowledge_transfer_generate_prompt():
    task = KnowledgeTransferTask()
    sample = create_sample_dataset("knowledge_transfer", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_confidence_calibration_generate_prompt():
    task = ConfidenceCalibrationTask()
    sample = create_sample_dataset("confidence_calibration", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and "Confidence" in prompt


def test_knowledge_boundary_generate_prompt():
    task = KnowledgeBoundaryTask()
    sample = create_sample_dataset("knowledge_boundary", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_self_assessment_generate_prompt():
    task = SelfAssessmentTask()
    sample = create_sample_dataset("self_assessment", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_selective_attention_generate_prompt():
    task = SelectiveAttentionTask()
    sample = create_sample_dataset("selective_attention", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_sustained_attention_generate_prompt():
    task = SustainedAttentionTask()
    sample = create_sample_dataset("sustained_attention", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_divided_attention_generate_prompt():
    task = DividedAttentionTask()
    sample = create_sample_dataset("divided_attention", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_planning_generate_prompt():
    task = PlanningTask()
    sample = create_sample_dataset("planning", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and "Goal" in prompt


def test_cognitive_flexibility_generate_prompt():
    task = CognitiveFlexibilityTask()
    sample = create_sample_dataset("cognitive_flexibility", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_working_memory_generate_prompt():
    task = WorkingMemoryTask()
    sample = create_sample_dataset("working_memory", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_theory_of_mind_generate_prompt():
    task = TheoryOfMindTask()
    sample = create_sample_dataset("theory_of_mind", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_emotion_recognition_generate_prompt():
    task = EmotionRecognitionTask()
    sample = create_sample_dataset("emotion_recognition", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


def test_perspective_taking_generate_prompt():
    task = PerspectiveTakingTask()
    sample = create_sample_dataset("perspective_taking", 1)[0]
    prompt = task.generate_prompt(sample)
    assert isinstance(prompt, str) and len(prompt) > 0


# ---------------------------------------------------------------------------
# score tests — must return float in [0, 1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task_name,TaskClass,model_output",
    [
        ("few_shot_learning", FewShotLearningTask, "fruit"),
        ("in_context_learning", InContextLearningTask, "three"),
        ("knowledge_transfer", KnowledgeTransferTask, "color scheme"),
        ("confidence_calibration", ConfidenceCalibrationTask, "Answer: paris\nConfidence: 80"),
        ("knowledge_boundary", KnowledgeBoundaryTask, "I don't know"),
        ("self_assessment", SelfAssessmentTask, "Yes, the answer is correct."),
        ("selective_attention", SelectiveAttentionTask, "ALPHA-0"),
        (
            "sustained_attention",
            SustainedAttentionTask,
            "1. positive\n2. negative\n3. positive\n4. negative\n5. positive\n6. negative\n7. positive\n8. negative\n9. positive\n10. negative",
        ),
        ("divided_attention", DividedAttentionTask, "The first color is red and the first number is 7."),
        (
            "planning",
            PlanningTask,
            "1. boil water\n2. add tea bag\n3. pour water\n4. steep\n5. remove bag",
        ),
        ("cognitive_flexibility", CognitiveFlexibilityTask, "blue\nred"),
        ("working_memory", WorkingMemoryTask, "0 1 2 3 4"),
        ("theory_of_mind", TheoryOfMindTask, "Alice thinks the book is on the table."),
        ("emotion_recognition", EmotionRecognitionTask, "joy"),
        ("perspective_taking", PerspectiveTakingTask, "The manager wants productivity, the employee has family obligations."),
    ],
)
def test_score_returns_valid_float(task_name, TaskClass, model_output):
    task = TaskClass()
    sample = create_sample_dataset(task_name, 1)[0]
    score = task.score(model_output, sample)
    assert isinstance(score, float), f"score should be float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"score {score} out of [0, 1] for {task_name}"


# ---------------------------------------------------------------------------
# evaluate tests — must return a dict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task_name,TaskClass,model_output",
    [
        ("few_shot_learning", FewShotLearningTask, "fruit"),
        ("in_context_learning", InContextLearningTask, "three"),
        ("knowledge_transfer", KnowledgeTransferTask, "color scheme"),
        ("confidence_calibration", ConfidenceCalibrationTask, "Answer: paris\nConfidence: 80"),
        ("knowledge_boundary", KnowledgeBoundaryTask, "I don't know"),
        ("self_assessment", SelfAssessmentTask, "Yes, the answer is correct."),
        ("selective_attention", SelectiveAttentionTask, "ALPHA-0"),
        (
            "sustained_attention",
            SustainedAttentionTask,
            "1. positive\n2. negative",
        ),
        ("divided_attention", DividedAttentionTask, "red and 7"),
        (
            "planning",
            PlanningTask,
            "1. boil water\n2. pour water",
        ),
        ("cognitive_flexibility", CognitiveFlexibilityTask, "blue red"),
        ("working_memory", WorkingMemoryTask, "0 1 2 3 4"),
        ("theory_of_mind", TheoryOfMindTask, "on the table"),
        ("emotion_recognition", EmotionRecognitionTask, "joy"),
        ("perspective_taking", PerspectiveTakingTask, "productivity and family"),
    ],
)
def test_evaluate_returns_dict(task_name, TaskClass, model_output):
    task = TaskClass()
    sample = create_sample_dataset(task_name, 1)[0]
    result = task.evaluate(model_output, sample)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

def test_compute_ece_perfect_calibration():
    confidences = [0.9] * 10
    accuracies = [1.0] * 9 + [0.0]
    ece = compute_ece(confidences, accuracies)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_compute_ece_empty():
    assert compute_ece([], []) == 0.0


def test_compute_ece_range():
    import random
    random.seed(42)
    confidences = [random.random() for _ in range(100)]
    accuracies = [float(random.random() > 0.5) for _ in range(100)]
    ece = compute_ece(confidences, accuracies)
    assert 0.0 <= ece <= 1.0


def test_compute_composite_score(all_results):
    score = compute_composite_score(all_results)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert abs(score - 0.75) < 1e-6


def test_compute_composite_score_empty():
    assert compute_composite_score([]) == 0.0


def test_compute_category_score(all_results):
    score = compute_category_score(all_results, "learning")
    assert abs(score - 0.75) < 1e-6


def test_compute_category_score_missing():
    assert compute_category_score([], "learning") == 0.0


def test_generate_report(all_results):
    report = generate_report(all_results)
    assert "composite_score" in report
    assert "category_scores" in report
    assert "task_scores" in report
    assert "strengths" in report
    assert "weaknesses" in report
    assert 0.0 <= report["composite_score"] <= 1.0
    for cat_score in report["category_scores"].values():
        assert 0.0 <= cat_score <= 1.0


# ---------------------------------------------------------------------------
# Data utils tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task_name",
    [
        "few_shot_learning",
        "in_context_learning",
        "knowledge_transfer",
        "confidence_calibration",
        "knowledge_boundary",
        "self_assessment",
        "selective_attention",
        "sustained_attention",
        "divided_attention",
        "planning",
        "cognitive_flexibility",
        "working_memory",
        "theory_of_mind",
        "emotion_recognition",
        "perspective_taking",
    ],
)
def test_create_sample_dataset(task_name):
    samples = create_sample_dataset(task_name, n_samples=5)
    assert isinstance(samples, list)
    assert len(samples) == 5
    for s in samples:
        assert isinstance(s, dict)


def test_save_and_load_results(all_results):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "results.json")
        save_results(all_results, output_path)
        assert os.path.exists(output_path)
        with open(output_path) as fh:
            loaded = json.load(fh)
        assert len(loaded) == len(all_results)
        for item in loaded:
            assert "task_name" in item
            assert "score" in item


def test_load_task_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [{"question": "Q1", "expected": "A1"}]
        path = os.path.join(tmpdir, "test_task.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        loaded = load_task_data("test_task", tmpdir)
        assert loaded == data


# ---------------------------------------------------------------------------
# BenchmarkResult model tests
# ---------------------------------------------------------------------------

def test_benchmark_result_valid(sample_result):
    assert 0.0 <= sample_result.score <= 1.0
    assert sample_result.task_name == "few_shot_learning"


def test_benchmark_result_score_bounds():
    with pytest.raises(Exception):
        BenchmarkResult(
            task_name="test",
            category="learning",
            score=1.5,
            metrics={},
            num_samples=1,
        )
