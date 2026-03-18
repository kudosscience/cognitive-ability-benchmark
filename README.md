# Cognitive Ability Benchmark

A comprehensive benchmark for evaluating cognitive abilities in AI models, developed as a Kaggle Hackathon entry.

## Overview

This benchmark systematically measures five core cognitive ability categories across 15 distinct tasks. Each task takes a model's text output and scores it against a ground truth, producing a normalized score in \[0, 1\].

---

## Cognitive Ability Categories

| Category | Tasks | Description |
|---|---|---|
| **Learning** | Few-Shot, In-Context, Knowledge Transfer | Ability to learn from examples and transfer knowledge |
| **Metacognition** | Confidence Calibration, Knowledge Boundary, Self-Assessment | Awareness of one's own knowledge and confidence |
| **Attention** | Selective, Sustained, Divided | Focusing, maintaining, and splitting attention |
| **Executive Functions** | Planning, Cognitive Flexibility, Working Memory | Goal-directed action, rule-switching, information maintenance |
| **Social Cognition** | Theory of Mind, Emotion Recognition, Perspective Taking | Understanding others' mental states and emotions |

---

## Installation

### From source

```bash
git clone https://github.com/kudosscience/cognitive-ability-benchmark
cd cognitive-ability-benchmark
pip install -e ".[dev]"
```

### Requirements

```
numpy, pandas, scipy, scikit-learn, pydantic>=2.0, pyyaml
```

---

## Quick Start

```python
from benchmark.tasks import FewShotLearningTask, TheoryOfMindTask
from benchmark.evaluation.metrics import compute_composite_score, generate_report
from benchmark.utils.data_utils import create_sample_dataset
from benchmark.base import BenchmarkResult

# --- Few-shot learning ---
task = FewShotLearningTask()
sample = create_sample_dataset("few_shot_learning", n_samples=1)[0]

prompt = task.generate_prompt(sample)
print(prompt)

# Simulate a model response
model_output = "fruit"
score = task.score(model_output, sample)
details = task.evaluate(model_output, sample)
print(f"Score: {score:.2f}")
print(f"Details: {details}")

# --- Collect results and generate a full report ---
results = [
    BenchmarkResult(
        task_name=task.name,
        category=task.category,
        score=score,
        metrics=details,
        num_samples=1,
    )
]

report = generate_report(results)
print(report)
```

---

## Benchmark Structure

```
cognitive-ability-benchmark/
├── benchmark/
│   ├── base.py                  # BenchmarkTask ABC + BenchmarkResult model
│   ├── tasks/
│   │   ├── learning.py          # FewShot, InContext, KnowledgeTransfer
│   │   ├── metacognition.py     # ConfidenceCalibration, KnowledgeBoundary, SelfAssessment
│   │   ├── attention.py         # Selective, Sustained, Divided
│   │   ├── executive_functions.py  # Planning, CognitiveFlexibility, WorkingMemory
│   │   └── social_cognition.py  # TheoryOfMind, EmotionRecognition, PerspectiveTaking
│   ├── evaluation/
│   │   └── metrics.py           # ECE, composite/category scores, report generation
│   └── utils/
│       └── data_utils.py        # load/save data, sample dataset generation
└── tests/
    └── test_tasks.py            # 87 pytest tests
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `composite_score` | Weighted average across all 5 categories (0.2 each) |
| `learning_score` | Average of the 3 learning task scores |
| `metacognition_score` | Average of the 3 metacognition task scores |
| `attention_score` | Average of the 3 attention task scores |
| `executive_functions_score` | Average of the 3 executive function task scores |
| `social_cognition_score` | Average of the 3 social cognition task scores |

### Expected Calibration Error (ECE)

ECE measures how well a model's stated confidence matches its actual accuracy. The `ConfidenceCalibrationTask` scores models using `score = max(0, 1 - ECE)` so a higher score means better calibration.

### Report

`generate_report(results)` returns:
- `composite_score` — overall benchmark score
- `category_scores` — per-category averages
- `task_scores` — per-task scores
- `strengths` — categories scoring ≥ 0.7
- `weaknesses` — categories scoring < 0.5

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Kaggle Hackathon Context

This benchmark was built as an entry for the Kaggle Hackathon on measuring cognitive abilities in AI. The benchmark metadata is described in [`benchmark.yaml`](benchmark.yaml).

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-task`)
3. Add your task class inheriting from `BenchmarkTask`
4. Add corresponding tests in `tests/test_tasks.py`
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.
