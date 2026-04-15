# CogniFlex Executive Functions Benchmark

CogniFlex is a high-rigor benchmark suite designed for the Kaggle "Measuring Progress Toward AGI - Cognitive Abilities" hackathon. The suite focuses on the Executive Functions track and isolates three cognitive capabilities:

1. Inhibitory control
2. Cognitive flexibility
3. Complex planning

## Implemented Task Set

### 1) Habit Override (`habit_override`)

Purpose: test whether a model can suppress habitual pattern continuation when a local rule inversion appears.

Output check: exact path match (normalized whitespace).

### 2) Rule Shift (`rule_shift`)

Purpose: test whether a model can continue multi-step computation after operation semantics shift midway.

Output check: integer extraction and exact integer match.

### 3) Conflict Planning (`conflict_planning`)

Purpose: test whether a model can find the shortest valid action plan while ignoring tempting but destructive trap actions.

Output check: exact action sequence match with CSV normalization.

## Design Principles

- Programmatic generation only (no ambiguous labels)
- Difficulty levels 1-5 cycled across datasets
- Deterministic generation via explicit random seeds
- Exact-match grading aligned with defensible benchmark scoring

## Project Structure

```text
benchmark/
  cogniflex_suite.py
  tasks/
    base_task.py
    habit_override.py
    rule_shift.py
    conflict_planning.py
  utils/
    generator.py
tests/
  test_generators.py
  test_tasks.py
```

## Quick Start

```python
from benchmark.cogniflex_suite import CogniFlexSuite

seed = 20260415
samples_per_task = 3

datasets = CogniFlexSuite.generate_samples(
    num_samples_per_task=samples_per_task,
    seed=seed,
)

habit_samples = datasets["habit_override"]
first_sample = habit_samples[0]
print(first_sample.prompt)
print(first_sample.expected_output)
```

## Scoring Example

```python
from benchmark.tasks.rule_shift import RuleShiftTask

sample = RuleShiftTask.generate_samples(num_samples=1, seed=17)[0]
model_output = f"Final value is {sample.expected_output}."
score = RuleShiftTask.score_sample(model_output, sample)
print(score)  # 1.0
```

## Running Tests

```bash
python -m pytest tests/
```

## Current Status

Implemented in this phase:

- Generator framework for all three tasks
- Task classes with deterministic exact-match scoring
- Unit tests for determinism, correctness, and unique shortest-plan behavior

Next implementation steps:

- Kaggle benchmark adapter and run configuration
- Pilot model sweeps for performance gradient tuning
- Final writeup aligned to the Kaggle submission template
