from __future__ import annotations

import pytest

from benchmark.tasks.conflict_planning import ConflictPlanningTask
from benchmark.tasks.habit_override import HabitOverrideTask
from benchmark.tasks.rule_shift import RuleShiftTask
from benchmark.utils.generator import parse_action_sequence


def test_habit_override_task_generates_records() -> None:
    records = HabitOverrideTask.generate_records(num_samples=4, seed=13)

    assert len(records) == 4
    required_keys = {"prompt", "expected_output", "difficulty", "metadata"}
    for record in records:
        assert required_keys.issubset(record.keys())


def test_rule_shift_task_generates_records() -> None:
    records = RuleShiftTask.generate_records(num_samples=4, seed=17)

    assert len(records) == 4
    required_keys = {"prompt", "expected_output", "difficulty", "metadata"}
    for record in records:
        assert required_keys.issubset(record.keys())


def test_conflict_planning_task_generates_records() -> None:
    records = ConflictPlanningTask.generate_records(num_samples=4, seed=19)

    assert len(records) == 4
    required_keys = {"prompt", "expected_output", "difficulty", "metadata"}
    for record in records:
        assert required_keys.issubset(record.keys())


def test_habit_override_exact_match_scoring() -> None:
    sample = HabitOverrideTask.generate_samples(num_samples=1, seed=23)[0]

    assert HabitOverrideTask.score_sample(sample.expected_output, sample) == pytest.approx(1.0)
    assert HabitOverrideTask.score_sample("INCORRECT", sample) == pytest.approx(0.0)


def test_rule_shift_scoring_extracts_integer() -> None:
    sample = RuleShiftTask.generate_samples(num_samples=1, seed=29)[0]

    answer_with_context = f"Final value is {sample.expected_output}."
    wrong_answer = "No integer present"

    assert RuleShiftTask.score_sample(answer_with_context, sample) == pytest.approx(1.0)
    assert RuleShiftTask.score_sample(wrong_answer, sample) == pytest.approx(0.0)


def test_conflict_planning_scoring_normalizes_commas_and_spaces() -> None:
    sample = ConflictPlanningTask.generate_samples(num_samples=1, seed=31)[0]
    expected_sequence = parse_action_sequence(sample.expected_output)
    spaced_sequence = ", ".join(expected_sequence)

    assert ConflictPlanningTask.score_sample(spaced_sequence, sample) == pytest.approx(1.0)


def test_conflict_planning_validate_plan_accepts_canonical_rejects_trap_first() -> None:
    sample = ConflictPlanningTask.generate_samples(num_samples=1, seed=37)[0]
    canonical_plan = parse_action_sequence(sample.expected_output)
    trap_action_id = sample.metadata["trap_action_ids"][0]

    assert ConflictPlanningTask.validate_plan(sample=sample, action_sequence=canonical_plan)
    assert not ConflictPlanningTask.validate_plan(
        sample=sample,
        action_sequence=[trap_action_id, *canonical_plan],
    )
