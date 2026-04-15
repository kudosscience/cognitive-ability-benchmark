from __future__ import annotations

import random

from benchmark.utils.generator import (
    GOAL_RESOURCE,
    generate_conflict_planning_sample,
    generate_habit_override_dataset,
    generate_rule_shift_dataset,
    parse_action_sequence,
    simulate_conflict_plan,
)


def _collect_valid_plans(sample, max_depth: int) -> list[tuple[str, ...]]:
    actions_by_id = {action["action_id"]: action for action in sample.metadata["actions"]}
    ordered_action_ids = sorted(actions_by_id)
    start_inventory = dict(sample.metadata["start_inventory"])
    plans: list[tuple[str, ...]] = []

    def dfs(inventory: dict[str, int], used: set[str], plan: list[str]) -> None:
        if inventory.get(GOAL_RESOURCE, 0) > 0:
            plans.append(tuple(plan))
            return

        if len(plan) >= max_depth:
            return

        for action_id in ordered_action_ids:
            if action_id in used:
                continue

            action = actions_by_id[action_id]
            consumed_resource = action["consumes"]
            if inventory.get(consumed_resource, 0) < 1:
                continue

            next_inventory = dict(inventory)
            next_inventory[consumed_resource] -= 1
            if next_inventory[consumed_resource] == 0:
                del next_inventory[consumed_resource]

            produced_resource = action["produces"]
            next_inventory[produced_resource] = next_inventory.get(produced_resource, 0) + 1

            dfs(next_inventory, used | {action_id}, plan + [action_id])

    dfs(start_inventory, set(), [])
    return plans


def test_habit_override_dataset_is_deterministic() -> None:
    first_dataset = generate_habit_override_dataset(num_samples=12, seed=99)
    second_dataset = generate_habit_override_dataset(num_samples=12, seed=99)

    first_expected = [sample.expected_output for sample in first_dataset]
    second_expected = [sample.expected_output for sample in second_dataset]

    assert first_expected == second_expected


def test_habit_override_difficulty_cycles_1_to_5() -> None:
    dataset = generate_habit_override_dataset(num_samples=12, seed=7)
    observed = [sample.difficulty for sample in dataset]
    expected_cycle = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]

    assert observed == expected_cycle


def test_habit_override_path_matches_metadata() -> None:
    sample = generate_habit_override_dataset(num_samples=1, seed=101)[0]
    visited_path = sample.expected_output.split(" > ")

    assert visited_path[0] == sample.metadata["start_letter"]
    assert len(visited_path) == sample.metadata["steps"] + 1


def test_rule_shift_dataset_is_deterministic() -> None:
    first_dataset = generate_rule_shift_dataset(num_samples=10, seed=314)
    second_dataset = generate_rule_shift_dataset(num_samples=10, seed=314)

    first_expected = [sample.expected_output for sample in first_dataset]
    second_expected = [sample.expected_output for sample in second_dataset]

    assert first_expected == second_expected


def test_rule_shift_expected_matches_trace_tail() -> None:
    sample = generate_rule_shift_dataset(num_samples=1, seed=2718)[0]
    final_state = sample.metadata["trace"][-1]

    assert sample.expected_output == str(final_state)


def test_conflict_planning_expected_plan_is_valid() -> None:
    sample = generate_conflict_planning_sample(difficulty=3, rng=random.Random(33))

    assert simulate_conflict_plan(sample=sample, action_sequence=sample.expected_output)


def test_conflict_planning_trap_first_breaks_plan() -> None:
    sample = generate_conflict_planning_sample(difficulty=3, rng=random.Random(44))
    trap_action_id = sample.metadata["trap_action_ids"][0]
    canonical_plan = parse_action_sequence(sample.expected_output)
    trap_first_plan = [trap_action_id, *canonical_plan]

    assert not simulate_conflict_plan(sample=sample, action_sequence=trap_first_plan)


def test_conflict_planning_has_unique_shortest_solution() -> None:
    sample = generate_conflict_planning_sample(difficulty=5, rng=random.Random(55))
    expected_plan = tuple(parse_action_sequence(sample.expected_output))
    max_depth = len(expected_plan)

    plans = _collect_valid_plans(sample=sample, max_depth=max_depth)
    assert plans, "No valid plan was found for generated sample."

    shortest_length = min(len(plan) for plan in plans)
    shortest_plans = [plan for plan in plans if len(plan) == shortest_length]

    assert shortest_length == len(expected_plan)
    assert shortest_plans == [expected_plan]
