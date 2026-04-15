from __future__ import annotations

from benchmark.evaluation.pilot_sweep import (
    PilotSweepConfig,
    default_profiles,
    run_pilot_sweep,
    summarize_pilot_sweep,
)


def test_pilot_sweep_produces_expected_row_count() -> None:
    profiles = default_profiles()
    config = PilotSweepConfig(
        seed=20260415,
        num_samples_per_task=10,
        model_profiles=profiles,
    )

    results = run_pilot_sweep(config)
    expected_rows = len(profiles) * 3 * config.num_samples_per_task

    assert len(results) == expected_rows


def test_pilot_sweep_overall_ranking_shows_gradient() -> None:
    profiles = default_profiles()
    config = PilotSweepConfig(
        seed=20260415,
        num_samples_per_task=120,
        model_profiles=profiles,
    )

    summary = summarize_pilot_sweep(run_pilot_sweep(config))
    overall_rows = sorted(summary["overall_summary"], key=lambda row: row["overall_score"])
    ordered_model_names = [row["model_name"] for row in overall_rows]

    assert ordered_model_names == [
        "pattern-matcher-small",
        "rule-aware-medium",
        "planner-large",
        "reasoning-frontier-xl",
        "oracle-upper-bound",
    ]


def test_pilot_sweep_difficulty_signal_degrades_for_small_model() -> None:
    profiles = default_profiles()
    config = PilotSweepConfig(
        seed=20260415,
        num_samples_per_task=150,
        model_profiles=profiles,
    )

    summary = summarize_pilot_sweep(run_pilot_sweep(config))
    difficulty_rows = summary["difficulty_summary"]

    target_rows = [
        row
        for row in difficulty_rows
        if row["model_name"] == "pattern-matcher-small"
        and row["task_name"] == "rule_shift"
    ]

    by_difficulty = {row["difficulty"]: row["mean_score"] for row in target_rows}
    assert by_difficulty[1] > by_difficulty[5]
