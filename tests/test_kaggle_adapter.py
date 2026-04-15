from __future__ import annotations

import json
from pathlib import Path

from benchmark.adapters.kaggle_benchmarks_adapter import (
    KaggleAdapterConfig,
    export_kaggle_assets,
)


def _count_jsonl_rows(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_export_kaggle_assets_writes_expected_files(tmp_path: Path) -> None:
    config = KaggleAdapterConfig(
        benchmark_name="Test CogniFlex",
        benchmark_slug="test-cogniflex",
        track="Executive Functions",
        num_samples_per_task=7,
        seed=101,
    )

    result = export_kaggle_assets(config=config, output_dir=tmp_path)

    metadata_path = Path(result["metadata_path"])
    notebook_path = Path(result["notebook_script"])

    assert metadata_path.exists()
    assert notebook_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task_names = [task["task_name"] for task in metadata["tasks"]]
    assert task_names == ["habit_override", "rule_shift", "conflict_planning"]

    for task in metadata["tasks"]:
        dataset_path = tmp_path / task["dataset_path"].replace("./", "")
        assert dataset_path.exists()
        assert _count_jsonl_rows(dataset_path) == config.num_samples_per_task

        first_row = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
        assert "expected_output" not in first_row
        assert "sample_id" in first_row

        if task["task_name"] == "habit_override":
            assert "visited_letters" not in first_row["metadata"]

        if task["task_name"] == "rule_shift":
            assert "trace" not in first_row["metadata"]

        if task["task_name"] == "conflict_planning":
            assert "canonical_plan" not in first_row["metadata"]
            assert "trap_action_ids" not in first_row["metadata"]


def test_export_kaggle_notebook_contains_choose_hint(tmp_path: Path) -> None:
    config = KaggleAdapterConfig(
        benchmark_name="Test CogniFlex",
        benchmark_slug="test-cogniflex",
        track="Executive Functions",
        num_samples_per_task=3,
        seed=99,
    )

    result = export_kaggle_assets(config=config, output_dir=tmp_path)
    notebook = Path(result["notebook_script"]).read_text(encoding="utf-8")

    assert "@kbench.task(name=\"cogniflex_executive_functions\")" in notebook
    assert "COGNIFLEX_PRIVATE_BUNDLE_PATH" in notebook
    assert "%choose cogniflex_executive_functions" in notebook
