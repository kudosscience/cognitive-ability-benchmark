from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.adapters.kaggle_benchmarks_adapter import KaggleAdapterConfig, export_kaggle_assets
from benchmark.cogniflex_suite import CogniFlexSuite
from benchmark.evaluation.secure_evaluator import (
    SecureEvaluationError,
    create_private_answer_bundle,
    parse_submission_jsonl,
    sanitize_for_judge_input,
    score_prediction,
    score_submission,
)

TEST_SIGNING_KEY = "unit-test-signing-key"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=True))
            file_handle.write("\n")


def _build_bundle(tmp_path: Path, num_samples_per_task: int = 5) -> tuple[Path, Path, dict[str, list[dict]]]:
    config = KaggleAdapterConfig(
        benchmark_name="Secure CogniFlex",
        benchmark_slug="secure-cogniflex",
        track="Executive Functions",
        num_samples_per_task=num_samples_per_task,
        seed=20260415,
    )

    result = export_kaggle_assets(config=config, output_dir=tmp_path)
    public_data_dir = tmp_path / "data"

    suite_records = CogniFlexSuite.generate_records(
        num_samples_per_task=config.num_samples_per_task,
        seed=config.seed,
    )
    bundle = create_private_answer_bundle(
        public_data_dir=public_data_dir,
        suite_records=suite_records,
        signing_key=TEST_SIGNING_KEY,
    )

    private_bundle_path = tmp_path / "private_answer_key.json"
    private_bundle_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    assert Path(result["metadata_path"]).exists()
    return public_data_dir, private_bundle_path, suite_records


def test_submission_scoring_uses_private_bundle_and_reaches_perfect_score(tmp_path: Path) -> None:
    public_data_dir, private_bundle_path, suite_records = _build_bundle(tmp_path=tmp_path)

    submission_rows: list[dict] = []
    for task_name, records in suite_records.items():
        for sample_id, record in enumerate(records, start=1):
            submission_rows.append(
                {
                    "task_name": task_name,
                    "sample_id": sample_id,
                    "prediction": record["expected_output"],
                }
            )

    submission_path = tmp_path / "submission_predictions.jsonl"
    _write_jsonl(path=submission_path, rows=submission_rows)

    result = score_submission(
        public_data_dir=public_data_dir,
        submission_path=submission_path,
        private_bundle_path=private_bundle_path,
        signing_key=TEST_SIGNING_KEY,
    )

    assert result["overall_score"] == pytest.approx(1.0)
    assert result["num_missing_predictions"] == 0


def test_submission_scoring_rejects_tampered_public_data(tmp_path: Path) -> None:
    public_data_dir, private_bundle_path, suite_records = _build_bundle(tmp_path=tmp_path)

    submission_rows: list[dict] = []
    for task_name, records in suite_records.items():
        for sample_id, record in enumerate(records, start=1):
            submission_rows.append(
                {
                    "task_name": task_name,
                    "sample_id": sample_id,
                    "prediction": record["expected_output"],
                }
            )

    submission_path = tmp_path / "submission_predictions.jsonl"
    _write_jsonl(path=submission_path, rows=submission_rows)

    tampered_path = public_data_dir / "habit_override.jsonl"
    tampered_path.write_text(
        tampered_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SecureEvaluationError):
        score_submission(
            public_data_dir=public_data_dir,
            submission_path=submission_path,
            private_bundle_path=private_bundle_path,
            signing_key=TEST_SIGNING_KEY,
        )


def test_submission_parser_rejects_duplicate_prediction_keys(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission_predictions.jsonl"
    _write_jsonl(
        path=submission_path,
        rows=[
            {"task_name": "rule_shift", "sample_id": 1, "prediction": "1"},
            {"task_name": "rule_shift", "sample_id": 1, "prediction": "2"},
        ],
    )

    with pytest.raises(SecureEvaluationError):
        parse_submission_jsonl(submission_path=submission_path)


def test_sanitized_inputs_remove_hidden_characters() -> None:
    raw = "A\u200b >\u200d B\n\t"
    sanitized = sanitize_for_judge_input(raw)
    assert sanitized == "A > B"


def test_scoring_resists_output_manipulation_patterns() -> None:
    assert score_prediction("rule_shift", "Final value is 12", "12") == pytest.approx(0.0)
    assert score_prediction("rule_shift", "12", "12") == pytest.approx(1.0)
    assert score_prediction("conflict_planning", "C1,C1,GOAL", "C1,C2,GOAL") == pytest.approx(0.0)
