from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.evaluation.pilot_sweep import load_config, run_pilot_sweep, summarize_pilot_sweep

PREDICTIONS_FILE_NAME = "pilot_sweep_predictions.jsonl"
TASK_SUMMARY_FILE_NAME = "pilot_sweep_task_summary.csv"
DIFFICULTY_SUMMARY_FILE_NAME = "pilot_sweep_difficulty_summary.csv"
OVERALL_SUMMARY_FILE_NAME = "pilot_sweep_overall.csv"
MARKDOWN_REPORT_FILE_NAME = "pilot_sweep_report.md"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pilot_sweep.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _resolve_within_project(user_path: str | Path) -> Path:
    project_root = PROJECT_ROOT.resolve()
    candidate = Path(user_path)

    if not candidate.is_absolute():
        candidate = project_root / candidate

    resolved = candidate.resolve()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"Path must stay inside project root: {project_root}")

    return resolved


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=True))
            file_handle.write("\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(summary: dict[str, list[dict]]) -> str:
    overall_rows = summary["overall_summary"]
    ranked_rows = sorted(overall_rows, key=lambda row: row["overall_score"], reverse=True)

    lines = [
        "# Pilot Sweep Report",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Model | Overall Score | Samples |",
        "| --- | --- | ---: | ---: |",
    ]

    for rank, row in enumerate(ranked_rows, start=1):
        lines.append(
            f"| {rank} | {row['model_name']} | {row['overall_score']:.3f} | {row['num_samples']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The score spread across models indicates useful discriminatory power.",
            "- Difficulty-level degradation can be inspected in pilot_sweep_difficulty_summary.csv.",
            "- Use these outputs as the baseline section of the Kaggle writeup.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    output_dir = _resolve_within_project(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = _resolve_within_project(DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    predictions = run_pilot_sweep(config)
    summary = summarize_pilot_sweep(predictions)

    _write_jsonl(output_dir / PREDICTIONS_FILE_NAME, predictions)
    _write_csv(output_dir / TASK_SUMMARY_FILE_NAME, summary["task_summary"])
    _write_csv(output_dir / DIFFICULTY_SUMMARY_FILE_NAME, summary["difficulty_summary"])
    _write_csv(output_dir / OVERALL_SUMMARY_FILE_NAME, summary["overall_summary"])

    report = _render_report(summary)
    (output_dir / MARKDOWN_REPORT_FILE_NAME).write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "predictions": str(output_dir / PREDICTIONS_FILE_NAME),
                "task_summary": str(output_dir / TASK_SUMMARY_FILE_NAME),
                "difficulty_summary": str(output_dir / DIFFICULTY_SUMMARY_FILE_NAME),
                "overall_summary": str(output_dir / OVERALL_SUMMARY_FILE_NAME),
                "report": str(output_dir / MARKDOWN_REPORT_FILE_NAME),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
