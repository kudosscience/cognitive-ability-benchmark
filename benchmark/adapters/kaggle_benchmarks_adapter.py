from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from benchmark.cogniflex_suite import CogniFlexSuite

DATA_DIR_NAME = "data"
NOTEBOOKS_DIR_NAME = "notebooks"
METADATA_FILE_NAME = "benchmark_metadata.json"
DEFAULT_SAMPLE_COUNT = 200
DEFAULT_SEED = 20260415
JSON_INDENT = 2


@dataclass(frozen=True)
class KaggleAdapterConfig:
    benchmark_name: str
    benchmark_slug: str
    track: str
    num_samples_per_task: int = DEFAULT_SAMPLE_COUNT
    seed: int = DEFAULT_SEED


def _serialize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for row_index, record in enumerate(records, start=1):
        normalized_record = {
            "sample_id": row_index,
            "prompt": record["prompt"],
            "expected_output": record["expected_output"],
            "difficulty": record["difficulty"],
            "metadata": record["metadata"],
        }
        serialized.append(normalized_record)
    return serialized


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=True))
            file_handle.write("\n")


def _build_metadata(
    config: KaggleAdapterConfig,
    dataset_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "benchmark_name": config.benchmark_name,
        "benchmark_slug": config.benchmark_slug,
        "track": config.track,
        "num_samples_per_task": config.num_samples_per_task,
        "seed": config.seed,
        "tasks": [
            {
                "task_name": task_name,
                "dataset_path": dataset_path,
            }
            for task_name, dataset_path in dataset_paths.items()
        ],
        "kaggle_notes": {
            "task_creation_url": "https://www.kaggle.com/benchmarks/tasks/new",
            "docs_url": "https://www.kaggle.com/docs/benchmarks",
            "sdk_repo": "https://github.com/Kaggle/kaggle-benchmarks",
        },
    }


def generate_kaggle_notebook_script(dataset_paths: dict[str, str]) -> str:
    notebook_template = f'''import os
import re
import pandas as pd
import kaggle_benchmarks as kbench

RENDER_SUBRUNS_ENV_KEY = "RENDER_SUBRUNS"
os.environ[RENDER_SUBRUNS_ENV_KEY] = "False"

DATASET_PATHS = {json.dumps(dataset_paths, indent=2)}

CSV_SPLIT_PATTERN = re.compile(r"\\s*,\\s*")
INTEGER_PATTERN = re.compile(r"-?\\d+")


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _normalize_csv_sequence(text: str) -> str:
    segments = [segment.strip().upper() for segment in CSV_SPLIT_PATTERN.split(str(text)) if segment.strip()]
    return ",".join(segments)


def _extract_integer(text: str) -> str:
    match = INTEGER_PATTERN.search(str(text))
    return match.group(0) if match else ""


def _is_match(task_name: str, prediction: str, expected_output: str) -> bool:
    if task_name == "rule_shift":
        return _extract_integer(prediction) == _extract_integer(expected_output)
    if task_name == "conflict_planning":
        return _normalize_csv_sequence(prediction) == _normalize_csv_sequence(expected_output)
    return _normalize_text(prediction) == _normalize_text(expected_output)


@kbench.task(name="cogniflex_score_row", store_task=False)
def cogniflex_score_row(llm, task_name: str, prompt: str, expected_output: str) -> bool:
    response = llm.prompt(prompt)
    return _is_match(task_name=task_name, prediction=response, expected_output=expected_output)


@kbench.task(name="cogniflex_executive_functions")
def cogniflex_executive_functions(llm) -> tuple[float, float]:
    task_scores = []

    for task_name, dataset_path in DATASET_PATHS.items():
        df = pd.read_json(dataset_path, lines=True)
        df = df[["prompt", "expected_output"]].copy()
        df["task_name"] = task_name

        runs = cogniflex_score_row.evaluate(
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,
            timeout=120,
            remove_run_files=True,
        )

        task_accuracy = float(runs.as_dataframe()["result"].mean())
        task_scores.append(task_accuracy)

    mean_score = float(sum(task_scores) / len(task_scores))
    centered_squares = [(score - mean_score) ** 2 for score in task_scores]
    variance = float(sum(centered_squares) / len(centered_squares))
    std_score = float(variance ** 0.5)

    return mean_score, std_score


run = cogniflex_executive_functions.run(kbench.llm)
run

# In the final Kaggle notebook cell keep only this task/run pair.
# %choose cogniflex_executive_functions
'''
    return notebook_template


def export_kaggle_assets(
    config: KaggleAdapterConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    data_path = output_path / DATA_DIR_NAME
    notebooks_path = output_path / NOTEBOOKS_DIR_NAME

    data_path.mkdir(parents=True, exist_ok=True)
    notebooks_path.mkdir(parents=True, exist_ok=True)

    suite_records = CogniFlexSuite.generate_records(
        num_samples_per_task=config.num_samples_per_task,
        seed=config.seed,
    )

    dataset_paths: dict[str, str] = {}
    for task_name, task_records in suite_records.items():
        output_file_name = f"{task_name}.jsonl"
        absolute_dataset_path = data_path / output_file_name
        serialized_records = _serialize_records(records=task_records)
        _write_jsonl(path=absolute_dataset_path, rows=serialized_records)
        dataset_paths[task_name] = f"./{DATA_DIR_NAME}/{output_file_name}"

    notebook_script = generate_kaggle_notebook_script(dataset_paths=dataset_paths)
    notebook_script_path = notebooks_path / "cogniflex_task.py"
    notebook_script_path.write_text(notebook_script, encoding="utf-8")

    metadata = _build_metadata(config=config, dataset_paths=dataset_paths)
    metadata_path = output_path / METADATA_FILE_NAME
    metadata_path.write_text(
        json.dumps(metadata, indent=JSON_INDENT, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "config": asdict(config),
        "output_dir": str(output_path),
        "dataset_paths": dataset_paths,
        "notebook_script": str(notebook_script_path),
        "metadata_path": str(metadata_path),
    }
