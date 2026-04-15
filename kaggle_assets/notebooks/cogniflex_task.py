import os
import re
import pandas as pd
import kaggle_benchmarks as kbench

RENDER_SUBRUNS_ENV_KEY = "RENDER_SUBRUNS"
os.environ[RENDER_SUBRUNS_ENV_KEY] = "False"

DATASET_PATHS = {
  "habit_override": "./data/habit_override.jsonl",
  "rule_shift": "./data/rule_shift.jsonl",
  "conflict_planning": "./data/conflict_planning.jsonl"
}

CSV_SPLIT_PATTERN = re.compile(r"\s*,\s*")
INTEGER_PATTERN = re.compile(r"-?\d+")


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
