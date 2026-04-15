import os
import re
import json
import unicodedata
from pathlib import Path
import pandas as pd
import kaggle_benchmarks as kbench

RENDER_SUBRUNS_ENV_KEY = "RENDER_SUBRUNS"
os.environ[RENDER_SUBRUNS_ENV_KEY] = "False"

PRIVATE_BUNDLE_ENV_KEY = "COGNIFLEX_PRIVATE_BUNDLE_PATH"
DEFAULT_PRIVATE_BUNDLE_PATH = "./private/private_answer_key.json"
KAGGLE_INPUT_PRIVATE_BUNDLE_PATH = "/kaggle/input/cogniflex-private/private_answer_key.json"

DATASET_PATHS = {
  "habit_override": "./data/habit_override.jsonl",
  "rule_shift": "./data/rule_shift.jsonl",
  "conflict_planning": "./data/conflict_planning.jsonl"
}

CSV_SPLIT_PATTERN = re.compile(r"\s*,\s*")
INTEGER_PATTERN = re.compile(r"^-?\d{1,18}$")
HABIT_PATTERN = re.compile(r"^[A-Z](?:\s*>\s*[A-Z])*$")
ACTION_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
MAX_ACTION_COUNT = 128
MAX_OUTPUT_CHARS = 2048

ZERO_WIDTH_CHARACTERS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
}


def _sanitize_for_judge_input(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    sanitized = []
    for character in normalized:
        if character in ZERO_WIDTH_CHARACTERS:
            continue
        category = unicodedata.category(character)
        if category.startswith("C") and character not in {"\t", "\n", "\r"}:
            continue
        sanitized.append(character)

    collapsed = " ".join("".join(sanitized).strip().split())
    return collapsed[:MAX_OUTPUT_CHARS]


def _canonicalize_habit_override(output: str) -> str | None:
    cleaned = _sanitize_for_judge_input(output).upper()
    if not HABIT_PATTERN.fullmatch(cleaned):
        return None
    tokens = [segment.strip() for segment in cleaned.split(">")]
    if any(len(token) != 1 for token in tokens):
        return None
    return " > ".join(tokens)


def _canonicalize_rule_shift(output: str) -> str | None:
    cleaned = _sanitize_for_judge_input(output)
    if not INTEGER_PATTERN.fullmatch(cleaned):
        return None
    return str(int(cleaned))


def _canonicalize_conflict_planning(output: str) -> str | None:
    cleaned = _sanitize_for_judge_input(output).upper()
    if not cleaned:
        return None

    tokens = [segment.strip() for segment in cleaned.split(",") if segment.strip()]
    if not tokens or len(tokens) > MAX_ACTION_COUNT:
        return None
    if len(set(tokens)) != len(tokens):
        return None
    if any(not ACTION_PATTERN.fullmatch(token) for token in tokens):
        return None
    return ",".join(tokens)


def _canonicalize_output(task_name: str, output: str) -> str | None:
    if task_name == "habit_override":
        return _canonicalize_habit_override(output)
    if task_name == "rule_shift":
        return _canonicalize_rule_shift(output)
    if task_name == "conflict_planning":
        return _canonicalize_conflict_planning(output)
    return None


def _resolve_private_bundle_path() -> Path:
    location_key = os.environ.get(PRIVATE_BUNDLE_ENV_KEY, "default").strip().lower()
    if location_key in {"", "default"}:
        return (Path.cwd() / Path(DEFAULT_PRIVATE_BUNDLE_PATH)).resolve()
    if location_key == "kaggle_input":
        return Path(KAGGLE_INPUT_PRIVATE_BUNDLE_PATH).resolve()

    raise RuntimeError(
        "Unsupported private bundle location key. Use 'default' or 'kaggle_input'."
    )


def _load_private_answers() -> dict[tuple[str, int], str]:
    private_path = _resolve_private_bundle_path()
    if not private_path.exists():
        raise RuntimeError(
            "Private answer bundle is required for official scoring and is intentionally not shipped in public assets."
        )

    with private_path.open("r", encoding="utf-8") as private_file:
        bundle = json.load(private_file)
    lookup: dict[tuple[str, int], str] = {}
    for row in bundle.get("answers", []):
        task_name = str(row["task_name"])
        sample_id = int(row["sample_id"])
        expected_output = str(row["expected_output"])
        canonical = _canonicalize_output(task_name=task_name, output=expected_output)
        if canonical is None:
            raise RuntimeError("Private answer bundle contains non-canonical answers.")
        lookup[(task_name, sample_id)] = canonical

    return lookup


PRIVATE_ANSWER_LOOKUP = _load_private_answers()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _normalize_csv_sequence(text: str) -> str:
    segments = [segment.strip().upper() for segment in CSV_SPLIT_PATTERN.split(str(text)) if segment.strip()]
    return ",".join(segments)


def _extract_integer(text: str) -> str:
    match = INTEGER_PATTERN.search(str(text))
    return match.group(0) if match else ""


def _is_match(task_name: str, prediction: str, expected_output: str) -> bool:
    canonical_prediction = _canonicalize_output(task_name=task_name, output=prediction)
    canonical_expected = _canonicalize_output(task_name=task_name, output=expected_output)
    if canonical_expected is None or canonical_prediction is None:
        return False
    return canonical_prediction == canonical_expected


@kbench.task(name="cogniflex_score_row", store_task=False)
def cogniflex_score_row(llm, task_name: str, sample_id: int, prompt: str) -> bool:
    response = llm.prompt(prompt)
    expected_output = PRIVATE_ANSWER_LOOKUP.get((task_name, int(sample_id)))
    if expected_output is None:
        return False
    return _is_match(task_name=task_name, prediction=response, expected_output=expected_output)


@kbench.task(name="cogniflex_executive_functions")
def cogniflex_executive_functions(llm) -> tuple[float, float]:
    task_scores = []

    for task_name, dataset_path in DATASET_PATHS.items():
        df = pd.read_json(dataset_path, lines=True)
        df = df[["sample_id", "prompt"]].copy()
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


benchmark_run = cogniflex_executive_functions.run(kbench.llm)

# In the final Kaggle notebook cell keep only this task/run pair.
# %choose cogniflex_executive_functions
