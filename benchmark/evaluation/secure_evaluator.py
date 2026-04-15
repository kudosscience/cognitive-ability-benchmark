from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

ALLOWED_TASK_NAMES = ("habit_override", "rule_shift", "conflict_planning")
PUBLIC_DATASET_FILE_BY_TASK = {
    "habit_override": "habit_override.jsonl",
    "rule_shift": "rule_shift.jsonl",
    "conflict_planning": "conflict_planning.jsonl",
}

SCHEMA_VERSION = 1
MAX_SUBMISSION_LINE_BYTES = 8192
MAX_PREDICTION_CHARACTERS = 2048
MAX_ACTION_COUNT = 128

HABIT_PATTERN = re.compile(r"^[A-Z](?:\s*>\s*[A-Z])*$")
INTEGER_PATTERN = re.compile(r"^-?\d{1,18}$")
ACTION_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

ZERO_WIDTH_CHARACTERS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
}


@dataclass(frozen=True)
class PrivateAnswerRecord:
    task_name: str
    sample_id: int
    expected_output: str


class SecureEvaluationError(ValueError):
    """Raised when evaluation input, bundles, or submissions are invalid."""


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_unsafe_characters(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    sanitized: list[str] = []
    for character in normalized:
        if character in ZERO_WIDTH_CHARACTERS:
            continue

        category = unicodedata.category(character)
        if category.startswith("C") and character not in {"\t", "\n", "\r"}:
            continue

        sanitized.append(character)

    return "".join(sanitized)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def sanitize_for_judge_input(text: str, max_characters: int = MAX_PREDICTION_CHARACTERS) -> str:
    """Sanitize text before any judge-facing usage to reduce injection risk."""
    stripped = _strip_unsafe_characters(str(text))
    collapsed = _normalize_whitespace(stripped)
    return collapsed[:max_characters]


def canonicalize_habit_override(output: str) -> str | None:
    cleaned = sanitize_for_judge_input(output).upper()
    if not cleaned or not HABIT_PATTERN.fullmatch(cleaned):
        return None

    tokens = [segment.strip() for segment in cleaned.split(">")]
    if any(len(token) != 1 for token in tokens):
        return None

    return " > ".join(tokens)


def canonicalize_rule_shift(output: str) -> str | None:
    cleaned = sanitize_for_judge_input(output)
    if not INTEGER_PATTERN.fullmatch(cleaned):
        return None

    # Canonical integer form blocks formatting tricks (e.g., +0012).
    return str(int(cleaned))


def canonicalize_conflict_planning(output: str) -> str | None:
    cleaned = sanitize_for_judge_input(output).upper()
    if not cleaned:
        return None

    segments = [segment.strip() for segment in cleaned.split(",") if segment.strip()]
    if not segments or len(segments) > MAX_ACTION_COUNT:
        return None

    for segment in segments:
        if not ACTION_PATTERN.fullmatch(segment):
            return None

    if len(set(segments)) != len(segments):
        # Duplicate action IDs are structurally invalid for this benchmark.
        return None

    return ",".join(segments)


def canonicalize_output(task_name: str, output: str) -> str | None:
    if task_name == "habit_override":
        return canonicalize_habit_override(output)
    if task_name == "rule_shift":
        return canonicalize_rule_shift(output)
    if task_name == "conflict_planning":
        return canonicalize_conflict_planning(output)

    raise SecureEvaluationError(f"Unsupported task: {task_name}")


def score_prediction(task_name: str, model_output: str, expected_output: str) -> float:
    canonical_prediction = canonicalize_output(task_name=task_name, output=model_output)
    canonical_expected = canonicalize_output(task_name=task_name, output=expected_output)

    if canonical_expected is None:
        raise SecureEvaluationError(
            f"Private answer for task '{task_name}' is not canonical and cannot be scored safely."
        )

    if canonical_prediction is None:
        return 0.0

    return float(canonical_prediction == canonical_expected)


def sanitize_public_record(task_name: str, sample_id: int, record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {}))

    if task_name == "habit_override":
        safe_metadata = {
            "start_letter": metadata.get("start_letter"),
            "steps": metadata.get("steps"),
            "override_letters": metadata.get("override_letters"),
            "override_delta": metadata.get("override_delta"),
        }
    elif task_name == "rule_shift":
        safe_metadata = {
            "initial_state": metadata.get("initial_state"),
            "shift_after_index": metadata.get("shift_after_index"),
            "operations": metadata.get("operations"),
        }
    elif task_name == "conflict_planning":
        raw_actions = metadata.get("actions", [])
        safe_actions = [
            {
                "action_id": action.get("action_id"),
                "consumes": action.get("consumes"),
                "produces": action.get("produces"),
            }
            for action in raw_actions
        ]
        safe_metadata = {
            "start_inventory": metadata.get("start_inventory"),
            "actions": safe_actions,
        }
    else:
        raise SecureEvaluationError(f"Unsupported task: {task_name}")

    return {
        "sample_id": sample_id,
        "prompt": str(record["prompt"]),
        "difficulty": int(record["difficulty"]),
        "metadata": safe_metadata,
    }


def _build_dataset_manifest(public_data_dir: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for task_name, file_name in PUBLIC_DATASET_FILE_BY_TASK.items():
        dataset_path = public_data_dir / file_name
        if not dataset_path.exists():
            raise SecureEvaluationError(f"Missing public dataset file: {dataset_path}")

        line_count = len(dataset_path.read_text(encoding="utf-8").splitlines())
        manifest[task_name] = {
            "file_name": file_name,
            "sha256": _sha256_file(dataset_path),
            "row_count": line_count,
        }

    return manifest


def _serialize_answers(answers: list[PrivateAnswerRecord]) -> list[dict[str, Any]]:
    return [
        {
            "task_name": answer.task_name,
            "sample_id": answer.sample_id,
            "expected_output": answer.expected_output,
        }
        for answer in answers
    ]


def _build_bundle_payload(
    dataset_manifest: dict[str, dict[str, Any]],
    answers: list[PrivateAnswerRecord],
) -> dict[str, Any]:
    answers_as_dicts = _serialize_answers(answers)
    answers_digest = _sha256_text(json.dumps(answers_as_dicts, sort_keys=True, ensure_ascii=True))

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest": dataset_manifest,
        "answer_count": len(answers_as_dicts),
        "answers_digest": answers_digest,
        "answers": answers_as_dicts,
    }


def _sign_bundle_payload(payload: dict[str, Any], signing_key: str) -> str:
    signing_bytes = _canonical_json_bytes(payload)
    signature = hmac.new(signing_key.encode("utf-8"), signing_bytes, hashlib.sha256).hexdigest()
    return signature


def create_private_answer_bundle(
    public_data_dir: str | Path,
    suite_records: dict[str, list[dict[str, Any]]],
    signing_key: str,
) -> dict[str, Any]:
    public_path = Path(public_data_dir)
    dataset_manifest = _build_dataset_manifest(public_data_dir=public_path)

    answers: list[PrivateAnswerRecord] = []
    for task_name in ALLOWED_TASK_NAMES:
        records = suite_records.get(task_name, [])
        for sample_id, record in enumerate(records, start=1):
            expected_output = str(record.get("expected_output", ""))
            canonical_expected = canonicalize_output(task_name=task_name, output=expected_output)
            if canonical_expected is None:
                raise SecureEvaluationError(
                    f"Cannot create private bundle because sample is non-canonical for task '{task_name}'."
                )

            answers.append(
                PrivateAnswerRecord(
                    task_name=task_name,
                    sample_id=sample_id,
                    expected_output=canonical_expected,
                )
            )

    payload = _build_bundle_payload(dataset_manifest=dataset_manifest, answers=answers)
    payload["signature"] = _sign_bundle_payload(payload=payload, signing_key=signing_key)
    return payload


def load_private_answer_bundle(private_bundle_path: str | Path, signing_key: str) -> dict[str, Any]:
    bundle = json.loads(Path(private_bundle_path).read_text(encoding="utf-8"))

    if int(bundle.get("schema_version", -1)) != SCHEMA_VERSION:
        raise SecureEvaluationError("Unsupported private bundle schema version.")

    provided_signature = str(bundle.get("signature", ""))
    if not provided_signature:
        raise SecureEvaluationError("Private bundle signature is missing.")

    unsigned_payload = dict(bundle)
    unsigned_payload.pop("signature", None)

    expected_signature = _sign_bundle_payload(payload=unsigned_payload, signing_key=signing_key)
    if not hmac.compare_digest(provided_signature, expected_signature):
        raise SecureEvaluationError("Private bundle signature verification failed.")

    serialized_answers = unsigned_payload.get("answers", [])
    expected_digest = str(unsigned_payload.get("answers_digest", ""))
    actual_digest = _sha256_text(json.dumps(serialized_answers, sort_keys=True, ensure_ascii=True))
    if expected_digest != actual_digest:
        raise SecureEvaluationError("Private bundle answer digest mismatch.")

    return bundle


def verify_public_data_integrity(public_data_dir: str | Path, private_bundle: dict[str, Any]) -> None:
    public_path = Path(public_data_dir)
    expected_manifest = private_bundle.get("dataset_manifest", {})

    for task_name, expected_entry in expected_manifest.items():
        file_name = str(expected_entry["file_name"])
        dataset_path = public_path / file_name
        if not dataset_path.exists():
            raise SecureEvaluationError(f"Expected dataset file is missing: {dataset_path}")

        observed_hash = _sha256_file(dataset_path)
        if observed_hash != str(expected_entry["sha256"]):
            raise SecureEvaluationError(f"Dataset integrity check failed for {dataset_path}.")

        observed_rows = len(dataset_path.read_text(encoding="utf-8").splitlines())
        if observed_rows != int(expected_entry["row_count"]):
            raise SecureEvaluationError(f"Dataset row count mismatch for {dataset_path}.")


def _parse_submission_line(line_index: int, raw_line: str) -> dict[str, Any] | None:
    line_bytes = raw_line.encode("utf-8")
    if len(line_bytes) > MAX_SUBMISSION_LINE_BYTES:
        raise SecureEvaluationError(
            f"Submission line {line_index} exceeds {MAX_SUBMISSION_LINE_BYTES} bytes."
        )

    stripped = raw_line.strip()
    if not stripped:
        return None

    try:
        row = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise SecureEvaluationError(
            f"Submission line {line_index} is not valid JSON: {error}"
        ) from error

    if not isinstance(row, dict):
        raise SecureEvaluationError(f"Submission line {line_index} must be a JSON object.")

    return row


def _extract_submission_record(line_index: int, row: dict[str, Any]) -> tuple[tuple[str, int], str]:
    allowed_fields = {"task_name", "sample_id", "prediction"}
    unknown_fields = set(row) - allowed_fields
    if unknown_fields:
        joined = ", ".join(sorted(unknown_fields))
        raise SecureEvaluationError(
            f"Submission line {line_index} contains unsupported fields: {joined}."
        )

    task_name = str(row.get("task_name", "")).strip()
    if task_name not in ALLOWED_TASK_NAMES:
        raise SecureEvaluationError(
            f"Submission line {line_index} has unsupported task_name: {task_name}"
        )

    sample_id = row.get("sample_id")
    if not isinstance(sample_id, int) or sample_id < 1:
        raise SecureEvaluationError(
            f"Submission line {line_index} has invalid sample_id: {sample_id}"
        )

    prediction = row.get("prediction")
    if not isinstance(prediction, str):
        raise SecureEvaluationError(
            f"Submission line {line_index} has non-string prediction."
        )

    if len(prediction) > MAX_PREDICTION_CHARACTERS:
        raise SecureEvaluationError(
            f"Submission line {line_index} prediction exceeds {MAX_PREDICTION_CHARACTERS} characters."
        )

    return (task_name, sample_id), prediction


def parse_submission_jsonl(submission_path: str | Path) -> dict[tuple[str, int], str]:
    parsed: dict[tuple[str, int], str] = {}

    with Path(submission_path).open("r", encoding="utf-8") as file_handle:
        for line_index, raw_line in enumerate(file_handle, start=1):
            row = _parse_submission_line(line_index=line_index, raw_line=raw_line)
            if row is None:
                continue

            lookup_key, prediction = _extract_submission_record(
                line_index=line_index,
                row=row,
            )
            if lookup_key in parsed:
                task_name, sample_id = lookup_key
                raise SecureEvaluationError(
                    f"Duplicate prediction found for task '{task_name}' sample_id {sample_id}."
                )

            parsed[lookup_key] = prediction

    return parsed


def score_submission(
    public_data_dir: str | Path,
    submission_path: str | Path,
    private_bundle_path: str | Path,
    signing_key: str,
) -> dict[str, Any]:
    private_bundle = load_private_answer_bundle(
        private_bundle_path=private_bundle_path,
        signing_key=signing_key,
    )
    verify_public_data_integrity(public_data_dir=public_data_dir, private_bundle=private_bundle)

    predictions = parse_submission_jsonl(submission_path=submission_path)

    expected_rows = private_bundle["answers"]
    expected_lookup: dict[tuple[str, int], str] = {
        (str(row["task_name"]), int(row["sample_id"])): str(row["expected_output"])
        for row in expected_rows
    }

    extra_predictions = [
        lookup_key for lookup_key in predictions if lookup_key not in expected_lookup
    ]
    if extra_predictions:
        raise SecureEvaluationError(
            "Submission includes predictions for unknown sample IDs or tasks."
        )

    task_scores: dict[str, list[float]] = {task_name: [] for task_name in ALLOWED_TASK_NAMES}
    missing_predictions = 0

    for lookup_key, expected_output in expected_lookup.items():
        task_name, _sample_id = lookup_key
        prediction = predictions.get(lookup_key)
        if prediction is None:
            task_scores[task_name].append(0.0)
            missing_predictions += 1
            continue

        score = score_prediction(
            task_name=task_name,
            model_output=prediction,
            expected_output=expected_output,
        )
        task_scores[task_name].append(score)

    summarized_task_scores = {
        task_name: (sum(scores) / len(scores) if scores else 0.0)
        for task_name, scores in task_scores.items()
    }

    total_scores = [score for scores in task_scores.values() for score in scores]
    overall_score = sum(total_scores) / len(total_scores) if total_scores else 0.0

    return {
        "overall_score": overall_score,
        "task_scores": summarized_task_scores,
        "num_scored_samples": len(total_scores),
        "num_missing_predictions": missing_predictions,
    }
