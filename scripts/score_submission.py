from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.evaluation.secure_evaluator import score_submission  # noqa: E402

DEFAULT_PUBLIC_DATA_DIR = PROJECT_ROOT / "kaggle_assets" / "data"
DEFAULT_PRIVATE_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "private" / "private_answer_key.json"
DEFAULT_SUBMISSION_PATH = PROJECT_ROOT / "outputs" / "submission_predictions.jsonl"
SIGNING_KEY_ENV = "COGNIFLEX_SIGNING_KEY"


def _resolve_within_project(user_path: str | Path) -> Path:
    project_root = PROJECT_ROOT.resolve()
    candidate = Path(user_path)

    if not candidate.is_absolute():
        candidate = project_root / candidate

    resolved = candidate.resolve()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"Path must stay inside project root: {project_root}")

    return resolved


def main() -> None:
    signing_key = os.environ.get(SIGNING_KEY_ENV, "")
    if not signing_key:
        raise RuntimeError(
            f"{SIGNING_KEY_ENV} is required for signed private-bundle verification."
        )

    public_data_dir = _resolve_within_project(DEFAULT_PUBLIC_DATA_DIR)
    private_bundle_path = _resolve_within_project(DEFAULT_PRIVATE_BUNDLE_PATH)
    submission_path = _resolve_within_project(DEFAULT_SUBMISSION_PATH)

    result = score_submission(
        public_data_dir=public_data_dir,
        submission_path=submission_path,
        private_bundle_path=private_bundle_path,
        signing_key=signing_key,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
