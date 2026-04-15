from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.adapters.kaggle_benchmarks_adapter import (  # noqa: E402
    KaggleAdapterConfig,
    export_private_answer_bundle,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "kaggle_adapter.json"
DEFAULT_PUBLIC_OUTPUT_DIR = PROJECT_ROOT / "kaggle_assets"
DEFAULT_PRIVATE_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "private" / "private_answer_key.json"
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


def load_config(config_path: Path) -> KaggleAdapterConfig:
    validated_config_path = _resolve_within_project(config_path)
    raw = json.loads(validated_config_path.read_text(encoding="utf-8"))

    return KaggleAdapterConfig(
        benchmark_name=raw["benchmark_name"],
        benchmark_slug=raw["benchmark_slug"],
        track=raw["track"],
        num_samples_per_task=int(raw["num_samples_per_task"]),
        seed=int(raw["seed"]),
    )


def main() -> None:
    signing_key = os.environ.get(SIGNING_KEY_ENV, "")
    if not signing_key:
        raise RuntimeError(
            f"{SIGNING_KEY_ENV} is required to export a tamper-proof private answer bundle."
        )

    config = load_config(DEFAULT_CONFIG_PATH)
    public_output_dir = _resolve_within_project(DEFAULT_PUBLIC_OUTPUT_DIR)
    private_bundle_path = _resolve_within_project(DEFAULT_PRIVATE_BUNDLE_PATH)

    result = export_private_answer_bundle(
        config=config,
        public_output_dir=public_output_dir,
        private_bundle_path=private_bundle_path,
        signing_key=signing_key,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
