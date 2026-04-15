from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.adapters.kaggle_benchmarks_adapter import (
    KaggleAdapterConfig,
    export_kaggle_assets,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "kaggle_adapter.json"


def _resolve_within_project(user_path: str | Path) -> Path:
    project_root = PROJECT_ROOT.resolve()
    candidate = Path(user_path)

    if not candidate.is_absolute():
        candidate = project_root / candidate

    resolved = candidate.resolve()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"Path must stay inside project root: {project_root}")

    return resolved


def load_config(config_path: Path) -> tuple[KaggleAdapterConfig, str]:
    validated_config_path = _resolve_within_project(config_path)
    raw = json.loads(validated_config_path.read_text(encoding="utf-8"))

    adapter_config = KaggleAdapterConfig(
        benchmark_name=raw["benchmark_name"],
        benchmark_slug=raw["benchmark_slug"],
        track=raw["track"],
        num_samples_per_task=int(raw["num_samples_per_task"]),
        seed=int(raw["seed"]),
    )
    output_dir = str(_resolve_within_project(raw["output_dir"]))
    return adapter_config, output_dir


def main() -> None:
    adapter_config, output_dir = load_config(DEFAULT_CONFIG_PATH)
    result = export_kaggle_assets(config=adapter_config, output_dir=output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
