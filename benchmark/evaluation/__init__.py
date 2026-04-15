"""Evaluation utilities for CogniFlex benchmark analysis."""

from benchmark.evaluation.pilot_sweep import (
    ModelProfile,
    PilotSweepConfig,
    run_pilot_sweep,
    summarize_pilot_sweep,
)
from benchmark.evaluation.secure_evaluator import (
    SecureEvaluationError,
    canonicalize_output,
    create_private_answer_bundle,
    parse_submission_jsonl,
    sanitize_for_judge_input,
    score_prediction,
    score_submission,
    verify_public_data_integrity,
)

__all__ = [
    "ModelProfile",
    "PilotSweepConfig",
    "run_pilot_sweep",
    "summarize_pilot_sweep",
    "SecureEvaluationError",
    "sanitize_for_judge_input",
    "canonicalize_output",
    "score_prediction",
    "create_private_answer_bundle",
    "parse_submission_jsonl",
    "verify_public_data_integrity",
    "score_submission",
]
