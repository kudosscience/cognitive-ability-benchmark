"""Evaluation metrics for the Cognitive Ability Benchmark."""

from __future__ import annotations

from benchmark.base import BenchmarkResult

CATEGORY_WEIGHTS: dict[str, float] = {
    "learning": 0.2,
    "metacognition": 0.2,
    "attention": 0.2,
    "executive_functions": 0.2,
    "social_cognition": 0.2,
}


def compute_ece(confidences: list[float], accuracies: list[float], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Args:
        confidences: Model confidence values in [0, 1].
        accuracies:  Binary correctness flags (1.0 / 0.0) per sample.
        n_bins:      Number of equal-width bins.

    Returns:
        ECE as a float in [0, 1].
    """
    if not confidences:
        return 0.0
    n = len(confidences)
    bin_size = 1.0 / n_bins
    ece = 0.0
    for b in range(n_bins):
        lower = b * bin_size
        upper = lower + bin_size
        indices = [
            i for i, c in enumerate(confidences) if lower <= c < upper
        ]
        if not indices:
            continue
        bin_acc = sum(accuracies[i] for i in indices) / len(indices)
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        ece += (len(indices) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_category_score(
    task_results: list[BenchmarkResult], category: str
) -> float:
    """Return the average score for a given category."""
    relevant = [r for r in task_results if r.category == category]
    if not relevant:
        return 0.0
    return sum(r.score for r in relevant) / len(relevant)


def compute_composite_score(task_results: list[BenchmarkResult]) -> float:
    """Compute a weighted composite score across all categories."""
    if not task_results:
        return 0.0
    composite = 0.0
    for category, weight in CATEGORY_WEIGHTS.items():
        cat_score = compute_category_score(task_results, category)
        composite += weight * cat_score
    return float(composite)


def generate_report(task_results: list[BenchmarkResult]) -> dict:
    """Generate a full evaluation report.

    Returns a dict with composite_score, category_scores, task_scores,
    strengths (categories scoring >= 0.7), and weaknesses (< 0.5).
    """
    composite = compute_composite_score(task_results)
    category_scores = {
        cat: compute_category_score(task_results, cat)
        for cat in CATEGORY_WEIGHTS
    }
    task_scores = {r.task_name: r.score for r in task_results}
    strengths = [cat for cat, score in category_scores.items() if score >= 0.7]
    weaknesses = [cat for cat, score in category_scores.items() if score < 0.5]
    return {
        "composite_score": composite,
        "category_scores": category_scores,
        "task_scores": task_scores,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }
