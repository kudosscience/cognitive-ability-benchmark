"""Cognitive Ability Benchmark — base classes and result models."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkTask(ABC):
    """Abstract base class for all benchmark tasks."""

    name: str
    category: str
    description: str

    def __init__(self, name: str, category: str, description: str) -> None:
        self.name = name
        self.category = category
        self.description = description

    @abstractmethod
    def evaluate(self, model_output: str, ground_truth: dict) -> dict:
        """Compute detailed evaluation metrics."""

    @abstractmethod
    def score(self, model_output: str, ground_truth: dict) -> float:
        """Return a single float score in [0, 1]."""

    @abstractmethod
    def generate_prompt(self, sample: dict) -> str:
        """Build the prompt string for a single sample."""


class BenchmarkResult(BaseModel):
    """Pydantic model capturing the result of running a benchmark task."""

    model_config = {"arbitrary_types_allowed": True}

    task_name: str
    category: str
    score: float = Field(ge=0.0, le=1.0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    num_samples: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
