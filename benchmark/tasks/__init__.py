"""Task modules."""

from benchmark.tasks.learning import FewShotLearningTask, InContextLearningTask, KnowledgeTransferTask
from benchmark.tasks.metacognition import ConfidenceCalibrationTask, KnowledgeBoundaryTask, SelfAssessmentTask
from benchmark.tasks.attention import SelectiveAttentionTask, SustainedAttentionTask, DividedAttentionTask
from benchmark.tasks.executive_functions import PlanningTask, CognitiveFlexibilityTask, WorkingMemoryTask
from benchmark.tasks.social_cognition import TheoryOfMindTask, EmotionRecognitionTask, PerspectiveTakingTask

__all__ = [
    "FewShotLearningTask",
    "InContextLearningTask",
    "KnowledgeTransferTask",
    "ConfidenceCalibrationTask",
    "KnowledgeBoundaryTask",
    "SelfAssessmentTask",
    "SelectiveAttentionTask",
    "SustainedAttentionTask",
    "DividedAttentionTask",
    "PlanningTask",
    "CognitiveFlexibilityTask",
    "WorkingMemoryTask",
    "TheoryOfMindTask",
    "EmotionRecognitionTask",
    "PerspectiveTakingTask",
]
