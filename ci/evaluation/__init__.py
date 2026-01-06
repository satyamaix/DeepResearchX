"""
DRX Evaluation Pipeline Package.

This package provides comprehensive evaluation capabilities for the DRX
Deep Research system, including:

- DeepEval integration for agent trajectory and task completion metrics
- Ragas integration for faithfulness, context precision, answer relevancy
- TAU-bench style scenario testing
- CI/CD gate integration with configurable thresholds
- Dataset Flywheel for continuous learning (WP-3A)

WP11: Evaluation Pipeline Implementation
WP-3A: Dataset Flywheel Implementation

Usage:
    # Run all evaluation tests
    pytest ci/evaluation/ -m eval

    # Run only CI gate tests
    pytest ci/evaluation/ -m ci_gate

    # Run with coverage
    pytest ci/evaluation/ --cov=src --cov-report=html

    # Use Dataset Flywheel
    from ci.evaluation import DatasetCollector, FeedbackStore
"""

from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__package_name__ = "drx_evaluation"

# Paths
EVALUATION_DIR = Path(__file__).parent
SCENARIOS_DIR = EVALUATION_DIR / "scenarios"
RESULTS_PATH = EVALUATION_DIR / "eval_results.json"
TRAINING_DATA_DIR = EVALUATION_DIR.parent.parent / "data" / "training"

# Evaluation thresholds (CI gate configuration)
THRESHOLDS = {
    # Hard gates (must pass)
    "faithfulness": 0.8,
    "task_completion": 0.7,
    "hallucination_max": 0.2,
    "policy_violations": 0,
    # Soft gates (warnings)
    "answer_relevancy": 0.7,
    "context_precision": 0.6,
    "context_recall": 0.6,
}

# Dataset Flywheel imports (WP-3A)
from ci.evaluation.dataset_collector import (
    DatasetCollector,
    DatasetStatistics,
    QualityTier,
    SessionRecord,
    classify_quality_tier,
    create_dataset_collector,
)
from ci.evaluation.feedback_store import (
    AggregateMetrics,
    FeedbackRecord,
    FeedbackStore,
    create_feedback_store,
)

__all__ = [
    # Paths and configuration
    "EVALUATION_DIR",
    "SCENARIOS_DIR",
    "RESULTS_PATH",
    "TRAINING_DATA_DIR",
    "THRESHOLDS",
    # Dataset Collector (WP-3A)
    "DatasetCollector",
    "DatasetStatistics",
    "QualityTier",
    "SessionRecord",
    "classify_quality_tier",
    "create_dataset_collector",
    # Feedback Store (WP-3A)
    "AggregateMetrics",
    "FeedbackRecord",
    "FeedbackStore",
    "create_feedback_store",
]
