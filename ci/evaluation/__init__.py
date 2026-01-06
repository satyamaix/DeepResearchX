"""
DRX Evaluation Pipeline Package.

This package provides comprehensive evaluation capabilities for the DRX
Deep Research system, including:

- DeepEval integration for agent trajectory and task completion metrics
- Ragas integration for faithfulness, context precision, answer relevancy
- TAU-bench style scenario testing
- CI/CD gate integration with configurable thresholds
- Dataset Flywheel for continuous learning (WP-3A)
- Live API evaluation runner (WP-0B)
- DeepEval/Ragas OpenRouter configuration (WP-0C)
- Report generator for markdown and JSON output (WP-1A)

WP11: Evaluation Pipeline Implementation
WP-3A: Dataset Flywheel Implementation
WP-0B: Live Evaluation Runner
WP-0C: DeepEval/Ragas OpenRouter Configuration
WP-1A: Report Generator

Usage:
    # Run all evaluation tests
    pytest ci/evaluation/ -m eval

    # Run only CI gate tests
    pytest ci/evaluation/ -m ci_gate

    # Run with coverage
    pytest ci/evaluation/ --cov=src --cov-report=html

    # Use Dataset Flywheel
    from ci.evaluation import DatasetCollector, FeedbackStore

    # Run live API evaluations
    python ci/evaluation/run_evaluation.py --scenarios scenarios/research_tasks.yaml

    # Configure DeepEval with OpenRouter
    from ci.evaluation import configure_deepeval_for_openrouter, create_all_metrics
    configure_deepeval_for_openrouter()
    metrics = create_all_metrics()

    # Generate evaluation reports
    from ci.evaluation import ReportGenerator
    generator = ReportGenerator()
    generator.load_results(Path("eval_results.json"))
    report = generator.generate_report()
    print(generator.to_markdown(report))
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

# Live Evaluation Runner (WP-0B)
from ci.evaluation.run_evaluation import (
    EvaluationResult,
    EvaluationRunner,
    load_scenarios,
    save_results,
)

# DeepEval/Ragas OpenRouter Configuration (WP-0C)
from ci.evaluation.deepeval_config import (
    DEEPEVAL_AVAILABLE,
    RAGAS_AVAILABLE,
    BatchEvaluationResult,
    EvaluationResult as DeepEvalScenario,
    MetricResult,
    configure_deepeval_for_openrouter,
    create_all_metrics,
    create_answer_relevancy_metric,
    create_faithfulness_metric,
    create_hallucination_metric,
    create_ragas_dataset,
    create_test_case,
    evaluate_scenarios,
    get_eval_model,
    get_openrouter_judge,
    run_full_evaluation,
    run_ragas_evaluation,
    run_ragas_evaluation_async,
)

# Report Generator (WP-1A)
from ci.evaluation.report_generator import (
    EvaluationReport,
    GateResult,
    ReportGenerator,
    ScenarioReport,
    ThresholdConfig,
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
    # Evaluation Runner (WP-0B)
    "EvaluationResult",
    "EvaluationRunner",
    "load_scenarios",
    "save_results",
    # DeepEval/Ragas Configuration (WP-0C)
    "DEEPEVAL_AVAILABLE",
    "RAGAS_AVAILABLE",
    "BatchEvaluationResult",
    "DeepEvalScenario",
    "MetricResult",
    "configure_deepeval_for_openrouter",
    "create_all_metrics",
    "create_answer_relevancy_metric",
    "create_faithfulness_metric",
    "create_hallucination_metric",
    "create_ragas_dataset",
    "create_test_case",
    "evaluate_scenarios",
    "get_eval_model",
    "get_openrouter_judge",
    "run_full_evaluation",
    "run_ragas_evaluation",
    "run_ragas_evaluation_async",
    # Report Generator (WP-1A)
    "EvaluationReport",
    "GateResult",
    "ReportGenerator",
    "ScenarioReport",
    "ThresholdConfig",
]
