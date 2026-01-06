"""DeepEval and Ragas Configuration with OpenRouter Integration.

This module configures DeepEval and Ragas to use OpenRouter as the LLM judge,
enabling evaluation of research agent outputs using various metrics.

WP-0C: DeepEval/Ragas OpenRouter Configuration

Key Features:
- OpenRouter LLM judge configuration for DeepEval metrics
- Metric factory functions for faithfulness, hallucination, and answer relevancy
- Batch evaluation support for multiple scenarios
- Ragas integration for RAG-specific metrics
- Retry logic with exponential backoff for rate limits
- Graceful handling of missing dependencies
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, TypedDict

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================


class MetricResult(TypedDict):
    """Result from a single metric evaluation."""

    metric_name: str
    score: float
    threshold: float
    passed: bool
    reason: str | None


class EvaluationResult(TypedDict):
    """Result structure for evaluation scenarios."""

    scenario_id: str
    input_query: str
    actual_output: str
    retrieval_context: list[str]
    expected_output: str | None
    ground_truth: str | None
    metadata: dict[str, Any]


class BatchEvaluationResult(TypedDict):
    """Result from batch evaluation."""

    scenario_id: str
    metrics: list[MetricResult]
    passed: bool
    error: str | None


# =============================================================================
# Environment Configuration
# =============================================================================


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment.

    Returns:
        OpenRouter API key

    Raises:
        ValueError: If API key is not set
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please set it to use OpenRouter as the LLM judge."
        )
    return api_key


def get_openrouter_base_url() -> str:
    """Get OpenRouter base URL from environment.

    Returns:
        OpenRouter API base URL
    """
    return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def get_eval_model() -> str:
    """Get the evaluation model from environment.

    Returns:
        Model identifier to use for evaluation
    """
    return os.environ.get("EVAL_MODEL", "openai/gpt-4o-mini")


# =============================================================================
# Dependency Availability Checks
# =============================================================================


def _check_deepeval_available() -> bool:
    """Check if DeepEval is available."""
    try:
        import deepeval  # noqa: F401

        return True
    except ImportError:
        return False


def _check_ragas_available() -> bool:
    """Check if Ragas is available."""
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


DEEPEVAL_AVAILABLE = _check_deepeval_available()
RAGAS_AVAILABLE = _check_ragas_available()


# =============================================================================
# OpenRouter LLM Judge Configuration
# =============================================================================


def get_openrouter_judge(model: str = "openai/gpt-4o-mini") -> Any:
    """Configure an LLM judge using OpenRouter API.

    DeepEval expects a model that can be used for evaluation.
    OpenRouter provides access to many models via a unified API.

    This function configures DeepEval to use OpenRouter by setting
    the appropriate environment variables for OpenAI-compatible endpoints.

    Args:
        model: Model identifier to use (default: openai/gpt-4o-mini)

    Returns:
        Model string for use with DeepEval metrics

    Note:
        DeepEval uses environment variables for configuration:
        - OPENAI_API_KEY: Set to OpenRouter API key
        - OPENAI_BASE_URL: Set to OpenRouter base URL

        This function sets these environment variables to redirect
        DeepEval's OpenAI calls to OpenRouter.
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is not installed. Install with: pip install deepeval"
        )

    # Get OpenRouter configuration
    api_key = get_openrouter_api_key()
    base_url = get_openrouter_base_url()

    # Configure environment for DeepEval to use OpenRouter
    # DeepEval uses OpenAI SDK which respects these environment variables
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url

    # Use the specified model or environment override
    eval_model = os.environ.get("EVAL_MODEL", model)

    logger.info(
        f"Configured OpenRouter judge with model: {eval_model}",
        extra={"model": eval_model, "base_url": base_url},
    )

    return eval_model


def configure_deepeval_for_openrouter(model: str | None = None) -> str:
    """Configure DeepEval to use OpenRouter as the LLM provider.

    This is a convenience function that sets up the environment
    and returns the configured model string.

    Args:
        model: Optional model override

    Returns:
        Configured model string for use with metrics
    """
    eval_model = model or get_eval_model()
    return get_openrouter_judge(eval_model)


# =============================================================================
# Metric Factory Functions
# =============================================================================


def create_faithfulness_metric(threshold: float = 0.8) -> Any:
    """Create a faithfulness metric with OpenRouter judge.

    Faithfulness measures whether the generated output is grounded
    in the provided retrieval context without introducing external
    information or hallucinations.

    Args:
        threshold: Minimum score required to pass (default: 0.8)

    Returns:
        Configured FaithfulnessMetric instance

    Raises:
        ImportError: If DeepEval is not installed
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is not installed. Install with: pip install deepeval"
        )

    from deepeval.metrics import FaithfulnessMetric

    # Configure OpenRouter
    model = configure_deepeval_for_openrouter()

    return FaithfulnessMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
    )


def create_hallucination_metric(threshold: float = 0.2) -> Any:
    """Create a hallucination metric with OpenRouter judge.

    Hallucination score measures the proportion of claims in the output
    that are not supported by the retrieval context. Lower scores are better.

    Args:
        threshold: Maximum score allowed to pass (default: 0.2)
                   Note: For hallucination, scores above threshold fail.

    Returns:
        Configured HallucinationMetric instance

    Raises:
        ImportError: If DeepEval is not installed
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is not installed. Install with: pip install deepeval"
        )

    from deepeval.metrics import HallucinationMetric

    # Configure OpenRouter
    model = configure_deepeval_for_openrouter()

    return HallucinationMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
    )


def create_answer_relevancy_metric(threshold: float = 0.7) -> Any:
    """Create an answer relevancy metric with OpenRouter judge.

    Answer relevancy measures how well the generated output addresses
    the user's original query.

    Args:
        threshold: Minimum score required to pass (default: 0.7)

    Returns:
        Configured AnswerRelevancyMetric instance

    Raises:
        ImportError: If DeepEval is not installed
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is not installed. Install with: pip install deepeval"
        )

    from deepeval.metrics import AnswerRelevancyMetric

    # Configure OpenRouter
    model = configure_deepeval_for_openrouter()

    return AnswerRelevancyMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
    )


def create_all_metrics(thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Create all evaluation metrics with configured thresholds.

    Args:
        thresholds: Optional dictionary of metric thresholds.
                    Keys: faithfulness, hallucination, answer_relevancy

    Returns:
        Dictionary mapping metric names to configured metric instances
    """
    default_thresholds = {
        "faithfulness": 0.8,
        "hallucination": 0.2,
        "answer_relevancy": 0.7,
    }

    if thresholds:
        default_thresholds.update(thresholds)

    return {
        "faithfulness": create_faithfulness_metric(default_thresholds["faithfulness"]),
        "hallucination": create_hallucination_metric(default_thresholds["hallucination"]),
        "answer_relevancy": create_answer_relevancy_metric(
            default_thresholds["answer_relevancy"]
        ),
    }


# =============================================================================
# Test Case Creation
# =============================================================================


def create_test_case(
    input_query: str,
    actual_output: str,
    retrieval_context: list[str],
    expected_output: str | None = None,
) -> Any:
    """Create a DeepEval LLMTestCase from components.

    Args:
        input_query: The user's research query
        actual_output: The generated response
        retrieval_context: Retrieved documents/context
        expected_output: Optional expected/ground truth output

    Returns:
        Configured LLMTestCase for evaluation

    Raises:
        ImportError: If DeepEval is not installed
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError(
            "DeepEval is not installed. Install with: pip install deepeval"
        )

    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
    )


# =============================================================================
# Batch Evaluation
# =============================================================================


async def evaluate_scenarios(
    results: list[EvaluationResult],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Run DeepEval and Ragas metrics on collected results.

    This function evaluates multiple research scenarios using configured
    metrics with retry logic for handling rate limits.

    Args:
        results: List of evaluation results containing scenario data
        thresholds: Dictionary of metric thresholds

    Returns:
        Dictionary with metric scores for each scenario:
        {
            "scenarios": [BatchEvaluationResult, ...],
            "summary": {
                "total": int,
                "passed": int,
                "failed": int,
                "pass_rate": float,
                "avg_scores": dict[str, float],
            }
        }
    """
    if not DEEPEVAL_AVAILABLE:
        logger.warning("DeepEval not available, skipping evaluation")
        return {
            "scenarios": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_scores": {},
                "error": "DeepEval not installed",
            },
        }

    if not results:
        return {
            "scenarios": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_scores": {},
            },
        }

    # Configure OpenRouter
    configure_deepeval_for_openrouter()

    # Create metrics
    metrics = create_all_metrics(thresholds)

    scenario_results: list[BatchEvaluationResult] = []
    score_accumulators: dict[str, list[float]] = {
        "faithfulness": [],
        "hallucination": [],
        "answer_relevancy": [],
    }

    for result in results:
        scenario_metrics: list[MetricResult] = []
        scenario_passed = True
        scenario_error: str | None = None

        try:
            # Create test case
            test_case = create_test_case(
                input_query=result["input_query"],
                actual_output=result["actual_output"],
                retrieval_context=result["retrieval_context"],
                expected_output=result.get("expected_output"),
            )

            # Evaluate each metric with retry logic
            for metric_name, metric in metrics.items():
                metric_result = await _evaluate_metric_with_retry(
                    metric=metric,
                    test_case=test_case,
                    metric_name=metric_name,
                    threshold=thresholds.get(metric_name, 0.5),
                )

                scenario_metrics.append(metric_result)
                score_accumulators[metric_name].append(metric_result["score"])

                # Check pass condition
                if metric_name == "hallucination":
                    # For hallucination, lower is better
                    if metric_result["score"] > metric_result["threshold"]:
                        scenario_passed = False
                else:
                    # For other metrics, higher is better
                    if metric_result["score"] < metric_result["threshold"]:
                        scenario_passed = False

        except Exception as e:
            logger.error(
                f"Error evaluating scenario {result.get('scenario_id', 'unknown')}: {e}",
                exc_info=True,
            )
            scenario_error = str(e)
            scenario_passed = False

        scenario_results.append(
            BatchEvaluationResult(
                scenario_id=result.get("scenario_id", "unknown"),
                metrics=scenario_metrics,
                passed=scenario_passed,
                error=scenario_error,
            )
        )

    # Calculate summary statistics
    total = len(scenario_results)
    passed = sum(1 for r in scenario_results if r["passed"])
    failed = total - passed
    pass_rate = passed / total if total > 0 else 0.0

    avg_scores: dict[str, float] = {}
    for metric_name, scores in score_accumulators.items():
        if scores:
            avg_scores[metric_name] = sum(scores) / len(scores)

    return {
        "scenarios": scenario_results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
            "avg_scores": avg_scores,
        },
    }


async def _evaluate_metric_with_retry(
    metric: Any,
    test_case: Any,
    metric_name: str,
    threshold: float,
    max_retries: int = 3,
) -> MetricResult:
    """Evaluate a single metric with retry logic for rate limits.

    Args:
        metric: DeepEval metric instance
        test_case: LLMTestCase to evaluate
        metric_name: Name of the metric
        threshold: Threshold for pass/fail
        max_retries: Maximum retry attempts

    Returns:
        MetricResult with score and pass/fail status
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    ):
        with attempt:
            # Run metric evaluation in thread pool (DeepEval metrics are sync)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, metric.measure, test_case)

            # Determine pass/fail based on metric type
            if metric_name == "hallucination":
                passed = metric.score <= threshold
            else:
                passed = metric.score >= threshold

            return MetricResult(
                metric_name=metric_name,
                score=metric.score,
                threshold=threshold,
                passed=passed,
                reason=getattr(metric, "reason", None),
            )

    # Should not reach here due to reraise=True
    return MetricResult(
        metric_name=metric_name,
        score=0.0,
        threshold=threshold,
        passed=False,
        reason="Evaluation failed after retries",
    )


# =============================================================================
# Ragas Integration
# =============================================================================


def create_ragas_dataset(results: list[EvaluationResult]) -> Any:
    """Convert evaluation results to Ragas Dataset format.

    Args:
        results: List of evaluation results

    Returns:
        HuggingFace Dataset compatible with Ragas evaluation

    Raises:
        ImportError: If Ragas or datasets is not installed
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("Ragas is not installed. Install with: pip install ragas")

    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets package is not installed. Install with: pip install datasets"
        )

    # Configure OpenRouter for Ragas
    # Ragas also uses OpenAI SDK internally
    configure_deepeval_for_openrouter()

    # Convert to Ragas format
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for result in results:
        data["question"].append(result["input_query"])
        data["answer"].append(result["actual_output"])
        data["contexts"].append(result["retrieval_context"])
        data["ground_truth"].append(result.get("ground_truth") or result.get("expected_output") or "")

    return Dataset.from_dict(data)


def run_ragas_evaluation(dataset: Any) -> dict[str, float]:
    """Run Ragas metrics (context_precision, context_recall, faithfulness).

    Args:
        dataset: Ragas-compatible dataset

    Returns:
        Dictionary with metric scores:
        {
            "context_precision": float,
            "context_recall": float,
            "faithfulness": float,
            "answer_relevancy": float,
        }

    Raises:
        ImportError: If Ragas is not installed
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("Ragas is not installed. Install with: pip install ragas")

    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Configure OpenRouter for Ragas
    configure_deepeval_for_openrouter()

    # Run evaluation
    result = ragas_evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    return {
        "context_precision": result.get("context_precision", 0.0),
        "context_recall": result.get("context_recall", 0.0),
        "faithfulness": result.get("faithfulness", 0.0),
        "answer_relevancy": result.get("answer_relevancy", 0.0),
    }


async def run_ragas_evaluation_async(
    results: list[EvaluationResult],
) -> dict[str, float]:
    """Run Ragas evaluation asynchronously.

    Wrapper that runs the synchronous Ragas evaluation in a thread pool.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with Ragas metric scores
    """
    if not results:
        return {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
        }

    # Create dataset
    dataset = create_ragas_dataset(results)

    # Run evaluation in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_ragas_evaluation, dataset)


# =============================================================================
# Combined Evaluation
# =============================================================================


async def run_full_evaluation(
    results: list[EvaluationResult],
    thresholds: dict[str, float] | None = None,
    include_ragas: bool = True,
) -> dict[str, Any]:
    """Run complete evaluation pipeline with DeepEval and Ragas.

    This function runs both DeepEval metrics (faithfulness, hallucination,
    answer relevancy) and Ragas metrics (context precision, context recall)
    on the provided evaluation results.

    Args:
        results: List of evaluation results
        thresholds: Optional custom thresholds
        include_ragas: Whether to include Ragas metrics (default: True)

    Returns:
        Combined evaluation results:
        {
            "deepeval": { ... },
            "ragas": { ... },
            "combined_score": float,
            "passed": bool,
        }
    """
    default_thresholds = {
        "faithfulness": 0.8,
        "hallucination": 0.2,
        "answer_relevancy": 0.7,
        "context_precision": 0.6,
        "context_recall": 0.6,
    }

    if thresholds:
        default_thresholds.update(thresholds)

    # Run DeepEval evaluation
    deepeval_results = await evaluate_scenarios(results, default_thresholds)

    # Run Ragas evaluation if requested and available
    ragas_results: dict[str, float] = {}
    if include_ragas and RAGAS_AVAILABLE:
        try:
            ragas_results = await run_ragas_evaluation_async(results)
        except Exception as e:
            logger.warning(f"Ragas evaluation failed: {e}")
            ragas_results = {"error": str(e)}

    # Calculate combined score
    scores: list[float] = []

    # Add DeepEval scores
    if deepeval_results.get("summary", {}).get("avg_scores"):
        for metric_name, score in deepeval_results["summary"]["avg_scores"].items():
            if metric_name == "hallucination":
                # Invert hallucination score (lower is better)
                scores.append(1.0 - score)
            else:
                scores.append(score)

    # Add Ragas scores
    if ragas_results and "error" not in ragas_results:
        for score in ragas_results.values():
            if isinstance(score, (int, float)):
                scores.append(score)

    combined_score = sum(scores) / len(scores) if scores else 0.0

    # Determine overall pass/fail
    deepeval_passed = deepeval_results.get("summary", {}).get("pass_rate", 0) >= 0.5

    ragas_passed = True
    if ragas_results and "error" not in ragas_results:
        ragas_passed = (
            ragas_results.get("faithfulness", 0) >= default_thresholds.get("faithfulness", 0.8)
            and ragas_results.get("context_precision", 0)
            >= default_thresholds.get("context_precision", 0.6)
        )

    overall_passed = deepeval_passed and ragas_passed

    return {
        "deepeval": deepeval_results,
        "ragas": ragas_results,
        "combined_score": combined_score,
        "passed": overall_passed,
    }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Type definitions
    "MetricResult",
    "EvaluationResult",
    "BatchEvaluationResult",
    # Configuration
    "get_openrouter_api_key",
    "get_openrouter_base_url",
    "get_eval_model",
    "get_openrouter_judge",
    "configure_deepeval_for_openrouter",
    # Availability flags
    "DEEPEVAL_AVAILABLE",
    "RAGAS_AVAILABLE",
    # Metric factories
    "create_faithfulness_metric",
    "create_hallucination_metric",
    "create_answer_relevancy_metric",
    "create_all_metrics",
    # Test case creation
    "create_test_case",
    # Batch evaluation
    "evaluate_scenarios",
    # Ragas integration
    "create_ragas_dataset",
    "run_ragas_evaluation",
    "run_ragas_evaluation_async",
    # Combined evaluation
    "run_full_evaluation",
]
