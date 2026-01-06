"""
Phoenix Observability Configuration and Utilities.

This module provides integration with Arize Phoenix for LLM tracing,
evaluation visualization, and span annotations. It uses OpenTelemetry
with OpenInference instrumentation for comprehensive observability.

Key Features:
- OpenTelemetry tracer provider setup with Phoenix OTLP exporter
- Auto-instrumentation for LangChain/LangGraph workflows
- Custom span annotation utilities for evaluation metrics
- Evaluation span creation for Ragas/DeepEval scores
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Tracer
from opentelemetry.trace.span import Span

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

# Global tracer instance (initialized via setup_phoenix)
_tracer: Tracer | None = None
_tracer_provider: TracerProvider | None = None
_is_initialized: bool = False


# =============================================================================
# Phoenix Configuration
# =============================================================================


@dataclass
class PhoenixConfig:
    """
    Configuration for Phoenix observability integration.

    Attributes:
        collector_endpoint: OTLP collector endpoint (gRPC format).
        project_name: Phoenix project name for trace organization.
        enable_auto_instrument: Whether to auto-instrument LangChain/LangGraph.
        service_name: OpenTelemetry service name for this application.
        service_version: Application version for trace metadata.
        deployment_environment: Environment name (development/staging/production).
        sample_rate: Trace sampling rate (1.0 = 100% sampling).
        max_export_batch_size: Maximum spans per export batch.
        export_timeout_millis: Export timeout in milliseconds.
        enabled: Master switch to enable/disable all tracing.
    """

    collector_endpoint: str = "localhost:4317"
    project_name: str = "drx-research"
    enable_auto_instrument: bool = True
    service_name: str = "drx-deep-research"
    service_version: str = "1.0.0"
    deployment_environment: str = "development"
    sample_rate: float = 1.0
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000
    enabled: bool = True

    # Additional Phoenix-specific settings
    batch_schedule_delay_millis: int = 5000
    max_queue_size: int = 2048

    @classmethod
    def from_settings(cls) -> "PhoenixConfig":
        """
        Create PhoenixConfig from application settings.

        Returns:
            PhoenixConfig instance populated from environment variables.
        """
        from src.config import get_settings

        settings = get_settings()

        return cls(
            collector_endpoint=settings.PHOENIX_COLLECTOR_ENDPOINT,
            project_name=settings.PHOENIX_PROJECT_NAME,
            enable_auto_instrument=True,
            deployment_environment=settings.APP_ENV,
            enabled=True,
        )


# =============================================================================
# Phoenix Setup and Initialization
# =============================================================================


def setup_phoenix(config: PhoenixConfig | None = None) -> Tracer:
    """
    Initialize Phoenix tracing with OpenTelemetry and auto-instrumentation.

    This function sets up:
    1. OpenTelemetry tracer provider with Phoenix OTLP exporter
    2. LangChain/LangGraph auto-instrumentation via OpenInference
    3. Global tracer instance for custom spans

    Args:
        config: Phoenix configuration. If None, loads from settings.

    Returns:
        Configured OpenTelemetry Tracer instance.

    Raises:
        RuntimeError: If Phoenix initialization fails.

    Example:
        >>> config = PhoenixConfig(project_name="my-project")
        >>> tracer = setup_phoenix(config)
        >>> with tracer.start_as_current_span("my_operation") as span:
        ...     span.set_attribute("custom.attribute", "value")
    """
    global _tracer, _tracer_provider, _is_initialized

    if _is_initialized:
        logger.warning("Phoenix already initialized. Returning existing tracer.")
        return _tracer  # type: ignore

    if config is None:
        config = PhoenixConfig.from_settings()

    if not config.enabled:
        logger.info("Phoenix tracing is disabled. Using no-op tracer.")
        _tracer = trace.get_tracer(__name__)
        _is_initialized = True
        return _tracer

    try:
        # Import Phoenix OTEL registration
        from phoenix.otel import register

        # Register Phoenix tracer provider with OTLP exporter
        # This automatically configures the global tracer provider
        _tracer_provider = register(
            project_name=config.project_name,
            endpoint=config.collector_endpoint,
        )

        logger.info(
            "Phoenix tracer provider initialized",
            extra={
                "endpoint": config.collector_endpoint,
                "project": config.project_name,
            },
        )

        # Auto-instrument LangChain/LangGraph if enabled
        if config.enable_auto_instrument:
            _setup_auto_instrumentation()

        # Get the configured tracer
        _tracer = trace.get_tracer(
            instrumenting_module_name=config.service_name,
            instrumenting_library_version=config.service_version,
        )

        _is_initialized = True

        logger.info(
            "Phoenix observability initialized successfully",
            extra={
                "service_name": config.service_name,
                "auto_instrument": config.enable_auto_instrument,
            },
        )

        return _tracer

    except ImportError as e:
        logger.error(f"Failed to import Phoenix dependencies: {e}")
        logger.warning("Falling back to no-op tracer")
        _tracer = trace.get_tracer(__name__)
        _is_initialized = True
        return _tracer

    except Exception as e:
        logger.error(f"Failed to initialize Phoenix: {e}")
        raise RuntimeError(f"Phoenix initialization failed: {e}") from e


def _setup_auto_instrumentation() -> None:
    """
    Configure auto-instrumentation for LangChain and LangGraph.

    Uses OpenInference instrumentation packages to automatically
    capture spans for LLM calls, chain executions, and tool invocations.
    """
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # Check if already instrumented to avoid duplicate instrumentation
        if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
            LangChainInstrumentor().instrument()
            logger.info("LangChain auto-instrumentation enabled")
        else:
            logger.debug("LangChain already instrumented")

    except ImportError:
        logger.warning(
            "openinference-instrumentation-langchain not installed. "
            "LangChain auto-instrumentation disabled."
        )
    except Exception as e:
        logger.warning(f"Failed to enable LangChain auto-instrumentation: {e}")

    # Optionally instrument OpenAI client directly for non-LangChain calls
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        if not OpenAIInstrumentor().is_instrumented_by_opentelemetry:
            OpenAIInstrumentor().instrument()
            logger.info("OpenAI auto-instrumentation enabled")

    except ImportError:
        logger.debug("OpenAI instrumentation not available")
    except Exception as e:
        logger.debug(f"OpenAI instrumentation skipped: {e}")


def shutdown_phoenix() -> None:
    """
    Gracefully shutdown Phoenix tracing.

    Flushes any pending spans and releases resources.
    Should be called during application shutdown.
    """
    global _tracer, _tracer_provider, _is_initialized

    if _tracer_provider is not None:
        try:
            # Force flush any pending spans
            if hasattr(_tracer_provider, 'force_flush'):
                _tracer_provider.force_flush(timeout_millis=5000)

            # Shutdown the provider
            if hasattr(_tracer_provider, 'shutdown'):
                _tracer_provider.shutdown()

            logger.info("Phoenix tracer provider shut down successfully")

        except Exception as e:
            logger.error(f"Error during Phoenix shutdown: {e}")
        finally:
            _tracer = None
            _tracer_provider = None
            _is_initialized = False


def is_phoenix_initialized() -> bool:
    """Check if Phoenix has been initialized."""
    return _is_initialized


# =============================================================================
# Tracer Access
# =============================================================================


def get_tracer(name: str | None = None) -> Tracer:
    """
    Get a configured OpenTelemetry tracer instance.

    If Phoenix hasn't been initialized, this will initialize it with
    default settings. Use a custom name for component-specific tracers.

    Args:
        name: Optional tracer name. If None, returns the global tracer.

    Returns:
        OpenTelemetry Tracer instance.

    Example:
        >>> tracer = get_tracer("planner_agent")
        >>> with tracer.start_as_current_span("plan_research") as span:
        ...     # Your code here
        ...     pass
    """
    global _tracer

    if not _is_initialized:
        setup_phoenix()

    if name is not None:
        return trace.get_tracer(name)

    return _tracer or trace.get_tracer(__name__)


def get_current_span() -> Span:
    """
    Get the currently active span.

    Returns:
        The current span from the trace context, or a non-recording
        span if no span is active.
    """
    return trace.get_current_span()


# =============================================================================
# Span Annotation Utilities
# =============================================================================


def annotate_span(
    attributes: dict[str, Any],
    span: Span | None = None,
) -> None:
    """
    Add custom attributes to a span for Phoenix visualization.

    Attributes are automatically prefixed with 'drx.' namespace to
    distinguish custom attributes from standard OpenTelemetry attributes.

    Args:
        attributes: Dictionary of attribute key-value pairs.
        span: Target span. If None, uses the current active span.

    Supported attribute types:
        - str, int, float, bool: Direct assignment
        - list: Converted to JSON string
        - dict: Converted to JSON string

    Example:
        >>> annotate_span({
        ...     "agent_type": "planner",
        ...     "iteration": 2,
        ...     "coverage_score": 0.85,
        ...     "gaps": ["market analysis", "competitor data"],
        ... })
    """
    import json

    if span is None:
        span = get_current_span()

    if not span.is_recording():
        return

    for key, value in attributes.items():
        # Prefix with namespace
        attr_key = f"drx.{key}" if not key.startswith("drx.") else key

        # Handle complex types
        if isinstance(value, (list, dict)):
            try:
                span.set_attribute(attr_key, json.dumps(value))
            except (TypeError, ValueError):
                span.set_attribute(attr_key, str(value))
        elif isinstance(value, (str, int, float, bool)):
            span.set_attribute(attr_key, value)
        elif value is None:
            span.set_attribute(attr_key, "null")
        else:
            span.set_attribute(attr_key, str(value))


def annotate_span_with_state(
    state: dict[str, Any],
    include_keys: list[str] | None = None,
    exclude_keys: list[str] | None = None,
    span: Span | None = None,
) -> None:
    """
    Annotate span with AgentState fields.

    Selectively adds state information to the span for debugging
    and analysis in Phoenix.

    Args:
        state: AgentState dictionary.
        include_keys: If specified, only include these keys.
        exclude_keys: Keys to exclude from annotation.
        span: Target span. If None, uses current span.
    """
    if span is None:
        span = get_current_span()

    if not span.is_recording():
        return

    exclude_keys = exclude_keys or []
    # Default exclusions for large/sensitive fields
    default_exclude = {"messages", "synthesis", "final_report"}
    exclude_set = set(exclude_keys) | default_exclude

    if include_keys:
        keys_to_process = [k for k in include_keys if k not in exclude_set]
    else:
        keys_to_process = [k for k in state.keys() if k not in exclude_set]

    state_attrs = {}
    for key in keys_to_process:
        if key in state:
            value = state[key]
            state_attrs[f"state.{key}"] = value

    annotate_span(state_attrs, span)


def set_span_status(
    status: StatusCode,
    description: str | None = None,
    span: Span | None = None,
) -> None:
    """
    Set the status of a span.

    Args:
        status: StatusCode (OK, ERROR, UNSET).
        description: Optional description (typically for errors).
        span: Target span. If None, uses current span.
    """
    if span is None:
        span = get_current_span()

    span.set_status(Status(status, description))


def record_exception(
    exception: BaseException,
    span: Span | None = None,
    set_status: bool = True,
) -> None:
    """
    Record an exception on a span.

    Args:
        exception: The exception to record.
        span: Target span. If None, uses current span.
        set_status: Whether to set span status to ERROR.
    """
    if span is None:
        span = get_current_span()

    span.record_exception(exception)

    if set_status:
        span.set_status(Status(StatusCode.ERROR, str(exception)))


# =============================================================================
# Evaluation Span Creation
# =============================================================================


@contextmanager
def create_evaluation_span(
    evaluation_name: str,
    evaluation_type: str = "ragas",
    parent_span: Span | None = None,
) -> Generator[Span, None, None]:
    """
    Create a span specifically for evaluation metrics.

    This context manager creates a span optimized for Phoenix evaluation
    visualization, with standard attributes for evaluation frameworks.

    Args:
        evaluation_name: Name of the evaluation (e.g., "faithfulness", "answer_relevancy").
        evaluation_type: Evaluation framework ("ragas", "deepeval", "custom").
        parent_span: Optional parent span. If None, uses current context.

    Yields:
        The evaluation span for adding metrics.

    Example:
        >>> with create_evaluation_span("faithfulness", "ragas") as eval_span:
        ...     score = calculate_faithfulness(response, context)
        ...     log_evaluation_metric("faithfulness", score, span=eval_span)
    """
    tracer = get_tracer("drx.evaluation")

    # Set up span context
    context = None
    if parent_span is not None:
        context = trace.set_span_in_context(parent_span)

    with tracer.start_as_current_span(
        name=f"evaluation.{evaluation_name}",
        context=context,
    ) as span:
        # Set standard evaluation attributes
        span.set_attribute("evaluation.name", evaluation_name)
        span.set_attribute("evaluation.type", evaluation_type)
        span.set_attribute("openinference.span.kind", "EVALUATION")

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def log_evaluation_metric(
    metric_name: str,
    score: float,
    threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
    span: Span | None = None,
) -> None:
    """
    Log an evaluation metric to a span for Phoenix visualization.

    Creates structured attributes that Phoenix can parse for
    evaluation dashboards and comparisons.

    Args:
        metric_name: Name of the metric (e.g., "faithfulness", "relevancy").
        score: Metric score (typically 0.0-1.0).
        threshold: Optional threshold for pass/fail determination.
        metadata: Additional metric metadata.
        span: Target span. If None, uses current span.

    Example:
        >>> log_evaluation_metric(
        ...     metric_name="faithfulness",
        ...     score=0.92,
        ...     threshold=0.7,
        ...     metadata={"model": "gpt-4", "context_length": 1500},
        ... )
    """
    import json

    if span is None:
        span = get_current_span()

    if not span.is_recording():
        logger.debug(f"Skipping metric logging: span not recording ({metric_name}={score})")
        return

    # Standard metric attributes
    span.set_attribute(f"evaluation.metric.{metric_name}", score)
    span.set_attribute("evaluation.score", score)  # For Phoenix recognition

    if threshold is not None:
        span.set_attribute(f"evaluation.threshold.{metric_name}", threshold)
        span.set_attribute("evaluation.passed", score >= threshold)

    if metadata:
        span.set_attribute(
            f"evaluation.metadata.{metric_name}",
            json.dumps(metadata),
        )

    logger.debug(
        f"Logged evaluation metric: {metric_name}={score}",
        extra={"threshold": threshold, "passed": score >= threshold if threshold else None},
    )


def log_ragas_scores(
    scores: dict[str, float],
    span: Span | None = None,
) -> None:
    """
    Log multiple Ragas evaluation scores to a span.

    Convenience function for logging common Ragas metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall

    Args:
        scores: Dictionary of metric_name -> score.
        span: Target span. If None, uses current span.

    Example:
        >>> log_ragas_scores({
        ...     "faithfulness": 0.95,
        ...     "answer_relevancy": 0.88,
        ...     "context_precision": 0.72,
        ...     "context_recall": 0.81,
        ... })
    """
    if span is None:
        span = get_current_span()

    # Standard thresholds for Ragas metrics
    thresholds = {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_precision": 0.6,
        "context_recall": 0.6,
        "harmfulness": 0.0,  # Lower is better
        "coherence": 0.7,
    }

    for metric_name, score in scores.items():
        threshold = thresholds.get(metric_name)
        log_evaluation_metric(
            metric_name=metric_name,
            score=score,
            threshold=threshold,
            metadata={"framework": "ragas"},
            span=span,
        )


def log_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int | None = None,
    model: str | None = None,
    span: Span | None = None,
) -> None:
    """
    Log LLM token usage to a span.

    Uses OpenInference semantic conventions for token tracking.

    Args:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens (computed if not provided).
        model: Model name for attribution.
        span: Target span. If None, uses current span.
    """
    if span is None:
        span = get_current_span()

    if not span.is_recording():
        return

    total = total_tokens or (prompt_tokens + completion_tokens)

    # OpenInference semantic conventions
    span.set_attribute("llm.token_count.prompt", prompt_tokens)
    span.set_attribute("llm.token_count.completion", completion_tokens)
    span.set_attribute("llm.token_count.total", total)

    # Additional DRX attributes
    span.set_attribute("drx.tokens.prompt", prompt_tokens)
    span.set_attribute("drx.tokens.completion", completion_tokens)
    span.set_attribute("drx.tokens.total", total)

    if model:
        span.set_attribute("llm.model_name", model)
        span.set_attribute("drx.model", model)


# =============================================================================
# Batch Evaluation Utilities
# =============================================================================


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    metric_name: str
    score: float
    threshold: float | None = None
    passed: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold is not None and self.passed is None:
            self.passed = self.score >= self.threshold


def log_batch_evaluation(
    results: list[EvaluationResult],
    batch_name: str = "batch_evaluation",
    span: Span | None = None,
) -> dict[str, Any]:
    """
    Log multiple evaluation results as a batch.

    Creates a summary of the batch evaluation and logs individual
    metrics to the span.

    Args:
        results: List of evaluation results.
        batch_name: Name for the batch operation.
        span: Target span. If None, uses current span.

    Returns:
        Summary dictionary with aggregate statistics.
    """
    import json

    if span is None:
        span = get_current_span()

    # Log individual metrics
    for result in results:
        log_evaluation_metric(
            metric_name=result.metric_name,
            score=result.score,
            threshold=result.threshold,
            metadata=result.metadata,
            span=span,
        )

    # Compute summary statistics
    scores = [r.score for r in results]
    passed_count = sum(1 for r in results if r.passed is True)

    summary = {
        "batch_name": batch_name,
        "total_metrics": len(results),
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "passed_count": passed_count,
        "pass_rate": passed_count / len(results) if results else 0,
    }

    if span and span.is_recording():
        span.set_attribute("evaluation.batch.name", batch_name)
        span.set_attribute("evaluation.batch.summary", json.dumps(summary))
        span.set_attribute("evaluation.batch.avg_score", summary["avg_score"])
        span.set_attribute("evaluation.batch.pass_rate", summary["pass_rate"])

    return summary


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Configuration
    "PhoenixConfig",
    # Setup functions
    "setup_phoenix",
    "shutdown_phoenix",
    "is_phoenix_initialized",
    # Tracer access
    "get_tracer",
    "get_current_span",
    # Span annotation
    "annotate_span",
    "annotate_span_with_state",
    "set_span_status",
    "record_exception",
    # Evaluation spans
    "create_evaluation_span",
    "log_evaluation_metric",
    "log_ragas_scores",
    "log_token_usage",
    # Batch evaluation
    "EvaluationResult",
    "log_batch_evaluation",
]
