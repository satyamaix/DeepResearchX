"""
DRX Observability Package.

This package provides comprehensive observability for the DRX Deep Research
system using Arize Phoenix for LLM tracing, evaluation visualization, and
span annotations.

Components:
- phoenix: Phoenix configuration, setup, and span annotation utilities
- tracing: Custom decorators for agents, tools, and LLM calls

Quick Start:
    >>> from src.observability import setup_phoenix, trace_agent, trace_tool
    >>>
    >>> # Initialize Phoenix tracing
    >>> tracer = setup_phoenix()
    >>>
    >>> # Decorate agent functions
    >>> @trace_agent("planner")
    ... async def plan_research(state: AgentState) -> AgentState:
    ...     ...
    >>>
    >>> # Decorate tool functions
    >>> @trace_tool("tavily_search")
    ... async def search(query: str) -> list[SearchResult]:
    ...     ...
"""

from src.observability.phoenix import (
    # Configuration
    PhoenixConfig,
    # Setup functions
    setup_phoenix,
    shutdown_phoenix,
    is_phoenix_initialized,
    # Tracer access
    get_tracer,
    get_current_span,
    # Span annotation
    annotate_span,
    annotate_span_with_state,
    set_span_status,
    record_exception,
    # Evaluation spans
    create_evaluation_span,
    log_evaluation_metric,
    log_ragas_scores,
    log_token_usage,
    # Batch evaluation
    EvaluationResult,
    log_batch_evaluation,
)

from src.observability.tracing import (
    # Constants
    SpanAttributes,
    # Decorators
    trace_agent,
    trace_tool,
    trace_llm_call,
    trace_workflow_node,
    trace_conditional_edge,
    # Context managers
    TraceContext,
    trace_span,
    async_trace_span,
)

__all__ = [
    # Phoenix Configuration
    "PhoenixConfig",
    # Setup
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
    # Evaluation
    "create_evaluation_span",
    "log_evaluation_metric",
    "log_ragas_scores",
    "log_token_usage",
    "EvaluationResult",
    "log_batch_evaluation",
    # Constants
    "SpanAttributes",
    # Decorators
    "trace_agent",
    "trace_tool",
    "trace_llm_call",
    "trace_workflow_node",
    "trace_conditional_edge",
    # Context managers
    "TraceContext",
    "trace_span",
    "async_trace_span",
]
