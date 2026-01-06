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

from src.observability.metrics import (
    # Metrics
    TOKENS_TOTAL,
    REQUEST_LATENCY,
    SESSIONS_ACTIVE,
    SESSIONS_TOTAL,
    AGENT_EXECUTIONS,
    AGENT_LATENCY,
    LLM_CALLS_TOTAL,
    LLM_LATENCY,
    FINDINGS_TOTAL,
    CITATIONS_TOTAL,
    COST_TOTAL,
    APP_INFO,
    # Helper functions
    track_tokens,
    track_llm_call,
    track_agent_execution,
    track_session_start,
    track_session_end,
    track_finding,
    track_citation,
    track_request_latency,
    track_agent_latency,
    agent_metrics,
    # Endpoint
    get_metrics,
    get_metrics_content_type,
    set_app_info,
    # Middleware
    MetricsMiddleware,
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
    # Prometheus Metrics
    "TOKENS_TOTAL",
    "REQUEST_LATENCY",
    "SESSIONS_ACTIVE",
    "SESSIONS_TOTAL",
    "AGENT_EXECUTIONS",
    "AGENT_LATENCY",
    "LLM_CALLS_TOTAL",
    "LLM_LATENCY",
    "FINDINGS_TOTAL",
    "CITATIONS_TOTAL",
    "COST_TOTAL",
    "APP_INFO",
    # Metrics Helper functions
    "track_tokens",
    "track_llm_call",
    "track_agent_execution",
    "track_session_start",
    "track_session_end",
    "track_finding",
    "track_citation",
    "track_request_latency",
    "track_agent_latency",
    "agent_metrics",
    # Metrics Endpoint
    "get_metrics",
    "get_metrics_content_type",
    "set_app_info",
    # Metrics Middleware
    "MetricsMiddleware",
]
