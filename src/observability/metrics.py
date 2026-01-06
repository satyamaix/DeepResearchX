"""
Prometheus Metrics for DRX Deep Research System.

Provides metrics collection for monitoring research sessions,
agent performance, token usage, and API latency.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# =============================================================================
# Core Metrics Definitions
# =============================================================================

# Token usage tracking
TOKENS_TOTAL = Counter(
    "drx_tokens_total",
    "Total tokens consumed",
    ["model", "agent", "direction"],  # direction: input/output
)

# Request latency
REQUEST_LATENCY = Histogram(
    "drx_request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint", "method", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Active sessions gauge
SESSIONS_ACTIVE = Gauge(
    "drx_sessions_active",
    "Number of currently active research sessions",
)

# Session counters
SESSIONS_TOTAL = Counter(
    "drx_sessions_total",
    "Total research sessions",
    ["status"],  # status: started/completed/failed/cancelled
)

# Agent execution metrics
AGENT_EXECUTIONS = Counter(
    "drx_agent_executions_total",
    "Total agent executions",
    ["agent", "status"],  # agent: planner/searcher/reader/etc, status: success/error
)

AGENT_LATENCY = Histogram(
    "drx_agent_latency_seconds",
    "Agent execution latency",
    ["agent"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# LLM call metrics
LLM_CALLS_TOTAL = Counter(
    "drx_llm_calls_total",
    "Total LLM API calls",
    ["model", "status"],
)

LLM_LATENCY = Histogram(
    "drx_llm_latency_seconds",
    "LLM API call latency",
    ["model"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# Research quality metrics
FINDINGS_TOTAL = Counter(
    "drx_findings_total",
    "Total findings extracted",
    ["confidence_bucket"],  # high/medium/low
)

CITATIONS_TOTAL = Counter(
    "drx_citations_total",
    "Total citations retrieved",
)

# Cost tracking
COST_TOTAL = Counter(
    "drx_cost_dollars_total",
    "Total cost in dollars",
    ["model"],
)

# Application info
APP_INFO = Info(
    "drx_app",
    "DRX application information",
)


# =============================================================================
# Metrics Helpers
# =============================================================================


def track_tokens(
    model: str,
    agent: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Track token usage for a model call."""
    TOKENS_TOTAL.labels(model=model, agent=agent, direction="input").inc(input_tokens)
    TOKENS_TOTAL.labels(model=model, agent=agent, direction="output").inc(output_tokens)


def track_llm_call(
    model: str,
    latency_seconds: float,
    success: bool,
    cost: float | None = None,
) -> None:
    """Track an LLM API call."""
    status = "success" if success else "error"
    LLM_CALLS_TOTAL.labels(model=model, status=status).inc()
    LLM_LATENCY.labels(model=model).observe(latency_seconds)
    if cost is not None:
        COST_TOTAL.labels(model=model).inc(cost)


def track_agent_execution(
    agent: str,
    latency_seconds: float,
    success: bool,
) -> None:
    """Track an agent execution."""
    status = "success" if success else "error"
    AGENT_EXECUTIONS.labels(agent=agent, status=status).inc()
    AGENT_LATENCY.labels(agent=agent).observe(latency_seconds)


def track_session_start() -> None:
    """Track a new session starting."""
    SESSIONS_ACTIVE.inc()
    SESSIONS_TOTAL.labels(status="started").inc()


def track_session_end(status: str = "completed") -> None:
    """Track a session ending."""
    SESSIONS_ACTIVE.dec()
    SESSIONS_TOTAL.labels(status=status).inc()


def track_finding(confidence: float) -> None:
    """Track a finding extraction."""
    if confidence >= 0.7:
        bucket = "high"
    elif confidence >= 0.4:
        bucket = "medium"
    else:
        bucket = "low"
    FINDINGS_TOTAL.labels(confidence_bucket=bucket).inc()


def track_citation() -> None:
    """Track a citation retrieval."""
    CITATIONS_TOTAL.inc()


@contextmanager
def track_request_latency(
    endpoint: str,
    method: str,
) -> Generator[None, None, None]:
    """Context manager to track request latency."""
    start = time.perf_counter()
    status = "200"
    try:
        yield
    except Exception:
        status = "500"
        raise
    finally:
        latency = time.perf_counter() - start
        REQUEST_LATENCY.labels(
            endpoint=endpoint,
            method=method,
            status=status,
        ).observe(latency)


@contextmanager
def track_agent_latency(agent: str) -> Generator[None, None, None]:
    """Context manager to track agent execution latency."""
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        track_agent_execution(agent, latency, success)


def agent_metrics(agent_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to track agent execution metrics."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with track_agent_latency(agent_name):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with track_agent_latency(agent_name):
                return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Metrics Endpoint
# =============================================================================


def get_metrics() -> bytes:
    """
    Generate Prometheus metrics in the standard exposition format.

    Returns:
        Prometheus-formatted metrics as bytes
    """
    return generate_latest(REGISTRY)


def get_metrics_content_type() -> str:
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST


def set_app_info(version: str, environment: str) -> None:
    """Set application info labels."""
    APP_INFO.info({
        "version": version,
        "environment": environment,
    })


# =============================================================================
# FastAPI Integration
# =============================================================================


class MetricsMiddleware:
    """
    FastAPI/Starlette middleware for automatic request metrics.

    Usage:
        app.add_middleware(MetricsMiddleware)
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        method = scope.get("method", "GET")

        # Skip metrics endpoint itself
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = "500"

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = str(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            latency = time.perf_counter() - start
            # Normalize path to avoid high cardinality
            normalized_path = self._normalize_path(path)
            REQUEST_LATENCY.labels(
                endpoint=normalized_path,
                method=method,
                status=status_code,
            ).observe(latency)

    def _normalize_path(self, path: str) -> str:
        """Normalize path to reduce cardinality."""
        # Replace UUIDs with placeholder
        import re

        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path,
            flags=re.IGNORECASE,
        )
        # Replace numeric IDs
        path = re.sub(r"/\d+(?=/|$)", "/{id}", path)
        return path


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Metrics
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
    # Helper functions
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
    # Endpoint
    "get_metrics",
    "get_metrics_content_type",
    "set_app_info",
    # Middleware
    "MetricsMiddleware",
]
