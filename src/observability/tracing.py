"""
Custom Tracing Decorators and Utilities for DRX Deep Research.

This module provides tracing decorators and context managers for
instrumenting agents, tools, and LLM calls with OpenTelemetry spans.

Key Components:
- @trace_agent: Decorator for agent invocation tracing
- @trace_tool: Decorator for tool execution tracing
- @trace_llm_call: Decorator for LLM API call tracing
- TraceContext: Context manager for custom spans with nesting support

These decorators integrate with Phoenix for visualization and analysis
of the research workflow execution.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    ParamSpec,
    TypeVar,
)

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span, SpanKind

from src.observability.phoenix import (
    annotate_span,
    get_current_span,
    get_tracer,
    log_token_usage,
    record_exception,
)

if TYPE_CHECKING:
    from src.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

# Type variables for generic decorators
P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Span Attribute Constants (OpenInference Semantic Conventions)
# =============================================================================


class SpanAttributes:
    """OpenInference and custom span attribute names."""

    # OpenInference standard attributes
    INPUT_VALUE = "input.value"
    INPUT_MIME_TYPE = "input.mime_type"
    OUTPUT_VALUE = "output.value"
    OUTPUT_MIME_TYPE = "output.mime_type"

    # LLM attributes
    LLM_MODEL_NAME = "llm.model_name"
    LLM_INVOCATION_PARAMS = "llm.invocation_parameters"
    LLM_PROMPTS = "llm.prompts"
    LLM_PROMPT_TEMPLATE = "llm.prompt_template"
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"

    # Tool attributes
    TOOL_NAME = "tool.name"
    TOOL_PARAMETERS = "tool.parameters"
    TOOL_DESCRIPTION = "tool.description"

    # Agent attributes (DRX custom)
    AGENT_TYPE = "drx.agent.type"
    AGENT_NAME = "drx.agent.name"
    AGENT_ITERATION = "drx.agent.iteration"
    AGENT_INPUT_STATE = "drx.agent.input_state"
    AGENT_OUTPUT_STATE = "drx.agent.output_state"

    # Session attributes
    SESSION_ID = "drx.session.id"
    USER_QUERY = "drx.user.query"

    # Performance attributes
    LATENCY_MS = "drx.latency_ms"
    START_TIME = "drx.start_time"
    END_TIME = "drx.end_time"


# =============================================================================
# Agent Tracing Decorator
# =============================================================================


def trace_agent(
    agent_name: str,
    agent_type: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
    include_state_fields: list[str] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing agent invocations.

    Creates a span for each agent execution, recording:
    - Input/output state
    - Token usage (if available in state)
    - Agent-specific attributes
    - Execution duration
    - Errors and exceptions

    Args:
        agent_name: Human-readable name for the agent.
        agent_type: Agent type (planner, searcher, etc.). Inferred if not provided.
        capture_input: Whether to capture input state.
        capture_output: Whether to capture output state.
        include_state_fields: Specific state fields to capture (default captures safe fields).

    Returns:
        Decorated function with tracing enabled.

    Example:
        >>> @trace_agent("planner", agent_type="planner")
        ... async def plan_research(state: AgentState) -> AgentState:
        ...     # Planning logic here
        ...     return updated_state
    """
    # Default state fields to capture (avoid large/sensitive fields)
    default_fields = [
        "session_id",
        "current_phase",
        "iteration_count",
        "tokens_used",
        "tokens_remaining",
    ]
    state_fields = include_state_fields or default_fields

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Determine if function is async
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer("drx.agents")
            span_name = f"agent.{agent_name}"

            with tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
            ) as span:
                start_time = time.perf_counter()

                # Set basic agent attributes
                span.set_attribute("openinference.span.kind", "AGENT")
                span.set_attribute(SpanAttributes.AGENT_NAME, agent_name)
                span.set_attribute(
                    SpanAttributes.AGENT_TYPE,
                    agent_type or _infer_agent_type(agent_name),
                )

                # Capture input state
                input_state = _extract_state_from_args(args, kwargs)
                if capture_input and input_state:
                    _annotate_input_state(span, input_state, state_fields)

                try:
                    # Execute the agent function
                    result = await func(*args, **kwargs)

                    # Capture output state
                    if capture_output and isinstance(result, dict):
                        _annotate_output_state(span, result, state_fields)

                        # Track token usage if available
                        if "tokens_used" in result:
                            input_tokens = input_state.get("tokens_used", 0) if input_state else 0
                            new_tokens = result["tokens_used"] - input_tokens
                            if new_tokens > 0:
                                span.set_attribute("drx.tokens.consumed", new_tokens)

                    # Record duration
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)
                    record_exception(e, span)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer("drx.agents")
            span_name = f"agent.{agent_name}"

            with tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
            ) as span:
                start_time = time.perf_counter()

                span.set_attribute("openinference.span.kind", "AGENT")
                span.set_attribute(SpanAttributes.AGENT_NAME, agent_name)
                span.set_attribute(
                    SpanAttributes.AGENT_TYPE,
                    agent_type or _infer_agent_type(agent_name),
                )

                input_state = _extract_state_from_args(args, kwargs)
                if capture_input and input_state:
                    _annotate_input_state(span, input_state, state_fields)

                try:
                    result = func(*args, **kwargs)

                    if capture_output and isinstance(result, dict):
                        _annotate_output_state(span, result, state_fields)

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)
                    record_exception(e, span)
                    raise

        return async_wrapper if is_async else sync_wrapper  # type: ignore

    return decorator


def _infer_agent_type(agent_name: str) -> str:
    """Infer agent type from agent name."""
    agent_types = ["planner", "searcher", "reader", "synthesizer", "critic", "reporter"]
    name_lower = agent_name.lower()
    for agent_type in agent_types:
        if agent_type in name_lower:
            return agent_type
    return "unknown"


def _extract_state_from_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract AgentState from function arguments."""
    # Check kwargs first
    if "state" in kwargs:
        return kwargs["state"]

    # Check positional args
    for arg in args:
        if isinstance(arg, dict) and "session_id" in arg:
            return arg

    return None


def _annotate_input_state(
    span: Span,
    state: dict[str, Any],
    fields: list[str],
) -> None:
    """Annotate span with input state fields."""
    for field_name in fields:
        if field_name in state:
            value = state[field_name]
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"drx.input.{field_name}", value)
            elif value is not None:
                try:
                    span.set_attribute(f"drx.input.{field_name}", json.dumps(value))
                except (TypeError, ValueError):
                    span.set_attribute(f"drx.input.{field_name}", str(value))


def _annotate_output_state(
    span: Span,
    state: dict[str, Any],
    fields: list[str],
) -> None:
    """Annotate span with output state fields."""
    for field_name in fields:
        if field_name in state:
            value = state[field_name]
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"drx.output.{field_name}", value)
            elif value is not None:
                try:
                    span.set_attribute(f"drx.output.{field_name}", json.dumps(value))
                except (TypeError, ValueError):
                    span.set_attribute(f"drx.output.{field_name}", str(value))


# =============================================================================
# Tool Tracing Decorator
# =============================================================================


def trace_tool(
    tool_name: str,
    tool_description: str | None = None,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    max_output_length: int = 10000,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing tool invocations.

    Creates a span for each tool execution, recording:
    - Tool inputs (parameters)
    - Tool outputs (results)
    - Execution latency
    - Errors and exceptions

    Args:
        tool_name: Name of the tool (e.g., "tavily_search", "web_scraper").
        tool_description: Human-readable description of the tool.
        capture_inputs: Whether to capture input parameters.
        capture_outputs: Whether to capture output results.
        max_output_length: Maximum length for output capture (truncated if exceeded).

    Returns:
        Decorated function with tracing enabled.

    Example:
        >>> @trace_tool("tavily_search", "Search the web using Tavily API")
        ... async def search(query: str, max_results: int = 10) -> list[SearchResult]:
        ...     # Search implementation
        ...     return results
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer("drx.tools")
            span_name = f"tool.{tool_name}"

            with tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
            ) as span:
                start_time = time.perf_counter()

                # Set tool attributes
                span.set_attribute("openinference.span.kind", "TOOL")
                span.set_attribute(SpanAttributes.TOOL_NAME, tool_name)
                if tool_description:
                    span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, tool_description)

                # Capture inputs
                if capture_inputs:
                    _annotate_tool_inputs(span, func, args, kwargs)

                try:
                    result = await func(*args, **kwargs)

                    # Capture outputs
                    if capture_outputs:
                        _annotate_tool_output(span, result, max_output_length)

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)
                    record_exception(e, span)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer("drx.tools")
            span_name = f"tool.{tool_name}"

            with tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
            ) as span:
                start_time = time.perf_counter()

                span.set_attribute("openinference.span.kind", "TOOL")
                span.set_attribute(SpanAttributes.TOOL_NAME, tool_name)
                if tool_description:
                    span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, tool_description)

                if capture_inputs:
                    _annotate_tool_inputs(span, func, args, kwargs)

                try:
                    result = func(*args, **kwargs)

                    if capture_outputs:
                        _annotate_tool_output(span, result, max_output_length)

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)
                    record_exception(e, span)
                    raise

        return async_wrapper if is_async else sync_wrapper  # type: ignore

    return decorator


def _annotate_tool_inputs(
    span: Span,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Annotate span with tool input parameters."""
    # Get function signature to map args to parameter names
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Build parameters dict
    tool_params = {}

    # Map positional args
    for i, arg in enumerate(args):
        if i < len(params):
            param_name = params[i]
            # Skip 'self' and 'cls'
            if param_name not in ("self", "cls"):
                tool_params[param_name] = _safe_serialize(arg)

    # Add kwargs
    for key, value in kwargs.items():
        tool_params[key] = _safe_serialize(value)

    # Set as span attribute
    try:
        span.set_attribute(SpanAttributes.TOOL_PARAMETERS, json.dumps(tool_params))
        span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(tool_params))
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
    except (TypeError, ValueError):
        span.set_attribute(SpanAttributes.INPUT_VALUE, str(tool_params))


def _annotate_tool_output(
    span: Span,
    result: Any,
    max_length: int,
) -> None:
    """Annotate span with tool output."""
    try:
        if result is None:
            output_str = "null"
        elif isinstance(result, (str, int, float, bool)):
            output_str = str(result)
        else:
            output_str = json.dumps(_safe_serialize(result))

        # Truncate if needed
        if len(output_str) > max_length:
            output_str = output_str[:max_length] + "...[truncated]"

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_str)
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")

    except (TypeError, ValueError):
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result)[:max_length])


def _safe_serialize(value: Any, max_depth: int = 3) -> Any:
    """Safely serialize a value for JSON encoding."""
    if max_depth <= 0:
        return str(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_safe_serialize(v, max_depth - 1) for v in value[:100]]  # Limit list length
    elif isinstance(value, dict):
        return {
            str(k): _safe_serialize(v, max_depth - 1)
            for k, v in list(value.items())[:50]  # Limit dict size
        }
    else:
        return str(value)


# =============================================================================
# LLM Call Tracing Decorator
# =============================================================================


def trace_llm_call(
    func: Callable[P, T] | None = None,
    *,
    model_param: str = "model",
    capture_prompts: bool = True,
    capture_completions: bool = True,
    max_prompt_length: int = 50000,
    max_completion_length: int = 50000,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing LLM API calls.

    Creates a span for each LLM invocation, recording:
    - Model name and invocation parameters
    - Prompt/messages (optional)
    - Completion/response (optional)
    - Token usage (prompt, completion, total)
    - Latency

    Can be used with or without arguments:
        @trace_llm_call
        async def chat_completion(...): ...

        @trace_llm_call(model_param="model_name")
        async def chat_completion(...): ...

    Args:
        func: The function to decorate (when used without arguments).
        model_param: Parameter name for the model (default: "model").
        capture_prompts: Whether to capture prompt/messages.
        capture_completions: Whether to capture completions.
        max_prompt_length: Maximum length for prompt capture.
        max_completion_length: Maximum length for completion capture.

    Returns:
        Decorated function with LLM tracing enabled.

    Example:
        >>> @trace_llm_call
        ... async def chat_completion(
        ...     messages: list[dict],
        ...     model: str = "gpt-4",
        ...     temperature: float = 0.7,
        ... ) -> ChatCompletionResponse:
        ...     # LLM API call
        ...     return response
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        is_async = asyncio.iscoroutinefunction(fn)

        @functools.wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer("drx.llm")

            # Extract model name
            model = _extract_param(fn, args, kwargs, model_param) or "unknown"
            span_name = f"llm.{model.replace('/', '_')}"

            with tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                start_time = time.perf_counter()

                # Set LLM attributes
                span.set_attribute("openinference.span.kind", "LLM")
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)

                # Capture invocation parameters
                _annotate_llm_params(span, fn, args, kwargs, model_param)

                # Capture prompts/messages
                if capture_prompts:
                    _annotate_llm_prompts(span, fn, args, kwargs, max_prompt_length)

                try:
                    result = await fn(*args, **kwargs)

                    # Capture completion
                    if capture_completions:
                        _annotate_llm_completion(span, result, max_completion_length)

                    # Extract and log token usage
                    _extract_and_log_tokens(span, result)

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)
                    record_exception(e, span)
                    raise

        @functools.wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer("drx.llm")

            model = _extract_param(fn, args, kwargs, model_param) or "unknown"
            span_name = f"llm.{model.replace('/', '_')}"

            with tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                start_time = time.perf_counter()

                span.set_attribute("openinference.span.kind", "LLM")
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)

                _annotate_llm_params(span, fn, args, kwargs, model_param)

                if capture_prompts:
                    _annotate_llm_prompts(span, fn, args, kwargs, max_prompt_length)

                try:
                    result = fn(*args, **kwargs)

                    if capture_completions:
                        _annotate_llm_completion(span, result, max_completion_length)

                    _extract_and_log_tokens(span, result)

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)
                    record_exception(e, span)
                    raise

        return async_wrapper if is_async else sync_wrapper  # type: ignore

    # Support both @trace_llm_call and @trace_llm_call()
    if func is not None:
        return decorator(func)
    return decorator


def _extract_param(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    param_name: str,
) -> Any | None:
    """Extract a parameter value by name from function arguments."""
    if param_name in kwargs:
        return kwargs[param_name]

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    for i, p in enumerate(params):
        if p == param_name and i < len(args):
            return args[i]

    return None


def _annotate_llm_params(
    span: Span,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    model_param: str,
) -> None:
    """Annotate span with LLM invocation parameters."""
    # Common LLM parameters to capture
    param_names = ["temperature", "max_tokens", "top_p", "top_k", "stop", "presence_penalty", "frequency_penalty"]

    invocation_params = {}
    for param_name in param_names:
        value = _extract_param(func, args, kwargs, param_name)
        if value is not None:
            invocation_params[param_name] = value
            # Also set as individual attribute
            span.set_attribute(f"llm.{param_name}", value if isinstance(value, (str, int, float, bool)) else str(value))

    if invocation_params:
        try:
            span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMS, json.dumps(invocation_params))
        except (TypeError, ValueError):
            pass


def _annotate_llm_prompts(
    span: Span,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    max_length: int,
) -> None:
    """Annotate span with LLM prompts/messages."""
    # Try common parameter names for prompts
    prompt_params = ["messages", "prompt", "input", "text"]

    for param_name in prompt_params:
        value = _extract_param(func, args, kwargs, param_name)
        if value is not None:
            try:
                if isinstance(value, str):
                    prompt_str = value[:max_length]
                else:
                    prompt_str = json.dumps(value)[:max_length]

                span.set_attribute(SpanAttributes.LLM_PROMPTS, prompt_str)
                span.set_attribute(SpanAttributes.INPUT_VALUE, prompt_str)
                span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
            except (TypeError, ValueError):
                span.set_attribute(SpanAttributes.INPUT_VALUE, str(value)[:max_length])
            break


def _annotate_llm_completion(
    span: Span,
    result: Any,
    max_length: int,
) -> None:
    """Annotate span with LLM completion/response."""
    try:
        # Handle various response formats
        completion_text = None

        if isinstance(result, str):
            completion_text = result
        elif isinstance(result, dict):
            # OpenAI-style response
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice:
                    completion_text = choice["message"].get("content", "")
                elif "text" in choice:
                    completion_text = choice["text"]
            elif "content" in result:
                completion_text = result["content"]
            elif "text" in result:
                completion_text = result["text"]
        elif hasattr(result, "choices") and result.choices:
            # Pydantic model response
            choice = result.choices[0]
            if hasattr(choice, "message"):
                completion_text = getattr(choice.message, "content", "")
            elif hasattr(choice, "text"):
                completion_text = choice.text
        elif hasattr(result, "content"):
            completion_text = result.content

        if completion_text:
            truncated = completion_text[:max_length]
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, truncated)
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")

    except Exception as e:
        logger.debug(f"Failed to extract completion: {e}")


def _extract_and_log_tokens(span: Span, result: Any) -> None:
    """Extract and log token usage from LLM response."""
    try:
        usage = None

        if isinstance(result, dict):
            usage = result.get("usage")
        elif hasattr(result, "usage"):
            usage = result.usage

        if usage:
            prompt_tokens = 0
            completion_tokens = 0

            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)

            log_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                span=span,
            )

    except Exception as e:
        logger.debug(f"Failed to extract token usage: {e}")


# =============================================================================
# TraceContext - Custom Span Context Manager
# =============================================================================


@dataclass
class TraceContext:
    """
    Context manager for creating custom spans with nesting support.

    Provides a flexible way to create spans with automatic error recording
    and attribute management. Supports both sync and async contexts.

    Attributes:
        name: Span name.
        span_kind: OpenTelemetry span kind (INTERNAL, CLIENT, SERVER, etc.).
        attributes: Initial span attributes.
        record_exceptions: Whether to automatically record exceptions.
        set_status_on_error: Whether to set ERROR status on exceptions.

    Example:
        >>> async with TraceContext("process_documents", attributes={"doc_count": 10}) as ctx:
        ...     ctx.add_attribute("processed", 5)
        ...     result = await process_batch()
        ...     ctx.add_attribute("processed", 10)

        >>> # Nested contexts
        >>> with TraceContext("outer") as outer:
        ...     with TraceContext("inner", parent=outer.span) as inner:
        ...         # Inner span is child of outer
        ...         pass
    """

    name: str
    span_kind: SpanKind = SpanKind.INTERNAL
    attributes: dict[str, Any] = field(default_factory=dict)
    record_exceptions: bool = True
    set_status_on_error: bool = True
    parent: Span | None = None

    # Internal state
    _span: Span | None = field(default=None, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _tracer_name: str = field(default="drx.custom", init=False, repr=False)

    @property
    def span(self) -> Span | None:
        """Get the current span."""
        return self._span

    def add_attribute(self, key: str, value: Any) -> None:
        """Add an attribute to the current span."""
        if self._span and self._span.is_recording():
            if isinstance(value, (str, int, float, bool)):
                self._span.set_attribute(key, value)
            else:
                try:
                    self._span.set_attribute(key, json.dumps(value))
                except (TypeError, ValueError):
                    self._span.set_attribute(key, str(value))

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to the current span."""
        if self._span:
            self._span.add_event(name, attributes=attributes)

    def set_status(self, status: StatusCode, description: str | None = None) -> None:
        """Set the span status."""
        if self._span:
            self._span.set_status(Status(status, description))

    def record_exception(self, exception: BaseException) -> None:
        """Record an exception on the span."""
        if self._span:
            self._span.record_exception(exception)
            if self.set_status_on_error:
                self._span.set_status(Status(StatusCode.ERROR, str(exception)))

    def __enter__(self) -> "TraceContext":
        """Enter sync context."""
        return self._start_span()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit sync context."""
        self._end_span(exc_val)

    async def __aenter__(self) -> "TraceContext":
        """Enter async context."""
        return self._start_span()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context."""
        self._end_span(exc_val)

    def _start_span(self) -> "TraceContext":
        """Start the span and set initial attributes."""
        tracer = get_tracer(self._tracer_name)

        # Set up context with parent if provided
        context = None
        if self.parent is not None:
            context = trace.set_span_in_context(self.parent)

        # Start the span
        self._span = tracer.start_span(
            name=self.name,
            kind=self.span_kind,
            context=context,
        )

        # Make it the current span
        trace.use_span(self._span, end_on_exit=False).__enter__()

        self._start_time = time.perf_counter()

        # Set initial attributes
        for key, value in self.attributes.items():
            self.add_attribute(key, value)

        self._span.set_attribute(SpanAttributes.START_TIME, datetime.utcnow().isoformat())

        return self

    def _end_span(self, exception: BaseException | None) -> None:
        """End the span and record final state."""
        if self._span is None:
            return

        # Record duration
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        self._span.set_attribute(SpanAttributes.LATENCY_MS, duration_ms)
        self._span.set_attribute(SpanAttributes.END_TIME, datetime.utcnow().isoformat())

        # Handle exceptions
        if exception is not None:
            if self.record_exceptions:
                self._span.record_exception(exception)
            if self.set_status_on_error:
                self._span.set_status(Status(StatusCode.ERROR, str(exception)))
        else:
            self._span.set_status(Status(StatusCode.OK))

        # End the span
        self._span.end()


# =============================================================================
# Convenience Context Managers
# =============================================================================


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    span_kind: SpanKind = SpanKind.INTERNAL,
) -> Generator[Span, None, None]:
    """
    Simple context manager for creating a traced span.

    Args:
        name: Span name.
        attributes: Optional initial attributes.
        span_kind: OpenTelemetry span kind.

    Yields:
        The created span.

    Example:
        >>> with trace_span("my_operation", {"key": "value"}) as span:
        ...     span.set_attribute("result", "success")
    """
    tracer = get_tracer("drx.custom")

    with tracer.start_as_current_span(name, kind=span_kind) as span:
        if attributes:
            for key, value in attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
                else:
                    try:
                        span.set_attribute(key, json.dumps(value))
                    except (TypeError, ValueError):
                        span.set_attribute(key, str(value))
        yield span


@asynccontextmanager
async def async_trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    span_kind: SpanKind = SpanKind.INTERNAL,
) -> AsyncGenerator[Span, None]:
    """
    Async context manager for creating a traced span.

    Args:
        name: Span name.
        attributes: Optional initial attributes.
        span_kind: OpenTelemetry span kind.

    Yields:
        The created span.

    Example:
        >>> async with async_trace_span("async_operation") as span:
        ...     result = await some_async_work()
        ...     span.set_attribute("result", result)
    """
    tracer = get_tracer("drx.custom")

    with tracer.start_as_current_span(name, kind=span_kind) as span:
        if attributes:
            for key, value in attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
                else:
                    try:
                        span.set_attribute(key, json.dumps(value))
                    except (TypeError, ValueError):
                        span.set_attribute(key, str(value))
        yield span


# =============================================================================
# Evaluation Metric Logging
# =============================================================================


def log_evaluation_metric(
    metric_name: str,
    score: float,
    threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
    span: Span | None = None,
) -> None:
    """
    Log an evaluation metric to the current or specified span.

    This is a convenience re-export from phoenix module for use
    in tracing contexts.

    Args:
        metric_name: Name of the metric (e.g., "faithfulness").
        score: Metric score value.
        threshold: Optional threshold for pass/fail determination.
        metadata: Additional metadata for the metric.
        span: Target span. Uses current span if None.

    Example:
        >>> with trace_span("evaluate_response") as span:
        ...     score = compute_faithfulness(response, context)
        ...     log_evaluation_metric("faithfulness", score, threshold=0.7)
    """
    from src.observability.phoenix import log_evaluation_metric as phoenix_log_metric

    phoenix_log_metric(
        metric_name=metric_name,
        score=score,
        threshold=threshold,
        metadata=metadata,
        span=span,
    )


# =============================================================================
# Workflow Tracing Utilities
# =============================================================================


def trace_workflow_node(
    node_name: str,
    workflow_name: str = "research_workflow",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing LangGraph workflow nodes.

    Optimized for LangGraph node functions that receive and return state.

    Args:
        node_name: Name of the workflow node.
        workflow_name: Name of the parent workflow.

    Returns:
        Decorated function.

    Example:
        >>> @trace_workflow_node("plan_node")
        ... async def plan_node(state: AgentState) -> dict:
        ...     # Node logic
        ...     return {"plan": new_plan}
    """
    return trace_agent(
        agent_name=node_name,
        agent_type=_infer_agent_type(node_name),
        include_state_fields=[
            "session_id",
            "current_phase",
            "iteration_count",
            "tokens_used",
            "next_node",
        ],
    )


def trace_conditional_edge(
    edge_name: str,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing LangGraph conditional edge functions.

    Args:
        edge_name: Name of the conditional edge.

    Returns:
        Decorated function.

    Example:
        >>> @trace_conditional_edge("should_continue")
        ... def should_continue(state: AgentState) -> str:
        ...     if state["should_terminate"]:
        ...         return "end"
        ...     return state["next_node"] or "continue"
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer("drx.workflow")

            with tracer.start_as_current_span(f"edge.{edge_name}") as span:
                span.set_attribute("openinference.span.kind", "CHAIN")
                span.set_attribute("drx.edge.name", edge_name)

                result = func(*args, **kwargs)

                # Log the routing decision
                if isinstance(result, str):
                    span.set_attribute("drx.edge.decision", result)

                return result

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
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
    # Utilities
    "log_evaluation_metric",
]
