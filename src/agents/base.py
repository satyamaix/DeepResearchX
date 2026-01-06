"""
Base Agent Class for DRX Deep Research System.

Provides:
- BaseAgent: Abstract base class for all research agents
- AgentResponse: Structured response container
- Common utilities for LLM interaction, token tracking, and state management
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import get_settings

if TYPE_CHECKING:
    from ..orchestrator.state import AgentState, AgentType

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Response Dataclass
# =============================================================================


@dataclass
class AgentResponse:
    """
    Structured response container for agent outputs.

    Provides a standardized format for all agent responses,
    enabling consistent handling across the workflow.
    """

    # Whether the agent completed successfully
    success: bool

    # The agent's output data (type varies by agent)
    data: Any = None

    # Error message if success is False
    error: str | None = None

    # Agent that produced this response
    agent_name: str = ""

    # Token usage for this invocation
    tokens_used: int = 0

    # Prompt tokens consumed
    prompt_tokens: int = 0

    # Completion tokens generated
    completion_tokens: int = 0

    # Execution time in milliseconds
    latency_ms: int = 0

    # Model used for this invocation
    model: str = ""

    # Unique trace identifier for debugging
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Timestamp of completion
    completed_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_response(
        cls,
        data: Any,
        agent_name: str,
        tokens_used: int = 0,
        **kwargs,
    ) -> AgentResponse:
        """Create a successful response."""
        return cls(
            success=True,
            data=data,
            agent_name=agent_name,
            tokens_used=tokens_used,
            **kwargs,
        )

    @classmethod
    def error_response(
        cls,
        error: str,
        agent_name: str,
        **kwargs,
    ) -> AgentResponse:
        """Create an error response."""
        return cls(
            success=False,
            error=error,
            agent_name=agent_name,
            **kwargs,
        )


# =============================================================================
# LLM Client Interface (Abstract)
# =============================================================================


class LLMClient(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (uses default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            Dict containing 'content', 'usage', and metadata
        """
        pass


# =============================================================================
# Base Agent Abstract Class
# =============================================================================


class BaseAgent(ABC):
    """
    Abstract base class for all DRX research agents.

    Provides:
    - Common LLM interaction patterns
    - Token tracking and budget management
    - State update utilities
    - Logging and tracing
    - Error handling

    Subclasses must implement:
    - name: Agent identifier
    - description: Human-readable description
    - system_prompt: The system prompt for LLM interactions
    - _process(): Core agent logic
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
    ):
        """
        Initialize base agent.

        Args:
            llm_client: LLM client for API calls (required for LLM-based agents)
            model: Model identifier (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            max_output_tokens: Maximum tokens to generate per request
        """
        self._llm_client = llm_client
        self._settings = get_settings()

        self._model = model or self._settings.DEFAULT_MODEL
        self._temperature = temperature or self._settings.DEFAULT_TEMPERATURE
        self._max_output_tokens = max_output_tokens

        # Statistics tracking
        self._invocation_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0
        self._error_count = 0

    # =========================================================================
    # Abstract Properties (MUST be implemented by subclasses)
    # =========================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this agent.

        Used for logging, tracing, and agent routing.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of the agent's purpose.

        Used for documentation and LLM context.
        """
        pass

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """
        The agent type enum value.

        Used for state tracking and routing.
        """
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        The system prompt for LLM interactions.

        Defines the agent's role, capabilities, and output format.
        """
        pass

    # =========================================================================
    # Optional Overridable Properties
    # =========================================================================

    @property
    def model(self) -> str:
        """Model identifier for this agent."""
        return self._model

    @property
    def temperature(self) -> float:
        """Sampling temperature for this agent."""
        return self._temperature

    # =========================================================================
    # Public Interface
    # =========================================================================

    async def invoke(self, state: AgentState) -> AgentState:
        """
        Main entry point for agent invocation.

        Handles:
        - Token budget checking
        - Pre/post processing hooks
        - Error handling and logging
        - State updates

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        start_time = time.monotonic()
        trace_id = str(uuid.uuid4())

        logger.info(
            f"[{trace_id}] {self.name} invoked for session {state['session_id']}"
        )

        self._invocation_count += 1

        try:
            # Check token budget
            if not self._check_token_budget(state):
                logger.warning(f"[{trace_id}] Token budget exhausted")
                return self._update_state_with_error(
                    state, "Token budget exhausted", trace_id
                )

            # Pre-processing hook
            state = await self._pre_process(state)

            # Main processing
            response = await self._process(state)

            # Track tokens
            if response.tokens_used > 0:
                self._total_tokens += response.tokens_used
                state = self._update_token_usage(state, response.tokens_used)

            # Post-processing hook
            state = await self._post_process(state, response)

            # Calculate latency
            latency_ms = int((time.monotonic() - start_time) * 1000)
            self._total_latency_ms += latency_ms

            logger.info(
                f"[{trace_id}] {self.name} completed in {latency_ms}ms, "
                f"tokens={response.tokens_used}, success={response.success}"
            )

            return state

        except Exception as e:
            self._error_count += 1
            latency_ms = int((time.monotonic() - start_time) * 1000)
            self._total_latency_ms += latency_ms

            logger.exception(f"[{trace_id}] {self.name} error: {e}")
            return self._update_state_with_error(state, str(e), trace_id)

    # =========================================================================
    # Abstract Method (MUST be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    async def _process(self, state: AgentState) -> AgentResponse:
        """
        Core agent logic.

        Subclasses must implement this method to perform their
        specific processing and return an AgentResponse.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with results
        """
        pass

    # =========================================================================
    # Overridable Hooks
    # =========================================================================

    async def _pre_process(self, state: AgentState) -> AgentState:
        """
        Pre-processing hook called before _process().

        Override to add validation, transformation, or setup logic.
        Default implementation returns state unchanged.

        Args:
            state: Current workflow state

        Returns:
            Potentially modified state
        """
        return state

    async def _post_process(
        self, state: AgentState, response: AgentResponse
    ) -> AgentState:
        """
        Post-processing hook called after _process().

        Override to add state updates based on response.
        Default implementation returns state unchanged.

        Args:
            state: Current workflow state
            response: Agent response from _process()

        Returns:
            Updated workflow state
        """
        return state

    # =========================================================================
    # LLM Interaction Helpers
    # =========================================================================

    async def _call_llm(
        self,
        user_message: str,
        system_override: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> AgentResponse:
        """
        Call the LLM with the agent's system prompt.

        Args:
            user_message: The user/input message content
            system_override: Override the default system prompt
            temperature: Override the default temperature
            max_tokens: Override the default max tokens
            response_format: JSON schema for structured output

        Returns:
            AgentResponse with LLM response or error
        """
        if self._llm_client is None:
            return AgentResponse.error_response(
                "LLM client not configured",
                self.name,
            )

        messages = [
            {"role": "system", "content": system_override or self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            start_time = time.monotonic()

            kwargs = {}
            if response_format:
                kwargs["response_format"] = response_format

            result = await self._llm_client.chat_completion(
                messages=messages,
                model=self._model,
                temperature=temperature or self._temperature,
                max_tokens=max_tokens or self._max_output_tokens,
                **kwargs,
            )

            latency_ms = int((time.monotonic() - start_time) * 1000)

            # Extract usage information
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            return AgentResponse.success_response(
                data=result.get("content", ""),
                agent_name=self.name,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                model=self._model,
            )

        except Exception as e:
            logger.exception(f"{self.name} LLM call failed: {e}")
            return AgentResponse.error_response(str(e), self.name)

    async def _call_llm_with_history(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AgentResponse:
        """
        Call the LLM with a full message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override the default temperature
            max_tokens: Override the default max tokens

        Returns:
            AgentResponse with LLM response or error
        """
        if self._llm_client is None:
            return AgentResponse.error_response(
                "LLM client not configured",
                self.name,
            )

        # Ensure system message is first
        if not messages or messages[0].get("role") != "system":
            messages = [
                {"role": "system", "content": self.system_prompt},
                *messages,
            ]

        try:
            start_time = time.monotonic()

            result = await self._llm_client.chat_completion(
                messages=messages,
                model=self._model,
                temperature=temperature or self._temperature,
                max_tokens=max_tokens or self._max_output_tokens,
            )

            latency_ms = int((time.monotonic() - start_time) * 1000)

            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            return AgentResponse.success_response(
                data=result.get("content", ""),
                agent_name=self.name,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                model=self._model,
            )

        except Exception as e:
            logger.exception(f"{self.name} LLM call failed: {e}")
            return AgentResponse.error_response(str(e), self.name)

    # =========================================================================
    # State Management Helpers
    # =========================================================================

    def _check_token_budget(self, state: AgentState) -> bool:
        """
        Check if there's sufficient token budget remaining.

        Args:
            state: Current workflow state

        Returns:
            True if budget allows execution, False otherwise
        """
        # Reserve 10% buffer for safety
        buffer = int(state["token_budget"] * 0.1)
        return state["tokens_remaining"] > buffer

    def _update_token_usage(self, state: AgentState, tokens: int) -> AgentState:
        """
        Update token usage in state.

        Args:
            state: Current workflow state
            tokens: Tokens consumed

        Returns:
            State with updated token counts
        """
        new_used = state["tokens_used"] + tokens
        new_remaining = max(0, state["token_budget"] - new_used)

        return {
            **state,
            "tokens_used": new_used,
            "tokens_remaining": new_remaining,
        }

    def _update_state_with_error(
        self,
        state: AgentState,
        error: str,
        trace_id: str,
    ) -> AgentState:
        """
        Update state with error information.

        Args:
            state: Current workflow state
            error: Error message
            trace_id: Trace identifier

        Returns:
            State with error information
        """
        return {
            **state,
            "error": f"[{self.name}] {error}",
            "current_phase": "failed",
            "should_terminate": True,
        }

    # =========================================================================
    # Message Conversion Helpers
    # =========================================================================

    def _state_messages_to_dicts(
        self, state: AgentState
    ) -> list[dict[str, str]]:
        """
        Convert LangChain messages in state to dict format.

        Args:
            state: Current workflow state

        Returns:
            List of message dicts
        """
        result = []
        for msg in state.get("messages", []):
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, dict):
                result.append(msg)
        return result

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get agent usage statistics.

        Returns:
            Dict with invocation counts, tokens, latency, and error rate
        """
        avg_latency = (
            self._total_latency_ms / self._invocation_count
            if self._invocation_count > 0
            else 0
        )
        avg_tokens = (
            self._total_tokens / self._invocation_count
            if self._invocation_count > 0
            else 0
        )
        error_rate = (
            self._error_count / self._invocation_count
            if self._invocation_count > 0
            else 0
        )

        return {
            "agent_name": self.name,
            "agent_type": self.agent_type,
            "model": self._model,
            "invocations": self._invocation_count,
            "total_tokens": self._total_tokens,
            "avg_tokens_per_call": int(avg_tokens),
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": int(avg_latency),
            "errors": self._error_count,
            "error_rate": round(error_rate, 4),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, model={self._model})>"


# =============================================================================
# Utility Functions
# =============================================================================


def create_finding_id() -> str:
    """Generate a unique finding ID."""
    return f"finding_{uuid.uuid4().hex[:12]}"


def create_citation_id() -> str:
    """Generate a unique citation ID."""
    return f"cite_{uuid.uuid4().hex[:8]}"


def create_subtask_id(prefix: str = "task") -> str:
    """Generate a unique subtask ID."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def timestamp_now() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


# =============================================================================
# OpenRouter LLM Client Adapter
# =============================================================================


class OpenRouterLLMClient(LLMClient):
    """
    Adapter that wraps OpenRouterClient to implement the LLMClient interface.

    This bridges the gap between the abstract LLMClient expected by agents
    and the concrete OpenRouterClient implementation.
    """

    def __init__(self, openrouter_client: Any):
        """
        Initialize with an OpenRouterClient instance.

        Args:
            openrouter_client: An initialized OpenRouterClient
        """
        self._client = openrouter_client

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute a chat completion request with Phoenix tracing.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (uses default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            Dict containing 'content', 'usage', and metadata
        """
        # Import tracing utilities
        try:
            from src.observability import get_tracer, log_token_usage
            tracer = get_tracer("drx.llm")
        except Exception:
            tracer = None

        # Create span for LLM call if tracing is available
        if tracer:
            with tracer.start_as_current_span("llm.chat_completion") as span:
                # Set span attributes for OpenInference
                span.set_attribute("llm.model_name", model or "default")
                span.set_attribute("llm.invocation_parameters", str({
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }))
                span.set_attribute("openinference.span.kind", "LLM")

                # Log input messages (truncated for large inputs)
                if messages:
                    input_text = messages[-1].get("content", "")[:500]
                    span.set_attribute("llm.input_messages", str(messages[:3]))  # First 3 messages
                    span.set_attribute("input.value", input_text)

                try:
                    # Call the underlying OpenRouterClient
                    response = await self._client.chat_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                    # Log output and token usage
                    span.set_attribute("output.value", response.content[:1000] if response.content else "")
                    span.set_attribute("llm.model_name", response.model or model or "unknown")

                    if response.usage:
                        log_token_usage(
                            prompt_tokens=response.usage.get("prompt_tokens", 0),
                            completion_tokens=response.usage.get("completion_tokens", 0),
                            total_tokens=response.usage.get("total_tokens", 0),
                            model=response.model,
                            span=span,
                        )

                    if response.latency_ms:
                        span.set_attribute("drx.latency_ms", response.latency_ms)

                    return {
                        "content": response.content,
                        "usage": response.usage,
                        "model": response.model,
                        "finish_reason": response.finish_reason,
                        "tool_calls": response.tool_calls,
                        "latency_ms": response.latency_ms,
                    }

                except Exception as e:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    raise
        else:
            # No tracing available, just call directly
            response = await self._client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return {
                "content": response.content,
                "usage": response.usage,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "tool_calls": response.tool_calls,
                "latency_ms": response.latency_ms,
            }


# Module-level LLM client singleton
_llm_client: OpenRouterLLMClient | None = None


async def get_llm_client() -> OpenRouterLLMClient:
    """
    Get or create the LLM client singleton.

    Returns:
        OpenRouterLLMClient: Configured LLM client adapter
    """
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    # Import here to avoid circular imports
    from src.services.openrouter_client import get_openrouter_client

    openrouter_client = await get_openrouter_client()
    _llm_client = OpenRouterLLMClient(openrouter_client)

    return _llm_client


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AgentResponse",
    "BaseAgent",
    "LLMClient",
    "OpenRouterLLMClient",
    "get_llm_client",
    "create_finding_id",
    "create_citation_id",
    "create_subtask_id",
    "timestamp_now",
]
