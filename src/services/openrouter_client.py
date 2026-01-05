"""OpenRouter API Client for DRX.

Provides async access to OpenRouter API for LLM completions with:
- Multiple model support (Gemini, DeepSeek, Claude, etc.)
- Retry logic with exponential backoff
- Streaming support for SSE
- Token counting and usage tracking
- Rate limit handling
- Web search plugin support
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Literal

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton
_openrouter_client: OpenRouterClient | None = None
_client_lock = asyncio.Lock()


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""

    pass


class OpenRouterRateLimitError(OpenRouterError):
    """Exception raised when rate limited."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class OpenRouterAuthError(OpenRouterError):
    """Exception raised for authentication errors."""

    pass


class OpenRouterAPIError(OpenRouterError):
    """Exception raised for API errors."""

    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class OpenRouterTimeoutError(OpenRouterError):
    """Exception raised when request times out."""

    pass


class MessageRole(str, Enum):
    """Valid message roles for chat completion."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMModelConfig:
    """Configuration for a specific LLM model."""

    model_id: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    context_length: int = 128000
    supports_tools: bool = True
    supports_vision: bool = False
    supports_json_mode: bool = True
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameters."""
        return {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


# Backwards compatibility alias
ModelConfig = LLMModelConfig


# Pre-configured model configurations
MODELS: dict[str, LLMModelConfig] = {
    "google/gemini-2.5-flash-preview": LLMModelConfig(
        model_id="google/gemini-2.5-flash-preview",
        max_tokens=65536,
        context_length=1000000,
        supports_vision=True,
        cost_per_million_input=0.15,
        cost_per_million_output=0.6,
    ),
    "google/gemini-2.5-pro-preview": LLMModelConfig(
        model_id="google/gemini-2.5-pro-preview",
        max_tokens=65536,
        context_length=1000000,
        supports_vision=True,
        cost_per_million_input=1.25,
        cost_per_million_output=10.0,
    ),
    "deepseek/deepseek-r1": LLMModelConfig(
        model_id="deepseek/deepseek-r1",
        max_tokens=64000,
        context_length=64000,
        temperature=0.6,
        supports_tools=False,
        cost_per_million_input=0.55,
        cost_per_million_output=2.19,
    ),
    "deepseek/deepseek-chat": LLMModelConfig(
        model_id="deepseek/deepseek-chat",
        max_tokens=64000,
        context_length=64000,
        cost_per_million_input=0.14,
        cost_per_million_output=0.28,
    ),
    "anthropic/claude-sonnet-4": LLMModelConfig(
        model_id="anthropic/claude-sonnet-4",
        max_tokens=8192,
        context_length=200000,
        supports_vision=True,
        cost_per_million_input=3.0,
        cost_per_million_output=15.0,
    ),
    "openai/gpt-4o": LLMModelConfig(
        model_id="openai/gpt-4o",
        max_tokens=16384,
        context_length=128000,
        supports_vision=True,
        cost_per_million_input=2.5,
        cost_per_million_output=10.0,
    ),
}


@dataclass
class ChatMessage:
    """Represents a chat message."""

    role: MessageRole | str
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        msg: dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
        }
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


@dataclass
class CompletionResponse:
    """Response from a chat completion request."""

    content: str
    model: str
    finish_reason: str | None
    usage: dict[str, int]
    raw_response: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] | None = None
    latency_ms: float = 0.0

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.usage.get("total_tokens", 0)


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""

    content: str
    finish_reason: str | None = None
    model: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None


class OpenRouterClient:
    """Async client for OpenRouter API.

    Provides methods for chat completions with retry logic,
    streaming support, and usage tracking.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_referer: str = "https://drx.local",
        app_name: str = "DRX Deep Research",
    ):
        """Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key. Uses config if not provided.
            base_url: API base URL. Uses config if not provided.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
            http_referer: HTTP referer for request attribution.
            app_name: Application name for request attribution.
        """
        settings = get_settings()

        self.api_key = api_key or settings.openrouter_api_key_value
        self.base_url = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.OPENROUTER_TIMEOUT
        self.max_retries = max_retries or settings.OPENROUTER_MAX_RETRIES
        self.http_referer = http_referer
        self.app_name = app_name

        self._client: httpx.AsyncClient | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

        # Usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0

    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.http_referer,
                    "X-Title": self.app_name,
                    "Content-Type": "application/json",
                },
                http2=True,
            )

            self._initialized = True
            logger.info("OpenRouter client initialized")

    async def _ensure_initialized(self) -> httpx.AsyncClient:
        """Ensure client is initialized and return it."""
        if not self._initialized or self._client is None:
            await self.initialize()
        return self._client  # type: ignore

    def _get_model_config(self, model: str) -> LLMModelConfig:
        """Get model configuration, creating default if not found."""
        if model in MODELS:
            return MODELS[model]
        return LLMModelConfig(model_id=model)

    def _build_request_body(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        plugins: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the request body for chat completion."""
        model_config = self._get_model_config(model)

        # Convert messages to dict format
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted_messages.append(msg.to_dict())
            else:
                formatted_messages.append(msg)

        body: dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens or model_config.max_tokens,
            "temperature": temperature if temperature is not None else model_config.temperature,
            "top_p": top_p if top_p is not None else model_config.top_p,
            "stream": stream,
        }

        # Add optional parameters
        if tools and model_config.supports_tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice

        if response_format and model_config.supports_json_mode:
            body["response_format"] = response_format

        if plugins:
            body["plugins"] = plugins

        # Add any extra kwargs
        for key, value in kwargs.items():
            if value is not None:
                body[key] = value

        return body

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error responses from the API."""
        status_code = response.status_code

        try:
            error_body = response.json()
            error_message = error_body.get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text

        if status_code == 401:
            raise OpenRouterAuthError(f"Authentication failed: {error_message}")

        if status_code == 429:
            retry_after = response.headers.get("retry-after")
            retry_seconds = float(retry_after) if retry_after else None
            raise OpenRouterRateLimitError(
                f"Rate limited: {error_message}",
                retry_after=retry_seconds,
            )

        if status_code >= 500:
            raise OpenRouterAPIError(
                f"Server error: {error_message}",
                status_code=status_code,
                response_body=response.text,
            )

        raise OpenRouterAPIError(
            f"API error: {error_message}",
            status_code=status_code,
            response_body=response.text,
        )

    async def chat_completion(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Make a chat completion request.

        Args:
            messages: List of chat messages.
            model: Model ID. Uses default from config if not provided.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            tools: List of tools for function calling.
            tool_choice: Tool selection strategy.
            response_format: Response format (e.g., JSON mode).
            **kwargs: Additional API parameters.

        Returns:
            CompletionResponse: The completion response.

        Raises:
            OpenRouterError: If the request fails.
        """
        client = await self._ensure_initialized()
        settings = get_settings()
        model = model or settings.DEFAULT_MODEL

        body = self._build_request_body(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            **kwargs,
        )

        start_time = time.time()

        # Retry logic with tenacity
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((OpenRouterRateLimitError, OpenRouterAPIError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await client.post("/chat/completions", json=body)

                    if response.status_code != 200:
                        self._handle_error_response(response)

                    data = response.json()
                    latency = (time.time() - start_time) * 1000

                    # Extract response data
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    usage = data.get("usage", {})

                    # Update usage tracking
                    self._total_input_tokens += usage.get("prompt_tokens", 0)
                    self._total_output_tokens += usage.get("completion_tokens", 0)
                    self._total_requests += 1

                    return CompletionResponse(
                        content=message.get("content", ""),
                        model=data.get("model", model),
                        finish_reason=choice.get("finish_reason"),
                        usage=usage,
                        raw_response=data,
                        tool_calls=message.get("tool_calls"),
                        latency_ms=latency,
                    )

                except httpx.TimeoutException as e:
                    logger.warning(f"Request timed out (attempt {attempt.retry_state.attempt_number})")
                    raise OpenRouterTimeoutError(f"Request timed out: {e}") from e

        # This should not be reached due to reraise=True
        raise OpenRouterAPIError("Request failed after all retries")

    async def chat_completion_with_search(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Make a chat completion request with web search plugin.

        Enables the OpenRouter web search plugin for grounding
        responses with real-time web data.

        Args:
            messages: List of chat messages.
            model: Model ID.
            **kwargs: Additional API parameters.

        Returns:
            CompletionResponse: The completion response with search results.
        """
        # Enable web search plugin
        plugins = [{"id": "web-search"}]

        return await self.chat_completion(
            messages=messages,
            model=model,
            plugins=plugins,
            **kwargs,
        )

    async def stream_chat_completion(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion response.

        Args:
            messages: List of chat messages.
            model: Model ID.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            tools: List of tools for function calling.
            tool_choice: Tool selection strategy.
            **kwargs: Additional API parameters.

        Yields:
            StreamChunk: Streaming response chunks.

        Raises:
            OpenRouterError: If the request fails.
        """
        client = await self._ensure_initialized()
        settings = get_settings()
        model = model or settings.DEFAULT_MODEL

        body = self._build_request_body(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((OpenRouterRateLimitError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                try:
                    async with client.stream("POST", "/chat/completions", json=body) as response:
                        if response.status_code != 200:
                            # Read full response for error handling
                            await response.aread()
                            self._handle_error_response(response)

                        accumulated_content = ""
                        accumulated_tool_calls: list[dict[str, Any]] = []

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue

                            data_str = line[6:]  # Remove "data: " prefix

                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})

                            content = delta.get("content", "")
                            if content:
                                accumulated_content += content

                            tool_calls = delta.get("tool_calls")

                            yield StreamChunk(
                                content=content,
                                finish_reason=choice.get("finish_reason"),
                                model=data.get("model"),
                                tool_calls=tool_calls,
                                usage=data.get("usage"),
                            )

                        # Update usage tracking from final chunk
                        # (OpenRouter includes usage in final streaming chunk)
                        self._total_requests += 1

                except httpx.TimeoutException as e:
                    logger.warning(f"Stream timed out (attempt {attempt.retry_state.attempt_number})")
                    raise OpenRouterTimeoutError(f"Stream timed out: {e}") from e

    async def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of available models from OpenRouter.

        Returns:
            list: Available model information.
        """
        client = await self._ensure_initialized()

        response = await client.get("/models")

        if response.status_code != 200:
            self._handle_error_response(response)

        data = response.json()
        return data.get("data", [])

    def get_usage_stats(self) -> dict[str, int]:
        """Get cumulative usage statistics.

        Returns:
            dict: Usage statistics including token counts.
        """
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "total_requests": self._total_requests,
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0

    async def health_check(self) -> dict[str, Any]:
        """Check API connectivity.

        Returns:
            dict: Health status.
        """
        try:
            client = await self._ensure_initialized()
            start = time.time()

            # Use models endpoint as health check
            response = await client.get("/models", timeout=10.0)
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "latency_ms": round(latency, 2),
                }

            return {
                "status": "degraded",
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._initialized = False
            logger.info("OpenRouter client closed")


# ==================== Token Counting Utilities ====================


def estimate_token_count(text: str, model: str | None = None) -> int:
    """Estimate token count for text.

    Uses a simple heuristic based on character count.
    For more accurate counts, use tiktoken with appropriate encoding.

    Args:
        text: Text to count tokens for.
        model: Model name (currently unused, for future tiktoken support).

    Returns:
        int: Estimated token count.
    """
    # Simple heuristic: ~4 characters per token on average
    # This is a rough estimate; actual tokenization varies by model
    return max(1, len(text) // 4)


def estimate_messages_tokens(
    messages: list[ChatMessage | dict[str, Any]],
    model: str | None = None,
) -> int:
    """Estimate token count for a list of messages.

    Args:
        messages: List of chat messages.
        model: Model name.

    Returns:
        int: Estimated total token count.
    """
    total = 0

    for msg in messages:
        if isinstance(msg, ChatMessage):
            content = msg.content
        else:
            content = msg.get("content", "")

        if isinstance(content, str):
            total += estimate_token_count(content, model)
        elif isinstance(content, list):
            # Multi-modal content
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += estimate_token_count(part["text"], model)

        # Add overhead for message structure
        total += 4  # Approximate overhead per message

    return total


# ==================== Singleton Factory ====================


async def get_openrouter_client() -> OpenRouterClient:
    """Get or create the OpenRouter client singleton.

    Returns:
        OpenRouterClient: Configured client instance.
    """
    global _openrouter_client

    if _openrouter_client is not None and _openrouter_client._initialized:
        return _openrouter_client

    async with _client_lock:
        if _openrouter_client is not None and _openrouter_client._initialized:
            return _openrouter_client

        _openrouter_client = OpenRouterClient()
        await _openrouter_client.initialize()

        return _openrouter_client


async def close_openrouter_client() -> None:
    """Close the singleton OpenRouter client."""
    global _openrouter_client

    if _openrouter_client is not None:
        await _openrouter_client.close()
        _openrouter_client = None


# ==================== Convenience Functions ====================


async def quick_completion(
    prompt: str,
    model: str | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> str:
    """Quick helper for simple completions.

    Args:
        prompt: User prompt.
        model: Model ID.
        system_prompt: Optional system prompt.
        **kwargs: Additional parameters.

    Returns:
        str: The completion text.
    """
    client = await get_openrouter_client()

    messages: list[dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    response = await client.chat_completion(messages=messages, model=model, **kwargs)

    return response.content
