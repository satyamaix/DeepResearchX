"""
Base tool classes and utilities for DRX Deep Research system.

Provides:
- BaseTool: Abstract base class for all tools
- ToolResult: Standardized result container
- SearchResult: Web search result dataclass
- tool_with_retry: Decorator for retry logic with exponential backoff
- RateLimiter: Token bucket rate limiter
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar, ParamSpec, Generic

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class ToolStatus(Enum):
    """Status codes for tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """
    Standardized result container for tool operations.

    Attributes:
        success: Whether the operation completed successfully
        data: The result data (type depends on tool)
        error: Error message if success is False
        metadata: Additional tracing/debugging information
    """
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Add default metadata fields."""
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.utcnow().isoformat()
        if "trace_id" not in self.metadata:
            self.metadata["trace_id"] = str(uuid.uuid4())

    @classmethod
    def ok(cls, data: Any, **metadata) -> ToolResult:
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> ToolResult:
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)

    @classmethod
    def rate_limited(cls, retry_after: float | None = None) -> ToolResult:
        """Create a rate-limited result."""
        return cls(
            success=False,
            error="Rate limit exceeded",
            metadata={
                "status": ToolStatus.RATE_LIMITED.value,
                "retry_after": retry_after
            }
        )


@dataclass
class SearchResult:
    """
    Standardized web search result.

    Attributes:
        url: Source URL of the result
        title: Title of the page/document
        snippet: Text excerpt or summary
        score: Relevance score (0.0 to 1.0, higher is better)
        metadata: Additional source-specific metadata
    """
    url: str
    title: str
    snippet: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize fields."""
        # Clamp score to [0, 1]
        self.score = max(0.0, min(1.0, self.score))
        # Ensure URL is stripped
        self.url = self.url.strip()
        # Add retrieval timestamp if not present
        if "retrieved_at" not in self.metadata:
            self.metadata["retrieved_at"] = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        """Create from dictionary."""
        return cls(
            url=data.get("url", ""),
            title=data.get("title", ""),
            snippet=data.get("snippet", ""),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {})
        )


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Supports both per-second and per-period (e.g., monthly) limits.
    Thread-safe for asyncio operations.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int | None = None,
        max_requests_per_period: int | None = None,
        period_seconds: float = 2592000.0  # 30 days default
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Sustained request rate
            burst_size: Maximum burst size (defaults to requests_per_second)
            max_requests_per_period: Hard limit for period (e.g., 1000/month)
            period_seconds: Length of period in seconds
        """
        self.rate = requests_per_second
        self.burst_size = burst_size or int(requests_per_second)
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

        # Period-based limiting
        self.max_requests_per_period = max_requests_per_period
        self.period_seconds = period_seconds
        self.period_start = time.monotonic()
        self.period_requests = 0

    async def acquire(self, tokens: int = 1) -> tuple[bool, float | None]:
        """
        Attempt to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Tuple of (success, wait_time_if_failed)
        """
        async with self._lock:
            now = time.monotonic()

            # Check period limit
            if self.max_requests_per_period is not None:
                # Reset period if elapsed
                if now - self.period_start >= self.period_seconds:
                    self.period_start = now
                    self.period_requests = 0

                # Check if period limit exceeded
                if self.period_requests >= self.max_requests_per_period:
                    wait_time = self.period_seconds - (now - self.period_start)
                    return False, wait_time

            # Refill tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.period_requests += tokens
                return True, None

            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.rate
            return False, wait_time

    async def wait_and_acquire(self, tokens: int = 1, max_wait: float = 30.0) -> bool:
        """
        Wait until tokens are available and acquire them.

        Args:
            tokens: Number of tokens to acquire
            max_wait: Maximum time to wait in seconds

        Returns:
            True if acquired, False if max_wait exceeded
        """
        total_waited = 0.0

        while total_waited < max_wait:
            success, wait_time = await self.acquire(tokens)
            if success:
                return True

            if wait_time is None or wait_time > max_wait - total_waited:
                return False

            await asyncio.sleep(min(wait_time, max_wait - total_waited))
            total_waited += wait_time

        return False

    @property
    def remaining_in_period(self) -> int | None:
        """Get remaining requests in current period."""
        if self.max_requests_per_period is None:
            return None
        return max(0, self.max_requests_per_period - self.period_requests)


def tool_with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Exception types that trigger retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Calculate delay with exponential backoff + jitter
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        # Add jitter (10-30% of delay)
                        import random
                        jitter = delay * (0.1 + 0.2 * random.random())
                        actual_delay = delay + jitter

                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {actual_delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(actual_delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries failed for {func.__name__}: {e}"
                        )

            # Re-raise last exception if all retries failed
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected state in retry wrapper for {func.__name__}")

        return wrapper
    return decorator


class BaseTool(ABC):
    """
    Abstract base class for all DRX tools.

    Provides:
    - Consistent interface for tool invocation
    - Rate limiting support
    - Logging and tracing hooks
    - Error handling
    """

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 30.0
    ):
        """
        Initialize base tool.

        Args:
            rate_limiter: Optional rate limiter instance
            timeout: Default timeout for operations in seconds
        """
        self._rate_limiter = rate_limiter
        self._timeout = timeout
        self._invocation_count = 0
        self._error_count = 0
        self._total_latency = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for identification and logging."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM context."""
        pass

    @property
    def schema(self) -> dict[str, Any]:
        """
        JSON schema for tool input.
        Override in subclasses for custom input validation.
        """
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input query or command"
                }
            },
            "required": ["input"]
        }

    async def invoke(self, input: str, **kwargs) -> ToolResult:
        """
        Main entry point for tool invocation.

        Handles rate limiting, timing, and error handling.
        Subclasses should override _execute() instead.

        Args:
            input: The input query or command
            **kwargs: Additional tool-specific parameters

        Returns:
            ToolResult with success status and data or error
        """
        start_time = time.monotonic()
        self._invocation_count += 1

        trace_id = str(uuid.uuid4())
        logger.info(f"[{trace_id}] {self.name} invoked with input: {input[:100]}...")

        try:
            # Check rate limit
            if self._rate_limiter:
                acquired = await self._rate_limiter.wait_and_acquire(
                    max_wait=self._timeout / 2
                )
                if not acquired:
                    logger.warning(f"[{trace_id}] {self.name} rate limited")
                    return ToolResult.rate_limited()

            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute(input, **kwargs),
                timeout=self._timeout
            )

            # Add tracing metadata
            latency = time.monotonic() - start_time
            self._total_latency += latency
            result.metadata["trace_id"] = trace_id
            result.metadata["latency_ms"] = int(latency * 1000)
            result.metadata["tool_name"] = self.name

            logger.info(
                f"[{trace_id}] {self.name} completed in {latency:.3f}s, "
                f"success={result.success}"
            )

            return result

        except asyncio.TimeoutError:
            self._error_count += 1
            logger.error(f"[{trace_id}] {self.name} timed out after {self._timeout}s")
            return ToolResult(
                success=False,
                error=f"Operation timed out after {self._timeout}s",
                metadata={
                    "trace_id": trace_id,
                    "status": ToolStatus.TIMEOUT.value
                }
            )
        except Exception as e:
            self._error_count += 1
            logger.exception(f"[{trace_id}] {self.name} error: {e}")
            return ToolResult.fail(
                str(e),
                trace_id=trace_id,
                exception_type=type(e).__name__
            )

    @abstractmethod
    async def _execute(self, input: str, **kwargs) -> ToolResult:
        """
        Execute the tool operation.

        Subclasses must implement this method.

        Args:
            input: The input query or command
            **kwargs: Additional tool-specific parameters

        Returns:
            ToolResult with operation outcome
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get tool usage statistics."""
        avg_latency = (
            self._total_latency / self._invocation_count
            if self._invocation_count > 0 else 0
        )
        return {
            "name": self.name,
            "invocations": self._invocation_count,
            "errors": self._error_count,
            "error_rate": (
                self._error_count / self._invocation_count
                if self._invocation_count > 0 else 0
            ),
            "avg_latency_ms": int(avg_latency * 1000),
            "rate_limiter_remaining": (
                self._rate_limiter.remaining_in_period
                if self._rate_limiter else None
            )
        }

    def to_langchain_tool(self) -> dict[str, Any]:
        """
        Convert to LangChain tool format.

        Returns:
            Dict compatible with LangChain tool schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.schema,
            "function": self.invoke
        }


class SearchTool(BaseTool):
    """
    Base class for search-based tools.

    Provides common functionality for web search tools.
    """

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 30.0,
        default_max_results: int = 5
    ):
        super().__init__(rate_limiter=rate_limiter, timeout=timeout)
        self.default_max_results = default_max_results

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": self.default_max_results
                }
            },
            "required": ["query"]
        }

    async def search(
        self,
        query: str,
        max_results: int | None = None
    ) -> list[SearchResult]:
        """
        Execute a search query.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        max_results = max_results or self.default_max_results
        result = await self.invoke(query, max_results=max_results)

        if result.success and isinstance(result.data, list):
            return result.data
        return []

    async def _execute(self, input: str, **kwargs) -> ToolResult:
        """Execute search - delegates to _search method."""
        max_results = kwargs.get("max_results", self.default_max_results)
        try:
            results = await self._search(input, max_results)
            return ToolResult.ok(
                results,
                result_count=len(results),
                query=input
            )
        except Exception as e:
            return ToolResult.fail(str(e), query=input)

    @abstractmethod
    async def _search(
        self,
        query: str,
        max_results: int
    ) -> list[SearchResult]:
        """
        Execute the actual search.

        Subclasses must implement this method.
        """
        pass


# Type aliases for convenience
ToolFactory = Callable[[], BaseTool]
SearchResults = list[SearchResult]
