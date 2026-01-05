"""Active State Redis Service for DRX Deep Research System.

Provides real-time agent health monitoring, metrics tracking, and circuit breaker
state management using Redis as the backing store.

This service implements the Agentic Metadata functionality from R10.3 of the DRX spec,
enabling:
- Per-agent invocation tracking with timestamps
- Token burn rate calculations over sliding windows
- Latency percentile computations (p50, p99)
- Error rate monitoring
- Circuit breaker state management
- Background metrics aggregation

Redis Key Patterns:
    drx:agent:{agent_id}:invocations - Sorted set: score=timestamp, value=json(InvocationRecord)
    drx:agent:{agent_id}:errors      - Sorted set: score=timestamp, value=error_type
    drx:agent:{agent_id}:health      - Hash: status, last_check, failure_count, etc.
    drx:agent:{agent_id}:metrics     - Hash: tokens_1m, tokens_5m, latency_p50, etc.
    drx:agent:{agent_id}:circuit     - String: closed|open|half_open
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

logger = logging.getLogger(__name__)


# =============================================================================
# TypedDict Definitions
# =============================================================================


class AgentMetrics(TypedDict):
    """Metrics for an agent over various time windows.

    Attributes:
        tokens_1m: Total tokens consumed in the last 1 minute
        tokens_5m: Total tokens consumed in the last 5 minutes
        requests_1m: Number of requests in the last 1 minute
        latency_p50: 50th percentile latency in milliseconds
        latency_p99: 99th percentile latency in milliseconds
        error_rate_5m: Error rate (0.0-1.0) over the last 5 minutes
        last_updated: ISO 8601 timestamp of last metrics update
    """
    tokens_1m: int
    tokens_5m: int
    requests_1m: int
    latency_p50: float
    latency_p99: float
    error_rate_5m: float
    last_updated: str


class AgentHealthStatus(TypedDict):
    """Complete health status for an agent.

    Attributes:
        agent_id: Unique identifier for the agent
        status: Overall health status (healthy, degraded, unhealthy)
        circuit_status: Circuit breaker state (closed, open, half_open)
        failure_count: Current consecutive failure count
        last_success: ISO timestamp of last successful invocation, or None
        last_failure: ISO timestamp of last failed invocation, or None
        metrics: Current agent metrics
    """
    agent_id: str
    status: Literal["healthy", "degraded", "unhealthy"]
    circuit_status: Literal["closed", "open", "half_open"]
    failure_count: int
    last_success: str | None
    last_failure: str | None
    metrics: AgentMetrics


class InvocationRecord(TypedDict):
    """Record of a single agent invocation.

    Attributes:
        timestamp: ISO 8601 timestamp of the invocation
        tokens_used: Number of tokens consumed
        latency_ms: Request latency in milliseconds
        success: Whether the invocation succeeded
        error_type: Type of error if failed, None if successful
    """
    timestamp: str
    tokens_used: int
    latency_ms: int
    success: bool
    error_type: str | None


class CircuitBreakerConfig(TypedDict):
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before circuit opens
        recovery_timeout_seconds: Time before attempting recovery
        half_open_max_calls: Maximum calls allowed in half-open state
        error_rate_threshold: Error rate threshold to trigger circuit open (0.0-1.0)
    """
    failure_threshold: int
    recovery_timeout_seconds: int
    half_open_max_calls: int
    error_rate_threshold: float


# =============================================================================
# Constants
# =============================================================================

# Key prefix for all DRX agent keys
KEY_PREFIX = "drx:agent"

# Default circuit breaker configuration
DEFAULT_CIRCUIT_CONFIG: CircuitBreakerConfig = {
    "failure_threshold": 5,
    "recovery_timeout_seconds": 60,
    "half_open_max_calls": 3,
    "error_rate_threshold": 0.5,
}

# Default retention period for invocation data (1 hour)
DEFAULT_RETENTION_SECONDS = 3600

# Time windows for metrics calculations
WINDOW_1M = 60
WINDOW_5M = 300


# =============================================================================
# Exceptions
# =============================================================================


class ActiveStateError(Exception):
    """Base exception for Active State service errors."""
    pass


class ActiveStateConnectionError(ActiveStateError):
    """Raised when Redis connection fails."""
    pass


class ActiveStateDataError(ActiveStateError):
    """Raised when data operations fail."""
    pass


# =============================================================================
# ActiveStateService Class
# =============================================================================


class ActiveStateService:
    """Service for managing agent active state in Redis.

    This service provides real-time tracking of agent invocations, health status,
    metrics, and circuit breaker state. It uses Redis sorted sets for time-series
    data (invocations, errors) and hashes for current state (health, metrics).

    Example:
        >>> service = ActiveStateService(redis_client)
        >>> await service.initialize()
        >>> await service.record_invocation("planner", tokens=1500, latency_ms=250, success=True)
        >>> health = await service.get_agent_health("planner")
        >>> print(health["status"])  # "healthy"

    Attributes:
        redis: The Redis client instance
        _initialized: Whether the service has been initialized
        _lock: Async lock for thread-safe initialization
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        circuit_config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize the Active State Service.

        Args:
            redis_client: An initialized async Redis client instance
            circuit_config: Optional circuit breaker configuration override
        """
        self.redis = redis_client
        self.circuit_config = circuit_config or DEFAULT_CIRCUIT_CONFIG
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the service and verify Redis connection.

        Raises:
            ActiveStateConnectionError: If Redis connection fails
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                await self.redis.ping()
                self._initialized = True
                logger.info("ActiveStateService initialized successfully")
            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise ActiveStateConnectionError(f"Redis connection failed: {e}") from e

    async def _ensure_initialized(self) -> None:
        """Ensure the service is initialized before operations."""
        if not self._initialized:
            await self.initialize()

    # =========================================================================
    # Redis Key Helpers
    # =========================================================================

    def _invocations_key(self, agent_id: str) -> str:
        """Get the Redis key for agent invocations sorted set.

        Args:
            agent_id: The agent identifier

        Returns:
            Redis key string: drx:agent:{agent_id}:invocations
        """
        return f"{KEY_PREFIX}:{agent_id}:invocations"

    def _errors_key(self, agent_id: str) -> str:
        """Get the Redis key for agent errors sorted set.

        Args:
            agent_id: The agent identifier

        Returns:
            Redis key string: drx:agent:{agent_id}:errors
        """
        return f"{KEY_PREFIX}:{agent_id}:errors"

    def _health_key(self, agent_id: str) -> str:
        """Get the Redis key for agent health hash.

        Args:
            agent_id: The agent identifier

        Returns:
            Redis key string: drx:agent:{agent_id}:health
        """
        return f"{KEY_PREFIX}:{agent_id}:health"

    def _metrics_key(self, agent_id: str) -> str:
        """Get the Redis key for agent metrics hash.

        Args:
            agent_id: The agent identifier

        Returns:
            Redis key string: drx:agent:{agent_id}:metrics
        """
        return f"{KEY_PREFIX}:{agent_id}:metrics"

    def _circuit_key(self, agent_id: str) -> str:
        """Get the Redis key for agent circuit breaker state.

        Args:
            agent_id: The agent identifier

        Returns:
            Redis key string: drx:agent:{agent_id}:circuit
        """
        return f"{KEY_PREFIX}:{agent_id}:circuit"

    def _circuit_opened_at_key(self, agent_id: str) -> str:
        """Get the Redis key for circuit breaker open timestamp.

        Args:
            agent_id: The agent identifier

        Returns:
            Redis key string: drx:agent:{agent_id}:circuit_opened_at
        """
        return f"{KEY_PREFIX}:{agent_id}:circuit_opened_at"

    def _half_open_count_key(self, agent_id: str) -> str:
        """Get the Redis key for half-open request count.

        Args:
            agent_id: The agent identifier

        Returns:
            Redis key string: drx:agent:{agent_id}:half_open_count
        """
        return f"{KEY_PREFIX}:{agent_id}:half_open_count"

    # =========================================================================
    # Invocation Recording
    # =========================================================================

    async def record_invocation(
        self,
        agent_id: str,
        tokens: int,
        latency_ms: int,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        """Record an agent invocation with its metrics.

        This method atomically records the invocation in a sorted set (for time
        queries) and updates the health hash with the latest timestamps.

        Args:
            agent_id: The agent identifier (e.g., "planner", "researcher")
            tokens: Number of tokens consumed by this invocation
            latency_ms: Request latency in milliseconds
            success: Whether the invocation succeeded
            error_type: Type of error if failed (e.g., "timeout", "rate_limit")

        Raises:
            ActiveStateDataError: If the recording operation fails
        """
        await self._ensure_initialized()

        now = time.time()
        timestamp_iso = datetime.now(timezone.utc).isoformat()

        # Create invocation record
        record: InvocationRecord = {
            "timestamp": timestamp_iso,
            "tokens_used": tokens,
            "latency_ms": latency_ms,
            "success": success,
            "error_type": error_type,
        }

        try:
            # Use pipeline for atomic operations
            async with self.redis.pipeline(transaction=True) as pipe:
                # Add invocation to sorted set (score = timestamp for range queries)
                invocations_key = self._invocations_key(agent_id)
                record_json = json.dumps(record)
                await pipe.zadd(invocations_key, {record_json: now})

                # Set expiration on invocations key
                await pipe.expire(invocations_key, DEFAULT_RETENTION_SECONDS)

                # Update health hash
                health_key = self._health_key(agent_id)
                health_updates: dict[str, str] = {
                    "last_check": timestamp_iso,
                }

                if success:
                    health_updates["last_success"] = timestamp_iso
                    # Reset failure count on success
                    await pipe.hset(health_key, mapping=health_updates)
                    await pipe.hdel(health_key, "failure_count")
                else:
                    health_updates["last_failure"] = timestamp_iso
                    await pipe.hset(health_key, mapping=health_updates)
                    # Increment failure count
                    await pipe.hincrby(health_key, "failure_count", 1)

                    # Record error in errors sorted set
                    errors_key = self._errors_key(agent_id)
                    error_value = error_type or "unknown"
                    await pipe.zadd(errors_key, {f"{now}:{error_value}": now})
                    await pipe.expire(errors_key, DEFAULT_RETENTION_SECONDS)

                # Execute pipeline
                await pipe.execute()

            logger.debug(
                f"Recorded invocation for agent {agent_id}: "
                f"tokens={tokens}, latency={latency_ms}ms, success={success}"
            )

        except redis.RedisError as e:
            logger.error(f"Failed to record invocation for {agent_id}: {e}")
            raise ActiveStateDataError(f"Failed to record invocation: {e}") from e

    async def get_invocations_in_window(
        self,
        agent_id: str,
        window_seconds: int,
    ) -> list[InvocationRecord]:
        """Get all invocations within a time window.

        Args:
            agent_id: The agent identifier
            window_seconds: Time window in seconds (from now)

        Returns:
            List of InvocationRecord objects within the window

        Raises:
            ActiveStateDataError: If the query fails
        """
        await self._ensure_initialized()

        now = time.time()
        window_start = now - window_seconds

        try:
            invocations_key = self._invocations_key(agent_id)
            # Get all records within the time window
            records_raw = await self.redis.zrangebyscore(
                invocations_key,
                min=window_start,
                max=now,
            )

            records: list[InvocationRecord] = []
            for record_json in records_raw:
                try:
                    record = json.loads(record_json)
                    records.append(record)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in invocations: {record_json}")
                    continue

            return records

        except redis.RedisError as e:
            logger.error(f"Failed to get invocations for {agent_id}: {e}")
            raise ActiveStateDataError(f"Failed to get invocations: {e}") from e

    # =========================================================================
    # Health and Metrics Methods
    # =========================================================================

    async def get_token_burn_rate(
        self,
        agent_id: str,
        window_seconds: int = WINDOW_1M,
    ) -> float:
        """Calculate token burn rate for an agent over a time window.

        Args:
            agent_id: The agent identifier
            window_seconds: Time window in seconds (default: 60)

        Returns:
            Tokens per second burn rate

        Raises:
            ActiveStateDataError: If the calculation fails
        """
        records = await self.get_invocations_in_window(agent_id, window_seconds)

        if not records:
            return 0.0

        total_tokens = sum(r["tokens_used"] for r in records)
        return total_tokens / window_seconds

    async def get_error_rate(
        self,
        agent_id: str,
        window_seconds: int = WINDOW_5M,
    ) -> float:
        """Calculate error rate for an agent over a time window.

        Args:
            agent_id: The agent identifier
            window_seconds: Time window in seconds (default: 300)

        Returns:
            Error rate as a float between 0.0 and 1.0

        Raises:
            ActiveStateDataError: If the calculation fails
        """
        records = await self.get_invocations_in_window(agent_id, window_seconds)

        if not records:
            return 0.0

        total = len(records)
        failures = sum(1 for r in records if not r["success"])

        return failures / total

    async def calculate_percentiles(
        self,
        agent_id: str,
        window_seconds: int = WINDOW_5M,
    ) -> tuple[float, float]:
        """Calculate latency percentiles (p50, p99) for an agent.

        Args:
            agent_id: The agent identifier
            window_seconds: Time window in seconds (default: 300)

        Returns:
            Tuple of (p50_latency_ms, p99_latency_ms)

        Raises:
            ActiveStateDataError: If the calculation fails
        """
        records = await self.get_invocations_in_window(agent_id, window_seconds)

        if not records:
            return (0.0, 0.0)

        latencies = sorted(r["latency_ms"] for r in records)
        n = len(latencies)

        # Calculate p50 (median)
        p50_idx = int(n * 0.5)
        p50 = float(latencies[min(p50_idx, n - 1)])

        # Calculate p99
        p99_idx = int(n * 0.99)
        p99 = float(latencies[min(p99_idx, n - 1)])

        return (p50, p99)

    async def get_metrics(self, agent_id: str) -> AgentMetrics:
        """Get current metrics for an agent.

        This method calculates real-time metrics from the invocation data.
        For cached metrics (from background aggregation), use get_cached_metrics().

        Args:
            agent_id: The agent identifier

        Returns:
            AgentMetrics with current values

        Raises:
            ActiveStateDataError: If metrics calculation fails
        """
        await self._ensure_initialized()

        try:
            # Get invocations for different windows
            records_1m = await self.get_invocations_in_window(agent_id, WINDOW_1M)
            records_5m = await self.get_invocations_in_window(agent_id, WINDOW_5M)

            # Calculate token counts
            tokens_1m = sum(r["tokens_used"] for r in records_1m)
            tokens_5m = sum(r["tokens_used"] for r in records_5m)

            # Calculate request count
            requests_1m = len(records_1m)

            # Calculate percentiles from 5m window
            p50, p99 = await self.calculate_percentiles(agent_id, WINDOW_5M)

            # Calculate error rate from 5m window
            error_rate = await self.get_error_rate(agent_id, WINDOW_5M)

            timestamp_iso = datetime.now(timezone.utc).isoformat()

            metrics: AgentMetrics = {
                "tokens_1m": tokens_1m,
                "tokens_5m": tokens_5m,
                "requests_1m": requests_1m,
                "latency_p50": p50,
                "latency_p99": p99,
                "error_rate_5m": error_rate,
                "last_updated": timestamp_iso,
            }

            return metrics

        except redis.RedisError as e:
            logger.error(f"Failed to get metrics for {agent_id}: {e}")
            raise ActiveStateDataError(f"Failed to get metrics: {e}") from e

    async def get_cached_metrics(self, agent_id: str) -> AgentMetrics | None:
        """Get cached metrics from Redis hash (set by background aggregation).

        Args:
            agent_id: The agent identifier

        Returns:
            AgentMetrics if cached, None if not available
        """
        await self._ensure_initialized()

        try:
            metrics_key = self._metrics_key(agent_id)
            data = await self.redis.hgetall(metrics_key)

            if not data:
                return None

            return AgentMetrics(
                tokens_1m=int(data.get("tokens_1m", 0)),
                tokens_5m=int(data.get("tokens_5m", 0)),
                requests_1m=int(data.get("requests_1m", 0)),
                latency_p50=float(data.get("latency_p50", 0.0)),
                latency_p99=float(data.get("latency_p99", 0.0)),
                error_rate_5m=float(data.get("error_rate_5m", 0.0)),
                last_updated=data.get("last_updated", ""),
            )

        except redis.RedisError as e:
            logger.warning(f"Failed to get cached metrics for {agent_id}: {e}")
            return None

    def _determine_health_status(
        self,
        error_rate: float,
        circuit_status: Literal["closed", "open", "half_open"],
        failure_count: int,
    ) -> Literal["healthy", "degraded", "unhealthy"]:
        """Determine overall health status based on metrics.

        Args:
            error_rate: Current error rate (0.0-1.0)
            circuit_status: Current circuit breaker state
            failure_count: Current consecutive failure count

        Returns:
            Health status string
        """
        # Unhealthy if circuit is open
        if circuit_status == "open":
            return "unhealthy"

        # Unhealthy if error rate is very high
        if error_rate > 0.5:
            return "unhealthy"

        # Degraded if circuit is half-open or error rate is elevated
        if circuit_status == "half_open" or error_rate > 0.1:
            return "degraded"

        # Degraded if there are recent failures
        if failure_count > 0:
            return "degraded"

        return "healthy"

    async def get_agent_health(self, agent_id: str) -> AgentHealthStatus:
        """Get complete health status for an agent.

        This provides a comprehensive view of agent health including:
        - Overall health status (healthy, degraded, unhealthy)
        - Circuit breaker state
        - Failure counts and timestamps
        - Current metrics

        Args:
            agent_id: The agent identifier

        Returns:
            AgentHealthStatus with all health information

        Raises:
            ActiveStateDataError: If health retrieval fails
        """
        await self._ensure_initialized()

        try:
            # Get health hash data
            health_key = self._health_key(agent_id)
            health_data = await self.redis.hgetall(health_key)

            # Get circuit status
            circuit_status = await self.get_circuit_status(agent_id)

            # Get failure count
            failure_count = int(health_data.get("failure_count", 0))

            # Get metrics
            metrics = await self.get_metrics(agent_id)

            # Determine overall status
            status = self._determine_health_status(
                metrics["error_rate_5m"],
                circuit_status,
                failure_count,
            )

            return AgentHealthStatus(
                agent_id=agent_id,
                status=status,
                circuit_status=circuit_status,
                failure_count=failure_count,
                last_success=health_data.get("last_success"),
                last_failure=health_data.get("last_failure"),
                metrics=metrics,
            )

        except redis.RedisError as e:
            logger.error(f"Failed to get health for {agent_id}: {e}")
            raise ActiveStateDataError(f"Failed to get health: {e}") from e

    # =========================================================================
    # Circuit Breaker State Management
    # =========================================================================

    async def get_circuit_status(
        self,
        agent_id: str,
    ) -> Literal["closed", "open", "half_open"]:
        """Get the current circuit breaker status for an agent.

        Circuit breaker states:
        - closed: Normal operation, requests flow through
        - open: Circuit is tripped, requests are rejected
        - half_open: Testing if service has recovered

        Args:
            agent_id: The agent identifier

        Returns:
            Circuit breaker status string

        Raises:
            ActiveStateDataError: If status retrieval fails
        """
        await self._ensure_initialized()

        try:
            circuit_key = self._circuit_key(agent_id)
            status = await self.redis.get(circuit_key)

            if status is None:
                return "closed"

            if status in ("closed", "open", "half_open"):
                return status  # type: ignore

            # Invalid status, default to closed
            logger.warning(f"Invalid circuit status for {agent_id}: {status}")
            return "closed"

        except redis.RedisError as e:
            logger.error(f"Failed to get circuit status for {agent_id}: {e}")
            raise ActiveStateDataError(f"Failed to get circuit status: {e}") from e

    async def set_circuit_status(
        self,
        agent_id: str,
        status: Literal["closed", "open", "half_open"],
    ) -> None:
        """Set the circuit breaker status for an agent.

        Args:
            agent_id: The agent identifier
            status: New circuit breaker status

        Raises:
            ActiveStateDataError: If status update fails
        """
        await self._ensure_initialized()

        try:
            circuit_key = self._circuit_key(agent_id)
            opened_at_key = self._circuit_opened_at_key(agent_id)
            half_open_key = self._half_open_count_key(agent_id)

            async with self.redis.pipeline(transaction=True) as pipe:
                await pipe.set(circuit_key, status)

                if status == "open":
                    # Record when circuit was opened for recovery timeout
                    await pipe.set(opened_at_key, str(time.time()))
                    logger.warning(f"Circuit breaker OPENED for agent: {agent_id}")
                elif status == "half_open":
                    # Reset half-open call counter
                    await pipe.delete(half_open_key)
                    logger.info(f"Circuit breaker HALF_OPEN for agent: {agent_id}")
                elif status == "closed":
                    # Clean up circuit breaker state
                    await pipe.delete(opened_at_key)
                    await pipe.delete(half_open_key)
                    logger.info(f"Circuit breaker CLOSED for agent: {agent_id}")

                await pipe.execute()

        except redis.RedisError as e:
            logger.error(f"Failed to set circuit status for {agent_id}: {e}")
            raise ActiveStateDataError(f"Failed to set circuit status: {e}") from e

    async def increment_failure_count(self, agent_id: str) -> int:
        """Increment the consecutive failure count for an agent.

        This also checks if the failure threshold is exceeded and opens
        the circuit breaker if necessary.

        Args:
            agent_id: The agent identifier

        Returns:
            New failure count after increment

        Raises:
            ActiveStateDataError: If operation fails
        """
        await self._ensure_initialized()

        try:
            health_key = self._health_key(agent_id)
            new_count = await self.redis.hincrby(health_key, "failure_count", 1)

            # Check if we should open the circuit
            threshold = self.circuit_config["failure_threshold"]
            if new_count >= threshold:
                current_status = await self.get_circuit_status(agent_id)
                if current_status == "closed":
                    await self.set_circuit_status(agent_id, "open")
                    logger.warning(
                        f"Circuit opened for {agent_id} after {new_count} failures"
                    )

            return new_count

        except redis.RedisError as e:
            logger.error(f"Failed to increment failure count for {agent_id}: {e}")
            raise ActiveStateDataError(f"Failed to increment failure count: {e}") from e

    async def reset_failure_count(self, agent_id: str) -> None:
        """Reset the consecutive failure count for an agent.

        This is typically called after a successful request to reset
        the failure tracking.

        Args:
            agent_id: The agent identifier

        Raises:
            ActiveStateDataError: If operation fails
        """
        await self._ensure_initialized()

        try:
            health_key = self._health_key(agent_id)
            await self.redis.hdel(health_key, "failure_count")

            # If circuit is half-open and we had a success, close it
            current_status = await self.get_circuit_status(agent_id)
            if current_status == "half_open":
                await self.set_circuit_status(agent_id, "closed")
                logger.info(f"Circuit closed for {agent_id} after successful request")

        except redis.RedisError as e:
            logger.error(f"Failed to reset failure count for {agent_id}: {e}")
            raise ActiveStateDataError(f"Failed to reset failure count: {e}") from e

    async def should_allow_request(
        self,
        agent_id: str,
        config: CircuitBreakerConfig | None = None,
    ) -> bool:
        """Determine if a request should be allowed based on circuit state.

        This method implements the circuit breaker pattern:
        - CLOSED: Allow all requests
        - OPEN: Reject all requests, unless recovery timeout has passed
        - HALF_OPEN: Allow limited requests to test recovery

        Args:
            agent_id: The agent identifier
            config: Optional circuit breaker config override

        Returns:
            True if request should be allowed, False otherwise

        Raises:
            ActiveStateDataError: If check fails
        """
        await self._ensure_initialized()

        config = config or self.circuit_config

        try:
            current_status = await self.get_circuit_status(agent_id)

            # CLOSED: Allow all requests
            if current_status == "closed":
                return True

            # OPEN: Check if recovery timeout has passed
            if current_status == "open":
                opened_at_key = self._circuit_opened_at_key(agent_id)
                opened_at_str = await self.redis.get(opened_at_key)

                if opened_at_str:
                    opened_at = float(opened_at_str)
                    elapsed = time.time() - opened_at
                    recovery_timeout = config["recovery_timeout_seconds"]

                    if elapsed >= recovery_timeout:
                        # Transition to half-open
                        await self.set_circuit_status(agent_id, "half_open")
                        return True

                # Still in open state, reject
                return False

            # HALF_OPEN: Allow limited requests
            if current_status == "half_open":
                half_open_key = self._half_open_count_key(agent_id)

                # Increment and check count atomically
                count = await self.redis.incr(half_open_key)

                # Set expiry if this is the first increment
                if count == 1:
                    await self.redis.expire(
                        half_open_key,
                        config["recovery_timeout_seconds"],
                    )

                max_calls = config["half_open_max_calls"]
                return count <= max_calls

            return True

        except redis.RedisError as e:
            logger.error(f"Failed to check request allowance for {agent_id}: {e}")
            # On error, default to allowing the request
            return True

    async def record_circuit_success(self, agent_id: str) -> None:
        """Record a successful request and potentially close the circuit.

        Args:
            agent_id: The agent identifier
        """
        await self.reset_failure_count(agent_id)

    async def record_circuit_failure(
        self,
        agent_id: str,
        error_type: str | None = None,
    ) -> Literal["closed", "open", "half_open"]:
        """Record a failed request and potentially open the circuit.

        Args:
            agent_id: The agent identifier
            error_type: Optional error type for logging

        Returns:
            New circuit status after recording failure
        """
        await self.increment_failure_count(agent_id)
        return await self.get_circuit_status(agent_id)

    # =========================================================================
    # Background Metrics Aggregation
    # =========================================================================

    async def aggregate_metrics(self, agent_ids: list[str]) -> dict[str, AgentMetrics]:
        """Aggregate and cache metrics for multiple agents.

        This method is designed to be run as a background task to pre-compute
        metrics and store them in Redis hashes for fast retrieval.

        Args:
            agent_ids: List of agent identifiers to aggregate metrics for

        Returns:
            Dictionary mapping agent_id to computed AgentMetrics

        Raises:
            ActiveStateDataError: If aggregation fails
        """
        await self._ensure_initialized()

        results: dict[str, AgentMetrics] = {}

        for agent_id in agent_ids:
            try:
                # Compute current metrics
                metrics = await self.get_metrics(agent_id)
                results[agent_id] = metrics

                # Cache in Redis hash
                metrics_key = self._metrics_key(agent_id)
                metrics_hash = {
                    "tokens_1m": str(metrics["tokens_1m"]),
                    "tokens_5m": str(metrics["tokens_5m"]),
                    "requests_1m": str(metrics["requests_1m"]),
                    "latency_p50": str(metrics["latency_p50"]),
                    "latency_p99": str(metrics["latency_p99"]),
                    "error_rate_5m": str(metrics["error_rate_5m"]),
                    "last_updated": metrics["last_updated"],
                }

                await self.redis.hset(metrics_key, mapping=metrics_hash)
                # Set expiration on cached metrics (5 minutes)
                await self.redis.expire(metrics_key, WINDOW_5M)

                logger.debug(f"Aggregated metrics for agent {agent_id}")

            except Exception as e:
                logger.error(f"Failed to aggregate metrics for {agent_id}: {e}")
                continue

        return results

    async def cleanup_old_data(
        self,
        agent_ids: list[str],
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    ) -> dict[str, int]:
        """Clean up old invocation and error data beyond retention period.

        This method removes entries from sorted sets that are older than the
        retention period to prevent unbounded memory growth.

        Args:
            agent_ids: List of agent identifiers to clean up
            retention_seconds: Data retention period in seconds (default: 1 hour)

        Returns:
            Dictionary mapping agent_id to number of entries removed

        Raises:
            ActiveStateDataError: If cleanup fails
        """
        await self._ensure_initialized()

        cutoff_time = time.time() - retention_seconds
        results: dict[str, int] = {}

        for agent_id in agent_ids:
            try:
                total_removed = 0

                # Clean up invocations sorted set
                invocations_key = self._invocations_key(agent_id)
                removed = await self.redis.zremrangebyscore(
                    invocations_key,
                    min=0,
                    max=cutoff_time,
                )
                total_removed += removed

                # Clean up errors sorted set
                errors_key = self._errors_key(agent_id)
                removed = await self.redis.zremrangebyscore(
                    errors_key,
                    min=0,
                    max=cutoff_time,
                )
                total_removed += removed

                results[agent_id] = total_removed

                if total_removed > 0:
                    logger.debug(
                        f"Cleaned up {total_removed} old entries for agent {agent_id}"
                    )

            except redis.RedisError as e:
                logger.error(f"Failed to cleanup data for {agent_id}: {e}")
                results[agent_id] = 0
                continue

        return results

    async def get_all_agent_ids(self) -> list[str]:
        """Get all known agent IDs from Redis keys.

        This scans for agent keys to find all tracked agents.

        Returns:
            List of agent identifiers
        """
        await self._ensure_initialized()

        try:
            # Scan for agent health keys to discover agents
            pattern = f"{KEY_PREFIX}:*:health"
            agent_ids: set[str] = set()

            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100,
                )

                for key in keys:
                    # Extract agent_id from key pattern: drx:agent:{agent_id}:health
                    parts = key.split(":")
                    if len(parts) >= 3:
                        agent_ids.add(parts[2])

                if cursor == 0:
                    break

            return list(agent_ids)

        except redis.RedisError as e:
            logger.error(f"Failed to get agent IDs: {e}")
            return []

    async def run_background_aggregation(
        self,
        interval_seconds: int = 30,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Run background metrics aggregation loop.

        This is designed to be run as an asyncio task to periodically
        aggregate metrics for all known agents.

        Args:
            interval_seconds: Time between aggregation runs (default: 30s)
            stop_event: Optional event to signal shutdown
        """
        logger.info(f"Starting background metrics aggregation (interval: {interval_seconds}s)")

        while True:
            try:
                # Check for stop signal
                if stop_event and stop_event.is_set():
                    logger.info("Background aggregation stopped by signal")
                    break

                # Get all known agents
                agent_ids = await self.get_all_agent_ids()

                if agent_ids:
                    # Aggregate metrics
                    await self.aggregate_metrics(agent_ids)

                    # Cleanup old data
                    await self.cleanup_old_data(agent_ids)

                    logger.debug(f"Background aggregation completed for {len(agent_ids)} agents")

            except Exception as e:
                logger.error(f"Background aggregation error: {e}")

            # Wait for next interval
            try:
                if stop_event:
                    # Use wait_for to allow early exit on stop signal
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=interval_seconds,
                    )
                    break
                else:
                    await asyncio.sleep(interval_seconds)
            except asyncio.TimeoutError:
                # Timeout is expected, continue loop
                continue


# =============================================================================
# Factory Functions
# =============================================================================


async def create_active_state_service(
    redis_url: str,
    circuit_config: CircuitBreakerConfig | None = None,
) -> ActiveStateService:
    """Create and initialize an ActiveStateService instance.

    This factory function creates a Redis connection and initializes
    the ActiveStateService for immediate use.

    Args:
        redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
        circuit_config: Optional circuit breaker configuration

    Returns:
        Initialized ActiveStateService instance

    Raises:
        ActiveStateConnectionError: If Redis connection fails

    Example:
        >>> service = await create_active_state_service("redis://localhost:6379/0")
        >>> await service.record_invocation("planner", tokens=1000, latency_ms=200, success=True)
    """
    try:
        # Create Redis client
        pool = ConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=50,
        )
        redis_client = redis.Redis(connection_pool=pool)

        # Create and initialize service
        service = ActiveStateService(
            redis_client=redis_client,
            circuit_config=circuit_config,
        )
        await service.initialize()

        return service

    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
        raise ActiveStateConnectionError(f"Redis connection failed: {e}") from e


def create_active_state_service_sync(
    redis_client: redis.Redis,
    circuit_config: CircuitBreakerConfig | None = None,
) -> ActiveStateService:
    """Create an ActiveStateService instance synchronously.

    This factory function creates the service without initializing
    the Redis connection. Call initialize() before use.

    Args:
        redis_client: Pre-configured Redis client instance
        circuit_config: Optional circuit breaker configuration

    Returns:
        ActiveStateService instance (not yet initialized)

    Example:
        >>> service = create_active_state_service_sync(redis_client)
        >>> await service.initialize()  # Must be called before use
    """
    return ActiveStateService(
        redis_client=redis_client,
        circuit_config=circuit_config,
    )


# =============================================================================
# Module-level Singleton
# =============================================================================

_active_state_service: ActiveStateService | None = None
_service_lock = asyncio.Lock()


async def get_active_state_service(
    redis_url: str | None = None,
) -> ActiveStateService:
    """Get or create the ActiveStateService singleton.

    This provides a module-level singleton instance for shared use
    across the application.

    Args:
        redis_url: Redis URL (only used for first initialization)

    Returns:
        Initialized ActiveStateService singleton

    Raises:
        ActiveStateConnectionError: If Redis connection fails
        ValueError: If redis_url not provided on first call
    """
    global _active_state_service

    if _active_state_service is not None:
        return _active_state_service

    async with _service_lock:
        if _active_state_service is not None:
            return _active_state_service

        if redis_url is None:
            # Try to get from config
            try:
                from src.config import get_settings
                redis_url = get_settings().redis_url_str
            except ImportError:
                raise ValueError("redis_url must be provided on first call")

        _active_state_service = await create_active_state_service(redis_url)
        return _active_state_service


async def close_active_state_service() -> None:
    """Close the ActiveStateService singleton and release resources."""
    global _active_state_service

    if _active_state_service is not None:
        try:
            await _active_state_service.redis.close()
        except Exception as e:
            logger.warning(f"Error closing ActiveStateService: {e}")
        _active_state_service = None
        logger.info("ActiveStateService singleton closed")


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    # TypedDicts
    "AgentMetrics",
    "AgentHealthStatus",
    "InvocationRecord",
    "CircuitBreakerConfig",
    # Constants
    "KEY_PREFIX",
    "DEFAULT_CIRCUIT_CONFIG",
    "DEFAULT_RETENTION_SECONDS",
    "WINDOW_1M",
    "WINDOW_5M",
    # Exceptions
    "ActiveStateError",
    "ActiveStateConnectionError",
    "ActiveStateDataError",
    # Service class
    "ActiveStateService",
    # Factory functions
    "create_active_state_service",
    "create_active_state_service_sync",
    "get_active_state_service",
    "close_active_state_service",
]
