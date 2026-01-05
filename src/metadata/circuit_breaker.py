"""Circuit Breaker Implementation for DRX Deep Research System.

Implements the circuit breaker pattern for fault tolerance and cascading failure prevention.
Integrates with ActiveStateService for Redis-backed state management and with AgentManifest
for configuration.

The circuit breaker has three states:
- CLOSED: Normal operation, requests flow through
- OPEN: Circuit is tripped, requests are rejected
- HALF_OPEN: Testing recovery with limited requests

Part of WP-M5: Circuit Breaker Implementation for the DRX spec.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Literal, TypedDict

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class CircuitState(str, Enum):
    """Circuit breaker state enumeration.

    The circuit breaker pattern uses three states to manage failure handling:

    - CLOSED: Normal operation. Requests flow through and failures are counted.
              When failure threshold is exceeded, circuit transitions to OPEN.

    - OPEN: Circuit is tripped. All requests are immediately rejected without
            attempting execution. After timeout, transitions to HALF_OPEN.

    - HALF_OPEN: Testing recovery. A limited number of requests are allowed
                 through to test if the underlying service has recovered.
                 Success closes the circuit; failure reopens it.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# TypedDict Definitions
# =============================================================================


class CircuitBreakerConfigDict(TypedDict, total=False):
    """Configuration for circuit breaker behavior.

    This is the canonical circuit breaker configuration TypedDict used across
    the entire DRX codebase. Import this from src.metadata.circuit_breaker
    instead of defining locally.

    Attributes:
        failure_threshold: Number of consecutive failures before opening circuit.
                          Default: 5
        success_threshold: Number of consecutive successes in half-open state
                          before closing circuit. Default: 3
        timeout_seconds: Time to wait in open state before attempting recovery.
                        Default: 30 seconds (standard across all components)
        half_open_max_calls: Maximum number of calls allowed in half-open state.
                            Default: 3
        error_rate_threshold: Error rate threshold (0.0-1.0) that can trigger
                             circuit open. Default: 0.5
    """

    failure_threshold: int
    success_threshold: int
    timeout_seconds: int
    half_open_max_calls: int
    error_rate_threshold: float


class CircuitStatsDict(TypedDict):
    """Statistics for a circuit breaker instance.

    Attributes:
        agent_id: The agent this circuit breaker protects
        state: Current circuit state
        failure_count: Current consecutive failure count
        success_count: Current consecutive success count in half-open
        total_failures: Total failures since last reset
        total_successes: Total successes since last reset
        last_failure_time: ISO timestamp of last failure, or None
        last_success_time: ISO timestamp of last success, or None
        last_state_change: ISO timestamp of last state transition
        opened_at: ISO timestamp when circuit was opened, or None
        time_in_current_state_seconds: Time spent in current state
    """

    agent_id: str
    state: str
    failure_count: int
    success_count: int
    total_failures: int
    total_successes: int
    last_failure_time: str | None
    last_success_time: str | None
    last_state_change: str
    opened_at: str | None
    time_in_current_state_seconds: float


class CircuitStateChangeEvent(TypedDict):
    """Event emitted when circuit state changes.

    Attributes:
        agent_id: The agent whose circuit changed
        previous_state: State before the change
        new_state: State after the change
        timestamp: ISO timestamp of the change
        reason: Reason for the state change
        failure_count: Current failure count at time of change
    """

    agent_id: str
    previous_state: str
    new_state: str
    timestamp: str
    reason: str
    failure_count: int


# =============================================================================
# Default Configuration
# =============================================================================


DEFAULT_CIRCUIT_BREAKER_CONFIG: CircuitBreakerConfigDict = {
    "failure_threshold": 5,
    "success_threshold": 3,
    "timeout_seconds": 30,
    "half_open_max_calls": 3,
    "error_rate_threshold": 0.5,
}


# =============================================================================
# Exceptions
# =============================================================================


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, agent_id: str, message: str | None = None) -> None:
        """Initialize CircuitOpenError.

        Args:
            agent_id: The agent whose circuit is open
            message: Optional custom message
        """
        self.agent_id = agent_id
        super().__init__(
            message or f"Circuit breaker is open for agent: {agent_id}"
        )


class CircuitHalfOpenExhaustedError(CircuitBreakerError):
    """Raised when half-open call limit is exhausted."""

    def __init__(self, agent_id: str, max_calls: int) -> None:
        """Initialize CircuitHalfOpenExhaustedError.

        Args:
            agent_id: The agent whose half-open limit is reached
            max_calls: The maximum calls allowed in half-open state
        """
        self.agent_id = agent_id
        self.max_calls = max_calls
        super().__init__(
            f"Half-open call limit ({max_calls}) exhausted for agent: {agent_id}"
        )


# =============================================================================
# CircuitBreaker Class
# =============================================================================


class CircuitBreaker:
    """Circuit breaker implementation for agent fault tolerance.

    This class implements the circuit breaker pattern to prevent cascading
    failures when an agent or its dependencies are unhealthy. It integrates
    with the ActiveStateService for persistent state management in Redis.

    The circuit breaker tracks failures and successes for each agent,
    automatically transitioning between states based on configured thresholds.

    Example:
        >>> from src.services.active_state import ActiveStateService
        >>> circuit_breaker = CircuitBreaker(active_state_service)
        >>>
        >>> # Check if request is allowed
        >>> if await circuit_breaker.can_execute("searcher_v1"):
        ...     try:
        ...         result = await execute_agent_request()
        ...         await circuit_breaker.record_success("searcher_v1")
        ...     except Exception as e:
        ...         await circuit_breaker.record_failure("searcher_v1", e)
        ... else:
        ...     # Handle circuit open - use fallback or reject

    Attributes:
        active_state: The ActiveStateService for Redis operations
        config: Circuit breaker configuration
        _state_change_listeners: List of callbacks for state change events
        _lock: Async lock for thread-safe state transitions
    """

    def __init__(
        self,
        active_state_service: Any,
        config: CircuitBreakerConfigDict | None = None,
        manifest_registry: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the CircuitBreaker.

        Args:
            active_state_service: The ActiveStateService instance for Redis ops
            config: Optional circuit breaker configuration override
            manifest_registry: Optional registry mapping agent_id to AgentManifest
                              for per-agent configuration overrides
        """
        self.active_state = active_state_service
        self.config = config or DEFAULT_CIRCUIT_BREAKER_CONFIG.copy()
        self.manifest_registry = manifest_registry or {}

        # State tracking for half-open success counting
        self._half_open_successes: dict[str, int] = {}
        self._half_open_calls: dict[str, int] = {}

        # Internal state tracking for in-memory operations
        self._failure_counts: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}
        self._total_failures: dict[str, int] = {}
        self._total_successes: dict[str, int] = {}
        self._last_failure_time: dict[str, str] = {}
        self._last_success_time: dict[str, str] = {}
        self._last_state_change: dict[str, str] = {}
        self._opened_at: dict[str, str] = {}

        # Event listeners for state changes
        self._state_change_listeners: list[Callable[[CircuitStateChangeEvent], Any]] = []

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    def _get_agent_config(self, agent_id: str) -> CircuitBreakerConfigDict:
        """Get circuit breaker configuration for a specific agent.

        If the agent has a manifest with custom circuit breaker settings,
        those are used. Otherwise, falls back to default config.

        Args:
            agent_id: The agent identifier

        Returns:
            CircuitBreakerConfigDict with the agent's configuration
        """
        if agent_id in self.manifest_registry:
            manifest = self.manifest_registry[agent_id]
            # Check if manifest has circuit_breaker config
            if hasattr(manifest, "circuit_breaker"):
                cb_config = manifest.circuit_breaker
                return CircuitBreakerConfigDict(
                    failure_threshold=getattr(
                        cb_config, "failure_threshold", self.config.get("failure_threshold", 5)
                    ),
                    success_threshold=getattr(
                        cb_config, "success_threshold", self.config.get("success_threshold", 3)
                    ),
                    timeout_seconds=getattr(
                        cb_config, "timeout_seconds", self.config.get("timeout_seconds", 30)
                    ),
                    half_open_max_calls=self.config.get("half_open_max_calls", 3),
                )
        return self.config

    def add_state_change_listener(
        self, listener: Callable[[CircuitStateChangeEvent], Any]
    ) -> None:
        """Add a listener for circuit state change events.

        Args:
            listener: Callback function that receives CircuitStateChangeEvent
        """
        self._state_change_listeners.append(listener)

    def remove_state_change_listener(
        self, listener: Callable[[CircuitStateChangeEvent], Any]
    ) -> None:
        """Remove a state change listener.

        Args:
            listener: The listener to remove
        """
        if listener in self._state_change_listeners:
            self._state_change_listeners.remove(listener)

    async def _emit_state_change(
        self,
        agent_id: str,
        previous_state: CircuitState,
        new_state: CircuitState,
        reason: str,
    ) -> None:
        """Emit a state change event to all listeners.

        Args:
            agent_id: The agent whose state changed
            previous_state: State before the change
            new_state: State after the change
            reason: Reason for the state change
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        self._last_state_change[agent_id] = timestamp

        if new_state == CircuitState.OPEN:
            self._opened_at[agent_id] = timestamp

        event = CircuitStateChangeEvent(
            agent_id=agent_id,
            previous_state=previous_state.value,
            new_state=new_state.value,
            timestamp=timestamp,
            reason=reason,
            failure_count=self._failure_counts.get(agent_id, 0),
        )

        logger.info(
            f"Circuit state change for {agent_id}: "
            f"{previous_state.value} -> {new_state.value} ({reason})"
        )

        # Call listeners (non-blocking)
        for listener in self._state_change_listeners:
            try:
                result = listener(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.warning(f"State change listener error: {e}")

    async def can_execute(self, agent_id: str) -> bool:
        """Check if a request is allowed for the given agent.

        This method checks the current circuit state and determines if
        a request should be allowed through:

        - CLOSED: Always allows requests
        - OPEN: Rejects requests, unless timeout has passed (transitions to HALF_OPEN)
        - HALF_OPEN: Allows limited requests up to half_open_max_calls

        Args:
            agent_id: The agent identifier to check

        Returns:
            True if request is allowed, False otherwise

        Example:
            >>> if await circuit_breaker.can_execute("searcher_v1"):
            ...     # Proceed with request
            ...     pass
            ... else:
            ...     # Handle rejection
            ...     pass
        """
        async with self._lock:
            current_state = await self.get_state(agent_id)
            config = self._get_agent_config(agent_id)

            if current_state == CircuitState.CLOSED:
                return True

            if current_state == CircuitState.OPEN:
                # Check if timeout has passed
                if await self._should_transition_to_half_open(agent_id, config):
                    await self._set_state(agent_id, CircuitState.HALF_OPEN)
                    self._half_open_calls[agent_id] = 0
                    self._half_open_successes[agent_id] = 0
                    await self._emit_state_change(
                        agent_id,
                        CircuitState.OPEN,
                        CircuitState.HALF_OPEN,
                        "Timeout elapsed, testing recovery",
                    )
                    return True
                return False

            if current_state == CircuitState.HALF_OPEN:
                return self._should_allow_half_open(agent_id, config)

            return True

    async def _should_transition_to_half_open(
        self, agent_id: str, config: CircuitBreakerConfigDict
    ) -> bool:
        """Check if circuit should transition from OPEN to HALF_OPEN.

        Args:
            agent_id: The agent identifier
            config: Circuit breaker configuration

        Returns:
            True if timeout has elapsed and should transition
        """
        opened_at = self._opened_at.get(agent_id)
        if not opened_at:
            # Also check Redis for opened_at timestamp
            try:
                status = await self.active_state.get_circuit_status(agent_id)
                if status == "open":
                    # Get opened_at from Redis
                    opened_at_key = self.active_state._circuit_opened_at_key(agent_id)
                    opened_at_str = await self.active_state.redis.get(opened_at_key)
                    if opened_at_str:
                        opened_at_ts = float(opened_at_str)
                        elapsed = time.time() - opened_at_ts
                        return elapsed >= config.get("timeout_seconds", 30)
            except Exception as e:
                logger.warning(f"Error checking Redis for opened_at: {e}")
            return True  # Default to allowing transition if no timestamp

        try:
            opened_at_dt = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - opened_at_dt).total_seconds()
            return elapsed >= config.get("timeout_seconds", 30)
        except (ValueError, TypeError):
            return True

    def _should_allow_half_open(
        self, agent_id: str, config: CircuitBreakerConfigDict
    ) -> bool:
        """Check if a request should be allowed in half-open state.

        Args:
            agent_id: The agent identifier
            config: Circuit breaker configuration

        Returns:
            True if under the half-open call limit
        """
        current_calls = self._half_open_calls.get(agent_id, 0)
        max_calls = config.get("half_open_max_calls", 3)

        if current_calls < max_calls:
            self._half_open_calls[agent_id] = current_calls + 1
            return True
        return False

    async def _set_state(self, agent_id: str, state: CircuitState) -> None:
        """Set the circuit state in Redis.

        Args:
            agent_id: The agent identifier
            state: The new circuit state
        """
        await self.active_state.set_circuit_status(agent_id, state.value)

    async def get_state(self, agent_id: str) -> CircuitState:
        """Get the current circuit state for an agent.

        Retrieves the circuit state from Redis. If no state is set,
        returns CLOSED as the default.

        Args:
            agent_id: The agent identifier

        Returns:
            Current CircuitState for the agent

        Example:
            >>> state = await circuit_breaker.get_state("searcher_v1")
            >>> if state == CircuitState.OPEN:
            ...     print("Circuit is open!")
        """
        try:
            status = await self.active_state.get_circuit_status(agent_id)
            return CircuitState(status)
        except (ValueError, Exception) as e:
            logger.warning(f"Error getting circuit state for {agent_id}: {e}")
            return CircuitState.CLOSED

    async def record_success(self, agent_id: str) -> None:
        """Record a successful execution for the given agent.

        This method updates the circuit breaker state based on success:

        - CLOSED: Resets failure count (no state change)
        - HALF_OPEN: Increments success count. If success_threshold is reached,
                     transitions to CLOSED state.
        - OPEN: Should not normally happen (requests should be rejected)

        Also records the invocation in ActiveStateService for metrics.

        Args:
            agent_id: The agent identifier

        Example:
            >>> try:
            ...     result = await execute_agent_request()
            ...     await circuit_breaker.record_success("searcher_v1")
            ... except Exception as e:
            ...     await circuit_breaker.record_failure("searcher_v1", e)
        """
        async with self._lock:
            current_state = await self.get_state(agent_id)
            config = self._get_agent_config(agent_id)

            timestamp = datetime.now(timezone.utc).isoformat()
            self._last_success_time[agent_id] = timestamp
            self._total_successes[agent_id] = self._total_successes.get(agent_id, 0) + 1

            if current_state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_counts[agent_id] = 0
                await self.active_state.reset_failure_count(agent_id)
                logger.debug(f"Success recorded for {agent_id} (CLOSED state)")

            elif current_state == CircuitState.HALF_OPEN:
                # Increment half-open success count
                self._half_open_successes[agent_id] = (
                    self._half_open_successes.get(agent_id, 0) + 1
                )
                success_threshold = config.get("success_threshold", 3)

                if self._half_open_successes[agent_id] >= success_threshold:
                    # Transition to CLOSED
                    await self._set_state(agent_id, CircuitState.CLOSED)
                    self._failure_counts[agent_id] = 0
                    self._half_open_calls[agent_id] = 0
                    self._half_open_successes[agent_id] = 0
                    self._opened_at.pop(agent_id, None)

                    await self._emit_state_change(
                        agent_id,
                        CircuitState.HALF_OPEN,
                        CircuitState.CLOSED,
                        f"Success threshold ({success_threshold}) reached",
                    )
                    logger.info(
                        f"Circuit CLOSED for {agent_id} after "
                        f"{success_threshold} successes in half-open"
                    )
                else:
                    logger.debug(
                        f"Half-open success {self._half_open_successes[agent_id]}/"
                        f"{success_threshold} for {agent_id}"
                    )

            elif current_state == CircuitState.OPEN:
                # This shouldn't happen normally, but log it
                logger.warning(
                    f"Success recorded for {agent_id} while circuit is OPEN"
                )

    async def record_failure(
        self, agent_id: str, error: Exception | None = None
    ) -> None:
        """Record a failed execution for the given agent.

        This method updates the circuit breaker state based on failure:

        - CLOSED: Increments failure count. If failure_threshold is reached,
                  transitions to OPEN state.
        - HALF_OPEN: Immediately transitions back to OPEN state.
        - OPEN: Updates failure tracking (should not normally happen)

        Args:
            agent_id: The agent identifier
            error: Optional exception that caused the failure

        Example:
            >>> try:
            ...     result = await execute_agent_request()
            ...     await circuit_breaker.record_success("searcher_v1")
            ... except Exception as e:
            ...     await circuit_breaker.record_failure("searcher_v1", e)
        """
        async with self._lock:
            current_state = await self.get_state(agent_id)
            config = self._get_agent_config(agent_id)

            timestamp = datetime.now(timezone.utc).isoformat()
            self._last_failure_time[agent_id] = timestamp
            self._total_failures[agent_id] = self._total_failures.get(agent_id, 0) + 1
            self._failure_counts[agent_id] = self._failure_counts.get(agent_id, 0) + 1

            error_type = type(error).__name__ if error else "unknown"
            logger.warning(
                f"Failure recorded for {agent_id}: {error_type} "
                f"(count: {self._failure_counts[agent_id]})"
            )

            if current_state == CircuitState.CLOSED:
                failure_threshold = config.get("failure_threshold", 5)

                if self._failure_counts[agent_id] >= failure_threshold:
                    # Transition to OPEN
                    await self._set_state(agent_id, CircuitState.OPEN)

                    await self._emit_state_change(
                        agent_id,
                        CircuitState.CLOSED,
                        CircuitState.OPEN,
                        f"Failure threshold ({failure_threshold}) exceeded: {error_type}",
                    )
                    logger.warning(
                        f"Circuit OPENED for {agent_id} after "
                        f"{failure_threshold} failures"
                    )
                else:
                    # Record failure in ActiveStateService
                    await self.active_state.increment_failure_count(agent_id)

            elif current_state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately reopens circuit
                await self._set_state(agent_id, CircuitState.OPEN)
                self._half_open_calls[agent_id] = 0
                self._half_open_successes[agent_id] = 0

                await self._emit_state_change(
                    agent_id,
                    CircuitState.HALF_OPEN,
                    CircuitState.OPEN,
                    f"Failure during recovery test: {error_type}",
                )
                logger.warning(
                    f"Circuit reopened for {agent_id} due to failure in half-open"
                )

            elif current_state == CircuitState.OPEN:
                # Already open, just log
                logger.debug(
                    f"Additional failure recorded for {agent_id} "
                    f"while circuit is OPEN"
                )

    async def force_open(self, agent_id: str, reason: str = "Manual override") -> None:
        """Manually open the circuit for an agent.

        Forces the circuit to open state regardless of current state.
        Useful for administrative actions or proactive protection.

        Args:
            agent_id: The agent identifier
            reason: Reason for forcing the circuit open

        Example:
            >>> # Proactively open circuit before maintenance
            >>> await circuit_breaker.force_open("searcher_v1", "Scheduled maintenance")
        """
        async with self._lock:
            current_state = await self.get_state(agent_id)

            if current_state != CircuitState.OPEN:
                await self._set_state(agent_id, CircuitState.OPEN)
                self._half_open_calls[agent_id] = 0
                self._half_open_successes[agent_id] = 0

                await self._emit_state_change(
                    agent_id,
                    current_state,
                    CircuitState.OPEN,
                    f"Forced open: {reason}",
                )
                logger.warning(f"Circuit force-opened for {agent_id}: {reason}")
            else:
                logger.info(f"Circuit already open for {agent_id}")

    async def force_close(self, agent_id: str, reason: str = "Manual override") -> None:
        """Manually close the circuit for an agent.

        Forces the circuit to closed state regardless of current state.
        Useful for recovery after manual intervention or testing.

        Args:
            agent_id: The agent identifier
            reason: Reason for forcing the circuit closed

        Example:
            >>> # Re-enable agent after fixing issue
            >>> await circuit_breaker.force_close("searcher_v1", "Issue resolved")
        """
        async with self._lock:
            current_state = await self.get_state(agent_id)

            if current_state != CircuitState.CLOSED:
                await self._set_state(agent_id, CircuitState.CLOSED)
                self._failure_counts[agent_id] = 0
                self._half_open_calls[agent_id] = 0
                self._half_open_successes[agent_id] = 0
                self._opened_at.pop(agent_id, None)

                await self._emit_state_change(
                    agent_id,
                    current_state,
                    CircuitState.CLOSED,
                    f"Forced close: {reason}",
                )
                logger.info(f"Circuit force-closed for {agent_id}: {reason}")
            else:
                logger.info(f"Circuit already closed for {agent_id}")

    async def get_stats(self, agent_id: str) -> CircuitStatsDict:
        """Get comprehensive statistics for a circuit breaker.

        Returns detailed statistics about the circuit breaker's current state,
        failure/success counts, and timing information.

        Args:
            agent_id: The agent identifier

        Returns:
            CircuitStatsDict with all statistics

        Example:
            >>> stats = await circuit_breaker.get_stats("searcher_v1")
            >>> print(f"State: {stats['state']}, Failures: {stats['failure_count']}")
        """
        current_state = await self.get_state(agent_id)
        timestamp = datetime.now(timezone.utc)

        # Calculate time in current state
        last_change = self._last_state_change.get(agent_id)
        time_in_state = 0.0
        if last_change:
            try:
                last_change_dt = datetime.fromisoformat(
                    last_change.replace("Z", "+00:00")
                )
                time_in_state = (timestamp - last_change_dt).total_seconds()
            except (ValueError, TypeError):
                pass

        return CircuitStatsDict(
            agent_id=agent_id,
            state=current_state.value,
            failure_count=self._failure_counts.get(agent_id, 0),
            success_count=self._half_open_successes.get(agent_id, 0),
            total_failures=self._total_failures.get(agent_id, 0),
            total_successes=self._total_successes.get(agent_id, 0),
            last_failure_time=self._last_failure_time.get(agent_id),
            last_success_time=self._last_success_time.get(agent_id),
            last_state_change=self._last_state_change.get(agent_id, timestamp.isoformat()),
            opened_at=self._opened_at.get(agent_id),
            time_in_current_state_seconds=time_in_state,
        )

    async def get_alternative_agent(
        self,
        failed_agent: str,
        required_capabilities: list[str] | None = None,
    ) -> str | None:
        """Find a healthy alternative agent to handle requests.

        When a circuit is open, this method can be used to find another
        agent of the same type or with the same capabilities that can
        handle the request.

        Args:
            failed_agent: The agent whose circuit is open
            required_capabilities: Optional list of required capabilities

        Returns:
            Agent ID of a healthy alternative, or None if no alternative found

        Example:
            >>> if not await circuit_breaker.can_execute("searcher_v1"):
            ...     alt = await circuit_breaker.get_alternative_agent("searcher_v1")
            ...     if alt:
            ...         result = await execute_with_agent(alt)
        """
        if not self.manifest_registry:
            logger.debug("No manifest registry available for alternative lookup")
            return None

        # Get the failed agent's type and capabilities
        failed_manifest = self.manifest_registry.get(failed_agent)
        if not failed_manifest:
            logger.debug(f"No manifest found for {failed_agent}")
            return None

        failed_agent_type = getattr(failed_manifest, "agent_type", None)
        failed_capabilities = getattr(failed_manifest, "capabilities", [])

        if required_capabilities is None:
            required_capabilities = failed_capabilities

        required_set = set(required_capabilities)

        # Search for alternatives
        for agent_id, manifest in self.manifest_registry.items():
            if agent_id == failed_agent:
                continue

            # Check if agent is active
            if not getattr(manifest, "is_active", True):
                continue

            # Check agent type matches or has required capabilities
            agent_type = getattr(manifest, "agent_type", None)
            agent_capabilities = set(getattr(manifest, "capabilities", []))

            type_matches = agent_type == failed_agent_type
            capabilities_match = required_set.issubset(agent_capabilities)

            if type_matches or capabilities_match:
                # Check if this agent's circuit is healthy
                state = await self.get_state(agent_id)
                if state == CircuitState.CLOSED:
                    logger.info(
                        f"Found alternative agent {agent_id} for {failed_agent}"
                    )
                    return agent_id

        logger.warning(f"No healthy alternative found for {failed_agent}")
        return None

    async def reset_all(self, agent_id: str) -> None:
        """Reset all circuit breaker state for an agent.

        Clears all failure counts, success counts, and resets to CLOSED state.

        Args:
            agent_id: The agent identifier
        """
        async with self._lock:
            await self._set_state(agent_id, CircuitState.CLOSED)

            self._failure_counts.pop(agent_id, None)
            self._success_counts.pop(agent_id, None)
            self._total_failures.pop(agent_id, None)
            self._total_successes.pop(agent_id, None)
            self._last_failure_time.pop(agent_id, None)
            self._last_success_time.pop(agent_id, None)
            self._last_state_change.pop(agent_id, None)
            self._opened_at.pop(agent_id, None)
            self._half_open_calls.pop(agent_id, None)
            self._half_open_successes.pop(agent_id, None)

            await self.active_state.reset_failure_count(agent_id)

            logger.info(f"Circuit breaker reset for {agent_id}")

    async def get_all_circuit_states(self) -> dict[str, CircuitState]:
        """Get circuit states for all known agents.

        Returns:
            Dictionary mapping agent_id to CircuitState
        """
        states: dict[str, CircuitState] = {}

        try:
            agent_ids = await self.active_state.get_all_agent_ids()
            for agent_id in agent_ids:
                states[agent_id] = await self.get_state(agent_id)
        except Exception as e:
            logger.error(f"Error getting all circuit states: {e}")

        return states


# =============================================================================
# Factory Functions
# =============================================================================


async def create_circuit_breaker(
    active_state_service: Any,
    config: CircuitBreakerConfigDict | None = None,
    manifest_registry: dict[str, Any] | None = None,
) -> CircuitBreaker:
    """Create a CircuitBreaker instance.

    Factory function to create and configure a CircuitBreaker.

    Args:
        active_state_service: The ActiveStateService instance
        config: Optional circuit breaker configuration
        manifest_registry: Optional agent manifest registry

    Returns:
        Configured CircuitBreaker instance

    Example:
        >>> from src.services.active_state import create_active_state_service
        >>> active_state = await create_active_state_service("redis://localhost:6379/0")
        >>> circuit_breaker = await create_circuit_breaker(active_state)
    """
    return CircuitBreaker(
        active_state_service=active_state_service,
        config=config,
        manifest_registry=manifest_registry,
    )


def create_circuit_breaker_sync(
    active_state_service: Any,
    config: CircuitBreakerConfigDict | None = None,
    manifest_registry: dict[str, Any] | None = None,
) -> CircuitBreaker:
    """Create a CircuitBreaker instance synchronously.

    Args:
        active_state_service: The ActiveStateService instance
        config: Optional circuit breaker configuration
        manifest_registry: Optional agent manifest registry

    Returns:
        CircuitBreaker instance
    """
    return CircuitBreaker(
        active_state_service=active_state_service,
        config=config,
        manifest_registry=manifest_registry,
    )


# =============================================================================
# Module-level Singleton
# =============================================================================


_circuit_breaker: CircuitBreaker | None = None
_cb_lock = asyncio.Lock()


async def get_circuit_breaker(
    active_state_service: Any | None = None,
    config: CircuitBreakerConfigDict | None = None,
) -> CircuitBreaker:
    """Get or create the CircuitBreaker singleton.

    Provides a module-level singleton for shared use across the application.

    Args:
        active_state_service: ActiveStateService (required on first call)
        config: Optional circuit breaker configuration

    Returns:
        CircuitBreaker singleton instance

    Raises:
        ValueError: If active_state_service not provided on first call
    """
    global _circuit_breaker

    if _circuit_breaker is not None:
        return _circuit_breaker

    async with _cb_lock:
        if _circuit_breaker is not None:
            return _circuit_breaker

        if active_state_service is None:
            raise ValueError(
                "active_state_service must be provided on first call"
            )

        _circuit_breaker = await create_circuit_breaker(
            active_state_service=active_state_service,
            config=config,
        )
        return _circuit_breaker


def reset_circuit_breaker_singleton() -> None:
    """Reset the CircuitBreaker singleton.

    Used primarily for testing to reset global state.
    """
    global _circuit_breaker
    _circuit_breaker = None


# =============================================================================
# Type Exports
# =============================================================================


__all__ = [
    # Enums
    "CircuitState",
    # TypedDicts
    "CircuitBreakerConfigDict",
    "CircuitStatsDict",
    "CircuitStateChangeEvent",
    # Constants
    "DEFAULT_CIRCUIT_BREAKER_CONFIG",
    # Exceptions
    "CircuitBreakerError",
    "CircuitOpenError",
    "CircuitHalfOpenExhaustedError",
    # Main class
    "CircuitBreaker",
    # Factory functions
    "create_circuit_breaker",
    "create_circuit_breaker_sync",
    "get_circuit_breaker",
    "reset_circuit_breaker_singleton",
]
