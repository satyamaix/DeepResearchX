"""
DRX Circuit Breaker Behavior Tests.

Tests for circuit breaker pattern implementation including:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold enforcement
- Success threshold for recovery
- Automatic rerouting when circuit opens
- Timeout-based recovery attempts

Part of WP-M8: Metadata-Aware Evaluation Implementation.

Test Categories:
- State transition tests: Verify correct state machine behavior
- Threshold tests: Verify failure/success thresholds are respected
- Recovery tests: Verify half-open state and recovery logic
- Rerouting tests: Verify fallback agent selection

Usage:
    # Run all circuit breaker tests
    pytest ci/evaluation/test_circuit_breaker.py -v

    # Run state transition tests
    pytest ci/evaluation/test_circuit_breaker.py -k state -v

    # Run with markers
    pytest ci/evaluation/test_circuit_breaker.py -m ci_gate -v
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Pytest Configuration
# =============================================================================


pytestmark = [
    pytest.mark.eval,
    pytest.mark.asyncio,
]


# Import shared mock classes from conftest
from ci.evaluation.conftest import (
    MockActiveStateService,
    MockCircuitBreakerConfig,
    MockManifest,
    MockRedisClient,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def redis_client() -> MockRedisClient:
    """Return a mock Redis client."""
    return MockRedisClient()


@pytest.fixture
def active_state_service(redis_client: MockRedisClient) -> MockActiveStateService:
    """Return a mock ActiveStateService."""
    return MockActiveStateService(redis_client)


@pytest.fixture
def circuit_breaker_config() -> dict[str, int]:
    """Return default circuit breaker configuration."""
    return {
        "failure_threshold": 5,
        "success_threshold": 3,
        "timeout_seconds": 30,
        "half_open_max_calls": 3,
    }


@pytest.fixture
def manifest_registry() -> dict[str, MockManifest]:
    """Return a registry of mock manifests for testing."""
    return {
        "searcher_v1": MockManifest(
            agent_id="searcher_v1",
            agent_type="searcher",
            capabilities=["web_search", "source_discovery"],
        ),
        "searcher_v2": MockManifest(
            agent_id="searcher_v2",
            agent_type="searcher",
            capabilities=["web_search", "source_discovery"],
        ),
        "reader_v1": MockManifest(
            agent_id="reader_v1",
            agent_type="reader",
            capabilities=["content_extraction", "text_processing"],
        ),
        "synthesizer_v1": MockManifest(
            agent_id="synthesizer_v1",
            agent_type="synthesizer",
            capabilities=["information_synthesis"],
        ),
    }


@pytest.fixture
async def circuit_breaker(
    active_state_service: MockActiveStateService,
    circuit_breaker_config: dict[str, int],
    manifest_registry: dict[str, MockManifest],
) -> Any:
    """Return a CircuitBreaker instance for testing."""
    try:
        from src.metadata.circuit_breaker import CircuitBreaker
    except ImportError:
        pytest.skip("CircuitBreaker not available")

    return CircuitBreaker(
        active_state_service=active_state_service,
        config=circuit_breaker_config,
        manifest_registry=manifest_registry,
    )


# =============================================================================
# State Transition Tests
# =============================================================================


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    async def test_initial_state_is_closed(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test that circuit starts in CLOSED state."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        state = await circuit_breaker.get_state("new_agent_v1")
        assert state == CircuitState.CLOSED

    async def test_can_execute_in_closed_state(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test that requests are allowed in CLOSED state."""
        can_execute = await circuit_breaker.can_execute("searcher_v1")
        assert can_execute is True

    async def test_circuit_opens_after_failure_threshold(
        self,
        circuit_breaker: Any,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """Test that circuit opens after failure threshold is reached."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"
        threshold = circuit_breaker_config["failure_threshold"]

        # Record failures up to threshold
        for i in range(threshold):
            await circuit_breaker.record_failure(agent_id, Exception(f"Error {i}"))

        # Circuit should now be OPEN
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.OPEN

    async def test_cannot_execute_in_open_state(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
    ) -> None:
        """Test that requests are blocked in OPEN state."""
        agent_id = "searcher_v1"

        # Force circuit to OPEN state
        await active_state_service.set_circuit_status(agent_id, "open")

        can_execute = await circuit_breaker.can_execute(agent_id)
        assert can_execute is False

    async def test_circuit_transitions_to_half_open_after_timeout(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
        redis_client: MockRedisClient,
    ) -> None:
        """Test that circuit transitions to HALF_OPEN after timeout."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"

        # Force circuit to OPEN state with old timestamp
        await active_state_service.set_circuit_status(agent_id, "open")

        # Set opened_at to time in past (beyond timeout)
        past_time = time.time() - 60  # 60 seconds ago
        await redis_client.set(f"drx:agent:{agent_id}:circuit_opened_at", str(past_time))

        # Request should trigger transition to HALF_OPEN
        can_execute = await circuit_breaker.can_execute(agent_id)

        # Should be allowed (half-open allows limited requests)
        assert can_execute is True

        # State should now be HALF_OPEN
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.HALF_OPEN

    async def test_circuit_closes_after_success_threshold_in_half_open(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """Test that circuit closes after success threshold in HALF_OPEN."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"
        success_threshold = circuit_breaker_config["success_threshold"]

        # Force circuit to HALF_OPEN state
        await active_state_service.set_circuit_status(agent_id, "half_open")

        # Record successes up to threshold
        for _ in range(success_threshold):
            await circuit_breaker.record_success(agent_id)

        # Circuit should now be CLOSED
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.CLOSED

    async def test_circuit_reopens_on_failure_in_half_open(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
    ) -> None:
        """Test that circuit reopens on any failure in HALF_OPEN."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"

        # Force circuit to HALF_OPEN state
        await active_state_service.set_circuit_status(agent_id, "half_open")

        # Record one success followed by failure
        await circuit_breaker.record_success(agent_id)
        await circuit_breaker.record_failure(agent_id, Exception("Test error"))

        # Circuit should reopen
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.OPEN


# =============================================================================
# Threshold Tests
# =============================================================================


class TestCircuitBreakerThresholds:
    """Tests for circuit breaker threshold enforcement."""

    async def test_failure_threshold_not_exceeded(
        self,
        circuit_breaker: Any,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """Test circuit stays closed when below failure threshold."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"
        threshold = circuit_breaker_config["failure_threshold"]

        # Record failures just below threshold
        for i in range(threshold - 1):
            await circuit_breaker.record_failure(agent_id, Exception(f"Error {i}"))

        # Circuit should still be CLOSED
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.CLOSED

    async def test_failure_count_resets_on_success(
        self,
        circuit_breaker: Any,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """Test that failure count resets on successful request."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"
        threshold = circuit_breaker_config["failure_threshold"]

        # Record failures just below threshold
        for i in range(threshold - 1):
            await circuit_breaker.record_failure(agent_id, Exception(f"Error {i}"))

        # Record success to reset count
        await circuit_breaker.record_success(agent_id)

        # Record more failures (should start from 0)
        for i in range(threshold - 1):
            await circuit_breaker.record_failure(agent_id, Exception(f"Error {i}"))

        # Circuit should still be CLOSED (count was reset)
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.CLOSED

    async def test_half_open_limited_requests(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """Test that HALF_OPEN state limits number of requests."""
        agent_id = "searcher_v1"
        max_calls = circuit_breaker_config["half_open_max_calls"]

        # Force circuit to HALF_OPEN state
        await active_state_service.set_circuit_status(agent_id, "half_open")

        # First N requests should be allowed
        allowed_count = 0
        for _ in range(max_calls + 2):
            if await circuit_breaker.can_execute(agent_id):
                allowed_count += 1

        # Should have allowed exactly max_calls
        assert allowed_count == max_calls

    @pytest.mark.parametrize(
        "failure_threshold,failures_before_open",
        [
            (3, 3),
            (5, 5),
            (10, 10),
        ],
    )
    async def test_configurable_failure_threshold(
        self,
        active_state_service: MockActiveStateService,
        manifest_registry: dict[str, MockManifest],
        failure_threshold: int,
        failures_before_open: int,
    ) -> None:
        """Test that failure threshold is configurable."""
        try:
            from src.metadata.circuit_breaker import CircuitBreaker, CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        config = {
            "failure_threshold": failure_threshold,
            "success_threshold": 3,
            "timeout_seconds": 30,
            "half_open_max_calls": 3,
        }

        cb = CircuitBreaker(
            active_state_service=active_state_service,
            config=config,
            manifest_registry=manifest_registry,
        )

        agent_id = "searcher_v1"

        # Record exactly threshold failures
        for i in range(failures_before_open):
            await cb.record_failure(agent_id, Exception(f"Error {i}"))

        state = await cb.get_state(agent_id)
        assert state == CircuitState.OPEN


# =============================================================================
# Recovery Tests
# =============================================================================


class TestCircuitBreakerRecovery:
    """Tests for circuit breaker recovery behavior."""

    async def test_recovery_after_timeout(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
        redis_client: MockRedisClient,
    ) -> None:
        """Test recovery attempt after timeout period."""
        agent_id = "searcher_v1"

        # Force circuit to OPEN state
        await active_state_service.set_circuit_status(agent_id, "open")

        # Set opened_at to time in past
        past_time = time.time() - 60
        await redis_client.set(f"drx:agent:{agent_id}:circuit_opened_at", str(past_time))

        # Attempt request - should be allowed for recovery test
        can_execute = await circuit_breaker.can_execute(agent_id)
        assert can_execute is True

    async def test_successful_recovery_closes_circuit(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """Test that successful recovery closes the circuit."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"
        success_threshold = circuit_breaker_config["success_threshold"]

        # Start in HALF_OPEN
        await active_state_service.set_circuit_status(agent_id, "half_open")

        # Record sufficient successes
        for _ in range(success_threshold):
            await circuit_breaker.record_success(agent_id)

        # Circuit should be CLOSED
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.CLOSED

        # Requests should be allowed
        can_execute = await circuit_breaker.can_execute(agent_id)
        assert can_execute is True

    async def test_failed_recovery_reopens_circuit(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
    ) -> None:
        """Test that failed recovery reopens the circuit."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"

        # Start in HALF_OPEN
        await active_state_service.set_circuit_status(agent_id, "half_open")

        # Fail during recovery
        await circuit_breaker.record_failure(agent_id, Exception("Recovery failed"))

        # Circuit should be OPEN again
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.OPEN

    async def test_partial_recovery_then_failure(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """Test partial recovery followed by failure."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"
        success_threshold = circuit_breaker_config["success_threshold"]

        # Start in HALF_OPEN
        await active_state_service.set_circuit_status(agent_id, "half_open")

        # Partial recovery (not enough successes)
        for _ in range(success_threshold - 1):
            await circuit_breaker.record_success(agent_id)

        # Then fail
        await circuit_breaker.record_failure(agent_id, Exception("Late failure"))

        # Circuit should be OPEN
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.OPEN


# =============================================================================
# Rerouting Tests
# =============================================================================


class TestCircuitBreakerRerouting:
    """Tests for automatic rerouting when circuit opens."""

    async def test_get_alternative_agent_finds_healthy_agent(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
    ) -> None:
        """Test finding alternative agent when circuit is open."""
        # Open circuit for searcher_v1
        await active_state_service.set_circuit_status("searcher_v1", "open")

        # Find alternative
        alternative = await circuit_breaker.get_alternative_agent(
            failed_agent="searcher_v1",
            required_capabilities=["web_search"],
        )

        # Should find searcher_v2 as alternative
        assert alternative == "searcher_v2"

    async def test_get_alternative_agent_with_matching_type(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
    ) -> None:
        """Test finding alternative agent by type."""
        # Open circuit for searcher_v1
        await active_state_service.set_circuit_status("searcher_v1", "open")

        alternative = await circuit_breaker.get_alternative_agent(
            failed_agent="searcher_v1",
        )

        # Should find another searcher
        assert alternative is not None
        assert "searcher" in alternative

    async def test_get_alternative_agent_no_healthy_alternatives(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
    ) -> None:
        """Test when no healthy alternatives are available."""
        # Open circuits for all searchers
        await active_state_service.set_circuit_status("searcher_v1", "open")
        await active_state_service.set_circuit_status("searcher_v2", "open")

        alternative = await circuit_breaker.get_alternative_agent(
            failed_agent="searcher_v1",
            required_capabilities=["web_search"],
        )

        # No healthy alternative available
        assert alternative is None

    async def test_get_alternative_agent_no_matching_capabilities(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test when no agent has required capabilities."""
        alternative = await circuit_breaker.get_alternative_agent(
            failed_agent="searcher_v1",
            required_capabilities=["nonexistent_capability"],
        )

        # No agent with required capability
        assert alternative is None


# =============================================================================
# Force Open/Close Tests
# =============================================================================


class TestCircuitBreakerManualControl:
    """Tests for manual circuit breaker control."""

    async def test_force_open_circuit(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test manually forcing circuit open."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"

        # Force open
        await circuit_breaker.force_open(agent_id, "Manual test")

        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.OPEN

        # Requests should be blocked
        can_execute = await circuit_breaker.can_execute(agent_id)
        assert can_execute is False

    async def test_force_close_circuit(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
    ) -> None:
        """Test manually forcing circuit closed."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"

        # First open the circuit
        await active_state_service.set_circuit_status(agent_id, "open")

        # Force close
        await circuit_breaker.force_close(agent_id, "Manual recovery")

        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.CLOSED

        # Requests should be allowed
        can_execute = await circuit_breaker.can_execute(agent_id)
        assert can_execute is True

    async def test_reset_all_clears_state(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test that reset_all clears all circuit state."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"

        # Record some failures
        for i in range(3):
            await circuit_breaker.record_failure(agent_id, Exception(f"Error {i}"))

        # Reset
        await circuit_breaker.reset_all(agent_id)

        # State should be CLOSED
        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.CLOSED

        # Stats should be cleared
        stats = await circuit_breaker.get_stats(agent_id)
        assert stats["failure_count"] == 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestCircuitBreakerStatistics:
    """Tests for circuit breaker statistics."""

    async def test_get_stats_returns_correct_structure(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test that get_stats returns expected structure."""
        stats = await circuit_breaker.get_stats("searcher_v1")

        assert "agent_id" in stats
        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "total_failures" in stats
        assert "total_successes" in stats

    async def test_stats_track_failures(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test that stats correctly track failures."""
        agent_id = "searcher_v1"

        # Record some failures
        await circuit_breaker.record_failure(agent_id, Exception("Error 1"))
        await circuit_breaker.record_failure(agent_id, Exception("Error 2"))

        stats = await circuit_breaker.get_stats(agent_id)
        assert stats["failure_count"] == 2
        assert stats["total_failures"] == 2

    async def test_stats_track_successes(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test that stats correctly track successes."""
        agent_id = "searcher_v1"

        # Record some successes
        await circuit_breaker.record_success(agent_id)
        await circuit_breaker.record_success(agent_id)
        await circuit_breaker.record_success(agent_id)

        stats = await circuit_breaker.get_stats(agent_id)
        assert stats["total_successes"] == 3


# =============================================================================
# Event Listener Tests
# =============================================================================


class TestCircuitBreakerEventListeners:
    """Tests for circuit breaker event listeners."""

    async def test_state_change_listener_called(
        self,
        circuit_breaker: Any,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """Test that state change listeners are called."""
        events: list[dict] = []

        def listener(event: dict) -> None:
            events.append(event)

        circuit_breaker.add_state_change_listener(listener)

        agent_id = "searcher_v1"
        threshold = circuit_breaker_config["failure_threshold"]

        # Trigger state change
        for i in range(threshold):
            await circuit_breaker.record_failure(agent_id, Exception(f"Error {i}"))

        # Should have received state change event
        assert len(events) > 0
        assert events[-1]["new_state"] == "open"

    async def test_remove_state_change_listener(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test removing state change listeners."""
        events: list[dict] = []

        def listener(event: dict) -> None:
            events.append(event)

        circuit_breaker.add_state_change_listener(listener)
        circuit_breaker.remove_state_change_listener(listener)

        # Force state change
        await circuit_breaker.force_open("searcher_v1", "Test")

        # Listener should not have been called
        assert len(events) == 0


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestCircuitBreakerConcurrency:
    """Tests for circuit breaker concurrent access."""

    async def test_concurrent_failure_recording(
        self,
        circuit_breaker: Any,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """Test concurrent failure recording is thread-safe."""
        agent_id = "searcher_v1"

        # Record many failures concurrently
        tasks = [
            circuit_breaker.record_failure(agent_id, Exception(f"Error {i}"))
            for i in range(20)
        ]

        await asyncio.gather(*tasks)

        # Circuit should be open (exceeded threshold)
        try:
            from src.metadata.circuit_breaker import CircuitState
            state = await circuit_breaker.get_state(agent_id)
            assert state == CircuitState.OPEN
        except ImportError:
            pytest.skip("CircuitBreaker not available")

    async def test_concurrent_can_execute_checks(
        self,
        circuit_breaker: Any,
    ) -> None:
        """Test concurrent can_execute checks."""
        agent_id = "searcher_v1"

        # Run many concurrent checks
        tasks = [
            circuit_breaker.can_execute(agent_id)
            for _ in range(50)
        ]

        results = await asyncio.gather(*tasks)

        # All should return True (circuit is closed)
        assert all(results)


# =============================================================================
# CI Gate Tests
# =============================================================================


@pytest.mark.ci_gate
class TestCIGateCircuitBreaker:
    """CI gate tests for circuit breaker."""

    async def test_ci_gate_circuit_opens_on_failures(
        self,
        circuit_breaker: Any,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """CI gate: Circuit must open after failure threshold."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"
        threshold = circuit_breaker_config["failure_threshold"]

        for i in range(threshold):
            await circuit_breaker.record_failure(agent_id, Exception(f"Error {i}"))

        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.OPEN, "Circuit must open after threshold failures"

    async def test_ci_gate_circuit_recovers(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
        circuit_breaker_config: dict[str, int],
    ) -> None:
        """CI gate: Circuit must recover after successful requests."""
        try:
            from src.metadata.circuit_breaker import CircuitState
        except ImportError:
            pytest.skip("CircuitBreaker not available")

        agent_id = "searcher_v1"
        success_threshold = circuit_breaker_config["success_threshold"]

        # Start in HALF_OPEN
        await active_state_service.set_circuit_status(agent_id, "half_open")

        for _ in range(success_threshold):
            await circuit_breaker.record_success(agent_id)

        state = await circuit_breaker.get_state(agent_id)
        assert state == CircuitState.CLOSED, "Circuit must close after successful recovery"

    async def test_ci_gate_alternative_agent_found(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
    ) -> None:
        """CI gate: Alternative agent must be found when circuit opens."""
        # Open circuit for primary
        await active_state_service.set_circuit_status("searcher_v1", "open")

        alternative = await circuit_breaker.get_alternative_agent(
            failed_agent="searcher_v1",
        )

        assert alternative is not None, "Must find alternative agent when circuit opens"

    async def test_ci_gate_requests_blocked_when_open(
        self,
        circuit_breaker: Any,
        active_state_service: MockActiveStateService,
    ) -> None:
        """CI gate: Requests must be blocked when circuit is open."""
        agent_id = "searcher_v1"

        # Force open
        await active_state_service.set_circuit_status(agent_id, "open")

        can_execute = await circuit_breaker.can_execute(agent_id)
        assert can_execute is False, "Requests must be blocked when circuit is open"
