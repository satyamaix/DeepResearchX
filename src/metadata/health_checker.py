"""Health Checker Implementation for DRX Deep Research System.

Provides comprehensive health checking for agents, including:
- Error rate monitoring
- Latency threshold checking
- Token burn rate analysis
- Overall health status determination

Integrates with ActiveStateService for metrics and CircuitBreaker for
fault tolerance decisions.

Part of WP-M5: Circuit Breaker Implementation for the DRX spec.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)


# =============================================================================
# TypedDict Definitions
# =============================================================================


class HealthCheckResult(TypedDict):
    """Result of a single health check.

    Attributes:
        check_name: Name of the health check performed
        passed: Whether the check passed
        value: Actual value measured
        threshold: Threshold value for comparison
        message: Human-readable result message
    """

    check_name: str
    passed: bool
    value: float
    threshold: float
    message: str


class HealthStatusDict(TypedDict):
    """Complete health status for an agent.

    Attributes:
        agent_id: The agent identifier
        status: Overall health status (healthy, degraded, unhealthy, unknown)
        is_healthy: Simple boolean indicating health
        timestamp: ISO timestamp of health check
        checks: List of individual health check results
        error_rate: Current error rate
        latency_p50: 50th percentile latency in ms
        latency_p99: 99th percentile latency in ms
        token_burn_rate: Tokens per second burn rate
        circuit_state: Current circuit breaker state
        recommendations: List of recommended actions
    """

    agent_id: str
    status: Literal["healthy", "degraded", "unhealthy", "unknown"]
    is_healthy: bool
    timestamp: str
    checks: list[HealthCheckResult]
    error_rate: float
    latency_p50: float
    latency_p99: float
    token_burn_rate: float
    circuit_state: str
    recommendations: list[str]


class HealthThresholdsDict(TypedDict, total=False):
    """Configurable thresholds for health checks.

    Attributes:
        max_error_rate: Maximum acceptable error rate (0.0-1.0). Default: 0.1
        max_latency_p50_ms: Maximum acceptable p50 latency. Default: 5000
        max_latency_p99_ms: Maximum acceptable p99 latency. Default: 30000
        max_token_burn_rate: Maximum tokens per second. Default: 1000
        min_success_rate: Minimum required success rate. Default: 0.9
        degraded_error_rate: Error rate for degraded status. Default: 0.05
        degraded_latency_p99_ms: P99 latency for degraded status. Default: 15000
    """

    max_error_rate: float
    max_latency_p50_ms: int
    max_latency_p99_ms: int
    max_token_burn_rate: float
    min_success_rate: float
    degraded_error_rate: float
    degraded_latency_p99_ms: int


# =============================================================================
# Default Configuration
# =============================================================================


DEFAULT_HEALTH_THRESHOLDS: HealthThresholdsDict = {
    "max_error_rate": 0.1,
    "max_latency_p50_ms": 5000,
    "max_latency_p99_ms": 30000,
    "max_token_burn_rate": 1000,
    "min_success_rate": 0.9,
    "degraded_error_rate": 0.05,
    "degraded_latency_p99_ms": 15000,
}


# =============================================================================
# HealthChecker Class
# =============================================================================


class HealthChecker:
    """Health checker for agent monitoring and status determination.

    This class performs comprehensive health checks on agents, analyzing
    metrics from ActiveStateService to determine overall health status.
    It integrates with AgentManifest for per-agent threshold configuration.

    The health checker evaluates:
    - Error rate: Percentage of failed requests
    - Latency: Response time percentiles (p50, p99)
    - Token burn rate: Resource consumption rate
    - Circuit breaker state: Current circuit status

    Example:
        >>> from src.services.active_state import ActiveStateService
        >>> health_checker = HealthChecker(active_state_service)
        >>>
        >>> # Quick health check
        >>> if await health_checker.is_healthy("searcher_v1"):
        ...     # Agent is healthy
        ...     pass
        >>>
        >>> # Comprehensive health report
        >>> report = await health_checker.get_health_report("searcher_v1")
        >>> print(f"Status: {report['status']}")

    Attributes:
        active_state: The ActiveStateService for metrics retrieval
        default_thresholds: Default health thresholds
        manifest_registry: Optional registry for per-agent configuration
    """

    def __init__(
        self,
        active_state_service: Any,
        default_thresholds: HealthThresholdsDict | None = None,
        manifest_registry: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the HealthChecker.

        Args:
            active_state_service: The ActiveStateService instance for metrics
            default_thresholds: Optional default health thresholds override
            manifest_registry: Optional registry mapping agent_id to AgentManifest
        """
        self.active_state = active_state_service
        self.default_thresholds = default_thresholds or DEFAULT_HEALTH_THRESHOLDS.copy()
        self.manifest_registry = manifest_registry or {}

        # Cache for recent health checks
        self._health_cache: dict[str, tuple[HealthStatusDict, float]] = {}
        self._cache_ttl_seconds = 5.0

    def _get_agent_thresholds(self, agent_id: str) -> HealthThresholdsDict:
        """Get health thresholds for a specific agent.

        If the agent has a manifest with custom thresholds, those are used.
        Otherwise, falls back to default thresholds.

        Args:
            agent_id: The agent identifier

        Returns:
            HealthThresholdsDict with the agent's thresholds
        """
        if agent_id in self.manifest_registry:
            manifest = self.manifest_registry[agent_id]
            if hasattr(manifest, "health_thresholds"):
                ht = manifest.health_thresholds
                return HealthThresholdsDict(
                    max_error_rate=getattr(
                        ht, "max_error_rate",
                        self.default_thresholds.get("max_error_rate", 0.1)
                    ),
                    max_latency_p50_ms=self.default_thresholds.get("max_latency_p50_ms", 5000),
                    max_latency_p99_ms=getattr(
                        ht, "max_latency_ms",
                        self.default_thresholds.get("max_latency_p99_ms", 30000)
                    ),
                    max_token_burn_rate=self.default_thresholds.get(
                        "max_token_burn_rate", 1000
                    ),
                    min_success_rate=getattr(
                        ht, "min_success_rate",
                        self.default_thresholds.get("min_success_rate", 0.9)
                    ),
                    degraded_error_rate=self.default_thresholds.get(
                        "degraded_error_rate", 0.05
                    ),
                    degraded_latency_p99_ms=self.default_thresholds.get(
                        "degraded_latency_p99_ms", 15000
                    ),
                )
        return self.default_thresholds

    async def check_agent_health(self, agent_id: str) -> HealthStatusDict:
        """Perform a comprehensive health check on an agent.

        Analyzes all health metrics and returns a detailed status report
        including individual check results and recommendations.

        Args:
            agent_id: The agent identifier

        Returns:
            HealthStatusDict with complete health information

        Example:
            >>> status = await health_checker.check_agent_health("searcher_v1")
            >>> if status["status"] == "unhealthy":
            ...     print("Recommendations:", status["recommendations"])
        """
        thresholds = self._get_agent_thresholds(agent_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        checks: list[HealthCheckResult] = []
        recommendations: list[str] = []

        # Get metrics from ActiveStateService
        try:
            metrics = await self.active_state.get_metrics(agent_id)
            error_rate = metrics.get("error_rate_5m", 0.0)
            latency_p50 = metrics.get("latency_p50", 0.0)
            latency_p99 = metrics.get("latency_p99", 0.0)
        except Exception as e:
            logger.warning(f"Failed to get metrics for {agent_id}: {e}")
            error_rate = 0.0
            latency_p50 = 0.0
            latency_p99 = 0.0

        # Get token burn rate
        try:
            token_burn_rate = await self.active_state.get_token_burn_rate(agent_id)
        except Exception as e:
            logger.warning(f"Failed to get token burn rate for {agent_id}: {e}")
            token_burn_rate = 0.0

        # Get circuit state
        try:
            circuit_state = await self.active_state.get_circuit_status(agent_id)
        except Exception as e:
            logger.warning(f"Failed to get circuit status for {agent_id}: {e}")
            circuit_state = "unknown"

        # Perform individual checks
        error_check = self._check_error_rate(
            error_rate,
            thresholds.get("max_error_rate", 0.1),
        )
        checks.append(error_check)
        if not error_check["passed"]:
            recommendations.append(
                f"Error rate ({error_rate:.2%}) exceeds threshold. "
                "Consider investigating recent failures."
            )

        latency_p50_check = self._check_latency(
            latency_p50,
            thresholds.get("max_latency_p50_ms", 5000),
            "p50",
        )
        checks.append(latency_p50_check)

        latency_p99_check = self._check_latency(
            latency_p99,
            thresholds.get("max_latency_p99_ms", 30000),
            "p99",
        )
        checks.append(latency_p99_check)
        if not latency_p99_check["passed"]:
            recommendations.append(
                f"P99 latency ({latency_p99:.0f}ms) is high. "
                "Consider scaling or investigating slow requests."
            )

        token_check = self._check_token_burn(
            token_burn_rate,
            thresholds.get("max_token_burn_rate", 1000),
        )
        checks.append(token_check)
        if not token_check["passed"]:
            recommendations.append(
                f"Token burn rate ({token_burn_rate:.1f}/s) is high. "
                "Consider optimizing prompts or adding rate limiting."
            )

        # Circuit breaker check
        circuit_check = self._check_circuit_state(circuit_state)
        checks.append(circuit_check)
        if not circuit_check["passed"]:
            if circuit_state == "open":
                recommendations.append(
                    "Circuit breaker is OPEN. Agent is not accepting requests. "
                    "Wait for recovery or investigate underlying issues."
                )
            elif circuit_state == "half_open":
                recommendations.append(
                    "Circuit breaker is testing recovery. "
                    "Monitor success rate of test requests."
                )

        # Determine overall status
        status = self._determine_status(checks, error_rate, latency_p99, thresholds)

        health_status = HealthStatusDict(
            agent_id=agent_id,
            status=status,
            is_healthy=status == "healthy",
            timestamp=timestamp,
            checks=checks,
            error_rate=error_rate,
            latency_p50=latency_p50,
            latency_p99=latency_p99,
            token_burn_rate=token_burn_rate,
            circuit_state=circuit_state,
            recommendations=recommendations,
        )

        # Cache the result
        import time
        self._health_cache[agent_id] = (health_status, time.time())

        return health_status

    async def is_healthy(self, agent_id: str) -> bool:
        """Quick health check returning a simple boolean.

        Uses cached results if available and fresh, otherwise performs
        a new health check.

        Args:
            agent_id: The agent identifier

        Returns:
            True if the agent is healthy, False otherwise

        Example:
            >>> if await health_checker.is_healthy("searcher_v1"):
            ...     # Proceed with request
            ...     pass
        """
        import time

        # Check cache first
        if agent_id in self._health_cache:
            cached_status, cached_time = self._health_cache[agent_id]
            if time.time() - cached_time < self._cache_ttl_seconds:
                return cached_status["is_healthy"]

        # Perform fresh check
        status = await self.check_agent_health(agent_id)
        return status["is_healthy"]

    async def get_health_report(self, agent_id: str) -> dict[str, Any]:
        """Get a detailed health report for an agent.

        Returns a comprehensive report suitable for logging, monitoring
        dashboards, or API responses.

        Args:
            agent_id: The agent identifier

        Returns:
            Dictionary with detailed health information

        Example:
            >>> report = await health_checker.get_health_report("searcher_v1")
            >>> print(json.dumps(report, indent=2))
        """
        status = await self.check_agent_health(agent_id)

        # Add additional diagnostic information
        try:
            agent_health = await self.active_state.get_agent_health(agent_id)
            failure_count = agent_health.get("failure_count", 0)
            last_success = agent_health.get("last_success")
            last_failure = agent_health.get("last_failure")
        except Exception:
            failure_count = 0
            last_success = None
            last_failure = None

        report: dict[str, Any] = {
            **status,
            "diagnostic": {
                "failure_count": failure_count,
                "last_success": last_success,
                "last_failure": last_failure,
                "thresholds": self._get_agent_thresholds(agent_id),
            },
        }

        return report

    def _check_error_rate(
        self,
        error_rate: float,
        threshold: float,
    ) -> HealthCheckResult:
        """Check if error rate is within acceptable bounds.

        Args:
            error_rate: Current error rate (0.0-1.0)
            threshold: Maximum acceptable error rate

        Returns:
            HealthCheckResult for the error rate check
        """
        passed = error_rate <= threshold
        return HealthCheckResult(
            check_name="error_rate",
            passed=passed,
            value=error_rate,
            threshold=threshold,
            message=(
                f"Error rate {error_rate:.2%} is within threshold"
                if passed
                else f"Error rate {error_rate:.2%} exceeds threshold {threshold:.2%}"
            ),
        )

    def _check_latency(
        self,
        latency_ms: float,
        threshold_ms: int,
        percentile: str,
    ) -> HealthCheckResult:
        """Check if latency is within acceptable bounds.

        Args:
            latency_ms: Latency value in milliseconds
            threshold_ms: Maximum acceptable latency
            percentile: Percentile label (e.g., "p50", "p99")

        Returns:
            HealthCheckResult for the latency check
        """
        passed = latency_ms <= threshold_ms
        return HealthCheckResult(
            check_name=f"latency_{percentile}",
            passed=passed,
            value=latency_ms,
            threshold=float(threshold_ms),
            message=(
                f"Latency {percentile} ({latency_ms:.0f}ms) is within threshold"
                if passed
                else f"Latency {percentile} ({latency_ms:.0f}ms) exceeds "
                     f"threshold {threshold_ms}ms"
            ),
        )

    def _check_token_burn(
        self,
        burn_rate: float,
        threshold: float,
    ) -> HealthCheckResult:
        """Check if token burn rate is within acceptable bounds.

        Args:
            burn_rate: Current tokens per second burn rate
            threshold: Maximum acceptable burn rate

        Returns:
            HealthCheckResult for the token burn check
        """
        passed = burn_rate <= threshold
        return HealthCheckResult(
            check_name="token_burn_rate",
            passed=passed,
            value=burn_rate,
            threshold=threshold,
            message=(
                f"Token burn rate ({burn_rate:.1f}/s) is within threshold"
                if passed
                else f"Token burn rate ({burn_rate:.1f}/s) exceeds "
                     f"threshold {threshold:.1f}/s"
            ),
        )

    def _check_circuit_state(
        self,
        circuit_state: str,
    ) -> HealthCheckResult:
        """Check if circuit breaker state is acceptable.

        Args:
            circuit_state: Current circuit breaker state

        Returns:
            HealthCheckResult for the circuit state check
        """
        passed = circuit_state in ("closed", "unknown")
        return HealthCheckResult(
            check_name="circuit_state",
            passed=passed,
            value=1.0 if passed else 0.0,
            threshold=1.0,
            message=(
                f"Circuit breaker is {circuit_state}"
                if passed
                else f"Circuit breaker is {circuit_state} - agent may be impaired"
            ),
        )

    def _determine_status(
        self,
        checks: list[HealthCheckResult],
        error_rate: float,
        latency_p99: float,
        thresholds: HealthThresholdsDict,
    ) -> Literal["healthy", "degraded", "unhealthy", "unknown"]:
        """Determine overall health status from check results.

        Args:
            checks: List of individual health check results
            error_rate: Current error rate
            latency_p99: Current p99 latency
            thresholds: Health thresholds for evaluation

        Returns:
            Overall health status
        """
        # Count failures
        failed_checks = [c for c in checks if not c["passed"]]
        critical_failures = [
            c for c in failed_checks
            if c["check_name"] in ("error_rate", "circuit_state")
        ]

        # Unhealthy if circuit is open or error rate exceeds threshold
        if critical_failures:
            return "unhealthy"

        # Unhealthy if multiple checks fail
        if len(failed_checks) >= 2:
            return "unhealthy"

        # Check for degraded conditions
        degraded_error_rate = thresholds.get("degraded_error_rate", 0.05)
        degraded_latency = thresholds.get("degraded_latency_p99_ms", 15000)

        if error_rate > degraded_error_rate or latency_p99 > degraded_latency:
            return "degraded"

        # Check if any non-critical checks failed
        if failed_checks:
            return "degraded"

        return "healthy"

    async def get_all_agent_health(self) -> dict[str, HealthStatusDict]:
        """Get health status for all known agents.

        Returns:
            Dictionary mapping agent_id to HealthStatusDict
        """
        results: dict[str, HealthStatusDict] = {}

        try:
            agent_ids = await self.active_state.get_all_agent_ids()
            for agent_id in agent_ids:
                try:
                    results[agent_id] = await self.check_agent_health(agent_id)
                except Exception as e:
                    logger.warning(f"Failed to check health for {agent_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to get agent IDs: {e}")

        return results

    async def get_unhealthy_agents(self) -> list[str]:
        """Get list of unhealthy agent IDs.

        Returns:
            List of agent IDs with unhealthy or degraded status
        """
        all_health = await self.get_all_agent_health()
        return [
            agent_id
            for agent_id, status in all_health.items()
            if status["status"] in ("unhealthy", "degraded")
        ]

    def clear_cache(self, agent_id: str | None = None) -> None:
        """Clear health check cache.

        Args:
            agent_id: Optional specific agent to clear, or None for all
        """
        if agent_id:
            self._health_cache.pop(agent_id, None)
        else:
            self._health_cache.clear()


# =============================================================================
# Factory Functions
# =============================================================================


async def create_health_checker(
    active_state_service: Any,
    thresholds: HealthThresholdsDict | None = None,
    manifest_registry: dict[str, Any] | None = None,
) -> HealthChecker:
    """Create a HealthChecker instance.

    Factory function to create and configure a HealthChecker.

    Args:
        active_state_service: The ActiveStateService instance
        thresholds: Optional default health thresholds
        manifest_registry: Optional agent manifest registry

    Returns:
        Configured HealthChecker instance

    Example:
        >>> from src.services.active_state import create_active_state_service
        >>> active_state = await create_active_state_service("redis://localhost:6379/0")
        >>> health_checker = await create_health_checker(active_state)
    """
    return HealthChecker(
        active_state_service=active_state_service,
        default_thresholds=thresholds,
        manifest_registry=manifest_registry,
    )


def create_health_checker_sync(
    active_state_service: Any,
    thresholds: HealthThresholdsDict | None = None,
    manifest_registry: dict[str, Any] | None = None,
) -> HealthChecker:
    """Create a HealthChecker instance synchronously.

    Args:
        active_state_service: The ActiveStateService instance
        thresholds: Optional default health thresholds
        manifest_registry: Optional agent manifest registry

    Returns:
        HealthChecker instance
    """
    return HealthChecker(
        active_state_service=active_state_service,
        default_thresholds=thresholds,
        manifest_registry=manifest_registry,
    )


# =============================================================================
# Module-level Singleton
# =============================================================================


_health_checker: HealthChecker | None = None
_hc_lock = asyncio.Lock()


async def get_health_checker(
    active_state_service: Any | None = None,
    thresholds: HealthThresholdsDict | None = None,
) -> HealthChecker:
    """Get or create the HealthChecker singleton.

    Provides a module-level singleton for shared use across the application.

    Args:
        active_state_service: ActiveStateService (required on first call)
        thresholds: Optional health thresholds

    Returns:
        HealthChecker singleton instance

    Raises:
        ValueError: If active_state_service not provided on first call
    """
    global _health_checker

    if _health_checker is not None:
        return _health_checker

    async with _hc_lock:
        if _health_checker is not None:
            return _health_checker

        if active_state_service is None:
            raise ValueError(
                "active_state_service must be provided on first call"
            )

        _health_checker = await create_health_checker(
            active_state_service=active_state_service,
            thresholds=thresholds,
        )
        return _health_checker


def reset_health_checker_singleton() -> None:
    """Reset the HealthChecker singleton.

    Used primarily for testing to reset global state.
    """
    global _health_checker
    _health_checker = None


# =============================================================================
# Type Exports
# =============================================================================


__all__ = [
    # TypedDicts
    "HealthCheckResult",
    "HealthStatusDict",
    "HealthThresholdsDict",
    # Constants
    "DEFAULT_HEALTH_THRESHOLDS",
    # Main class
    "HealthChecker",
    # Factory functions
    "create_health_checker",
    "create_health_checker_sync",
    "get_health_checker",
    "reset_health_checker_singleton",
]
