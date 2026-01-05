"""
Policy Firewall Middleware for DRX Deep Research Platform.

Provides comprehensive policy enforcement for tool invocations including:
- Domain validation (allowed/blocked domains)
- Budget enforcement (max spend per session/agent)
- Rate limiting (requests and tokens per minute)
- Capability validation (tool permissions)

Integrates with:
- Agent Manifest (WP-M1) for policy configuration
- Active State Redis Service (WP-M2) for real-time tracking
- Event streaming for violation notifications
- PostgreSQL for violation audit logging

Part of WP-M6: Metadata Firewall Middleware implementation.

Usage:
    from src.middleware import PolicyFirewall

    firewall = PolicyFirewall(
        active_state_service=state_service,
        db_pool=pool,
    )

    # Check policy before tool execution
    result = await firewall.check_policy(
        agent_id="searcher_v1",
        tool_name="web_search",
        tool_args={"url": "https://example.com"},
    )

    if result["allowed"]:
        # Execute tool
        pass
    else:
        # Handle violation
        for violation in result["violations"]:
            await firewall.log_violation(agent_id, violation)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypedDict, TypeVar

from src.middleware.domain_validator import DomainValidator

if TYPE_CHECKING:
    from src.metadata.manifest import AgentManifest
    from src.services.active_state import ActiveStateService
    from src.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# =============================================================================
# TypedDict Definitions
# =============================================================================


class PolicyViolation(TypedDict):
    """
    Record of a policy violation detected during tool invocation.

    Attributes:
        violation_id: Unique identifier for this violation
        agent_id: ID of the agent that triggered the violation
        violation_type: Category of the violation
        severity: How serious the violation is
        details: Additional context about the violation
        timestamp: ISO 8601 timestamp when violation occurred
    """

    violation_id: str
    agent_id: str
    violation_type: Literal[
        "domain_blocked",
        "budget_exceeded",
        "rate_limited",
        "capability_denied",
    ]
    severity: Literal["warning", "error", "critical"]
    details: dict[str, Any]
    timestamp: str


class PolicyCheckResult(TypedDict):
    """
    Result of a policy check for a tool invocation.

    Attributes:
        allowed: Whether the tool invocation is allowed
        reason: Human-readable explanation if not allowed
        violations: List of policy violations detected
        recommendations: Suggested actions to resolve violations
    """

    allowed: bool
    reason: str | None
    violations: list[PolicyViolation]
    recommendations: list[str]


class PolicyConfig(TypedDict, total=False):
    """
    Configuration for policy enforcement behavior.

    Attributes:
        enforce_domains: Whether to enforce domain restrictions
        enforce_budget: Whether to enforce budget limits
        enforce_rate_limits: Whether to enforce rate limits
        enforce_capabilities: Whether to enforce tool capabilities
        log_violations: Whether to log violations to database
        emit_events: Whether to emit violation events
        default_budget_usd: Default budget if not specified in manifest
        default_rpm: Default requests per minute if not specified
        default_tpm: Default tokens per minute if not specified
    """

    enforce_domains: bool
    enforce_budget: bool
    enforce_rate_limits: bool
    enforce_capabilities: bool
    log_violations: bool
    emit_events: bool
    default_budget_usd: float
    default_rpm: int
    default_tpm: int


class BudgetState(TypedDict):
    """
    Current budget state for an agent/session.

    Attributes:
        total_spent: Total amount spent in USD
        max_budget: Maximum budget in USD
        remaining: Remaining budget in USD
        last_updated: ISO timestamp of last update
    """

    total_spent: float
    max_budget: float
    remaining: float
    last_updated: str


class RateLimitState(TypedDict):
    """
    Current rate limit state for an agent.

    Attributes:
        requests_count: Number of requests in current window
        requests_limit: Maximum requests per minute
        tokens_count: Number of tokens in current window
        tokens_limit: Maximum tokens per minute
        window_start: ISO timestamp of window start
        retry_after_seconds: Seconds until rate limit resets
    """

    requests_count: int
    requests_limit: int
    tokens_count: int
    tokens_limit: int
    window_start: str
    retry_after_seconds: float | None


# =============================================================================
# Exceptions
# =============================================================================


class PolicyFirewallError(Exception):
    """Base exception for policy firewall errors."""

    pass


class BudgetExceededError(PolicyFirewallError):
    """Raised when an agent exceeds their budget."""

    def __init__(self, agent_id: str, current_spend: float, max_budget: float):
        self.agent_id = agent_id
        self.current_spend = current_spend
        self.max_budget = max_budget
        super().__init__(
            f"Agent {agent_id} has exceeded budget: "
            f"${current_spend:.4f} / ${max_budget:.4f}"
        )


class RateLimitExceededError(PolicyFirewallError):
    """Raised when an agent exceeds their rate limit."""

    def __init__(
        self,
        agent_id: str,
        limit_type: Literal["requests", "tokens"],
        current: int,
        limit: int,
        retry_after: float,
    ):
        self.agent_id = agent_id
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(
            f"Agent {agent_id} rate limited ({limit_type}): "
            f"{current}/{limit}, retry after {retry_after:.1f}s"
        )


class DomainBlockedError(PolicyFirewallError):
    """Raised when an agent attempts to access a blocked domain."""

    def __init__(self, agent_id: str, domain: str, reason: str):
        self.agent_id = agent_id
        self.domain = domain
        self.reason = reason
        super().__init__(f"Domain {domain} blocked for agent {agent_id}: {reason}")


class CapabilityDeniedError(PolicyFirewallError):
    """Raised when an agent lacks capability for a tool."""

    def __init__(self, agent_id: str, tool_name: str, required_capability: str):
        self.agent_id = agent_id
        self.tool_name = tool_name
        self.required_capability = required_capability
        super().__init__(
            f"Agent {agent_id} lacks capability '{required_capability}' "
            f"required for tool '{tool_name}'"
        )


# =============================================================================
# Constants
# =============================================================================

# Redis key patterns for policy state
BUDGET_KEY_PREFIX = "drx:policy:budget"
RATE_LIMIT_KEY_PREFIX = "drx:policy:ratelimit"
VIOLATION_STREAM_KEY = "drx:events:policy_violations"

# Default policy configuration
DEFAULT_POLICY_CONFIG: PolicyConfig = {
    "enforce_domains": True,
    "enforce_budget": True,
    "enforce_rate_limits": True,
    "enforce_capabilities": True,
    "log_violations": True,
    "emit_events": True,
    "default_budget_usd": 1.0,
    "default_rpm": 60,
    "default_tpm": 100000,
}

# Violation severity mapping
VIOLATION_SEVERITY_MAP: dict[str, Literal["warning", "error", "critical"]] = {
    "domain_blocked": "error",
    "budget_exceeded": "critical",
    "rate_limited": "warning",
    "capability_denied": "error",
}

# Tool to capability mapping (tool_name -> required capabilities)
TOOL_CAPABILITY_MAP: dict[str, list[str]] = {
    "web_search": ["web_search", "source_discovery"],
    "url_fetch": ["content_extraction", "web_search"],
    "html_parse": ["content_extraction", "text_processing"],
    "pdf_extract": ["content_extraction"],
    "synthesize": ["information_synthesis"],
    "report_create": ["report_generation"],
}


# =============================================================================
# PolicyFirewall Class
# =============================================================================


class PolicyFirewall:
    """
    Middleware for enforcing agent policies on tool invocations.

    The PolicyFirewall intercepts tool invocations and validates them against
    the agent's manifest configuration. It enforces:

    1. Domain restrictions (allowed/blocked domains for web access)
    2. Budget limits (maximum spend per session/agent)
    3. Rate limits (requests and tokens per minute)
    4. Capability requirements (tool-specific permissions)

    Violations are logged to the database and optionally emitted as events
    for real-time monitoring.

    Example:
        >>> firewall = PolicyFirewall(state_service, db_pool)
        >>> result = await firewall.check_policy(
        ...     agent_id="searcher_v1",
        ...     tool_name="web_search",
        ...     tool_args={"url": "https://example.com"}
        ... )
        >>> if result["allowed"]:
        ...     # Proceed with tool execution
        ...     pass

    Attributes:
        active_state: Redis service for real-time state
        db_pool: PostgreSQL connection pool for logging
        domain_validator: Validator for URL/domain checks
        config: Policy enforcement configuration
    """

    def __init__(
        self,
        active_state_service: "ActiveStateService | None" = None,
        db_pool: Any = None,
        domain_validator: DomainValidator | None = None,
        config: PolicyConfig | None = None,
        manifest_loader: Callable[[str], Any] | None = None,
    ) -> None:
        """
        Initialize the PolicyFirewall middleware.

        Args:
            active_state_service: Redis service for real-time tracking
            db_pool: PostgreSQL pool for violation logging
            domain_validator: Custom domain validator instance
            config: Policy configuration overrides
            manifest_loader: Custom function to load agent manifests
        """
        self.active_state = active_state_service
        self.db_pool = db_pool
        self.domain_validator = domain_validator or DomainValidator(
            manifest_loader=manifest_loader
        )
        self.config = {**DEFAULT_POLICY_CONFIG, **(config or {})}
        self._manifest_loader = manifest_loader
        self._manifest_cache: dict[str, "AgentManifest"] = {}
        self._lock = asyncio.Lock()

    def _get_config_float(self, key: str, default: float) -> float:
        """Get a float config value with type safety."""
        value = self.config.get(key, default)
        return float(value) if value is not None else default

    def _get_config_int(self, key: str, default: int) -> int:
        """Get an int config value with type safety."""
        value = self.config.get(key, default)
        return int(value) if value is not None else default

    def _get_config_bool(self, key: str, default: bool) -> bool:
        """Get a bool config value with type safety."""
        value = self.config.get(key, default)
        return bool(value) if value is not None else default

    async def _get_manifest(self, agent_id: str) -> "AgentManifest | None":
        """
        Get agent manifest with caching.

        Args:
            agent_id: The agent identifier

        Returns:
            AgentManifest if found, None otherwise
        """
        if agent_id in self._manifest_cache:
            return self._manifest_cache[agent_id]

        if self._manifest_loader:
            try:
                manifest = await self._manifest_loader(agent_id)
                self._manifest_cache[agent_id] = manifest
                return manifest
            except Exception as e:
                logger.warning(f"Failed to load manifest for {agent_id}: {e}")
                return None

        return None

    def clear_cache(self) -> None:
        """Clear all internal caches."""
        self._manifest_cache.clear()
        self.domain_validator.clear_cache()

    # =========================================================================
    # Main Policy Check Method
    # =========================================================================

    async def check_policy(
        self,
        agent_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        session_id: str | None = None,
        estimated_cost: float | None = None,
    ) -> PolicyCheckResult:
        """
        Check if a tool invocation is allowed by policy.

        Performs comprehensive policy validation including:
        1. Domain validation (if URL present in args)
        2. Budget check (if cost estimation available)
        3. Rate limit check
        4. Capability validation

        Args:
            agent_id: The agent attempting the invocation
            tool_name: Name of the tool being invoked
            tool_args: Arguments being passed to the tool
            session_id: Optional session ID for session-scoped tracking
            estimated_cost: Optional estimated cost of the invocation

        Returns:
            PolicyCheckResult with allowed status and any violations

        Example:
            >>> result = await firewall.check_policy(
            ...     agent_id="searcher_v1",
            ...     tool_name="web_search",
            ...     tool_args={"url": "https://blocked.com"}
            ... )
            >>> print(result["allowed"])
            False
            >>> print(result["violations"][0]["violation_type"])
            'domain_blocked'
        """
        violations: list[PolicyViolation] = []
        recommendations: list[str] = []

        # Get agent manifest
        manifest = await self._get_manifest(agent_id)

        # 1. Check domain restrictions
        if self._get_config_bool("enforce_domains", True):
            domain_violations = await self._check_domain_policy(
                agent_id=agent_id,
                tool_name=tool_name,
                tool_args=tool_args,
                manifest=manifest,
            )
            violations.extend(domain_violations)
            if domain_violations:
                recommendations.append(
                    "Remove blocked URLs or use allowed domains"
                )

        # 2. Check budget limits
        if self._get_config_bool("enforce_budget", True) and estimated_cost:
            budget_violation = await self._check_budget_policy(
                agent_id=agent_id,
                estimated_cost=estimated_cost,
                session_id=session_id,
                manifest=manifest,
            )
            if budget_violation:
                violations.append(budget_violation)
                recommendations.append(
                    "Reduce operation scope or request budget increase"
                )

        # 3. Check rate limits
        if self._get_config_bool("enforce_rate_limits", True):
            rate_limit_violation = await self._check_rate_limit_policy(
                agent_id=agent_id,
                manifest=manifest,
            )
            if rate_limit_violation:
                violations.append(rate_limit_violation)
                recommendations.append(
                    f"Wait {rate_limit_violation['details'].get('retry_after', 60)}s before retrying"
                )

        # 4. Check capability requirements
        if self._get_config_bool("enforce_capabilities", True):
            capability_violation = await self._check_capability_policy(
                agent_id=agent_id,
                tool_name=tool_name,
                manifest=manifest,
            )
            if capability_violation:
                violations.append(capability_violation)
                recommendations.append(
                    "Use an agent with required capabilities"
                )

        # Determine if allowed
        allowed = len(violations) == 0
        reason = None if allowed else violations[0]["details"].get("message", "Policy violation")

        return PolicyCheckResult(
            allowed=allowed,
            reason=reason,
            violations=violations,
            recommendations=recommendations,
        )

    # =========================================================================
    # Domain Policy Check
    # =========================================================================

    async def _check_domain_policy(
        self,
        agent_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        manifest: "AgentManifest | None" = None,
    ) -> list[PolicyViolation]:
        """
        Check domain restrictions for URLs in tool arguments.

        Extracts URLs from tool arguments and validates them against
        the agent's allowed/blocked domain lists.

        Args:
            agent_id: The agent identifier
            tool_name: Name of the tool being invoked
            tool_args: Tool arguments (may contain URLs)
            manifest: Optional pre-loaded manifest

        Returns:
            List of domain-related violations (empty if all URLs allowed)
        """
        violations: list[PolicyViolation] = []

        # Get domain lists from manifest
        allowed_domains: list[str] = []
        blocked_domains: list[str] = []

        if manifest:
            allowed_domains = list(manifest.allowed_domains)
            blocked_domains = list(manifest.blocked_domains)

        # Extract URLs from tool arguments
        urls = self._extract_urls_from_args(tool_args)

        if not urls:
            return violations

        # Validate each URL
        for url in urls:
            try:
                is_allowed, reason = await self.domain_validator.validate_url(
                    agent_id=agent_id,
                    url=url,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )

                if not is_allowed:
                    violation = PolicyViolation(
                        violation_id=str(uuid.uuid4()),
                        agent_id=agent_id,
                        violation_type="domain_blocked",
                        severity=VIOLATION_SEVERITY_MAP["domain_blocked"],
                        details={
                            "message": reason or f"Domain not allowed for URL: {url}",
                            "url": url,
                            "domain": self.domain_validator.extract_domain(url),
                            "tool_name": tool_name,
                            "allowed_domains": allowed_domains[:10],  # Truncate for logs
                            "blocked_domains": blocked_domains[:10],
                        },
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    violations.append(violation)

            except Exception as e:
                logger.warning(f"Error validating URL {url}: {e}")
                # Invalid URL - create a violation
                violation = PolicyViolation(
                    violation_id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    violation_type="domain_blocked",
                    severity="warning",
                    details={
                        "message": f"Invalid URL format: {e}",
                        "url": url,
                        "tool_name": tool_name,
                        "error": str(e),
                    },
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                violations.append(violation)

        return violations

    def _extract_urls_from_args(self, args: dict[str, Any]) -> list[str]:
        """
        Extract URL values from tool arguments.

        Looks for common URL argument names and extracts their values.

        Args:
            args: Tool arguments dictionary

        Returns:
            List of URL strings found in arguments
        """
        urls: list[str] = []
        url_keys = ["url", "urls", "link", "links", "source", "sources", "target", "endpoint"]

        for key, value in args.items():
            # Check known URL keys
            if key.lower() in url_keys:
                if isinstance(value, str):
                    urls.append(value)
                elif isinstance(value, list):
                    urls.extend(v for v in value if isinstance(v, str))

            # Check for URL-like values in any key
            elif isinstance(value, str) and self._looks_like_url(value):
                urls.append(value)

        return urls

    def _looks_like_url(self, value: str) -> bool:
        """Check if a string looks like a URL."""
        return value.startswith(("http://", "https://", "ftp://", "//"))

    # =========================================================================
    # Budget Policy Check
    # =========================================================================

    async def _check_budget_policy(
        self,
        agent_id: str,
        estimated_cost: float,
        session_id: str | None = None,
        manifest: "AgentManifest | None" = None,
    ) -> PolicyViolation | None:
        """
        Check if estimated cost would exceed agent's budget.

        Args:
            agent_id: The agent identifier
            estimated_cost: Estimated cost of the operation in USD
            session_id: Optional session ID for session-scoped budget
            manifest: Optional pre-loaded manifest

        Returns:
            PolicyViolation if budget would be exceeded, None otherwise
        """
        # Get max budget from manifest or default
        max_budget = self._get_config_float("default_budget_usd", 1.0)
        if manifest:
            max_budget = manifest.max_budget_usd

        # Get current spend
        current_spend = await self._get_current_spend(agent_id, session_id)

        # Check if this operation would exceed budget
        projected_spend = current_spend + estimated_cost

        if projected_spend > max_budget:
            return PolicyViolation(
                violation_id=str(uuid.uuid4()),
                agent_id=agent_id,
                violation_type="budget_exceeded",
                severity=VIOLATION_SEVERITY_MAP["budget_exceeded"],
                details={
                    "message": (
                        f"Operation would exceed budget: "
                        f"${projected_spend:.4f} > ${max_budget:.4f}"
                    ),
                    "current_spend": current_spend,
                    "estimated_cost": estimated_cost,
                    "projected_spend": projected_spend,
                    "max_budget": max_budget,
                    "remaining_budget": max_budget - current_spend,
                    "session_id": session_id,
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        return None

    async def _get_current_spend(
        self,
        agent_id: str,
        session_id: str | None = None,
    ) -> float:
        """
        Get the current spend for an agent/session.

        Args:
            agent_id: The agent identifier
            session_id: Optional session ID for session-scoped tracking

        Returns:
            Current spend in USD
        """
        if self.active_state:
            try:
                # Build the budget key
                key = f"{BUDGET_KEY_PREFIX}:{agent_id}"
                if session_id:
                    key = f"{key}:{session_id}"

                spend_str = await self.active_state.redis.get(key)
                if spend_str:
                    return float(spend_str)
            except Exception as e:
                logger.warning(f"Failed to get current spend from Redis: {e}")

        return 0.0

    async def _record_spend(
        self,
        agent_id: str,
        amount: float,
        session_id: str | None = None,
    ) -> float:
        """
        Record spend for an agent/session.

        Args:
            agent_id: The agent identifier
            amount: Amount spent in USD
            session_id: Optional session ID

        Returns:
            New total spend
        """
        if self.active_state:
            try:
                key = f"{BUDGET_KEY_PREFIX}:{agent_id}"
                if session_id:
                    key = f"{key}:{session_id}"

                # Atomically increment spend
                new_total = await self.active_state.redis.incrbyfloat(key, amount)

                # Set expiration (24 hours for session budgets)
                if session_id:
                    await self.active_state.redis.expire(key, 86400)

                return float(new_total)

            except Exception as e:
                logger.warning(f"Failed to record spend in Redis: {e}")

        return amount

    async def enforce_budget(
        self,
        agent_id: str,
        estimated_cost: float,
        session_id: str | None = None,
    ) -> bool:
        """
        Enforce budget limits before an operation.

        Checks if the operation would exceed budget and returns whether
        it should proceed. Does not record the spend (call _record_spend
        after successful operation).

        Args:
            agent_id: The agent identifier
            estimated_cost: Estimated cost of the operation
            session_id: Optional session ID

        Returns:
            True if operation is within budget, False otherwise

        Raises:
            BudgetExceededError: If operation would exceed budget and
                                 enforcement is enabled
        """
        manifest = await self._get_manifest(agent_id)
        violation = await self._check_budget_policy(
            agent_id=agent_id,
            estimated_cost=estimated_cost,
            session_id=session_id,
            manifest=manifest,
        )

        if violation:
            if self._get_config_bool("log_violations", True):
                await self.log_violation(agent_id, violation, session_id)

            max_budget = manifest.max_budget_usd if manifest else self._get_config_float("default_budget_usd", 1.0)
            current_spend = await self._get_current_spend(agent_id, session_id)

            raise BudgetExceededError(
                agent_id=agent_id,
                current_spend=current_spend,
                max_budget=max_budget,
            )

        return True

    # =========================================================================
    # Rate Limit Policy Check
    # =========================================================================

    async def _check_rate_limit_policy(
        self,
        agent_id: str,
        manifest: "AgentManifest | None" = None,
    ) -> PolicyViolation | None:
        """
        Check if agent is within rate limits.

        Args:
            agent_id: The agent identifier
            manifest: Optional pre-loaded manifest

        Returns:
            PolicyViolation if rate limited, None otherwise
        """
        # Get rate limits from manifest or defaults
        rpm_limit: int = self._get_config_int("default_rpm", 60)
        tpm_limit: int = self._get_config_int("default_tpm", 100000)

        if manifest:
            rpm_limit = manifest.rate_limits.requests_per_minute
            tpm_limit = manifest.rate_limits.tokens_per_minute

        # Get current rate limit state
        rate_state = await self._get_rate_limit_state(agent_id)

        # Check requests per minute
        if rate_state["requests_count"] >= rpm_limit:
            retry_after = rate_state.get("retry_after_seconds", 60.0) or 60.0
            return PolicyViolation(
                violation_id=str(uuid.uuid4()),
                agent_id=agent_id,
                violation_type="rate_limited",
                severity=VIOLATION_SEVERITY_MAP["rate_limited"],
                details={
                    "message": f"Rate limit exceeded: {rate_state['requests_count']}/{rpm_limit} requests/min",
                    "limit_type": "requests",
                    "current": rate_state["requests_count"],
                    "limit": rpm_limit,
                    "retry_after": retry_after,
                    "window_start": rate_state["window_start"],
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Check tokens per minute
        if rate_state["tokens_count"] >= tpm_limit:
            retry_after = rate_state.get("retry_after_seconds", 60.0) or 60.0
            return PolicyViolation(
                violation_id=str(uuid.uuid4()),
                agent_id=agent_id,
                violation_type="rate_limited",
                severity=VIOLATION_SEVERITY_MAP["rate_limited"],
                details={
                    "message": f"Token limit exceeded: {rate_state['tokens_count']}/{tpm_limit} tokens/min",
                    "limit_type": "tokens",
                    "current": rate_state["tokens_count"],
                    "limit": tpm_limit,
                    "retry_after": retry_after,
                    "window_start": rate_state["window_start"],
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        return None

    async def _get_rate_limit_state(self, agent_id: str) -> RateLimitState:
        """
        Get current rate limit state for an agent.

        Args:
            agent_id: The agent identifier

        Returns:
            RateLimitState with current counts and limits
        """
        default_state: RateLimitState = {
            "requests_count": 0,
            "requests_limit": self._get_config_int("default_rpm", 60),
            "tokens_count": 0,
            "tokens_limit": self._get_config_int("default_tpm", 100000),
            "window_start": datetime.now(timezone.utc).isoformat(),
            "retry_after_seconds": None,
        }

        if not self.active_state:
            return default_state

        try:
            key = f"{RATE_LIMIT_KEY_PREFIX}:{agent_id}"

            # Use pipeline to get all rate limit data atomically
            async with self.active_state.redis.pipeline() as pipe:
                await pipe.hgetall(key)
                await pipe.ttl(key)
                results = await pipe.execute()

            data = results[0] or {}
            ttl = results[1] if results[1] > 0 else 60

            return RateLimitState(
                requests_count=int(data.get("requests_count", 0)),
                requests_limit=int(data.get("requests_limit", default_state["requests_limit"])),
                tokens_count=int(data.get("tokens_count", 0)),
                tokens_limit=int(data.get("tokens_limit", default_state["tokens_limit"])),
                window_start=data.get("window_start", default_state["window_start"]),
                retry_after_seconds=float(ttl) if ttl > 0 else None,
            )

        except Exception as e:
            logger.warning(f"Failed to get rate limit state from Redis: {e}")
            return default_state

    async def _increment_rate_limit(
        self,
        agent_id: str,
        requests: int = 1,
        tokens: int = 0,
    ) -> RateLimitState:
        """
        Increment rate limit counters for an agent.

        Args:
            agent_id: The agent identifier
            requests: Number of requests to add
            tokens: Number of tokens to add

        Returns:
            Updated RateLimitState
        """
        if not self.active_state:
            return await self._get_rate_limit_state(agent_id)

        try:
            key = f"{RATE_LIMIT_KEY_PREFIX}:{agent_id}"
            now = datetime.now(timezone.utc).isoformat()

            async with self.active_state.redis.pipeline(transaction=True) as pipe:
                # Increment counters
                await pipe.hincrby(key, "requests_count", requests)
                await pipe.hincrby(key, "tokens_count", tokens)
                await pipe.hset(key, "window_start", now)

                # Set 1-minute expiration (sliding window)
                await pipe.expire(key, 60)

                results = await pipe.execute()

            return RateLimitState(
                requests_count=int(results[0]),
                requests_limit=self._get_config_int("default_rpm", 60),
                tokens_count=int(results[1]),
                tokens_limit=self._get_config_int("default_tpm", 100000),
                window_start=now,
                retry_after_seconds=60.0,
            )

        except Exception as e:
            logger.warning(f"Failed to increment rate limit in Redis: {e}")
            return await self._get_rate_limit_state(agent_id)

    async def enforce_rate_limit(
        self,
        agent_id: str,
        tokens: int = 0,
    ) -> bool:
        """
        Enforce rate limits before an operation.

        Checks current rate limit state and increments counters.

        Args:
            agent_id: The agent identifier
            tokens: Number of tokens this operation will use

        Returns:
            True if within rate limits, False otherwise

        Raises:
            RateLimitExceededError: If rate limit exceeded and enforcement enabled
        """
        manifest = await self._get_manifest(agent_id)
        violation = await self._check_rate_limit_policy(agent_id, manifest)

        if violation:
            if self._get_config_bool("log_violations", True):
                await self.log_violation(agent_id, violation)

            details = violation["details"]
            raise RateLimitExceededError(
                agent_id=agent_id,
                limit_type=details.get("limit_type", "requests"),
                current=details.get("current", 0),
                limit=details.get("limit", 60),
                retry_after=details.get("retry_after", 60.0),
            )

        # Increment rate limit counters
        await self._increment_rate_limit(agent_id, requests=1, tokens=tokens)

        return True

    # =========================================================================
    # Capability Policy Check
    # =========================================================================

    async def _check_capability_policy(
        self,
        agent_id: str,
        tool_name: str,
        manifest: "AgentManifest | None" = None,
    ) -> PolicyViolation | None:
        """
        Check if agent has capability to use a tool.

        Args:
            agent_id: The agent identifier
            tool_name: Name of the tool being invoked
            manifest: Optional pre-loaded manifest

        Returns:
            PolicyViolation if capability denied, None otherwise
        """
        if not manifest:
            # Without manifest, allow by default
            return None

        # Get required capabilities for tool
        required_capabilities = TOOL_CAPABILITY_MAP.get(tool_name, [])

        if not required_capabilities:
            # Tool has no specific capability requirements
            return None

        # Get agent capabilities
        agent_capabilities = set(manifest.capabilities)

        # Check if agent has at least one required capability
        has_capability = bool(agent_capabilities.intersection(required_capabilities))

        # Also check allowed_tools
        if tool_name in manifest.allowed_tools:
            has_capability = True

        if not has_capability:
            return PolicyViolation(
                violation_id=str(uuid.uuid4()),
                agent_id=agent_id,
                violation_type="capability_denied",
                severity=VIOLATION_SEVERITY_MAP["capability_denied"],
                details={
                    "message": f"Agent lacks capability for tool '{tool_name}'",
                    "tool_name": tool_name,
                    "required_capabilities": required_capabilities,
                    "agent_capabilities": list(agent_capabilities),
                    "allowed_tools": manifest.allowed_tools,
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        return None

    # =========================================================================
    # Violation Logging
    # =========================================================================

    async def log_violation(
        self,
        agent_id: str,
        violation: PolicyViolation,
        session_id: str | None = None,
        step_id: str | None = None,
        tool_invocation_id: str | None = None,
    ) -> None:
        """
        Log a policy violation to the database.

        Args:
            agent_id: The agent identifier
            violation: The violation to log
            session_id: Optional session context
            step_id: Optional step context
            tool_invocation_id: Optional tool invocation context
        """
        if not self._get_config_bool("log_violations", True):
            return

        logger.warning(
            f"Policy violation: {violation['violation_type']} - "
            f"Agent: {agent_id}, Details: {violation['details'].get('message', 'Unknown')}"
        )

        # Log to database
        if self.db_pool:
            await self._write_violation_to_db(
                violation=violation,
                session_id=session_id,
                step_id=step_id,
                tool_invocation_id=tool_invocation_id,
            )

        # Emit event
        if self._get_config_bool("emit_events", True):
            await self._emit_violation_event(
                violation=violation,
                session_id=session_id,
            )

    async def _write_violation_to_db(
        self,
        violation: PolicyViolation,
        session_id: str | None = None,
        step_id: str | None = None,
        tool_invocation_id: str | None = None,
    ) -> None:
        """
        Write violation record to PostgreSQL.

        Args:
            violation: The violation to write
            session_id: Optional session context
            step_id: Optional step context
            tool_invocation_id: Optional tool invocation context
        """
        if not self.db_pool:
            return

        try:
            query = """
                INSERT INTO policy_violations (
                    id,
                    session_id,
                    step_id,
                    agent_id,
                    tool_invocation_id,
                    violation_type,
                    violation_code,
                    severity,
                    description,
                    context,
                    trigger_action,
                    trigger_input,
                    blocked_execution,
                    created_at
                ) VALUES (
                    $1::uuid,
                    $2::uuid,
                    $3::uuid,
                    $4,
                    $5::uuid,
                    $6,
                    $7,
                    $8::violation_severity,
                    $9,
                    $10::jsonb,
                    $11,
                    $12::jsonb,
                    $13,
                    $14
                )
            """

            # Generate violation code
            type_codes = {
                "domain_blocked": "DOM_001",
                "budget_exceeded": "BUD_001",
                "rate_limited": "RAT_001",
                "capability_denied": "CAP_001",
            }
            violation_code = type_codes.get(violation["violation_type"], "POL_001")

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    uuid.UUID(violation["violation_id"]),
                    uuid.UUID(session_id) if session_id else None,
                    uuid.UUID(step_id) if step_id else None,
                    violation["agent_id"],
                    uuid.UUID(tool_invocation_id) if tool_invocation_id else None,
                    violation["violation_type"],
                    violation_code,
                    violation["severity"],
                    violation["details"].get("message", "Policy violation"),
                    json.dumps(violation["details"]),
                    violation["details"].get("tool_name"),
                    json.dumps({"args": violation["details"].get("url")}),
                    True,  # blocked_execution
                    datetime.fromisoformat(violation["timestamp"].replace("Z", "+00:00")),
                )

            logger.debug(f"Violation {violation['violation_id']} logged to database")

        except Exception as e:
            logger.error(f"Failed to write violation to database: {e}")

    async def _emit_violation_event(
        self,
        violation: PolicyViolation,
        session_id: str | None = None,
    ) -> None:
        """
        Emit violation event to Redis stream.

        Args:
            violation: The violation to emit
            session_id: Optional session context
        """
        if not self.active_state:
            return

        try:
            event_data = {
                "event_type": "policy_violated",
                "violation_id": violation["violation_id"],
                "agent_id": violation["agent_id"],
                "violation_type": violation["violation_type"],
                "severity": violation["severity"],
                "message": violation["details"].get("message", ""),
                "session_id": session_id or "",
                "timestamp": violation["timestamp"],
            }

            # Add to Redis stream
            await self.active_state.redis.xadd(
                VIOLATION_STREAM_KEY,
                event_data,
                maxlen=10000,  # Keep last 10k events
            )

            logger.debug(f"Violation event emitted: {violation['violation_id']}")

        except Exception as e:
            logger.warning(f"Failed to emit violation event: {e}")

    # =========================================================================
    # Tool Execution Wrapper
    # =========================================================================

    def wrap_tool_execution(
        self,
        tool: "BaseTool",
        agent_id: str,
        session_id: str | None = None,
    ) -> Callable[..., Awaitable[ToolResult]]:
        """
        Create a wrapped tool execution function with policy enforcement.

        Returns a function that:
        1. Checks policy before execution
        2. Records spend and rate limit usage after execution
        3. Logs any violations

        Args:
            tool: The tool to wrap
            agent_id: The agent using the tool
            session_id: Optional session context

        Returns:
            Wrapped execution function

        Example:
            >>> wrapped_search = firewall.wrap_tool_execution(
            ...     search_tool,
            ...     agent_id="searcher_v1",
            ...     session_id="session-123",
            ... )
            >>> result = await wrapped_search("query", max_results=10)
        """
        original_invoke = tool.invoke

        @wraps(original_invoke)
        async def wrapped_invoke(input_str: str, **kwargs: Any) -> "ToolResult":
            from src.tools.base import ToolResult

            # Build tool args from input and kwargs
            tool_args = {"input": input_str, **kwargs}

            # Estimate cost (simple heuristic - override in production)
            estimated_cost = 0.001  # $0.001 per invocation default

            # Check policy
            try:
                result = await self.check_policy(
                    agent_id=agent_id,
                    tool_name=tool.name,
                    tool_args=tool_args,
                    session_id=session_id,
                    estimated_cost=estimated_cost,
                )
            except Exception as e:
                logger.error(f"Policy check failed: {e}")
                return ToolResult.fail(f"Policy check failed: {e}")

            if not result["allowed"]:
                # Log violations
                for violation in result["violations"]:
                    await self.log_violation(
                        agent_id=agent_id,
                        violation=violation,
                        session_id=session_id,
                    )

                return ToolResult.fail(
                    f"Policy violation: {result['reason']}",
                    violations=[v["violation_type"] for v in result["violations"]],
                    recommendations=result["recommendations"],
                )

            # Execute the tool
            try:
                tool_result = await original_invoke(input_str, **kwargs)

                # Record spend after successful execution
                if tool_result.success:
                    actual_cost = tool_result.metadata.get("cost_usd", estimated_cost)
                    await self._record_spend(agent_id, actual_cost, session_id)

                    # Record tokens used
                    tokens_used = tool_result.metadata.get("tokens_used", 0)
                    if tokens_used > 0:
                        await self._increment_rate_limit(
                            agent_id,
                            requests=0,  # Already counted
                            tokens=tokens_used,
                        )

                return tool_result

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return ToolResult.fail(f"Tool execution failed: {e}")

        return wrapped_invoke

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def get_budget_state(
        self,
        agent_id: str,
        session_id: str | None = None,
    ) -> BudgetState:
        """
        Get current budget state for an agent/session.

        Args:
            agent_id: The agent identifier
            session_id: Optional session ID

        Returns:
            BudgetState with current spend and limits
        """
        manifest = await self._get_manifest(agent_id)
        max_budget = manifest.max_budget_usd if manifest else self._get_config_float("default_budget_usd", 1.0)
        current_spend = await self._get_current_spend(agent_id, session_id)

        return BudgetState(
            total_spent=current_spend,
            max_budget=max_budget,
            remaining=max(0.0, max_budget - current_spend),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    async def reset_rate_limit(self, agent_id: str) -> None:
        """
        Reset rate limit counters for an agent.

        Args:
            agent_id: The agent identifier
        """
        if self.active_state:
            key = f"{RATE_LIMIT_KEY_PREFIX}:{agent_id}"
            await self.active_state.redis.delete(key)
            logger.info(f"Rate limit reset for agent {agent_id}")

    async def reset_budget(
        self,
        agent_id: str,
        session_id: str | None = None,
    ) -> None:
        """
        Reset budget tracking for an agent/session.

        Args:
            agent_id: The agent identifier
            session_id: Optional session ID
        """
        if self.active_state:
            key = f"{BUDGET_KEY_PREFIX}:{agent_id}"
            if session_id:
                key = f"{key}:{session_id}"
            await self.active_state.redis.delete(key)
            logger.info(f"Budget reset for agent {agent_id}")


# =============================================================================
# Decorator for Policy Enforcement
# =============================================================================


def enforce_policy(
    firewall: PolicyFirewall,
    agent_id: str,
    session_id: str | None = None,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]],
    Callable[P, Coroutine[Any, Any, R]],
]:
    """
    Decorator to enforce policy on async tool methods.

    Usage:
        @enforce_policy(firewall, "searcher_v1")
        async def search(url: str) -> dict:
            # Tool implementation
            pass

    Args:
        firewall: PolicyFirewall instance
        agent_id: Agent identifier
        session_id: Optional session context

    Returns:
        Decorated async function with policy enforcement
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Extract tool name from function
            tool_name = func.__name__

            # Build args dict
            tool_args = dict(kwargs)

            # Check policy
            result = await firewall.check_policy(
                agent_id=agent_id,
                tool_name=tool_name,
                tool_args=tool_args,
                session_id=session_id,
            )

            if not result["allowed"]:
                for violation in result["violations"]:
                    await firewall.log_violation(
                        agent_id=agent_id,
                        violation=violation,
                        session_id=session_id,
                    )
                raise PolicyFirewallError(
                    f"Policy violation: {result['reason']}"
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    # TypedDicts
    "PolicyViolation",
    "PolicyCheckResult",
    "PolicyConfig",
    "BudgetState",
    "RateLimitState",
    # Main class
    "PolicyFirewall",
    # Exceptions
    "PolicyFirewallError",
    "BudgetExceededError",
    "RateLimitExceededError",
    "DomainBlockedError",
    "CapabilityDeniedError",
    # Constants
    "DEFAULT_POLICY_CONFIG",
    "VIOLATION_SEVERITY_MAP",
    "TOOL_CAPABILITY_MAP",
    # Decorator
    "enforce_policy",
]
