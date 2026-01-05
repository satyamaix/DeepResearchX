"""
DRX Metadata-Aware Evaluation Assertions.

This module provides assertion helpers for validating agent compliance with
metadata policies during evaluation. Used by test_metadata_compliance.py
and test_circuit_breaker.py for systematic testing of the agentic infrastructure.

Key Features:
- Trace-based assertion helpers for budget, domain, rate limit compliance
- Capability matching assertions for task routing
- Detailed error messages for debugging policy violations
- Integration with AgentManifest and PolicyFirewall

Part of WP-M8: Metadata-Aware Evaluation Implementation.

Usage:
    from ci.evaluation.metadata_assertions import (
        assert_budget_compliance,
        assert_domain_compliance,
        assert_rate_limit_compliance,
        assert_capability_match,
    )

    # Check budget compliance in a trace
    assert_budget_compliance(trace, manifest)

    # Check domain access compliance
    assert_domain_compliance(trace, manifest)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict


# =============================================================================
# TypedDict Definitions
# =============================================================================


class ToolInvocationRecord(TypedDict, total=False):
    """Record of a single tool invocation within a trace.

    Attributes:
        tool_name: Name of the tool invoked
        tool_args: Arguments passed to the tool
        timestamp: ISO 8601 timestamp of invocation
        duration_ms: Duration of the invocation in milliseconds
        success: Whether the invocation succeeded
        error: Error message if failed
        cost_usd: Cost of the invocation in USD
        tokens_used: Number of tokens consumed
        url: URL accessed (if applicable)
        domain: Domain accessed (if applicable)
    """

    tool_name: str
    tool_args: dict[str, Any]
    timestamp: str
    duration_ms: int
    success: bool
    error: str | None
    cost_usd: float
    tokens_used: int
    url: str | None
    domain: str | None


class AgentInvocationRecord(TypedDict, total=False):
    """Record of an agent invocation within a trace.

    Attributes:
        agent_id: Identifier of the agent invoked
        agent_type: Type of the agent (searcher, reader, etc.)
        timestamp: ISO 8601 timestamp of invocation
        duration_ms: Duration of the agent execution in milliseconds
        success: Whether the agent succeeded
        error: Error message if failed
        tool_invocations: List of tool invocations made by the agent
        cost_usd: Total cost of the agent invocation
        tokens_used: Total tokens consumed
        input_tokens: Input tokens used
        output_tokens: Output tokens generated
    """

    agent_id: str
    agent_type: str
    timestamp: str
    duration_ms: int
    success: bool
    error: str | None
    tool_invocations: list[ToolInvocationRecord]
    cost_usd: float
    tokens_used: int
    input_tokens: int
    output_tokens: int


class PolicyViolationRecord(TypedDict):
    """Record of a policy violation detected in a trace.

    Attributes:
        violation_type: Type of violation
        agent_id: Agent that caused the violation
        severity: Severity level
        details: Additional violation details
        timestamp: When the violation occurred
    """

    violation_type: Literal[
        "domain_blocked",
        "budget_exceeded",
        "rate_limited",
        "capability_denied",
    ]
    agent_id: str
    severity: Literal["warning", "error", "critical"]
    details: dict[str, Any]
    timestamp: str


class Trace(TypedDict, total=False):
    """Execution trace for a research session.

    Represents the full execution path including all agent invocations,
    tool calls, and policy violations. Used for evaluation and compliance
    testing.

    Attributes:
        session_id: Unique session identifier
        user_query: Original user query
        started_at: ISO timestamp when session started
        completed_at: ISO timestamp when session completed
        agent_invocations: List of agent invocations
        policy_violations: List of policy violations detected
        total_cost_usd: Total cost of the session
        total_tokens: Total tokens consumed
        total_requests: Total number of requests made
        domains_accessed: List of unique domains accessed
        urls_accessed: List of URLs accessed
        final_status: Final session status
        error: Error message if session failed
    """

    session_id: str
    user_query: str
    started_at: str
    completed_at: str | None
    agent_invocations: list[AgentInvocationRecord]
    policy_violations: list[PolicyViolationRecord]
    total_cost_usd: float
    total_tokens: int
    total_requests: int
    domains_accessed: list[str]
    urls_accessed: list[str]
    final_status: Literal["completed", "failed", "blocked", "timeout"]
    error: str | None


class SubTask(TypedDict, total=False):
    """A subtask from a research plan requiring agent assignment.

    Attributes:
        task_id: Unique task identifier
        description: Description of what needs to be done
        required_capabilities: Capabilities needed to execute this task
        preferred_agent_type: Preferred type of agent for this task
        priority: Task priority (higher = more important)
        dependencies: Task IDs this task depends on
        constraints: Additional constraints for execution
    """

    task_id: str
    description: str
    required_capabilities: list[str]
    preferred_agent_type: str | None
    priority: int
    dependencies: list[str]
    constraints: dict[str, Any]


class ComplianceResult(TypedDict):
    """Result of a compliance assertion check.

    Attributes:
        compliant: Whether the check passed
        violations: List of violations found
        warnings: List of warnings generated
        metrics: Computed metrics from the check
    """

    compliant: bool
    violations: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


# =============================================================================
# Exception Classes
# =============================================================================


class MetadataAssertionError(AssertionError):
    """Base exception for metadata assertion failures.

    Provides detailed context about what policy was violated and how.
    """

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        violation_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize MetadataAssertionError.

        Args:
            message: Human-readable error message
            agent_id: Agent that caused the violation (if applicable)
            violation_type: Type of violation
            details: Additional context about the violation
        """
        self.agent_id = agent_id
        self.violation_type = violation_type
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

        # Build detailed message
        full_message = self._build_detailed_message(message)
        super().__init__(full_message)

    def _build_detailed_message(self, base_message: str) -> str:
        """Build a detailed error message with context."""
        parts = [base_message]

        if self.agent_id:
            parts.append(f"Agent: {self.agent_id}")

        if self.violation_type:
            parts.append(f"Violation Type: {self.violation_type}")

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")

        return " | ".join(parts)


class BudgetComplianceError(MetadataAssertionError):
    """Raised when budget limits are violated.

    Contains specifics about the budget violation including current spend,
    limit, and amount exceeded.
    """

    def __init__(
        self,
        message: str,
        agent_id: str,
        current_spend: float,
        max_budget: float,
        exceeded_by: float,
    ) -> None:
        """Initialize BudgetComplianceError.

        Args:
            message: Error description
            agent_id: Agent that exceeded budget
            current_spend: Current total spend in USD
            max_budget: Maximum allowed budget in USD
            exceeded_by: Amount exceeded by in USD
        """
        self.current_spend = current_spend
        self.max_budget = max_budget
        self.exceeded_by = exceeded_by

        super().__init__(
            message=message,
            agent_id=agent_id,
            violation_type="budget_exceeded",
            details={
                "current_spend_usd": current_spend,
                "max_budget_usd": max_budget,
                "exceeded_by_usd": exceeded_by,
            },
        )


class DomainComplianceError(MetadataAssertionError):
    """Raised when domain access policies are violated.

    Contains specifics about which domain was accessed and why it was blocked.
    """

    def __init__(
        self,
        message: str,
        agent_id: str,
        domain: str,
        url: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Initialize DomainComplianceError.

        Args:
            message: Error description
            agent_id: Agent that violated domain policy
            domain: Domain that was accessed
            url: Full URL that was accessed (if available)
            reason: Reason the domain was blocked
        """
        self.domain = domain
        self.url = url
        self.reason = reason

        super().__init__(
            message=message,
            agent_id=agent_id,
            violation_type="domain_blocked",
            details={
                "domain": domain,
                "url": url,
                "reason": reason,
            },
        )


class RateLimitComplianceError(MetadataAssertionError):
    """Raised when rate limits are violated.

    Contains specifics about which rate limit was exceeded.
    """

    def __init__(
        self,
        message: str,
        agent_id: str,
        limit_type: Literal["requests", "tokens"],
        current_value: int,
        limit_value: int,
        window_seconds: int = 60,
    ) -> None:
        """Initialize RateLimitComplianceError.

        Args:
            message: Error description
            agent_id: Agent that exceeded rate limit
            limit_type: Type of limit exceeded (requests or tokens)
            current_value: Current count
            limit_value: Maximum allowed value
            window_seconds: Time window for the limit
        """
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.window_seconds = window_seconds

        super().__init__(
            message=message,
            agent_id=agent_id,
            violation_type="rate_limited",
            details={
                "limit_type": limit_type,
                "current": current_value,
                "limit": limit_value,
                "window_seconds": window_seconds,
                "exceeded_by": current_value - limit_value,
            },
        )


class CapabilityComplianceError(MetadataAssertionError):
    """Raised when capability requirements are not met.

    Contains specifics about which capabilities were required but missing.
    """

    def __init__(
        self,
        message: str,
        agent_id: str,
        required_capabilities: list[str],
        agent_capabilities: list[str],
        missing_capabilities: list[str],
    ) -> None:
        """Initialize CapabilityComplianceError.

        Args:
            message: Error description
            agent_id: Agent being checked
            required_capabilities: Capabilities required for the task
            agent_capabilities: Capabilities the agent has
            missing_capabilities: Capabilities that are missing
        """
        self.required_capabilities = required_capabilities
        self.agent_capabilities = agent_capabilities
        self.missing_capabilities = missing_capabilities

        super().__init__(
            message=message,
            agent_id=agent_id,
            violation_type="capability_denied",
            details={
                "required": required_capabilities,
                "available": agent_capabilities,
                "missing": missing_capabilities,
            },
        )


# =============================================================================
# Helper Functions
# =============================================================================


def extract_total_cost(trace: Trace) -> float:
    """Extract total cost from a trace.

    Sums up costs from all agent invocations, falling back to trace-level
    total if available.

    Args:
        trace: Execution trace to analyze

    Returns:
        Total cost in USD
    """
    if "total_cost_usd" in trace:
        return trace["total_cost_usd"]

    total = 0.0
    for agent_inv in trace.get("agent_invocations", []):
        total += agent_inv.get("cost_usd", 0.0)

        for tool_inv in agent_inv.get("tool_invocations", []):
            total += tool_inv.get("cost_usd", 0.0)

    return total


def extract_total_tokens(trace: Trace) -> int:
    """Extract total tokens used from a trace.

    Sums up tokens from all agent invocations.

    Args:
        trace: Execution trace to analyze

    Returns:
        Total tokens consumed
    """
    if "total_tokens" in trace:
        return trace["total_tokens"]

    total = 0
    for agent_inv in trace.get("agent_invocations", []):
        total += agent_inv.get("tokens_used", 0)

    return total


def extract_total_requests(trace: Trace) -> int:
    """Extract total request count from a trace.

    Counts all tool invocations across all agents.

    Args:
        trace: Execution trace to analyze

    Returns:
        Total number of requests/invocations
    """
    if "total_requests" in trace:
        return trace["total_requests"]

    total = 0
    for agent_inv in trace.get("agent_invocations", []):
        total += len(agent_inv.get("tool_invocations", []))

    return total


def extract_domains_accessed(trace: Trace) -> set[str]:
    """Extract all domains accessed in a trace.

    Args:
        trace: Execution trace to analyze

    Returns:
        Set of unique domains accessed
    """
    domains: set[str] = set()

    if "domains_accessed" in trace:
        domains.update(trace["domains_accessed"])

    for agent_inv in trace.get("agent_invocations", []):
        for tool_inv in agent_inv.get("tool_invocations", []):
            if tool_inv.get("domain"):
                domains.add(tool_inv["domain"])

    return domains


def extract_urls_accessed(trace: Trace) -> list[str]:
    """Extract all URLs accessed in a trace.

    Args:
        trace: Execution trace to analyze

    Returns:
        List of URLs accessed (may contain duplicates)
    """
    urls: list[str] = []

    if "urls_accessed" in trace:
        urls.extend(trace["urls_accessed"])

    for agent_inv in trace.get("agent_invocations", []):
        for tool_inv in agent_inv.get("tool_invocations", []):
            if tool_inv.get("url"):
                urls.append(tool_inv["url"])

    return urls


def extract_agent_requests_per_minute(
    trace: Trace,
    agent_id: str,
) -> dict[str, int]:
    """Extract requests per minute for a specific agent.

    Groups requests by minute and returns counts per minute window.

    Args:
        trace: Execution trace to analyze
        agent_id: Agent to filter for

    Returns:
        Dictionary mapping minute timestamp to request count
    """
    from datetime import datetime

    requests_per_minute: dict[str, int] = {}

    for agent_inv in trace.get("agent_invocations", []):
        if agent_inv.get("agent_id") != agent_id:
            continue

        for tool_inv in agent_inv.get("tool_invocations", []):
            timestamp = tool_inv.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    minute_key = dt.strftime("%Y-%m-%dT%H:%M")
                    requests_per_minute[minute_key] = (
                        requests_per_minute.get(minute_key, 0) + 1
                    )
                except ValueError:
                    continue

    return requests_per_minute


def extract_agent_tokens_per_minute(
    trace: Trace,
    agent_id: str,
) -> dict[str, int]:
    """Extract tokens per minute for a specific agent.

    Groups tokens by minute and returns totals per minute window.

    Args:
        trace: Execution trace to analyze
        agent_id: Agent to filter for

    Returns:
        Dictionary mapping minute timestamp to token count
    """
    from datetime import datetime

    tokens_per_minute: dict[str, int] = {}

    for agent_inv in trace.get("agent_invocations", []):
        if agent_inv.get("agent_id") != agent_id:
            continue

        timestamp = agent_inv.get("timestamp", "")
        tokens = agent_inv.get("tokens_used", 0)

        if timestamp and tokens:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                minute_key = dt.strftime("%Y-%m-%dT%H:%M")
                tokens_per_minute[minute_key] = (
                    tokens_per_minute.get(minute_key, 0) + tokens
                )
            except ValueError:
                continue

    return tokens_per_minute


def get_agent_cost(trace: Trace, agent_id: str) -> float:
    """Get total cost for a specific agent in the trace.

    Args:
        trace: Execution trace to analyze
        agent_id: Agent to filter for

    Returns:
        Total cost in USD for the agent
    """
    total = 0.0

    for agent_inv in trace.get("agent_invocations", []):
        if agent_inv.get("agent_id") == agent_id:
            total += agent_inv.get("cost_usd", 0.0)

            for tool_inv in agent_inv.get("tool_invocations", []):
                total += tool_inv.get("cost_usd", 0.0)

    return total


def count_violations_by_type(
    trace: Trace,
    violation_type: str,
) -> int:
    """Count violations of a specific type in a trace.

    Args:
        trace: Execution trace to analyze
        violation_type: Type of violation to count

    Returns:
        Number of violations of the specified type
    """
    count = 0

    for violation in trace.get("policy_violations", []):
        if violation.get("violation_type") == violation_type:
            count += 1

    return count


def get_violations_for_agent(
    trace: Trace,
    agent_id: str,
) -> list[PolicyViolationRecord]:
    """Get all violations for a specific agent.

    Args:
        trace: Execution trace to analyze
        agent_id: Agent to filter for

    Returns:
        List of violations for the specified agent
    """
    return [
        v for v in trace.get("policy_violations", [])
        if v.get("agent_id") == agent_id
    ]


# =============================================================================
# Assertion Functions
# =============================================================================


def assert_budget_compliance(
    trace: Trace,
    manifest: Any,
    session_budget: float | None = None,
) -> None:
    """Assert that the trace complies with budget limits.

    Validates that the total cost in the trace does not exceed the agent's
    budget as defined in the manifest.

    Args:
        trace: Execution trace to validate
        manifest: AgentManifest with budget configuration
        session_budget: Optional override for session-level budget

    Raises:
        BudgetComplianceError: If budget is exceeded

    Example:
        >>> assert_budget_compliance(trace, manifest)
        >>> # Passes silently if compliant, raises if not
    """
    # Get max budget from manifest
    max_budget = session_budget
    if max_budget is None:
        max_budget = getattr(manifest, "max_budget_usd", 1.0)

    # Extract total cost from trace
    total_cost = extract_total_cost(trace)

    # Check compliance
    if total_cost > max_budget:
        agent_id = getattr(manifest, "id", "unknown")
        exceeded_by = total_cost - max_budget

        raise BudgetComplianceError(
            message=f"Budget exceeded: ${total_cost:.4f} > ${max_budget:.4f}",
            agent_id=agent_id,
            current_spend=total_cost,
            max_budget=max_budget,
            exceeded_by=exceeded_by,
        )

    # Also check for budget_exceeded violations in trace
    budget_violations = count_violations_by_type(trace, "budget_exceeded")
    if budget_violations > 0:
        agent_id = getattr(manifest, "id", "unknown")
        raise BudgetComplianceError(
            message=f"Budget violation recorded in trace ({budget_violations} violations)",
            agent_id=agent_id,
            current_spend=total_cost,
            max_budget=max_budget,
            exceeded_by=0.0,
        )


def assert_domain_compliance(
    trace: Trace,
    manifest: Any,
) -> None:
    """Assert that the trace complies with domain access policies.

    Validates that all domains accessed in the trace are allowed by the
    agent's manifest configuration.

    Args:
        trace: Execution trace to validate
        manifest: AgentManifest with domain configuration

    Raises:
        DomainComplianceError: If a blocked domain was accessed

    Example:
        >>> assert_domain_compliance(trace, manifest)
        >>> # Passes silently if compliant, raises if not
    """
    agent_id = getattr(manifest, "id", "unknown")

    # Get domain lists from manifest
    allowed_domains: list[str] = list(getattr(manifest, "allowed_domains", []))
    blocked_domains: list[str] = list(getattr(manifest, "blocked_domains", []))

    # Extract domains accessed in trace
    domains_accessed = extract_domains_accessed(trace)

    # Check for domain_blocked violations in trace
    domain_violations = [
        v for v in trace.get("policy_violations", [])
        if v.get("violation_type") == "domain_blocked"
    ]

    if domain_violations:
        first_violation = domain_violations[0]
        domain = first_violation.get("details", {}).get("domain", "unknown")
        url = first_violation.get("details", {}).get("url")
        reason = first_violation.get("details", {}).get("message", "Domain blocked")

        raise DomainComplianceError(
            message=f"Blocked domain access recorded: {domain}",
            agent_id=agent_id,
            domain=domain,
            url=url,
            reason=reason,
        )

    # Validate each accessed domain
    for domain in domains_accessed:
        domain_lower = domain.lower()

        # Check blocked domains
        for blocked in blocked_domains:
            blocked_lower = blocked.lower()
            if _domain_matches(domain_lower, blocked_lower):
                raise DomainComplianceError(
                    message=f"Accessed blocked domain: {domain}",
                    agent_id=agent_id,
                    domain=domain,
                    reason=f"Domain matches blocked pattern: {blocked}",
                )

        # If allowed domains are specified, domain must be in list
        if allowed_domains:
            is_allowed = any(
                _domain_matches(domain_lower, allowed.lower())
                for allowed in allowed_domains
            )
            if not is_allowed:
                raise DomainComplianceError(
                    message=f"Accessed domain not in allowed list: {domain}",
                    agent_id=agent_id,
                    domain=domain,
                    reason="Domain not in allowed domains list",
                )


def _domain_matches(domain: str, pattern: str) -> bool:
    """Check if domain matches a pattern (supports wildcards).

    Args:
        domain: Domain to check
        pattern: Pattern to match against (may contain wildcards)

    Returns:
        True if domain matches the pattern
    """
    import fnmatch

    # Exact match
    if domain == pattern:
        return True

    # Wildcard match
    if "*" in pattern or "?" in pattern:
        if fnmatch.fnmatch(domain, pattern):
            return True

    # Subdomain match (pattern is parent domain)
    if domain.endswith(f".{pattern}"):
        return True

    return False


def assert_rate_limit_compliance(
    trace: Trace,
    manifest: Any,
) -> None:
    """Assert that the trace complies with rate limits.

    Validates that requests per minute and tokens per minute do not exceed
    the limits defined in the agent's manifest.

    Args:
        trace: Execution trace to validate
        manifest: AgentManifest with rate limit configuration

    Raises:
        RateLimitComplianceError: If rate limits are exceeded

    Example:
        >>> assert_rate_limit_compliance(trace, manifest)
        >>> # Passes silently if compliant, raises if not
    """
    agent_id = getattr(manifest, "id", "unknown")

    # Get rate limits from manifest
    rate_limits = getattr(manifest, "rate_limits", None)
    if rate_limits:
        rpm_limit = getattr(rate_limits, "requests_per_minute", 60)
        tpm_limit = getattr(rate_limits, "tokens_per_minute", 100000)
    else:
        rpm_limit = 60
        tpm_limit = 100000

    # Check for rate_limited violations in trace
    rate_violations = [
        v for v in trace.get("policy_violations", [])
        if v.get("violation_type") == "rate_limited"
    ]

    if rate_violations:
        first_violation = rate_violations[0]
        details = first_violation.get("details", {})
        limit_type = details.get("limit_type", "requests")
        current = details.get("current", 0)
        limit = details.get("limit", rpm_limit)

        raise RateLimitComplianceError(
            message=f"Rate limit violation recorded: {limit_type}",
            agent_id=agent_id,
            limit_type=limit_type,  # type: ignore
            current_value=current,
            limit_value=limit,
        )

    # Check requests per minute
    requests_per_minute = extract_agent_requests_per_minute(trace, agent_id)
    for minute_key, count in requests_per_minute.items():
        if count > rpm_limit:
            raise RateLimitComplianceError(
                message=f"Requests per minute exceeded at {minute_key}: {count} > {rpm_limit}",
                agent_id=agent_id,
                limit_type="requests",
                current_value=count,
                limit_value=rpm_limit,
            )

    # Check tokens per minute
    tokens_per_minute = extract_agent_tokens_per_minute(trace, agent_id)
    for minute_key, count in tokens_per_minute.items():
        if count > tpm_limit:
            raise RateLimitComplianceError(
                message=f"Tokens per minute exceeded at {minute_key}: {count} > {tpm_limit}",
                agent_id=agent_id,
                limit_type="tokens",
                current_value=count,
                limit_value=tpm_limit,
            )


def assert_capability_match(
    agent_id: str,
    task: SubTask,
    manifest: Any,
) -> None:
    """Assert that an agent has the capabilities required for a task.

    Validates that the agent's capabilities include at least one of the
    capabilities required by the task.

    Args:
        agent_id: Agent identifier to check
        task: SubTask with required capabilities
        manifest: AgentManifest with capability configuration

    Raises:
        CapabilityComplianceError: If agent lacks required capabilities

    Example:
        >>> assert_capability_match("searcher_v1", task, manifest)
        >>> # Passes silently if agent has required capabilities
    """
    # Get required capabilities from task
    required_capabilities = task.get("required_capabilities", [])

    if not required_capabilities:
        # No capabilities required, always passes
        return

    # Get agent capabilities from manifest
    agent_capabilities: list[str] = list(getattr(manifest, "capabilities", []))

    # Check if agent has at least one required capability
    required_set = set(required_capabilities)
    agent_set = set(agent_capabilities)

    # Find intersection (matching capabilities)
    matching = required_set.intersection(agent_set)

    if not matching:
        missing = list(required_set - agent_set)
        raise CapabilityComplianceError(
            message=f"Agent lacks required capabilities for task '{task.get('task_id', 'unknown')}'",
            agent_id=agent_id,
            required_capabilities=required_capabilities,
            agent_capabilities=agent_capabilities,
            missing_capabilities=missing,
        )


def assert_no_violations(trace: Trace) -> None:
    """Assert that a trace has no policy violations.

    Args:
        trace: Execution trace to validate

    Raises:
        MetadataAssertionError: If any violations are present
    """
    violations = trace.get("policy_violations", [])

    if violations:
        first = violations[0]
        raise MetadataAssertionError(
            message=f"Trace contains {len(violations)} policy violation(s)",
            agent_id=first.get("agent_id"),
            violation_type=first.get("violation_type"),
            details=first.get("details", {}),
        )


def check_compliance(
    trace: Trace,
    manifest: Any,
    session_budget: float | None = None,
) -> ComplianceResult:
    """Check all compliance rules and return a result object.

    Unlike the assert_* functions, this returns a result object instead
    of raising exceptions, making it suitable for aggregating results.

    Args:
        trace: Execution trace to validate
        manifest: AgentManifest with policy configuration
        session_budget: Optional session-level budget override

    Returns:
        ComplianceResult with compliance status and details

    Example:
        >>> result = check_compliance(trace, manifest)
        >>> if not result["compliant"]:
        ...     print(f"Violations: {result['violations']}")
    """
    violations: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    # Check budget compliance
    try:
        assert_budget_compliance(trace, manifest, session_budget)
        metrics["budget_compliant"] = True
    except BudgetComplianceError as e:
        violations.append(str(e))
        metrics["budget_compliant"] = False
        metrics["budget_exceeded_by"] = e.exceeded_by

    # Check domain compliance
    try:
        assert_domain_compliance(trace, manifest)
        metrics["domain_compliant"] = True
    except DomainComplianceError as e:
        violations.append(str(e))
        metrics["domain_compliant"] = False
        metrics["blocked_domain"] = e.domain

    # Check rate limit compliance
    try:
        assert_rate_limit_compliance(trace, manifest)
        metrics["rate_limit_compliant"] = True
    except RateLimitComplianceError as e:
        violations.append(str(e))
        metrics["rate_limit_compliant"] = False
        metrics["rate_limit_type"] = e.limit_type
        metrics["rate_limit_exceeded_by"] = e.current_value - e.limit_value

    # Add summary metrics
    metrics["total_cost_usd"] = extract_total_cost(trace)
    metrics["total_tokens"] = extract_total_tokens(trace)
    metrics["total_requests"] = extract_total_requests(trace)
    metrics["domains_accessed_count"] = len(extract_domains_accessed(trace))
    metrics["violation_count"] = len(trace.get("policy_violations", []))

    return ComplianceResult(
        compliant=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        metrics=metrics,
    )


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    # TypedDicts
    "ToolInvocationRecord",
    "AgentInvocationRecord",
    "PolicyViolationRecord",
    "Trace",
    "SubTask",
    "ComplianceResult",
    # Exceptions
    "MetadataAssertionError",
    "BudgetComplianceError",
    "DomainComplianceError",
    "RateLimitComplianceError",
    "CapabilityComplianceError",
    # Helper functions
    "extract_total_cost",
    "extract_total_tokens",
    "extract_total_requests",
    "extract_domains_accessed",
    "extract_urls_accessed",
    "extract_agent_requests_per_minute",
    "extract_agent_tokens_per_minute",
    "get_agent_cost",
    "count_violations_by_type",
    "get_violations_for_agent",
    # Assertion functions
    "assert_budget_compliance",
    "assert_domain_compliance",
    "assert_rate_limit_compliance",
    "assert_capability_match",
    "assert_no_violations",
    "check_compliance",
]
