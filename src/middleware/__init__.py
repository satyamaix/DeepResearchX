"""
Middleware Module for DRX Deep Research Platform.

This module provides policy enforcement middleware components that intercept
and validate tool invocations before execution. It implements the Metadata
Firewall (WP-M6) for security and compliance enforcement.

Key Components:
- PolicyFirewall: Main middleware class for policy enforcement
- DomainValidator: URL/domain validation with wildcard support
- PolicyCheckResult: Result of policy validation
- PolicyViolation: Violation record for audit logging

The middleware integrates with:
- WP-M1: Agent Manifest (policy configuration)
- WP-M2: Active State Redis Service (rate limiting, budget tracking)
- WP-M3: Context Propagation (session context)

Usage:
    from src.middleware import PolicyFirewall, DomainValidator

    # Create firewall instance
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

    if not result["allowed"]:
        # Handle policy violation
        for violation in result["violations"]:
            await firewall.log_violation(agent_id, violation)
"""

from __future__ import annotations

from src.middleware.domain_validator import (
    DomainValidator,
    DomainValidationError,
)
from src.middleware.policy_firewall import (
    PolicyCheckResult,
    PolicyViolation,
    PolicyConfig,
    PolicyFirewall,
    PolicyFirewallError,
    BudgetExceededError,
    RateLimitExceededError,
    DomainBlockedError,
    CapabilityDeniedError,
)

__all__ = [
    # Domain validation
    "DomainValidator",
    "DomainValidationError",
    # TypedDicts
    "PolicyCheckResult",
    "PolicyViolation",
    "PolicyConfig",
    # Main firewall class
    "PolicyFirewall",
    # Exceptions
    "PolicyFirewallError",
    "BudgetExceededError",
    "RateLimitExceededError",
    "DomainBlockedError",
    "CapabilityDeniedError",
]
