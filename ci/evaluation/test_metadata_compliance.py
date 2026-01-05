"""
DRX Metadata Compliance Tests.

DeepEval-integrated tests for validating agent compliance with metadata policies.
Tests cover budget limits, domain restrictions, rate limits, and capability matching.

Part of WP-M8: Metadata-Aware Evaluation Implementation.

Test Categories:
- Budget compliance: Tests that agents respect spending limits
- Domain compliance: Tests that agents only access allowed domains
- Rate limit compliance: Tests that agents respect request/token limits
- Capability compliance: Tests that tasks are routed to capable agents

Usage:
    # Run all compliance tests
    pytest ci/evaluation/test_metadata_compliance.py -v

    # Run only budget tests
    pytest ci/evaluation/test_metadata_compliance.py -k budget -v

    # Run with markers
    pytest ci/evaluation/test_metadata_compliance.py -m ci_gate -v
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import assertion helpers and types
from ci.evaluation.metadata_assertions import (
    AgentInvocationRecord,
    BudgetComplianceError,
    CapabilityComplianceError,
    ComplianceResult,
    DomainComplianceError,
    PolicyViolationRecord,
    RateLimitComplianceError,
    SubTask,
    ToolInvocationRecord,
    Trace,
    assert_budget_compliance,
    assert_capability_match,
    assert_domain_compliance,
    assert_no_violations,
    assert_rate_limit_compliance,
    check_compliance,
    extract_domains_accessed,
    extract_total_cost,
    extract_total_tokens,
)


# =============================================================================
# Pytest Configuration
# =============================================================================


pytestmark = [
    pytest.mark.eval,
    pytest.mark.asyncio,
]


# =============================================================================
# Mock Manifest Fixture
# =============================================================================


class MockRateLimits:
    """Mock rate limits object for testing."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute


class MockCircuitBreaker:
    """Mock circuit breaker config for testing."""

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_seconds: int = 30,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds


class MockManifest:
    """Mock AgentManifest for testing compliance assertions."""

    def __init__(
        self,
        agent_id: str = "test_agent_v1",
        agent_type: str = "searcher",
        capabilities: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_budget_usd: float = 1.0,
        rate_limits: MockRateLimits | None = None,
        circuit_breaker: MockCircuitBreaker | None = None,
        allowed_tools: list[str] | None = None,
        is_active: bool = True,
    ) -> None:
        self.id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities or ["web_search", "source_discovery"]
        self.allowed_domains = allowed_domains or []
        self.blocked_domains = blocked_domains or []
        self.max_budget_usd = max_budget_usd
        self.rate_limits = rate_limits or MockRateLimits()
        self.circuit_breaker = circuit_breaker or MockCircuitBreaker()
        self.allowed_tools = allowed_tools or ["web_search", "url_fetch"]
        self.is_active = is_active


# =============================================================================
# Trace Generation Fixtures
# =============================================================================


@pytest.fixture
def base_timestamp() -> str:
    """Return a base timestamp for test traces."""
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def sample_manifest() -> MockManifest:
    """Return a sample manifest for testing."""
    return MockManifest(
        agent_id="searcher_v1",
        agent_type="searcher",
        capabilities=["web_search", "source_discovery", "query_expansion"],
        allowed_domains=["*.gov", "*.edu", "example.com"],
        blocked_domains=["malicious.com", "*.spam.net"],
        max_budget_usd=5.0,
        rate_limits=MockRateLimits(requests_per_minute=30, tokens_per_minute=50000),
    )


@pytest.fixture
def compliant_trace(base_timestamp: str) -> Trace:
    """Return a trace that complies with all policies."""
    return Trace(
        session_id=str(uuid.uuid4()),
        user_query="What are the latest research findings?",
        started_at=base_timestamp,
        completed_at=base_timestamp,
        agent_invocations=[
            AgentInvocationRecord(
                agent_id="searcher_v1",
                agent_type="searcher",
                timestamp=base_timestamp,
                duration_ms=500,
                success=True,
                error=None,
                tool_invocations=[
                    ToolInvocationRecord(
                        tool_name="web_search",
                        tool_args={"query": "research findings"},
                        timestamp=base_timestamp,
                        duration_ms=200,
                        success=True,
                        error=None,
                        cost_usd=0.01,
                        tokens_used=100,
                        url="https://research.gov/findings",
                        domain="research.gov",
                    ),
                ],
                cost_usd=0.05,
                tokens_used=500,
                input_tokens=300,
                output_tokens=200,
            ),
        ],
        policy_violations=[],
        total_cost_usd=0.05,
        total_tokens=500,
        total_requests=1,
        domains_accessed=["research.gov"],
        urls_accessed=["https://research.gov/findings"],
        final_status="completed",
        error=None,
    )


@pytest.fixture
def budget_exceeded_trace(base_timestamp: str) -> Trace:
    """Return a trace that exceeds budget limits."""
    return Trace(
        session_id=str(uuid.uuid4()),
        user_query="Comprehensive research on all topics",
        started_at=base_timestamp,
        completed_at=base_timestamp,
        agent_invocations=[
            AgentInvocationRecord(
                agent_id="searcher_v1",
                agent_type="searcher",
                timestamp=base_timestamp,
                duration_ms=5000,
                success=True,
                error=None,
                tool_invocations=[],
                cost_usd=10.0,  # Exceeds typical budget
                tokens_used=50000,
                input_tokens=25000,
                output_tokens=25000,
            ),
        ],
        policy_violations=[
            PolicyViolationRecord(
                violation_type="budget_exceeded",
                agent_id="searcher_v1",
                severity="critical",
                details={
                    "message": "Budget exceeded: $10.00 > $5.00",
                    "current_spend": 10.0,
                    "max_budget": 5.0,
                },
                timestamp=base_timestamp,
            ),
        ],
        total_cost_usd=10.0,
        total_tokens=50000,
        total_requests=10,
        domains_accessed=["example.com"],
        urls_accessed=[],
        final_status="blocked",
        error="Budget exceeded",
    )


@pytest.fixture
def domain_blocked_trace(base_timestamp: str) -> Trace:
    """Return a trace with blocked domain access."""
    return Trace(
        session_id=str(uuid.uuid4()),
        user_query="Find information on malicious.com",
        started_at=base_timestamp,
        completed_at=base_timestamp,
        agent_invocations=[
            AgentInvocationRecord(
                agent_id="searcher_v1",
                agent_type="searcher",
                timestamp=base_timestamp,
                duration_ms=100,
                success=False,
                error="Domain blocked",
                tool_invocations=[
                    ToolInvocationRecord(
                        tool_name="url_fetch",
                        tool_args={"url": "https://malicious.com/data"},
                        timestamp=base_timestamp,
                        duration_ms=50,
                        success=False,
                        error="Domain blocked",
                        cost_usd=0.0,
                        tokens_used=0,
                        url="https://malicious.com/data",
                        domain="malicious.com",
                    ),
                ],
                cost_usd=0.01,
                tokens_used=100,
                input_tokens=100,
                output_tokens=0,
            ),
        ],
        policy_violations=[
            PolicyViolationRecord(
                violation_type="domain_blocked",
                agent_id="searcher_v1",
                severity="error",
                details={
                    "message": "Domain 'malicious.com' is in blocked list",
                    "domain": "malicious.com",
                    "url": "https://malicious.com/data",
                },
                timestamp=base_timestamp,
            ),
        ],
        total_cost_usd=0.01,
        total_tokens=100,
        total_requests=1,
        domains_accessed=["malicious.com"],
        urls_accessed=["https://malicious.com/data"],
        final_status="blocked",
        error="Domain blocked",
    )


@pytest.fixture
def rate_limited_trace(base_timestamp: str) -> Trace:
    """Return a trace that exceeds rate limits."""
    # Generate many tool invocations to simulate rate limit breach
    tool_invocations: list[ToolInvocationRecord] = []
    for i in range(100):
        tool_invocations.append(
            ToolInvocationRecord(
                tool_name="web_search",
                tool_args={"query": f"search {i}"},
                timestamp=base_timestamp,  # All in same minute
                duration_ms=50,
                success=True if i < 30 else False,
                error=None if i < 30 else "Rate limited",
                cost_usd=0.001,
                tokens_used=100,
                url=f"https://example.com/page{i}",
                domain="example.com",
            )
        )

    return Trace(
        session_id=str(uuid.uuid4()),
        user_query="Run many searches quickly",
        started_at=base_timestamp,
        completed_at=base_timestamp,
        agent_invocations=[
            AgentInvocationRecord(
                agent_id="searcher_v1",
                agent_type="searcher",
                timestamp=base_timestamp,
                duration_ms=5000,
                success=False,
                error="Rate limited",
                tool_invocations=tool_invocations,
                cost_usd=0.1,
                tokens_used=10000,
                input_tokens=5000,
                output_tokens=5000,
            ),
        ],
        policy_violations=[
            PolicyViolationRecord(
                violation_type="rate_limited",
                agent_id="searcher_v1",
                severity="warning",
                details={
                    "message": "Rate limit exceeded: 100/30 requests/min",
                    "limit_type": "requests",
                    "current": 100,
                    "limit": 30,
                    "retry_after": 60,
                },
                timestamp=base_timestamp,
            ),
        ],
        total_cost_usd=0.1,
        total_tokens=10000,
        total_requests=100,
        domains_accessed=["example.com"],
        urls_accessed=[],
        final_status="failed",
        error="Rate limited",
    )


@pytest.fixture
def sample_subtask() -> SubTask:
    """Return a sample subtask for capability testing."""
    return SubTask(
        task_id="task_001",
        description="Search for competitor information",
        required_capabilities=["web_search", "source_discovery"],
        preferred_agent_type="searcher",
        priority=1,
        dependencies=[],
        constraints={},
    )


# =============================================================================
# Budget Compliance Tests
# =============================================================================


class TestBudgetCompliance:
    """Tests for budget compliance assertions."""

    def test_budget_compliance_passes_when_under_limit(
        self,
        compliant_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that compliant traces pass budget check."""
        # Should not raise
        assert_budget_compliance(compliant_trace, sample_manifest)

    def test_budget_compliance_fails_when_exceeded(
        self,
        budget_exceeded_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that budget exceeded traces fail check."""
        with pytest.raises(BudgetComplianceError) as exc_info:
            assert_budget_compliance(budget_exceeded_trace, sample_manifest)

        assert exc_info.value.current_spend == 10.0
        assert exc_info.value.max_budget == 5.0
        assert "budget_exceeded" in str(exc_info.value)

    def test_budget_compliance_respects_session_override(
        self,
        budget_exceeded_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that session budget override is respected."""
        # With higher session budget, should pass
        assert_budget_compliance(
            budget_exceeded_trace,
            sample_manifest,
            session_budget=20.0,
        )

    def test_budget_compliance_with_zero_cost(
        self,
        base_timestamp: str,
        sample_manifest: MockManifest,
    ) -> None:
        """Test compliance check with zero cost trace."""
        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Simple query",
            started_at=base_timestamp,
            agent_invocations=[],
            policy_violations=[],
            total_cost_usd=0.0,
            total_tokens=0,
            total_requests=0,
            domains_accessed=[],
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        # Should not raise
        assert_budget_compliance(trace, sample_manifest)

    def test_budget_compliance_fails_on_violation_record(
        self,
        base_timestamp: str,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that budget violation record fails check even if total cost is low."""
        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Query",
            started_at=base_timestamp,
            agent_invocations=[],
            policy_violations=[
                PolicyViolationRecord(
                    violation_type="budget_exceeded",
                    agent_id="searcher_v1",
                    severity="critical",
                    details={"message": "Budget exceeded"},
                    timestamp=base_timestamp,
                ),
            ],
            total_cost_usd=0.01,  # Low cost but violation recorded
            total_tokens=100,
            total_requests=1,
            domains_accessed=[],
            urls_accessed=[],
            final_status="blocked",
            error="Budget exceeded",
        )

        with pytest.raises(BudgetComplianceError):
            assert_budget_compliance(trace, sample_manifest)

    @pytest.mark.parametrize(
        "cost,budget,should_pass",
        [
            (0.99, 1.0, True),
            (1.0, 1.0, True),
            (1.01, 1.0, False),
            (5.0, 5.0, True),
            (5.01, 5.0, False),
            (0.0, 0.01, True),
        ],
    )
    def test_budget_compliance_boundary_conditions(
        self,
        base_timestamp: str,
        cost: float,
        budget: float,
        should_pass: bool,
    ) -> None:
        """Test budget compliance at boundary conditions."""
        manifest = MockManifest(max_budget_usd=budget)
        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test query",
            started_at=base_timestamp,
            agent_invocations=[
                AgentInvocationRecord(
                    agent_id="test_agent_v1",
                    agent_type="searcher",
                    timestamp=base_timestamp,
                    duration_ms=100,
                    success=True,
                    error=None,
                    tool_invocations=[],
                    cost_usd=cost,
                    tokens_used=100,
                    input_tokens=50,
                    output_tokens=50,
                ),
            ],
            policy_violations=[],
            total_cost_usd=cost,
            total_tokens=100,
            total_requests=1,
            domains_accessed=[],
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        if should_pass:
            assert_budget_compliance(trace, manifest)
        else:
            with pytest.raises(BudgetComplianceError):
                assert_budget_compliance(trace, manifest)


# =============================================================================
# Domain Compliance Tests
# =============================================================================


class TestDomainCompliance:
    """Tests for domain compliance assertions."""

    def test_domain_compliance_passes_allowed_domain(
        self,
        compliant_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that allowed domains pass compliance check."""
        # research.gov matches *.gov pattern
        assert_domain_compliance(compliant_trace, sample_manifest)

    def test_domain_compliance_fails_blocked_domain(
        self,
        domain_blocked_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that blocked domains fail compliance check."""
        with pytest.raises(DomainComplianceError) as exc_info:
            assert_domain_compliance(domain_blocked_trace, sample_manifest)

        assert exc_info.value.domain == "malicious.com"
        assert "domain_blocked" in str(exc_info.value)

    def test_domain_compliance_wildcard_matching(
        self,
        base_timestamp: str,
        sample_manifest: MockManifest,
    ) -> None:
        """Test wildcard pattern matching for domains."""
        # Test *.gov pattern allows subdomains
        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Search government sites",
            started_at=base_timestamp,
            agent_invocations=[
                AgentInvocationRecord(
                    agent_id="searcher_v1",
                    agent_type="searcher",
                    timestamp=base_timestamp,
                    duration_ms=100,
                    success=True,
                    tool_invocations=[
                        ToolInvocationRecord(
                            tool_name="url_fetch",
                            tool_args={},
                            timestamp=base_timestamp,
                            duration_ms=50,
                            success=True,
                            cost_usd=0.01,
                            tokens_used=100,
                            url="https://data.census.gov/api",
                            domain="data.census.gov",
                        ),
                    ],
                    cost_usd=0.01,
                    tokens_used=100,
                ),
            ],
            policy_violations=[],
            total_cost_usd=0.01,
            total_tokens=100,
            total_requests=1,
            domains_accessed=["data.census.gov"],
            urls_accessed=["https://data.census.gov/api"],
            final_status="completed",
            error=None,
        )

        # Should pass because data.census.gov matches *.gov
        assert_domain_compliance(trace, sample_manifest)

    def test_domain_compliance_blocked_subdomain(
        self,
        base_timestamp: str,
    ) -> None:
        """Test that subdomains of blocked domains are also blocked."""
        manifest = MockManifest(
            blocked_domains=["spam.net"],
        )

        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test",
            started_at=base_timestamp,
            agent_invocations=[],
            policy_violations=[],
            total_cost_usd=0.0,
            total_tokens=0,
            total_requests=0,
            domains_accessed=["api.spam.net"],  # Subdomain of blocked
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        with pytest.raises(DomainComplianceError):
            assert_domain_compliance(trace, manifest)

    def test_domain_compliance_not_in_allowed_list(
        self,
        base_timestamp: str,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that domains not in allowed list fail when list is specified."""
        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test",
            started_at=base_timestamp,
            agent_invocations=[],
            policy_violations=[],
            total_cost_usd=0.0,
            total_tokens=0,
            total_requests=0,
            domains_accessed=["random-domain.com"],  # Not in allowed list
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        with pytest.raises(DomainComplianceError) as exc_info:
            assert_domain_compliance(trace, sample_manifest)

        assert "not in allowed" in str(exc_info.value).lower()

    def test_domain_compliance_empty_allowed_list_allows_all(
        self,
        base_timestamp: str,
    ) -> None:
        """Test that empty allowed list allows all non-blocked domains."""
        manifest = MockManifest(
            allowed_domains=[],  # Empty allows all
            blocked_domains=["blocked.com"],
        )

        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test",
            started_at=base_timestamp,
            agent_invocations=[],
            policy_violations=[],
            total_cost_usd=0.0,
            total_tokens=0,
            total_requests=0,
            domains_accessed=["any-domain.com"],
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        # Should pass because allowed_domains is empty
        assert_domain_compliance(trace, manifest)

    @pytest.mark.parametrize(
        "domain,allowed,blocked,should_pass",
        [
            ("example.com", ["example.com"], [], True),
            ("example.com", [], ["example.com"], False),
            ("sub.example.com", ["*.example.com"], [], True),
            ("example.com", ["*.gov"], [], False),
            ("data.gov", ["*.gov"], [], True),
            ("evil.com", ["good.com"], ["evil.com"], False),
            ("neutral.org", [], [], True),  # Empty lists = all allowed
        ],
    )
    def test_domain_compliance_various_patterns(
        self,
        base_timestamp: str,
        domain: str,
        allowed: list[str],
        blocked: list[str],
        should_pass: bool,
    ) -> None:
        """Test domain compliance with various patterns."""
        manifest = MockManifest(
            allowed_domains=allowed,
            blocked_domains=blocked,
        )

        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test",
            started_at=base_timestamp,
            agent_invocations=[],
            policy_violations=[],
            total_cost_usd=0.0,
            total_tokens=0,
            total_requests=0,
            domains_accessed=[domain],
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        if should_pass:
            assert_domain_compliance(trace, manifest)
        else:
            with pytest.raises(DomainComplianceError):
                assert_domain_compliance(trace, manifest)


# =============================================================================
# Rate Limit Compliance Tests
# =============================================================================


class TestRateLimitCompliance:
    """Tests for rate limit compliance assertions."""

    def test_rate_limit_compliance_passes_under_limit(
        self,
        compliant_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that traces under rate limits pass check."""
        assert_rate_limit_compliance(compliant_trace, sample_manifest)

    def test_rate_limit_compliance_fails_when_exceeded(
        self,
        rate_limited_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that rate limited traces fail check."""
        with pytest.raises(RateLimitComplianceError) as exc_info:
            assert_rate_limit_compliance(rate_limited_trace, sample_manifest)

        assert exc_info.value.limit_type == "requests"
        assert "rate_limited" in str(exc_info.value)

    def test_rate_limit_compliance_requests_per_minute(
        self,
        base_timestamp: str,
    ) -> None:
        """Test request rate limit enforcement."""
        manifest = MockManifest(
            rate_limits=MockRateLimits(requests_per_minute=10),
        )

        # Create trace with 15 requests in same minute
        tool_invocations = [
            ToolInvocationRecord(
                tool_name="search",
                tool_args={},
                timestamp=base_timestamp,
                duration_ms=50,
                success=True,
                cost_usd=0.001,
                tokens_used=10,
            )
            for _ in range(15)
        ]

        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test",
            started_at=base_timestamp,
            agent_invocations=[
                AgentInvocationRecord(
                    agent_id="test_agent_v1",
                    agent_type="searcher",
                    timestamp=base_timestamp,
                    duration_ms=1000,
                    success=True,
                    tool_invocations=tool_invocations,
                    cost_usd=0.015,
                    tokens_used=150,
                ),
            ],
            policy_violations=[],
            total_cost_usd=0.015,
            total_tokens=150,
            total_requests=15,
            domains_accessed=[],
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        with pytest.raises(RateLimitComplianceError) as exc_info:
            assert_rate_limit_compliance(trace, manifest)

        assert exc_info.value.limit_type == "requests"
        assert exc_info.value.current_value == 15
        assert exc_info.value.limit_value == 10

    def test_rate_limit_compliance_tokens_per_minute(
        self,
        base_timestamp: str,
    ) -> None:
        """Test token rate limit enforcement."""
        manifest = MockManifest(
            rate_limits=MockRateLimits(tokens_per_minute=1000),
        )

        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test",
            started_at=base_timestamp,
            agent_invocations=[
                AgentInvocationRecord(
                    agent_id="test_agent_v1",
                    agent_type="searcher",
                    timestamp=base_timestamp,
                    duration_ms=1000,
                    success=True,
                    tool_invocations=[],
                    cost_usd=0.1,
                    tokens_used=5000,  # Exceeds 1000 limit
                ),
            ],
            policy_violations=[],
            total_cost_usd=0.1,
            total_tokens=5000,
            total_requests=1,
            domains_accessed=[],
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        with pytest.raises(RateLimitComplianceError) as exc_info:
            assert_rate_limit_compliance(trace, manifest)

        assert exc_info.value.limit_type == "tokens"
        assert exc_info.value.current_value == 5000
        assert exc_info.value.limit_value == 1000

    def test_rate_limit_compliance_violation_record(
        self,
        base_timestamp: str,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that rate limit violation record fails check."""
        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test",
            started_at=base_timestamp,
            agent_invocations=[],
            policy_violations=[
                PolicyViolationRecord(
                    violation_type="rate_limited",
                    agent_id="searcher_v1",
                    severity="warning",
                    details={
                        "limit_type": "tokens",
                        "current": 100000,
                        "limit": 50000,
                    },
                    timestamp=base_timestamp,
                ),
            ],
            total_cost_usd=0.0,
            total_tokens=0,
            total_requests=0,
            domains_accessed=[],
            urls_accessed=[],
            final_status="failed",
            error="Rate limited",
        )

        with pytest.raises(RateLimitComplianceError):
            assert_rate_limit_compliance(trace, sample_manifest)


# =============================================================================
# Capability Compliance Tests
# =============================================================================


class TestCapabilityCompliance:
    """Tests for capability matching assertions."""

    def test_capability_match_passes_with_matching_capabilities(
        self,
        sample_subtask: SubTask,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that agents with matching capabilities pass."""
        assert_capability_match(
            agent_id="searcher_v1",
            task=sample_subtask,
            manifest=sample_manifest,
        )

    def test_capability_match_fails_with_missing_capabilities(
        self,
        sample_subtask: SubTask,
    ) -> None:
        """Test that agents without required capabilities fail."""
        manifest = MockManifest(
            agent_id="reader_v1",
            capabilities=["content_extraction", "text_processing"],
        )

        with pytest.raises(CapabilityComplianceError) as exc_info:
            assert_capability_match(
                agent_id="reader_v1",
                task=sample_subtask,
                manifest=manifest,
            )

        assert "web_search" in exc_info.value.missing_capabilities
        assert "capability_denied" in str(exc_info.value)

    def test_capability_match_passes_with_partial_match(
        self,
    ) -> None:
        """Test that having at least one required capability passes."""
        task = SubTask(
            task_id="task_002",
            description="Process content",
            required_capabilities=["content_extraction", "web_search"],
            priority=1,
        )

        manifest = MockManifest(
            agent_id="reader_v1",
            capabilities=["content_extraction"],  # Has one of two
        )

        # Should pass because it has at least one required capability
        assert_capability_match(
            agent_id="reader_v1",
            task=task,
            manifest=manifest,
        )

    def test_capability_match_passes_with_empty_requirements(
        self,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that tasks with no requirements always pass."""
        task = SubTask(
            task_id="task_003",
            description="Simple task",
            required_capabilities=[],  # No requirements
            priority=1,
        )

        assert_capability_match(
            agent_id="searcher_v1",
            task=task,
            manifest=sample_manifest,
        )

    @pytest.mark.parametrize(
        "agent_caps,required_caps,should_pass",
        [
            (["web_search"], ["web_search"], True),
            (["web_search", "content_extraction"], ["web_search"], True),
            (["content_extraction"], ["web_search"], False),
            ([], ["web_search"], False),
            (["web_search"], [], True),
            (["a", "b", "c"], ["b", "d"], True),  # Has "b"
            (["a", "b", "c"], ["x", "y", "z"], False),
        ],
    )
    def test_capability_match_various_scenarios(
        self,
        agent_caps: list[str],
        required_caps: list[str],
        should_pass: bool,
    ) -> None:
        """Test capability matching with various scenarios."""
        manifest = MockManifest(capabilities=agent_caps)
        task = SubTask(
            task_id="task_test",
            description="Test task",
            required_capabilities=required_caps,
            priority=1,
        )

        if should_pass:
            assert_capability_match("test_agent_v1", task, manifest)
        else:
            with pytest.raises(CapabilityComplianceError):
                assert_capability_match("test_agent_v1", task, manifest)


# =============================================================================
# Combined Compliance Tests
# =============================================================================


class TestCombinedCompliance:
    """Tests for combined compliance checking."""

    def test_assert_no_violations_passes_clean_trace(
        self,
        compliant_trace: Trace,
    ) -> None:
        """Test that clean traces pass no violations check."""
        assert_no_violations(compliant_trace)

    def test_assert_no_violations_fails_with_violations(
        self,
        budget_exceeded_trace: Trace,
    ) -> None:
        """Test that traces with violations fail check."""
        from ci.evaluation.metadata_assertions import MetadataAssertionError

        with pytest.raises(MetadataAssertionError):
            assert_no_violations(budget_exceeded_trace)

    def test_check_compliance_returns_result_object(
        self,
        compliant_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test that check_compliance returns proper result object."""
        result = check_compliance(compliant_trace, sample_manifest)

        assert isinstance(result, dict)
        assert "compliant" in result
        assert "violations" in result
        assert "warnings" in result
        assert "metrics" in result

    def test_check_compliance_compliant_trace(
        self,
        compliant_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test check_compliance with compliant trace."""
        result = check_compliance(compliant_trace, sample_manifest)

        assert result["compliant"] is True
        assert len(result["violations"]) == 0
        assert result["metrics"]["budget_compliant"] is True
        assert result["metrics"]["domain_compliant"] is True
        assert result["metrics"]["rate_limit_compliant"] is True

    def test_check_compliance_budget_exceeded_trace(
        self,
        budget_exceeded_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """Test check_compliance with budget exceeded trace."""
        result = check_compliance(budget_exceeded_trace, sample_manifest)

        assert result["compliant"] is False
        assert len(result["violations"]) > 0
        assert result["metrics"]["budget_compliant"] is False

    def test_check_compliance_multiple_violations(
        self,
        base_timestamp: str,
    ) -> None:
        """Test check_compliance with multiple violations."""
        manifest = MockManifest(
            max_budget_usd=0.01,
            allowed_domains=["allowed.com"],
            blocked_domains=[],
        )

        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test",
            started_at=base_timestamp,
            agent_invocations=[
                AgentInvocationRecord(
                    agent_id="test_agent_v1",
                    agent_type="searcher",
                    timestamp=base_timestamp,
                    duration_ms=100,
                    success=True,
                    tool_invocations=[],
                    cost_usd=1.0,  # Exceeds budget
                    tokens_used=100,
                ),
            ],
            policy_violations=[],
            total_cost_usd=1.0,
            total_tokens=100,
            total_requests=1,
            domains_accessed=["blocked.com"],  # Not in allowed list
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        result = check_compliance(trace, manifest)

        assert result["compliant"] is False
        assert len(result["violations"]) >= 2  # Budget + domain
        assert result["metrics"]["budget_compliant"] is False
        assert result["metrics"]["domain_compliant"] is False


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for trace extraction helper functions."""

    def test_extract_total_cost(
        self,
        compliant_trace: Trace,
    ) -> None:
        """Test total cost extraction from trace."""
        cost = extract_total_cost(compliant_trace)
        assert cost == 0.05

    def test_extract_total_tokens(
        self,
        compliant_trace: Trace,
    ) -> None:
        """Test total tokens extraction from trace."""
        tokens = extract_total_tokens(compliant_trace)
        assert tokens == 500

    def test_extract_domains_accessed(
        self,
        compliant_trace: Trace,
    ) -> None:
        """Test domains extraction from trace."""
        domains = extract_domains_accessed(compliant_trace)
        assert "research.gov" in domains

    def test_extract_domains_from_tool_invocations(
        self,
        base_timestamp: str,
    ) -> None:
        """Test domain extraction from tool invocations."""
        trace = Trace(
            session_id=str(uuid.uuid4()),
            user_query="Test",
            started_at=base_timestamp,
            agent_invocations=[
                AgentInvocationRecord(
                    agent_id="test_agent_v1",
                    agent_type="searcher",
                    timestamp=base_timestamp,
                    duration_ms=100,
                    success=True,
                    tool_invocations=[
                        ToolInvocationRecord(
                            tool_name="fetch",
                            tool_args={},
                            timestamp=base_timestamp,
                            duration_ms=50,
                            success=True,
                            cost_usd=0.01,
                            tokens_used=100,
                            domain="domain1.com",
                        ),
                        ToolInvocationRecord(
                            tool_name="fetch",
                            tool_args={},
                            timestamp=base_timestamp,
                            duration_ms=50,
                            success=True,
                            cost_usd=0.01,
                            tokens_used=100,
                            domain="domain2.com",
                        ),
                    ],
                    cost_usd=0.02,
                    tokens_used=200,
                ),
            ],
            policy_violations=[],
            total_cost_usd=0.02,
            total_tokens=200,
            total_requests=2,
            domains_accessed=[],  # Empty at trace level
            urls_accessed=[],
            final_status="completed",
            error=None,
        )

        domains = extract_domains_accessed(trace)
        assert "domain1.com" in domains
        assert "domain2.com" in domains


# =============================================================================
# Integration Tests with Mocked Services
# =============================================================================


@pytest.mark.integration
class TestPolicyFirewallIntegration:
    """Integration tests with PolicyFirewall."""

    @pytest.fixture
    def mock_active_state(self) -> MagicMock:
        """Create mock ActiveStateService."""
        mock = MagicMock()
        mock.redis = MagicMock()
        mock.redis.get = AsyncMock(return_value=None)
        mock.redis.set = AsyncMock(return_value=True)
        mock.redis.incrbyfloat = AsyncMock(return_value=0.0)
        mock.redis.hincrby = AsyncMock(return_value=0)
        mock.redis.hgetall = AsyncMock(return_value={})
        mock.redis.pipeline = MagicMock()
        return mock

    async def test_firewall_budget_check_integration(
        self,
        mock_active_state: MagicMock,
    ) -> None:
        """Test PolicyFirewall budget checking integration."""
        try:
            from src.middleware.policy_firewall import PolicyFirewall
        except ImportError:
            pytest.skip("PolicyFirewall not available")

        firewall = PolicyFirewall(
            active_state_service=mock_active_state,
            config={
                "enforce_budget": True,
                "enforce_domains": False,
                "enforce_rate_limits": False,
                "enforce_capabilities": False,
                "default_budget_usd": 1.0,
            },
        )

        result = await firewall.check_policy(
            agent_id="test_agent_v1",
            tool_name="web_search",
            tool_args={"query": "test"},
            estimated_cost=0.01,
        )

        assert result["allowed"] is True
        assert len(result["violations"]) == 0

    async def test_firewall_domain_check_integration(
        self,
        mock_active_state: MagicMock,
    ) -> None:
        """Test PolicyFirewall domain checking integration."""
        try:
            from src.middleware.policy_firewall import PolicyFirewall
        except ImportError:
            pytest.skip("PolicyFirewall not available")

        # Create manifest loader that returns manifest with blocked domains
        async def mock_loader(agent_id: str):
            return MockManifest(
                agent_id=agent_id,
                blocked_domains=["blocked.com"],
            )

        firewall = PolicyFirewall(
            active_state_service=mock_active_state,
            manifest_loader=mock_loader,
            config={
                "enforce_budget": False,
                "enforce_domains": True,
                "enforce_rate_limits": False,
                "enforce_capabilities": False,
            },
        )

        result = await firewall.check_policy(
            agent_id="test_agent_v1",
            tool_name="url_fetch",
            tool_args={"url": "https://blocked.com/page"},
        )

        assert result["allowed"] is False
        assert len(result["violations"]) > 0
        assert result["violations"][0]["violation_type"] == "domain_blocked"


# =============================================================================
# CI Gate Tests
# =============================================================================


@pytest.mark.ci_gate
class TestCIGateCompliance:
    """CI gate tests for metadata compliance."""

    def test_ci_gate_budget_compliance(
        self,
        compliant_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """CI gate: Budget compliance must pass."""
        assert_budget_compliance(compliant_trace, sample_manifest)

    def test_ci_gate_domain_compliance(
        self,
        compliant_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """CI gate: Domain compliance must pass."""
        assert_domain_compliance(compliant_trace, sample_manifest)

    def test_ci_gate_rate_limit_compliance(
        self,
        compliant_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """CI gate: Rate limit compliance must pass."""
        assert_rate_limit_compliance(compliant_trace, sample_manifest)

    def test_ci_gate_no_violations(
        self,
        compliant_trace: Trace,
    ) -> None:
        """CI gate: No policy violations in trace."""
        assert_no_violations(compliant_trace)

    def test_ci_gate_full_compliance_check(
        self,
        compliant_trace: Trace,
        sample_manifest: MockManifest,
    ) -> None:
        """CI gate: Full compliance check must pass."""
        result = check_compliance(compliant_trace, sample_manifest)
        assert result["compliant"] is True, f"Compliance failed: {result['violations']}"
