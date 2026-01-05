"""
DRX Evaluation Pipeline - Pytest Fixtures and Configuration.

This module provides pytest fixtures for the evaluation pipeline, including:
- Mock LLM clients for deterministic testing
- Sample agent states for unit tests
- Evaluation datasets for DeepEval/Ragas integration
- Async test support configuration

WP11: Evaluation Pipeline Implementation
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# Ensure test environment variables are loaded first
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/drx_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("OPENROUTER_API_KEY", "test-api-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "DEBUG")


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers for evaluation tests."""
    config.addinivalue_line(
        "markers", "eval: mark test as evaluation test (requires eval dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires services)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "ci_gate: mark test as CI gate (blocking on failure)"
    )


# =============================================================================
# Event Loop Configuration (pytest-asyncio)
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create an event loop for the test session.

    Uses session scope to share the loop across all async tests,
    improving test performance and avoiding loop creation overhead.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def evaluation_dir() -> Path:
    """Return the evaluation directory path."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def scenarios_dir(evaluation_dir: Path) -> Path:
    """Return the scenarios directory path."""
    return evaluation_dir / "scenarios"


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def research_scenarios(scenarios_dir: Path) -> dict[str, Any]:
    """
    Load research task scenarios from YAML file.

    Returns:
        Dictionary containing TAU-bench style scenario definitions
    """
    scenario_file = scenarios_dir / "research_tasks.yaml"
    if scenario_file.exists():
        with open(scenario_file) as f:
            return yaml.safe_load(f)
    return {"scenarios": [], "evaluation_config": {}}


@pytest.fixture
def sample_query() -> str:
    """Sample research query for testing."""
    return "What are the top 3 payment processing competitors to Stripe?"


@pytest.fixture
def sample_context() -> list[str]:
    """Sample retrieval context for evaluation tests."""
    return [
        "PayPal is a leading digital payment platform that processes over $1 trillion "
        "in payment volume annually. Founded in 1998, it competes directly with Stripe "
        "in the online payment processing space.",

        "Square (now Block, Inc.) offers a suite of payment solutions including "
        "point-of-sale systems, online payments, and business banking. Square "
        "processed $186 billion in gross payment volume in 2023.",

        "Adyen is a Dutch fintech company providing payment processing services "
        "to major enterprises including Netflix, Uber, and Spotify. Adyen's "
        "single-platform approach competes with Stripe's API-first strategy.",

        "Stripe was founded in 2010 and has become one of the most valuable "
        "private fintech companies, offering payment processing APIs used by "
        "millions of businesses worldwide.",
    ]


@pytest.fixture
def sample_answer() -> str:
    """Sample generated answer for evaluation tests."""
    return """
Based on my research, the top 3 payment processing competitors to Stripe are:

1. **PayPal** - The largest digital payment platform by volume, processing over
   $1 trillion annually. PayPal offers a comprehensive suite of payment solutions
   including Braintree for developers.

2. **Square (Block, Inc.)** - A major competitor with $186 billion in gross payment
   volume in 2023. Square excels in omnichannel payments and small business solutions.

3. **Adyen** - A Dutch enterprise payment processor serving major clients like
   Netflix, Uber, and Spotify. Adyen's unified platform approach provides an
   alternative to Stripe's modular API design.

These three competitors together with Stripe dominate the modern payment
processing landscape, each with distinct strengths in different market segments.
"""


@pytest.fixture
def sample_citations() -> list[dict[str, Any]]:
    """Sample citations for evaluation tests."""
    return [
        {
            "id": "cit_001",
            "url": "https://investor.pypl.com/financials/annual-reports",
            "title": "PayPal Annual Report 2023",
            "snippet": "Total payment volume exceeded $1.36 trillion",
            "relevance_score": 0.92,
            "domain": "investor.pypl.com",
        },
        {
            "id": "cit_002",
            "url": "https://block.xyz/investor-relations",
            "title": "Block Q4 2023 Earnings",
            "snippet": "Gross payment volume reached $186 billion",
            "relevance_score": 0.88,
            "domain": "block.xyz",
        },
        {
            "id": "cit_003",
            "url": "https://www.adyen.com/investors",
            "title": "Adyen H2 2023 Results",
            "snippet": "Processed volume grew 22% year-over-year",
            "relevance_score": 0.85,
            "domain": "adyen.com",
        },
    ]


@pytest.fixture
def sample_findings() -> list[dict[str, Any]]:
    """Sample research findings for evaluation tests."""
    return [
        {
            "id": "find_001",
            "claim": "PayPal processes over $1 trillion in payment volume annually",
            "evidence": "According to PayPal's 2023 annual report, total payment volume exceeded $1.36 trillion",
            "source_urls": ["https://investor.pypl.com/financials/annual-reports"],
            "citation_ids": ["cit_001"],
            "confidence_score": 0.95,
            "agent_source": "reader",
            "tags": ["market_data", "competitor_analysis"],
            "verified": True,
        },
        {
            "id": "find_002",
            "claim": "Square processed $186 billion in gross payment volume in 2023",
            "evidence": "Block's Q4 2023 earnings report shows GPV of $186 billion",
            "source_urls": ["https://block.xyz/investor-relations"],
            "citation_ids": ["cit_002"],
            "confidence_score": 0.92,
            "agent_source": "reader",
            "tags": ["market_data", "competitor_analysis"],
            "verified": True,
        },
    ]


# =============================================================================
# AgentState Fixtures
# =============================================================================


@pytest.fixture
def sample_steerability_params() -> dict[str, Any]:
    """Sample steerability parameters for testing."""
    return {
        "tone": "technical",
        "format": "markdown",
        "max_sources": 10,
        "focus_areas": ["fintech", "payment processing"],
        "exclude_topics": [],
        "preferred_domains": ["techcrunch.com", "bloomberg.com"],
        "language": "en",
        "custom_instructions": None,
    }


@pytest.fixture
def sample_quality_metrics() -> dict[str, Any]:
    """Sample quality metrics for testing."""
    return {
        "coverage_score": 0.85,
        "avg_confidence": 0.88,
        "verified_findings": 5,
        "total_findings": 6,
        "unique_sources": 8,
        "citation_density": 2.5,
        "consistency_score": 0.92,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }


@pytest.fixture
def sample_research_plan() -> dict[str, Any]:
    """Sample research plan for testing."""
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "dag_nodes": [
            {
                "id": "task_001",
                "description": "Search for Stripe competitors",
                "agent_type": "searcher",
                "dependencies": [],
                "status": "completed",
                "inputs": {"query": "Stripe payment processing competitors"},
                "outputs": {"urls": ["https://example.com/1", "https://example.com/2"]},
                "quality_score": 0.9,
                "started_at": now,
                "completed_at": now,
                "error": None,
            },
            {
                "id": "task_002",
                "description": "Extract competitor information",
                "agent_type": "reader",
                "dependencies": ["task_001"],
                "status": "completed",
                "inputs": {"urls": ["https://example.com/1"]},
                "outputs": {"findings": []},
                "quality_score": 0.85,
                "started_at": now,
                "completed_at": now,
                "error": None,
            },
        ],
        "current_iteration": 1,
        "max_iterations": 5,
        "coverage_score": 0.85,
        "created_at": now,
        "updated_at": now,
        "sub_questions": [
            "Who are the main competitors to Stripe?",
            "What are their market shares?",
            "How do their features compare?",
        ],
    }


@pytest.fixture
def sample_state(
    sample_query: str,
    sample_steerability_params: dict[str, Any],
    sample_research_plan: dict[str, Any],
    sample_findings: list[dict[str, Any]],
    sample_citations: list[dict[str, Any]],
    sample_quality_metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a sample AgentState for testing.

    This fixture provides a fully populated state object that can be used
    for unit tests and evaluation scenarios.
    """
    now = datetime.utcnow().isoformat() + "Z"

    return {
        "messages": [],
        "session_id": "test-session-001",
        "user_query": sample_query,
        "started_at": now,
        "steerability": sample_steerability_params,
        "plan": sample_research_plan,
        "findings": sample_findings,
        "citations": sample_citations,
        "synthesis": "Preliminary synthesis of competitor analysis...",
        "final_report": None,
        "iteration_count": 1,
        "max_iterations": 5,
        "gaps": ["Need more information on pricing comparison"],
        "quality_metrics": sample_quality_metrics,
        "policy_violations": [],
        "blocked": False,
        "token_budget": 500000,
        "tokens_used": 15000,
        "tokens_remaining": 485000,
        "current_phase": "researching",
        "next_node": "synthesizer",
        "should_terminate": False,
        "error": None,
    }


@pytest.fixture
def completed_state(sample_state: dict[str, Any], sample_answer: str) -> dict[str, Any]:
    """Create a completed AgentState with final report."""
    state = sample_state.copy()
    state["final_report"] = sample_answer
    state["current_phase"] = "complete"
    state["should_terminate"] = True
    state["iteration_count"] = 2
    return state


# =============================================================================
# Mock LLM Client Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Standard mock LLM response structure."""
    return {
        "id": "mock-response-001",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": "google/gemini-2.5-flash-preview",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock LLM response for testing purposes.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


@pytest.fixture
def mock_openrouter(mock_llm_response: dict[str, Any]) -> Generator[MagicMock, None, None]:
    """
    Mock OpenRouter client for deterministic testing.

    Patches the httpx.AsyncClient to return controlled responses,
    enabling tests to run without actual API calls.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_llm_response
    mock_response.raise_for_status = MagicMock()

    async def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
        return mock_response

    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_tavily_response() -> dict[str, Any]:
    """Mock Tavily search response."""
    return {
        "query": "Stripe competitors payment processing",
        "results": [
            {
                "title": "Top Stripe Alternatives in 2024",
                "url": "https://example.com/stripe-alternatives",
                "content": "PayPal, Square, and Adyen are leading competitors...",
                "score": 0.95,
            },
            {
                "title": "Payment Processing Market Analysis",
                "url": "https://example.com/market-analysis",
                "content": "The payment processing market is dominated by...",
                "score": 0.88,
            },
        ],
    }


@pytest.fixture
def mock_tavily(mock_tavily_response: dict[str, Any]) -> Generator[MagicMock, None, None]:
    """Mock Tavily search client."""
    mock_client = MagicMock()

    async def mock_search(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return mock_tavily_response

    mock_client.search = AsyncMock(side_effect=mock_search)

    with patch("tavily.TavilyClient", return_value=mock_client):
        yield mock_client


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture
async def mock_db_connection() -> AsyncGenerator[MagicMock, None]:
    """
    Mock database connection for testing.

    Provides a mock asyncpg connection that can be used for
    testing database operations without a real database.
    """
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock(return_value="INSERT 1")
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=None)
    mock_conn.close = AsyncMock()

    yield mock_conn


@pytest.fixture
async def mock_redis() -> AsyncGenerator[MagicMock, None]:
    """
    Mock Redis client for testing.

    Provides an in-memory mock for Redis operations.
    """
    mock_client = MagicMock()
    cache: dict[str, Any] = {}

    async def mock_get(key: str) -> Any:
        return cache.get(key)

    async def mock_set(key: str, value: Any, **kwargs: Any) -> bool:
        cache[key] = value
        return True

    async def mock_delete(key: str) -> int:
        if key in cache:
            del cache[key]
            return 1
        return 0

    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.set = AsyncMock(side_effect=mock_set)
    mock_client.delete = AsyncMock(side_effect=mock_delete)
    mock_client.close = AsyncMock()

    yield mock_client


# =============================================================================
# Orchestrator Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator(
    mock_openrouter: MagicMock,
    mock_tavily: MagicMock,
    sample_state: dict[str, Any],
) -> MagicMock:
    """
    Mock ResearchOrchestrator for testing.

    Provides a mock orchestrator that simulates the research workflow
    without executing actual agent logic or API calls.
    """
    orchestrator = MagicMock()
    orchestrator.state = sample_state

    async def mock_run(query: str, **kwargs: Any) -> dict[str, Any]:
        return sample_state

    async def mock_resume(session_id: str) -> dict[str, Any]:
        return sample_state

    orchestrator.run = AsyncMock(side_effect=mock_run)
    orchestrator.resume = AsyncMock(side_effect=mock_resume)
    orchestrator.get_state = MagicMock(return_value=sample_state)

    return orchestrator


@pytest.fixture
async def orchestrator(
    mock_openrouter: MagicMock,
    mock_tavily: MagicMock,
    mock_db_connection: MagicMock,
    mock_redis: MagicMock,
) -> AsyncGenerator[MagicMock, None]:
    """
    ResearchOrchestrator instance with mocked dependencies.

    This fixture provides an orchestrator that can be used for
    integration testing with controlled mock responses.
    """
    # Import here to avoid circular imports and ensure mocks are in place
    orchestrator = MagicMock()
    orchestrator.llm_client = mock_openrouter
    orchestrator.search_client = mock_tavily
    orchestrator.db = mock_db_connection
    orchestrator.cache = mock_redis

    yield orchestrator


# =============================================================================
# DeepEval / Ragas Fixtures
# =============================================================================


@pytest.fixture
def eval_test_case(
    sample_query: str,
    sample_answer: str,
    sample_context: list[str],
) -> dict[str, Any]:
    """
    Create a test case structure for DeepEval evaluation.

    Returns:
        Dictionary compatible with LLMTestCase construction
    """
    return {
        "input": sample_query,
        "actual_output": sample_answer,
        "retrieval_context": sample_context,
        "expected_output": None,  # Can be set for specific tests
    }


@pytest.fixture
def eval_dataset(
    research_scenarios: dict[str, Any],
    sample_context: list[str],
) -> list[dict[str, Any]]:
    """
    Create an evaluation dataset from scenarios.

    Transforms TAU-bench style scenarios into DeepEval-compatible
    test cases for batch evaluation.
    """
    dataset = []

    scenarios = research_scenarios.get("scenarios", [])
    for scenario in scenarios:
        test_case = {
            "input": scenario.get("input", ""),
            "actual_output": "",  # To be filled during evaluation
            "retrieval_context": sample_context,
            "expected_output": scenario.get("expected_outputs", {}).get("ground_truth"),
            "metadata": {
                "scenario_id": scenario.get("id", "unknown"),
                "description": scenario.get("description", ""),
                "constraints": scenario.get("constraints", {}),
                "evaluation": scenario.get("evaluation", {}),
            },
        }
        dataset.append(test_case)

    return dataset


@pytest.fixture
def evaluation_thresholds() -> dict[str, float]:
    """
    Evaluation metric thresholds for CI gates.

    These thresholds determine pass/fail criteria for the evaluation pipeline.
    Hard gates block deployment; soft gates generate warnings.
    """
    return {
        # Hard gates (must pass)
        "faithfulness": 0.8,
        "task_completion": 0.7,
        "hallucination_max": 0.2,  # Maximum allowed (lower is better)
        "policy_violations": 0,  # Must be zero

        # Soft gates (warnings)
        "answer_relevancy": 0.7,
        "context_precision": 0.6,
        "context_recall": 0.6,
    }


# =============================================================================
# Results Output Fixtures
# =============================================================================


@pytest.fixture
def eval_results_path(evaluation_dir: Path) -> Path:
    """Path for evaluation results JSON output."""
    return evaluation_dir / "eval_results.json"


@pytest.fixture
def save_eval_results(eval_results_path: Path):
    """
    Factory fixture for saving evaluation results.

    Returns a callable that saves results to JSON for CI consumption.
    """
    def _save_results(results: dict[str, Any]) -> Path:
        results["timestamp"] = datetime.utcnow().isoformat() + "Z"
        results["version"] = "1.0.0"

        with open(eval_results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return eval_results_path

    return _save_results


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
async def cleanup_after_test() -> AsyncGenerator[None, None]:
    """
    Automatic cleanup after each test.

    Ensures test isolation by cleaning up any resources.
    """
    yield
    # Cleanup logic here if needed
    await asyncio.sleep(0)  # Allow any pending async operations to complete


# =============================================================================
# Shared Mock Classes
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


class MockCircuitBreakerConfig:
    """Mock circuit breaker configuration for testing."""

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
    """
    Mock AgentManifest for testing compliance assertions and circuit breaker.

    This consolidated mock supports both metadata compliance testing
    (test_metadata_compliance.py) and circuit breaker testing
    (test_circuit_breaker.py).
    """

    def __init__(
        self,
        agent_id: str = "test_agent_v1",
        agent_type: str = "searcher",
        capabilities: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_budget_usd: float = 1.0,
        rate_limits: MockRateLimits | None = None,
        circuit_breaker: MockCircuitBreakerConfig | None = None,
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
        self.circuit_breaker = circuit_breaker or MockCircuitBreakerConfig()
        self.allowed_tools = allowed_tools or ["web_search", "url_fetch"]
        self.is_active = is_active


class MockRedisClient:
    """
    Mock Redis client for testing.

    Provides full hash operation support for circuit breaker state management
    and general key-value operations for caching.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._hash_data: dict[str, dict[str, str]] = {}

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, **kwargs: Any) -> bool:
        self._data[key] = value
        return True

    async def delete(self, key: str) -> int:
        if key in self._data:
            del self._data[key]
            return 1
        return 0

    async def hget(self, key: str, field: str) -> str | None:
        if key in self._hash_data:
            return self._hash_data[key].get(field)
        return None

    async def hset(self, key: str, mapping: dict[str, str] | None = None, **kwargs: Any) -> int:
        if key not in self._hash_data:
            self._hash_data[key] = {}
        if mapping:
            self._hash_data[key].update(mapping)
        return len(mapping) if mapping else 0

    async def hdel(self, key: str, *fields: str) -> int:
        count = 0
        if key in self._hash_data:
            for field in fields:
                if field in self._hash_data[key]:
                    del self._hash_data[key][field]
                    count += 1
        return count

    async def hgetall(self, key: str) -> dict[str, str]:
        return self._hash_data.get(key, {})

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        if key not in self._hash_data:
            self._hash_data[key] = {}
        current = int(self._hash_data[key].get(field, "0"))
        new_value = current + amount
        self._hash_data[key][field] = str(new_value)
        return new_value

    async def incrbyfloat(self, key: str, amount: float) -> float:
        current = float(self._data.get(key, "0"))
        new_value = current + amount
        self._data[key] = str(new_value)
        return new_value

    async def expire(self, key: str, seconds: int) -> bool:
        return True

    async def incr(self, key: str) -> int:
        current = int(self._data.get(key, "0"))
        new_value = current + 1
        self._data[key] = str(new_value)
        return new_value

    async def ping(self) -> bool:
        return True

    def pipeline(self, transaction: bool = True) -> "MockRedisPipeline":
        return MockRedisPipeline(self)


class MockRedisPipeline:
    """Mock Redis pipeline for atomic operations."""

    def __init__(self, client: MockRedisClient) -> None:
        self._client = client
        self._commands: list[tuple[str, tuple, dict]] = []

    async def __aenter__(self) -> "MockRedisPipeline":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def set(self, key: str, value: str, **kwargs: Any) -> "MockRedisPipeline":
        self._commands.append(("set", (key, value), kwargs))
        return self

    async def delete(self, key: str) -> "MockRedisPipeline":
        self._commands.append(("delete", (key,), {}))
        return self

    async def hset(self, key: str, mapping: dict[str, str] | None = None, **kwargs: Any) -> "MockRedisPipeline":
        self._commands.append(("hset", (key, mapping), kwargs))
        return self

    async def hdel(self, key: str, *fields: str) -> "MockRedisPipeline":
        self._commands.append(("hdel", (key, *fields), {}))
        return self

    async def hincrby(self, key: str, field: str, amount: int = 1) -> "MockRedisPipeline":
        self._commands.append(("hincrby", (key, field, amount), {}))
        return self

    async def expire(self, key: str, seconds: int) -> "MockRedisPipeline":
        self._commands.append(("expire", (key, seconds), {}))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for cmd, args, kwargs in self._commands:
            if cmd == "set":
                await self._client.set(args[0], args[1], **kwargs)
                results.append(True)
            elif cmd == "delete":
                result = await self._client.delete(args[0])
                results.append(result)
            elif cmd == "hset":
                result = await self._client.hset(args[0], args[1], **kwargs)
                results.append(result)
            elif cmd == "hdel":
                result = await self._client.hdel(args[0], *args[1:])
                results.append(result)
            elif cmd == "hincrby":
                result = await self._client.hincrby(args[0], args[1], args[2])
                results.append(result)
            elif cmd == "expire":
                results.append(True)
            else:
                results.append(None)
        self._commands.clear()
        return results


class MockActiveStateService:
    """
    Mock ActiveStateService for circuit breaker and policy firewall testing.

    Provides in-memory state management for agent circuit status and
    failure/success counting.
    """

    def __init__(self, redis_client: MockRedisClient | None = None) -> None:
        import time
        self._time = time
        self.redis = redis_client or MockRedisClient()
        self._initialized = True
        self._circuit_states: dict[str, str] = {}  # "closed", "open", "half_open"
        self._failure_counts: dict[str, int] = {}

    async def initialize(self) -> None:
        self._initialized = True

    async def get_circuit_status(self, agent_id: str) -> str:
        return self._circuit_states.get(agent_id, "closed")

    async def set_circuit_status(self, agent_id: str, status: str) -> None:
        self._circuit_states[agent_id] = status
        await self.redis.set(f"drx:agent:{agent_id}:circuit", status)

        if status == "open":
            await self.redis.set(
                f"drx:agent:{agent_id}:circuit_opened_at",
                str(self._time.time()),
            )

    async def increment_failure_count(self, agent_id: str) -> int:
        self._failure_counts[agent_id] = self._failure_counts.get(agent_id, 0) + 1
        return self._failure_counts[agent_id]

    async def reset_failure_count(self, agent_id: str) -> None:
        self._failure_counts[agent_id] = 0
        await self.redis.hdel(f"drx:agent:{agent_id}:health", "failure_count")

    async def get_all_agent_ids(self) -> list[str]:
        return list(self._circuit_states.keys())

    def _circuit_opened_at_key(self, agent_id: str) -> str:
        return f"drx:agent:{agent_id}:circuit_opened_at"


# =============================================================================
# Shared Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_manifest() -> MockManifest:
    """
    Return a default MockManifest for testing.

    Provides a standard agent manifest with common defaults suitable
    for most test scenarios.
    """
    return MockManifest(
        agent_id="test_agent_v1",
        agent_type="searcher",
        capabilities=["web_search", "source_discovery"],
        max_budget_usd=5.0,
    )


@pytest.fixture
def mock_redis_client() -> MockRedisClient:
    """
    Return a MockRedisClient for testing.

    Provides full Redis operation support including hash operations
    for circuit breaker state management.
    """
    return MockRedisClient()


@pytest.fixture
def mock_active_state_service(mock_redis_client: MockRedisClient) -> MockActiveStateService:
    """
    Return a MockActiveStateService for testing.

    Provides in-memory agent state management for circuit breaker
    and policy firewall testing.
    """
    return MockActiveStateService(mock_redis_client)


# =============================================================================
# Marker-based Skipping
# =============================================================================


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Modify test collection based on markers and environment.

    Skips integration tests when running in unit-test-only mode,
    and skips eval tests when dependencies are not installed.
    """
    skip_integration = pytest.mark.skip(reason="Integration tests disabled")
    skip_eval = pytest.mark.skip(reason="Eval dependencies not installed")

    # Check if eval dependencies are available
    try:
        import deepeval  # noqa: F401
        eval_available = True
    except ImportError:
        eval_available = False

    for item in items:
        # Skip integration tests if not enabled
        if "integration" in item.keywords:
            if os.environ.get("SKIP_INTEGRATION_TESTS", "false").lower() == "true":
                item.add_marker(skip_integration)

        # Skip eval tests if dependencies not installed
        if "eval" in item.keywords and not eval_available:
            item.add_marker(skip_eval)
