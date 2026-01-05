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
