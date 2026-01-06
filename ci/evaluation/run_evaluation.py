"""Live Evaluation Runner for DRX Deep Research.

Executes research requests against the live DRX API and collects outputs
for evaluation. Supports scenario-based testing, configurable timeouts,
and handling of both positive and negative test cases.

WP-0B: Evaluation Pipeline Implementation

Usage:
    python ci/evaluation/run_evaluation.py --scenarios scenarios/research_tasks.yaml --output results.json
    python ci/evaluation/run_evaluation.py --api-url http://localhost:8000 --scenario-ids competitor_analysis,quick_fact_check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import httpx
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions (TypedDict for LangGraph compatibility)
# =============================================================================


class EvaluationResult(TypedDict):
    """Result of evaluating a single research scenario.

    Contains all outputs needed for downstream evaluation metrics
    including faithfulness, relevancy, and policy compliance.
    """
    scenario_id: str
    input: str
    actual_output: str  # The final report
    retrieval_context: list[str]  # Retrieved sources/citations
    citations: list[dict[str, Any]]
    duration_seconds: float
    success: bool
    error: str | None
    policy_blocked: bool


class ScenarioConfig(TypedDict, total=False):
    """Configuration for a research scenario."""
    id: str
    name: str
    description: str
    category: str
    priority: str
    input: str
    expected_outputs: list[dict[str, Any]]
    constraints: dict[str, Any]
    evaluation: dict[str, Any]
    steerability: dict[str, Any]
    metadata: dict[str, Any]


class RunnerConfig(TypedDict, total=False):
    """Configuration for the evaluation runner."""
    api_base_url: str
    default_timeout: int
    max_retries: int
    retry_delay: float
    poll_interval: float


# =============================================================================
# EvaluationRunner Implementation
# =============================================================================


class EvaluationRunner:
    """Execute research requests and collect outputs for evaluation.

    This class integrates with the DRX API to:
    1. Submit research requests
    2. Poll for completion or stream SSE responses
    3. Collect outputs for downstream evaluation
    4. Handle timeouts, errors, and policy blocks

    Attributes:
        api_base_url: Base URL for the DRX API.
        client: Async HTTP client for API requests.
        default_timeout: Default timeout in seconds for requests.
        max_retries: Maximum number of retries for failed requests.
        retry_delay: Delay between retries in seconds.
        poll_interval: Interval between polling requests in seconds.
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        default_timeout: int = 600,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        poll_interval: float = 2.0,
    ) -> None:
        """Initialize the evaluation runner.

        Args:
            api_base_url: Base URL for the DRX API.
            default_timeout: Default timeout in seconds for requests.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Delay between retries in seconds.
            poll_interval: Interval between polling requests in seconds.
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.poll_interval = poll_interval

        # HTTP client will be created in context manager
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "EvaluationRunner":
        """Enter async context and create HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.api_base_url,
            timeout=httpx.Timeout(30.0, read=None),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, ensuring it exists."""
        if self._client is None:
            raise RuntimeError(
                "EvaluationRunner must be used as an async context manager"
            )
        return self._client

    async def run_scenario(self, scenario: dict[str, Any]) -> EvaluationResult:
        """Execute a single research scenario.

        This method:
        1. Submits the research request to the API
        2. Polls for completion or handles streaming
        3. Collects the final output and metadata
        4. Returns structured evaluation results

        Args:
            scenario: Scenario configuration dictionary.

        Returns:
            EvaluationResult with collected outputs.
        """
        scenario_id = scenario.get("id", "unknown")
        input_query = scenario.get("input", "")
        constraints = scenario.get("constraints", {})
        steerability = scenario.get("steerability", {})

        # Get timeout - support both formats:
        # - constraints.timeout_seconds (research_tasks.yaml)
        # - timeout_seconds (curated_test_cases.yaml)
        timeout = (
            constraints.get("timeout_seconds")
            or scenario.get("timeout_seconds")
            or self.default_timeout
        )

        # Check if this is a negative test (expects policy block)
        # Support both formats:
        # - evaluation.must_block (research_tasks.yaml)
        # - is_negative_test (curated_test_cases.yaml)
        evaluation_config = scenario.get("evaluation", {})
        expects_block = evaluation_config.get("must_block", False) or scenario.get("is_negative_test", False)

        logger.info(f"Running scenario: {scenario_id}")
        start_time = time.time()

        try:
            # Submit research request
            interaction_id, initial_status = await self._submit_request(
                query=input_query,
                steerability=steerability,
                constraints=constraints,
            )

            # Check for immediate policy block
            if initial_status in ("blocked", "rejected"):
                duration = time.time() - start_time
                return EvaluationResult(
                    scenario_id=scenario_id,
                    input=input_query,
                    actual_output="",
                    retrieval_context=[],
                    citations=[],
                    duration_seconds=duration,
                    success=expects_block,  # Success if we expected the block
                    error=None,
                    policy_blocked=True,
                )

            # Poll for completion
            result = await self.poll_for_completion(
                interaction_id=interaction_id,
                timeout=timeout,
            )

            duration = time.time() - start_time

            # Check if research was blocked during execution
            if result.get("status") in ("blocked", "rejected"):
                return EvaluationResult(
                    scenario_id=scenario_id,
                    input=input_query,
                    actual_output="",
                    retrieval_context=[],
                    citations=[],
                    duration_seconds=duration,
                    success=expects_block,
                    error=None,
                    policy_blocked=True,
                )

            # Extract results
            result_data = result.get("result", {})
            final_report = result_data.get("final_report", "") or ""
            citations = result_data.get("citations", []) or []
            findings = result_data.get("findings", []) or []

            # Build retrieval context from citations and findings
            retrieval_context = self._build_retrieval_context(citations, findings)

            # Check for error state
            error = result.get("error")
            status = result.get("status", "unknown")
            success = status == "completed" and not error

            # For negative tests, invert success logic
            if expects_block and not result_data:
                # If we expected a block but didn't get one, check if report is empty
                success = not bool(final_report)
                if not success:
                    error = "Expected policy block but received output"

            return EvaluationResult(
                scenario_id=scenario_id,
                input=input_query,
                actual_output=final_report,
                retrieval_context=retrieval_context,
                citations=citations,
                duration_seconds=duration,
                success=success,
                error=error,
                policy_blocked=False,
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(f"Scenario {scenario_id} timed out after {timeout}s")
            return EvaluationResult(
                scenario_id=scenario_id,
                input=input_query,
                actual_output="",
                retrieval_context=[],
                citations=[],
                duration_seconds=duration,
                success=False,
                error=f"Timeout after {timeout} seconds",
                policy_blocked=False,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Scenario {scenario_id} failed: {e}", exc_info=True)
            return EvaluationResult(
                scenario_id=scenario_id,
                input=input_query,
                actual_output="",
                retrieval_context=[],
                citations=[],
                duration_seconds=duration,
                success=False,
                error=str(e),
                policy_blocked=False,
            )

    async def run_all(
        self,
        scenarios: list[dict[str, Any]],
        parallel: bool = False,
    ) -> list[EvaluationResult]:
        """Execute all scenarios and collect results.

        Args:
            scenarios: List of scenario configurations.
            parallel: Whether to run scenarios in parallel.

        Returns:
            List of EvaluationResult for each scenario.
        """
        results: list[EvaluationResult] = []
        total = len(scenarios)

        logger.info(f"Running {total} scenarios...")

        if parallel:
            # Run scenarios in parallel
            tasks = [self.run_scenario(s) for s in scenarios]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            processed_results: list[EvaluationResult] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    scenario = scenarios[i]
                    processed_results.append(
                        EvaluationResult(
                            scenario_id=scenario.get("id", f"scenario_{i}"),
                            input=scenario.get("input", ""),
                            actual_output="",
                            retrieval_context=[],
                            citations=[],
                            duration_seconds=0.0,
                            success=False,
                            error=str(result),
                            policy_blocked=False,
                        )
                    )
                else:
                    processed_results.append(result)
            results = processed_results
        else:
            # Run scenarios sequentially
            for i, scenario in enumerate(scenarios, 1):
                logger.info(f"Progress: {i}/{total}")
                result = await self.run_scenario(scenario)
                results.append(result)

        # Log summary
        success_count = sum(1 for r in results if r["success"])
        logger.info(f"Completed: {success_count}/{total} scenarios passed")

        return results

    async def poll_for_completion(
        self,
        interaction_id: str,
        timeout: int,
    ) -> dict[str, Any]:
        """Poll the API until research completes or times out.

        Polls the interaction status endpoint at regular intervals
        until the interaction reaches a terminal state or timeout.

        Args:
            interaction_id: ID of the interaction to poll.
            timeout: Maximum time to wait in seconds.

        Returns:
            Final interaction state dictionary.

        Raises:
            asyncio.TimeoutError: If timeout is reached.
        """
        start_time = time.time()
        terminal_states = {"completed", "complete", "failed", "cancelled", "blocked", "rejected"}

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise asyncio.TimeoutError(
                    f"Polling timed out after {timeout} seconds"
                )

            try:
                response = await self.client.get(
                    f"/api/v1/interactions/{interaction_id}"
                )

                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")

                    logger.debug(
                        f"Interaction {interaction_id} status: {status} "
                        f"(elapsed: {elapsed:.1f}s)"
                    )

                    if status in terminal_states:
                        return data

                elif response.status_code == 404:
                    logger.warning(
                        f"Interaction {interaction_id} not found, continuing poll..."
                    )

                else:
                    logger.warning(
                        f"Unexpected status code {response.status_code} "
                        f"while polling {interaction_id}"
                    )

            except httpx.RequestError as e:
                logger.warning(f"Request error while polling: {e}")

            # Wait before next poll
            await asyncio.sleep(self.poll_interval)

    async def _submit_request(
        self,
        query: str,
        steerability: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Submit a research request to the API.

        Args:
            query: Research query.
            steerability: Steerability configuration.
            constraints: Research constraints.

        Returns:
            Tuple of (interaction_id, initial_status).

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        # Build request payload
        payload: dict[str, Any] = {"input": query}

        if steerability:
            payload["steerability"] = steerability

        if constraints:
            # Map constraints to config format
            config: dict[str, Any] = {}
            if "max_iterations" in constraints:
                config["max_iterations"] = constraints["max_iterations"]
            if "timeout_seconds" in constraints:
                config["timeout_seconds"] = constraints["timeout_seconds"]
            if config:
                payload["config"] = config

        # Submit request with retries
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    "/api/v1/interactions",
                    json=payload,
                )

                if response.status_code in (200, 201, 202):
                    data = response.json()
                    interaction_id = data.get("id", "")
                    status = data.get("status", "pending")
                    logger.info(
                        f"Created interaction {interaction_id} with status {status}"
                    )
                    return interaction_id, status

                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                elif response.status_code == 403:
                    # Policy block
                    logger.info("Request blocked by policy")
                    return "", "blocked"

                elif response.status_code == 400:
                    # Bad request - might be policy rejection
                    error_data = response.json()
                    error_msg = error_data.get("detail", "Bad request")
                    if "policy" in error_msg.lower() or "blocked" in error_msg.lower():
                        return "", "rejected"
                    raise httpx.HTTPStatusError(
                        f"Bad request: {error_msg}",
                        request=response.request,
                        response=response,
                    )

                else:
                    response.raise_for_status()

            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        if last_error:
            raise last_error
        raise RuntimeError("Failed to submit request after retries")

    def _build_retrieval_context(
        self,
        citations: list[dict[str, Any]],
        findings: list[dict[str, Any]],
    ) -> list[str]:
        """Build retrieval context from citations and findings.

        Creates a list of context strings for evaluation metrics
        like faithfulness and context precision.

        Args:
            citations: List of citation records.
            findings: List of finding records.

        Returns:
            List of context strings.
        """
        context: list[str] = []

        # Add citation snippets
        for citation in citations:
            snippet = citation.get("snippet", "")
            if snippet:
                title = citation.get("title", "")
                url = citation.get("url", "")
                context_str = f"{title}: {snippet}"
                if url:
                    context_str += f" (Source: {url})"
                context.append(context_str)

        # Add finding evidence
        for finding in findings:
            evidence = finding.get("evidence", "")
            if evidence:
                claim = finding.get("claim", "")
                context_str = f"{claim} - Evidence: {evidence}"
                context.append(context_str)

        return context


# =============================================================================
# Scenario Loading Utilities
# =============================================================================


def load_scenarios(
    file_path: Path,
    scenario_ids: list[str] | None = None,
    group: str | None = None,
) -> list[dict[str, Any]]:
    """Load scenarios from a YAML file.

    Supports both formats:
    - research_tasks.yaml: uses 'scenarios' and 'scenario_groups'
    - curated_test_cases.yaml: uses 'test_cases' and 'test_groups'

    Args:
        file_path: Path to the scenarios YAML file.
        scenario_ids: Optional list of specific scenario IDs to load.
        group: Optional scenario group name to load.

    Returns:
        List of scenario dictionaries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Support both formats: 'scenarios' and 'test_cases'
    all_scenarios = data.get("scenarios", []) or data.get("test_cases", [])

    # Support both formats: 'scenario_groups' and 'test_groups'
    scenario_groups = data.get("scenario_groups", {}) or data.get("test_groups", {})

    # Build scenario lookup
    scenario_lookup = {s["id"]: s for s in all_scenarios if "id" in s}

    # Filter by group if specified
    if group:
        group_config = scenario_groups.get(group, {})
        # Support both 'scenarios' and 'test_ids' keys
        group_ids = group_config.get("scenarios", []) or group_config.get("test_ids", [])
        if not group_ids:
            logger.warning(f"Scenario group '{group}' not found or empty")
            return []
        return [scenario_lookup[sid] for sid in group_ids if sid in scenario_lookup]

    # Filter by specific IDs if specified
    if scenario_ids:
        return [scenario_lookup[sid] for sid in scenario_ids if sid in scenario_lookup]

    # Return all scenarios
    return all_scenarios


def save_results(
    results: list[EvaluationResult],
    output_path: Path,
    include_metadata: bool = True,
) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: List of evaluation results.
        output_path: Path to save the results.
        include_metadata: Whether to include run metadata.
    """
    output: dict[str, Any] = {
        "results": results,
    }

    if include_metadata:
        output["metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_scenarios": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "policy_blocked": sum(1 for r in results if r["policy_blocked"]),
            "total_duration_seconds": sum(r["duration_seconds"] for r in results),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run DRX evaluation scenarios against the live API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scenarios from a file
  python run_evaluation.py --scenarios scenarios/research_tasks.yaml

  # Run specific scenarios by ID
  python run_evaluation.py --scenarios scenarios/research_tasks.yaml --scenario-ids competitor_analysis,quick_fact_check

  # Run a scenario group
  python run_evaluation.py --scenarios scenarios/research_tasks.yaml --group smoke_test

  # Run against a different API URL
  python run_evaluation.py --scenarios scenarios/research_tasks.yaml --api-url http://api.example.com:8000
        """,
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        required=True,
        help="Path to scenarios YAML file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.json"),
        help="Path to save results JSON (default: eval_results.json)",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the DRX API (default: http://localhost:8000)",
    )

    parser.add_argument(
        "--scenario-ids",
        type=str,
        default=None,
        help="Comma-separated list of scenario IDs to run",
    )

    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Scenario group name to run",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Default timeout in seconds (default: 600)",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run scenarios in parallel",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point for the evaluation runner."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate scenarios file exists
    if not args.scenarios.exists():
        logger.error(f"Scenarios file not found: {args.scenarios}")
        return 1

    # Parse scenario IDs if provided
    scenario_ids = None
    if args.scenario_ids:
        scenario_ids = [s.strip() for s in args.scenario_ids.split(",")]

    # Load scenarios
    try:
        scenarios = load_scenarios(
            file_path=args.scenarios,
            scenario_ids=scenario_ids,
            group=args.group,
        )
    except Exception as e:
        logger.error(f"Failed to load scenarios: {e}")
        return 1

    if not scenarios:
        logger.error("No scenarios to run")
        return 1

    logger.info(f"Loaded {len(scenarios)} scenarios")

    # Run evaluation
    async with EvaluationRunner(
        api_base_url=args.api_url,
        default_timeout=args.timeout,
    ) as runner:
        results = await runner.run_all(
            scenarios=scenarios,
            parallel=args.parallel,
        )

    # Save results
    save_results(results, args.output)

    # Return exit code based on results
    failed = sum(1 for r in results if not r["success"])
    if failed > 0:
        logger.warning(f"{failed} scenario(s) failed")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "EvaluationResult",
    "EvaluationRunner",
    "load_scenarios",
    "save_results",
]
