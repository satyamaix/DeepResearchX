"""
Parallel Execution Engine for DRX Deep Research System.

Enables concurrent execution of independent DAG tasks with
proper dependency resolution, fan-out/fan-in patterns, and
result aggregation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from src.agents.base import BaseAgent
    from src.orchestrator.state import AgentState, ResearchPlan, SubTask

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================


@dataclass
class TaskResult:
    """Result from executing a single task."""
    task_id: str
    success: bool
    output: dict[str, Any]
    error: str | None = None
    tokens_used: int = 0
    execution_time_ms: float = 0
    completed_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


@dataclass
class AggregatedResult:
    """Aggregated results from parallel task execution."""
    results: list[TaskResult]
    total_tokens: int
    successful_count: int
    failed_count: int
    execution_time_ms: float

    @property
    def all_successful(self) -> bool:
        return self.failed_count == 0

    def get_outputs(self) -> dict[str, dict[str, Any]]:
        """Get outputs keyed by task ID."""
        return {r.task_id: r.output for r in self.results if r.success}


@dataclass
class DependencyGraph:
    """Represents task dependencies for execution ordering."""
    # task_id -> list of task_ids it depends on
    dependencies: dict[str, list[str]]
    # task_id -> list of task_ids that depend on it
    dependents: dict[str, list[str]]
    # All task IDs in topological order
    topological_order: list[str]

    def get_ready_tasks(self, completed: set[str]) -> list[str]:
        """Get tasks whose dependencies are all completed."""
        ready = []
        for task_id, deps in self.dependencies.items():
            if task_id not in completed:
                if all(d in completed for d in deps):
                    ready.append(task_id)
        return ready


# =============================================================================
# Parallel Executor
# =============================================================================


class ParallelExecutor:
    """
    Execute DAG tasks in parallel where dependencies allow.

    Features:
    - Automatic dependency resolution
    - Concurrent execution of independent tasks
    - Fan-out (parallel spawn) and fan-in (result aggregation)
    - Configurable concurrency limits
    - Error handling and partial failure recovery

    Example:
        ```python
        executor = ParallelExecutor(agents, max_concurrency=5)

        while not is_plan_complete(state["plan"]):
            results = await executor.execute_ready_tasks(state["plan"], state)
            state = merge_task_results(state, results)
        ```
    """

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        max_concurrency: int = 5,
        task_timeout: float = 300.0,  # 5 minutes per task
    ):
        """
        Initialize the parallel executor.

        Args:
            agents: Map of agent_type to agent instance
            max_concurrency: Maximum concurrent tasks
            task_timeout: Timeout per task in seconds
        """
        self._agents = agents
        self._max_concurrency = max_concurrency
        self._task_timeout = task_timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def resolve_dependencies(self, plan: ResearchPlan) -> DependencyGraph:
        """
        Build dependency graph from research plan.

        Args:
            plan: The research plan with DAG nodes

        Returns:
            DependencyGraph for execution ordering
        """
        dependencies: dict[str, list[str]] = {}
        dependents: dict[str, list[str]] = {}

        for task in plan.get("dag_nodes", []):
            task_id = task["id"]
            dependencies[task_id] = task.get("dependencies", [])

            # Build reverse mapping
            if task_id not in dependents:
                dependents[task_id] = []
            for dep_id in task.get("dependencies", []):
                if dep_id not in dependents:
                    dependents[dep_id] = []
                dependents[dep_id].append(task_id)

        # Topological sort using Kahn's algorithm
        topological_order = self._topological_sort(dependencies)

        return DependencyGraph(
            dependencies=dependencies,
            dependents=dependents,
            topological_order=topological_order,
        )

    def _topological_sort(self, dependencies: dict[str, list[str]]) -> list[str]:
        """Kahn's algorithm for topological sorting."""
        # Calculate in-degrees
        in_degree: dict[str, int] = {k: 0 for k in dependencies}
        for deps in dependencies.values():
            for dep in deps:
                if dep in in_degree:
                    pass  # dep should already be counted

        for task_id, deps in dependencies.items():
            in_degree[task_id] = len(deps)

        # Start with nodes that have no dependencies
        queue = [k for k, v in in_degree.items() if v == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Decrease in-degree of dependents
            for task_id, deps in dependencies.items():
                if node in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)

        return result

    def get_ready_tasks(self, plan: ResearchPlan) -> list[SubTask]:
        """
        Get tasks that are ready for execution.

        A task is ready when:
        - Status is "pending"
        - All dependencies are "completed"

        Args:
            plan: Current research plan

        Returns:
            List of ready tasks
        """
        completed_ids = {
            task["id"]
            for task in plan.get("dag_nodes", [])
            if task["status"] == "completed"
        }

        ready = []
        for task in plan.get("dag_nodes", []):
            if task["status"] == "pending":
                deps_satisfied = all(
                    dep_id in completed_ids
                    for dep_id in task.get("dependencies", [])
                )
                if deps_satisfied:
                    ready.append(task)

        return ready

    async def execute_ready_tasks(
        self,
        plan: ResearchPlan,
        state: AgentState,
    ) -> AggregatedResult:
        """
        Execute all ready tasks in parallel.

        Args:
            plan: Research plan with tasks
            state: Current agent state

        Returns:
            Aggregated results from all executed tasks
        """
        import time
        start_time = time.perf_counter()

        ready_tasks = self.get_ready_tasks(plan)

        if not ready_tasks:
            return AggregatedResult(
                results=[],
                total_tokens=0,
                successful_count=0,
                failed_count=0,
                execution_time_ms=0,
            )

        logger.info(f"Executing {len(ready_tasks)} tasks in parallel")

        # Fan-out: Create coroutines for all ready tasks
        coroutines = [
            self._execute_task_with_semaphore(task, state)
            for task in ready_tasks
        ]

        # Execute concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Fan-in: Aggregate results
        task_results: list[TaskResult] = []
        for i, result in enumerate(results):
            task = ready_tasks[i]

            if isinstance(result, Exception):
                task_results.append(TaskResult(
                    task_id=task["id"],
                    success=False,
                    output={},
                    error=str(result),
                ))
            else:
                task_results.append(result)

        execution_time = (time.perf_counter() - start_time) * 1000

        return AggregatedResult(
            results=task_results,
            total_tokens=sum(r.tokens_used for r in task_results),
            successful_count=sum(1 for r in task_results if r.success),
            failed_count=sum(1 for r in task_results if not r.success),
            execution_time_ms=execution_time,
        )

    async def _execute_task_with_semaphore(
        self,
        task: SubTask,
        state: AgentState,
    ) -> TaskResult:
        """Execute a single task with concurrency limiting."""
        async with self._semaphore:
            return await self._execute_task(task, state)

    async def _execute_task(
        self,
        task: SubTask,
        state: AgentState,
    ) -> TaskResult:
        """Execute a single task."""
        import time
        start_time = time.perf_counter()

        task_id = task["id"]
        agent_type = task["agent_type"]

        logger.debug(f"Executing task {task_id} with agent {agent_type}")

        # Get the appropriate agent
        agent = self._agents.get(agent_type)
        if agent is None:
            return TaskResult(
                task_id=task_id,
                success=False,
                output={},
                error=f"No agent found for type: {agent_type}",
            )

        try:
            # Execute with timeout
            async with asyncio.timeout(self._task_timeout):
                # Prepare task-specific state
                task_state = self._prepare_task_state(task, state)

                # Run the agent
                result = await agent.run(task_state)

                execution_time = (time.perf_counter() - start_time) * 1000

                if result.success:
                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        output=result.data,
                        tokens_used=result.tokens_used,
                        execution_time_ms=execution_time,
                    )
                else:
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        output={},
                        error=result.error,
                        tokens_used=result.tokens_used,
                        execution_time_ms=execution_time,
                    )

        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task_id,
                success=False,
                output={},
                error=f"Task timed out after {self._task_timeout}s",
            )
        except Exception as e:
            logger.exception(f"Task {task_id} failed with exception")
            return TaskResult(
                task_id=task_id,
                success=False,
                output={},
                error=str(e),
            )

    def _prepare_task_state(
        self,
        task: SubTask,
        state: AgentState,
    ) -> AgentState:
        """Prepare state for a specific task."""
        # Include task inputs in the state
        task_inputs = task.get("inputs", {})

        # Create a copy with task-specific context
        task_state = dict(state)
        task_state["current_task"] = task
        task_state["task_inputs"] = task_inputs

        return task_state  # type: ignore


# =============================================================================
# State Merge Utilities
# =============================================================================


def merge_task_results(
    state: AgentState,
    results: AggregatedResult,
) -> AgentState:
    """
    Merge task results back into the agent state.

    Args:
        state: Current state
        results: Aggregated task results

    Returns:
        Updated state with task outputs merged
    """
    from src.orchestrator.state import update_tokens_used

    plan = state.get("plan")
    if not plan:
        return state

    # Update task statuses in plan
    for task_result in results.results:
        for task in plan["dag_nodes"]:
            if task["id"] == task_result.task_id:
                if task_result.success:
                    task["status"] = "completed"
                    task["outputs"] = task_result.output
                    task["completed_at"] = task_result.completed_at
                else:
                    task["status"] = "failed"
                    task["error"] = task_result.error
                break

    # Merge outputs into state based on agent type
    findings = list(state.get("findings", []))
    citations = list(state.get("citations", []))
    synthesis = state.get("synthesis", "")
    gaps = list(state.get("gaps", []))

    for task_result in results.results:
        if not task_result.success:
            continue

        output = task_result.output

        # Merge findings
        if "findings" in output:
            findings.extend(output["findings"])

        # Merge citations
        if "citations" in output:
            citations.extend(output["citations"])

        # Update synthesis
        if "synthesis" in output and output["synthesis"]:
            synthesis = output["synthesis"]

        # Merge gaps
        if "gaps" in output:
            gaps.extend(output["gaps"])

    # Update token usage
    token_update = update_tokens_used(state, results.total_tokens)

    return {
        **state,
        "plan": plan,
        "findings": findings,
        "citations": citations,
        "synthesis": synthesis,
        "gaps": gaps,
        **token_update,
    }


def is_plan_complete(plan: ResearchPlan | None) -> bool:
    """Check if all tasks in the plan are completed or failed."""
    if not plan:
        return True

    for task in plan.get("dag_nodes", []):
        if task["status"] not in ("completed", "failed"):
            return False
    return True


def has_critical_failures(plan: ResearchPlan | None) -> bool:
    """Check if there are failed tasks that block progress."""
    if not plan:
        return False

    failed_ids = {
        task["id"]
        for task in plan.get("dag_nodes", [])
        if task["status"] == "failed"
    }

    # Check if any pending task depends on a failed task
    for task in plan.get("dag_nodes", []):
        if task["status"] == "pending":
            for dep_id in task.get("dependencies", []):
                if dep_id in failed_ids:
                    return True

    return False


# =============================================================================
# Factory Function
# =============================================================================


def create_parallel_executor(
    agents: dict[str, BaseAgent] | None = None,
    **kwargs: Any,
) -> ParallelExecutor:
    """
    Factory function to create a ParallelExecutor.

    Args:
        agents: Map of agent types to instances. If None, creates defaults.
        **kwargs: Additional configuration

    Returns:
        Configured ParallelExecutor
    """
    if agents is None:
        # Import and create default agents
        from src.agents import (
            create_planner_agent,
            create_searcher_agent,
            create_reader_agent,
            create_synthesizer_agent,
            create_critic_agent,
            create_reporter_agent,
        )

        agents = {
            "planner": create_planner_agent(),
            "searcher": create_searcher_agent(),
            "reader": create_reader_agent(),
            "synthesizer": create_synthesizer_agent(),
            "critic": create_critic_agent(),
            "reporter": create_reporter_agent(),
        }

    return ParallelExecutor(agents=agents, **kwargs)


__all__ = [
    "ParallelExecutor",
    "TaskResult",
    "AggregatedResult",
    "DependencyGraph",
    "merge_task_results",
    "is_plan_complete",
    "has_critical_failures",
    "create_parallel_executor",
]
