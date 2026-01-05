"""
LangGraph StateGraph Workflow Definition for DRX Deep Research System.

This module defines the main research workflow as a LangGraph StateGraph,
implementing a cyclic DAG with conditional routing for iterative research.

Workflow Architecture:
```
                    +-------------------------------------+
                    |                                     |
                    v                                     |
START --> plan --> search --> read --> synthesize --> critique
                                                         |
                                          should_continue?
                                          |            |
                                    "continue"    "report"
                                          |            |
                                          +------------+
                                                       v
                                                 policy_check
                                                       |
                                                       v
                                                      END
```

Key Features:
- Cyclic execution with iteration limits
- Conditional routing based on gap analysis
- Async checkpointing with PostgreSQL
- Event streaming for progress tracking
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Literal,
    TypedDict,
)

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import get_settings
from src.orchestrator.checkpointer import (
    get_checkpointer,
    checkpointer_context,
    get_thread_state,
    list_thread_checkpoints,
)
from src.orchestrator.nodes import (
    check_policy,
    critique_synthesis,
    generate_report,
    has_policy_violation,
    plan_research,
    read_documents,
    search_sources,
    should_continue,
    synthesize_findings,
)
from src.orchestrator.state import (
    AgentState,
    SteerabilityParams,
    create_initial_state,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import CheckpointTuple

logger = logging.getLogger(__name__)


# =============================================================================
# Stream Event Types
# =============================================================================


class StreamEventType(str, Enum):
    """Types of events emitted during workflow execution."""

    # Workflow lifecycle events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_INTERRUPTED = "workflow_interrupted"

    # Node execution events
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"

    # Progress events
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"

    # Content events
    PLAN_CREATED = "plan_created"
    SOURCES_FOUND = "sources_found"
    FINDINGS_EXTRACTED = "findings_extracted"
    SYNTHESIS_GENERATED = "synthesis_generated"
    GAPS_IDENTIFIED = "gaps_identified"
    REPORT_GENERATED = "report_generated"

    # Token events
    TOKEN_UPDATE = "token_update"

    # Error events
    ERROR = "error"
    WARNING = "warning"

    # Checkpoint events
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESUMED = "checkpoint_resumed"


@dataclass
class StreamEvent:
    """
    Event emitted during workflow execution for progress tracking.

    Provides real-time updates on workflow state and progress.
    """

    event_type: StreamEventType
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    session_id: str = ""
    node_name: str | None = None
    iteration: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "node_name": self.node_name,
            "iteration": self.iteration,
            "data": self.data,
            "error": self.error,
        }


# =============================================================================
# Workflow Graph Builder
# =============================================================================


def create_research_workflow() -> StateGraph:
    """
    Create the research workflow StateGraph.

    Builds a cyclic graph with the following structure:
    - plan: Decompose query into research tasks
    - search: Find relevant sources
    - read: Extract information from sources
    - synthesize: Combine findings into coherent analysis
    - critique: Evaluate quality and identify gaps
    - policy_check: Verify content compliance
    - report: Generate final output

    Returns:
        StateGraph: Uncompiled workflow graph.
    """
    # Create the StateGraph with AgentState
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("plan", plan_research)
    workflow.add_node("search", search_sources)
    workflow.add_node("read", read_documents)
    workflow.add_node("synthesize", synthesize_findings)
    workflow.add_node("critique", critique_synthesis)
    workflow.add_node("policy_check", check_policy)
    workflow.add_node("report", generate_report)

    # Add edges for the main flow
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "read")
    workflow.add_edge("read", "synthesize")
    workflow.add_edge("synthesize", "critique")

    # Add conditional edge from critique
    # This creates the cyclic behavior for iterative research
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "continue": "search",  # Loop back for more research
            "report": "policy_check",  # Proceed to final output
        },
    )

    # Add conditional edge from policy_check
    workflow.add_conditional_edges(
        "policy_check",
        has_policy_violation,
        {
            "safe": "report",  # Generate report if policy passes
            "violation": END,  # End if blocked by policy
        },
    )

    # Final edge from report to END
    workflow.add_edge("report", END)

    logger.info("Research workflow graph created")
    return workflow


def compile_workflow(
    checkpointer: AsyncPostgresSaver | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph:
    """
    Compile the research workflow with optional checkpointing.

    Args:
        checkpointer: AsyncPostgresSaver for state persistence.
        interrupt_before: Nodes to pause before (for human-in-the-loop).
        interrupt_after: Nodes to pause after.

    Returns:
        CompiledStateGraph: Ready-to-execute workflow.
    """
    workflow = create_research_workflow()

    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )

    logger.info(
        "Workflow compiled",
        extra={
            "has_checkpointer": checkpointer is not None,
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
        },
    )

    return compiled


# =============================================================================
# Research Orchestrator Class
# =============================================================================


class ResearchOrchestrator:
    """
    High-level orchestrator for DRX research workflows.

    Manages workflow execution, checkpointing, and event streaming.
    Provides convenient methods for running, resuming, and monitoring
    research sessions.

    Example:
        ```python
        orchestrator = ResearchOrchestrator()
        await orchestrator.initialize()

        async for event in orchestrator.run(
            query="What are the latest advances in quantum computing?",
            config={"max_iterations": 3}
        ):
            print(f"{event.event_type}: {event.data}")
        ```
    """

    def __init__(
        self,
        db_uri: str | None = None,
        checkpointer: AsyncPostgresSaver | None = None,
    ):
        """
        Initialize the ResearchOrchestrator.

        Args:
            db_uri: PostgreSQL connection URI for checkpointing.
                   Uses DATABASE_URL from config if not provided.
            checkpointer: Optional pre-configured checkpointer.
        """
        self._db_uri = db_uri
        self._checkpointer = checkpointer
        self._workflow: CompiledStateGraph | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the orchestrator with checkpointing.

        Must be called before running workflows.
        """
        if self._initialized:
            return

        # Get or create checkpointer
        if self._checkpointer is None:
            self._checkpointer = await get_checkpointer(self._db_uri)

        # Compile workflow with checkpointer
        self._workflow = compile_workflow(checkpointer=self._checkpointer)
        self._initialized = True

        logger.info("ResearchOrchestrator initialized")

    async def run(
        self,
        query: str,
        config: dict[str, Any] | None = None,
        session_id: str | None = None,
        steerability: SteerabilityParams | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Run a research workflow and stream events.

        Args:
            query: The research question to investigate.
            config: Optional configuration overrides.
            session_id: Optional session ID (generated if not provided).
            steerability: Optional steerability parameters.

        Yields:
            StreamEvent: Events during workflow execution.

        Example:
            ```python
            async for event in orchestrator.run("What is quantum entanglement?"):
                if event.event_type == StreamEventType.SYNTHESIS_GENERATED:
                    print(event.data.get("synthesis"))
            ```
        """
        if not self._initialized:
            await self.initialize()

        # Generate session ID if not provided
        session_id = session_id or str(uuid.uuid4())

        # Merge configuration
        settings = get_settings()
        run_config = {
            "max_iterations": settings.MAX_RESEARCH_ITERATIONS,
            "token_budget": settings.TOKEN_BUDGET_PER_SESSION,
            **(config or {}),
        }

        # Create initial state
        initial_state = create_initial_state(
            session_id=session_id,
            user_query=query,
            steerability=steerability,
            token_budget=run_config.get("token_budget", 500000),
            max_iterations=run_config.get("max_iterations", 5),
        )

        # Emit workflow started event
        yield StreamEvent(
            event_type=StreamEventType.WORKFLOW_STARTED,
            session_id=session_id,
            data={
                "query": query,
                "config": run_config,
            },
        )

        try:
            # Create LangGraph config with thread_id for checkpointing
            langgraph_config: RunnableConfig = {
                "configurable": {
                    "thread_id": session_id,
                },
                "recursion_limit": (run_config.get("max_iterations", 5) + 1) * 10,
            }

            # Stream workflow execution
            current_node: str | None = None
            last_iteration = 0

            async for event in self._workflow.astream_events(
                initial_state,
                config=langgraph_config,
                version="v2",
            ):
                # Process LangGraph events and emit StreamEvents
                event_kind = event.get("event")

                if event_kind == "on_chain_start":
                    node_name = event.get("name")
                    if node_name and node_name not in ("RunnableSequence", "LangGraph"):
                        current_node = node_name
                        yield StreamEvent(
                            event_type=StreamEventType.NODE_STARTED,
                            session_id=session_id,
                            node_name=current_node,
                            iteration=last_iteration,
                        )

                elif event_kind == "on_chain_end":
                    node_name = event.get("name")
                    if node_name and node_name not in ("RunnableSequence", "LangGraph"):
                        output = event.get("data", {}).get("output", {})

                        # Emit node-specific events
                        async for node_event in self._process_node_output(
                            node_name,
                            output,
                            session_id,
                            last_iteration,
                        ):
                            yield node_event

                        # Track iteration changes
                        if "iteration_count" in output:
                            new_iteration = output["iteration_count"]
                            if new_iteration > last_iteration:
                                yield StreamEvent(
                                    event_type=StreamEventType.ITERATION_COMPLETED,
                                    session_id=session_id,
                                    iteration=last_iteration,
                                )
                                last_iteration = new_iteration
                                yield StreamEvent(
                                    event_type=StreamEventType.ITERATION_STARTED,
                                    session_id=session_id,
                                    iteration=last_iteration,
                                )

                        yield StreamEvent(
                            event_type=StreamEventType.NODE_COMPLETED,
                            session_id=session_id,
                            node_name=node_name,
                            iteration=last_iteration,
                            data=self._summarize_output(output),
                        )

                elif event_kind == "on_chain_error":
                    error_msg = str(event.get("data", {}).get("error", "Unknown error"))
                    yield StreamEvent(
                        event_type=StreamEventType.NODE_FAILED,
                        session_id=session_id,
                        node_name=current_node,
                        error=error_msg,
                    )

            # Get final state
            final_state = await self.get_state(session_id)

            yield StreamEvent(
                event_type=StreamEventType.WORKFLOW_COMPLETED,
                session_id=session_id,
                iteration=last_iteration,
                data={
                    "final_report": final_state.get("final_report") if final_state else None,
                    "tokens_used": final_state.get("tokens_used", 0) if final_state else 0,
                    "findings_count": len(final_state.get("findings", [])) if final_state else 0,
                    "citations_count": len(final_state.get("citations", [])) if final_state else 0,
                },
            )

        except asyncio.CancelledError:
            yield StreamEvent(
                event_type=StreamEventType.WORKFLOW_INTERRUPTED,
                session_id=session_id,
            )
            raise

        except Exception as e:
            logger.exception(f"Workflow failed for session {session_id}")
            yield StreamEvent(
                event_type=StreamEventType.WORKFLOW_FAILED,
                session_id=session_id,
                error=str(e),
            )
            raise

    async def run_sync(
        self,
        query: str,
        config: dict[str, Any] | None = None,
        session_id: str | None = None,
        steerability: SteerabilityParams | None = None,
    ) -> AgentState:
        """
        Run a research workflow synchronously (blocking).

        Returns the final state after workflow completion.
        Use this when you don't need streaming events.

        Args:
            query: The research question.
            config: Optional configuration.
            session_id: Optional session ID.
            steerability: Optional steerability params.

        Returns:
            AgentState: Final workflow state.
        """
        if not self._initialized:
            await self.initialize()

        session_id = session_id or str(uuid.uuid4())

        settings = get_settings()
        run_config = {
            "max_iterations": settings.MAX_RESEARCH_ITERATIONS,
            "token_budget": settings.TOKEN_BUDGET_PER_SESSION,
            **(config or {}),
        }

        initial_state = create_initial_state(
            session_id=session_id,
            user_query=query,
            steerability=steerability,
            token_budget=run_config.get("token_budget", 500000),
            max_iterations=run_config.get("max_iterations", 5),
        )

        langgraph_config: RunnableConfig = {
            "configurable": {
                "thread_id": session_id,
            },
            "recursion_limit": (run_config.get("max_iterations", 5) + 1) * 10,
        }

        # Invoke workflow (blocking)
        result = await self._workflow.ainvoke(
            initial_state,
            config=langgraph_config,
        )

        return result

    async def get_state(self, session_id: str) -> AgentState | None:
        """
        Retrieve the current state for a session.

        Args:
            session_id: The session/thread ID.

        Returns:
            AgentState if found, None otherwise.
        """
        if not self._initialized:
            await self.initialize()

        state = await get_thread_state(
            thread_id=session_id,
            checkpointer=self._checkpointer,
        )

        return state

    async def resume(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Resume a workflow from a checkpoint.

        Args:
            session_id: The session/thread ID to resume.
            checkpoint_id: Optional specific checkpoint to resume from.
                          Uses latest if not specified.
            updates: Optional state updates to apply before resuming.

        Yields:
            StreamEvent: Events during workflow execution.
        """
        if not self._initialized:
            await self.initialize()

        yield StreamEvent(
            event_type=StreamEventType.CHECKPOINT_RESUMED,
            session_id=session_id,
            data={"checkpoint_id": checkpoint_id},
        )

        try:
            # Build config for resumption
            langgraph_config: RunnableConfig = {
                "configurable": {
                    "thread_id": session_id,
                },
            }

            if checkpoint_id:
                langgraph_config["configurable"]["checkpoint_id"] = checkpoint_id

            # Get current state
            current_state = await self.get_state(session_id)
            if current_state is None:
                yield StreamEvent(
                    event_type=StreamEventType.ERROR,
                    session_id=session_id,
                    error=f"No checkpoint found for session {session_id}",
                )
                return

            # Apply updates if provided
            if updates:
                current_state = {**current_state, **updates}

            # Resume workflow
            current_node: str | None = None
            last_iteration = current_state.get("iteration_count", 0)

            async for event in self._workflow.astream_events(
                current_state,
                config=langgraph_config,
                version="v2",
            ):
                event_kind = event.get("event")

                if event_kind == "on_chain_start":
                    node_name = event.get("name")
                    if node_name and node_name not in ("RunnableSequence", "LangGraph"):
                        current_node = node_name
                        yield StreamEvent(
                            event_type=StreamEventType.NODE_STARTED,
                            session_id=session_id,
                            node_name=current_node,
                            iteration=last_iteration,
                        )

                elif event_kind == "on_chain_end":
                    node_name = event.get("name")
                    if node_name and node_name not in ("RunnableSequence", "LangGraph"):
                        output = event.get("data", {}).get("output", {})

                        if "iteration_count" in output:
                            last_iteration = output["iteration_count"]

                        yield StreamEvent(
                            event_type=StreamEventType.NODE_COMPLETED,
                            session_id=session_id,
                            node_name=node_name,
                            iteration=last_iteration,
                            data=self._summarize_output(output),
                        )

            # Final state
            final_state = await self.get_state(session_id)

            yield StreamEvent(
                event_type=StreamEventType.WORKFLOW_COMPLETED,
                session_id=session_id,
                iteration=last_iteration,
                data={
                    "final_report": final_state.get("final_report") if final_state else None,
                },
            )

        except Exception as e:
            logger.exception(f"Resume failed for session {session_id}")
            yield StreamEvent(
                event_type=StreamEventType.WORKFLOW_FAILED,
                session_id=session_id,
                error=str(e),
            )
            raise

    async def list_checkpoints(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        List available checkpoints for a session.

        Args:
            session_id: The session/thread ID.
            limit: Maximum checkpoints to return.

        Returns:
            List of checkpoint metadata.
        """
        if not self._initialized:
            await self.initialize()

        return await list_thread_checkpoints(
            thread_id=session_id,
            limit=limit,
            checkpointer=self._checkpointer,
        )

    async def cancel(self, session_id: str) -> bool:
        """
        Cancel a running workflow.

        Note: This sets a flag but doesn't immediately stop execution.
        The workflow will check for cancellation at the next node boundary.

        Args:
            session_id: The session to cancel.

        Returns:
            True if cancellation was initiated.
        """
        # In LangGraph, cancellation is typically handled by
        # raising asyncio.CancelledError in the streaming context
        # This is a placeholder for future implementation
        logger.info(f"Cancellation requested for session {session_id}")
        return True

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _process_node_output(
        self,
        node_name: str,
        output: dict[str, Any],
        session_id: str,
        iteration: int,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Process node output and emit appropriate events."""

        if node_name == "plan" and "plan" in output:
            yield StreamEvent(
                event_type=StreamEventType.PLAN_CREATED,
                session_id=session_id,
                node_name=node_name,
                iteration=iteration,
                data={
                    "task_count": len(output["plan"].get("dag_nodes", [])),
                    "sub_questions": output["plan"].get("sub_questions", []),
                },
            )

        elif node_name == "search" and "citations" in output:
            yield StreamEvent(
                event_type=StreamEventType.SOURCES_FOUND,
                session_id=session_id,
                node_name=node_name,
                iteration=iteration,
                data={
                    "citation_count": len(output.get("citations", [])),
                },
            )

        elif node_name == "read" and "findings" in output:
            yield StreamEvent(
                event_type=StreamEventType.FINDINGS_EXTRACTED,
                session_id=session_id,
                node_name=node_name,
                iteration=iteration,
                data={
                    "finding_count": len(output.get("findings", [])),
                },
            )

        elif node_name == "synthesize" and "synthesis" in output:
            yield StreamEvent(
                event_type=StreamEventType.SYNTHESIS_GENERATED,
                session_id=session_id,
                node_name=node_name,
                iteration=iteration,
                data={
                    "synthesis_length": len(output.get("synthesis", "")),
                },
            )

        elif node_name == "critique" and "gaps" in output:
            yield StreamEvent(
                event_type=StreamEventType.GAPS_IDENTIFIED,
                session_id=session_id,
                node_name=node_name,
                iteration=iteration,
                data={
                    "gap_count": len(output.get("gaps", [])),
                    "gaps": output.get("gaps", [])[:5],  # First 5 gaps
                    "quality_metrics": output.get("quality_metrics"),
                },
            )

        elif node_name == "report" and "final_report" in output:
            yield StreamEvent(
                event_type=StreamEventType.REPORT_GENERATED,
                session_id=session_id,
                node_name=node_name,
                iteration=iteration,
                data={
                    "report_length": len(output.get("final_report", "")),
                },
            )

        # Token update for any node
        if "tokens_used" in output:
            yield StreamEvent(
                event_type=StreamEventType.TOKEN_UPDATE,
                session_id=session_id,
                node_name=node_name,
                iteration=iteration,
                data={
                    "tokens_used": output.get("tokens_used", 0),
                    "tokens_remaining": output.get("tokens_remaining", 0),
                },
            )

    def _summarize_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """Create a summary of node output for events."""
        summary = {}

        # Include counts and metrics, not full data
        if "plan" in output:
            plan = output["plan"]
            summary["plan_tasks"] = len(plan.get("dag_nodes", [])) if plan else 0

        if "citations" in output:
            summary["citations_count"] = len(output["citations"])

        if "findings" in output:
            summary["findings_count"] = len(output["findings"])

        if "synthesis" in output:
            summary["synthesis_length"] = len(output["synthesis"])

        if "gaps" in output:
            summary["gaps_count"] = len(output["gaps"])

        if "quality_metrics" in output:
            metrics = output["quality_metrics"]
            if metrics:
                summary["coverage_score"] = metrics.get("coverage_score")

        if "tokens_used" in output:
            summary["tokens_used"] = output["tokens_used"]

        if "error" in output:
            summary["error"] = output["error"]

        return summary


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_research(
    query: str,
    config: dict[str, Any] | None = None,
    db_uri: str | None = None,
) -> AgentState:
    """
    Convenience function to run a single research query.

    Creates an orchestrator, runs the query, and returns the result.
    Use this for simple one-off research without event streaming.

    Args:
        query: The research question.
        config: Optional configuration.
        db_uri: Optional database URI for checkpointing.

    Returns:
        Final AgentState with research results.

    Example:
        ```python
        result = await run_research("What is machine learning?")
        print(result["final_report"])
        ```
    """
    orchestrator = ResearchOrchestrator(db_uri=db_uri)
    await orchestrator.initialize()
    return await orchestrator.run_sync(query, config)


async def stream_research(
    query: str,
    config: dict[str, Any] | None = None,
    db_uri: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Convenience function to stream research events.

    Args:
        query: The research question.
        config: Optional configuration.
        db_uri: Optional database URI.

    Yields:
        StreamEvent: Events during execution.

    Example:
        ```python
        async for event in stream_research("What is AI?"):
            print(event.event_type, event.data)
        ```
    """
    orchestrator = ResearchOrchestrator(db_uri=db_uri)
    await orchestrator.initialize()

    async for event in orchestrator.run(query, config):
        yield event


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core classes
    "ResearchOrchestrator",
    # Event types
    "StreamEvent",
    "StreamEventType",
    # Graph builders
    "create_research_workflow",
    "compile_workflow",
    # Convenience functions
    "run_research",
    "stream_research",
]
