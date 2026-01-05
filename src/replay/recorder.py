"""Event Recording for Deterministic Replay.

Provides the EventRecorder class for recording research session events
to the database. Events capture all inputs, outputs, tool calls, and
LLM invocations needed for deterministic replay.

The recording system is designed for:
- Debugging: Replay failed sessions to understand what went wrong
- Training: Generate training data from successful research runs
- Reproducibility: Repeat experiments with identical results
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from psycopg import AsyncConnection

from src.db.connection import get_async_connection

logger = logging.getLogger(__name__)


# =============================================================================
# TypedDict Definitions
# =============================================================================


class ToolCallRecord(TypedDict, total=False):
    """Record of a tool invocation for replay.

    Captures all information needed to replay or mock a tool call,
    including full inputs and outputs.

    Attributes:
        tool_call_id: Unique identifier for this tool call.
        tool_name: Name of the tool invoked.
        tool_version: Version of the tool (for compatibility checking).
        inputs: Full input parameters passed to the tool.
        outputs: Full output returned by the tool.
        status: Execution status (succeeded, failed, timeout).
        error_message: Error message if status is failed.
        latency_ms: Execution time in milliseconds.
        timestamp: ISO 8601 timestamp of the invocation.
        metadata: Additional metadata for the tool call.
    """

    tool_call_id: str
    tool_name: str
    tool_version: str | None
    inputs: dict[str, Any]
    outputs: dict[str, Any] | None
    status: str
    error_message: str | None
    latency_ms: int
    timestamp: str
    metadata: dict[str, Any]


class LLMCallRecord(TypedDict, total=False):
    """Record of an LLM invocation for replay.

    Captures all information needed to replay or mock an LLM call,
    including full prompts and responses for deterministic replay.

    Attributes:
        llm_call_id: Unique identifier for this LLM call.
        model: Model identifier (e.g., 'claude-sonnet-4-20250514').
        model_config: Model configuration (temperature, max_tokens, etc.).
        messages: Full message history sent to the LLM.
        response: Full response from the LLM.
        tool_calls: Tool calls requested by the LLM (if any).
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
        cost_usd: Cost of this call in USD.
        latency_ms: Response time in milliseconds.
        timestamp: ISO 8601 timestamp of the invocation.
        finish_reason: Reason the LLM stopped generating (stop, tool_use, etc.).
        metadata: Additional metadata for the LLM call.
    """

    llm_call_id: str
    model: str
    model_config: dict[str, Any]
    messages: list[dict[str, Any]]
    response: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    timestamp: str
    finish_reason: str | None
    metadata: dict[str, Any]


class ReplayEvent(TypedDict, total=False):
    """Event record for deterministic replay.

    A complete record of a single step in a research session,
    capturing all information needed for exact replay.

    Attributes:
        event_id: Unique identifier for this event.
        session_id: Research session this event belongs to.
        checkpoint_id: Associated checkpoint ID for resumability.
        event_type: Type of event (node_start, node_end, tool_call, etc.).
        node_name: Name of the graph node that generated this event.
        inputs: Input state to the node.
        outputs: Output state from the node.
        tool_calls: List of tool calls made during this event.
        llm_calls: List of LLM calls made during this event.
        timestamp: ISO 8601 timestamp of the event.
        deterministic_seed: Seed for reproducible random operations.
        sequence_number: Order of this event in the session.
        parent_event_id: Parent event ID for hierarchical events.
        metadata: Additional event metadata.
    """

    event_id: str
    session_id: str
    checkpoint_id: str | None
    event_type: str
    node_name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    tool_calls: list[ToolCallRecord]
    llm_calls: list[LLMCallRecord]
    timestamp: str
    deterministic_seed: int | None
    sequence_number: int
    parent_event_id: str | None
    metadata: dict[str, Any]


# =============================================================================
# EventRecorder Class
# =============================================================================


class RecorderError(Exception):
    """Base exception for recorder operations."""

    pass


class EventRecorder:
    """Records research session events for deterministic replay.

    The EventRecorder captures all events during a research session
    and persists them to the database for later replay. Events include
    node executions, tool calls, LLM invocations, and state transitions.

    Events are stored in the research_steps table with additional
    replay-specific metadata in the checkpoint_data JSONB column.

    Example:
        recorder = EventRecorder()

        # Record a node execution event
        event: ReplayEvent = {
            "event_id": str(uuid.uuid4()),
            "session_id": session_id,
            "event_type": "node_end",
            "node_name": "searcher",
            "inputs": {"query": "AI research"},
            "outputs": {"results": [...]},
            "tool_calls": [...],
            "llm_calls": [...],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await recorder.record_event(session_id, event)
    """

    def __init__(
        self,
        conn: AsyncConnection[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the EventRecorder.

        Args:
            conn: Optional database connection. If not provided, will
                  acquire connections from the pool as needed.
        """
        self._conn = conn
        self._sequence_counters: dict[str, int] = {}

    def _get_next_sequence(self, session_id: str) -> int:
        """Get the next sequence number for a session.

        Args:
            session_id: The session identifier.

        Returns:
            Next sequence number for the session.
        """
        if session_id not in self._sequence_counters:
            self._sequence_counters[session_id] = 0
        self._sequence_counters[session_id] += 1
        return self._sequence_counters[session_id]

    async def record_event(
        self,
        session_id: str,
        event: ReplayEvent,
    ) -> str:
        """Record a replay event to the database.

        Persists the event to the research_steps table with replay
        metadata stored in the checkpoint_data column.

        Args:
            session_id: The research session ID.
            event: The replay event to record.

        Returns:
            The event_id of the recorded event.

        Raises:
            RecorderError: If recording fails.
        """
        # Ensure event has required fields
        event_id = event.get("event_id") or str(uuid.uuid4())
        timestamp = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
        sequence_number = event.get("sequence_number") or self._get_next_sequence(
            session_id
        )

        # Build replay metadata
        replay_metadata: dict[str, Any] = {
            "replay_version": "1.0.0",
            "event_id": event_id,
            "event_type": event.get("event_type", "unknown"),
            "node_name": event.get("node_name", ""),
            "inputs": event.get("inputs", {}),
            "outputs": event.get("outputs", {}),
            "tool_calls": event.get("tool_calls", []),
            "llm_calls": event.get("llm_calls", []),
            "deterministic_seed": event.get("deterministic_seed"),
            "sequence_number": sequence_number,
            "parent_event_id": event.get("parent_event_id"),
            "metadata": event.get("metadata", {}),
        }

        # Map event_type to step_type for database schema
        step_type_mapping = {
            "node_start": "reasoning",
            "node_end": "reasoning",
            "tool_call": "search_execution",
            "llm_call": "reasoning",
            "checkpoint": "synthesis",
            "error": "quality_check",
        }
        step_type = step_type_mapping.get(
            event.get("event_type", ""), "reasoning"
        )

        # Map node_name to agent_type
        agent_type_mapping = {
            "planner": "planner",
            "searcher": "searcher",
            "reader": "reader",
            "reasoner": "reasoner",
            "writer": "writer",
            "critic": "critic",
            "synthesizer": "synthesizer",
            "orchestrator": "orchestrator",
        }
        node_name = event.get("node_name", "")
        agent_type = agent_type_mapping.get(
            node_name.lower().split("_")[0], "orchestrator"
        )

        try:
            async with self._get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO research_steps (
                        id, session_id, step_type, agent_type, step_name,
                        status, inputs, outputs, checkpoint_data,
                        created_at, started_at, completed_at, metadata
                    ) VALUES (
                        %(id)s, %(session_id)s, %(step_type)s, %(agent_type)s,
                        %(step_name)s, %(status)s, %(inputs)s, %(outputs)s,
                        %(checkpoint_data)s, %(created_at)s, %(started_at)s,
                        %(completed_at)s, %(metadata)s
                    )
                    """,
                    {
                        "id": event_id,
                        "session_id": session_id,
                        "step_type": step_type,
                        "agent_type": agent_type,
                        "step_name": event.get("node_name", "replay_event"),
                        "status": "succeeded",
                        "inputs": json.dumps(event.get("inputs", {})),
                        "outputs": json.dumps(event.get("outputs", {})),
                        "checkpoint_data": json.dumps(replay_metadata),
                        "created_at": timestamp,
                        "started_at": timestamp,
                        "completed_at": timestamp,
                        "metadata": json.dumps({
                            "replay": True,
                            "checkpoint_id": event.get("checkpoint_id"),
                            "event_metadata": event.get("metadata", {}),
                        }),
                    },
                )

            logger.debug(
                f"Recorded replay event {event_id} for session {session_id}",
                extra={
                    "event_id": event_id,
                    "session_id": session_id,
                    "event_type": event.get("event_type"),
                },
            )

            return event_id

        except Exception as e:
            logger.error(f"Failed to record event: {e}")
            raise RecorderError(f"Failed to record event: {e}") from e

    async def record_events_batch(
        self,
        session_id: str,
        events: list[ReplayEvent],
    ) -> list[str]:
        """Record multiple events in a single transaction.

        More efficient than recording events one at a time.

        Args:
            session_id: The research session ID.
            events: List of replay events to record.

        Returns:
            List of event_ids for the recorded events.

        Raises:
            RecorderError: If recording fails.
        """
        event_ids: list[str] = []

        try:
            async with self._get_connection() as conn:
                # Use a transaction for batch insert
                await conn.set_autocommit(False)
                try:
                    for event in events:
                        event_id = await self.record_event(session_id, event)
                        event_ids.append(event_id)
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
                finally:
                    await conn.set_autocommit(True)

            return event_ids

        except Exception as e:
            logger.error(f"Failed to record event batch: {e}")
            raise RecorderError(f"Failed to record event batch: {e}") from e

    async def get_events(
        self,
        session_id: str,
        from_checkpoint: str | None = None,
        limit: int | None = None,
        event_types: list[str] | None = None,
    ) -> list[ReplayEvent]:
        """Retrieve recorded events for a session.

        Fetches events from the database, optionally starting from a
        specific checkpoint and filtering by event type.

        Args:
            session_id: The research session ID.
            from_checkpoint: Optional checkpoint ID to start from.
            limit: Maximum number of events to return.
            event_types: Optional list of event types to filter by.

        Returns:
            List of ReplayEvent dictionaries ordered by sequence.

        Raises:
            RecorderError: If retrieval fails.
        """
        try:
            async with self._get_connection() as conn:
                # Build query
                query = """
                    SELECT id, session_id, step_name, inputs, outputs,
                           checkpoint_data, created_at, metadata
                    FROM research_steps
                    WHERE session_id = %(session_id)s
                """
                params: dict[str, Any] = {"session_id": session_id}

                # Filter by checkpoint if specified
                if from_checkpoint:
                    query += """
                        AND (checkpoint_data->>'sequence_number')::int >= (
                            SELECT (checkpoint_data->>'sequence_number')::int
                            FROM research_steps
                            WHERE session_id = %(session_id)s
                            AND checkpoint_data->>'event_id' = %(checkpoint_id)s
                            LIMIT 1
                        )
                    """
                    params["checkpoint_id"] = from_checkpoint

                # Filter by event types if specified
                if event_types:
                    query += """
                        AND checkpoint_data->>'event_type' = ANY(%(event_types)s)
                    """
                    params["event_types"] = event_types

                # Order by sequence
                query += """
                    ORDER BY (checkpoint_data->>'sequence_number')::int ASC
                """

                # Apply limit
                if limit:
                    query += " LIMIT %(limit)s"
                    params["limit"] = limit

                result = await conn.execute(query, params)
                rows = await result.fetchall()

                events: list[ReplayEvent] = []
                for row in rows:
                    checkpoint_data = row.get("checkpoint_data", {})
                    if isinstance(checkpoint_data, str):
                        checkpoint_data = json.loads(checkpoint_data)

                    metadata = row.get("metadata", {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    inputs = row.get("inputs", {})
                    if isinstance(inputs, str):
                        inputs = json.loads(inputs)

                    outputs = row.get("outputs", {})
                    if isinstance(outputs, str):
                        outputs = json.loads(outputs)

                    event: ReplayEvent = {
                        "event_id": checkpoint_data.get("event_id", row["id"]),
                        "session_id": row["session_id"],
                        "checkpoint_id": metadata.get("checkpoint_id"),
                        "event_type": checkpoint_data.get("event_type", "unknown"),
                        "node_name": checkpoint_data.get(
                            "node_name", row.get("step_name", "")
                        ),
                        "inputs": checkpoint_data.get("inputs", inputs),
                        "outputs": checkpoint_data.get("outputs", outputs),
                        "tool_calls": checkpoint_data.get("tool_calls", []),
                        "llm_calls": checkpoint_data.get("llm_calls", []),
                        "timestamp": (
                            row["created_at"].isoformat()
                            if hasattr(row["created_at"], "isoformat")
                            else str(row["created_at"])
                        ),
                        "deterministic_seed": checkpoint_data.get("deterministic_seed"),
                        "sequence_number": checkpoint_data.get("sequence_number", 0),
                        "parent_event_id": checkpoint_data.get("parent_event_id"),
                        "metadata": checkpoint_data.get("metadata", {}),
                    }
                    events.append(event)

                logger.debug(
                    f"Retrieved {len(events)} events for session {session_id}",
                    extra={"session_id": session_id, "from_checkpoint": from_checkpoint},
                )

                return events

        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            raise RecorderError(f"Failed to retrieve events: {e}") from e

    async def get_event_by_id(
        self,
        event_id: str,
    ) -> ReplayEvent | None:
        """Retrieve a specific event by its ID.

        Args:
            event_id: The event identifier.

        Returns:
            ReplayEvent if found, None otherwise.

        Raises:
            RecorderError: If retrieval fails.
        """
        try:
            async with self._get_connection() as conn:
                result = await conn.execute(
                    """
                    SELECT id, session_id, step_name, inputs, outputs,
                           checkpoint_data, created_at, metadata
                    FROM research_steps
                    WHERE id = %(event_id)s
                       OR checkpoint_data->>'event_id' = %(event_id)s
                    LIMIT 1
                    """,
                    {"event_id": event_id},
                )
                row = await result.fetchone()

                if not row:
                    return None

                checkpoint_data = row.get("checkpoint_data", {})
                if isinstance(checkpoint_data, str):
                    checkpoint_data = json.loads(checkpoint_data)

                metadata = row.get("metadata", {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                inputs = row.get("inputs", {})
                if isinstance(inputs, str):
                    inputs = json.loads(inputs)

                outputs = row.get("outputs", {})
                if isinstance(outputs, str):
                    outputs = json.loads(outputs)

                event: ReplayEvent = {
                    "event_id": checkpoint_data.get("event_id", row["id"]),
                    "session_id": row["session_id"],
                    "checkpoint_id": metadata.get("checkpoint_id"),
                    "event_type": checkpoint_data.get("event_type", "unknown"),
                    "node_name": checkpoint_data.get(
                        "node_name", row.get("step_name", "")
                    ),
                    "inputs": checkpoint_data.get("inputs", inputs),
                    "outputs": checkpoint_data.get("outputs", outputs),
                    "tool_calls": checkpoint_data.get("tool_calls", []),
                    "llm_calls": checkpoint_data.get("llm_calls", []),
                    "timestamp": (
                        row["created_at"].isoformat()
                        if hasattr(row["created_at"], "isoformat")
                        else str(row["created_at"])
                    ),
                    "deterministic_seed": checkpoint_data.get("deterministic_seed"),
                    "sequence_number": checkpoint_data.get("sequence_number", 0),
                    "parent_event_id": checkpoint_data.get("parent_event_id"),
                    "metadata": checkpoint_data.get("metadata", {}),
                }

                return event

        except Exception as e:
            logger.error(f"Failed to retrieve event {event_id}: {e}")
            raise RecorderError(f"Failed to retrieve event: {e}") from e

    async def delete_events(
        self,
        session_id: str,
        before_checkpoint: str | None = None,
    ) -> int:
        """Delete recorded events for a session.

        Useful for cleaning up old replay data or resetting a session.

        Args:
            session_id: The research session ID.
            before_checkpoint: If specified, only delete events before this checkpoint.

        Returns:
            Number of events deleted.

        Raises:
            RecorderError: If deletion fails.
        """
        try:
            async with self._get_connection() as conn:
                if before_checkpoint:
                    # Delete events before the specified checkpoint
                    result = await conn.execute(
                        """
                        DELETE FROM research_steps
                        WHERE session_id = %(session_id)s
                        AND (checkpoint_data->>'sequence_number')::int < (
                            SELECT (checkpoint_data->>'sequence_number')::int
                            FROM research_steps
                            WHERE session_id = %(session_id)s
                            AND checkpoint_data->>'event_id' = %(checkpoint_id)s
                            LIMIT 1
                        )
                        AND metadata->>'replay' = 'true'
                        """,
                        {
                            "session_id": session_id,
                            "checkpoint_id": before_checkpoint,
                        },
                    )
                else:
                    # Delete all replay events for the session
                    result = await conn.execute(
                        """
                        DELETE FROM research_steps
                        WHERE session_id = %(session_id)s
                        AND metadata->>'replay' = 'true'
                        """,
                        {"session_id": session_id},
                    )

                deleted_count = result.rowcount or 0

                logger.info(
                    f"Deleted {deleted_count} events for session {session_id}",
                    extra={
                        "session_id": session_id,
                        "before_checkpoint": before_checkpoint,
                    },
                )

                return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete events: {e}")
            raise RecorderError(f"Failed to delete events: {e}") from e

    async def get_session_summary(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Get a summary of recorded events for a session.

        Provides aggregated statistics about the recorded events.

        Args:
            session_id: The research session ID.

        Returns:
            Dictionary with event statistics.

        Raises:
            RecorderError: If retrieval fails.
        """
        try:
            async with self._get_connection() as conn:
                result = await conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_events,
                        MIN(created_at) as first_event_at,
                        MAX(created_at) as last_event_at,
                        COUNT(DISTINCT checkpoint_data->>'node_name') as unique_nodes,
                        jsonb_agg(DISTINCT checkpoint_data->>'event_type') as event_types
                    FROM research_steps
                    WHERE session_id = %(session_id)s
                    AND metadata->>'replay' = 'true'
                    """,
                    {"session_id": session_id},
                )
                row = await result.fetchone()

                if not row or row["total_events"] == 0:
                    return {
                        "session_id": session_id,
                        "total_events": 0,
                        "first_event_at": None,
                        "last_event_at": None,
                        "unique_nodes": 0,
                        "event_types": [],
                        "duration_ms": 0,
                    }

                first_at = row["first_event_at"]
                last_at = row["last_event_at"]

                if hasattr(first_at, "timestamp") and hasattr(last_at, "timestamp"):
                    duration_ms = int((last_at.timestamp() - first_at.timestamp()) * 1000)
                else:
                    duration_ms = 0

                return {
                    "session_id": session_id,
                    "total_events": row["total_events"],
                    "first_event_at": (
                        first_at.isoformat() if hasattr(first_at, "isoformat") else str(first_at)
                    ),
                    "last_event_at": (
                        last_at.isoformat() if hasattr(last_at, "isoformat") else str(last_at)
                    ),
                    "unique_nodes": row["unique_nodes"],
                    "event_types": row["event_types"] or [],
                    "duration_ms": duration_ms,
                }

        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            raise RecorderError(f"Failed to get session summary: {e}") from e

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[AsyncConnection[dict[str, Any]]]:
        """Get a database connection.

        Returns the injected connection if available, otherwise
        acquires one from the pool.

        Yields:
            AsyncConnection: Database connection for executing queries.
        """
        if self._conn is not None:
            # Use the injected connection directly (no cleanup needed)
            yield self._conn
        else:
            # Acquire from pool and ensure proper cleanup
            async with get_async_connection() as conn:
                yield conn


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "EventRecorder",
    "RecorderError",
    "ReplayEvent",
    "ToolCallRecord",
    "LLMCallRecord",
]
