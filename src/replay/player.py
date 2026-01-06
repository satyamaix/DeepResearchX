"""Replay Player for Deterministic Session Replay.

Provides the ReplayPlayer class for replaying recorded research sessions
with support for modifications, comparison, and debugging.

The replay system supports:
- Exact replay: Reproduce sessions identically using recorded data
- Modified replay: Change inputs/parameters and observe different outcomes
- Comparison: Diff original vs replay runs for debugging
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.orchestrator.checkpointer import get_checkpointer
from src.replay.recorder import EventRecorder, ReplayEvent

logger = logging.getLogger(__name__)


# =============================================================================
# TypedDict Definitions
# =============================================================================


class FieldDiff(TypedDict, total=False):
    """Represents a difference in a single field.

    Attributes:
        field_path: JSON path to the differing field (e.g., "outputs.results[0].score").
        original_value: Value in the original run.
        replay_value: Value in the replay run.
        diff_type: Type of difference (added, removed, changed, type_change).
        is_significant: Whether this diff is likely to affect outcomes.
    """

    field_path: str
    original_value: Any
    replay_value: Any
    diff_type: str
    is_significant: bool


class EventDiff(TypedDict, total=False):
    """Differences for a single event between original and replay.

    Attributes:
        event_id: Original event ID.
        replay_event_id: Replay event ID (may differ).
        node_name: Name of the node.
        event_type: Type of event.
        input_diffs: Field-level diffs in inputs.
        output_diffs: Field-level diffs in outputs.
        tool_call_diffs: Diffs in tool calls.
        llm_call_diffs: Diffs in LLM calls.
        is_match: Whether events are considered equivalent.
        match_score: Similarity score (0.0 to 1.0).
    """

    event_id: str
    replay_event_id: str | None
    node_name: str
    event_type: str
    input_diffs: list[FieldDiff]
    output_diffs: list[FieldDiff]
    tool_call_diffs: list[FieldDiff]
    llm_call_diffs: list[FieldDiff]
    is_match: bool
    match_score: float


class DiffReport(TypedDict, total=False):
    """Complete comparison report between original and replay sessions.

    Provides comprehensive diff information for debugging and
    analysis of replay accuracy.

    Attributes:
        original_session_id: Original session identifier.
        replay_session_id: Replay session identifier.
        original_checkpoint_id: Starting checkpoint for original.
        replay_checkpoint_id: Starting checkpoint for replay.
        created_at: Timestamp when comparison was made.
        total_events_original: Number of events in original.
        total_events_replay: Number of events in replay.
        matched_events: Number of events that match.
        mismatched_events: Number of events that differ.
        missing_events: Number of events missing in replay.
        extra_events: Number of extra events in replay.
        event_diffs: List of per-event differences.
        summary: High-level summary of differences.
        is_deterministic: Whether replay was fully deterministic.
        determinism_score: Overall score for replay accuracy (0.0 to 1.0).
        metadata: Additional comparison metadata.
    """

    original_session_id: str
    replay_session_id: str
    original_checkpoint_id: str | None
    replay_checkpoint_id: str | None
    created_at: str
    total_events_original: int
    total_events_replay: int
    matched_events: int
    mismatched_events: int
    missing_events: int
    extra_events: int
    event_diffs: list[EventDiff]
    summary: str
    is_deterministic: bool
    determinism_score: float
    metadata: dict[str, Any]


class ReplayConfig(TypedDict, total=False):
    """Configuration for replay execution.

    Attributes:
        deterministic_seed: Seed for reproducible random operations.
        temperature: LLM temperature (0 for deterministic).
        mock_tool_calls: Whether to mock tool calls with recorded data.
        mock_llm_calls: Whether to mock LLM calls with recorded data.
        timeout_seconds: Maximum replay duration.
        stop_on_divergence: Whether to stop if replay diverges.
        divergence_threshold: Threshold for divergence detection.
    """

    deterministic_seed: int | None
    temperature: float
    mock_tool_calls: bool
    mock_llm_calls: bool
    timeout_seconds: int
    stop_on_divergence: bool
    divergence_threshold: float


# =============================================================================
# ReplayPlayer Class
# =============================================================================


class ReplayError(Exception):
    """Base exception for replay operations."""

    pass


class ReplayDivergenceError(ReplayError):
    """Exception raised when replay diverges from original."""

    def __init__(self, message: str, diff: EventDiff | None = None) -> None:
        super().__init__(message)
        self.diff = diff


class ReplayPlayer:
    """Replays recorded research sessions for debugging and analysis.

    The ReplayPlayer can:
    - Replay sessions from any checkpoint
    - Inject modifications to test different scenarios
    - Compare original vs replay for debugging
    - Generate deterministic outputs for training data

    Example:
        player = ReplayPlayer()

        # Exact replay from checkpoint
        async for event in player.replay_from_checkpoint(session_id, checkpoint_id):
            print(f"Replayed: {event['node_name']}")

        # Replay with modifications
        modifications = {"inputs": {"query": "modified query"}}
        async for event in player.replay_with_modifications(session_id, modifications):
            print(f"Modified replay: {event['node_name']}")

        # Compare original vs replay
        diff = await player.compare_runs(original_id, replay_id)
        print(f"Determinism score: {diff['determinism_score']}")
    """

    DEFAULT_CONFIG: ReplayConfig = {
        "deterministic_seed": 42,
        "temperature": 0.0,
        "mock_tool_calls": True,
        "mock_llm_calls": True,
        "timeout_seconds": 600,
        "stop_on_divergence": False,
        "divergence_threshold": 0.8,
    }

    def __init__(
        self,
        recorder: EventRecorder | None = None,
        checkpointer: AsyncPostgresSaver | None = None,
        conn: AsyncConnection[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the ReplayPlayer.

        Args:
            recorder: Event recorder for reading/writing events.
            checkpointer: LangGraph checkpointer for state access.
            conn: Optional database connection.
        """
        self._recorder = recorder or EventRecorder(conn)
        self._checkpointer = checkpointer
        self._conn = conn
        self._active_replays: dict[str, asyncio.Task[None]] = {}

    async def _ensure_checkpointer(self) -> AsyncPostgresSaver:
        """Ensure checkpointer is initialized.

        Returns:
            Initialized checkpointer instance.
        """
        if self._checkpointer is None:
            self._checkpointer = await get_checkpointer()
        return self._checkpointer

    async def replay_from_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        config: ReplayConfig | None = None,
    ) -> AsyncGenerator[ReplayEvent, None]:
        """Replay a session from a specific checkpoint.

        Retrieves recorded events from the checkpoint forward and
        yields them for processing. Tool and LLM calls can be mocked
        using recorded data for deterministic replay.

        Args:
            session_id: The session to replay.
            checkpoint_id: Checkpoint to start from.
            config: Optional replay configuration.

        Yields:
            ReplayEvent dictionaries in sequence.

        Raises:
            ReplayError: If replay fails.
        """
        replay_config = {**self.DEFAULT_CONFIG, **(config or {})}
        replay_session_id = str(uuid.uuid4())

        logger.info(
            f"Starting replay of session {session_id} from checkpoint {checkpoint_id}",
            extra={
                "session_id": session_id,
                "checkpoint_id": checkpoint_id,
                "replay_session_id": replay_session_id,
            },
        )

        try:
            # Get events from checkpoint
            events = await self._recorder.get_events(
                session_id=session_id,
                from_checkpoint=checkpoint_id,
            )

            if not events:
                logger.warning(f"No events found for session {session_id}")
                return

            # Yield replay start event
            start_event: ReplayEvent = {
                "event_id": str(uuid.uuid4()),
                "session_id": replay_session_id,
                "checkpoint_id": checkpoint_id,
                "event_type": "replay_start",
                "node_name": "replay_controller",
                "inputs": {
                    "original_session_id": session_id,
                    "checkpoint_id": checkpoint_id,
                    "config": replay_config,
                },
                "outputs": {},
                "tool_calls": [],
                "llm_calls": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "deterministic_seed": replay_config.get("deterministic_seed"),
                "sequence_number": 0,
                "metadata": {"replay": True, "original_session_id": session_id},
            }
            yield start_event

            # Replay each event
            for idx, event in enumerate(events):
                replayed_event = await self._replay_event(
                    event=event,
                    replay_session_id=replay_session_id,
                    sequence_number=idx + 1,
                    config=replay_config,
                )

                # Record the replayed event
                await self._recorder.record_event(replay_session_id, replayed_event)

                yield replayed_event

            # Yield replay complete event
            complete_event: ReplayEvent = {
                "event_id": str(uuid.uuid4()),
                "session_id": replay_session_id,
                "checkpoint_id": checkpoint_id,
                "event_type": "replay_complete",
                "node_name": "replay_controller",
                "inputs": {},
                "outputs": {
                    "total_events": len(events),
                    "original_session_id": session_id,
                },
                "tool_calls": [],
                "llm_calls": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "deterministic_seed": replay_config.get("deterministic_seed"),
                "sequence_number": len(events) + 1,
                "metadata": {"replay": True, "original_session_id": session_id},
            }
            yield complete_event

            logger.info(
                f"Completed replay of session {session_id}, "
                f"replayed {len(events)} events",
                extra={
                    "session_id": session_id,
                    "replay_session_id": replay_session_id,
                    "event_count": len(events),
                },
            )

        except Exception as e:
            logger.error(f"Replay failed: {e}", exc_info=True)
            raise ReplayError(f"Replay failed: {e}") from e

    async def replay_with_modifications(
        self,
        session_id: str,
        modifications: dict[str, Any],
        checkpoint_id: str | None = None,
        config: ReplayConfig | None = None,
    ) -> AsyncGenerator[ReplayEvent, None]:
        """Replay a session with modifications to inputs or parameters.

        Allows testing "what if" scenarios by changing inputs, tool
        responses, or LLM prompts during replay.

        Args:
            session_id: The session to replay.
            modifications: Dict of modifications to apply. Keys can be:
                - "inputs": Override input values
                - "tool_overrides": Map of tool_name -> response to inject
                - "llm_overrides": Map of node_name -> response to inject
                - "config_overrides": Override replay config
            checkpoint_id: Optional checkpoint to start from (default: beginning).
            config: Optional replay configuration.

        Yields:
            ReplayEvent dictionaries with modifications applied.

        Raises:
            ReplayError: If replay fails.
        """
        replay_config = {**self.DEFAULT_CONFIG, **(config or {})}
        # Apply config overrides from modifications
        if "config_overrides" in modifications:
            replay_config.update(modifications["config_overrides"])

        replay_session_id = f"modified_replay_{uuid.uuid4().hex[:12]}"

        logger.info(
            f"Starting modified replay of session {session_id}",
            extra={
                "session_id": session_id,
                "checkpoint_id": checkpoint_id,
                "modifications": list(modifications.keys()),
            },
        )

        try:
            # Get events from checkpoint (or beginning)
            events = await self._recorder.get_events(
                session_id=session_id,
                from_checkpoint=checkpoint_id,
            )

            if not events:
                logger.warning(f"No events found for session {session_id}")
                return

            # Yield modified replay start event
            start_event: ReplayEvent = {
                "event_id": str(uuid.uuid4()),
                "session_id": replay_session_id,
                "checkpoint_id": checkpoint_id,
                "event_type": "modified_replay_start",
                "node_name": "replay_controller",
                "inputs": {
                    "original_session_id": session_id,
                    "checkpoint_id": checkpoint_id,
                    "modifications": modifications,
                    "config": replay_config,
                },
                "outputs": {},
                "tool_calls": [],
                "llm_calls": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "deterministic_seed": replay_config.get("deterministic_seed"),
                "sequence_number": 0,
                "metadata": {
                    "replay": True,
                    "modified": True,
                    "original_session_id": session_id,
                },
            }
            yield start_event

            # Replay each event with modifications
            for idx, event in enumerate(events):
                # Apply input modifications
                modified_event = self._apply_modifications(event, modifications)

                # Replay the modified event
                replayed_event = await self._replay_event(
                    event=modified_event,
                    replay_session_id=replay_session_id,
                    sequence_number=idx + 1,
                    config=replay_config,
                    tool_overrides=modifications.get("tool_overrides", {}),
                    llm_overrides=modifications.get("llm_overrides", {}),
                )

                # Record the replayed event
                await self._recorder.record_event(replay_session_id, replayed_event)

                yield replayed_event

            # Yield modified replay complete event
            complete_event: ReplayEvent = {
                "event_id": str(uuid.uuid4()),
                "session_id": replay_session_id,
                "checkpoint_id": checkpoint_id,
                "event_type": "modified_replay_complete",
                "node_name": "replay_controller",
                "inputs": {},
                "outputs": {
                    "total_events": len(events),
                    "original_session_id": session_id,
                    "modifications_applied": list(modifications.keys()),
                },
                "tool_calls": [],
                "llm_calls": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "deterministic_seed": replay_config.get("deterministic_seed"),
                "sequence_number": len(events) + 1,
                "metadata": {
                    "replay": True,
                    "modified": True,
                    "original_session_id": session_id,
                },
            }
            yield complete_event

        except Exception as e:
            logger.error(f"Modified replay failed: {e}", exc_info=True)
            raise ReplayError(f"Modified replay failed: {e}") from e

    async def compare_runs(
        self,
        original_session: str,
        replay_session: str,
        include_field_diffs: bool = True,
    ) -> DiffReport:
        """Compare original session with a replay session.

        Performs field-level comparison between original and replay
        events to identify divergences and measure determinism.

        Args:
            original_session: Original session ID.
            replay_session: Replay session ID.
            include_field_diffs: Whether to include detailed field diffs.

        Returns:
            DiffReport with comprehensive comparison results.

        Raises:
            ReplayError: If comparison fails.
        """
        logger.info(
            f"Comparing sessions: {original_session} vs {replay_session}",
            extra={
                "original_session": original_session,
                "replay_session": replay_session,
            },
        )

        try:
            # Get events from both sessions
            original_events = await self._recorder.get_events(original_session)
            replay_events = await self._recorder.get_events(replay_session)

            # Filter out replay control events from replay session
            replay_events = [
                e
                for e in replay_events
                if e.get("event_type") not in (
                    "replay_start",
                    "replay_complete",
                    "modified_replay_start",
                    "modified_replay_complete",
                )
            ]

            # Build event diff list
            event_diffs: list[EventDiff] = []
            matched = 0
            mismatched = 0

            # Compare events by sequence
            max_len = max(len(original_events), len(replay_events))

            for i in range(max_len):
                orig = original_events[i] if i < len(original_events) else None
                repl = replay_events[i] if i < len(replay_events) else None

                if orig is None:
                    # Extra event in replay
                    event_diffs.append(
                        self._create_missing_event_diff(
                            repl, is_extra=True  # type: ignore
                        )
                    )
                    mismatched += 1
                elif repl is None:
                    # Missing event in replay
                    event_diffs.append(
                        self._create_missing_event_diff(orig, is_extra=False)
                    )
                    mismatched += 1
                else:
                    # Compare events
                    diff = self._compare_events(
                        orig, repl, include_field_diffs=include_field_diffs
                    )
                    event_diffs.append(diff)
                    if diff["is_match"]:
                        matched += 1
                    else:
                        mismatched += 1

            # Calculate determinism score
            total_events = max(len(original_events), len(replay_events))
            determinism_score = matched / total_events if total_events > 0 else 1.0
            is_deterministic = determinism_score >= 0.99

            # Generate summary
            summary = self._generate_diff_summary(
                matched=matched,
                mismatched=mismatched,
                missing=len(original_events) - len(replay_events)
                if len(original_events) > len(replay_events)
                else 0,
                extra=len(replay_events) - len(original_events)
                if len(replay_events) > len(original_events)
                else 0,
                determinism_score=determinism_score,
            )

            report: DiffReport = {
                "original_session_id": original_session,
                "replay_session_id": replay_session,
                "original_checkpoint_id": None,
                "replay_checkpoint_id": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_events_original": len(original_events),
                "total_events_replay": len(replay_events),
                "matched_events": matched,
                "mismatched_events": mismatched,
                "missing_events": max(0, len(original_events) - len(replay_events)),
                "extra_events": max(0, len(replay_events) - len(original_events)),
                "event_diffs": event_diffs,
                "summary": summary,
                "is_deterministic": is_deterministic,
                "determinism_score": determinism_score,
                "metadata": {
                    "comparison_version": "1.0.0",
                    "include_field_diffs": include_field_diffs,
                },
            }

            logger.info(
                f"Comparison complete: {determinism_score:.2%} deterministic",
                extra={
                    "original_session": original_session,
                    "replay_session": replay_session,
                    "determinism_score": determinism_score,
                    "matched": matched,
                    "mismatched": mismatched,
                },
            )

            return report

        except Exception as e:
            logger.error(f"Comparison failed: {e}", exc_info=True)
            raise ReplayError(f"Comparison failed: {e}") from e

    async def get_divergence_point(
        self,
        original_session: str,
        replay_session: str,
    ) -> ReplayEvent | None:
        """Find the first point where replay diverges from original.

        Useful for debugging determinism issues.

        Args:
            original_session: Original session ID.
            replay_session: Replay session ID.

        Returns:
            The original event where divergence first occurred, or None if identical.
        """
        try:
            diff = await self.compare_runs(
                original_session,
                replay_session,
                include_field_diffs=False,
            )

            for event_diff in diff["event_diffs"]:
                if not event_diff["is_match"]:
                    # Found divergence - return original event
                    return await self._recorder.get_event_by_id(event_diff["event_id"])

            return None

        except Exception as e:
            logger.error(f"Failed to find divergence point: {e}")
            raise ReplayError(f"Failed to find divergence point: {e}") from e

    async def cancel_replay(self, replay_session_id: str) -> bool:
        """Cancel an active replay.

        Args:
            replay_session_id: ID of the replay session to cancel.

        Returns:
            True if cancelled, False if not found.
        """
        task = self._active_replays.get(replay_session_id)
        if task and not task.done():
            task.cancel()
            del self._active_replays[replay_session_id]
            logger.info(f"Cancelled replay: {replay_session_id}")
            return True
        return False

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _replay_event(
        self,
        event: ReplayEvent,
        replay_session_id: str,
        sequence_number: int,
        config: ReplayConfig,
        tool_overrides: dict[str, Any] | None = None,
        llm_overrides: dict[str, Any] | None = None,
    ) -> ReplayEvent:
        """Replay a single event.

        Args:
            event: The original event to replay.
            replay_session_id: ID of the replay session.
            sequence_number: Sequence number in replay.
            config: Replay configuration.
            tool_overrides: Optional tool response overrides.
            llm_overrides: Optional LLM response overrides.

        Returns:
            The replayed event.
        """
        # Create replayed event with new IDs
        replayed: ReplayEvent = {
            "event_id": str(uuid.uuid4()),
            "session_id": replay_session_id,
            "checkpoint_id": event.get("checkpoint_id"),
            "event_type": event.get("event_type", "unknown"),
            "node_name": event.get("node_name", ""),
            "inputs": event.get("inputs", {}).copy(),
            "outputs": event.get("outputs", {}).copy(),
            "tool_calls": [],
            "llm_calls": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "deterministic_seed": config.get("deterministic_seed"),
            "sequence_number": sequence_number,
            "parent_event_id": event.get("parent_event_id"),
            "metadata": {
                **event.get("metadata", {}),
                "replay": True,
                "original_event_id": event.get("event_id"),
            },
        }

        # Mock or replay tool calls
        if config.get("mock_tool_calls", True):
            replayed["tool_calls"] = event.get("tool_calls", [])
        else:
            # Apply tool overrides if any
            tool_calls = event.get("tool_calls", [])
            for call in tool_calls:
                tool_name = call.get("tool_name", "")
                if tool_overrides and tool_name in tool_overrides:
                    call["outputs"] = tool_overrides[tool_name]
            replayed["tool_calls"] = tool_calls

        # Mock or replay LLM calls
        if config.get("mock_llm_calls", True):
            replayed["llm_calls"] = event.get("llm_calls", [])
        else:
            # Apply LLM overrides if any
            llm_calls = event.get("llm_calls", [])
            node_name = event.get("node_name", "")
            if llm_overrides and node_name in llm_overrides:
                for call in llm_calls:
                    call["response"] = llm_overrides[node_name]
            replayed["llm_calls"] = llm_calls

        return replayed

    def _apply_modifications(
        self,
        event: ReplayEvent,
        modifications: dict[str, Any],
    ) -> ReplayEvent:
        """Apply modifications to an event before replay.

        Args:
            event: Original event.
            modifications: Modifications to apply.

        Returns:
            Modified event copy.
        """
        modified = dict(event)

        # Apply input modifications
        if "inputs" in modifications:
            original_inputs = modified.get("inputs", {})
            if isinstance(original_inputs, dict):
                modified["inputs"] = {**original_inputs, **modifications["inputs"]}

        return modified  # type: ignore

    def _compare_events(
        self,
        original: ReplayEvent,
        replay: ReplayEvent,
        include_field_diffs: bool = True,
    ) -> EventDiff:
        """Compare two events and generate diff.

        Args:
            original: Original event.
            replay: Replay event.
            include_field_diffs: Whether to include field-level diffs.

        Returns:
            EventDiff with comparison results.
        """
        input_diffs: list[FieldDiff] = []
        output_diffs: list[FieldDiff] = []
        tool_call_diffs: list[FieldDiff] = []
        llm_call_diffs: list[FieldDiff] = []

        if include_field_diffs:
            # Compare inputs
            input_diffs = self._diff_dicts(
                original.get("inputs", {}),
                replay.get("inputs", {}),
                "inputs",
            )

            # Compare outputs
            output_diffs = self._diff_dicts(
                original.get("outputs", {}),
                replay.get("outputs", {}),
                "outputs",
            )

            # Compare tool calls (simplified)
            if original.get("tool_calls") != replay.get("tool_calls"):
                tool_call_diffs.append({
                    "field_path": "tool_calls",
                    "original_value": len(original.get("tool_calls", [])),
                    "replay_value": len(replay.get("tool_calls", [])),
                    "diff_type": "changed",
                    "is_significant": True,
                })

            # Compare LLM calls (simplified)
            if original.get("llm_calls") != replay.get("llm_calls"):
                llm_call_diffs.append({
                    "field_path": "llm_calls",
                    "original_value": len(original.get("llm_calls", [])),
                    "replay_value": len(replay.get("llm_calls", [])),
                    "diff_type": "changed",
                    "is_significant": True,
                })

        # Calculate match score
        total_diffs = (
            len(input_diffs)
            + len(output_diffs)
            + len(tool_call_diffs)
            + len(llm_call_diffs)
        )
        match_score = 1.0 / (1 + total_diffs * 0.1)  # Decay with diffs
        is_match = total_diffs == 0

        return {
            "event_id": original.get("event_id", ""),
            "replay_event_id": replay.get("event_id"),
            "node_name": original.get("node_name", ""),
            "event_type": original.get("event_type", ""),
            "input_diffs": input_diffs,
            "output_diffs": output_diffs,
            "tool_call_diffs": tool_call_diffs,
            "llm_call_diffs": llm_call_diffs,
            "is_match": is_match,
            "match_score": match_score,
        }

    def _create_missing_event_diff(
        self,
        event: ReplayEvent,
        is_extra: bool,
    ) -> EventDiff:
        """Create a diff for a missing or extra event.

        Args:
            event: The event that is missing or extra.
            is_extra: True if event is extra in replay, False if missing.

        Returns:
            EventDiff representing the missing/extra event.
        """
        return {
            "event_id": event.get("event_id", ""),
            "replay_event_id": event.get("event_id") if is_extra else None,
            "node_name": event.get("node_name", ""),
            "event_type": event.get("event_type", ""),
            "input_diffs": [],
            "output_diffs": [],
            "tool_call_diffs": [],
            "llm_call_diffs": [],
            "is_match": False,
            "match_score": 0.0,
        }

    def _diff_dicts(
        self,
        original: dict[str, Any],
        replay: dict[str, Any],
        prefix: str = "",
    ) -> list[FieldDiff]:
        """Recursively diff two dictionaries.

        Args:
            original: Original dictionary.
            replay: Replay dictionary.
            prefix: Path prefix for nested fields.

        Returns:
            List of FieldDiff for all differences.
        """
        diffs: list[FieldDiff] = []
        all_keys = set(original.keys()) | set(replay.keys())

        for key in all_keys:
            path = f"{prefix}.{key}" if prefix else key
            orig_val = original.get(key)
            repl_val = replay.get(key)

            if key not in original:
                diffs.append({
                    "field_path": path,
                    "original_value": None,
                    "replay_value": repl_val,
                    "diff_type": "added",
                    "is_significant": True,
                })
            elif key not in replay:
                diffs.append({
                    "field_path": path,
                    "original_value": orig_val,
                    "replay_value": None,
                    "diff_type": "removed",
                    "is_significant": True,
                })
            elif type(orig_val) != type(repl_val):
                diffs.append({
                    "field_path": path,
                    "original_value": orig_val,
                    "replay_value": repl_val,
                    "diff_type": "type_change",
                    "is_significant": True,
                })
            elif isinstance(orig_val, dict) and isinstance(repl_val, dict):
                # Recurse for nested dicts
                diffs.extend(self._diff_dicts(orig_val, repl_val, path))
            elif orig_val != repl_val:
                # Check if difference is significant
                is_significant = not self._is_insignificant_diff(
                    key, orig_val, repl_val
                )
                diffs.append({
                    "field_path": path,
                    "original_value": orig_val,
                    "replay_value": repl_val,
                    "diff_type": "changed",
                    "is_significant": is_significant,
                })

        return diffs

    def _is_insignificant_diff(
        self,
        key: str,
        original: Any,
        replay: Any,
    ) -> bool:
        """Determine if a difference is insignificant.

        Some differences (like timestamps, IDs) are expected and
        shouldn't affect determinism scoring.

        Args:
            key: Field name.
            original: Original value.
            replay: Replay value.

        Returns:
            True if difference is insignificant.
        """
        # Timestamp fields are expected to differ
        if "timestamp" in key.lower() or "time" in key.lower():
            return True

        # Generated IDs are expected to differ
        if key.endswith("_id") and key != "checkpoint_id":
            return True

        # UUID fields differ
        if key == "id" or key == "event_id":
            return True

        # Latency measurements may vary
        if "latency" in key.lower() or "duration" in key.lower():
            return True

        return False

    def _generate_diff_summary(
        self,
        matched: int,
        mismatched: int,
        missing: int,
        extra: int,
        determinism_score: float,
    ) -> str:
        """Generate a human-readable diff summary.

        Args:
            matched: Number of matched events.
            mismatched: Number of mismatched events.
            missing: Number of missing events.
            extra: Number of extra events.
            determinism_score: Overall determinism score.

        Returns:
            Summary string.
        """
        if determinism_score >= 0.99:
            status = "DETERMINISTIC"
        elif determinism_score >= 0.9:
            status = "MOSTLY DETERMINISTIC"
        elif determinism_score >= 0.7:
            status = "PARTIALLY DETERMINISTIC"
        else:
            status = "NON-DETERMINISTIC"

        parts = [
            f"Status: {status}",
            f"Determinism Score: {determinism_score:.2%}",
            f"Events: {matched} matched, {mismatched} mismatched",
        ]

        if missing > 0:
            parts.append(f"{missing} missing in replay")
        if extra > 0:
            parts.append(f"{extra} extra in replay")

        return " | ".join(parts)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ReplayPlayer",
    "ReplayError",
    "ReplayDivergenceError",
    "ReplayConfig",
    "FieldDiff",
    "EventDiff",
    "DiffReport",
]
