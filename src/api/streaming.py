"""Server-Sent Events (SSE) Streaming Utilities for DRX API.

Provides SSE response generation with proper event formatting,
heartbeat keep-alive, and reconnection support via last_event_id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Callable

from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Stream Event Types
# =============================================================================


class StreamEventType(str, Enum):
    """Enumeration of SSE event types for research streaming.

    These event types align with the DRX streaming protocol and
    provide structured progress updates to clients.
    """

    # Lifecycle events
    INTERACTION_START = "interaction.start"
    INTERACTION_COMPLETE = "interaction.complete"
    INTERACTION_ERROR = "interaction.error"
    INTERACTION_CANCELLED = "interaction.cancelled"

    # Progress events
    THOUGHT_SUMMARY = "thought_summary"
    CONTENT_DELTA = "content.delta"
    CONTENT_COMPLETE = "content.complete"

    # Tool/Agent events
    TOOL_USE = "tool.use"
    TOOL_RESULT = "tool.result"
    AGENT_TRANSITION = "agent.transition"

    # Plan events
    PLAN_CREATED = "plan.created"
    PLAN_UPDATED = "plan.updated"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"

    # Quality events
    GAP_IDENTIFIED = "gap.identified"
    QUALITY_UPDATE = "quality.update"

    # System events
    HEARTBEAT = "heartbeat"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    WARNING = "warning"


# =============================================================================
# Stream Event Model
# =============================================================================


class StreamEvent(BaseModel):
    """Structured event for SSE streaming.

    Attributes:
        event_type: Type of event being streamed.
        data: Event payload data.
        checkpoint_id: Optional checkpoint ID for resumption.
        timestamp: Event timestamp.
        event_id: Unique event ID for reconnection tracking.
    """

    event_type: StreamEventType | str
    data: dict[str, Any]
    checkpoint_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str | None = None

    class Config:
        use_enum_values = True

    def to_sse_format(self) -> str:
        """Format event as SSE wire format.

        Returns:
            SSE-formatted string with event, id, and data fields.
        """
        lines = []

        # Event type
        event_name = (
            self.event_type.value
            if isinstance(self.event_type, StreamEventType)
            else self.event_type
        )
        lines.append(f"event: {event_name}")

        # Event ID for reconnection
        if self.event_id:
            lines.append(f"id: {self.event_id}")

        # Data payload
        payload = {
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.checkpoint_id:
            payload["checkpoint_id"] = self.checkpoint_id

        # SSE spec: each data line must be prefixed with "data: "
        data_json = json.dumps(payload, default=str, ensure_ascii=False)
        lines.append(f"data: {data_json}")

        # Empty line terminates event
        lines.append("")
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# SSE Manager
# =============================================================================


@dataclass
class SSEConfig:
    """Configuration for SSE streaming behavior."""

    heartbeat_interval: float = 15.0  # seconds between heartbeats
    retry_timeout: int = 3000  # milliseconds for client retry
    max_event_size: int = 65536  # maximum event payload size
    buffer_size: int = 100  # event buffer for reconnection


class SSEManager:
    """Manager for Server-Sent Events streaming.

    Handles SSE response generation with:
    - Proper event formatting per SSE spec
    - Heartbeat keep-alive to prevent connection timeout
    - Reconnection support via last_event_id
    - Event buffering for replay on reconnection
    """

    def __init__(
        self,
        event_generator: AsyncGenerator[StreamEvent | dict[str, Any], None],
        config: SSEConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize SSE manager.

        Args:
            event_generator: Async generator yielding stream events.
            config: SSE configuration options.
            session_id: Optional session ID for event tracking.
        """
        self._generator = event_generator
        self._config = config or SSEConfig()
        self._session_id = session_id or str(uuid.uuid4())

        # Event tracking for reconnection
        self._event_counter = 0
        self._event_buffer: list[tuple[str, StreamEvent]] = []
        self._last_sent_id: str | None = None

        # State
        self._cancelled = False
        self._started_at = time.time()

    def _generate_event_id(self) -> str:
        """Generate unique event ID.

        Returns:
            Unique event ID combining session and counter.
        """
        self._event_counter += 1
        return f"{self._session_id}:{self._event_counter}"

    def _buffer_event(self, event_id: str, event: StreamEvent) -> None:
        """Buffer event for potential replay on reconnection.

        Args:
            event_id: Event identifier.
            event: Event to buffer.
        """
        self._event_buffer.append((event_id, event))

        # Trim buffer if exceeds size limit
        if len(self._event_buffer) > self._config.buffer_size:
            self._event_buffer = self._event_buffer[-self._config.buffer_size :]

    def _get_replay_events(self, last_event_id: str) -> list[StreamEvent]:
        """Get events to replay after reconnection.

        Args:
            last_event_id: Last event ID received by client.

        Returns:
            List of events to replay.
        """
        replay = []
        found = False

        for event_id, event in self._event_buffer:
            if found:
                replay.append(event)
            elif event_id == last_event_id:
                found = True

        return replay

    def _normalize_event(self, event: StreamEvent | dict[str, Any]) -> StreamEvent:
        """Normalize event to StreamEvent instance.

        Args:
            event: Event as StreamEvent or dict.

        Returns:
            Normalized StreamEvent instance.
        """
        if isinstance(event, StreamEvent):
            return event

        # Convert dict to StreamEvent
        event_type = event.get("event_type", StreamEventType.CONTENT_DELTA)
        data = event.get("data", event)
        checkpoint_id = event.get("checkpoint_id")
        timestamp = event.get("timestamp")

        if timestamp and not isinstance(timestamp, datetime):
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        return StreamEvent(
            event_type=event_type,
            data=data if isinstance(data, dict) else {"value": data},
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
        )

    def _create_heartbeat(self) -> StreamEvent:
        """Create a heartbeat event.

        Returns:
            Heartbeat StreamEvent.
        """
        return StreamEvent(
            event_type=StreamEventType.HEARTBEAT,
            data={
                "session_id": self._session_id,
                "uptime_seconds": time.time() - self._started_at,
            },
        )

    async def stream(
        self,
        last_event_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate SSE stream with heartbeat and reconnection support.

        Args:
            last_event_id: Optional last event ID for replay.

        Yields:
            SSE-formatted event strings.
        """
        # Send retry directive
        yield f"retry: {self._config.retry_timeout}\n\n"

        # Replay buffered events if reconnecting
        if last_event_id:
            replay_events = self._get_replay_events(last_event_id)
            for event in replay_events:
                event_id = self._generate_event_id()
                event.event_id = event_id
                yield event.to_sse_format()
                logger.debug(f"Replayed event {event_id} after reconnection")

        # Create heartbeat task
        heartbeat_event = asyncio.Event()
        last_heartbeat = time.time()

        async def heartbeat_sender():
            """Background task to send periodic heartbeats."""
            nonlocal last_heartbeat
            while not self._cancelled:
                await asyncio.sleep(self._config.heartbeat_interval)
                if not self._cancelled:
                    heartbeat_event.set()

        heartbeat_task = asyncio.create_task(heartbeat_sender())

        try:
            async for raw_event in self._generator:
                if self._cancelled:
                    break

                # Normalize and process event
                event = self._normalize_event(raw_event)
                event_id = self._generate_event_id()
                event.event_id = event_id

                # Buffer for reconnection
                self._buffer_event(event_id, event)
                self._last_sent_id = event_id

                # Yield formatted event
                yield event.to_sse_format()

                # Check if heartbeat needed
                if heartbeat_event.is_set():
                    heartbeat = self._create_heartbeat()
                    heartbeat.event_id = self._generate_event_id()
                    yield heartbeat.to_sse_format()
                    heartbeat_event.clear()
                    last_heartbeat = time.time()

                # Check for terminal events
                if event.event_type in (
                    StreamEventType.INTERACTION_COMPLETE,
                    StreamEventType.INTERACTION_ERROR,
                    StreamEventType.INTERACTION_CANCELLED,
                ):
                    break

        except asyncio.CancelledError:
            logger.info(f"SSE stream cancelled for session {self._session_id}")
            # Send cancellation event
            cancel_event = StreamEvent(
                event_type=StreamEventType.INTERACTION_CANCELLED,
                data={"reason": "client_disconnect"},
                event_id=self._generate_event_id(),
            )
            yield cancel_event.to_sse_format()

        except Exception as e:
            logger.error(f"SSE stream error: {e}", exc_info=True)
            # Send error event
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(e), "type": type(e).__name__},
                event_id=self._generate_event_id(),
            )
            yield error_event.to_sse_format()

        finally:
            self._cancelled = True
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    def cancel(self) -> None:
        """Cancel the SSE stream."""
        self._cancelled = True


# =============================================================================
# SSE Response Factory
# =============================================================================


def create_sse_response(
    event_generator: AsyncGenerator[StreamEvent | dict[str, Any], None],
    session_id: str | None = None,
    last_event_id: str | None = None,
    config: SSEConfig | None = None,
    headers: dict[str, str] | None = None,
) -> StreamingResponse:
    """Create a FastAPI StreamingResponse for SSE.

    Args:
        event_generator: Async generator yielding stream events.
        session_id: Optional session ID for tracking.
        last_event_id: Optional last event ID for reconnection replay.
        config: Optional SSE configuration.
        headers: Additional response headers.

    Returns:
        StreamingResponse configured for SSE.
    """
    manager = SSEManager(
        event_generator=event_generator,
        config=config,
        session_id=session_id,
    )

    # Build headers
    response_headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
        "Access-Control-Allow-Origin": "*",  # CORS for SSE
    }
    if headers:
        response_headers.update(headers)

    return StreamingResponse(
        manager.stream(last_event_id=last_event_id),
        media_type="text/event-stream",
        headers=response_headers,
    )


# =============================================================================
# Stream Event Helpers
# =============================================================================


def create_start_event(
    interaction_id: str,
    query: str,
    **extra_data: Any,
) -> StreamEvent:
    """Create an interaction start event.

    Args:
        interaction_id: Unique interaction identifier.
        query: Research query.
        **extra_data: Additional event data.

    Returns:
        StreamEvent for interaction start.
    """
    return StreamEvent(
        event_type=StreamEventType.INTERACTION_START,
        data={
            "id": interaction_id,
            "query": query,
            "status": "started",
            **extra_data,
        },
    )


def create_thought_event(
    text: str,
    agent: str | None = None,
    phase: str | None = None,
    **extra_data: Any,
) -> StreamEvent:
    """Create a thought summary event.

    Args:
        text: Thought/reasoning text.
        agent: Optional agent name.
        phase: Optional workflow phase.
        **extra_data: Additional event data.

    Returns:
        StreamEvent for thought summary.
    """
    data = {"text": text}
    if agent:
        data["agent"] = agent
    if phase:
        data["phase"] = phase
    data.update(extra_data)

    return StreamEvent(
        event_type=StreamEventType.THOUGHT_SUMMARY,
        data=data,
    )


def create_content_delta_event(
    text: str,
    section: str | None = None,
    **extra_data: Any,
) -> StreamEvent:
    """Create a content delta event.

    Args:
        text: Content text delta.
        section: Optional section name.
        **extra_data: Additional event data.

    Returns:
        StreamEvent for content delta.
    """
    data = {"text": text}
    if section:
        data["section"] = section
    data.update(extra_data)

    return StreamEvent(
        event_type=StreamEventType.CONTENT_DELTA,
        data=data,
    )


def create_tool_event(
    tool_name: str,
    tool_input: dict[str, Any] | None = None,
    tool_result: Any = None,
    is_result: bool = False,
    **extra_data: Any,
) -> StreamEvent:
    """Create a tool use/result event.

    Args:
        tool_name: Name of the tool.
        tool_input: Tool input parameters.
        tool_result: Tool execution result.
        is_result: Whether this is a result event.
        **extra_data: Additional event data.

    Returns:
        StreamEvent for tool use or result.
    """
    data = {"tool": tool_name}
    if tool_input is not None:
        data["input"] = tool_input
    if tool_result is not None:
        data["result"] = tool_result
    data.update(extra_data)

    return StreamEvent(
        event_type=StreamEventType.TOOL_RESULT if is_result else StreamEventType.TOOL_USE,
        data=data,
    )


def create_complete_event(
    interaction_id: str,
    status: str = "completed",
    result: dict[str, Any] | None = None,
    **extra_data: Any,
) -> StreamEvent:
    """Create an interaction complete event.

    Args:
        interaction_id: Interaction identifier.
        status: Completion status.
        result: Optional result data.
        **extra_data: Additional event data.

    Returns:
        StreamEvent for interaction complete.
    """
    data = {
        "id": interaction_id,
        "status": status,
    }
    if result:
        data["result"] = result
    data.update(extra_data)

    return StreamEvent(
        event_type=StreamEventType.INTERACTION_COMPLETE,
        data=data,
    )


def create_error_event(
    error: str,
    error_type: str | None = None,
    interaction_id: str | None = None,
    recoverable: bool = False,
    **extra_data: Any,
) -> StreamEvent:
    """Create an error event.

    Args:
        error: Error message.
        error_type: Type of error.
        interaction_id: Optional interaction ID.
        recoverable: Whether error is recoverable.
        **extra_data: Additional event data.

    Returns:
        StreamEvent for error.
    """
    data = {
        "error": error,
        "recoverable": recoverable,
    }
    if error_type:
        data["type"] = error_type
    if interaction_id:
        data["id"] = interaction_id
    data.update(extra_data)

    return StreamEvent(
        event_type=StreamEventType.ERROR,
        data=data,
    )


# =============================================================================
# Async Event Generator Helpers
# =============================================================================


async def merge_generators(
    *generators: AsyncGenerator[StreamEvent, None],
) -> AsyncGenerator[StreamEvent, None]:
    """Merge multiple async generators into one.

    Useful for combining events from multiple sources.

    Args:
        *generators: Async generators to merge.

    Yields:
        Events from all generators as they arrive.
    """
    pending: set[asyncio.Task] = set()
    gen_map: dict[asyncio.Task, AsyncGenerator] = {}

    # Create initial tasks for each generator
    for gen in generators:
        task = asyncio.create_task(gen.__anext__())
        pending.add(task)
        gen_map[task] = gen

    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                gen = gen_map.pop(task)
                try:
                    event = task.result()
                    yield event

                    # Create new task for this generator
                    new_task = asyncio.create_task(gen.__anext__())
                    pending.add(new_task)
                    gen_map[new_task] = gen

                except StopAsyncIteration:
                    # Generator exhausted
                    pass

    finally:
        # Cancel any remaining tasks
        for task in pending:
            task.cancel()


async def timeout_generator(
    generator: AsyncGenerator[StreamEvent, None],
    timeout: float,
    timeout_event: StreamEvent | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Wrap generator with timeout.

    Args:
        generator: Source generator.
        timeout: Timeout in seconds.
        timeout_event: Optional event to yield on timeout.

    Yields:
        Events from generator until timeout.
    """
    start = time.time()

    async for event in generator:
        if time.time() - start > timeout:
            if timeout_event:
                yield timeout_event
            break
        yield event


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "StreamEventType",
    # Models
    "StreamEvent",
    # Config
    "SSEConfig",
    # Manager
    "SSEManager",
    # Factory
    "create_sse_response",
    # Event helpers
    "create_start_event",
    "create_thought_event",
    "create_content_delta_event",
    "create_tool_event",
    "create_complete_event",
    "create_error_event",
    # Generator helpers
    "merge_generators",
    "timeout_generator",
]
