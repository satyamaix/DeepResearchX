"""Replay API Routes for DRX Deep Research.

Provides REST endpoints for replay operations including:
- Starting session replay from checkpoints
- Retrieving recorded events
- Comparing original vs replay runs

These endpoints enable debugging, training data generation,
and reproducible research runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Query,
    status,
)
from pydantic import BaseModel, Field

from src.api.dependencies import (
    CurrentUserDep,
    DatabaseDep,
    RateLimitDependency,
    RedisDep,
    SettingsDep,
    rate_limit_heavy,
    rate_limit_standard,
)
from src.api.models import ErrorResponse
from src.api.streaming import (
    StreamEvent,
    StreamEventType,
    create_sse_response,
    SSEConfig,
)
from src.replay import EventRecorder, ReplayPlayer, ReplayEvent, DiffReport

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class ReplayRequest(BaseModel):
    """Request body for starting a replay."""

    checkpoint_id: str = Field(
        ...,
        description="Checkpoint ID to start replay from",
        min_length=1,
    )
    modifications: dict[str, Any] | None = Field(
        default=None,
        description="Optional modifications to apply during replay",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional replay configuration overrides",
    )


class ReplayResponse(BaseModel):
    """Response model for replay operations."""

    replay_session_id: str = Field(..., description="ID of the new replay session")
    original_session_id: str = Field(..., description="Original session being replayed")
    checkpoint_id: str = Field(..., description="Starting checkpoint")
    status: str = Field(..., description="Replay status")
    created_at: datetime = Field(..., description="When replay was started")
    stream_url: str | None = Field(None, description="URL to stream replay events")


class EventListResponse(BaseModel):
    """Response model for listing recorded events."""

    session_id: str = Field(..., description="Session ID")
    events: list[dict[str, Any]] = Field(..., description="List of events")
    total: int = Field(..., description="Total number of events")
    from_checkpoint: str | None = Field(None, description="Starting checkpoint filter")
    has_more: bool = Field(default=False, description="Whether more events exist")


class EventResponse(BaseModel):
    """Response model for a single event."""

    event_id: str
    session_id: str
    checkpoint_id: str | None
    event_type: str
    node_name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    timestamp: str
    sequence_number: int
    metadata: dict[str, Any]


class CompareRequest(BaseModel):
    """Request body for comparing sessions."""

    replay_session_id: str = Field(
        ...,
        description="Replay session to compare against original",
    )
    include_field_diffs: bool = Field(
        default=True,
        description="Whether to include detailed field-level diffs",
    )


class CompareResponse(BaseModel):
    """Response model for session comparison."""

    original_session_id: str
    replay_session_id: str
    total_events_original: int
    total_events_replay: int
    matched_events: int
    mismatched_events: int
    missing_events: int
    extra_events: int
    summary: str
    is_deterministic: bool
    determinism_score: float
    event_diffs: list[dict[str, Any]] | None = Field(
        None,
        description="Per-event diffs (if requested)",
    )
    created_at: datetime


class SessionSummaryResponse(BaseModel):
    """Response model for session event summary."""

    session_id: str
    total_events: int
    first_event_at: str | None
    last_event_at: str | None
    unique_nodes: int
    event_types: list[str]
    duration_ms: int


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["replay"])


# =============================================================================
# Replay Endpoints
# =============================================================================


@router.post(
    "/interactions/{interaction_id}/replay",
    response_model=ReplayResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start session replay",
    description="Start replaying a research session from a checkpoint",
    responses={
        202: {"description": "Replay started successfully"},
        404: {"model": ErrorResponse, "description": "Session or checkpoint not found"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def start_replay(
    interaction_id: Annotated[
        str,
        Path(description="Interaction ID to replay"),
    ],
    request: ReplayRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseDep,
    redis: RedisDep,
    user: CurrentUserDep,
    settings: SettingsDep,
    _rate_limit: Annotated[None, Depends(rate_limit_heavy)] = None,
) -> ReplayResponse:
    """Start replaying a research session from a checkpoint.

    This endpoint:
    1. Validates the session and checkpoint exist
    2. Creates a new replay session
    3. Starts replay in the background
    4. Returns immediately with replay session details

    The client can then stream replay events via the standard
    /interactions/{id}/stream endpoint using the replay_session_id.
    """
    # Verify original session exists and user has access
    result = await db.execute(
        """
        SELECT id, user_id, status
        FROM research_sessions
        WHERE id = %(id)s
        """,
        {"id": interaction_id},
    )
    row = await result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {interaction_id} not found",
        )

    # Verify ownership
    if row["user_id"] != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session",
        )

    # Create replay session record
    import uuid

    replay_session_id = f"replay_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc)

    try:
        await db.execute(
            """
            INSERT INTO research_sessions (
                id, user_id, query, status, metadata, created_at, updated_at
            ) VALUES (
                %(id)s, %(user_id)s, %(query)s, %(status)s, %(metadata)s,
                %(created_at)s, %(updated_at)s
            )
            """,
            {
                "id": replay_session_id,
                "user_id": user.id,
                "query": f"Replay of {interaction_id}",
                "status": "running",
                "metadata": {
                    "replay": True,
                    "original_session_id": interaction_id,
                    "checkpoint_id": request.checkpoint_id,
                    "modifications": request.modifications,
                },
                "created_at": now,
                "updated_at": now,
            },
        )
    except Exception as e:
        logger.error(f"Failed to create replay session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create replay session",
        )

    # Cache replay metadata in Redis
    await redis.hset(
        f"interaction:{replay_session_id}",
        mapping={
            "status": "running",
            "user_id": user.id,
            "original_session_id": interaction_id,
            "checkpoint_id": request.checkpoint_id,
            "created_at": now.isoformat(),
        },
    )
    await redis.expire(f"interaction:{replay_session_id}", 86400)  # 24h TTL

    # Start replay in background
    async def _run_replay():
        try:
            recorder = EventRecorder()
            player = ReplayPlayer(recorder=recorder)

            if request.modifications:
                event_gen = player.replay_with_modifications(
                    session_id=interaction_id,
                    modifications=request.modifications,
                    checkpoint_id=request.checkpoint_id,
                    config=request.config,
                )
            else:
                event_gen = player.replay_from_checkpoint(
                    session_id=interaction_id,
                    checkpoint_id=request.checkpoint_id,
                    config=request.config,
                )

            async for event in event_gen:
                # Publish event to Redis for streaming
                stream_event = StreamEvent(
                    event_type=event.get("event_type", "replay.event"),
                    data=event,
                    checkpoint_id=event.get("checkpoint_id"),
                )
                await redis.publish(
                    f"events:{replay_session_id}",
                    stream_event.model_dump_json(),
                )

            # Update status to completed
            await redis.hset(f"interaction:{replay_session_id}", "status", "completed")

            # Update database
            await db.execute(
                """
                UPDATE research_sessions
                SET status = 'completed', updated_at = %(updated_at)s
                WHERE id = %(id)s
                """,
                {"id": replay_session_id, "updated_at": datetime.now(timezone.utc)},
            )

        except Exception as e:
            logger.error(f"Replay failed: {e}", exc_info=True)
            await redis.hset(f"interaction:{replay_session_id}", "status", "failed")
            await redis.hset(f"interaction:{replay_session_id}", "error", str(e))

            # Publish error event
            error_event = StreamEvent(
                event_type=StreamEventType.INTERACTION_ERROR,
                data={"error": str(e), "replay_session_id": replay_session_id},
            )
            await redis.publish(
                f"events:{replay_session_id}",
                error_event.model_dump_json(),
            )

    background_tasks.add_task(_run_replay)

    logger.info(
        f"Started replay of {interaction_id} as {replay_session_id}",
        extra={
            "original_session_id": interaction_id,
            "replay_session_id": replay_session_id,
            "checkpoint_id": request.checkpoint_id,
            "user_id": user.id,
        },
    )

    return ReplayResponse(
        replay_session_id=replay_session_id,
        original_session_id=interaction_id,
        checkpoint_id=request.checkpoint_id,
        status="running",
        created_at=now,
        stream_url=f"/api/v1/interactions/{replay_session_id}/stream",
    )


@router.get(
    "/interactions/{interaction_id}/events",
    response_model=EventListResponse,
    summary="Get recorded events",
    description="Retrieve recorded events for a research session",
    responses={
        200: {"description": "Events retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_events(
    interaction_id: Annotated[
        str,
        Path(description="Interaction ID"),
    ],
    db: DatabaseDep,
    redis: RedisDep,
    user: CurrentUserDep,
    from_checkpoint: Annotated[
        str | None,
        Query(description="Checkpoint ID to start from"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description="Maximum events to return"),
    ] = 100,
    event_types: Annotated[
        list[str] | None,
        Query(description="Filter by event types"),
    ] = None,
    _rate_limit: Annotated[None, Depends(rate_limit_standard)] = None,
) -> EventListResponse:
    """Retrieve recorded events for a research session.

    Returns events ordered by sequence number, optionally filtered
    by checkpoint and event type.
    """
    # Verify session exists and user has access
    result = await db.execute(
        """
        SELECT id, user_id
        FROM research_sessions
        WHERE id = %(id)s
        """,
        {"id": interaction_id},
    )
    row = await result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {interaction_id} not found",
        )

    # Verify ownership
    if row["user_id"] != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session",
        )

    # Get events
    try:
        recorder = EventRecorder()
        events = await recorder.get_events(
            session_id=interaction_id,
            from_checkpoint=from_checkpoint,
            limit=limit + 1,  # Get one extra to check has_more
            event_types=event_types,
        )

        has_more = len(events) > limit
        if has_more:
            events = events[:limit]

        return EventListResponse(
            session_id=interaction_id,
            events=[dict(e) for e in events],
            total=len(events),
            from_checkpoint=from_checkpoint,
            has_more=has_more,
        )

    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve events",
        )


@router.get(
    "/interactions/{interaction_id}/events/{event_id}",
    response_model=EventResponse,
    summary="Get single event",
    description="Retrieve a specific recorded event by ID",
    responses={
        200: {"description": "Event retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Event not found"},
    },
)
async def get_event(
    interaction_id: Annotated[str, Path(description="Interaction ID")],
    event_id: Annotated[str, Path(description="Event ID")],
    db: DatabaseDep,
    user: CurrentUserDep,
    _rate_limit: Annotated[None, Depends(rate_limit_standard)] = None,
) -> EventResponse:
    """Retrieve a specific recorded event by ID."""
    # Verify session access
    result = await db.execute(
        """
        SELECT user_id FROM research_sessions WHERE id = %(id)s
        """,
        {"id": interaction_id},
    )
    row = await result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {interaction_id} not found",
        )

    if row["user_id"] != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session",
        )

    # Get specific event
    try:
        recorder = EventRecorder()
        event = await recorder.get_event_by_id(event_id)

        if not event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event {event_id} not found",
            )

        # Verify event belongs to session
        if event.get("session_id") != interaction_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event {event_id} not found in session {interaction_id}",
            )

        return EventResponse(
            event_id=event.get("event_id", ""),
            session_id=event.get("session_id", ""),
            checkpoint_id=event.get("checkpoint_id"),
            event_type=event.get("event_type", ""),
            node_name=event.get("node_name", ""),
            inputs=event.get("inputs", {}),
            outputs=event.get("outputs", {}),
            tool_calls=event.get("tool_calls", []),
            llm_calls=event.get("llm_calls", []),
            timestamp=event.get("timestamp", ""),
            sequence_number=event.get("sequence_number", 0),
            metadata=event.get("metadata", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve event",
        )


@router.get(
    "/interactions/{interaction_id}/events/summary",
    response_model=SessionSummaryResponse,
    summary="Get events summary",
    description="Get summary statistics for recorded session events",
    responses={
        200: {"description": "Summary retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_events_summary(
    interaction_id: Annotated[str, Path(description="Interaction ID")],
    db: DatabaseDep,
    user: CurrentUserDep,
    _rate_limit: Annotated[None, Depends(rate_limit_standard)] = None,
) -> SessionSummaryResponse:
    """Get summary statistics for recorded session events."""
    # Verify session access
    result = await db.execute(
        """
        SELECT user_id FROM research_sessions WHERE id = %(id)s
        """,
        {"id": interaction_id},
    )
    row = await result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {interaction_id} not found",
        )

    if row["user_id"] != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session",
        )

    try:
        recorder = EventRecorder()
        summary = await recorder.get_session_summary(interaction_id)

        return SessionSummaryResponse(
            session_id=summary["session_id"],
            total_events=summary["total_events"],
            first_event_at=summary.get("first_event_at"),
            last_event_at=summary.get("last_event_at"),
            unique_nodes=summary["unique_nodes"],
            event_types=summary.get("event_types", []),
            duration_ms=summary.get("duration_ms", 0),
        )

    except Exception as e:
        logger.error(f"Failed to get session summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session summary",
        )


@router.post(
    "/interactions/{interaction_id}/compare",
    response_model=CompareResponse,
    summary="Compare sessions",
    description="Compare original session with a replay session",
    responses={
        200: {"description": "Comparison completed successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        400: {"model": ErrorResponse, "description": "Invalid comparison request"},
    },
)
async def compare_sessions(
    interaction_id: Annotated[
        str,
        Path(description="Original interaction ID"),
    ],
    request: CompareRequest,
    db: DatabaseDep,
    user: CurrentUserDep,
    _rate_limit: Annotated[None, Depends(rate_limit_heavy)] = None,
) -> CompareResponse:
    """Compare original session with a replay session.

    Performs field-level comparison to identify divergences and
    measure replay determinism.
    """
    # Verify original session access
    result = await db.execute(
        """
        SELECT user_id FROM research_sessions WHERE id = %(id)s
        """,
        {"id": interaction_id},
    )
    row = await result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original session {interaction_id} not found",
        )

    if row["user_id"] != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to original session",
        )

    # Verify replay session access
    result = await db.execute(
        """
        SELECT user_id, metadata FROM research_sessions WHERE id = %(id)s
        """,
        {"id": request.replay_session_id},
    )
    replay_row = await result.fetchone()

    if not replay_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay session {request.replay_session_id} not found",
        )

    if replay_row["user_id"] != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to replay session",
        )

    # Verify this is a replay of the original session
    metadata = replay_row.get("metadata", {})
    if isinstance(metadata, str):
        import json
        metadata = json.loads(metadata)

    if metadata.get("original_session_id") != interaction_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session {request.replay_session_id} is not a replay of {interaction_id}",
        )

    # Perform comparison
    try:
        recorder = EventRecorder()
        player = ReplayPlayer(recorder=recorder)

        diff_report = await player.compare_runs(
            original_session=interaction_id,
            replay_session=request.replay_session_id,
            include_field_diffs=request.include_field_diffs,
        )

        return CompareResponse(
            original_session_id=diff_report.get("original_session_id", ""),
            replay_session_id=diff_report.get("replay_session_id", ""),
            total_events_original=diff_report.get("total_events_original", 0),
            total_events_replay=diff_report.get("total_events_replay", 0),
            matched_events=diff_report.get("matched_events", 0),
            mismatched_events=diff_report.get("mismatched_events", 0),
            missing_events=diff_report.get("missing_events", 0),
            extra_events=diff_report.get("extra_events", 0),
            summary=diff_report.get("summary", ""),
            is_deterministic=diff_report.get("is_deterministic", False),
            determinism_score=diff_report.get("determinism_score", 0.0),
            event_diffs=(
                [dict(d) for d in diff_report.get("event_diffs", [])]
                if request.include_field_diffs
                else None
            ),
            created_at=datetime.fromisoformat(diff_report.get("created_at", datetime.now(timezone.utc).isoformat())),
        )

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare sessions",
        )


@router.delete(
    "/interactions/{interaction_id}/events",
    status_code=status.HTTP_200_OK,
    summary="Delete recorded events",
    description="Delete recorded replay events for a session",
    responses={
        200: {"description": "Events deleted successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def delete_events(
    interaction_id: Annotated[str, Path(description="Interaction ID")],
    db: DatabaseDep,
    user: CurrentUserDep,
    before_checkpoint: Annotated[
        str | None,
        Query(description="Delete only events before this checkpoint"),
    ] = None,
    _rate_limit: Annotated[None, Depends(rate_limit_heavy)] = None,
) -> dict[str, Any]:
    """Delete recorded replay events for a session.

    Optionally delete only events before a specific checkpoint.
    """
    # Verify session access
    result = await db.execute(
        """
        SELECT user_id FROM research_sessions WHERE id = %(id)s
        """,
        {"id": interaction_id},
    )
    row = await result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {interaction_id} not found",
        )

    if row["user_id"] != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session",
        )

    try:
        recorder = EventRecorder()
        deleted_count = await recorder.delete_events(
            session_id=interaction_id,
            before_checkpoint=before_checkpoint,
        )

        logger.info(
            f"Deleted {deleted_count} events for session {interaction_id}",
            extra={
                "session_id": interaction_id,
                "before_checkpoint": before_checkpoint,
                "deleted_count": deleted_count,
            },
        )

        return {
            "session_id": interaction_id,
            "deleted_count": deleted_count,
            "before_checkpoint": before_checkpoint,
        }

    except Exception as e:
        logger.error(f"Failed to delete events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete events",
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "router",
    "ReplayRequest",
    "ReplayResponse",
    "EventListResponse",
    "EventResponse",
    "CompareRequest",
    "CompareResponse",
    "SessionSummaryResponse",
]
