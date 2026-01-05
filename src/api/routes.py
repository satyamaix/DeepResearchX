"""API Route Definitions for DRX Deep Research.

Provides REST endpoints for creating, managing, and streaming
research interactions with proper validation and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from pydantic import BaseModel, Field, field_validator

from src.api.dependencies import (
    CurrentUserDep,
    DatabaseDep,
    OrchestratorDep,
    RateLimitDependency,
    RedisDep,
    SettingsDep,
    User,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_streaming,
)
from src.api.models import ErrorResponse
from src.api.streaming import (
    SSEConfig,
    StreamEvent,
    StreamEventType,
    create_complete_event,
    create_error_event,
    create_sse_response,
    create_start_event,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class SteerabilityConfig(BaseModel):
    """User-configurable parameters for steering research output."""

    tone: str = Field(
        default="technical",
        description="Output tone: executive, technical, or casual",
        pattern="^(executive|technical|casual)$",
    )
    format: str = Field(
        default="markdown",
        description="Output format: markdown, markdown_table, or json",
        pattern="^(markdown|markdown_table|json)$",
    )
    max_sources: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of sources to include",
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Priority focus areas for research",
    )
    exclude_topics: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Topics to exclude from research",
    )
    preferred_domains: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Preferred source domains",
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=5,
        description="Output language (ISO 639-1 code)",
    )
    custom_instructions: str | None = Field(
        default=None,
        max_length=2000,
        description="Custom instructions for research",
    )


class ResearchConfig(BaseModel):
    """Configuration options for research execution."""

    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum research iterations",
    )
    token_budget: int = Field(
        default=500000,
        ge=10000,
        le=2000000,
        description="Maximum tokens to use",
    )
    timeout_seconds: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Maximum execution time in seconds",
    )
    enable_citations: bool = Field(
        default=True,
        description="Include source citations in output",
    )
    enable_quality_checks: bool = Field(
        default=True,
        description="Enable quality verification",
    )


class ResearchRequest(BaseModel):
    """Request body for creating a research interaction."""

    input: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Research query or question",
    )
    steerability: SteerabilityConfig | None = Field(
        default=None,
        description="Steerability parameters",
    )
    config: ResearchConfig | None = Field(
        default=None,
        description="Research configuration",
    )

    @field_validator("input")
    @classmethod
    def validate_input_not_empty(cls, v: str) -> str:
        """Ensure input is not just whitespace."""
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError("Input must contain at least 10 non-whitespace characters")
        return stripped


class InteractionResponse(BaseModel):
    """Response model for interaction operations."""

    id: str = Field(..., description="Unique interaction identifier")
    status: str = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")
    query: str | None = Field(None, description="Research query")
    result: dict[str, Any] | None = Field(None, description="Result data if completed")
    error: str | None = Field(None, description="Error message if failed")


class InteractionListResponse(BaseModel):
    """Response model for listing interactions."""

    interactions: list[InteractionResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    version: str
    timestamp: datetime
    checks: dict[str, dict[str, Any]]


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["research"])


# =============================================================================
# Health Check Endpoint
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and dependencies status",
    tags=["system"],
)
async def health_check(
    db: DatabaseDep,
    redis: RedisDep,
    settings: SettingsDep,
) -> HealthResponse:
    """Perform health check on API and dependencies.

    Returns status of database, Redis, and other services.
    """
    checks: dict[str, dict[str, Any]] = {}

    # Database check
    try:
        result = await db.execute("SELECT 1")
        row = await result.fetchone()
        checks["database"] = {
            "status": "healthy" if row else "unhealthy",
            "latency_ms": 0,  # Could add timing
        }
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}

    # Redis check
    try:
        pong = await redis.ping()
        checks["redis"] = {
            "status": "healthy" if pong else "unhealthy",
        }
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}

    # Overall status
    all_healthy = all(c.get("status") == "healthy" for c in checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )


# =============================================================================
# Interaction CRUD Endpoints
# =============================================================================


@router.post(
    "/interactions",
    response_model=InteractionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create research interaction",
    description="Queue a new research interaction for processing",
    responses={
        202: {"description": "Interaction queued successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_interaction(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseDep,
    redis: RedisDep,
    orchestrator: OrchestratorDep,
    user: CurrentUserDep,
    settings: SettingsDep,
    _rate_limit: Annotated[None, Depends(rate_limit_heavy)] = None,
) -> InteractionResponse:
    """Create a new research interaction.

    This endpoint:
    1. Validates the research request
    2. Creates an interaction record in the database
    3. Queues the research job for background processing
    4. Returns immediately with 202 Accepted

    The client should then connect to the /stream endpoint
    to receive real-time progress updates.
    """
    # Generate unique interaction ID
    interaction_id = f"int_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc)

    # Prepare configuration
    steerability = request.steerability or SteerabilityConfig()
    config = request.config or ResearchConfig()

    # Store interaction in database
    try:
        await db.execute(
            """
            INSERT INTO research_sessions (
                id, user_id, query, status, config, steerability,
                created_at, updated_at
            ) VALUES (
                %(id)s, %(user_id)s, %(query)s, %(status)s, %(config)s,
                %(steerability)s, %(created_at)s, %(updated_at)s
            )
            """,
            {
                "id": interaction_id,
                "user_id": user.id,
                "query": request.input,
                "status": "queued",
                "config": config.model_dump_json(),
                "steerability": steerability.model_dump_json(),
                "created_at": now,
                "updated_at": now,
            },
        )
    except Exception as e:
        logger.error(f"Failed to create interaction record: {e}")
        # Continue anyway - we can still process without persistence
        # In production, you might want to fail here

    # Cache interaction metadata in Redis for quick access
    await redis.hset(
        f"interaction:{interaction_id}",
        mapping={
            "status": "queued",
            "user_id": user.id,
            "query": request.input,
            "created_at": now.isoformat(),
        },
    )
    await redis.expire(f"interaction:{interaction_id}", 86400)  # 24h TTL

    # Queue background processing
    # In production, this would dispatch to Celery/Redis queue
    # For now, we'll use FastAPI background tasks
    background_tasks.add_task(
        _process_interaction,
        interaction_id=interaction_id,
        query=request.input,
        steerability=steerability.model_dump(),
        config=config.model_dump(),
        user_id=user.id,
        orchestrator=orchestrator,
        redis=redis,
    )

    logger.info(
        f"Created interaction {interaction_id} for user {user.id}",
        extra={"interaction_id": interaction_id, "user_id": user.id},
    )

    return InteractionResponse(
        id=interaction_id,
        status="queued",
        created_at=now,
        query=request.input,
    )


async def _process_interaction(
    interaction_id: str,
    query: str,
    steerability: dict[str, Any],
    config: dict[str, Any],
    user_id: str,
    orchestrator: Any,
    redis: Any,
) -> None:
    """Background task to process a research interaction.

    This runs the orchestrator and publishes events to Redis
    for consumption by the streaming endpoint.
    """
    try:
        # Update status to running
        await redis.hset(f"interaction:{interaction_id}", "status", "running")

        # Publish start event
        start_event = create_start_event(interaction_id, query)
        await redis.publish(
            f"events:{interaction_id}",
            start_event.model_dump_json(),
        )

        # Run the orchestrator
        async for event in orchestrator.run(
            query=query,
            session_id=interaction_id,
            config={**config, "steerability": steerability},
        ):
            # Publish event to Redis channel
            if isinstance(event, dict):
                event_data = StreamEvent(
                    event_type=event.get("event_type", "content.delta"),
                    data=event.get("data", event),
                    checkpoint_id=event.get("checkpoint_id"),
                )
            else:
                event_data = event

            await redis.publish(
                f"events:{interaction_id}",
                event_data.model_dump_json(),
            )

        # Update status to completed
        await redis.hset(f"interaction:{interaction_id}", "status", "completed")

        # Publish completion event
        complete_event = create_complete_event(interaction_id, "completed")
        await redis.publish(
            f"events:{interaction_id}",
            complete_event.model_dump_json(),
        )

    except asyncio.CancelledError:
        await redis.hset(f"interaction:{interaction_id}", "status", "cancelled")
        cancel_event = StreamEvent(
            event_type=StreamEventType.INTERACTION_CANCELLED,
            data={"id": interaction_id, "reason": "cancelled"},
        )
        await redis.publish(f"events:{interaction_id}", cancel_event.model_dump_json())

    except Exception as e:
        logger.error(f"Interaction {interaction_id} failed: {e}", exc_info=True)
        await redis.hset(f"interaction:{interaction_id}", "status", "failed")
        await redis.hset(f"interaction:{interaction_id}", "error", str(e))

        error_event = create_error_event(
            error=str(e),
            error_type=type(e).__name__,
            interaction_id=interaction_id,
        )
        await redis.publish(f"events:{interaction_id}", error_event.model_dump_json())


@router.get(
    "/interactions/{interaction_id}",
    response_model=InteractionResponse,
    summary="Get interaction status",
    description="Retrieve the current status and result of an interaction",
    responses={
        200: {"description": "Interaction found"},
        404: {"model": ErrorResponse, "description": "Interaction not found"},
    },
)
async def get_interaction(
    interaction_id: Annotated[str, Path(description="Interaction ID", pattern="^int_[a-f0-9]{16}$")],
    db: DatabaseDep,
    redis: RedisDep,
    user: CurrentUserDep,
    _rate_limit: Annotated[None, Depends(rate_limit_standard)] = None,
) -> InteractionResponse:
    """Get the current status of a research interaction.

    Returns the interaction status, and if completed, the result.
    """
    # Try Redis cache first for real-time status
    cached = await redis.hgetall(f"interaction:{interaction_id}")

    if cached:
        # Verify user owns this interaction
        if cached.get("user_id") != user.id and not getattr(user, "is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this interaction",
            )

        return InteractionResponse(
            id=interaction_id,
            status=cached.get("status", "unknown"),
            created_at=datetime.fromisoformat(cached["created_at"]) if "created_at" in cached else datetime.now(timezone.utc),
            query=cached.get("query"),
            error=cached.get("error"),
        )

    # Fall back to database
    try:
        result = await db.execute(
            """
            SELECT id, status, query, result, error, created_at, updated_at, user_id
            FROM research_sessions
            WHERE id = %(id)s
            """,
            {"id": interaction_id},
        )
        row = await result.fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interaction {interaction_id} not found",
            )

        # Verify ownership
        if row["user_id"] != user.id and not getattr(user, "is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this interaction",
            )

        return InteractionResponse(
            id=row["id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row.get("updated_at"),
            query=row.get("query"),
            result=row.get("result"),
            error=row.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch interaction {interaction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve interaction",
        )


@router.get(
    "/interactions/{interaction_id}/stream",
    summary="Stream interaction events",
    description="SSE endpoint for real-time progress updates",
    response_class=Response,
    responses={
        200: {"description": "SSE stream", "content": {"text/event-stream": {}}},
        404: {"model": ErrorResponse, "description": "Interaction not found"},
    },
)
async def stream_interaction(
    interaction_id: Annotated[str, Path(description="Interaction ID", pattern="^int_[a-f0-9]{16}$")],
    request: Request,
    redis: RedisDep,
    user: CurrentUserDep,
    settings: SettingsDep,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    _rate_limit: Annotated[None, Depends(rate_limit_streaming)] = None,
) -> Response:
    """Stream real-time events for a research interaction.

    This SSE endpoint streams:
    - interaction.start: When processing begins
    - thought_summary: Agent reasoning updates
    - content.delta: Incremental content updates
    - tool.use: Tool invocations
    - interaction.complete: When processing finishes
    - error: If an error occurs

    Supports reconnection via Last-Event-ID header.
    """
    # Verify interaction exists and user has access
    cached = await redis.hgetall(f"interaction:{interaction_id}")

    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interaction {interaction_id} not found",
        )

    if cached.get("user_id") != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this interaction",
        )

    # Create event generator from Redis pubsub
    async def event_generator():
        """Generate events from Redis pubsub channel."""
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"events:{interaction_id}")

        try:
            # Check if already completed
            current_status = await redis.hget(f"interaction:{interaction_id}", "status")
            if current_status in ("completed", "failed", "cancelled"):
                # Send final status event and return
                if current_status == "completed":
                    yield create_complete_event(interaction_id, "completed")
                elif current_status == "failed":
                    error_msg = await redis.hget(f"interaction:{interaction_id}", "error")
                    yield create_error_event(
                        error=error_msg or "Unknown error",
                        interaction_id=interaction_id,
                    )
                return

            # Stream events
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event_data = message["data"]
                        if isinstance(event_data, bytes):
                            event_data = event_data.decode("utf-8")
                        event = StreamEvent.model_validate_json(event_data)
                        yield event

                        # Check for terminal events
                        if event.event_type in (
                            StreamEventType.INTERACTION_COMPLETE,
                            StreamEventType.INTERACTION_ERROR,
                            StreamEventType.INTERACTION_CANCELLED,
                            "interaction.complete",
                            "interaction.error",
                            "interaction.cancelled",
                        ):
                            break
                    except Exception as e:
                        logger.warning(f"Failed to parse event: {e}")
                        continue

        finally:
            await pubsub.unsubscribe(f"events:{interaction_id}")
            await pubsub.aclose()

    # Return SSE response
    return create_sse_response(
        event_generator=event_generator(),
        session_id=interaction_id,
        last_event_id=last_event_id,
        config=SSEConfig(
            heartbeat_interval=15.0,
            retry_timeout=3000,
        ),
    )


@router.delete(
    "/interactions/{interaction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel interaction",
    description="Cancel a running research interaction",
    responses={
        204: {"description": "Interaction cancelled"},
        404: {"model": ErrorResponse, "description": "Interaction not found"},
        409: {"model": ErrorResponse, "description": "Interaction already completed"},
    },
)
async def cancel_interaction(
    interaction_id: Annotated[str, Path(description="Interaction ID", pattern="^int_[a-f0-9]{16}$")],
    db: DatabaseDep,
    redis: RedisDep,
    orchestrator: OrchestratorDep,
    user: CurrentUserDep,
    _rate_limit: Annotated[None, Depends(rate_limit_standard)] = None,
) -> Response:
    """Cancel a running research interaction.

    Sends cancellation signal to the orchestrator and updates status.
    """
    # Verify interaction exists
    cached = await redis.hgetall(f"interaction:{interaction_id}")

    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interaction {interaction_id} not found",
        )

    # Verify ownership
    if cached.get("user_id") != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this interaction",
        )

    # Check status
    current_status = cached.get("status")
    if current_status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel interaction in '{current_status}' status",
        )

    # Cancel via orchestrator
    cancelled = await orchestrator.cancel(interaction_id)

    if cancelled:
        # Update status
        await redis.hset(f"interaction:{interaction_id}", "status", "cancelled")

        # Update database
        try:
            await db.execute(
                """
                UPDATE research_sessions
                SET status = 'cancelled', updated_at = %(updated_at)s
                WHERE id = %(id)s
                """,
                {"id": interaction_id, "updated_at": datetime.now(timezone.utc)},
            )
        except Exception as e:
            logger.warning(f"Failed to update database for cancellation: {e}")

        # Publish cancellation event
        cancel_event = StreamEvent(
            event_type=StreamEventType.INTERACTION_CANCELLED,
            data={"id": interaction_id, "reason": "user_requested"},
        )
        await redis.publish(f"events:{interaction_id}", cancel_event.model_dump_json())

    logger.info(f"Cancelled interaction {interaction_id}")

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# =============================================================================
# Interaction List Endpoint
# =============================================================================


@router.get(
    "/interactions",
    response_model=InteractionListResponse,
    summary="List interactions",
    description="List user's research interactions with pagination",
)
async def list_interactions(
    db: DatabaseDep,
    user: CurrentUserDep,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    status_filter: Annotated[str | None, Query(alias="status", description="Filter by status")] = None,
    _rate_limit: Annotated[None, Depends(rate_limit_standard)] = None,
) -> InteractionListResponse:
    """List user's research interactions.

    Supports pagination and optional status filtering.
    """
    offset = (page - 1) * page_size

    # Build query
    query = """
        SELECT id, status, query, error, created_at, updated_at
        FROM research_sessions
        WHERE user_id = %(user_id)s
    """
    params: dict[str, Any] = {"user_id": user.id}

    if status_filter:
        query += " AND status = %(status)s"
        params["status"] = status_filter

    # Count total
    count_query = query.replace(
        "SELECT id, status, query, error, created_at, updated_at",
        "SELECT COUNT(*)",
    )
    count_result = await db.execute(count_query, params)
    count_row = await count_result.fetchone()
    total = count_row["count"] if count_row else 0

    # Fetch page
    query += " ORDER BY created_at DESC LIMIT %(limit)s OFFSET %(offset)s"
    params["limit"] = page_size
    params["offset"] = offset

    result = await db.execute(query, params)
    rows = await result.fetchall()

    interactions = [
        InteractionResponse(
            id=row["id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row.get("updated_at"),
            query=row.get("query"),
            error=row.get("error"),
        )
        for row in rows
    ]

    return InteractionListResponse(
        interactions=interactions,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(interactions)) < total,
    )


# =============================================================================
# Resume Interaction Endpoint
# =============================================================================


@router.post(
    "/interactions/{interaction_id}/resume",
    response_model=InteractionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resume interaction",
    description="Resume a paused or interrupted interaction from checkpoint",
)
async def resume_interaction(
    interaction_id: Annotated[str, Path(description="Interaction ID", pattern="^int_[a-f0-9]{16}$")],
    checkpoint_id: Annotated[str | None, Query(description="Checkpoint ID to resume from")] = None,
    background_tasks: BackgroundTasks = None,
    db: DatabaseDep = None,
    redis: RedisDep = None,
    orchestrator: OrchestratorDep = None,
    user: CurrentUserDep = None,
    _rate_limit: Annotated[None, Depends(rate_limit_heavy)] = None,
) -> InteractionResponse:
    """Resume a research interaction from a checkpoint.

    If no checkpoint_id is provided, resumes from the latest checkpoint.
    """
    # Verify interaction exists
    cached = await redis.hgetall(f"interaction:{interaction_id}")

    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interaction {interaction_id} not found",
        )

    # Verify ownership
    if cached.get("user_id") != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this interaction",
        )

    # Check status allows resume
    current_status = cached.get("status")
    if current_status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Interaction is already running",
        )

    if current_status == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot resume completed interaction",
        )

    now = datetime.now(timezone.utc)

    # Update status
    await redis.hset(f"interaction:{interaction_id}", "status", "resuming")

    # Queue background processing
    async def _resume_task():
        try:
            await redis.hset(f"interaction:{interaction_id}", "status", "running")

            async for event in orchestrator.resume(interaction_id, checkpoint_id or "latest"):
                if isinstance(event, dict):
                    event_data = StreamEvent(
                        event_type=event.get("event_type", "content.delta"),
                        data=event.get("data", event),
                        checkpoint_id=event.get("checkpoint_id"),
                    )
                else:
                    event_data = event

                await redis.publish(
                    f"events:{interaction_id}",
                    event_data.model_dump_json(),
                )

            await redis.hset(f"interaction:{interaction_id}", "status", "completed")
            complete_event = create_complete_event(interaction_id, "completed")
            await redis.publish(f"events:{interaction_id}", complete_event.model_dump_json())

        except Exception as e:
            logger.error(f"Resume failed for {interaction_id}: {e}")
            await redis.hset(f"interaction:{interaction_id}", "status", "failed")
            error_event = create_error_event(str(e), type(e).__name__, interaction_id)
            await redis.publish(f"events:{interaction_id}", error_event.model_dump_json())

    background_tasks.add_task(_resume_task)

    return InteractionResponse(
        id=interaction_id,
        status="resuming",
        created_at=datetime.fromisoformat(cached["created_at"]) if "created_at" in cached else now,
        updated_at=now,
        query=cached.get("query"),
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "router",
    "SteerabilityConfig",
    "ResearchConfig",
    "ResearchRequest",
    "InteractionResponse",
    "InteractionListResponse",
    "HealthResponse",
    "ErrorResponse",
]
