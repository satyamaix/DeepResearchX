"""API Route Definitions for DRX Deep Research.

Provides REST endpoints for creating, managing, and streaming
research interactions with proper validation and error handling.
"""

from __future__ import annotations

import asyncio
import json
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
from pydantic import BaseModel, Field, field_validator, model_validator

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
        description="Research query or question. Can also be provided as 'query' for backward compatibility.",
    )
    steerability: SteerabilityConfig | None = Field(
        default=None,
        description="Steerability parameters",
    )
    config: ResearchConfig | None = Field(
        default=None,
        description="Research configuration",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_query_as_input(cls, data: Any) -> Any:
        """Accept 'query' as an alias for 'input' for backward compatibility."""
        if isinstance(data, dict):
            # If 'query' is provided but 'input' is not, use 'query' as 'input'
            if "query" in data and "input" not in data:
                data["input"] = data.pop("query")
        return data

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
    # Generate unique interaction ID (UUID format for PostgreSQL compatibility)
    interaction_id = str(uuid.uuid4())
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
                "status": "pending",
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
            "status": "pending",
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
            # Publish event to Redis channel - handle both dict and dataclass events
            if isinstance(event, dict):
                event_json = json.dumps(event, default=str)
            elif hasattr(event, "to_dict"):
                # Workflow StreamEvent dataclass
                event_json = json.dumps(event.to_dict(), default=str)
            elif hasattr(event, "model_dump_json"):
                # Pydantic StreamEvent
                event_json = event.model_dump_json()
            else:
                # Fallback
                event_json = json.dumps({"event_type": str(event)}, default=str)

            await redis.publish(
                f"events:{interaction_id}",
                event_json,
            )

        # Get final state and persist result
        final_state = await orchestrator.get_state(interaction_id)
        result_data = None
        if final_state:
            result_data = {
                "final_report": final_state.get("final_report"),
                "findings": final_state.get("findings", []),
                "citations": final_state.get("citations", []),
                "tokens_used": final_state.get("tokens_used", 0),
                "iteration_count": final_state.get("iteration_count", 0),
            }
            # Store result in Redis for quick access
            await redis.hset(
                f"interaction:{interaction_id}",
                "result",
                json.dumps(result_data, default=str),
            )
            logger.info(f"Stored result for interaction {interaction_id}")

        # Update status to completed
        await redis.hset(f"interaction:{interaction_id}", "status", "completed")

        # Publish completion event with result
        complete_event = create_complete_event(interaction_id, "completed")
        if result_data:
            complete_event.data["result"] = result_data
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
    interaction_id: Annotated[str, Path(description="Interaction ID", pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")],
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

        # Parse result if present
        result_json = cached.get("result")
        result = json.loads(result_json) if result_json else None

        return InteractionResponse(
            id=interaction_id,
            status=cached.get("status", "unknown"),
            created_at=datetime.fromisoformat(cached["created_at"]) if "created_at" in cached else datetime.now(timezone.utc),
            query=cached.get("query"),
            result=result,
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
    interaction_id: Annotated[str, Path(description="Interaction ID", pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")],
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
    interaction_id: Annotated[str, Path(description="Interaction ID", pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")],
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
    interaction_id: Annotated[str, Path(description="Interaction ID", pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")],
    checkpoint_id: Annotated[str | None, Query(description="Checkpoint ID to resume from")] = None,
    *,
    background_tasks: BackgroundTasks,
    db: DatabaseDep,
    redis: RedisDep,
    orchestrator: OrchestratorDep,
    user: CurrentUserDep,
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
# Feedback Endpoint (WP-3A: Dataset Flywheel)
# =============================================================================


class FeedbackRequest(BaseModel):
    """User feedback submission for research session quality."""

    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1-5 (1=poor, 5=excellent)",
    )
    comment: str | None = Field(
        default=None,
        max_length=5000,
        description="Optional detailed feedback comment",
    )
    labels: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Feedback labels (e.g., 'accurate', 'comprehensive', 'well-cited')",
    )


class FeedbackResponse(BaseModel):
    """Feedback submission response."""

    feedback_id: str = Field(..., description="Unique feedback identifier")
    session_id: str = Field(..., description="Associated session ID")
    status: str = Field(..., description="Submission status")


@router.post(
    "/sessions/{session_id}/feedback",
    response_model=FeedbackResponse,
    summary="Submit session feedback",
    description="Submit user feedback for a completed research session",
    responses={
        200: {"description": "Feedback submitted successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        409: {"model": ErrorResponse, "description": "Session not yet completed"},
    },
)
async def submit_feedback(
    session_id: Annotated[
        str,
        Path(
            description="Session ID",
            pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        ),
    ],
    request: FeedbackRequest,
    db: DatabaseDep,
    redis: RedisDep,
    user: CurrentUserDep,
    _rate_limit: Annotated[None, Depends(rate_limit_standard)] = None,
) -> FeedbackResponse:
    """Submit feedback for a research session.

    This endpoint enables the Dataset Flywheel by collecting user feedback
    on research quality for continuous improvement and fine-tuning.

    The feedback includes:
    - Rating (1-5 scale)
    - Optional text comment
    - Optional categorical labels

    Feedback is stored in Redis for fast access and optionally persisted
    to PostgreSQL for long-term storage and training data collection.
    """
    # Import here to avoid circular imports
    from ci.evaluation.feedback_store import FeedbackStore

    # Verify session exists
    cached = await redis.hgetall(f"interaction:{session_id}")

    if not cached:
        # Try database
        try:
            result = await db.execute(
                """
                SELECT id, status, user_id FROM research_sessions
                WHERE id = %(id)s
                """,
                {"id": session_id},
            )
            row = await result.fetchone()

            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session {session_id} not found",
                )

            session_user_id = row["user_id"]
            session_status = row["status"]
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to query session {session_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
    else:
        session_user_id = cached.get("user_id")
        session_status = cached.get("status")

    # Verify ownership (unless admin)
    if session_user_id != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session",
        )

    # Verify session is completed (optional: could allow feedback on any status)
    if session_status not in ("completed", "complete"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot submit feedback for session in '{session_status}' status",
        )

    # Create feedback store and submit
    feedback_store = FeedbackStore(redis=redis, db_pool=None)

    try:
        feedback_id = await feedback_store.submit_feedback(
            session_id=session_id,
            rating=request.rating,
            comment=request.comment,
            labels=request.labels,
            user_id=user.id,
            metadata={
                "submitted_via": "api",
                "user_agent": None,  # Could extract from request headers
            },
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to submit feedback for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback",
        )

    # Also update dataset collector if available
    try:
        from ci.evaluation.dataset_collector import DatasetCollector

        collector = DatasetCollector()
        collector.add_feedback(
            session_id=session_id,
            rating=request.rating,
            feedback=request.comment,
            labels=request.labels,
        )
    except Exception as e:
        # Log but don't fail - dataset collection is secondary
        logger.warning(f"Failed to update dataset collector: {e}")

    logger.info(
        f"Feedback {feedback_id} submitted for session {session_id}",
        extra={
            "feedback_id": feedback_id,
            "session_id": session_id,
            "rating": request.rating,
            "user_id": user.id,
        },
    )

    return FeedbackResponse(
        feedback_id=feedback_id,
        session_id=session_id,
        status="submitted",
    )


@router.get(
    "/sessions/{session_id}/feedback",
    response_model=list[dict[str, Any]],
    summary="Get session feedback",
    description="Retrieve all feedback for a research session",
    responses={
        200: {"description": "Feedback retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_session_feedback(
    session_id: Annotated[
        str,
        Path(
            description="Session ID",
            pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        ),
    ],
    redis: RedisDep,
    user: CurrentUserDep,
    _rate_limit: Annotated[None, Depends(rate_limit_standard)] = None,
) -> list[dict[str, Any]]:
    """Get all feedback submitted for a research session.

    Returns a list of feedback records including ratings, comments,
    and categorical labels.
    """
    from ci.evaluation.feedback_store import FeedbackStore

    # Verify session exists and user has access
    cached = await redis.hgetall(f"interaction:{session_id}")

    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Verify ownership (unless admin)
    session_user_id = cached.get("user_id")
    if session_user_id != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session",
        )

    # Get feedback from store
    feedback_store = FeedbackStore(redis=redis, db_pool=None)
    feedback_records = await feedback_store.get_feedback(session_id)

    return feedback_records


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
    "FeedbackRequest",
    "FeedbackResponse",
]
