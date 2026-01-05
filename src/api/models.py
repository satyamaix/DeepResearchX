"""
Pydantic Models for DRX API Request/Response Handling.

These models are used exclusively for FastAPI request validation,
response serialization, and OpenAPI schema generation.

NOTE: These are separate from the TypedDict state definitions in
orchestrator/state.py which are used for LangGraph internal state.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


# =============================================================================
# Enums for API
# =============================================================================


class ToneEnum(str, Enum):
    """Output tone/style options."""

    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    CASUAL = "casual"


class FormatEnum(str, Enum):
    """Output format options."""

    MARKDOWN = "markdown"
    MARKDOWN_TABLE = "markdown_table"
    JSON = "json"


class ResearchStatus(str, Enum):
    """Research session status."""

    PENDING = "pending"
    PLANNING = "planning"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    CRITIQUING = "critiquing"
    REPORTING = "reporting"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StreamEventType(str, Enum):
    """Types of SSE stream events."""

    # Lifecycle events
    SESSION_STARTED = "session_started"
    SESSION_COMPLETE = "session_complete"
    SESSION_FAILED = "session_failed"

    # Phase transition events
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETE = "phase_complete"

    # Task events
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"

    # Content events
    FINDING_ADDED = "finding_added"
    CITATION_ADDED = "citation_added"
    SYNTHESIS_UPDATE = "synthesis_update"
    GAP_IDENTIFIED = "gap_identified"

    # Agent events
    AGENT_THINKING = "agent_thinking"
    AGENT_ACTION = "agent_action"
    AGENT_MESSAGE = "agent_message"

    # Token tracking
    TOKEN_UPDATE = "token_update"

    # Checkpoint events
    CHECKPOINT_SAVED = "checkpoint_saved"

    # Error events
    ERROR = "error"
    WARNING = "warning"


# =============================================================================
# Configuration Models
# =============================================================================


class SteerabilityConfig(BaseModel):
    """
    User-configurable parameters for steering research behavior and output.

    These settings influence how agents conduct research and format results.
    """

    model_config = ConfigDict(use_enum_values=True)

    tone: ToneEnum = Field(
        default=ToneEnum.TECHNICAL,
        description="Output tone/style for the final report",
    )

    format: FormatEnum = Field(
        default=FormatEnum.MARKDOWN,
        description="Output format preference",
    )

    max_sources: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of sources to include in the report",
    )

    focus_areas: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Specific areas to prioritize during research",
    )

    exclude_topics: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Topics or sources to exclude from research",
    )

    preferred_domains: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Preferred source domains (e.g., 'arxiv.org', 'nature.com')",
    )

    language: str = Field(
        default="en",
        min_length=2,
        max_length=5,
        description="ISO 639-1 language code for output",
    )

    custom_instructions: str | None = Field(
        default=None,
        max_length=2000,
        description="Additional custom instructions for research behavior",
    )


class ResearchConfig(BaseModel):
    """
    Technical configuration for the research session.

    Controls iteration limits, token budgets, and model behavior.
    """

    thinking_summaries: Literal["auto", "always", "never"] = Field(
        default="auto",
        description="When to include agent thinking summaries in responses",
    )

    max_iterations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum research iterations before completion",
    )

    token_budget: int = Field(
        default=500000,
        ge=10000,
        le=2000000,
        description="Maximum tokens to consume for this session",
    )

    timeout_seconds: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Maximum time allowed for research session",
    )

    enable_critic: bool = Field(
        default=True,
        description="Enable critic agent for quality evaluation",
    )

    min_coverage_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum coverage score to consider research complete",
    )


# =============================================================================
# Request Models
# =============================================================================


class ResearchRequest(BaseModel):
    """
    API request model for initiating a new research session.

    This is the primary input model for the /research endpoint.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input": "What are the latest advancements in quantum computing for drug discovery?",
                "steerability": {
                    "tone": "technical",
                    "format": "markdown",
                    "max_sources": 15,
                    "focus_areas": ["pharmaceutical applications", "recent breakthroughs"],
                },
                "config": {
                    "max_iterations": 3,
                    "token_budget": 300000,
                },
            }
        }
    )

    input: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The research query or question to investigate",
    )

    steerability: SteerabilityConfig = Field(
        default_factory=SteerabilityConfig,
        description="Parameters for steering research behavior",
    )

    config: ResearchConfig = Field(
        default_factory=ResearchConfig,
        description="Technical configuration for the session",
    )

    thread_id: str | None = Field(
        default=None,
        description="Optional thread ID to resume a previous session",
    )

    @field_validator("input")
    @classmethod
    def validate_input_not_empty(cls, v: str) -> str:
        """Ensure input contains meaningful content."""
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError("Research query must be at least 10 characters")
        return stripped


class ResumeRequest(BaseModel):
    """Request model for resuming a paused research session."""

    thread_id: str = Field(
        ...,
        description="Thread ID of the session to resume",
    )

    additional_input: str | None = Field(
        default=None,
        max_length=2000,
        description="Additional context or refinement for the research",
    )


class CancelRequest(BaseModel):
    """Request model for cancelling an active research session."""

    thread_id: str = Field(
        ...,
        description="Thread ID of the session to cancel",
    )

    reason: str | None = Field(
        default=None,
        max_length=500,
        description="Optional reason for cancellation",
    )


class FeedbackRequest(BaseModel):
    """Request model for providing feedback on research results."""

    thread_id: str = Field(
        ...,
        description="Thread ID of the completed session",
    )

    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1-5",
    )

    feedback_text: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional detailed feedback",
    )

    helpful_findings: list[str] | None = Field(
        default=None,
        description="IDs of findings that were particularly helpful",
    )

    unhelpful_findings: list[str] | None = Field(
        default=None,
        description="IDs of findings that were not helpful",
    )


# =============================================================================
# Response Models
# =============================================================================


class InteractionResponse(BaseModel):
    """
    Immediate response when a research session is initiated.

    Returned before streaming begins to provide the session ID.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "created_at": "2025-01-05T10:30:00Z",
                "updated_at": "2025-01-05T10:30:00Z",
                "stream_url": "/api/v1/research/550e8400-e29b-41d4-a716-446655440000/stream",
            }
        }
    )

    id: str = Field(
        ...,
        description="Unique session/thread identifier",
    )

    status: ResearchStatus = Field(
        ...,
        description="Current status of the research session",
    )

    created_at: datetime = Field(
        ...,
        description="Timestamp when session was created",
    )

    updated_at: datetime = Field(
        ...,
        description="Timestamp of last status update",
    )

    stream_url: str | None = Field(
        default=None,
        description="URL for SSE stream to receive updates",
    )

    estimated_duration_seconds: int | None = Field(
        default=None,
        description="Estimated time to completion in seconds",
    )


class CitationResponse(BaseModel):
    """Citation information in API responses."""

    id: str
    url: str
    title: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    domain: str
    retrieved_at: datetime


class FindingResponse(BaseModel):
    """Research finding in API responses."""

    id: str
    claim: str
    evidence: str
    source_urls: list[str]
    citation_ids: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    tags: list[str]
    verified: bool


class QualityMetricsResponse(BaseModel):
    """Quality metrics in API responses."""

    coverage_score: float = Field(ge=0.0, le=1.0)
    avg_confidence: float = Field(ge=0.0, le=1.0)
    verified_findings: int = Field(ge=0)
    total_findings: int = Field(ge=0)
    unique_sources: int = Field(ge=0)
    citation_density: float = Field(ge=0.0)
    consistency_score: float = Field(ge=0.0, le=1.0)


class TokenUsageResponse(BaseModel):
    """Token usage information in API responses."""

    budget: int
    used: int
    remaining: int
    percentage_used: float = Field(ge=0.0, le=100.0)


class ResearchMetadata(BaseModel):
    """Metadata about the research session."""

    session_id: str
    user_query: str
    iterations_completed: int
    duration_seconds: float
    token_usage: TokenUsageResponse
    quality_metrics: QualityMetricsResponse | None


class ResearchResult(BaseModel):
    """
    Complete research result returned after session completion.

    This is the primary output model containing the full report and all supporting data.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "complete",
                "report": "# Research Report\n\n## Executive Summary\n...",
                "citations": [],
                "findings": [],
                "metadata": {
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "user_query": "What are the latest advancements...",
                    "iterations_completed": 3,
                    "duration_seconds": 45.2,
                    "token_usage": {
                        "budget": 300000,
                        "used": 125000,
                        "remaining": 175000,
                        "percentage_used": 41.67,
                    },
                    "quality_metrics": None,
                },
            }
        }
    )

    id: str = Field(
        ...,
        description="Unique session identifier",
    )

    status: ResearchStatus = Field(
        ...,
        description="Final status of the research session",
    )

    report: str | None = Field(
        default=None,
        description="The final formatted research report",
    )

    citations: list[CitationResponse] = Field(
        default_factory=list,
        description="All citations used in the report",
    )

    findings: list[FindingResponse] = Field(
        default_factory=list,
        description="Individual research findings",
    )

    metadata: ResearchMetadata = Field(
        ...,
        description="Session metadata and metrics",
    )

    error: str | None = Field(
        default=None,
        description="Error message if session failed",
    )


# =============================================================================
# Streaming Models
# =============================================================================


class StreamEvent(BaseModel):
    """
    Server-Sent Event (SSE) model for real-time updates.

    Used to stream progress updates to clients during research.
    """

    model_config = ConfigDict(use_enum_values=True)

    event_type: StreamEventType = Field(
        ...,
        description="Type of event being streamed",
    )

    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data payload",
    )

    checkpoint_id: str | None = Field(
        default=None,
        description="Associated checkpoint ID if applicable",
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this event occurred",
    )

    sequence: int = Field(
        default=0,
        ge=0,
        description="Event sequence number for ordering",
    )

    def to_sse(self) -> str:
        """Format as SSE message."""
        import json

        data_json = json.dumps(
            {
                "event_type": self.event_type,
                "data": self.data,
                "checkpoint_id": self.checkpoint_id,
                "timestamp": self.timestamp.isoformat(),
                "sequence": self.sequence,
            }
        )
        return f"event: {self.event_type}\ndata: {data_json}\n\n"


class StreamEventData:
    """Factory for creating typed stream event data payloads."""

    @staticmethod
    def session_started(session_id: str, query: str) -> dict[str, Any]:
        return {"session_id": session_id, "query": query}

    @staticmethod
    def phase_started(phase: str, iteration: int) -> dict[str, Any]:
        return {"phase": phase, "iteration": iteration}

    @staticmethod
    def task_progress(
        task_id: str, agent: str, progress: float, message: str
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "agent": agent,
            "progress": progress,
            "message": message,
        }

    @staticmethod
    def finding_added(finding: FindingResponse) -> dict[str, Any]:
        return finding.model_dump()

    @staticmethod
    def citation_added(citation: CitationResponse) -> dict[str, Any]:
        return citation.model_dump()

    @staticmethod
    def token_update(used: int, remaining: int, budget: int) -> dict[str, Any]:
        return {
            "used": used,
            "remaining": remaining,
            "budget": budget,
            "percentage": round((used / budget) * 100, 2) if budget > 0 else 0,
        }

    @staticmethod
    def error(message: str, code: str | None = None) -> dict[str, Any]:
        return {"message": message, "code": code}


# =============================================================================
# List/Query Response Models
# =============================================================================


class SessionListItem(BaseModel):
    """Summary item for session listing."""

    id: str
    query_preview: str = Field(max_length=200)
    status: ResearchStatus
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    """Paginated list of research sessions."""

    items: list[SessionListItem]
    total: int
    page: int
    page_size: int
    has_more: bool


class HealthResponse(BaseModel):
    """API health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime
    components: dict[str, Literal["up", "down", "unknown"]]


# =============================================================================
# Error Response Models
# =============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str
    message: str
    field: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Standard API error response."""

    error: str
    message: str
    details: list[ErrorDetail] | None = None
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    # Enums
    "ToneEnum",
    "FormatEnum",
    "ResearchStatus",
    "StreamEventType",
    # Config Models
    "SteerabilityConfig",
    "ResearchConfig",
    # Request Models
    "ResearchRequest",
    "ResumeRequest",
    "CancelRequest",
    "FeedbackRequest",
    # Response Models
    "InteractionResponse",
    "CitationResponse",
    "FindingResponse",
    "QualityMetricsResponse",
    "TokenUsageResponse",
    "ResearchMetadata",
    "ResearchResult",
    # Streaming
    "StreamEvent",
    "StreamEventData",
    # List/Query
    "SessionListItem",
    "SessionListResponse",
    "HealthResponse",
    # Errors
    "ErrorDetail",
    "ErrorResponse",
]
