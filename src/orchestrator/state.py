"""
LangGraph State Definitions for DRX Deep Research System.

CRITICAL: Uses TypedDict (NOT Pydantic BaseModel) for LangGraph state
due to serialization compatibility with AsyncPostgresSaver checkpointing.

This module defines the core state types that flow through the LangGraph
workflow, enabling proper state management, checkpointing, and resumption.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# =============================================================================
# Agent Type Definitions
# =============================================================================

AgentType = Literal[
    "planner",
    "searcher",
    "reader",
    "synthesizer",
    "critic",
    "reporter",
]

TaskStatus = Literal["pending", "running", "completed", "failed"]

ToneType = Literal["executive", "technical", "casual"]

FormatType = Literal["markdown", "markdown_table", "json"]


# =============================================================================
# SubTask TypedDict
# =============================================================================


class SubTask(TypedDict):
    """
    Represents a single unit of work in the research DAG.

    Each subtask is executed by a specific agent and may depend on
    outputs from other subtasks, forming a directed acyclic graph.
    """

    # Unique identifier for this subtask
    id: str

    # Human-readable description of what this task accomplishes
    description: str

    # Which agent type should execute this task
    agent_type: AgentType

    # List of subtask IDs that must complete before this task can run
    dependencies: list[str]

    # Current execution status
    status: TaskStatus

    # Input data for the agent (query context, parameters, etc.)
    inputs: dict[str, Any]

    # Output data produced by the agent (None until completed)
    outputs: dict[str, Any] | None

    # Quality score assigned by critic agent (0.0-1.0, None if not evaluated)
    quality_score: float | None

    # Timestamp when task started execution
    started_at: str | None

    # Timestamp when task completed
    completed_at: str | None

    # Error message if task failed
    error: str | None


# =============================================================================
# ResearchPlan TypedDict
# =============================================================================


class ResearchPlan(TypedDict):
    """
    The DAG-structured research plan created by the planner agent.

    Contains all subtasks organized in a dependency graph, along with
    metadata for iterative refinement based on gap analysis.
    """

    # All subtasks in the research plan (DAG nodes)
    dag_nodes: list[SubTask]

    # Current iteration number (1-indexed)
    current_iteration: int

    # Maximum allowed iterations before forcing completion
    max_iterations: int

    # Coverage score from critic (0.0-1.0) measuring query coverage
    coverage_score: float | None

    # Timestamp when plan was created
    created_at: str

    # Timestamp when plan was last modified
    updated_at: str

    # Original decomposition of the query into sub-questions
    sub_questions: list[str]


# =============================================================================
# CitationRecord TypedDict
# =============================================================================


class CitationRecord(TypedDict):
    """
    A tracked citation from retrieved sources.

    Provides full provenance for facts included in the final report.
    """

    # Unique identifier for this citation
    id: str

    # Source URL
    url: str

    # Page/document title
    title: str

    # Relevant text snippet from the source
    snippet: str

    # Relevance score (0.0-1.0) computed during retrieval
    relevance_score: float

    # ISO 8601 timestamp when this was retrieved
    retrieved_at: str

    # Domain/hostname for display grouping
    domain: str

    # Agent that retrieved this citation
    retrieved_by: AgentType

    # Whether this citation was actually used in the final report
    used_in_report: bool


# =============================================================================
# Finding TypedDict
# =============================================================================


class Finding(TypedDict):
    """
    A research finding synthesized from one or more sources.

    Represents a discrete piece of knowledge that contributes to
    answering the user's query.
    """

    # Unique identifier for this finding
    id: str

    # The factual claim being made
    claim: str

    # Supporting evidence text
    evidence: str

    # URLs of sources supporting this claim
    source_urls: list[str]

    # Citation IDs linked to this finding
    citation_ids: list[str]

    # Confidence score (0.0-1.0) in this finding
    confidence_score: float

    # Which agent produced this finding
    agent_source: AgentType

    # Tags/categories for this finding
    tags: list[str]

    # Whether this finding was verified by critic
    verified: bool

    # Timestamp when finding was created
    created_at: str


# =============================================================================
# Steerability Configuration TypedDict
# =============================================================================


class SteerabilityParams(TypedDict):
    """
    User-specified parameters for steering research output.

    These parameters influence how agents conduct research and
    how the final report is formatted.
    """

    # Output tone/style
    tone: ToneType

    # Output format preference
    format: FormatType

    # Maximum number of sources to include
    max_sources: int

    # Focus areas to prioritize (optional)
    focus_areas: list[str]

    # Topics/sources to exclude (optional)
    exclude_topics: list[str]

    # Preferred source domains (optional)
    preferred_domains: list[str]

    # Language for output (ISO 639-1 code)
    language: str

    # Custom instructions from user
    custom_instructions: str | None


# =============================================================================
# Quality Metrics TypedDict
# =============================================================================


class QualityMetrics(TypedDict):
    """
    Aggregated quality metrics for the research session.

    Updated by the critic agent after each iteration.
    """

    # Overall coverage of the query (0.0-1.0)
    coverage_score: float

    # Average confidence across findings
    avg_confidence: float

    # Number of verified findings
    verified_findings: int

    # Total findings count
    total_findings: int

    # Number of unique sources used
    unique_sources: int

    # Citation density (citations per 100 words)
    citation_density: float

    # Factual consistency score
    consistency_score: float

    # Last update timestamp
    updated_at: str


# =============================================================================
# Agent State (MAIN WORKFLOW STATE)
# =============================================================================


class AgentState(TypedDict):
    """
    The primary state object for the LangGraph research workflow.

    This TypedDict is the single source of truth passed between all
    nodes in the workflow graph. It accumulates results and enables
    checkpointing/resumption via AsyncPostgresSaver.

    IMPORTANT: This must be TypedDict (not Pydantic) for LangGraph
    serialization compatibility with PostgreSQL checkpointing.
    """

    # =========================================================================
    # Message History (with LangGraph reducer)
    # =========================================================================

    # Accumulated messages with add_messages reducer for proper merging
    messages: Annotated[list[AnyMessage], add_messages]

    # =========================================================================
    # Session Identification
    # =========================================================================

    # Unique session/thread identifier
    session_id: str

    # User's original research query
    user_query: str

    # ISO 8601 timestamp when session started
    started_at: str

    # =========================================================================
    # Steerability Configuration
    # =========================================================================

    # User-specified research parameters
    steerability: SteerabilityParams

    # =========================================================================
    # Research Plan & Progress
    # =========================================================================

    # The current research plan (DAG of subtasks)
    plan: ResearchPlan | None

    # Accumulated research findings
    findings: list[Finding]

    # All retrieved citations
    citations: list[CitationRecord]

    # =========================================================================
    # Synthesis & Output
    # =========================================================================

    # Intermediate synthesis text (updated each iteration)
    synthesis: str

    # Final formatted report (None until reporter completes)
    final_report: str | None

    # =========================================================================
    # Iteration Control
    # =========================================================================

    # Current iteration number
    iteration_count: int

    # Maximum allowed iterations
    max_iterations: int

    # =========================================================================
    # Gap Analysis & Refinement
    # =========================================================================

    # Identified knowledge gaps requiring additional research
    gaps: list[str]

    # Aggregated quality metrics
    quality_metrics: QualityMetrics | None

    # =========================================================================
    # Policy & Safety
    # =========================================================================

    # Any policy violations detected
    policy_violations: list[str]

    # Whether research should be blocked due to policy
    blocked: bool

    # =========================================================================
    # Token Budget Management
    # =========================================================================

    # Total token budget for this session
    token_budget: int

    # Tokens consumed so far
    tokens_used: int

    # Estimated tokens remaining
    tokens_remaining: int

    # =========================================================================
    # Workflow Control
    # =========================================================================

    # Current workflow phase
    current_phase: Literal[
        "planning",
        "researching",
        "synthesizing",
        "critiquing",
        "reporting",
        "complete",
        "failed",
    ]

    # Next node to execute (for conditional routing)
    next_node: str | None

    # Whether workflow should terminate
    should_terminate: bool

    # Error message if workflow failed
    error: str | None


# =============================================================================
# State Factory Functions
# =============================================================================


def create_initial_state(
    session_id: str,
    user_query: str,
    steerability: SteerabilityParams | None = None,
    token_budget: int = 500000,
    max_iterations: int = 5,
) -> AgentState:
    """
    Create a new AgentState with sensible defaults.

    Args:
        session_id: Unique identifier for this research session
        user_query: The user's research question
        steerability: Optional steerability parameters
        token_budget: Maximum tokens to consume
        max_iterations: Maximum research iterations

    Returns:
        Initialized AgentState ready for workflow execution
    """
    now = datetime.utcnow().isoformat() + "Z"

    default_steerability: SteerabilityParams = {
        "tone": "technical",
        "format": "markdown",
        "max_sources": 20,
        "focus_areas": [],
        "exclude_topics": [],
        "preferred_domains": [],
        "language": "en",
        "custom_instructions": None,
    }

    return AgentState(
        messages=[],
        session_id=session_id,
        user_query=user_query,
        started_at=now,
        steerability=steerability or default_steerability,
        plan=None,
        findings=[],
        citations=[],
        synthesis="",
        final_report=None,
        iteration_count=0,
        max_iterations=max_iterations,
        gaps=[],
        quality_metrics=None,
        policy_violations=[],
        blocked=False,
        token_budget=token_budget,
        tokens_used=0,
        tokens_remaining=token_budget,
        current_phase="planning",
        next_node=None,
        should_terminate=False,
        error=None,
    )


def create_empty_plan(max_iterations: int = 5) -> ResearchPlan:
    """
    Create an empty research plan structure.

    Args:
        max_iterations: Maximum iterations for the plan

    Returns:
        Empty ResearchPlan ready to be populated by planner
    """
    now = datetime.utcnow().isoformat() + "Z"

    return ResearchPlan(
        dag_nodes=[],
        current_iteration=1,
        max_iterations=max_iterations,
        coverage_score=None,
        created_at=now,
        updated_at=now,
        sub_questions=[],
    )


def create_subtask(
    task_id: str,
    description: str,
    agent_type: AgentType,
    dependencies: list[str] | None = None,
    inputs: dict[str, Any] | None = None,
) -> SubTask:
    """
    Create a new subtask for the research plan.

    Args:
        task_id: Unique identifier for this task
        description: Human-readable task description
        agent_type: Agent that will execute this task
        dependencies: IDs of tasks that must complete first
        inputs: Input data for the task

    Returns:
        New SubTask in pending status
    """
    return SubTask(
        id=task_id,
        description=description,
        agent_type=agent_type,
        dependencies=dependencies or [],
        status="pending",
        inputs=inputs or {},
        outputs=None,
        quality_score=None,
        started_at=None,
        completed_at=None,
        error=None,
    )


def create_finding(
    finding_id: str,
    claim: str,
    evidence: str,
    source_urls: list[str],
    citation_ids: list[str],
    agent_source: AgentType,
    confidence_score: float = 0.5,
    tags: list[str] | None = None,
) -> Finding:
    """
    Create a new research finding.

    Args:
        finding_id: Unique identifier
        claim: The factual claim
        evidence: Supporting evidence text
        source_urls: URLs of supporting sources
        citation_ids: IDs of linked citations
        agent_source: Agent that produced this finding
        confidence_score: Confidence in the finding (0.0-1.0)
        tags: Optional categorization tags

    Returns:
        New Finding ready to be added to state
    """
    now = datetime.utcnow().isoformat() + "Z"

    return Finding(
        id=finding_id,
        claim=claim,
        evidence=evidence,
        source_urls=source_urls,
        citation_ids=citation_ids,
        confidence_score=confidence_score,
        agent_source=agent_source,
        tags=tags or [],
        verified=False,
        created_at=now,
    )


def create_citation(
    citation_id: str,
    url: str,
    title: str,
    snippet: str,
    relevance_score: float,
    retrieved_by: AgentType,
) -> CitationRecord:
    """
    Create a new citation record.

    Args:
        citation_id: Unique identifier
        url: Source URL
        title: Page/document title
        snippet: Relevant text snippet
        relevance_score: Relevance to query (0.0-1.0)
        retrieved_by: Agent that retrieved this

    Returns:
        New CitationRecord
    """
    from urllib.parse import urlparse

    now = datetime.utcnow().isoformat() + "Z"
    domain = urlparse(url).netloc

    return CitationRecord(
        id=citation_id,
        url=url,
        title=title,
        snippet=snippet,
        relevance_score=relevance_score,
        retrieved_at=now,
        domain=domain,
        retrieved_by=retrieved_by,
        used_in_report=False,
    )


# =============================================================================
# State Update Helpers
# =============================================================================


def update_tokens_used(state: AgentState, tokens: int) -> dict[str, int]:
    """
    Calculate updated token usage values.

    Args:
        state: Current agent state
        tokens: Number of tokens to add to usage

    Returns:
        Dict with updated tokens_used and tokens_remaining
    """
    new_used = state["tokens_used"] + tokens
    new_remaining = max(0, state["token_budget"] - new_used)

    return {
        "tokens_used": new_used,
        "tokens_remaining": new_remaining,
    }


def get_pending_tasks(plan: ResearchPlan) -> list[SubTask]:
    """
    Get all pending tasks whose dependencies are satisfied.

    Args:
        plan: Current research plan

    Returns:
        List of tasks ready for execution
    """
    completed_ids = {
        task["id"] for task in plan["dag_nodes"] if task["status"] == "completed"
    }

    ready_tasks = []
    for task in plan["dag_nodes"]:
        if task["status"] == "pending":
            deps_satisfied = all(dep in completed_ids for dep in task["dependencies"])
            if deps_satisfied:
                ready_tasks.append(task)

    return ready_tasks


def get_task_by_id(plan: ResearchPlan, task_id: str) -> SubTask | None:
    """
    Find a task by its ID.

    Args:
        plan: Research plan to search
        task_id: ID of task to find

    Returns:
        SubTask if found, None otherwise
    """
    for task in plan["dag_nodes"]:
        if task["id"] == task_id:
            return task
    return None


def is_plan_complete(plan: ResearchPlan) -> bool:
    """
    Check if all tasks in the plan are completed.

    Args:
        plan: Research plan to check

    Returns:
        True if all tasks completed, False otherwise
    """
    return all(task["status"] == "completed" for task in plan["dag_nodes"])


def has_failed_tasks(plan: ResearchPlan) -> bool:
    """
    Check if any tasks have failed.

    Args:
        plan: Research plan to check

    Returns:
        True if any tasks failed, False otherwise
    """
    return any(task["status"] == "failed" for task in plan["dag_nodes"])


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    # Type aliases
    "AgentType",
    "TaskStatus",
    "ToneType",
    "FormatType",
    # TypedDicts
    "SubTask",
    "ResearchPlan",
    "CitationRecord",
    "Finding",
    "SteerabilityParams",
    "QualityMetrics",
    "AgentState",
    # Factory functions
    "create_initial_state",
    "create_empty_plan",
    "create_subtask",
    "create_finding",
    "create_citation",
    # Helpers
    "update_tokens_used",
    "get_pending_tasks",
    "get_task_by_id",
    "is_plan_complete",
    "has_failed_tasks",
]
