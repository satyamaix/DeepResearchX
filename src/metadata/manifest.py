"""
Agent Manifest Pydantic Models and Validation.

This module provides strongly-typed Pydantic models for agent manifests,
matching the JSON Schema defined in schemas/agent_manifest.json.

Uses TypedDict for LangGraph compatibility where needed, and Pydantic
BaseModel for validation and serialization.

Part of the Agentic Metadata implementation (R10.8 from DRX.md spec).
"""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

from src.metadata.circuit_breaker import CircuitBreakerConfigDict


# =============================================================================
# Type Definitions
# =============================================================================

AgentTypeEnum = Literal[
    "planner",
    "searcher",
    "reader",
    "synthesizer",
    "critic",
    "reporter",
    "orchestrator",
    "reasoner",
    "writer",
]

HealthStatus = Literal["healthy", "degraded", "unhealthy", "unknown"]


# =============================================================================
# TypedDict Definitions (for LangGraph compatibility)
# =============================================================================


class RateLimitsDict(TypedDict, total=False):
    """Rate limiting configuration as TypedDict."""

    requests_per_minute: int
    tokens_per_minute: int


class HealthThresholdsDict(TypedDict, total=False):
    """Health monitoring thresholds as TypedDict."""

    max_error_rate: float
    max_latency_ms: int
    min_success_rate: float


# CircuitBreakerDict is an alias to the canonical CircuitBreakerConfigDict
# from circuit_breaker.py for backward compatibility
CircuitBreakerDict = CircuitBreakerConfigDict


class ModelConfigDict(TypedDict, total=False):
    """LLM model configuration as TypedDict."""

    model: str
    temperature: float
    max_tokens: int
    top_p: float


class AgentManifestDict(TypedDict, total=False):
    """
    Complete agent manifest as TypedDict.

    For use with LangGraph state when TypedDict is required.
    """

    id: str
    version: str
    agent_type: AgentTypeEnum
    display_name: str
    description: str
    capabilities: list[str]
    allowed_domains: list[str]
    blocked_domains: list[str]
    max_budget_usd: float
    rate_limits: RateLimitsDict
    health_thresholds: HealthThresholdsDict
    circuit_breaker: CircuitBreakerDict
    model_config: ModelConfigDict
    allowed_tools: list[str]
    tool_config: dict[str, Any]
    timeout_seconds: int
    max_concurrent_calls: int
    is_active: bool
    is_default: bool
    metadata: dict[str, Any]


# =============================================================================
# Pydantic Models
# =============================================================================


class RateLimits(BaseModel):
    """
    Rate limiting configuration for an agent.

    Controls the maximum throughput to prevent overloading
    downstream services and managing costs.
    """

    requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Maximum requests per minute",
    )
    tokens_per_minute: int = Field(
        default=100000,
        ge=1000,
        le=10000000,
        description="Maximum tokens per minute",
    )

    model_config = {"extra": "forbid"}


class HealthThresholds(BaseModel):
    """
    Health monitoring thresholds for an agent.

    Defines the boundaries for healthy operation. When exceeded,
    the agent's health status changes accordingly.
    """

    max_error_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable error rate (0.0-1.0)",
    )
    max_latency_ms: int = Field(
        default=30000,
        ge=100,
        le=600000,
        description="Maximum acceptable latency in milliseconds",
    )
    min_success_rate: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable success rate (0.0-1.0)",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_rate_consistency(self) -> "HealthThresholds":
        """Ensure error rate and success rate are consistent."""
        if self.max_error_rate + self.min_success_rate > 1.0:
            # Adjust to be consistent - success rate takes precedence
            self.max_error_rate = 1.0 - self.min_success_rate
        return self


class CircuitBreakerConfig(BaseModel):
    """
    Circuit breaker configuration for fault tolerance.

    Implements the circuit breaker pattern to prevent cascading
    failures when an agent or its dependencies are unhealthy.

    Note: This Pydantic model corresponds to CircuitBreakerConfigDict TypedDict.
    Default values are aligned with DEFAULT_CIRCUIT_BREAKER_CONFIG in
    src.metadata.circuit_breaker.
    """

    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of failures before circuit opens",
    )
    success_threshold: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of successes in half-open state before closing",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Seconds to wait before attempting recovery",
    )
    half_open_max_calls: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Maximum calls allowed in half-open state",
    )
    error_rate_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Error rate threshold (0.0-1.0) to trigger circuit open",
    )

    model_config = {"extra": "forbid"}


class ModelConfig(BaseModel):
    """
    LLM model configuration for an agent.

    Specifies which model to use and its generation parameters.
    """

    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model identifier to use",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens in response",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter",
    )

    model_config = {"extra": "allow"}  # Allow additional model-specific params


class AgentManifest(BaseModel):
    """
    Complete agent manifest with validation.

    This is the primary Pydantic model for agent configuration,
    providing full validation and serialization capabilities.

    Attributes:
        id: Unique agent identifier (format: agent_name_vN)
        version: Semantic version string
        agent_type: The role of the agent in the workflow
        capabilities: List of capabilities provided
        allowed_domains: Domains the agent can access
        blocked_domains: Domains the agent cannot access
        max_budget_usd: Maximum budget per session
        rate_limits: Rate limiting configuration
        health_thresholds: Health monitoring thresholds
        circuit_breaker: Circuit breaker configuration
        model_config_params: LLM model configuration
        allowed_tools: Tools the agent can use
        tool_config: Tool-specific configuration
        timeout_seconds: Default operation timeout
        max_concurrent_calls: Maximum concurrent API calls
        is_active: Whether agent is active
        is_default: Whether this is the default for its type
        metadata: Additional custom metadata
    """

    id: str = Field(
        ...,
        pattern=r"^[a-z_]+_v\d+$",
        description="Unique agent identifier (format: agent_name_vN)",
        examples=["planner_v1", "searcher_v2"],
    )
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic version string",
        examples=["1.0.0", "2.1.3"],
    )
    agent_type: AgentTypeEnum = Field(
        ...,
        description="The type/role of the agent in the research workflow",
    )
    display_name: str | None = Field(
        default=None,
        max_length=255,
        description="Human-readable display name",
    )
    description: str | None = Field(
        default=None,
        description="Detailed description of agent purpose",
    )
    capabilities: list[str] = Field(
        ...,
        min_length=1,
        description="List of capabilities this agent provides",
    )
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="Domains the agent is allowed to access",
    )
    blocked_domains: list[str] = Field(
        default_factory=list,
        description="Domains the agent is blocked from accessing",
    )
    max_budget_usd: float = Field(
        default=1.0,
        ge=0.0,
        le=1000.0,
        description="Maximum budget in USD per session",
    )
    rate_limits: RateLimits = Field(
        default_factory=RateLimits,
        description="Rate limiting configuration",
    )
    health_thresholds: HealthThresholds = Field(
        default_factory=HealthThresholds,
        description="Health monitoring thresholds",
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration",
    )
    # Named model_config_params to avoid conflict with Pydantic's model_config
    model_config_params: ModelConfig = Field(
        default_factory=ModelConfig,
        alias="model_config",
        description="LLM model configuration",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Tools the agent is permitted to use",
    )
    tool_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific configuration overrides",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Default timeout for operations",
    )
    max_concurrent_calls: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum concurrent API calls",
    )
    is_active: bool = Field(
        default=True,
        description="Whether agent is currently active",
    )
    is_default: bool = Field(
        default=False,
        description="Whether this is the default for its type",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata",
    )

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": "searcher_v1",
                    "version": "1.0.0",
                    "agent_type": "searcher",
                    "capabilities": ["web_search", "source_discovery"],
                    "max_budget_usd": 1.0,
                }
            ]
        },
    }

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Validate that ID matches the required pattern."""
        pattern = r"^[a-z_]+_v\d+$"
        if not re.match(pattern, v):
            raise ValueError(
                f"ID must match pattern '{pattern}' (e.g., 'searcher_v1')"
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate semantic version format."""
        pattern = r"^\d+\.\d+\.\d+$"
        if not re.match(pattern, v):
            raise ValueError(
                f"Version must match semantic version format (e.g., '1.0.0')"
            )
        return v

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities_not_empty(cls, v: list[str]) -> list[str]:
        """Ensure capabilities list is not empty."""
        if not v:
            raise ValueError("At least one capability is required")
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for cap in v:
            if cap not in seen:
                seen.add(cap)
                unique.append(cap)
        return unique

    @model_validator(mode="after")
    def validate_domain_lists(self) -> "AgentManifest":
        """Ensure allowed and blocked domains don't overlap."""
        if self.allowed_domains and self.blocked_domains:
            overlap = set(self.allowed_domains) & set(self.blocked_domains)
            if overlap:
                raise ValueError(
                    f"Domains cannot be both allowed and blocked: {overlap}"
                )
        return self

    def to_dict(self) -> AgentManifestDict:
        """
        Convert to TypedDict for LangGraph compatibility.

        Returns:
            AgentManifestDict representation of this manifest.
        """
        return AgentManifestDict(
            id=self.id,
            version=self.version,
            agent_type=self.agent_type,
            display_name=self.display_name or "",
            description=self.description or "",
            capabilities=self.capabilities,
            allowed_domains=self.allowed_domains,
            blocked_domains=self.blocked_domains,
            max_budget_usd=self.max_budget_usd,
            rate_limits=RateLimitsDict(
                requests_per_minute=self.rate_limits.requests_per_minute,
                tokens_per_minute=self.rate_limits.tokens_per_minute,
            ),
            health_thresholds=HealthThresholdsDict(
                max_error_rate=self.health_thresholds.max_error_rate,
                max_latency_ms=self.health_thresholds.max_latency_ms,
                min_success_rate=self.health_thresholds.min_success_rate,
            ),
            circuit_breaker=CircuitBreakerDict(
                failure_threshold=self.circuit_breaker.failure_threshold,
                success_threshold=self.circuit_breaker.success_threshold,
                timeout_seconds=self.circuit_breaker.timeout_seconds,
                half_open_max_calls=self.circuit_breaker.half_open_max_calls,
                error_rate_threshold=self.circuit_breaker.error_rate_threshold,
            ),
            model_config=ModelConfigDict(
                model=self.model_config_params.model,
                temperature=self.model_config_params.temperature,
                max_tokens=self.model_config_params.max_tokens,
                top_p=self.model_config_params.top_p,
            ),
            allowed_tools=self.allowed_tools,
            tool_config=self.tool_config,
            timeout_seconds=self.timeout_seconds,
            max_concurrent_calls=self.max_concurrent_calls,
            is_active=self.is_active,
            is_default=self.is_default,
            metadata=self.metadata,
        )


# =============================================================================
# Default Configurations Per Agent Type
# =============================================================================

_DEFAULT_AGENT_CONFIGS: dict[AgentTypeEnum, dict[str, Any]] = {
    "planner": {
        "display_name": "Research Planner",
        "description": "Decomposes research queries into executable plans",
        "capabilities": ["query_analysis", "plan_generation", "strategy_selection"],
        "allowed_tools": ["query_parse", "plan_create"],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.5,
            "max_tokens": 8192,
        },
    },
    "searcher": {
        "display_name": "Web Searcher",
        "description": "Executes web searches and retrieves relevant sources",
        "capabilities": ["web_search", "source_discovery", "query_expansion"],
        "allowed_tools": ["web_search", "url_fetch", "search_expand"],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.2,
            "max_tokens": 2048,
        },
    },
    "reader": {
        "display_name": "Content Reader",
        "description": "Extracts and processes content from web sources",
        "capabilities": ["content_extraction", "text_processing", "entity_recognition"],
        "allowed_tools": ["url_fetch", "html_parse", "pdf_extract"],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.1,
            "max_tokens": 16384,
        },
    },
    "synthesizer": {
        "display_name": "Knowledge Synthesizer",
        "description": "Combines information from multiple sources into coherent output",
        "capabilities": ["information_synthesis", "conflict_resolution", "knowledge_integration"],
        "allowed_tools": ["synthesize", "merge_findings"],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.5,
            "max_tokens": 8192,
        },
    },
    "critic": {
        "display_name": "Quality Critic",
        "description": "Reviews and critiques research output for quality",
        "capabilities": ["quality_assessment", "bias_detection", "completeness_check"],
        "allowed_tools": ["quality_score", "bias_detect"],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
            "max_tokens": 4096,
        },
    },
    "reporter": {
        "display_name": "Report Writer",
        "description": "Generates structured research reports and summaries",
        "capabilities": ["report_generation", "summarization", "citation_formatting"],
        "allowed_tools": ["report_create", "citation_format"],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.6,
            "max_tokens": 16384,
        },
    },
    "orchestrator": {
        "display_name": "Research Orchestrator",
        "description": "Coordinates research workflow and manages agent interactions",
        "capabilities": ["workflow_management", "task_delegation", "state_management"],
        "allowed_tools": ["dag_execute", "agent_invoke", "session_manage"],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
            "max_tokens": 4096,
        },
    },
    "reasoner": {
        "display_name": "Research Reasoner",
        "description": "Performs logical reasoning and analysis on gathered information",
        "capabilities": ["logical_reasoning", "fact_checking", "inference"],
        "allowed_tools": ["knowledge_query", "fact_verify"],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.4,
            "max_tokens": 8192,
        },
    },
    "writer": {
        "display_name": "Content Writer",
        "description": "Generates written content and documentation",
        "capabilities": ["content_generation", "editing", "formatting"],
        "allowed_tools": ["write", "format"],
        "model_config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.6,
            "max_tokens": 16384,
        },
    },
}


# =============================================================================
# Factory Functions
# =============================================================================


def get_default_manifest(agent_type: AgentTypeEnum) -> AgentManifest:
    """
    Create a default AgentManifest for a given agent type.

    Provides sensible defaults based on the agent's role in the
    research workflow.

    Args:
        agent_type: The type of agent to create a manifest for.

    Returns:
        AgentManifest with default configuration for the agent type.

    Raises:
        ValueError: If agent_type is not recognized.

    Example:
        >>> manifest = get_default_manifest("searcher")
        >>> print(manifest.id)
        'searcher_v1'
    """
    if agent_type not in _DEFAULT_AGENT_CONFIGS:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Valid types: {list(_DEFAULT_AGENT_CONFIGS.keys())}"
        )

    config = _DEFAULT_AGENT_CONFIGS[agent_type]

    return AgentManifest(
        id=f"{agent_type}_v1",
        version="1.0.0",
        agent_type=agent_type,
        display_name=config["display_name"],
        description=config["description"],
        capabilities=config["capabilities"],
        allowed_tools=config["allowed_tools"],
        model_config=ModelConfig(**config["model_config"]),
        is_default=True,
    )


def manifest_to_dict(manifest: AgentManifest) -> dict[str, Any]:
    """
    Convert an AgentManifest to a plain dictionary.

    Useful for serialization to JSON or database storage.

    Args:
        manifest: The AgentManifest to convert.

    Returns:
        Dictionary representation suitable for JSON serialization.

    Example:
        >>> manifest = get_default_manifest("planner")
        >>> data = manifest_to_dict(manifest)
        >>> print(data["agent_type"])
        'planner'
    """
    return manifest.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )


def validate_manifest(manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a manifest dictionary against the schema.

    Performs comprehensive validation including pattern matching,
    type checking, and cross-field validation.

    Args:
        manifest: Dictionary to validate as an AgentManifest.

    Returns:
        Tuple of (is_valid, error_messages).
        If valid, error_messages is an empty list.

    Example:
        >>> is_valid, errors = validate_manifest({"id": "bad id"})
        >>> print(is_valid)
        False
        >>> print(errors[0])
        "id: String should match pattern '^[a-z_]+_v\\d+$'"
    """
    errors: list[str] = []

    try:
        AgentManifest.model_validate(manifest)
        return True, []
    except Exception as e:
        # Parse Pydantic validation errors
        if hasattr(e, "errors"):
            for error in e.errors():
                field = ".".join(str(loc) for loc in error.get("loc", []))
                msg = error.get("msg", str(error))
                errors.append(f"{field}: {msg}")
        else:
            errors.append(str(e))

        return False, errors


async def load_manifest_from_registry(
    agent_id: str,
    db_pool: Any,
) -> AgentManifest:
    """
    Load an AgentManifest from the PostgreSQL agent_registry table.

    Fetches the agent configuration from the database and converts
    it to a validated AgentManifest instance.

    Args:
        agent_id: The unique identifier of the agent (e.g., 'searcher_v1').
        db_pool: asyncpg connection pool for database access.

    Returns:
        AgentManifest loaded and validated from the database.

    Raises:
        ValueError: If the agent is not found in the registry.
        ValidationError: If the database record fails validation.

    Example:
        >>> async with asyncpg.create_pool(dsn) as pool:
        ...     manifest = await load_manifest_from_registry("searcher_v1", pool)
        ...     print(manifest.capabilities)
    """
    query = """
        SELECT
            id,
            version,
            agent_type::text as agent_type,
            display_name,
            description,
            capabilities,
            allowed_domains,
            blocked_domains,
            max_budget_usd,
            rate_limit_rpm,
            rate_limit_tpm,
            model_config,
            allowed_tools,
            tool_config,
            timeout_seconds,
            max_concurrent_calls,
            is_active,
            is_default,
            metadata
        FROM agent_registry
        WHERE id = $1
    """

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(query, agent_id)

    if row is None:
        raise ValueError(f"Agent not found in registry: {agent_id}")

    # Map database columns to manifest fields
    manifest_data = {
        "id": row["id"],
        "version": row["version"],
        "agent_type": row["agent_type"],
        "display_name": row["display_name"],
        "description": row["description"],
        "capabilities": list(row["capabilities"]) if row["capabilities"] else [],
        "allowed_domains": list(row["allowed_domains"]) if row["allowed_domains"] else [],
        "blocked_domains": list(row["blocked_domains"]) if row["blocked_domains"] else [],
        "max_budget_usd": float(row["max_budget_usd"]) if row["max_budget_usd"] else 1.0,
        "rate_limits": {
            "requests_per_minute": row["rate_limit_rpm"] or 60,
            "tokens_per_minute": row["rate_limit_tpm"] or 100000,
        },
        "model_config": row["model_config"] if row["model_config"] else {},
        "allowed_tools": list(row["allowed_tools"]) if row["allowed_tools"] else [],
        "tool_config": row["tool_config"] if row["tool_config"] else {},
        "timeout_seconds": row["timeout_seconds"] or 60,
        "max_concurrent_calls": row["max_concurrent_calls"] or 5,
        "is_active": row["is_active"] if row["is_active"] is not None else True,
        "is_default": row["is_default"] if row["is_default"] is not None else False,
        "metadata": row["metadata"] if row["metadata"] else {},
    }

    return AgentManifest.model_validate(manifest_data)


async def save_manifest_to_registry(
    manifest: AgentManifest,
    db_pool: Any,
    upsert: bool = True,
) -> None:
    """
    Save an AgentManifest to the PostgreSQL agent_registry table.

    Args:
        manifest: The AgentManifest to save.
        db_pool: asyncpg connection pool for database access.
        upsert: If True, update existing record. If False, raise on conflict.

    Raises:
        IntegrityError: If upsert=False and agent already exists.

    Example:
        >>> manifest = get_default_manifest("searcher")
        >>> async with asyncpg.create_pool(dsn) as pool:
        ...     await save_manifest_to_registry(manifest, pool)
    """
    import json

    if upsert:
        query = """
            INSERT INTO agent_registry (
                id, version, agent_type, display_name, description,
                capabilities, allowed_domains, blocked_domains,
                max_budget_usd, rate_limit_rpm, rate_limit_tpm,
                model_config, allowed_tools, tool_config,
                timeout_seconds, max_concurrent_calls,
                is_active, is_default, metadata
            ) VALUES (
                $1, $2, $3::agent_type, $4, $5,
                $6, $7, $8,
                $9, $10, $11,
                $12::jsonb, $13, $14::jsonb,
                $15, $16,
                $17, $18, $19::jsonb
            )
            ON CONFLICT (id) DO UPDATE SET
                version = EXCLUDED.version,
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                capabilities = EXCLUDED.capabilities,
                allowed_domains = EXCLUDED.allowed_domains,
                blocked_domains = EXCLUDED.blocked_domains,
                max_budget_usd = EXCLUDED.max_budget_usd,
                rate_limit_rpm = EXCLUDED.rate_limit_rpm,
                rate_limit_tpm = EXCLUDED.rate_limit_tpm,
                model_config = EXCLUDED.model_config,
                allowed_tools = EXCLUDED.allowed_tools,
                tool_config = EXCLUDED.tool_config,
                timeout_seconds = EXCLUDED.timeout_seconds,
                max_concurrent_calls = EXCLUDED.max_concurrent_calls,
                is_active = EXCLUDED.is_active,
                is_default = EXCLUDED.is_default,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """
    else:
        query = """
            INSERT INTO agent_registry (
                id, version, agent_type, display_name, description,
                capabilities, allowed_domains, blocked_domains,
                max_budget_usd, rate_limit_rpm, rate_limit_tpm,
                model_config, allowed_tools, tool_config,
                timeout_seconds, max_concurrent_calls,
                is_active, is_default, metadata
            ) VALUES (
                $1, $2, $3::agent_type, $4, $5,
                $6, $7, $8,
                $9, $10, $11,
                $12::jsonb, $13, $14::jsonb,
                $15, $16,
                $17, $18, $19::jsonb
            )
        """

    async with db_pool.acquire() as conn:
        await conn.execute(
            query,
            manifest.id,
            manifest.version,
            manifest.agent_type,
            manifest.display_name,
            manifest.description,
            manifest.capabilities,
            manifest.allowed_domains,
            manifest.blocked_domains,
            manifest.max_budget_usd,
            manifest.rate_limits.requests_per_minute,
            manifest.rate_limits.tokens_per_minute,
            json.dumps(manifest.model_config_params.model_dump()),
            manifest.allowed_tools,
            json.dumps(manifest.tool_config),
            manifest.timeout_seconds,
            manifest.max_concurrent_calls,
            manifest.is_active,
            manifest.is_default,
            json.dumps(manifest.metadata),
        )


async def list_manifests_from_registry(
    db_pool: Any,
    agent_type: AgentTypeEnum | None = None,
    active_only: bool = True,
) -> list[AgentManifest]:
    """
    List all agent manifests from the registry.

    Args:
        db_pool: asyncpg connection pool for database access.
        agent_type: Optional filter by agent type.
        active_only: If True, only return active agents.

    Returns:
        List of AgentManifest instances.

    Example:
        >>> async with asyncpg.create_pool(dsn) as pool:
        ...     manifests = await list_manifests_from_registry(pool, agent_type="searcher")
        ...     for m in manifests:
        ...         print(m.id)
    """
    conditions = []
    params = []

    if active_only:
        conditions.append("is_active = true")

    if agent_type:
        params.append(agent_type)
        conditions.append(f"agent_type = ${len(params)}::agent_type")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            id,
            version,
            agent_type::text as agent_type,
            display_name,
            description,
            capabilities,
            allowed_domains,
            blocked_domains,
            max_budget_usd,
            rate_limit_rpm,
            rate_limit_tpm,
            model_config,
            allowed_tools,
            tool_config,
            timeout_seconds,
            max_concurrent_calls,
            is_active,
            is_default,
            metadata
        FROM agent_registry
        {where_clause}
        ORDER BY agent_type, id
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    manifests = []
    for row in rows:
        manifest_data = {
            "id": row["id"],
            "version": row["version"],
            "agent_type": row["agent_type"],
            "display_name": row["display_name"],
            "description": row["description"],
            "capabilities": list(row["capabilities"]) if row["capabilities"] else [],
            "allowed_domains": list(row["allowed_domains"]) if row["allowed_domains"] else [],
            "blocked_domains": list(row["blocked_domains"]) if row["blocked_domains"] else [],
            "max_budget_usd": float(row["max_budget_usd"]) if row["max_budget_usd"] else 1.0,
            "rate_limits": {
                "requests_per_minute": row["rate_limit_rpm"] or 60,
                "tokens_per_minute": row["rate_limit_tpm"] or 100000,
            },
            "model_config": row["model_config"] if row["model_config"] else {},
            "allowed_tools": list(row["allowed_tools"]) if row["allowed_tools"] else [],
            "tool_config": row["tool_config"] if row["tool_config"] else {},
            "timeout_seconds": row["timeout_seconds"] or 60,
            "max_concurrent_calls": row["max_concurrent_calls"] or 5,
            "is_active": row["is_active"] if row["is_active"] is not None else True,
            "is_default": row["is_default"] if row["is_default"] is not None else False,
            "metadata": row["metadata"] if row["metadata"] else {},
        }
        manifests.append(AgentManifest.model_validate(manifest_data))

    return manifests


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    # Type aliases
    "AgentTypeEnum",
    "HealthStatus",
    # TypedDicts
    "RateLimitsDict",
    "HealthThresholdsDict",
    "CircuitBreakerDict",
    "ModelConfigDict",
    "AgentManifestDict",
    # Pydantic models
    "RateLimits",
    "HealthThresholds",
    "CircuitBreakerConfig",
    "ModelConfig",
    "AgentManifest",
    # Factory functions
    "get_default_manifest",
    "manifest_to_dict",
    "validate_manifest",
    # Database functions
    "load_manifest_from_registry",
    "save_manifest_to_registry",
    "list_manifests_from_registry",
]
