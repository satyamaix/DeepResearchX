"""
DRX Metadata Module - Agent Manifest, Routing, Selection, Circuit Breaker, and Context.

This module provides:
- AgentManifest schema and Pydantic models for agent configuration
- Capability-based routing for intelligent agent selection
- High-level AgentSelector for task-to-agent mapping
- Circuit breaker pattern for fault tolerance
- Health checking for agent monitoring
- Context propagation system for managing research context across agents

Part of the Agentic Metadata implementation (R10.8 from DRX.md spec),
WP-M3: Context Propagation System, WP-M4: Capability-Based Routing,
and WP-M5: Circuit Breaker Implementation.

Usage:
    from src.metadata import AgentManifest, get_default_manifest

    # Create a default manifest for an agent type
    manifest = get_default_manifest("searcher")

    # Validate a manifest dictionary
    is_valid, errors = validate_manifest({"id": "test_v1", ...})

    # Load from database
    manifest = await load_manifest_from_registry("searcher_v1", db_pool)

    # Capability-based routing
    from src.metadata import AgentCapabilityRouter, create_requirements
    router = AgentCapabilityRouter(active_state_service)
    router.register_agent(manifest)
    agent_id = await router.find_agent(create_requirements(
        required_capabilities=["web_search"],
    ))

    # High-level agent selection
    from src.metadata import AgentSelector
    selector = AgentSelector(active_state_service)
    agent_id = await selector.select_agent_for_task(subtask)

    # Circuit breaker usage (WP-M5)
    from src.metadata import CircuitBreaker, CircuitState
    circuit_breaker = CircuitBreaker(active_state_service)
    if await circuit_breaker.can_execute("searcher_v1"):
        try:
            result = await execute_agent_request()
            await circuit_breaker.record_success("searcher_v1")
        except Exception as e:
            await circuit_breaker.record_failure("searcher_v1", e)

    # Health checking (WP-M5)
    from src.metadata import HealthChecker
    health_checker = HealthChecker(active_state_service)
    if await health_checker.is_healthy("searcher_v1"):
        # Agent is healthy
        pass

    # Context propagation (WP-M3)
    from src.metadata import ContextPropagator, create_context_store
    store = await create_context_store("hybrid")
    propagator = ContextPropagator(
        embedding_client=embeddings,
        llm_client=llm,
        context_store=store,
    )
    context = await propagator.create_context(agent_state)
"""

from __future__ import annotations

__version__ = "1.0.0"

# Import all public types and functions from manifest module
from src.metadata.manifest import (
    # Type aliases
    AgentTypeEnum,
    HealthStatus,
    # TypedDicts (for LangGraph compatibility)
    RateLimitsDict,
    HealthThresholdsDict,
    CircuitBreakerDict,
    ModelConfigDict,
    AgentManifestDict,
    # Pydantic models
    RateLimits,
    HealthThresholds,
    CircuitBreakerConfig,
    ModelConfig,
    AgentManifest,
    # Factory functions
    get_default_manifest,
    manifest_to_dict,
    validate_manifest,
    # Database functions
    load_manifest_from_registry,
    save_manifest_to_registry,
    list_manifests_from_registry,
)

# Import routing types and classes (WP-M4)
from src.metadata.routing import (
    # TypedDicts
    AgentRequirements,
    AgentInfo,
    RoutingDecision,
    # Constants
    COST_TIER_ORDER,
    COMPLIANCE_LEVEL_ORDER,
    # Classes
    AgentCapabilityRouter,
    # Factory functions
    create_requirements,
    capabilities_match,
)

# Import agent selector types and classes (WP-M4)
from src.metadata.agent_selector import (
    # Constants
    AGENT_TYPE_CAPABILITIES,
    AGENT_TYPE_PREFERRED_CAPABILITIES,
    AGENT_TYPE_COST_TIER,
    AGENT_TYPE_MAX_LATENCY,
    # Classes
    AgentSelector,
    # Exceptions
    AgentNotFoundError,
    AgentUnavailableError,
    # Factory functions
    create_agent_selector,
    get_capabilities_for_agent_type,
    # Singleton access
    get_agent_selector,
    reset_agent_selector,
)

# Import context store types and classes (WP-M3)
from src.metadata.context_store import (
    # Type definitions
    ContextStoreType,
    ContextStatus,
    # TypedDicts
    ResearchContext,
    ContextMetadata,
    ContextStoreConfig,
    # Abstract base class
    ContextStore,
    # Exceptions
    ContextStoreError,
    ContextNotFoundError,
    ContextStorageError,
    # Implementations
    RedisContextStore,
    PostgresContextStore,
    HybridContextStore,
    # Factory function
    create_context_store,
)

# Import context propagation types and classes (WP-M3)
from src.metadata.context import (
    # Protocols
    EmbeddingClient,
    LLMClient,
    ChunkStore,
    # Configuration
    ContextPropagatorConfig,
    # Main class
    ContextPropagator,
    # Utility functions
    create_empty_context,
    context_to_dict,
    estimate_context_tokens,
)

# Import circuit breaker types and classes (WP-M5)
from src.metadata.circuit_breaker import (
    # Enums
    CircuitState,
    # TypedDicts
    CircuitBreakerConfigDict,
    CircuitStatsDict,
    CircuitStateChangeEvent,
    # Constants
    DEFAULT_CIRCUIT_BREAKER_CONFIG,
    # Exceptions
    CircuitBreakerError,
    CircuitOpenError,
    CircuitHalfOpenExhaustedError,
    # Main class
    CircuitBreaker,
    # Factory functions
    create_circuit_breaker,
    create_circuit_breaker_sync,
    get_circuit_breaker,
    reset_circuit_breaker_singleton,
)

# Import health checker types and classes (WP-M5)
from src.metadata.health_checker import (
    # TypedDicts
    HealthCheckResult,
    HealthStatusDict,
    HealthThresholdsDict as HealthCheckerThresholdsDict,
    # Constants
    DEFAULT_HEALTH_THRESHOLDS,
    # Main class
    HealthChecker,
    # Factory functions
    create_health_checker,
    create_health_checker_sync,
    get_health_checker,
    reset_health_checker_singleton,
)

__all__ = [
    # Module version
    "__version__",
    # ==========================================================================
    # From manifest.py
    # ==========================================================================
    # Type aliases
    "AgentTypeEnum",
    "HealthStatus",
    # TypedDicts (for LangGraph compatibility)
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
    # ==========================================================================
    # From routing.py (WP-M4)
    # ==========================================================================
    # TypedDicts
    "AgentRequirements",
    "AgentInfo",
    "RoutingDecision",
    # Constants
    "COST_TIER_ORDER",
    "COMPLIANCE_LEVEL_ORDER",
    # Classes
    "AgentCapabilityRouter",
    # Factory functions
    "create_requirements",
    "capabilities_match",
    # ==========================================================================
    # From agent_selector.py (WP-M4)
    # ==========================================================================
    # Constants
    "AGENT_TYPE_CAPABILITIES",
    "AGENT_TYPE_PREFERRED_CAPABILITIES",
    "AGENT_TYPE_COST_TIER",
    "AGENT_TYPE_MAX_LATENCY",
    # Classes
    "AgentSelector",
    # Exceptions
    "AgentNotFoundError",
    "AgentUnavailableError",
    # Factory functions
    "create_agent_selector",
    "get_capabilities_for_agent_type",
    # Singleton access
    "get_agent_selector",
    "reset_agent_selector",
    # ==========================================================================
    # From context_store.py (WP-M3)
    # ==========================================================================
    # Type definitions
    "ContextStoreType",
    "ContextStatus",
    # TypedDicts
    "ResearchContext",
    "ContextMetadata",
    "ContextStoreConfig",
    # Abstract base class
    "ContextStore",
    # Exceptions
    "ContextStoreError",
    "ContextNotFoundError",
    "ContextStorageError",
    # Implementations
    "RedisContextStore",
    "PostgresContextStore",
    "HybridContextStore",
    # Factory function
    "create_context_store",
    # ==========================================================================
    # From context.py (WP-M3)
    # ==========================================================================
    # Protocols
    "EmbeddingClient",
    "LLMClient",
    "ChunkStore",
    # Configuration
    "ContextPropagatorConfig",
    # Main class
    "ContextPropagator",
    # Utility functions
    "create_empty_context",
    "context_to_dict",
    "estimate_context_tokens",
    # ==========================================================================
    # From circuit_breaker.py (WP-M5)
    # ==========================================================================
    # Enums
    "CircuitState",
    # TypedDicts
    "CircuitBreakerConfigDict",
    "CircuitStatsDict",
    "CircuitStateChangeEvent",
    # Constants
    "DEFAULT_CIRCUIT_BREAKER_CONFIG",
    # Exceptions
    "CircuitBreakerError",
    "CircuitOpenError",
    "CircuitHalfOpenExhaustedError",
    # Main class
    "CircuitBreaker",
    # Factory functions
    "create_circuit_breaker",
    "create_circuit_breaker_sync",
    "get_circuit_breaker",
    "reset_circuit_breaker_singleton",
    # ==========================================================================
    # From health_checker.py (WP-M5)
    # ==========================================================================
    # TypedDicts
    "HealthCheckResult",
    "HealthStatusDict",
    "HealthCheckerThresholdsDict",
    # Constants
    "DEFAULT_HEALTH_THRESHOLDS",
    # Main class
    "HealthChecker",
    # Factory functions
    "create_health_checker",
    "create_health_checker_sync",
    "get_health_checker",
    "reset_health_checker_singleton",
]
