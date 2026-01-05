"""
Agent Selector for DRX Deep Research System.

This module provides high-level agent selection logic that maps tasks
to agent requirements and uses the AgentCapabilityRouter for routing.

Part of WP-M4: Capability-Based Routing implementation for the DRX spec.

Key Components:
- AgentSelector: High-level class for task-to-agent mapping
- Task type to capability mapping
- Fallback agent determination
- Dynamic agent registration/unregistration

Integration Points:
- Uses AgentCapabilityRouter for low-level routing decisions
- Uses SubTask from orchestrator state for task context
- Uses AgentManifest for agent registration
- Designed to be used by orchestrator nodes for agent dispatch
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from src.metadata.routing import (
    AgentCapabilityRouter,
    AgentRequirements,
    RoutingDecision,
    create_requirements,
)

if TYPE_CHECKING:
    from src.services.active_state import ActiveStateService
    from src.metadata.manifest import AgentManifest
    from src.orchestrator.state import SubTask

logger = logging.getLogger(__name__)


# =============================================================================
# Task Type to Capability Mapping
# =============================================================================

# Maps agent_type from SubTask to the capabilities required for that type
AGENT_TYPE_CAPABILITIES: dict[str, list[str]] = {
    "planner": ["query_analysis", "plan_generation"],
    "searcher": ["web_search", "source_discovery"],
    "reader": ["content_extraction", "text_processing"],
    "synthesizer": ["information_synthesis", "knowledge_integration"],
    "critic": ["quality_assessment", "completeness_check"],
    "reporter": ["report_generation", "summarization"],
    "orchestrator": ["workflow_management", "task_delegation"],
    "reasoner": ["logical_reasoning", "inference"],
    "writer": ["content_generation", "formatting"],
}

# Maps agent_type to preferred (nice-to-have) capabilities
AGENT_TYPE_PREFERRED_CAPABILITIES: dict[str, list[str]] = {
    "planner": ["strategy_selection"],
    "searcher": ["query_expansion"],
    "reader": ["entity_recognition"],
    "synthesizer": ["conflict_resolution"],
    "critic": ["bias_detection"],
    "reporter": ["citation_formatting"],
    "orchestrator": ["state_management"],
    "reasoner": ["fact_checking"],
    "writer": ["editing"],
}

# Default cost tier by agent type
AGENT_TYPE_COST_TIER: dict[str, Literal["free", "standard", "premium"]] = {
    "planner": "standard",
    "searcher": "standard",
    "reader": "standard",
    "synthesizer": "standard",
    "critic": "standard",
    "reporter": "standard",
    "orchestrator": "premium",
    "reasoner": "premium",
    "writer": "standard",
}

# Default latency requirements by agent type (in milliseconds)
AGENT_TYPE_MAX_LATENCY: dict[str, int | None] = {
    "planner": 10000,  # Planning can take time
    "searcher": 5000,   # Search should be quick
    "reader": 15000,    # Reading content can take longer
    "synthesizer": 20000,  # Synthesis is compute-intensive
    "critic": 10000,    # Critique should be reasonably fast
    "reporter": 30000,  # Report generation can take time
    "orchestrator": 5000,  # Orchestrator should be responsive
    "reasoner": 20000,  # Reasoning can be complex
    "writer": 20000,    # Writing can take time
}


# =============================================================================
# AgentSelector Class
# =============================================================================


class AgentSelector:
    """
    High-level agent selection based on task requirements.

    The AgentSelector provides a simple interface for selecting agents
    to execute tasks, abstracting the complexity of capability matching
    and health-aware routing.

    Key Features:
        - Maps SubTask agent_type to capability requirements
        - Provides fallback agent lists for fault tolerance
        - Supports dynamic agent registration/unregistration
        - Caches routing decisions for performance

    Example:
        >>> selector = AgentSelector(active_state_service)
        >>> selector.register_agent(searcher_manifest)
        >>> selector.register_agent(searcher_v2_manifest)
        >>>
        >>> task = create_subtask("search-1", "Search web", "searcher", [], {})
        >>> agent_id = await selector.select_agent_for_task(task)
        >>> fallbacks = await selector.get_fallback_agents(agent_id)

    Attributes:
        router: The underlying AgentCapabilityRouter
        _task_cache: Cache of recent routing decisions
    """

    def __init__(
        self,
        active_state_service: "ActiveStateService",
        router: AgentCapabilityRouter | None = None,
    ) -> None:
        """
        Initialize the AgentSelector.

        Args:
            active_state_service: Service for health/metrics checks
            router: Optional pre-configured router (creates new if None)
        """
        self.router = router or AgentCapabilityRouter(active_state_service)
        self._task_cache: dict[str, RoutingDecision] = {}
        self._cache_ttl_seconds = 60  # Cache decisions for 1 minute

    def register_agent(self, manifest: "AgentManifest") -> None:
        """
        Register an agent for selection.

        Adds the agent to the routing registry, making it available
        for task assignment.

        Args:
            manifest: The agent manifest to register

        Example:
            >>> from src.metadata import get_default_manifest
            >>> manifest = get_default_manifest("searcher")
            >>> selector.register_agent(manifest)
        """
        self.router.register_agent(manifest)
        logger.info(f"Agent {manifest.id} registered for selection")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Remove an agent from selection.

        The agent will no longer be considered for task assignment.
        Active tasks assigned to this agent should be handled
        by the orchestrator.

        Args:
            agent_id: ID of the agent to remove

        Example:
            >>> selector.unregister_agent("searcher_v1")
        """
        removed = self.router.unregister_agent(agent_id)
        if removed:
            # Invalidate cache entries that might reference this agent
            self._invalidate_cache_for_agent(agent_id)
            logger.info(f"Agent {agent_id} unregistered from selection")
        else:
            logger.warning(f"Agent {agent_id} was not registered")

    def _invalidate_cache_for_agent(self, agent_id: str) -> None:
        """
        Invalidate cache entries that reference an agent.

        Args:
            agent_id: ID of the agent to invalidate
        """
        keys_to_remove = []
        for key, decision in self._task_cache.items():
            if (
                decision["selected_agent_id"] == agent_id
                or agent_id in decision["fallback_agents"]
            ):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._task_cache[key]

    def _map_task_to_requirements(
        self,
        task: "SubTask",
    ) -> AgentRequirements:
        """
        Map a SubTask to agent requirements.

        Converts the task's agent_type and metadata into a full
        AgentRequirements specification for routing.

        Args:
            task: The subtask to map

        Returns:
            AgentRequirements for the task

        Example:
            >>> task = create_subtask("s1", "Search", "searcher", [], {})
            >>> reqs = selector._map_task_to_requirements(task)
            >>> print(reqs["required_capabilities"])
            ['web_search', 'source_discovery']
        """
        agent_type = task["agent_type"]

        # Get required capabilities from mapping
        required_caps = AGENT_TYPE_CAPABILITIES.get(
            agent_type,
            [],  # Default to empty if unknown type
        )

        # Get preferred capabilities
        preferred_caps = AGENT_TYPE_PREFERRED_CAPABILITIES.get(
            agent_type,
            [],
        )

        # Get cost tier
        cost_tier = AGENT_TYPE_COST_TIER.get(agent_type, "standard")

        # Get max latency
        max_latency = AGENT_TYPE_MAX_LATENCY.get(agent_type, None)

        # Extract domain requirements from task inputs
        required_domains: list[str] | None = None
        if task.get("inputs"):
            inputs = task["inputs"]
            # Check for domain hints in inputs
            if "url" in inputs:
                from urllib.parse import urlparse
                parsed = urlparse(inputs["url"])
                if parsed.netloc:
                    required_domains = [parsed.netloc]
            elif "domains" in inputs:
                required_domains = inputs["domains"]

        # Determine compliance level from task metadata
        compliance_level: Literal["standard", "high", "critical"] = "standard"
        if task.get("inputs"):
            if task["inputs"].get("high_compliance"):
                compliance_level = "high"
            elif task["inputs"].get("critical_compliance"):
                compliance_level = "critical"

        return create_requirements(
            required_capabilities=required_caps,
            preferred_capabilities=preferred_caps,
            compliance_level=compliance_level,
            cost_tier=cost_tier,
            max_latency_ms=max_latency,
            required_domains=required_domains,
        )

    async def select_agent_for_task(
        self,
        task: "SubTask",
        use_cache: bool = True,
    ) -> str:
        """
        Select the best agent to execute a task.

        Maps the task to requirements and routes to the best available
        agent. Raises AgentNotFoundError if no suitable agent is found.

        Args:
            task: The subtask requiring an agent
            use_cache: Whether to use cached routing decisions

        Returns:
            ID of the selected agent

        Raises:
            AgentNotFoundError: If no suitable agent is available

        Example:
            >>> task = create_subtask("s1", "Search web", "searcher", [], {
            ...     "query": "machine learning trends"
            ... })
            >>> agent_id = await selector.select_agent_for_task(task)
            >>> print(f"Assigned to: {agent_id}")
        """
        task_id = task["id"]
        agent_type = task["agent_type"]

        # Check cache
        cache_key = f"{agent_type}:{task_id}"
        if use_cache and cache_key in self._task_cache:
            cached = self._task_cache[cache_key]
            if cached["selected_agent_id"]:
                logger.debug(f"Using cached selection for task {task_id}")
                return cached["selected_agent_id"]

        # Map task to requirements
        requirements = self._map_task_to_requirements(task)

        # Make routing decision
        decision = await self.router.make_routing_decision(requirements)

        # Cache the decision
        self._task_cache[cache_key] = decision

        if decision["selected_agent_id"] is None:
            logger.error(
                f"No agent available for task {task_id} ({agent_type}). "
                f"Rejections: {decision['rejection_reasons']}"
            )
            raise AgentNotFoundError(
                f"No agent available for task type '{agent_type}'. "
                f"Required capabilities: {requirements['required_capabilities']}"
            )

        logger.info(
            f"Selected agent {decision['selected_agent_id']} for task {task_id} "
            f"(score: {decision['score']:.2f})"
        )

        return decision["selected_agent_id"]

    async def get_fallback_agents(
        self,
        primary_agent: str,
        task: "SubTask | None" = None,
        max_fallbacks: int = 3,
    ) -> list[str]:
        """
        Get an ordered list of fallback agents for fault tolerance.

        If the primary agent fails, these fallbacks can be tried in order.
        The list excludes the primary agent and is ordered by fitness score.

        Args:
            primary_agent: ID of the primary agent
            task: Optional task for context-aware fallbacks
            max_fallbacks: Maximum number of fallbacks to return

        Returns:
            Ordered list of fallback agent IDs

        Example:
            >>> fallbacks = await selector.get_fallback_agents(
            ...     "searcher_v1",
            ...     task=search_task,
            ...     max_fallbacks=2
            ... )
            >>> print(fallbacks)  # ["searcher_v2", "searcher_v3"]
        """
        # Get manifest for primary agent to determine capabilities
        primary_manifest = self.router.get_agent_manifest(primary_agent)
        if not primary_manifest:
            logger.warning(f"Primary agent {primary_agent} not found")
            return []

        # Get all agents with similar capabilities
        primary_caps = set(primary_manifest.capabilities)
        candidates: list[tuple[str, float]] = []

        for agent_id in self.router.get_registered_agent_ids():
            if agent_id == primary_agent:
                continue

            manifest = self.router.get_agent_manifest(agent_id)
            if not manifest:
                continue

            # Check capability overlap
            agent_caps = set(manifest.capabilities)
            overlap = len(primary_caps & agent_caps) / len(primary_caps)

            # Only consider agents with significant overlap
            if overlap >= 0.5:
                # Score based on capability overlap and health
                try:
                    health = await self.router.active_state.get_agent_health(agent_id)
                    health_score = 1.0 if health["status"] == "healthy" else 0.5
                    score = overlap * health_score
                    candidates.append((agent_id, score))
                except Exception:
                    # Include without health bonus
                    candidates.append((agent_id, overlap * 0.5))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top N fallbacks
        fallbacks = [agent_id for agent_id, _ in candidates[:max_fallbacks]]

        logger.debug(
            f"Fallback agents for {primary_agent}: {fallbacks}"
        )

        return fallbacks

    async def select_with_fallbacks(
        self,
        task: "SubTask",
        max_attempts: int = 3,
    ) -> tuple[str, list[str]]:
        """
        Select an agent with prepared fallback list.

        Convenience method that returns both the selected agent
        and its fallbacks in a single call.

        Args:
            task: The subtask requiring an agent
            max_attempts: Maximum agents to try (1 primary + N-1 fallbacks)

        Returns:
            Tuple of (selected_agent_id, fallback_agent_ids)

        Raises:
            AgentNotFoundError: If no suitable agent is available

        Example:
            >>> agent_id, fallbacks = await selector.select_with_fallbacks(
            ...     task,
            ...     max_attempts=3
            ... )
        """
        primary = await self.select_agent_for_task(task)
        fallbacks = await self.get_fallback_agents(
            primary,
            task=task,
            max_fallbacks=max_attempts - 1,
        )
        return primary, fallbacks

    def get_available_agent_types(self) -> list[str]:
        """
        Get list of agent types that have registered agents.

        Returns:
            List of agent_type strings with at least one registered agent

        Example:
            >>> types = selector.get_available_agent_types()
            >>> print(types)  # ["planner", "searcher", "reader", ...]
        """
        available_types: set[str] = set()

        for agent_id in self.router.get_registered_agent_ids():
            manifest = self.router.get_agent_manifest(agent_id)
            if manifest:
                available_types.add(manifest.agent_type)

        return list(available_types)

    def can_handle_task_type(self, agent_type: str) -> bool:
        """
        Check if any agent can handle a given task type.

        Args:
            agent_type: The agent_type to check

        Returns:
            True if at least one agent can handle this type

        Example:
            >>> if selector.can_handle_task_type("searcher"):
            ...     print("Search capability available")
        """
        required_caps = AGENT_TYPE_CAPABILITIES.get(agent_type, [])
        if not required_caps:
            # Unknown type - check if any agent has this as their type
            for agent_id in self.router.get_registered_agent_ids():
                manifest = self.router.get_agent_manifest(agent_id)
                if manifest and manifest.agent_type == agent_type:
                    return True
            return False

        # Check if any agent has all required capabilities
        for agent_id in self.router.get_registered_agent_ids():
            if self.router._check_required_capabilities(agent_id, required_caps):
                return True

        return False

    def clear_cache(self) -> None:
        """
        Clear the routing decision cache.

        Useful after bulk agent registration/unregistration.
        """
        self._task_cache.clear()
        logger.debug("Routing cache cleared")


# =============================================================================
# Exceptions
# =============================================================================


class AgentNotFoundError(Exception):
    """Raised when no suitable agent is found for a task."""

    def __init__(self, message: str, agent_type: str | None = None) -> None:
        super().__init__(message)
        self.agent_type = agent_type


class AgentUnavailableError(Exception):
    """Raised when a specific agent is temporarily unavailable."""

    def __init__(self, message: str, agent_id: str) -> None:
        super().__init__(message)
        self.agent_id = agent_id


# =============================================================================
# Factory Functions
# =============================================================================


async def create_agent_selector(
    active_state_service: "ActiveStateService",
    register_defaults: bool = True,
) -> AgentSelector:
    """
    Create and initialize an AgentSelector.

    Optionally registers default agents for each agent type.

    Args:
        active_state_service: Service for health/metrics checks
        register_defaults: Whether to register default agent manifests

    Returns:
        Initialized AgentSelector

    Example:
        >>> service = await get_active_state_service()
        >>> selector = await create_agent_selector(service)
    """
    selector = AgentSelector(active_state_service)

    if register_defaults:
        from src.metadata.manifest import get_default_manifest, AgentTypeEnum

        # Register default agents for each type
        agent_types: list[AgentTypeEnum] = [
            "planner",
            "searcher",
            "reader",
            "synthesizer",
            "critic",
            "reporter",
        ]

        for agent_type in agent_types:
            try:
                manifest = get_default_manifest(agent_type)
                selector.register_agent(manifest)
            except Exception as e:
                logger.warning(
                    f"Failed to register default {agent_type} agent: {e}"
                )

    return selector


def get_capabilities_for_agent_type(agent_type: str) -> tuple[list[str], list[str]]:
    """
    Get required and preferred capabilities for an agent type.

    Args:
        agent_type: The agent type

    Returns:
        Tuple of (required_capabilities, preferred_capabilities)

    Example:
        >>> required, preferred = get_capabilities_for_agent_type("searcher")
        >>> print(required)  # ["web_search", "source_discovery"]
    """
    required = AGENT_TYPE_CAPABILITIES.get(agent_type, [])
    preferred = AGENT_TYPE_PREFERRED_CAPABILITIES.get(agent_type, [])
    return required, preferred


# =============================================================================
# Module-level Singleton
# =============================================================================

_agent_selector: AgentSelector | None = None


async def get_agent_selector(
    active_state_service: "ActiveStateService | None" = None,
) -> AgentSelector:
    """
    Get or create the AgentSelector singleton.

    Args:
        active_state_service: Service for health checks (required on first call)

    Returns:
        AgentSelector singleton instance

    Raises:
        ValueError: If active_state_service not provided on first call

    Example:
        >>> selector = await get_agent_selector(active_state_service)
        >>> agent = await selector.select_agent_for_task(task)
    """
    global _agent_selector

    if _agent_selector is not None:
        return _agent_selector

    if active_state_service is None:
        raise ValueError(
            "active_state_service must be provided on first call"
        )

    _agent_selector = await create_agent_selector(
        active_state_service,
        register_defaults=True,
    )

    return _agent_selector


def reset_agent_selector() -> None:
    """
    Reset the AgentSelector singleton.

    Useful for testing or reconfiguration.
    """
    global _agent_selector
    _agent_selector = None
    logger.info("AgentSelector singleton reset")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
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
]
