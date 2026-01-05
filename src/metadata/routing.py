"""
Capability-Based Routing for DRX Deep Research System.

This module provides intelligent agent routing based on capabilities,
health status, latency requirements, and cost considerations.

Part of WP-M4: Capability-Based Routing implementation for the DRX spec.

Key Components:
- AgentRequirements: TypedDict specifying what an agent needs to satisfy
- AgentInfo: TypedDict with agent metadata for routing decisions
- RoutingDecision: TypedDict capturing the routing result with reasoning
- AgentCapabilityRouter: Main router class for finding optimal agents

Integration Points:
- Uses ActiveStateService for health and metrics checks
- Uses AgentManifest for capability and configuration data
- Designed to be used by AgentSelector for high-level task routing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from src.services.active_state import ActiveStateService
    from src.metadata.manifest import AgentManifest

logger = logging.getLogger(__name__)


# =============================================================================
# TypedDict Definitions
# =============================================================================


class AgentRequirements(TypedDict):
    """
    Requirements specification for agent selection.

    Defines the criteria an agent must meet to handle a specific task.
    Used by the router to filter and score available agents.

    Attributes:
        required_capabilities: Capabilities the agent MUST have (all required)
        preferred_capabilities: Capabilities that improve agent score (nice to have)
        compliance_level: Required compliance tier for the task
        cost_tier: Maximum acceptable cost tier
        max_latency_ms: Maximum acceptable p99 latency (None = no limit)
        required_domains: Domains the agent must be able to access (None = any)
    """

    required_capabilities: list[str]
    preferred_capabilities: list[str]
    compliance_level: Literal["standard", "high", "critical"]
    cost_tier: Literal["free", "standard", "premium"]
    max_latency_ms: int | None
    required_domains: list[str] | None


class AgentInfo(TypedDict):
    """
    Runtime information about an available agent.

    Combines static manifest data with dynamic health/metrics data
    for routing decisions.

    Attributes:
        agent_id: Unique identifier for the agent
        capabilities: List of capabilities this agent provides
        health_status: Current health status (healthy, degraded, unhealthy)
        current_load: Current load factor (0.0-1.0, based on concurrent requests)
        latency_p99: 99th percentile latency in milliseconds
        cost_tier: Cost tier of this agent (free, standard, premium)
    """

    agent_id: str
    capabilities: list[str]
    health_status: str
    current_load: float
    latency_p99: float
    cost_tier: str


class RoutingDecision(TypedDict):
    """
    Result of a routing decision with full context.

    Provides transparency into why a particular agent was selected
    or why routing failed.

    Attributes:
        selected_agent_id: ID of the selected agent (None if no match)
        score: Fitness score of the selected agent (0.0-1.0)
        candidates_evaluated: Number of agents considered
        rejection_reasons: Map of agent_id to reason for rejection
        fallback_agents: Ordered list of fallback agent IDs
        routing_time_ms: Time taken to make the routing decision
    """

    selected_agent_id: str | None
    score: float
    candidates_evaluated: int
    rejection_reasons: dict[str, str]
    fallback_agents: list[str]
    routing_time_ms: float


# =============================================================================
# Cost Tier Ordering
# =============================================================================

COST_TIER_ORDER: dict[str, int] = {
    "free": 0,
    "standard": 1,
    "premium": 2,
}

COMPLIANCE_LEVEL_ORDER: dict[str, int] = {
    "standard": 0,
    "high": 1,
    "critical": 2,
}


# =============================================================================
# AgentCapabilityRouter Class
# =============================================================================


class AgentCapabilityRouter:
    """
    Routes tasks to agents based on capability matching and health.

    The router maintains a registry of available agents (from AgentManifest)
    and uses the ActiveStateService to check real-time health and metrics.

    Scoring Algorithm:
        1. Filter agents by required capabilities (must have ALL)
        2. Filter by cost tier (must be <= required tier)
        3. Filter by latency requirement (if specified)
        4. Filter by health status (exclude unhealthy)
        5. Score remaining agents:
           - +0.3 base if all required capabilities met
           - +0.1 per preferred capability (up to 0.3)
           - +0.2 if health is 'healthy' (vs 0.1 for 'degraded')
           - +0.2 based on load factor (lower is better)
           - -0.1 penalty per cost tier difference from free

    Example:
        >>> router = AgentCapabilityRouter(active_state_service)
        >>> router.register_agent(searcher_manifest)
        >>> agent_id = await router.find_agent({
        ...     "required_capabilities": ["web_search"],
        ...     "preferred_capabilities": ["query_expansion"],
        ...     "compliance_level": "standard",
        ...     "cost_tier": "standard",
        ...     "max_latency_ms": 5000,
        ...     "required_domains": None,
        ... })
        >>> print(agent_id)  # "searcher_v1"

    Attributes:
        active_state: Service for health/metrics checks
        _agents: Registry of agent manifests by ID
        _capability_index: Reverse index from capability to agent IDs
    """

    def __init__(self, active_state_service: "ActiveStateService") -> None:
        """
        Initialize the capability router.

        Args:
            active_state_service: Service for checking agent health and metrics
        """
        self.active_state = active_state_service
        self._agents: dict[str, "AgentManifest"] = {}
        self._capability_index: dict[str, set[str]] = {}

    def register_agent(self, manifest: "AgentManifest") -> None:
        """
        Register an agent manifest for routing.

        Adds the agent to the internal registry and updates the
        capability index for fast capability-based lookups.

        Args:
            manifest: The agent manifest to register

        Example:
            >>> from src.metadata import get_default_manifest
            >>> manifest = get_default_manifest("searcher")
            >>> router.register_agent(manifest)
        """
        agent_id = manifest.id
        self._agents[agent_id] = manifest

        # Update capability index
        for capability in manifest.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = set()
            self._capability_index[capability].add(agent_id)

        logger.debug(
            f"Registered agent {agent_id} with capabilities: {manifest.capabilities}"
        )

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the routing registry.

        Args:
            agent_id: ID of the agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        if agent_id not in self._agents:
            logger.warning(f"Agent {agent_id} not found in registry")
            return False

        manifest = self._agents.pop(agent_id)

        # Clean up capability index
        for capability in manifest.capabilities:
            if capability in self._capability_index:
                self._capability_index[capability].discard(agent_id)
                if not self._capability_index[capability]:
                    del self._capability_index[capability]

        logger.debug(f"Unregistered agent {agent_id}")
        return True

    def get_agent_manifest(self, agent_id: str) -> "AgentManifest | None":
        """
        Get the manifest for a registered agent.

        Args:
            agent_id: ID of the agent

        Returns:
            AgentManifest if found, None otherwise
        """
        return self._agents.get(agent_id)

    def get_registered_agent_ids(self) -> list[str]:
        """
        Get list of all registered agent IDs.

        Returns:
            List of registered agent IDs
        """
        return list(self._agents.keys())

    async def _check_health(self, agent_id: str) -> bool:
        """
        Check if an agent is healthy enough for routing.

        Uses the ActiveStateService to get current health status
        and circuit breaker state.

        Args:
            agent_id: ID of the agent to check

        Returns:
            True if agent is healthy or degraded, False if unhealthy
        """
        try:
            health = await self.active_state.get_agent_health(agent_id)
            status = health["status"]

            # Also check circuit breaker
            circuit_status = health["circuit_status"]
            if circuit_status == "open":
                logger.debug(f"Agent {agent_id} circuit is open, excluding")
                return False

            # Allow healthy and degraded, exclude unhealthy
            if status in ("healthy", "degraded"):
                return True

            logger.debug(f"Agent {agent_id} is unhealthy, excluding")
            return False

        except Exception as e:
            logger.warning(f"Failed to check health for {agent_id}: {e}")
            # On error, assume unhealthy to be safe
            return False

    async def _check_latency_requirement(
        self,
        agent_id: str,
        max_latency_ms: int,
    ) -> bool:
        """
        Check if an agent meets the latency requirement.

        Compares the agent's p99 latency against the requirement.

        Args:
            agent_id: ID of the agent to check
            max_latency_ms: Maximum acceptable p99 latency

        Returns:
            True if agent meets latency requirement, False otherwise
        """
        try:
            health = await self.active_state.get_agent_health(agent_id)
            metrics = health["metrics"]
            p99_latency = metrics["latency_p99"]

            if p99_latency <= max_latency_ms:
                return True

            logger.debug(
                f"Agent {agent_id} latency {p99_latency}ms exceeds max {max_latency_ms}ms"
            )
            return False

        except Exception as e:
            logger.warning(f"Failed to check latency for {agent_id}: {e}")
            # On error, be conservative and exclude
            return False

    def _check_cost_tier(
        self,
        agent_id: str,
        max_cost_tier: Literal["free", "standard", "premium"],
    ) -> bool:
        """
        Check if an agent's cost tier is acceptable.

        Args:
            agent_id: ID of the agent to check
            max_cost_tier: Maximum acceptable cost tier

        Returns:
            True if agent cost tier is acceptable, False otherwise
        """
        manifest = self._agents.get(agent_id)
        if not manifest:
            return False

        # Get cost tier from manifest metadata or default to "standard"
        agent_tier = manifest.metadata.get("cost_tier", "standard")
        max_tier_order = COST_TIER_ORDER.get(max_cost_tier, 1)
        agent_tier_order = COST_TIER_ORDER.get(agent_tier, 1)

        return agent_tier_order <= max_tier_order

    def _check_required_capabilities(
        self,
        agent_id: str,
        required_capabilities: list[str],
    ) -> bool:
        """
        Check if an agent has all required capabilities.

        Args:
            agent_id: ID of the agent to check
            required_capabilities: Capabilities that must all be present

        Returns:
            True if agent has ALL required capabilities, False otherwise
        """
        manifest = self._agents.get(agent_id)
        if not manifest:
            return False

        agent_capabilities = set(manifest.capabilities)
        required_set = set(required_capabilities)

        return required_set.issubset(agent_capabilities)

    def _check_domain_access(
        self,
        agent_id: str,
        required_domains: list[str],
    ) -> bool:
        """
        Check if an agent can access the required domains.

        Args:
            agent_id: ID of the agent to check
            required_domains: Domains the agent must be able to access

        Returns:
            True if agent can access all required domains, False otherwise
        """
        manifest = self._agents.get(agent_id)
        if not manifest:
            return False

        # If agent has no domain restrictions, it can access anything
        if not manifest.allowed_domains and not manifest.blocked_domains:
            return True

        # Check if all required domains are allowed
        for domain in required_domains:
            # Check blocked list first
            if manifest.blocked_domains:
                if any(
                    blocked in domain or domain in blocked
                    for blocked in manifest.blocked_domains
                ):
                    return False

            # If allowed_domains is set, domain must be in it
            if manifest.allowed_domains:
                if not any(
                    allowed in domain or domain in allowed
                    for allowed in manifest.allowed_domains
                ):
                    return False

        return True

    async def score_agent(
        self,
        agent_id: str,
        requirements: AgentRequirements,
    ) -> float:
        """
        Calculate a fitness score for an agent given requirements.

        Scoring Components:
            - Base capability match: +0.3 (required capabilities met)
            - Preferred capabilities: +0.1 each (max +0.3)
            - Health status: +0.2 (healthy) or +0.1 (degraded)
            - Load factor: +0.2 * (1 - current_load)
            - Cost efficiency: -0.1 per tier above free

        Args:
            agent_id: ID of the agent to score
            requirements: Requirements to score against

        Returns:
            Fitness score between 0.0 and 1.0

        Example:
            >>> score = await router.score_agent("searcher_v1", requirements)
            >>> print(f"Agent fitness: {score:.2f}")
        """
        manifest = self._agents.get(agent_id)
        if not manifest:
            return 0.0

        score = 0.0

        # Base capability match (+0.3)
        if self._check_required_capabilities(
            agent_id, requirements["required_capabilities"]
        ):
            score += 0.3
        else:
            # Missing required capabilities = zero score
            return 0.0

        # Preferred capabilities (+0.1 each, max +0.3)
        agent_caps = set(manifest.capabilities)
        preferred_count = sum(
            1 for cap in requirements["preferred_capabilities"]
            if cap in agent_caps
        )
        score += min(0.3, preferred_count * 0.1)

        # Health status (+0.2 healthy, +0.1 degraded)
        try:
            health = await self.active_state.get_agent_health(agent_id)
            if health["status"] == "healthy":
                score += 0.2
            elif health["status"] == "degraded":
                score += 0.1
            # unhealthy = no health bonus

            # Load factor (+0.2 * (1 - load))
            # Lower load = higher score
            metrics = health["metrics"]
            # Estimate load from requests_1m (normalized)
            requests_1m = metrics.get("requests_1m", 0)
            max_rpm = manifest.rate_limits.requests_per_minute
            current_load = min(1.0, requests_1m / max_rpm) if max_rpm > 0 else 0.0
            score += 0.2 * (1 - current_load)

        except Exception as e:
            logger.debug(f"Failed to get health for scoring {agent_id}: {e}")
            # Without health data, give minimal bonus
            score += 0.1

        # Cost efficiency penalty (-0.1 per tier above free)
        agent_tier = manifest.metadata.get("cost_tier", "standard")
        tier_order = COST_TIER_ORDER.get(agent_tier, 1)
        score -= tier_order * 0.1

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    async def find_agent(
        self,
        requirements: AgentRequirements,
    ) -> str | None:
        """
        Find the best agent matching the given requirements.

        Process:
            1. Get candidate agents with required capabilities
            2. Filter by cost tier
            3. Filter by domain access (if specified)
            4. Filter by health status
            5. Filter by latency (if specified)
            6. Score remaining candidates
            7. Return highest-scoring agent

        Args:
            requirements: The requirements specification

        Returns:
            Agent ID of the best match, or None if no suitable agent found

        Example:
            >>> agent_id = await router.find_agent({
            ...     "required_capabilities": ["web_search", "query_expansion"],
            ...     "preferred_capabilities": ["source_discovery"],
            ...     "compliance_level": "standard",
            ...     "cost_tier": "standard",
            ...     "max_latency_ms": 5000,
            ...     "required_domains": ["arxiv.org"],
            ... })
        """
        import time
        start_time = time.time()

        candidates: list[str] = []
        rejection_reasons: dict[str, str] = {}

        # Step 1: Get candidates with required capabilities
        required_caps = requirements["required_capabilities"]
        if not required_caps:
            # No required capabilities = all agents are candidates
            candidates = list(self._agents.keys())
        else:
            # Find agents with ALL required capabilities
            for agent_id in self._agents.keys():
                if self._check_required_capabilities(agent_id, required_caps):
                    candidates.append(agent_id)
                else:
                    rejection_reasons[agent_id] = "Missing required capabilities"

        if not candidates:
            logger.warning(
                f"No agents found with capabilities: {required_caps}"
            )
            return None

        # Step 2: Filter by cost tier
        max_tier = requirements["cost_tier"]
        filtered = []
        for agent_id in candidates:
            if self._check_cost_tier(agent_id, max_tier):
                filtered.append(agent_id)
            else:
                rejection_reasons[agent_id] = f"Cost tier exceeds {max_tier}"
        candidates = filtered

        if not candidates:
            logger.warning(f"No agents within cost tier: {max_tier}")
            return None

        # Step 3: Filter by domain access (if specified)
        required_domains = requirements.get("required_domains")
        if required_domains:
            filtered = []
            for agent_id in candidates:
                if self._check_domain_access(agent_id, required_domains):
                    filtered.append(agent_id)
                else:
                    rejection_reasons[agent_id] = "Domain access restricted"
            candidates = filtered

            if not candidates:
                logger.warning(
                    f"No agents with access to domains: {required_domains}"
                )
                return None

        # Step 4: Filter by health status
        filtered = []
        for agent_id in candidates:
            if await self._check_health(agent_id):
                filtered.append(agent_id)
            else:
                rejection_reasons[agent_id] = "Unhealthy or circuit open"
        candidates = filtered

        if not candidates:
            logger.warning("No healthy agents available")
            return None

        # Step 5: Filter by latency (if specified)
        max_latency = requirements.get("max_latency_ms")
        if max_latency is not None:
            filtered = []
            for agent_id in candidates:
                if await self._check_latency_requirement(agent_id, max_latency):
                    filtered.append(agent_id)
                else:
                    rejection_reasons[agent_id] = f"Latency exceeds {max_latency}ms"
            candidates = filtered

            if not candidates:
                logger.warning(f"No agents meeting latency requirement: {max_latency}ms")
                return None

        # Step 6: Score remaining candidates
        scored_agents: list[tuple[str, float]] = []
        for agent_id in candidates:
            score = await self.score_agent(agent_id, requirements)
            scored_agents.append((agent_id, score))

        # Sort by score descending
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        # Step 7: Return best agent
        if scored_agents:
            best_agent, best_score = scored_agents[0]
            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Selected agent {best_agent} (score: {best_score:.2f}) "
                f"from {len(scored_agents)} candidates in {elapsed_ms:.1f}ms"
            )
            return best_agent

        return None

    async def get_available_agents(
        self,
        capability: str,
    ) -> list[AgentInfo]:
        """
        Get all agents with a specific capability and their current status.

        Useful for displaying available agents or debugging routing issues.

        Args:
            capability: The capability to search for

        Returns:
            List of AgentInfo for all agents with the capability

        Example:
            >>> agents = await router.get_available_agents("web_search")
            >>> for agent in agents:
            ...     print(f"{agent['agent_id']}: {agent['health_status']}")
        """
        agent_ids = self._capability_index.get(capability, set())
        results: list[AgentInfo] = []

        for agent_id in agent_ids:
            manifest = self._agents.get(agent_id)
            if not manifest:
                continue

            try:
                health = await self.active_state.get_agent_health(agent_id)
                metrics = health["metrics"]

                # Calculate current load
                requests_1m = metrics.get("requests_1m", 0)
                max_rpm = manifest.rate_limits.requests_per_minute
                current_load = min(1.0, requests_1m / max_rpm) if max_rpm > 0 else 0.0

                agent_info: AgentInfo = {
                    "agent_id": agent_id,
                    "capabilities": manifest.capabilities,
                    "health_status": health["status"],
                    "current_load": current_load,
                    "latency_p99": metrics["latency_p99"],
                    "cost_tier": manifest.metadata.get("cost_tier", "standard"),
                }
                results.append(agent_info)

            except Exception as e:
                logger.warning(f"Failed to get info for {agent_id}: {e}")
                # Include with unknown/default values
                agent_info = {
                    "agent_id": agent_id,
                    "capabilities": manifest.capabilities,
                    "health_status": "unknown",
                    "current_load": 0.0,
                    "latency_p99": 0.0,
                    "cost_tier": manifest.metadata.get("cost_tier", "standard"),
                }
                results.append(agent_info)

        return results

    async def make_routing_decision(
        self,
        requirements: AgentRequirements,
    ) -> RoutingDecision:
        """
        Make a full routing decision with detailed context.

        Unlike find_agent(), this method returns complete routing
        context including rejection reasons and fallback agents.

        Args:
            requirements: The requirements specification

        Returns:
            RoutingDecision with full context

        Example:
            >>> decision = await router.make_routing_decision(requirements)
            >>> if decision["selected_agent_id"]:
            ...     print(f"Selected: {decision['selected_agent_id']}")
            ... else:
            ...     print(f"Rejected: {decision['rejection_reasons']}")
        """
        import time
        start_time = time.time()

        candidates: list[str] = []
        rejection_reasons: dict[str, str] = {}
        scored_agents: list[tuple[str, float]] = []

        # Get all agents with required capabilities
        required_caps = requirements["required_capabilities"]
        for agent_id in self._agents.keys():
            if not required_caps or self._check_required_capabilities(
                agent_id, required_caps
            ):
                candidates.append(agent_id)
            else:
                rejection_reasons[agent_id] = "Missing required capabilities"

        # Filter and score candidates
        for agent_id in candidates:
            # Cost tier check
            if not self._check_cost_tier(agent_id, requirements["cost_tier"]):
                rejection_reasons[agent_id] = "Cost tier too high"
                continue

            # Domain access check
            required_domains = requirements.get("required_domains")
            if required_domains and not self._check_domain_access(
                agent_id, required_domains
            ):
                rejection_reasons[agent_id] = "Domain access restricted"
                continue

            # Health check
            if not await self._check_health(agent_id):
                rejection_reasons[agent_id] = "Unhealthy or circuit open"
                continue

            # Latency check
            max_latency = requirements.get("max_latency_ms")
            if max_latency is not None:
                if not await self._check_latency_requirement(agent_id, max_latency):
                    rejection_reasons[agent_id] = f"Latency exceeds {max_latency}ms"
                    continue

            # Agent passed all filters, score it
            score = await self.score_agent(agent_id, requirements)
            scored_agents.append((agent_id, score))

        # Sort by score
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        elapsed_ms = (time.time() - start_time) * 1000

        if scored_agents:
            selected_id, selected_score = scored_agents[0]
            fallbacks = [agent_id for agent_id, _ in scored_agents[1:4]]

            return RoutingDecision(
                selected_agent_id=selected_id,
                score=selected_score,
                candidates_evaluated=len(self._agents),
                rejection_reasons=rejection_reasons,
                fallback_agents=fallbacks,
                routing_time_ms=elapsed_ms,
            )
        else:
            return RoutingDecision(
                selected_agent_id=None,
                score=0.0,
                candidates_evaluated=len(self._agents),
                rejection_reasons=rejection_reasons,
                fallback_agents=[],
                routing_time_ms=elapsed_ms,
            )


# =============================================================================
# Utility Functions
# =============================================================================


def create_requirements(
    required_capabilities: list[str],
    preferred_capabilities: list[str] | None = None,
    compliance_level: Literal["standard", "high", "critical"] = "standard",
    cost_tier: Literal["free", "standard", "premium"] = "standard",
    max_latency_ms: int | None = None,
    required_domains: list[str] | None = None,
) -> AgentRequirements:
    """
    Factory function to create AgentRequirements with defaults.

    Args:
        required_capabilities: Capabilities the agent must have
        preferred_capabilities: Nice-to-have capabilities
        compliance_level: Required compliance tier
        cost_tier: Maximum acceptable cost tier
        max_latency_ms: Maximum p99 latency
        required_domains: Required domain access

    Returns:
        AgentRequirements TypedDict

    Example:
        >>> reqs = create_requirements(
        ...     required_capabilities=["web_search"],
        ...     preferred_capabilities=["query_expansion"],
        ...     max_latency_ms=5000,
        ... )
    """
    return AgentRequirements(
        required_capabilities=required_capabilities,
        preferred_capabilities=preferred_capabilities or [],
        compliance_level=compliance_level,
        cost_tier=cost_tier,
        max_latency_ms=max_latency_ms,
        required_domains=required_domains,
    )


def capabilities_match(
    agent_capabilities: list[str],
    required: list[str],
    preferred: list[str] | None = None,
) -> tuple[bool, float]:
    """
    Check if capabilities match and calculate match score.

    Args:
        agent_capabilities: Capabilities the agent has
        required: Required capabilities (all must match)
        preferred: Preferred capabilities (partial match OK)

    Returns:
        Tuple of (all_required_met, match_score)

    Example:
        >>> met, score = capabilities_match(
        ...     ["web_search", "query_expansion"],
        ...     ["web_search"],
        ...     ["query_expansion", "source_discovery"]
        ... )
        >>> print(f"Required met: {met}, Score: {score:.2f}")
    """
    agent_caps = set(agent_capabilities)
    required_set = set(required)

    # Check required capabilities
    all_required_met = required_set.issubset(agent_caps)

    if not all_required_met:
        return False, 0.0

    # Calculate score
    score = 0.5  # Base score for meeting requirements

    # Add bonus for preferred capabilities
    if preferred:
        preferred_set = set(preferred)
        matched_preferred = len(preferred_set & agent_caps)
        score += 0.5 * (matched_preferred / len(preferred_set))

    return True, score


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
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
]
