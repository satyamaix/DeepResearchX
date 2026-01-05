"""
DRX Agent Module.

Provides specialized agents for the deep research pipeline:
- PlannerAgent: Decomposes queries into research DAG plans
- SearcherAgent: Performs web searches via Tavily/OpenRouter
- ReaderAgent: Extracts and structures information from sources
- SynthesizerAgent: Aggregates findings and resolves conflicts
- CriticAgent: Reviews quality and identifies gaps
- ReporterAgent: Generates final research reports

Each agent inherits from BaseAgent and implements:
- system_prompt: Agent-specific instructions
- _process(): Core agent logic
- _post_process(): State update logic
"""

# Base classes and utilities
from .base import (
    AgentResponse,
    BaseAgent,
    LLMClient,
    create_citation_id,
    create_finding_id,
    create_subtask_id,
    timestamp_now,
)

# Planner Agent
from .planner import (
    PlannerAgent,
    create_planner_agent,
    PLANNER_SYSTEM_PROMPT,
)

# Searcher Agent
from .searcher import (
    SearcherAgent,
    create_searcher_agent,
    SEARCHER_SYSTEM_PROMPT,
)

# Reader Agent
from .reader import (
    ReaderAgent,
    create_reader_agent,
    extract_entities_from_text,
    READER_SYSTEM_PROMPT,
)

# Synthesizer Agent
from .synthesizer import (
    SynthesizerAgent,
    create_synthesizer_agent,
    ArgumentGraph,
    SYNTHESIZER_SYSTEM_PROMPT,
)

# Critic Agent
from .critic import (
    CriticAgent,
    create_critic_agent,
    verify_finding_sources,
    CRITIC_SYSTEM_PROMPT,
)

# Reporter Agent
from .reporter import (
    ReporterAgent,
    create_reporter_agent,
    format_citation_inline,
    format_citation_full,
    create_executive_summary_prompt,
    REPORTER_SYSTEM_PROMPT,
)


# =============================================================================
# Agent Registry
# =============================================================================

AGENT_REGISTRY = {
    "planner": PlannerAgent,
    "searcher": SearcherAgent,
    "reader": ReaderAgent,
    "synthesizer": SynthesizerAgent,
    "critic": CriticAgent,
    "reporter": ReporterAgent,
}

AGENT_FACTORIES = {
    "planner": create_planner_agent,
    "searcher": create_searcher_agent,
    "reader": create_reader_agent,
    "synthesizer": create_synthesizer_agent,
    "critic": create_critic_agent,
    "reporter": create_reporter_agent,
}


def get_agent_class(agent_type: str) -> type[BaseAgent]:
    """
    Get agent class by type name.

    Args:
        agent_type: Agent type identifier

    Returns:
        Agent class

    Raises:
        KeyError: If agent type not found
    """
    return AGENT_REGISTRY[agent_type]


def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    Create an agent instance by type name.

    Args:
        agent_type: Agent type identifier
        **kwargs: Agent configuration options

    Returns:
        Configured agent instance

    Raises:
        KeyError: If agent type not found
    """
    factory = AGENT_FACTORIES[agent_type]
    return factory(**kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentResponse",
    "LLMClient",
    # Utility functions
    "create_citation_id",
    "create_finding_id",
    "create_subtask_id",
    "timestamp_now",
    # Agent classes
    "PlannerAgent",
    "SearcherAgent",
    "ReaderAgent",
    "SynthesizerAgent",
    "CriticAgent",
    "ReporterAgent",
    # Factory functions
    "create_planner_agent",
    "create_searcher_agent",
    "create_reader_agent",
    "create_synthesizer_agent",
    "create_critic_agent",
    "create_reporter_agent",
    # System prompts
    "PLANNER_SYSTEM_PROMPT",
    "SEARCHER_SYSTEM_PROMPT",
    "READER_SYSTEM_PROMPT",
    "SYNTHESIZER_SYSTEM_PROMPT",
    "CRITIC_SYSTEM_PROMPT",
    "REPORTER_SYSTEM_PROMPT",
    # Utilities
    "ArgumentGraph",
    "extract_entities_from_text",
    "verify_finding_sources",
    "format_citation_inline",
    "format_citation_full",
    "create_executive_summary_prompt",
    # Registry
    "AGENT_REGISTRY",
    "AGENT_FACTORIES",
    "get_agent_class",
    "create_agent",
]
