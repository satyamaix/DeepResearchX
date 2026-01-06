"""
LangGraph Node Functions for DRX Deep Research Workflow.

This module contains all node functions that execute within the LangGraph
StateGraph workflow. Each node function:
- Takes AgentState as input
- Returns a partial AgentState update (dict)
- Is responsible for a specific step in the research pipeline

Node Categories:
- Core Research Nodes: plan_research, search_sources, read_documents
- Processing Nodes: synthesize_findings, critique_synthesis
- Safety Nodes: check_policy
- Output Nodes: generate_report

Conditional Edge Functions:
- should_continue: Determines if research loop should continue
- has_policy_violation: Routes based on policy compliance
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.orchestrator.state import (
    AgentState,
    Finding,
    CitationRecord,
    KnowledgeGraphState,
    QualityMetrics,
    ResearchPlan,
    SubTask,
    create_citation,
    create_finding,
    get_pending_tasks,
    is_plan_complete,
    update_tokens_used,
)

if TYPE_CHECKING:
    from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

NodeResult = dict[str, Any]
ConditionResult = Literal["continue", "report", "safe", "violation"]


# =============================================================================
# Agent Registry (Lazy Loading Pattern)
# =============================================================================

# Global agent instances - lazily initialized
_agents: dict[str, "BaseAgent"] = {}


async def get_agent(agent_type: str) -> "BaseAgent":
    """
    Get or create an agent instance by type.

    Uses lazy initialization pattern to avoid circular imports
    and defer agent creation until needed.

    Args:
        agent_type: Type of agent ('planner', 'searcher', 'reader',
                   'synthesizer', 'critic', 'reporter')

    Returns:
        Initialized agent instance.

    Raises:
        ValueError: If agent_type is not recognized.
    """
    if agent_type in _agents:
        return _agents[agent_type]

    # Import agents and LLM client lazily to avoid circular imports
    try:
        from src.agents.base import get_llm_client
        llm_client = await get_llm_client()

        if agent_type == "planner":
            from src.agents.planner import PlannerAgent
            _agents[agent_type] = PlannerAgent(llm_client=llm_client)
        elif agent_type == "searcher":
            from src.agents.searcher import SearcherAgent
            _agents[agent_type] = SearcherAgent(llm_client=llm_client)
        elif agent_type == "reader":
            from src.agents.reader import ReaderAgent
            _agents[agent_type] = ReaderAgent(llm_client=llm_client)
        elif agent_type == "synthesizer":
            from src.agents.synthesizer import SynthesizerAgent
            _agents[agent_type] = SynthesizerAgent(llm_client=llm_client)
        elif agent_type == "critic":
            from src.agents.critic import CriticAgent
            _agents[agent_type] = CriticAgent(llm_client=llm_client)
        elif agent_type == "reporter":
            from src.agents.reporter import ReporterAgent
            _agents[agent_type] = ReporterAgent(llm_client=llm_client)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return _agents[agent_type]

    except ImportError as e:
        logger.warning(f"Agent {agent_type} not yet implemented: {e}")
        # Return a placeholder that raises NotImplementedError
        raise NotImplementedError(
            f"Agent '{agent_type}' is not yet implemented. "
            f"Please implement src/agents/{agent_type}.py"
        ) from e


def register_agent(agent_type: str, agent: "BaseAgent") -> None:
    """
    Register an agent instance manually.

    Useful for testing and dependency injection.

    Args:
        agent_type: Type identifier for the agent.
        agent: Agent instance to register.
    """
    _agents[agent_type] = agent


def clear_agents() -> None:
    """Clear all registered agents. Useful for testing."""
    _agents.clear()


# =============================================================================
# Core Research Node Functions
# =============================================================================


async def plan_research(state: AgentState) -> NodeResult:
    """
    Create or refine the research plan based on the user query.

    The planner agent:
    - Decomposes the query into sub-questions
    - Creates a DAG of research subtasks
    - On subsequent iterations, refines the plan based on gaps

    Args:
        state: Current workflow state.

    Returns:
        Partial state update with new/updated research plan.
    """
    logger.info(f"[{state['session_id']}] Planning research for: {state['user_query'][:50]}...")

    try:
        planner = await get_agent("planner")
        result = await planner.invoke(state)

        # Extract plan from agent result
        plan = result.get("plan")
        messages = result.get("messages", [])
        tokens = result.get("tokens_used", 0)

        # Update token tracking
        token_update = update_tokens_used(state, tokens)

        logger.info(
            f"[{state['session_id']}] Plan created with "
            f"{len(plan['dag_nodes']) if plan else 0} tasks"
        )

        return {
            "plan": plan,
            "messages": messages,
            "current_phase": "researching",
            **token_update,
        }

    except NotImplementedError:
        # Fallback: create a basic plan structure
        logger.warning("Planner agent not implemented, using fallback plan")
        return _create_fallback_plan(state)

    except Exception as e:
        logger.error(f"[{state['session_id']}] Planning failed: {e}")
        return {
            "error": str(e),
            "current_phase": "failed",
            "should_terminate": True,
        }


async def search_sources(state: AgentState) -> NodeResult:
    """
    Execute search tasks to find relevant sources.

    Processes all pending search tasks in the plan:
    - Executes searches in parallel (fan-out)
    - Aggregates results (fan-in)
    - Updates plan with search results

    Args:
        state: Current workflow state.

    Returns:
        Partial state update with search results and updated plan.
    """
    logger.info(f"[{state['session_id']}] Searching for sources...")

    if state["plan"] is None:
        logger.error("No plan available for search")
        return {"error": "No research plan available"}

    try:
        searcher = await get_agent("searcher")

        # Get pending search tasks
        search_tasks = [
            task for task in get_pending_tasks(state["plan"])
            if task["agent_type"] == "searcher"
        ]

        if not search_tasks:
            logger.info("No pending search tasks")
            return {"current_phase": "synthesizing"}

        # Execute searches in parallel using fan-out pattern
        results = await _parallel_execute(
            searcher,
            state,
            search_tasks,
            max_concurrent=5,
        )

        # Aggregate citations and update plan
        new_citations: list[CitationRecord] = []
        updated_plan = _update_plan_with_results(state["plan"], results)

        for result in results:
            if result.get("citations"):
                new_citations.extend(result["citations"])

        # Calculate token usage
        total_tokens = sum(r.get("tokens_used", 0) for r in results)
        token_update = update_tokens_used(state, total_tokens)

        logger.info(
            f"[{state['session_id']}] Search complete: "
            f"{len(new_citations)} citations found"
        )

        return {
            "plan": updated_plan,
            "citations": state["citations"] + new_citations,
            **token_update,
        }

    except NotImplementedError:
        logger.warning("Searcher agent not implemented, using fallback")
        return _create_fallback_citations(state)

    except Exception as e:
        logger.error(f"[{state['session_id']}] Search failed: {e}")
        return {"error": str(e)}


async def read_documents(state: AgentState) -> NodeResult:
    """
    Read and extract information from retrieved sources.

    The reader agent:
    - Processes URLs from search results
    - Extracts relevant content
    - Creates structured findings

    Args:
        state: Current workflow state.

    Returns:
        Partial state update with extracted findings.
    """
    logger.info(f"[{state['session_id']}] Reading documents...")

    if not state["citations"]:
        logger.info("No citations to read")
        return {"current_phase": "synthesizing"}

    try:
        reader = await get_agent("reader")

        # Get pending read tasks
        read_tasks = []
        if state["plan"]:
            read_tasks = [
                task for task in get_pending_tasks(state["plan"])
                if task["agent_type"] == "reader"
            ]

        # If no explicit read tasks, create implicit ones from citations
        if not read_tasks:
            read_tasks = _create_read_tasks_from_citations(state["citations"])

        # Execute reads in parallel
        results = await _parallel_execute(
            reader,
            state,
            read_tasks,
            max_concurrent=10,
        )

        # Aggregate findings
        new_findings: list[Finding] = []
        for result in results:
            if result.get("findings"):
                new_findings.extend(result["findings"])

        # Update plan if present
        updated_plan = state["plan"]
        if updated_plan:
            updated_plan = _update_plan_with_results(updated_plan, results)

        # Token tracking
        total_tokens = sum(r.get("tokens_used", 0) for r in results)
        token_update = update_tokens_used(state, total_tokens)

        logger.info(
            f"[{state['session_id']}] Reading complete: "
            f"{len(new_findings)} findings extracted"
        )

        return {
            "plan": updated_plan,
            "findings": state["findings"] + new_findings,
            "current_phase": "synthesizing",
            **token_update,
        }

    except NotImplementedError:
        logger.warning("Reader agent not implemented, using fallback")
        return _create_fallback_findings(state)

    except Exception as e:
        logger.error(f"[{state['session_id']}] Reading failed: {e}")
        return {"error": str(e)}


async def synthesize_findings(state: AgentState) -> NodeResult:
    """
    Synthesize findings into coherent analysis.

    The synthesizer agent:
    - Combines findings from multiple sources
    - Identifies patterns and themes
    - Creates a preliminary synthesis
    - Builds knowledge graph from entities, relations, and claims

    Args:
        state: Current workflow state.

    Returns:
        Partial state update with synthesis text and knowledge graph.
    """
    logger.info(f"[{state['session_id']}] Synthesizing findings...")

    if not state["findings"]:
        logger.warning("No findings to synthesize")
        return {
            "synthesis": "No findings available for synthesis.",
            "current_phase": "critiquing",
        }

    try:
        synthesizer = await get_agent("synthesizer")
        result = await synthesizer.invoke(state)

        synthesis = result.get("synthesis", "")
        messages = result.get("messages", [])
        tokens = result.get("tokens_used", 0)

        token_update = update_tokens_used(state, tokens)

        # Extract knowledge graph from synthesizer result if available
        # The synthesizer may build a KnowledgeGraph internally; we serialize it
        # for checkpointing compatibility with AsyncPostgresSaver
        knowledge_graph_update = _extract_knowledge_graph(result, state)

        logger.info(
            f"[{state['session_id']}] Synthesis complete: "
            f"{len(synthesis)} characters, "
            f"KG: {len(knowledge_graph_update.get('knowledge_graph', {}).get('entities', []))} entities"
        )

        return {
            "synthesis": synthesis,
            "messages": messages,
            "current_phase": "critiquing",
            **knowledge_graph_update,
            **token_update,
        }

    except NotImplementedError:
        logger.warning("Synthesizer agent not implemented, using fallback")
        return _create_fallback_synthesis(state)

    except Exception as e:
        logger.error(f"[{state['session_id']}] Synthesis failed: {e}")
        return {"error": str(e), "current_phase": "critiquing"}


async def critique_synthesis(state: AgentState) -> NodeResult:
    """
    Evaluate synthesis quality and identify gaps.

    The critic agent:
    - Assesses coverage of the original query
    - Identifies knowledge gaps requiring more research
    - Provides quality metrics

    Args:
        state: Current workflow state.

    Returns:
        Partial state update with gaps and quality metrics.
    """
    logger.info(f"[{state['session_id']}] Critiquing synthesis...")

    try:
        critic = await get_agent("critic")
        result = await critic.invoke(state)

        gaps = result.get("gaps", [])
        quality_metrics = result.get("quality_metrics")
        messages = result.get("messages", [])
        tokens = result.get("tokens_used", 0)

        token_update = update_tokens_used(state, tokens)

        # Increment iteration count
        new_iteration = state["iteration_count"] + 1

        logger.info(
            f"[{state['session_id']}] Critique complete: "
            f"{len(gaps)} gaps identified, iteration {new_iteration}"
        )

        return {
            "gaps": gaps,
            "quality_metrics": quality_metrics,
            "messages": messages,
            "iteration_count": new_iteration,
            **token_update,
        }

    except NotImplementedError:
        logger.warning("Critic agent not implemented, using fallback")
        return {
            "gaps": [],
            "quality_metrics": _create_fallback_metrics(state),
            "iteration_count": state["iteration_count"] + 1,
        }

    except Exception as e:
        logger.error(f"[{state['session_id']}] Critique failed: {e}")
        return {
            "error": str(e),
            "gaps": [],
            "iteration_count": state["iteration_count"] + 1,
        }


async def check_policy(state: AgentState) -> NodeResult:
    """
    Check synthesis and findings for policy compliance.

    Evaluates content against:
    - Harmful content policies
    - Copyright/attribution requirements
    - Factual accuracy standards

    Args:
        state: Current workflow state.

    Returns:
        Partial state update with policy violations if any.
    """
    logger.info(f"[{state['session_id']}] Checking policy compliance...")

    violations: list[str] = []

    # Check for empty or minimal content
    if not state["synthesis"] or len(state["synthesis"]) < 100:
        violations.append("Insufficient synthesis content")

    # Check for proper citations
    if not state["citations"]:
        violations.append("No citations provided")
    elif len(state["citations"]) < 2:
        violations.append("Insufficient source diversity (< 2 sources)")

    # Check for unverified findings
    unverified_count = sum(
        1 for f in state["findings"]
        if not f.get("verified", False)
    )
    if unverified_count > len(state["findings"]) * 0.5:
        violations.append(
            f"High proportion of unverified findings ({unverified_count}/{len(state['findings'])})"
        )

    # Add custom policy checks here
    # Example: Check for harmful content patterns
    # Example: Check for proper attribution

    if violations:
        logger.warning(
            f"[{state['session_id']}] Policy violations: {violations}"
        )
    else:
        logger.info(f"[{state['session_id']}] Policy check passed")

    return {
        "policy_violations": violations,
        "blocked": len(violations) > 0 and any(
            "harmful" in v.lower() or "blocked" in v.lower()
            for v in violations
        ),
    }


async def generate_report(state: AgentState) -> NodeResult:
    """
    Generate the final research report.

    The reporter agent:
    - Formats synthesis according to steerability params
    - Adds proper citations and attributions
    - Creates executive summary if requested

    Args:
        state: Current workflow state.

    Returns:
        Partial state update with final report.
    """
    logger.info(f"[{state['session_id']}] Generating final report...")

    # Check for blocking policy violations
    if state.get("blocked", False):
        logger.warning(f"[{state['session_id']}] Report blocked due to policy")
        return {
            "final_report": None,
            "error": "Report generation blocked due to policy violations",
            "current_phase": "failed",
            "should_terminate": True,
        }

    try:
        reporter = await get_agent("reporter")
        result = await reporter.invoke(state)

        final_report = result.get("final_report", "")
        messages = result.get("messages", [])
        tokens = result.get("tokens_used", 0)

        token_update = update_tokens_used(state, tokens)

        logger.info(
            f"[{state['session_id']}] Report generated: "
            f"{len(final_report)} characters"
        )

        return {
            "final_report": final_report,
            "messages": messages,
            "current_phase": "complete",
            "should_terminate": True,
            **token_update,
        }

    except NotImplementedError:
        logger.warning("Reporter agent not implemented, using fallback")
        return _create_fallback_report(state)

    except Exception as e:
        logger.error(f"[{state['session_id']}] Report generation failed: {e}")
        return {
            "error": str(e),
            "current_phase": "failed",
            "should_terminate": True,
        }


# =============================================================================
# Conditional Edge Functions
# =============================================================================


def should_continue(state: AgentState) -> Literal["continue", "report"]:
    """
    Determine if research loop should continue or proceed to report.

    Decision logic:
    - Continue if gaps exist AND iteration < max_iterations AND budget available
    - Report otherwise

    Args:
        state: Current workflow state.

    Returns:
        "continue" to loop back to search, "report" to generate final output.
    """
    # Check termination conditions
    if state.get("should_terminate", False):
        logger.info(f"[{state['session_id']}] Termination flag set -> report")
        return "report"

    # Check iteration limit
    max_iterations = state.get("max_iterations", 5)
    if state["iteration_count"] >= max_iterations:
        logger.info(
            f"[{state['session_id']}] Max iterations reached "
            f"({state['iteration_count']}/{max_iterations}) -> report"
        )
        return "report"

    # Check token budget
    if state["tokens_remaining"] < 10000:  # Reserve for report generation
        logger.info(
            f"[{state['session_id']}] Token budget exhausted "
            f"({state['tokens_remaining']} remaining) -> report"
        )
        return "report"

    # Check for gaps
    if not state.get("gaps"):
        logger.info(f"[{state['session_id']}] No gaps identified -> report")
        return "report"

    # Check quality threshold
    metrics = state.get("quality_metrics")
    if metrics and metrics.get("coverage_score", 0) >= 0.9:
        logger.info(
            f"[{state['session_id']}] Quality threshold met "
            f"(coverage: {metrics['coverage_score']:.2f}) -> report"
        )
        return "report"

    # Continue research loop
    logger.info(
        f"[{state['session_id']}] Gaps remain, continuing research loop "
        f"(iteration {state['iteration_count'] + 1})"
    )
    return "continue"


def has_policy_violation(state: AgentState) -> Literal["safe", "violation"]:
    """
    Route based on policy compliance status.

    Args:
        state: Current workflow state.

    Returns:
        "safe" if no violations, "violation" if policy issues detected.
    """
    if state.get("blocked", False):
        logger.warning(f"[{state['session_id']}] Blocked by policy")
        return "violation"

    violations = state.get("policy_violations", [])
    if violations:
        # Check severity - some violations are warnings, not blockers
        critical_violations = [
            v for v in violations
            if "harmful" in v.lower() or "blocked" in v.lower()
        ]
        if critical_violations:
            logger.warning(
                f"[{state['session_id']}] Critical policy violations: {critical_violations}"
            )
            return "violation"

    return "safe"


# =============================================================================
# Parallel Execution Helpers (Fan-Out/Fan-In Pattern)
# =============================================================================


async def _parallel_execute(
    agent: "BaseAgent",
    state: AgentState,
    tasks: list[SubTask],
    max_concurrent: int = 5,
) -> list[dict[str, Any]]:
    """
    Execute multiple tasks in parallel with concurrency limit.

    Implements fan-out/fan-in pattern:
    - Fan-out: Create concurrent tasks up to limit
    - Fan-in: Aggregate all results

    Args:
        agent: Agent to execute tasks.
        state: Current workflow state (shared context).
        tasks: List of subtasks to execute.
        max_concurrent: Maximum concurrent executions.

    Returns:
        List of results from all task executions.
    """
    if not tasks:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[dict[str, Any]] = []

    async def execute_with_semaphore(task: SubTask) -> dict[str, Any]:
        async with semaphore:
            try:
                # Create task-specific state
                task_state = {
                    **state,
                    "current_task": task,
                }
                result = await agent.invoke(task_state)
                return {
                    "task_id": task["id"],
                    "success": True,
                    **result,
                }
            except Exception as e:
                logger.error(f"Task {task['id']} failed: {e}")
                return {
                    "task_id": task["id"],
                    "success": False,
                    "error": str(e),
                }

    # Execute all tasks with concurrency limit
    async_tasks = [execute_with_semaphore(task) for task in tasks]
    results = await asyncio.gather(*async_tasks)

    return list(results)


async def parallel_node_execution(
    nodes: list[Callable[[AgentState], NodeResult]],
    state: AgentState,
) -> NodeResult:
    """
    Execute multiple nodes in parallel and merge results.

    Useful for independent operations that can run concurrently.

    Args:
        nodes: List of node functions to execute.
        state: Current workflow state.

    Returns:
        Merged results from all nodes.
    """
    async def run_node(node: Callable) -> NodeResult:
        if asyncio.iscoroutinefunction(node):
            return await node(state)
        return node(state)

    results = await asyncio.gather(*[run_node(n) for n in nodes])

    # Merge all results
    merged: NodeResult = {}
    for result in results:
        for key, value in result.items():
            if key in merged:
                # Handle list merging
                if isinstance(merged[key], list) and isinstance(value, list):
                    merged[key].extend(value)
                # Handle dict merging
                elif isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
                # Last value wins for other types
                else:
                    merged[key] = value
            else:
                merged[key] = value

    return merged


# =============================================================================
# Knowledge Graph Extraction
# =============================================================================


def _extract_knowledge_graph(
    result: dict[str, Any],
    state: AgentState,
) -> dict[str, KnowledgeGraphState]:
    """
    Extract knowledge graph from synthesizer result and serialize for checkpointing.

    The synthesizer agent may return a KnowledgeGraph object or raw entity/relation
    data. This function normalizes the output into a KnowledgeGraphState TypedDict
    that can be checkpointed by AsyncPostgresSaver.

    Args:
        result: Result from synthesizer agent invoke()
        state: Current workflow state (for merging with existing KG)

    Returns:
        Dict containing the updated knowledge_graph field
    """
    # Get existing knowledge graph from state (or empty if first synthesis)
    existing_kg = state.get("knowledge_graph", {
        "entities": [],
        "relations": [],
        "claims": [],
    })

    # Extract knowledge graph from result
    # The synthesizer may provide:
    # 1. A "knowledge_graph" key with serialized data
    # 2. An "_kg" or "kg" key with a KnowledgeGraph object
    # 3. Separate "entities", "relations", "claims" keys
    new_entities: list = []
    new_relations: list = []
    new_claims: list = []

    if "knowledge_graph" in result:
        kg_data = result["knowledge_graph"]
        if isinstance(kg_data, dict):
            new_entities = kg_data.get("entities", [])
            new_relations = kg_data.get("relations", [])
            new_claims = kg_data.get("claims", [])
        elif hasattr(kg_data, "_entities"):
            # It's a KnowledgeGraph object - extract values
            new_entities = list(kg_data._entities.values())
            new_relations = list(kg_data._relations.values())
            new_claims = list(kg_data._claims.values())

    elif "_kg" in result or "kg" in result:
        kg_obj = result.get("_kg") or result.get("kg")
        if hasattr(kg_obj, "_entities"):
            new_entities = list(kg_obj._entities.values())
            new_relations = list(kg_obj._relations.values())
            new_claims = list(kg_obj._claims.values())

    elif "entities" in result or "relations" in result or "claims" in result:
        new_entities = result.get("entities", [])
        new_relations = result.get("relations", [])
        new_claims = result.get("claims", [])

    # Merge with existing, avoiding duplicates by ID
    existing_entity_ids = {e["id"] for e in existing_kg.get("entities", [])}
    existing_relation_ids = {r["id"] for r in existing_kg.get("relations", [])}
    existing_claim_ids = {c["id"] for c in existing_kg.get("claims", [])}

    merged_entities = list(existing_kg.get("entities", []))
    merged_relations = list(existing_kg.get("relations", []))
    merged_claims = list(existing_kg.get("claims", []))

    for entity in new_entities:
        if entity.get("id") not in existing_entity_ids:
            merged_entities.append(entity)
            existing_entity_ids.add(entity["id"])

    for relation in new_relations:
        if relation.get("id") not in existing_relation_ids:
            merged_relations.append(relation)
            existing_relation_ids.add(relation["id"])

    for claim in new_claims:
        if claim.get("id") not in existing_claim_ids:
            merged_claims.append(claim)
            existing_claim_ids.add(claim["id"])

    return {
        "knowledge_graph": KnowledgeGraphState(
            entities=merged_entities,
            relations=merged_relations,
            claims=merged_claims,
        )
    }


# =============================================================================
# Fallback Implementations (Used When Agents Not Yet Implemented)
# =============================================================================


def _create_fallback_plan(state: AgentState) -> NodeResult:
    """Create a basic research plan when planner is not available."""
    now = datetime.utcnow().isoformat() + "Z"

    plan: ResearchPlan = {
        "dag_nodes": [
            {
                "id": "search-1",
                "description": f"Search for: {state['user_query']}",
                "agent_type": "searcher",
                "dependencies": [],
                "status": "pending",
                "inputs": {"query": state["user_query"]},
                "outputs": None,
                "quality_score": None,
                "started_at": None,
                "completed_at": None,
                "error": None,
            },
        ],
        "current_iteration": 1,
        "max_iterations": state.get("max_iterations", 5),
        "coverage_score": None,
        "created_at": now,
        "updated_at": now,
        "sub_questions": [state["user_query"]],
    }

    return {
        "plan": plan,
        "current_phase": "researching",
    }


def _create_fallback_citations(state: AgentState) -> NodeResult:
    """Create placeholder citations when searcher is not available."""
    return {
        "citations": [],
        "current_phase": "synthesizing",
    }


def _create_fallback_findings(state: AgentState) -> NodeResult:
    """Create placeholder findings when reader is not available."""
    return {
        "findings": [],
        "current_phase": "synthesizing",
    }


def _create_fallback_synthesis(state: AgentState) -> NodeResult:
    """Create basic synthesis when synthesizer is not available."""
    findings_text = "\n".join(
        f"- {f['claim']}" for f in state.get("findings", [])
    )

    synthesis = f"""
## Research Summary

Query: {state['user_query']}

### Findings:
{findings_text if findings_text else "No findings available."}

### Sources:
{len(state.get('citations', []))} sources consulted.
"""

    return {
        "synthesis": synthesis.strip(),
        "current_phase": "critiquing",
    }


def _create_fallback_metrics(state: AgentState) -> QualityMetrics:
    """Create basic quality metrics."""
    now = datetime.utcnow().isoformat() + "Z"

    findings = state.get("findings", [])
    verified = sum(1 for f in findings if f.get("verified", False))

    return {
        "coverage_score": 0.5,
        "avg_confidence": sum(
            f.get("confidence_score", 0.5) for f in findings
        ) / max(len(findings), 1),
        "verified_findings": verified,
        "total_findings": len(findings),
        "unique_sources": len(set(
            c["url"] for c in state.get("citations", [])
        )),
        "citation_density": 0.0,
        "consistency_score": 0.5,
        "updated_at": now,
    }


def _create_fallback_report(state: AgentState) -> NodeResult:
    """Generate basic report when reporter is not available."""
    steerability = state.get("steerability", {})
    tone = steerability.get("tone", "technical")

    # Build citations section
    citations_md = "\n".join(
        f"- [{c['title']}]({c['url']})"
        for c in state.get("citations", [])[:10]
    )

    report = f"""
# Research Report

## Query
{state['user_query']}

## Summary
{state.get('synthesis', 'No synthesis available.')}

## Sources
{citations_md if citations_md else 'No sources cited.'}

---
*Generated by DRX Deep Research System*
*Iteration count: {state['iteration_count']}*
*Tokens used: {state['tokens_used']}*
"""

    return {
        "final_report": report.strip(),
        "current_phase": "complete",
        "should_terminate": True,
    }


def _update_plan_with_results(
    plan: ResearchPlan,
    results: list[dict[str, Any]],
) -> ResearchPlan:
    """Update plan with task execution results."""
    now = datetime.utcnow().isoformat() + "Z"

    # Create a mutable copy
    updated_nodes = list(plan["dag_nodes"])

    for result in results:
        task_id = result.get("task_id")
        if not task_id:
            continue

        for i, node in enumerate(updated_nodes):
            if node["id"] == task_id:
                updated_nodes[i] = {
                    **node,
                    "status": "completed" if result.get("success") else "failed",
                    "outputs": result if result.get("success") else None,
                    "error": result.get("error") if not result.get("success") else None,
                    "completed_at": now,
                }
                break

    return {
        **plan,
        "dag_nodes": updated_nodes,
        "updated_at": now,
    }


def _create_read_tasks_from_citations(
    citations: list[CitationRecord],
) -> list[SubTask]:
    """Create read tasks from citation URLs."""
    tasks: list[SubTask] = []

    for i, citation in enumerate(citations[:20]):  # Limit to 20 reads
        tasks.append({
            "id": f"read-{i}",
            "description": f"Read content from {citation['domain']}",
            "agent_type": "reader",
            "dependencies": [],
            "status": "pending",
            "inputs": {
                "url": citation["url"],
                "citation_id": citation["id"],
            },
            "outputs": None,
            "quality_score": None,
            "started_at": None,
            "completed_at": None,
            "error": None,
        })

    return tasks


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Node functions
    "plan_research",
    "search_sources",
    "read_documents",
    "synthesize_findings",
    "critique_synthesis",
    "check_policy",
    "generate_report",
    # Conditional edge functions
    "should_continue",
    "has_policy_violation",
    # Parallel execution helpers
    "parallel_node_execution",
    # Agent management
    "get_agent",
    "register_agent",
    "clear_agents",
]
