"""
Planner Agent for DRX Deep Research System.

Responsible for:
- Decomposing user queries into research sub-questions
- Creating a DAG of subtasks with dependencies
- Optimizing task execution order for parallelism
- Adapting plans based on gap analysis feedback
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .base import (
    AgentResponse,
    BaseAgent,
    LLMClient,
    create_subtask_id,
    timestamp_now,
)

if TYPE_CHECKING:
    from ..orchestrator.state import (
        AgentState,
        AgentType,
        ResearchPlan,
        SubTask,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Planner Agent System Prompt
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are an expert research planner for a deep research system. Your role is to decompose complex queries into a structured research plan.

## Your Capabilities
1. Break down complex questions into atomic, answerable sub-questions
2. Identify dependencies between research tasks
3. Optimize for parallel execution where possible
4. Create search queries that will yield comprehensive results
5. Adapt plans based on feedback and gap analysis

## Output Format
You must respond with a valid JSON object containing:
```json
{
  "sub_questions": [
    "First sub-question to investigate",
    "Second sub-question to investigate"
  ],
  "tasks": [
    {
      "id": "search_1",
      "description": "Search for X to understand Y",
      "agent_type": "searcher",
      "dependencies": [],
      "inputs": {
        "query": "specific search query",
        "focus": "what aspect to focus on"
      }
    },
    {
      "id": "read_1",
      "description": "Extract key findings about Z",
      "agent_type": "reader",
      "dependencies": ["search_1"],
      "inputs": {
        "focus_areas": ["area1", "area2"],
        "extract_entities": true
      }
    }
  ],
  "reasoning": "Brief explanation of the research strategy"
}
```

## Task Types
- **searcher**: Web search tasks. Inputs: query, focus, max_results
- **reader**: Document extraction tasks. Inputs: focus_areas, extract_entities, content_type
- **synthesizer**: Aggregation tasks. Inputs: synthesis_type, conflict_resolution
- **critic**: Quality review tasks. Inputs: review_aspects, threshold

## Guidelines
1. Start with broad searches, then narrow based on initial findings
2. Create parallel search tasks for independent sub-questions
3. Add reader tasks after each search to extract structured information
4. Include synthesizer tasks to aggregate findings across sources
5. Add critic tasks to validate important claims
6. Limit total tasks to 10-15 for efficiency
7. Ensure all task IDs are unique and dependencies reference valid IDs

## Steerability Integration
Consider the user's preferences:
- Focus areas: Prioritize these topics
- Excluded topics: Avoid these subjects
- Preferred domains: Favor these sources
- Tone: Adjust depth and technicality accordingly

Respond ONLY with the JSON object, no additional text."""


# =============================================================================
# Gap-Aware Replanning Prompt
# =============================================================================

REPLANNING_PROMPT_TEMPLATE = """You are replanning a research task based on gap analysis feedback.

## Original Query
{user_query}

## Current Progress
- Iteration: {iteration}/{max_iterations}
- Findings so far: {findings_count}
- Coverage score: {coverage_score}

## Identified Gaps
{gaps}

## Previous Plan Summary
{previous_plan_summary}

## Instructions
Create additional tasks to address the identified gaps. Focus on:
1. Filling knowledge gaps with targeted searches
2. Verifying uncertain claims
3. Exploring underrepresented aspects
4. Finding alternative perspectives

Respond with the same JSON format as before, but only include NEW tasks needed to address gaps.
The task IDs should continue from where the previous plan left off (e.g., if last task was search_3, start with search_4)."""


# =============================================================================
# Planner Agent Implementation
# =============================================================================


class PlannerAgent(BaseAgent):
    """
    Research planning agent that decomposes queries into DAG-structured plans.

    The planner analyzes user queries and creates optimized research plans
    consisting of search, read, synthesis, and critique tasks organized
    as a directed acyclic graph for efficient parallel execution.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        temperature: float = 0.3,  # Lower temperature for structured output
        max_tasks: int = 15,
        min_tasks: int = 3,
    ):
        """
        Initialize the planner agent.

        Args:
            llm_client: LLM client for API calls
            model: Model identifier
            temperature: Sampling temperature (lower for consistency)
            max_tasks: Maximum tasks to generate in a plan
            min_tasks: Minimum tasks required for a valid plan
        """
        super().__init__(
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            max_output_tokens=4096,
        )
        self._max_tasks = max_tasks
        self._min_tasks = min_tasks

    # =========================================================================
    # Required Abstract Properties
    # =========================================================================

    @property
    def name(self) -> str:
        return "planner"

    @property
    def description(self) -> str:
        return (
            "Decomposes complex research queries into structured DAG plans "
            "with optimized task dependencies for parallel execution."
        )

    @property
    def agent_type(self) -> AgentType:
        return "planner"

    @property
    def system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT

    # =========================================================================
    # Core Processing
    # =========================================================================

    async def _process(self, state: AgentState) -> AgentResponse:
        """
        Generate or update the research plan.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with the research plan
        """
        # Check if this is initial planning or replanning
        if state["plan"] is None or state["iteration_count"] == 0:
            return await self._create_initial_plan(state)
        else:
            return await self._create_replan(state)

    async def _create_initial_plan(self, state: AgentState) -> AgentResponse:
        """Create the initial research plan."""
        user_message = self._format_initial_planning_prompt(state)

        response = await self._call_llm(user_message, temperature=0.3)

        if not response.success:
            return response

        try:
            plan = self._parse_plan_response(response.data, state)
            return AgentResponse.success_response(
                data=plan,
                agent_name=self.name,
                tokens_used=response.tokens_used,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms,
                model=response.model,
            )
        except Exception as e:
            logger.exception(f"Failed to parse plan: {e}")
            return AgentResponse.error_response(
                f"Failed to parse planning response: {e}",
                self.name,
            )

    async def _create_replan(self, state: AgentState) -> AgentResponse:
        """Create additional tasks based on gap analysis."""
        if not state["gaps"]:
            # No gaps, return existing plan
            return AgentResponse.success_response(
                data=state["plan"],
                agent_name=self.name,
                tokens_used=0,
            )

        user_message = self._format_replanning_prompt(state)

        response = await self._call_llm(user_message, temperature=0.3)

        if not response.success:
            return response

        try:
            updated_plan = self._merge_replan_response(
                response.data, state["plan"], state
            )
            return AgentResponse.success_response(
                data=updated_plan,
                agent_name=self.name,
                tokens_used=response.tokens_used,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms,
                model=response.model,
            )
        except Exception as e:
            logger.exception(f"Failed to parse replan: {e}")
            return AgentResponse.error_response(
                f"Failed to parse replanning response: {e}",
                self.name,
            )

    # =========================================================================
    # Prompt Formatting
    # =========================================================================

    def _format_initial_planning_prompt(self, state: AgentState) -> str:
        """Format the initial planning prompt with user query and preferences."""
        steerability = state["steerability"]

        prompt_parts = [
            f"## Research Query\n{state['user_query']}",
            "",
            "## User Preferences",
        ]

        if steerability.get("focus_areas"):
            prompt_parts.append(
                f"- Focus areas: {', '.join(steerability['focus_areas'])}"
            )

        if steerability.get("exclude_topics"):
            prompt_parts.append(
                f"- Exclude: {', '.join(steerability['exclude_topics'])}"
            )

        if steerability.get("preferred_domains"):
            prompt_parts.append(
                f"- Preferred sources: {', '.join(steerability['preferred_domains'])}"
            )

        prompt_parts.extend([
            f"- Output tone: {steerability.get('tone', 'technical')}",
            f"- Output format: {steerability.get('format', 'markdown')}",
            f"- Max sources: {steerability.get('max_sources', 20)}",
        ])

        if steerability.get("custom_instructions"):
            prompt_parts.append(
                f"\n## Custom Instructions\n{steerability['custom_instructions']}"
            )

        prompt_parts.extend([
            "",
            "## Task",
            "Create a comprehensive research plan to answer this query.",
            f"Generate between {self._min_tasks} and {self._max_tasks} tasks.",
        ])

        return "\n".join(prompt_parts)

    def _format_replanning_prompt(self, state: AgentState) -> str:
        """Format the replanning prompt with gap information."""
        plan = state["plan"]

        # Summarize previous plan
        task_summary = []
        for task in plan["dag_nodes"][:5]:  # First 5 tasks
            task_summary.append(
                f"- [{task['id']}] {task['description']} ({task['status']})"
            )

        gaps_formatted = "\n".join(f"- {gap}" for gap in state["gaps"])

        return REPLANNING_PROMPT_TEMPLATE.format(
            user_query=state["user_query"],
            iteration=state["iteration_count"],
            max_iterations=state["max_iterations"],
            findings_count=len(state["findings"]),
            coverage_score=plan.get("coverage_score", "N/A"),
            gaps=gaps_formatted,
            previous_plan_summary="\n".join(task_summary),
        )

    # =========================================================================
    # Response Parsing
    # =========================================================================

    def _parse_plan_response(
        self, response: str, state: AgentState
    ) -> ResearchPlan:
        """Parse LLM response into a ResearchPlan."""
        # Extract JSON from response
        json_data = self._extract_json(response)

        if not json_data:
            raise ValueError("No valid JSON found in response")

        # Validate required fields
        if "tasks" not in json_data:
            raise ValueError("Response missing 'tasks' field")

        tasks = json_data["tasks"]
        sub_questions = json_data.get("sub_questions", [])

        # Convert to SubTask format
        dag_nodes: list[SubTask] = []
        for task_data in tasks:
            subtask = self._create_subtask_from_dict(task_data)
            dag_nodes.append(subtask)

        # Validate DAG structure
        self._validate_dag(dag_nodes)

        now = timestamp_now()

        return {
            "dag_nodes": dag_nodes,
            "current_iteration": 1,
            "max_iterations": state["max_iterations"],
            "coverage_score": None,
            "created_at": now,
            "updated_at": now,
            "sub_questions": sub_questions,
        }

    def _merge_replan_response(
        self, response: str, existing_plan: ResearchPlan, state: AgentState
    ) -> ResearchPlan:
        """Merge new tasks from replan into existing plan."""
        json_data = self._extract_json(response)

        if not json_data or "tasks" not in json_data:
            # No new tasks, return existing plan
            return existing_plan

        new_tasks = json_data["tasks"]

        # Create SubTasks for new tasks
        new_subtasks = []
        for task_data in new_tasks:
            subtask = self._create_subtask_from_dict(task_data)
            new_subtasks.append(subtask)

        # Merge with existing tasks
        all_nodes = existing_plan["dag_nodes"] + new_subtasks

        # Validate merged DAG
        self._validate_dag(all_nodes)

        # Add new sub-questions if provided
        new_sub_questions = json_data.get("sub_questions", [])
        all_sub_questions = existing_plan["sub_questions"] + new_sub_questions

        return {
            **existing_plan,
            "dag_nodes": all_nodes,
            "updated_at": timestamp_now(),
            "sub_questions": all_sub_questions,
            "current_iteration": state["iteration_count"] + 1,
        }

    def _create_subtask_from_dict(self, task_data: dict[str, Any]) -> SubTask:
        """Create a SubTask from parsed JSON data."""
        task_id = task_data.get("id", create_subtask_id())
        agent_type = task_data.get("agent_type", "searcher")

        # Validate agent type
        valid_types = {"searcher", "reader", "synthesizer", "critic"}
        if agent_type not in valid_types:
            agent_type = "searcher"

        return {
            "id": task_id,
            "description": task_data.get("description", "Unnamed task"),
            "agent_type": agent_type,
            "dependencies": task_data.get("dependencies", []),
            "status": "pending",
            "inputs": task_data.get("inputs", {}),
            "outputs": None,
            "quality_score": None,
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

    def _extract_json(self, response: str) -> dict[str, Any] | None:
        """Extract JSON object from LLM response."""
        # Try to parse the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code blocks
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(code_block_pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Try to find JSON object pattern
        json_pattern = r"\{[\s\S]*\}"
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    # =========================================================================
    # DAG Validation
    # =========================================================================

    def _validate_dag(self, nodes: list[SubTask]) -> None:
        """
        Validate that the task graph is a valid DAG.

        Checks:
        1. All dependencies reference existing task IDs
        2. No circular dependencies
        3. At least one root task (no dependencies)

        Raises:
            ValueError: If validation fails
        """
        task_ids = {node["id"] for node in nodes}

        # Check dependency references
        for node in nodes:
            for dep_id in node["dependencies"]:
                if dep_id not in task_ids:
                    raise ValueError(
                        f"Task '{node['id']}' has invalid dependency '{dep_id}'"
                    )

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            node = next(n for n in nodes if n["id"] == node_id)
            for dep_id in node["dependencies"]:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node in nodes:
            if node["id"] not in visited:
                if has_cycle(node["id"]):
                    raise ValueError("Circular dependency detected in task graph")

        # Check for at least one root task
        root_tasks = [n for n in nodes if not n["dependencies"]]
        if not root_tasks:
            raise ValueError("No root tasks found (all tasks have dependencies)")

    # =========================================================================
    # State Updates
    # =========================================================================

    async def _post_process(
        self, state: AgentState, response: AgentResponse
    ) -> AgentState:
        """Update state with the new research plan."""
        if not response.success:
            return state

        plan = response.data

        return {
            **state,
            "plan": plan,
            "current_phase": "researching",
            "iteration_count": plan.get("current_iteration", 1),
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_planner_agent(
    llm_client: LLMClient | None = None,
    **kwargs,
) -> PlannerAgent:
    """
    Factory function to create a configured PlannerAgent.

    Args:
        llm_client: LLM client for API calls
        **kwargs: Additional configuration options

    Returns:
        Configured PlannerAgent instance
    """
    return PlannerAgent(llm_client=llm_client, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PlannerAgent",
    "create_planner_agent",
    "PLANNER_SYSTEM_PROMPT",
]
