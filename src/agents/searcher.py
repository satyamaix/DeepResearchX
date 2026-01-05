"""
Searcher Agent for DRX Deep Research System.

Responsible for:
- Executing web searches using configured search tools
- Query expansion and refinement for better results
- Deduplicating and ranking search results
- Managing search across multiple providers
- Creating citation records from search results
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .base import (
    AgentResponse,
    BaseAgent,
    LLMClient,
    create_citation_id,
    timestamp_now,
)

if TYPE_CHECKING:
    from ..orchestrator.state import (
        AgentState,
        AgentType,
        CitationRecord,
        SubTask,
    )
    from ..tools.base import SearchResult, SearchTool

logger = logging.getLogger(__name__)


# =============================================================================
# Searcher Agent System Prompt
# =============================================================================

SEARCHER_SYSTEM_PROMPT = """You are an expert search query optimizer for a research system. Your role is to expand and refine search queries for better results.

## Your Task
Given a research query and context, generate optimized search queries that will:
1. Cover different aspects of the topic
2. Use varied terminology and synonyms
3. Target specific facts, data, and authoritative sources
4. Avoid redundancy while maximizing coverage

## Output Format
Respond with a JSON object containing search queries:
```json
{
  "queries": [
    {
      "query": "optimized search query 1",
      "focus": "what this query targets",
      "priority": 1
    },
    {
      "query": "optimized search query 2",
      "focus": "what this query targets",
      "priority": 2
    }
  ],
  "reasoning": "Brief explanation of query strategy"
}
```

## Guidelines
1. Generate 2-4 queries per request
2. Make queries specific and targeted
3. Include exact phrases in quotes for precision
4. Consider synonyms and alternative phrasings
5. Prioritize queries by expected usefulness
6. Avoid overly broad or generic queries

Respond ONLY with the JSON object."""


# =============================================================================
# Query Expansion Prompt Template
# =============================================================================

QUERY_EXPANSION_TEMPLATE = """## Research Context
Original query: {original_query}

## Task Description
{task_description}

## Focus Areas
{focus_areas}

## Constraints
- Max queries: {max_queries}
- Preferred domains: {preferred_domains}
- Excluded topics: {excluded_topics}

Generate optimized search queries for this research task."""


# =============================================================================
# Searcher Agent Implementation
# =============================================================================


class SearcherAgent(BaseAgent):
    """
    Web search agent that executes optimized search queries.

    The searcher expands user queries into multiple targeted searches,
    executes them using configured search tools, deduplicates results,
    and creates citation records for downstream processing.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_tools: list[SearchTool] | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_queries_per_task: int = 3,
        max_results_per_query: int = 5,
        enable_query_expansion: bool = True,
        dedup_threshold: float = 0.9,
    ):
        """
        Initialize the searcher agent.

        Args:
            llm_client: LLM client for query expansion
            search_tools: List of search tool instances
            model: Model identifier
            temperature: Sampling temperature
            max_queries_per_task: Max expanded queries per search task
            max_results_per_query: Max results per search query
            enable_query_expansion: Whether to use LLM for query expansion
            dedup_threshold: URL similarity threshold for deduplication
        """
        super().__init__(
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            max_output_tokens=2048,
        )
        self._search_tools = search_tools or []
        self._max_queries = max_queries_per_task
        self._max_results = max_results_per_query
        self._enable_expansion = enable_query_expansion
        self._dedup_threshold = dedup_threshold

        # Track seen URLs for deduplication across searches
        self._seen_urls: set[str] = set()
        self._url_hashes: set[str] = set()

    # =========================================================================
    # Required Abstract Properties
    # =========================================================================

    @property
    def name(self) -> str:
        return "searcher"

    @property
    def description(self) -> str:
        return (
            "Executes optimized web searches using multiple tools, "
            "with query expansion, deduplication, and citation tracking."
        )

    @property
    def agent_type(self) -> AgentType:
        return "searcher"

    @property
    def system_prompt(self) -> str:
        return SEARCHER_SYSTEM_PROMPT

    # =========================================================================
    # Tool Management
    # =========================================================================

    def add_search_tool(self, tool: SearchTool) -> None:
        """Add a search tool to the agent."""
        self._search_tools.append(tool)

    def clear_seen_urls(self) -> None:
        """Clear the URL deduplication cache."""
        self._seen_urls.clear()
        self._url_hashes.clear()

    # =========================================================================
    # Core Processing
    # =========================================================================

    async def _process(self, state: AgentState) -> AgentResponse:
        """
        Execute search tasks from the research plan.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with search results and citations
        """
        if not self._search_tools:
            return AgentResponse.error_response(
                "No search tools configured",
                self.name,
            )

        plan = state.get("plan")
        if not plan:
            return AgentResponse.error_response(
                "No research plan available",
                self.name,
            )

        # Find pending search tasks
        search_tasks = self._get_pending_search_tasks(plan)

        if not search_tasks:
            return AgentResponse.success_response(
                data={"results": [], "citations": []},
                agent_name=self.name,
                tokens_used=0,
            )

        # Process all search tasks
        all_results = []
        all_citations = []
        total_tokens = 0

        for task in search_tasks:
            result = await self._execute_search_task(task, state)
            if result["results"]:
                all_results.extend(result["results"])
                all_citations.extend(result["citations"])
            total_tokens += result.get("tokens_used", 0)

        # Deduplicate across all results
        unique_results = self._deduplicate_results(all_results)
        unique_citations = self._deduplicate_citations(all_citations)

        return AgentResponse.success_response(
            data={
                "results": unique_results,
                "citations": unique_citations,
                "tasks_processed": len(search_tasks),
            },
            agent_name=self.name,
            tokens_used=total_tokens,
        )

    async def _execute_search_task(
        self, task: SubTask, state: AgentState
    ) -> dict[str, Any]:
        """Execute a single search task."""
        task_inputs = task.get("inputs", {})
        base_query = task_inputs.get("query", task.get("description", ""))

        # Expand query if enabled
        if self._enable_expansion and self._llm_client:
            queries = await self._expand_query(base_query, task, state)
        else:
            queries = [{"query": base_query, "focus": "main", "priority": 1}]

        # Execute searches
        all_results = []
        for query_info in queries[: self._max_queries]:
            results = await self._execute_search(query_info["query"])
            for result in results:
                result.metadata["focus"] = query_info.get("focus", "")
                result.metadata["original_query"] = base_query
            all_results.extend(results)

        # Create citations from results
        citations = self._create_citations_from_results(all_results)

        return {
            "results": all_results,
            "citations": citations,
            "tokens_used": 0,  # Updated if query expansion used tokens
        }

    # =========================================================================
    # Query Expansion
    # =========================================================================

    async def _expand_query(
        self, base_query: str, task: SubTask, state: AgentState
    ) -> list[dict[str, Any]]:
        """Use LLM to expand a base query into multiple targeted queries."""
        steerability = state.get("steerability", {})

        prompt = QUERY_EXPANSION_TEMPLATE.format(
            original_query=state.get("user_query", base_query),
            task_description=task.get("description", ""),
            focus_areas=", ".join(steerability.get("focus_areas", [])) or "None specified",
            max_queries=self._max_queries,
            preferred_domains=", ".join(steerability.get("preferred_domains", [])) or "Any",
            excluded_topics=", ".join(steerability.get("exclude_topics", [])) or "None",
        )

        response = await self._call_llm(prompt, temperature=0.3)

        if not response.success:
            logger.warning(f"Query expansion failed: {response.error}")
            return [{"query": base_query, "focus": "main", "priority": 1}]

        try:
            import json
            data = self._extract_json(response.data)
            if data and "queries" in data:
                return data["queries"]
        except Exception as e:
            logger.warning(f"Failed to parse expanded queries: {e}")

        return [{"query": base_query, "focus": "main", "priority": 1}]

    # =========================================================================
    # Search Execution
    # =========================================================================

    async def _execute_search(self, query: str) -> list[SearchResult]:
        """Execute search across all configured tools."""
        if not self._search_tools:
            return []

        all_results = []

        # Execute searches in parallel across tools
        search_coros = [
            tool.search(query, max_results=self._max_results)
            for tool in self._search_tools
        ]

        try:
            results_per_tool = await asyncio.gather(
                *search_coros, return_exceptions=True
            )

            for i, results in enumerate(results_per_tool):
                if isinstance(results, Exception):
                    logger.warning(
                        f"Search tool {self._search_tools[i].name} failed: {results}"
                    )
                    continue
                if isinstance(results, list):
                    for result in results:
                        result.metadata["search_tool"] = self._search_tools[i].name
                    all_results.extend(results)

        except Exception as e:
            logger.exception(f"Search execution failed: {e}")

        return all_results

    # =========================================================================
    # Deduplication
    # =========================================================================

    def _deduplicate_results(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Remove duplicate results based on URL and content similarity."""
        unique_results = []

        for result in results:
            url_hash = self._hash_url(result.url)

            # Skip if we've seen this URL
            if url_hash in self._url_hashes:
                continue

            # Check content similarity with existing results
            is_duplicate = False
            for existing in unique_results:
                if self._is_similar_content(result, existing):
                    is_duplicate = True
                    break

            if not is_duplicate:
                self._url_hashes.add(url_hash)
                self._seen_urls.add(result.url)
                unique_results.append(result)

        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)

        return unique_results

    def _deduplicate_citations(
        self, citations: list[CitationRecord]
    ) -> list[CitationRecord]:
        """Remove duplicate citations based on URL."""
        seen_urls = set()
        unique_citations = []

        for citation in citations:
            if citation["url"] not in seen_urls:
                seen_urls.add(citation["url"])
                unique_citations.append(citation)

        return unique_citations

    def _hash_url(self, url: str) -> str:
        """Create a normalized hash for URL comparison."""
        # Normalize URL
        parsed = urlparse(url)
        normalized = f"{parsed.netloc}{parsed.path}".lower()
        normalized = re.sub(r"[/\s]+$", "", normalized)  # Remove trailing slashes

        return hashlib.md5(normalized.encode()).hexdigest()

    def _is_similar_content(
        self, result1: SearchResult, result2: SearchResult
    ) -> bool:
        """Check if two results have similar content."""
        # Simple Jaccard similarity on words
        words1 = set(result1.snippet.lower().split())
        words2 = set(result2.snippet.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / union if union > 0 else 0

        return similarity >= self._dedup_threshold

    # =========================================================================
    # Citation Creation
    # =========================================================================

    def _create_citations_from_results(
        self, results: list[SearchResult]
    ) -> list[CitationRecord]:
        """Create citation records from search results."""
        citations = []

        for result in results:
            parsed_url = urlparse(result.url)
            domain = parsed_url.netloc

            citation: CitationRecord = {
                "id": create_citation_id(),
                "url": result.url,
                "title": result.title,
                "snippet": result.snippet[:500],  # Truncate long snippets
                "relevance_score": result.score,
                "retrieved_at": timestamp_now(),
                "domain": domain,
                "retrieved_by": "searcher",
                "used_in_report": False,
            }

            citations.append(citation)

        return citations

    # =========================================================================
    # Task Management
    # =========================================================================

    def _get_pending_search_tasks(self, plan: ResearchPlan) -> list[SubTask]:
        """Get pending search tasks that are ready for execution."""
        if not plan or not plan.get("dag_nodes"):
            return []

        completed_ids = {
            task["id"]
            for task in plan["dag_nodes"]
            if task["status"] == "completed"
        }

        pending_search_tasks = []
        for task in plan["dag_nodes"]:
            if task["agent_type"] == "searcher" and task["status"] == "pending":
                # Check if all dependencies are completed
                deps_satisfied = all(
                    dep_id in completed_ids
                    for dep_id in task.get("dependencies", [])
                )
                if deps_satisfied:
                    pending_search_tasks.append(task)

        return pending_search_tasks

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _extract_json(self, response: str) -> dict[str, Any] | None:
        """Extract JSON from LLM response."""
        import json
        import re

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try code block extraction
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(code_block_pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Try finding JSON object
        json_pattern = r"\{[\s\S]*\}"
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    # =========================================================================
    # State Updates
    # =========================================================================

    async def _post_process(
        self, state: AgentState, response: AgentResponse
    ) -> AgentState:
        """Update state with search results and citations."""
        if not response.success:
            return state

        data = response.data
        new_citations = data.get("citations", [])

        # Merge new citations with existing
        existing_citations = state.get("citations", [])
        all_citations = existing_citations + new_citations

        # Update task status in plan
        plan = state.get("plan")
        if plan:
            for task in plan["dag_nodes"]:
                if task["agent_type"] == "searcher" and task["status"] == "pending":
                    task["status"] = "completed"
                    task["completed_at"] = timestamp_now()
                    task["outputs"] = {
                        "result_count": len(data.get("results", [])),
                        "citation_ids": [c["id"] for c in new_citations],
                    }

        return {
            **state,
            "citations": all_citations,
            "plan": plan,
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_searcher_agent(
    llm_client: LLMClient | None = None,
    search_tools: list[SearchTool] | None = None,
    **kwargs,
) -> SearcherAgent:
    """
    Factory function to create a configured SearcherAgent.

    Args:
        llm_client: LLM client for query expansion
        search_tools: List of search tool instances
        **kwargs: Additional configuration options

    Returns:
        Configured SearcherAgent instance
    """
    return SearcherAgent(
        llm_client=llm_client,
        search_tools=search_tools,
        **kwargs,
    )


# =============================================================================
# Type Alias for Import Compatibility
# =============================================================================

# Import SearchResult from tools.base for type hints
try:
    from ..tools.base import SearchResult
except ImportError:
    # Define minimal SearchResult if tools.base not available
    from dataclasses import dataclass, field

    @dataclass
    class SearchResult:
        url: str
        title: str
        snippet: str
        score: float = 0.0
        metadata: dict = field(default_factory=dict)


# Also need ResearchPlan type
try:
    from ..orchestrator.state import ResearchPlan
except ImportError:
    pass


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SearcherAgent",
    "create_searcher_agent",
    "SEARCHER_SYSTEM_PROMPT",
]
