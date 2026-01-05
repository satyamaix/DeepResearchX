"""
Reader Agent for DRX Deep Research System.

Responsible for:
- Extracting structured claims and facts from documents
- Identifying key entities, numbers, and relationships
- Creating Finding objects with evidence
- Handling different content types (web pages, PDFs, etc.)
- Assessing source credibility
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
    create_finding_id,
    timestamp_now,
)

if TYPE_CHECKING:
    from ..orchestrator.state import (
        AgentState,
        AgentType,
        CitationRecord,
        Finding,
        SubTask,
        ResearchPlan,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Reader Agent System Prompt
# =============================================================================

READER_SYSTEM_PROMPT = """You are an expert information extraction agent for a research system. Your role is to extract structured facts and claims from source documents.

## Your Capabilities
1. Identify factual claims and statements
2. Extract key entities (people, organizations, places, dates)
3. Capture numerical data and statistics
4. Assess the credibility of claims
5. Identify supporting evidence for each claim

## Output Format
Respond with a JSON object containing extracted findings:
```json
{
  "findings": [
    {
      "claim": "Clear statement of the fact or claim",
      "evidence": "Direct quote or paraphrase supporting this claim",
      "confidence": 0.85,
      "entities": ["Entity1", "Entity2"],
      "numbers": ["$1.5 billion", "25%"],
      "tags": ["category1", "category2"]
    }
  ],
  "source_assessment": {
    "credibility": 0.8,
    "bias_indicators": ["any detected bias"],
    "content_type": "news_article|research_paper|blog|official_source|other",
    "date_relevance": "current|recent|dated|unknown"
  },
  "extraction_notes": "Any issues or limitations in extraction"
}
```

## Guidelines
1. Extract 3-8 findings per document
2. Each claim should be atomic and verifiable
3. Include direct quotes as evidence when possible
4. Assign confidence based on source quality and evidence strength
5. Tag findings by topic/category for organization
6. Note any conflicting information
7. Identify the most important/relevant findings

## Confidence Scoring
- 0.9-1.0: Multiple high-quality sources, direct evidence
- 0.7-0.89: Single reliable source, clear evidence
- 0.5-0.69: Reasonable source, indirect evidence
- 0.3-0.49: Questionable source or weak evidence
- 0.0-0.29: Speculative or unsupported

Respond ONLY with the JSON object."""


# =============================================================================
# Document Analysis Prompt Template
# =============================================================================

DOCUMENT_ANALYSIS_TEMPLATE = """## Research Context
Original query: {user_query}

## Source Information
- URL: {source_url}
- Title: {source_title}
- Domain: {source_domain}

## Content to Analyze
{content}

## Focus Areas
{focus_areas}

## Instructions
Extract all relevant facts, claims, and data that help answer the research query.
Focus on: {extraction_focus}

Pay special attention to:
1. Key facts directly answering the query
2. Statistical data and numbers
3. Expert opinions and quotes
4. Dates and timeline information
5. Relationships between entities"""


# =============================================================================
# Batch Extraction Prompt Template
# =============================================================================

BATCH_EXTRACTION_TEMPLATE = """## Research Context
Original query: {user_query}

## Sources to Analyze
{sources_content}

## Focus Areas
{focus_areas}

## Instructions
Extract all relevant facts, claims, and data from each source.
For each finding, note which source it came from by referencing the source URL.

Extract key information that helps answer the research query."""


# =============================================================================
# Reader Agent Implementation
# =============================================================================


class ReaderAgent(BaseAgent):
    """
    Document reading and information extraction agent.

    The reader analyzes documents from search results, extracts
    structured claims and facts, assesses source credibility,
    and creates Finding objects for synthesis.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        temperature: float = 0.2,  # Low temperature for factual extraction
        max_findings_per_doc: int = 8,
        min_confidence_threshold: float = 0.3,
        max_content_length: int = 8000,
        batch_size: int = 5,
    ):
        """
        Initialize the reader agent.

        Args:
            llm_client: LLM client for extraction
            model: Model identifier
            temperature: Sampling temperature (low for accuracy)
            max_findings_per_doc: Maximum findings to extract per document
            min_confidence_threshold: Minimum confidence to include finding
            max_content_length: Maximum content length to process
            batch_size: Number of documents to process in batch
        """
        super().__init__(
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            max_output_tokens=4096,
        )
        self._max_findings = max_findings_per_doc
        self._min_confidence = min_confidence_threshold
        self._max_content_length = max_content_length
        self._batch_size = batch_size

    # =========================================================================
    # Required Abstract Properties
    # =========================================================================

    @property
    def name(self) -> str:
        return "reader"

    @property
    def description(self) -> str:
        return (
            "Extracts structured claims, facts, and evidence from documents "
            "with credibility assessment and entity recognition."
        )

    @property
    def agent_type(self) -> AgentType:
        return "reader"

    @property
    def system_prompt(self) -> str:
        return READER_SYSTEM_PROMPT

    # =========================================================================
    # Core Processing
    # =========================================================================

    async def _process(self, state: AgentState) -> AgentResponse:
        """
        Process documents and extract findings.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with extracted findings
        """
        citations = state.get("citations", [])

        if not citations:
            return AgentResponse.success_response(
                data={"findings": [], "processed_count": 0},
                agent_name=self.name,
                tokens_used=0,
            )

        # Get pending reader tasks
        plan = state.get("plan")
        reader_tasks = self._get_pending_reader_tasks(plan) if plan else []

        # If no specific tasks, process unread citations in batches
        if not reader_tasks:
            unprocessed = self._get_unprocessed_citations(citations, state)
            if not unprocessed:
                return AgentResponse.success_response(
                    data={"findings": [], "processed_count": 0},
                    agent_name=self.name,
                    tokens_used=0,
                )
            return await self._process_citations_batch(unprocessed, state)

        # Process each task
        all_findings = []
        total_tokens = 0
        processed_count = 0

        for task in reader_tasks:
            result = await self._execute_reader_task(task, citations, state)
            all_findings.extend(result.get("findings", []))
            total_tokens += result.get("tokens_used", 0)
            processed_count += result.get("processed_count", 0)

        return AgentResponse.success_response(
            data={
                "findings": all_findings,
                "processed_count": processed_count,
            },
            agent_name=self.name,
            tokens_used=total_tokens,
        )

    async def _process_citations_batch(
        self,
        citations: list[CitationRecord],
        state: AgentState,
    ) -> AgentResponse:
        """Process citations in batches for efficiency."""
        all_findings = []
        total_tokens = 0
        processed_count = 0

        # Process in batches
        for i in range(0, len(citations), self._batch_size):
            batch = citations[i : i + self._batch_size]
            result = await self._extract_from_batch(batch, state)

            if result.success:
                all_findings.extend(result.data.get("findings", []))
                total_tokens += result.tokens_used
                processed_count += len(batch)

        return AgentResponse.success_response(
            data={
                "findings": all_findings,
                "processed_count": processed_count,
            },
            agent_name=self.name,
            tokens_used=total_tokens,
        )

    async def _execute_reader_task(
        self,
        task: SubTask | dict,
        citations: list[CitationRecord],
        state: AgentState,
    ) -> dict[str, Any]:
        """Execute a single reader task."""
        task_inputs = task.get("inputs", {})
        focus_areas = task_inputs.get("focus_areas", [])
        extraction_focus = task_inputs.get("extraction_focus", "all relevant information")

        # Get citations to process for this task
        citation_ids = task_inputs.get("citation_ids")
        if citation_ids:
            target_citations = [c for c in citations if c["id"] in citation_ids]
        else:
            target_citations = citations

        # Process in batches
        all_findings = []
        total_tokens = 0
        processed = 0

        for i in range(0, len(target_citations), self._batch_size):
            batch = target_citations[i : i + self._batch_size]
            result = await self._extract_from_batch(
                batch, state, focus_areas, extraction_focus
            )

            if result.success:
                all_findings.extend(result.data.get("findings", []))
                total_tokens += result.tokens_used
                processed += len(batch)

        return {
            "findings": all_findings,
            "tokens_used": total_tokens,
            "processed_count": processed,
        }

    async def _extract_from_batch(
        self,
        citations: list[CitationRecord],
        state: AgentState,
        focus_areas: list[str] | None = None,
        extraction_focus: str = "all relevant information",
    ) -> AgentResponse:
        """Extract findings from a batch of citations."""
        # Build combined content
        sources_content_parts = []
        for citation in citations:
            content = self._get_citation_content(citation)
            if content:
                sources_content_parts.append(
                    f"### Source: {citation.get('title', 'Unknown')}\n"
                    f"URL: {citation.get('url', '')}\n"
                    f"Domain: {citation.get('domain', '')}\n"
                    f"Content:\n{content}\n"
                )

        if not sources_content_parts:
            return AgentResponse.success_response(
                data={"findings": []},
                agent_name=self.name,
                tokens_used=0,
            )

        sources_content = "\n---\n".join(sources_content_parts)

        # Truncate if too long
        if len(sources_content) > self._max_content_length:
            sources_content = sources_content[: self._max_content_length] + "\n...[truncated]"

        steerability = state.get("steerability", {})
        combined_focus = focus_areas or steerability.get("focus_areas", [])

        prompt = BATCH_EXTRACTION_TEMPLATE.format(
            user_query=state.get("user_query", ""),
            sources_content=sources_content,
            focus_areas=", ".join(combined_focus) if combined_focus else "All relevant information",
        )

        response = await self._call_llm(prompt, temperature=0.2)

        if not response.success:
            return response

        try:
            data = self._extract_json(response.data)
            if not data:
                return AgentResponse.error_response(
                    "Failed to parse extraction response",
                    self.name,
                )

            # Convert to Finding objects
            findings = self._create_findings_from_batch_extraction(
                data, citations, state
            )

            return AgentResponse.success_response(
                data={"findings": findings},
                agent_name=self.name,
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.exception(f"Extraction parsing failed: {e}")
            return AgentResponse.error_response(str(e), self.name)

    async def _extract_from_document(
        self,
        content: str,
        citation: CitationRecord,
        state: AgentState,
        focus_areas: list[str],
        extraction_focus: str,
    ) -> AgentResponse:
        """Extract findings from a single document."""
        # Truncate content if needed
        if len(content) > self._max_content_length:
            content = content[: self._max_content_length] + "\n...[truncated]"

        prompt = DOCUMENT_ANALYSIS_TEMPLATE.format(
            user_query=state.get("user_query", ""),
            source_url=citation.get("url", ""),
            source_title=citation.get("title", "Unknown"),
            source_domain=citation.get("domain", "unknown"),
            content=content,
            focus_areas=", ".join(focus_areas) if focus_areas else "All relevant information",
            extraction_focus=extraction_focus,
        )

        response = await self._call_llm(prompt, temperature=0.2)

        if not response.success:
            return response

        try:
            data = self._extract_json(response.data)
            if not data:
                return AgentResponse.error_response(
                    "Failed to parse extraction response",
                    self.name,
                )

            # Convert to Finding objects
            findings = self._create_findings_from_extraction(
                data, citation, state
            )

            return AgentResponse.success_response(
                data={"findings": findings},
                agent_name=self.name,
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.exception(f"Extraction parsing failed: {e}")
            return AgentResponse.error_response(str(e), self.name)

    # =========================================================================
    # Finding Creation
    # =========================================================================

    def _create_findings_from_extraction(
        self,
        extraction_data: dict[str, Any],
        citation: CitationRecord,
        state: AgentState,
    ) -> list[Finding]:
        """Create Finding objects from single document extraction."""
        raw_findings = extraction_data.get("findings", [])
        source_assessment = extraction_data.get("source_assessment", {})

        # Apply source credibility adjustment
        source_credibility = source_assessment.get("credibility", 0.7)
        if isinstance(source_credibility, str):
            credibility_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
            source_credibility = credibility_map.get(source_credibility.lower(), 0.7)

        findings = []
        for raw in raw_findings[: self._max_findings]:
            # Calculate adjusted confidence
            raw_confidence = raw.get("confidence", 0.5)
            adjusted_confidence = (raw_confidence + source_credibility) / 2

            # Skip low confidence findings
            if adjusted_confidence < self._min_confidence:
                continue

            finding: Finding = {
                "id": create_finding_id(),
                "claim": raw.get("claim", ""),
                "evidence": raw.get("evidence", ""),
                "source_urls": [citation.get("url", "")],
                "citation_ids": [citation.get("id", "")],
                "confidence_score": round(adjusted_confidence, 2),
                "agent_source": "reader",
                "tags": raw.get("tags", []),
                "verified": False,
                "created_at": timestamp_now(),
            }

            # Add entity tags
            if "entities" in raw:
                finding["tags"].extend([f"entity:{e}" for e in raw["entities"][:3]])

            findings.append(finding)

        return findings

    def _create_findings_from_batch_extraction(
        self,
        extraction_data: dict[str, Any],
        citations: list[CitationRecord],
        state: AgentState,
    ) -> list[Finding]:
        """Create Finding objects from batch extraction."""
        raw_findings = extraction_data.get("findings", [])
        source_assessment = extraction_data.get("source_assessment", {})

        # Create URL to citation mapping
        url_to_citation = {c.get("url", ""): c for c in citations}

        # Default credibility
        default_credibility = source_assessment.get("credibility", 0.7)
        if isinstance(default_credibility, str):
            credibility_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
            default_credibility = credibility_map.get(default_credibility.lower(), 0.7)

        findings = []
        for raw in raw_findings:
            raw_confidence = raw.get("confidence", 0.5)
            adjusted_confidence = (raw_confidence + default_credibility) / 2

            if adjusted_confidence < self._min_confidence:
                continue

            # Try to match finding to source
            source_url = raw.get("source_url", "")
            matched_citation = url_to_citation.get(source_url)

            if matched_citation:
                source_urls = [matched_citation.get("url", "")]
                citation_ids = [matched_citation.get("id", "")]
            else:
                # Associate with all batch citations
                source_urls = [c.get("url", "") for c in citations]
                citation_ids = [c.get("id", "") for c in citations]

            finding: Finding = {
                "id": create_finding_id(),
                "claim": raw.get("claim", ""),
                "evidence": raw.get("evidence", ""),
                "source_urls": source_urls,
                "citation_ids": citation_ids,
                "confidence_score": round(adjusted_confidence, 2),
                "agent_source": "reader",
                "tags": raw.get("tags", []),
                "verified": False,
                "created_at": timestamp_now(),
            }

            if "entities" in raw:
                finding["tags"].extend([f"entity:{e}" for e in raw["entities"][:3]])

            findings.append(finding)

        return findings

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_citation_content(self, citation: CitationRecord) -> str:
        """Get content to process from a citation."""
        return citation.get("snippet", "")

    def _get_pending_reader_tasks(
        self, plan: ResearchPlan | None
    ) -> list[SubTask]:
        """Get pending reader tasks from the plan."""
        if not plan or not plan.get("dag_nodes"):
            return []

        completed_ids = {
            task["id"]
            for task in plan["dag_nodes"]
            if task["status"] == "completed"
        }

        pending_tasks = []
        for task in plan["dag_nodes"]:
            if task["agent_type"] == "reader" and task["status"] == "pending":
                deps_satisfied = all(
                    dep_id in completed_ids
                    for dep_id in task.get("dependencies", [])
                )
                if deps_satisfied:
                    pending_tasks.append(task)

        return pending_tasks

    def _get_unprocessed_citations(
        self,
        citations: list[CitationRecord],
        state: AgentState,
    ) -> list[CitationRecord]:
        """Get citations that haven't been processed yet."""
        existing_findings = state.get("findings", [])
        processed_urls = set()

        for finding in existing_findings:
            processed_urls.update(finding.get("source_urls", []))

        return [c for c in citations if c.get("url") not in processed_urls]

    def _extract_json(self, response: str) -> dict[str, Any] | None:
        """Extract JSON from LLM response."""
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
        """Update state with extracted findings."""
        if not response.success:
            return state

        data = response.data
        new_findings = data.get("findings", [])

        # Merge new findings with existing
        existing_findings = state.get("findings", [])
        all_findings = existing_findings + new_findings

        # Update task status in plan
        plan = state.get("plan")
        if plan:
            for task in plan["dag_nodes"]:
                if task["agent_type"] == "reader" and task["status"] == "pending":
                    task["status"] = "completed"
                    task["completed_at"] = timestamp_now()
                    task["outputs"] = {
                        "findings_count": len(new_findings),
                        "finding_ids": [f["id"] for f in new_findings],
                    }

        return {
            **state,
            "findings": all_findings,
            "plan": plan,
        }


# =============================================================================
# Entity Extraction Utilities
# =============================================================================


def extract_entities_from_text(text: str) -> dict[str, list[str]]:
    """
    Extract basic entities from text using patterns.

    This is a simple extraction - in production, use spaCy or similar.

    Args:
        text: Text to extract entities from

    Returns:
        Dict with entity types and lists of extracted entities
    """
    entities = {
        "dates": [],
        "numbers": [],
        "urls": [],
        "emails": [],
    }

    # Date patterns
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{4}\b",  # Years
    ]
    for pattern in date_patterns:
        entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))

    # Number patterns (with units)
    number_pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|percent|%|USD|EUR))?"
    entities["numbers"] = re.findall(number_pattern, text, re.IGNORECASE)

    # URL pattern
    url_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"
    entities["urls"] = re.findall(url_pattern, text)

    # Email pattern
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    entities["emails"] = re.findall(email_pattern, text)

    return entities


# =============================================================================
# Factory Function
# =============================================================================


def create_reader_agent(
    llm_client: LLMClient | None = None,
    **kwargs,
) -> ReaderAgent:
    """
    Factory function to create a configured ReaderAgent.

    Args:
        llm_client: LLM client for extraction
        **kwargs: Additional configuration options

    Returns:
        Configured ReaderAgent instance
    """
    return ReaderAgent(llm_client=llm_client, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ReaderAgent",
    "create_reader_agent",
    "extract_entities_from_text",
    "READER_SYSTEM_PROMPT",
]
