"""
Synthesizer Agent for DRX Deep Research System.

Responsible for:
- Aggregating findings from multiple sources
- Resolving conflicting information
- Building argument graphs
- Generating intermediate summaries
- Identifying consensus and disagreements
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .base import (
    AgentResponse,
    BaseAgent,
    LLMClient,
    timestamp_now,
)

if TYPE_CHECKING:
    from ..orchestrator.state import (
        AgentState,
        AgentType,
        Finding,
        SubTask,
        ResearchPlan,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Synthesizer Agent System Prompt
# =============================================================================

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert research synthesizer for a deep research system. Your role is to aggregate findings from multiple sources into coherent, well-organized summaries.

## Your Capabilities
1. Aggregate claims across multiple sources
2. Identify consensus and areas of disagreement
3. Resolve conflicts using source quality and evidence strength
4. Create coherent narratives from disparate findings
5. Highlight areas of uncertainty
6. Build logical argument structures

## Output Format
Respond with a JSON object containing synthesis results:
```json
{
  "synthesis": {
    "main_findings": [
      {
        "finding": "Key synthesized finding statement",
        "confidence": 0.9,
        "supporting_count": 3,
        "source_ids": ["finding_id1", "finding_id2"],
        "importance": "high|medium|low"
      }
    ],
    "conflicts": [
      {
        "topic": "Area of disagreement",
        "position_a": "First position with sources",
        "position_b": "Second position with sources",
        "resolution": "How resolved or why unresolved",
        "confidence_in_resolution": 0.7
      }
    ],
    "themes": [
      {
        "name": "Theme name",
        "description": "Theme description",
        "related_findings": ["finding_id1"]
      }
    ],
    "narrative": "Comprehensive synthesis narrative integrating all key findings"
  },
  "coverage_score": 0.85,
  "confidence_score": 0.8,
  "gaps_identified": ["Gap 1", "Gap 2"],
  "synthesis_notes": "Any notes about the synthesis process"
}
```

## Synthesis Guidelines
1. Prioritize accuracy over completeness
2. Clearly indicate confidence levels
3. Note when claims have weak or conflicting evidence
4. Group related findings into themes
5. Create a logical flow in the narrative
6. Flag any potential biases or limitations
7. Identify remaining knowledge gaps

## Conflict Resolution Strategy
1. Prefer recent sources over dated ones
2. Prefer primary sources over secondary
3. Weight by source credibility
4. When unresolvable, present both positions

Respond ONLY with the JSON object."""


# =============================================================================
# Synthesis Prompt Templates
# =============================================================================

SYNTHESIS_TEMPLATE = """## Research Question
{user_query}

## Findings to Synthesize
{findings_formatted}

## Source Statistics
- Total citations: {citation_count}
- Total findings: {finding_count}
- High confidence findings: {high_confidence_count}
- Low confidence findings: {low_confidence_count}

## Focus Areas
{focus_areas}

## Previous Synthesis (if any)
{previous_synthesis}

## Instructions
Synthesize all findings into a coherent, comprehensive analysis.
Identify key themes, resolve conflicts, and create a unified narrative that addresses the research question."""


INCREMENTAL_SYNTHESIS_TEMPLATE = """## Research Question
{user_query}

## Existing Synthesis
{existing_synthesis}

## New Findings to Integrate
{new_findings_formatted}

## Instructions
Update the synthesis to incorporate the new findings.
Note any changes to previous conclusions and any new conflicts or themes."""


# =============================================================================
# Synthesizer Agent Implementation
# =============================================================================


class SynthesizerAgent(BaseAgent):
    """
    Synthesis agent that aggregates and reconciles research findings.

    The synthesizer combines findings from multiple sources, resolves
    conflicts, identifies themes, and creates coherent summaries for
    the final report.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_findings_per_batch: int = 20,
        conflict_threshold: float = 0.3,
        min_confidence_for_synthesis: float = 0.4,
    ):
        """
        Initialize the synthesizer agent.

        Args:
            llm_client: LLM client for synthesis
            model: Model identifier
            temperature: Sampling temperature
            max_findings_per_batch: Max findings to process at once
            conflict_threshold: Threshold for detecting conflicting claims
            min_confidence_for_synthesis: Minimum finding confidence to include
        """
        super().__init__(
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            max_output_tokens=8192,  # Larger for synthesis
        )
        self._max_findings_batch = max_findings_per_batch
        self._conflict_threshold = conflict_threshold
        self._min_confidence = min_confidence_for_synthesis

    # =========================================================================
    # Required Abstract Properties
    # =========================================================================

    @property
    def name(self) -> str:
        return "synthesizer"

    @property
    def description(self) -> str:
        return (
            "Aggregates findings from multiple sources, resolves conflicts, "
            "and generates coherent synthesis with argument structures."
        )

    @property
    def agent_type(self) -> AgentType:
        return "synthesizer"

    @property
    def system_prompt(self) -> str:
        return SYNTHESIZER_SYSTEM_PROMPT

    # =========================================================================
    # Core Processing
    # =========================================================================

    async def _process(self, state: AgentState) -> AgentResponse:
        """
        Synthesize findings into coherent summary.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with synthesis results
        """
        findings = state.get("findings", [])

        if not findings:
            return AgentResponse.success_response(
                data={
                    "synthesis": {"narrative": "No findings to synthesize."},
                    "confidence_score": 0.0,
                    "coverage_score": 0.0,
                    "gaps_identified": ["No findings available"],
                },
                agent_name=self.name,
                tokens_used=0,
            )

        # Filter findings by confidence
        valid_findings = [
            f for f in findings
            if f.get("confidence_score", 0) >= self._min_confidence
        ]

        if not valid_findings:
            return AgentResponse.success_response(
                data={
                    "synthesis": {"narrative": "All findings below confidence threshold."},
                    "confidence_score": 0.0,
                    "coverage_score": 0.0,
                    "gaps_identified": ["Need higher confidence findings"],
                },
                agent_name=self.name,
                tokens_used=0,
            )

        # Check if this is incremental synthesis
        existing_synthesis = state.get("synthesis", "")
        if existing_synthesis and len(existing_synthesis) > 100:
            return await self._incremental_synthesis(valid_findings, state)
        else:
            return await self._full_synthesis(valid_findings, state)

    async def _full_synthesis(
        self,
        findings: list[Finding],
        state: AgentState,
    ) -> AgentResponse:
        """Perform full synthesis of all findings."""
        steerability = state.get("steerability", {})

        # Pre-analyze findings for conflicts and themes
        analysis = self._analyze_findings(findings)

        # Format findings for LLM
        findings_formatted = self._format_findings(findings)

        # Calculate statistics
        high_conf = len([f for f in findings if f.get("confidence_score", 0) >= 0.7])
        low_conf = len([f for f in findings if f.get("confidence_score", 0) < 0.5])

        prompt = SYNTHESIS_TEMPLATE.format(
            user_query=state.get("user_query", ""),
            findings_formatted=findings_formatted,
            citation_count=len(state.get("citations", [])),
            finding_count=len(findings),
            high_confidence_count=high_conf,
            low_confidence_count=low_conf,
            focus_areas=", ".join(steerability.get("focus_areas", [])) or "General",
            previous_synthesis="None - initial synthesis",
        )

        response = await self._call_llm(prompt, temperature=0.3)

        if not response.success:
            return response

        try:
            data = self._extract_json(response.data)
            if not data:
                # Fallback: create synthesis from raw response
                data = self._create_fallback_synthesis(response.data, findings)

            # Enhance with pre-analysis
            if "synthesis" in data and analysis.get("potential_conflicts"):
                existing_conflicts = data["synthesis"].get("conflicts", [])
                data["synthesis"]["conflicts"] = self._merge_conflicts(
                    existing_conflicts, analysis["potential_conflicts"]
                )

            return AgentResponse.success_response(
                data={
                    **data,
                    "finding_ids_processed": [f["id"] for f in findings],
                    "synthesized_at": timestamp_now(),
                },
                agent_name=self.name,
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.exception(f"Synthesis parsing failed: {e}")
            return AgentResponse.error_response(str(e), self.name)

    async def _incremental_synthesis(
        self,
        findings: list[Finding],
        state: AgentState,
    ) -> AgentResponse:
        """Update existing synthesis with new findings."""
        existing_synthesis = state.get("synthesis", "")

        # Identify new findings not yet synthesized
        # (In practice, track processed finding IDs in state)
        new_findings = findings[-10:]  # Simple: take recent findings

        if not new_findings:
            return AgentResponse.success_response(
                data={
                    "synthesis": {"narrative": existing_synthesis},
                    "confidence_score": 0.7,
                    "coverage_score": 0.7,
                    "incremental": True,
                },
                agent_name=self.name,
                tokens_used=0,
            )

        new_findings_formatted = self._format_findings(new_findings)

        prompt = INCREMENTAL_SYNTHESIS_TEMPLATE.format(
            user_query=state.get("user_query", ""),
            existing_synthesis=existing_synthesis[:3000],  # Truncate if long
            new_findings_formatted=new_findings_formatted,
        )

        response = await self._call_llm(prompt, temperature=0.3)

        if not response.success:
            return response

        try:
            data = self._extract_json(response.data)
            if not data:
                data = self._create_fallback_synthesis(response.data, findings)

            return AgentResponse.success_response(
                data={
                    **data,
                    "incremental": True,
                    "new_findings_count": len(new_findings),
                    "synthesized_at": timestamp_now(),
                },
                agent_name=self.name,
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.exception(f"Incremental synthesis failed: {e}")
            return AgentResponse.error_response(str(e), self.name)

    # =========================================================================
    # Finding Analysis
    # =========================================================================

    def _analyze_findings(self, findings: list[Finding]) -> dict[str, Any]:
        """Pre-analyze findings for patterns, themes, and conflicts."""
        analysis = {
            "tag_distribution": defaultdict(int),
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "potential_conflicts": [],
            "themes": [],
        }

        # Count tags
        for finding in findings:
            for tag in finding.get("tags", []):
                analysis["tag_distribution"][tag] += 1

            # Categorize confidence
            conf = finding.get("confidence_score", 0)
            if conf >= 0.7:
                analysis["confidence_distribution"]["high"] += 1
            elif conf >= 0.5:
                analysis["confidence_distribution"]["medium"] += 1
            else:
                analysis["confidence_distribution"]["low"] += 1

        # Detect potential conflicts (simple heuristic)
        claims = [(f["id"], f.get("claim", "").lower()) for f in findings]
        conflict_keywords = [
            ("increase", "decrease"),
            ("rise", "fall"),
            ("growth", "decline"),
            ("positive", "negative"),
            ("support", "oppose"),
        ]

        for i, (id1, claim1) in enumerate(claims):
            for id2, claim2 in claims[i + 1:]:
                for word1, word2 in conflict_keywords:
                    if (word1 in claim1 and word2 in claim2) or \
                       (word2 in claim1 and word1 in claim2):
                        analysis["potential_conflicts"].append({
                            "finding_ids": [id1, id2],
                            "conflict_type": f"{word1} vs {word2}",
                        })

        # Identify themes from frequent tags
        common_tags = sorted(
            analysis["tag_distribution"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        analysis["themes"] = [tag for tag, count in common_tags if count >= 2]

        return analysis

    # =========================================================================
    # Formatting Helpers
    # =========================================================================

    def _format_findings(self, findings: list[Finding]) -> str:
        """Format findings for LLM consumption."""
        formatted_parts = []

        for i, finding in enumerate(findings, 1):
            formatted_parts.append(
                f"### Finding {i} (ID: {finding.get('id', 'unknown')})\n"
                f"- **Claim**: {finding.get('claim', 'N/A')}\n"
                f"- **Evidence**: {finding.get('evidence', 'N/A')}\n"
                f"- **Confidence**: {finding.get('confidence_score', 'N/A')}\n"
                f"- **Tags**: {', '.join(finding.get('tags', []))}\n"
                f"- **Sources**: {len(finding.get('source_urls', []))} source(s)\n"
                f"- **Verified**: {finding.get('verified', False)}"
            )

        return "\n\n".join(formatted_parts)

    def _create_fallback_synthesis(
        self,
        raw_response: str,
        findings: list[Finding],
    ) -> dict[str, Any]:
        """Create synthesis structure from raw LLM response."""
        return {
            "synthesis": {
                "narrative": raw_response,
                "main_findings": [],
                "conflicts": [],
                "themes": [],
            },
            "confidence_score": 0.5,
            "coverage_score": 0.5,
            "gaps_identified": [],
            "synthesis_notes": "Created from unstructured response",
        }

    def _merge_conflicts(
        self,
        llm_conflicts: list[dict],
        detected_conflicts: list[dict],
    ) -> list[dict]:
        """Merge LLM-identified conflicts with pre-detected ones."""
        merged = list(llm_conflicts)

        for detected in detected_conflicts:
            # Check if already covered
            already_covered = False
            for existing in merged:
                if detected["conflict_type"].lower() in existing.get("topic", "").lower():
                    already_covered = True
                    break

            if not already_covered:
                merged.append({
                    "topic": f"Detected: {detected['conflict_type']}",
                    "position_a": "See findings",
                    "position_b": "See findings",
                    "resolution": "Requires further analysis",
                    "finding_ids": detected["finding_ids"],
                })

        return merged

    # =========================================================================
    # Utility Methods
    # =========================================================================

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
        """Update state with synthesis results."""
        if not response.success:
            return state

        data = response.data
        synthesis_data = data.get("synthesis", {})
        narrative = synthesis_data.get("narrative", "")

        # Extract gaps if identified
        gaps = data.get("gaps_identified", [])

        # Update quality metrics if available
        quality_metrics = state.get("quality_metrics")
        if quality_metrics and "coverage_score" in data:
            quality_metrics["coverage_score"] = data["coverage_score"]
            quality_metrics["updated_at"] = timestamp_now()

        # Update task status in plan
        plan = state.get("plan")
        if plan:
            for task in plan["dag_nodes"]:
                if task["agent_type"] == "synthesizer" and task["status"] == "pending":
                    task["status"] = "completed"
                    task["completed_at"] = timestamp_now()
                    task["outputs"] = {
                        "confidence_score": data.get("confidence_score"),
                        "coverage_score": data.get("coverage_score"),
                    }

        return {
            **state,
            "synthesis": narrative,
            "gaps": gaps if gaps else state.get("gaps", []),
            "quality_metrics": quality_metrics,
            "plan": plan,
            "current_phase": "critiquing",
        }


# =============================================================================
# Argument Graph Builder
# =============================================================================


class ArgumentGraph:
    """
    Builds argument structures from findings.

    Creates a graph of claims, supporting evidence, and relationships
    for visualization and analysis.
    """

    def __init__(self):
        self.nodes: dict[str, dict] = {}  # id -> node data
        self.edges: list[dict] = []  # list of edges

    def add_finding(self, finding: Finding) -> str:
        """Add a finding as a node."""
        node_id = finding.get("id", "")
        self.nodes[node_id] = {
            "id": node_id,
            "type": "claim",
            "content": finding.get("claim", ""),
            "confidence": finding.get("confidence_score", 0),
            "evidence": finding.get("evidence", ""),
        }
        return node_id

    def add_support_edge(self, from_id: str, to_id: str, strength: float = 1.0):
        """Add a supporting relationship."""
        self.edges.append({
            "from": from_id,
            "to": to_id,
            "type": "supports",
            "strength": strength,
        })

    def add_conflict_edge(self, from_id: str, to_id: str, severity: float = 1.0):
        """Add a conflicting relationship."""
        self.edges.append({
            "from": from_id,
            "to": to_id,
            "type": "conflicts",
            "severity": severity,
        })

    def to_dict(self) -> dict[str, Any]:
        """Export graph as dictionary."""
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
        }

    @classmethod
    def from_findings(cls, findings: list[Finding]) -> ArgumentGraph:
        """Build argument graph from findings."""
        graph = cls()

        for finding in findings:
            graph.add_finding(finding)

        # Detect relationships based on shared tags/topics
        finding_list = list(findings)
        for i, f1 in enumerate(finding_list):
            for f2 in finding_list[i + 1:]:
                shared_tags = set(f1.get("tags", [])) & set(f2.get("tags", []))
                if shared_tags:
                    # Related findings - add support edge
                    graph.add_support_edge(
                        f1["id"],
                        f2["id"],
                        strength=len(shared_tags) / 5,
                    )

        return graph


# =============================================================================
# Factory Function
# =============================================================================


def create_synthesizer_agent(
    llm_client: LLMClient | None = None,
    **kwargs,
) -> SynthesizerAgent:
    """
    Factory function to create a configured SynthesizerAgent.

    Args:
        llm_client: LLM client for synthesis
        **kwargs: Additional configuration options

    Returns:
        Configured SynthesizerAgent instance
    """
    return SynthesizerAgent(llm_client=llm_client, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SynthesizerAgent",
    "create_synthesizer_agent",
    "ArgumentGraph",
    "SYNTHESIZER_SYSTEM_PROMPT",
]
