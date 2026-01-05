"""
Reporter Agent for DRX Deep Research System.

Responsible for:
- Generating final structured research reports
- Formatting based on steerability preferences
- Including proper citations
- Creating executive summaries
- Respecting tone and format preferences
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
    timestamp_now,
)

if TYPE_CHECKING:
    from ..orchestrator.state import (
        AgentState,
        AgentType,
        CitationRecord,
        Finding,
        SteerabilityParams,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Reporter Agent System Prompts
# =============================================================================

REPORTER_SYSTEM_PROMPT = """You are an expert research report writer for a deep research system. Your role is to generate comprehensive, well-structured research reports.

## Your Capabilities
1. Create clear, well-organized reports
2. Adapt tone and format to user preferences
3. Include proper citations and references
4. Write executive summaries
5. Present findings with appropriate context
6. Handle multiple output formats

## Output Requirements
- Structure content logically with clear sections
- Use citations in [1], [2] format linking to sources
- Include an executive summary at the beginning
- Present key findings prominently
- Note limitations and areas of uncertainty
- Provide actionable conclusions when appropriate

## Quality Standards
- Accuracy: Only include well-supported claims
- Clarity: Use clear, precise language
- Completeness: Address all aspects of the query
- Objectivity: Present balanced perspectives
- Readability: Appropriate for target audience

Adapt your writing style based on the specified tone:
- **executive**: Concise, action-oriented, focus on key takeaways
- **technical**: Detailed, precise terminology, methodology focus
- **casual**: Accessible, conversational, general audience

Generate well-structured, professional content."""


# =============================================================================
# Report Generation Prompts by Format
# =============================================================================

MARKDOWN_REPORT_TEMPLATE = """## Report Parameters
- Query: {user_query}
- Tone: {tone}
- Max Sources: {max_sources}
- Language: {language}
- Focus Areas: {focus_areas}
{custom_instructions}

## Synthesis
{synthesis}

## Findings ({finding_count} total)
{findings_formatted}

## Sources ({citation_count} total)
{citations_formatted}

## Quality Metrics
- Coverage Score: {coverage_score}
- Confidence Score: {confidence_score}
- Unique Sources: {unique_sources}

## Instructions
Generate a comprehensive research report in Markdown format with:
1. Executive Summary (2-3 paragraphs)
2. Key Findings (bullet points with citations)
3. Detailed Analysis (multiple sections as needed)
4. Methodology Notes
5. Limitations and Caveats
6. Conclusion and Recommendations
7. References (numbered list of sources)

Use [1], [2], etc. for inline citations linking to the References section.
Match the specified tone throughout the report."""


JSON_REPORT_TEMPLATE = """## Report Parameters
- Query: {user_query}
- Tone: {tone}
- Focus Areas: {focus_areas}

## Synthesis
{synthesis}

## Findings
{findings_formatted}

## Sources
{citations_formatted}

## Instructions
Generate a research report as a JSON object with this structure:
{{
  "title": "Report title",
  "executive_summary": "2-3 paragraph summary",
  "key_findings": [
    {{
      "finding": "Key finding text",
      "confidence": 0.9,
      "citations": [1, 2]
    }}
  ],
  "sections": [
    {{
      "title": "Section Title",
      "content": "Section content with [1] citations"
    }}
  ],
  "methodology": "Brief methodology description",
  "limitations": ["Limitation 1", "Limitation 2"],
  "conclusion": "Final conclusion",
  "recommendations": ["Recommendation 1"],
  "references": [
    {{
      "id": 1,
      "title": "Source title",
      "url": "https://...",
      "accessed": "2024-01-01"
    }}
  ]
}}"""


MARKDOWN_TABLE_TEMPLATE = """## Report Parameters
- Query: {user_query}
- Tone: {tone}
- Focus Areas: {focus_areas}

## Synthesis
{synthesis}

## Findings
{findings_formatted}

## Sources
{citations_formatted}

## Instructions
Generate a research report in Markdown format optimized for tables:
1. Executive Summary
2. Key Findings Table:
   | Finding | Confidence | Sources | Evidence |
   |---------|------------|---------|----------|
   | ...     | ...        | ...     | ...      |
3. Comparative Analysis Tables where appropriate
4. Summary Statistics Table
5. Source Quality Table
6. Conclusion

Use clear section headers and appropriate tables for data presentation."""


# =============================================================================
# Reporter Agent Implementation
# =============================================================================


class ReporterAgent(BaseAgent):
    """
    Report generation agent that creates final research outputs.

    The reporter takes the synthesized research findings and generates
    well-formatted reports according to user steerability preferences
    for tone, format, and focus.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        temperature: float = 0.4,  # Slightly higher for better prose
        max_report_tokens: int = 8000,
        include_methodology: bool = True,
        include_limitations: bool = True,
    ):
        """
        Initialize the reporter agent.

        Args:
            llm_client: LLM client for report generation
            model: Model identifier
            temperature: Sampling temperature
            max_report_tokens: Maximum tokens for report generation
            include_methodology: Whether to include methodology section
            include_limitations: Whether to include limitations section
        """
        super().__init__(
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            max_output_tokens=max_report_tokens,
        )
        self._include_methodology = include_methodology
        self._include_limitations = include_limitations

    # =========================================================================
    # Required Abstract Properties
    # =========================================================================

    @property
    def name(self) -> str:
        return "reporter"

    @property
    def description(self) -> str:
        return (
            "Generates final research reports with proper formatting, "
            "citations, and steerability-based customization."
        )

    @property
    def agent_type(self) -> AgentType:
        return "reporter"

    @property
    def system_prompt(self) -> str:
        return REPORTER_SYSTEM_PROMPT

    # =========================================================================
    # Core Processing
    # =========================================================================

    async def _process(self, state: AgentState) -> AgentResponse:
        """
        Generate the final research report.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with the generated report
        """
        synthesis = state.get("synthesis", "")
        findings = state.get("findings", [])
        citations = state.get("citations", [])
        steerability = state.get("steerability", {})

        if not synthesis and not findings:
            return AgentResponse.error_response(
                "No synthesis or findings available for report generation",
                self.name,
            )

        # Prepare citations for references
        used_citations = self._prepare_citations(citations, findings)

        # Mark citations as used
        for citation in used_citations:
            citation["used_in_report"] = True

        # Format the report prompt based on format preference
        output_format = steerability.get("format", "markdown")
        prompt = self._format_report_prompt(
            state, steerability, used_citations, output_format
        )

        # Generate the report
        response = await self._call_llm(prompt, temperature=0.4)

        if not response.success:
            return response

        # Post-process the report
        report = self._post_process_report(
            response.data,
            output_format,
            used_citations,
            state,
        )

        return AgentResponse.success_response(
            data={
                "report": report,
                "format": output_format,
                "citations_used": len(used_citations),
                "findings_included": len(findings),
            },
            agent_name=self.name,
            tokens_used=response.tokens_used,
        )

    # =========================================================================
    # Citation Preparation
    # =========================================================================

    def _prepare_citations(
        self,
        citations: list[CitationRecord],
        findings: list[Finding],
    ) -> list[CitationRecord]:
        """
        Prepare citations for the report, numbering them sequentially.

        Args:
            citations: All available citations
            findings: Findings to include in report

        Returns:
            List of citations to include, numbered
        """
        # Get all citation IDs referenced in findings
        referenced_ids = set()
        for finding in findings:
            referenced_ids.update(finding.get("citation_ids", []))

        # Filter and sort citations
        used_citations = []
        seen_urls = set()

        # First, include citations referenced by findings
        for citation in citations:
            if citation["id"] in referenced_ids and citation["url"] not in seen_urls:
                used_citations.append(citation)
                seen_urls.add(citation["url"])

        # Then add remaining high-relevance citations
        remaining = [
            c for c in citations
            if c["url"] not in seen_urls and c.get("relevance_score", 0) >= 0.5
        ]
        remaining.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        for citation in remaining[:10]:  # Limit additional citations
            used_citations.append(citation)
            seen_urls.add(citation["url"])

        # Add reference numbers
        for i, citation in enumerate(used_citations, 1):
            citation["reference_number"] = i

        return used_citations

    # =========================================================================
    # Prompt Formatting
    # =========================================================================

    def _format_report_prompt(
        self,
        state: AgentState,
        steerability: SteerabilityParams,
        citations: list[CitationRecord],
        output_format: str,
    ) -> str:
        """Format the report generation prompt."""
        findings = state.get("findings", [])
        quality_metrics = state.get("quality_metrics", {})

        # Format findings
        findings_formatted = self._format_findings_for_prompt(findings)

        # Format citations
        citations_formatted = self._format_citations_for_prompt(citations)

        # Get custom instructions
        custom_instructions = ""
        if steerability.get("custom_instructions"):
            custom_instructions = f"\n## Custom Instructions\n{steerability['custom_instructions']}"

        # Base parameters
        params = {
            "user_query": state.get("user_query", ""),
            "tone": steerability.get("tone", "technical"),
            "max_sources": steerability.get("max_sources", 20),
            "language": steerability.get("language", "en"),
            "focus_areas": ", ".join(steerability.get("focus_areas", [])) or "General",
            "custom_instructions": custom_instructions,
            "synthesis": state.get("synthesis", "")[:4000],  # Truncate
            "findings_formatted": findings_formatted,
            "finding_count": len(findings),
            "citations_formatted": citations_formatted,
            "citation_count": len(citations),
            "coverage_score": quality_metrics.get("coverage_score", "N/A") if quality_metrics else "N/A",
            "confidence_score": quality_metrics.get("consistency_score", "N/A") if quality_metrics else "N/A",
            "unique_sources": quality_metrics.get("unique_sources", "N/A") if quality_metrics else "N/A",
        }

        # Select template based on format
        if output_format == "json":
            template = JSON_REPORT_TEMPLATE
        elif output_format == "markdown_table":
            template = MARKDOWN_TABLE_TEMPLATE
        else:
            template = MARKDOWN_REPORT_TEMPLATE

        return template.format(**params)

    def _format_findings_for_prompt(self, findings: list[Finding]) -> str:
        """Format findings for inclusion in the prompt."""
        formatted = []
        for i, finding in enumerate(findings[:20], 1):  # Limit to 20
            formatted.append(
                f"{i}. **{finding.get('claim', 'N/A')}**\n"
                f"   Evidence: {finding.get('evidence', 'N/A')[:200]}\n"
                f"   Confidence: {finding.get('confidence_score', 'N/A')}\n"
                f"   Citations: {finding.get('citation_ids', [])}"
            )
        return "\n\n".join(formatted)

    def _format_citations_for_prompt(self, citations: list[CitationRecord]) -> str:
        """Format citations for inclusion in the prompt."""
        formatted = []
        for citation in citations[:20]:  # Limit to 20
            ref_num = citation.get("reference_number", "?")
            formatted.append(
                f"[{ref_num}] {citation.get('title', 'Untitled')}\n"
                f"    URL: {citation.get('url', 'N/A')}\n"
                f"    Domain: {citation.get('domain', 'N/A')}\n"
                f"    Relevance: {citation.get('relevance_score', 'N/A')}"
            )
        return "\n\n".join(formatted)

    # =========================================================================
    # Report Post-Processing
    # =========================================================================

    def _post_process_report(
        self,
        raw_report: str,
        output_format: str,
        citations: list[CitationRecord],
        state: AgentState,
    ) -> str:
        """Post-process the generated report."""
        report = raw_report

        # Add report metadata header
        metadata = self._generate_report_metadata(state, citations)

        if output_format == "json":
            # Try to parse and validate JSON
            try:
                report_json = self._extract_json(raw_report)
                if report_json:
                    report_json["metadata"] = metadata
                    return json.dumps(report_json, indent=2)
            except Exception:
                pass
            # Return raw if JSON parsing fails
            return raw_report

        # For markdown formats, add metadata header
        header = self._format_markdown_header(metadata)
        report = f"{header}\n\n{report}"

        # Ensure references section exists
        if "## References" not in report and "# References" not in report:
            references = self._generate_references_section(citations)
            report = f"{report}\n\n{references}"

        # Add footer
        footer = self._generate_footer(state)
        report = f"{report}\n\n{footer}"

        return report

    def _generate_report_metadata(
        self,
        state: AgentState,
        citations: list[CitationRecord],
    ) -> dict[str, Any]:
        """Generate report metadata."""
        return {
            "query": state.get("user_query", ""),
            "generated_at": timestamp_now(),
            "session_id": state.get("session_id", ""),
            "iteration_count": state.get("iteration_count", 0),
            "total_sources": len(citations),
            "total_findings": len(state.get("findings", [])),
            "coverage_score": state.get("quality_metrics", {}).get("coverage_score") if state.get("quality_metrics") else None,
        }

    def _format_markdown_header(self, metadata: dict[str, Any]) -> str:
        """Format metadata as markdown header."""
        return f"""---
Generated: {metadata['generated_at']}
Session: {metadata['session_id']}
Sources: {metadata['total_sources']}
Findings: {metadata['total_findings']}
Coverage: {metadata.get('coverage_score', 'N/A')}
---"""

    def _generate_references_section(
        self,
        citations: list[CitationRecord],
    ) -> str:
        """Generate the references section."""
        lines = ["## References\n"]

        for citation in citations:
            ref_num = citation.get("reference_number", "?")
            title = citation.get("title", "Untitled")
            url = citation.get("url", "")
            domain = citation.get("domain", "")
            retrieved = citation.get("retrieved_at", "")[:10]  # Date only

            lines.append(
                f"[{ref_num}] {title}. *{domain}*. "
                f"Retrieved {retrieved}. [{url}]({url})"
            )

        return "\n".join(lines)

    def _generate_footer(self, state: AgentState) -> str:
        """Generate report footer."""
        return f"""---
*Report generated by DRX Deep Research System*
*Generated at: {timestamp_now()}*
*Sources consulted: {len(state.get('citations', []))}*
*Findings analyzed: {len(state.get('findings', []))}*
*Iterations: {state.get('iteration_count', 0)}*
"""

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _extract_json(self, response: str) -> dict[str, Any] | None:
        """Extract JSON from response."""
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
        """Update state with the final report."""
        if not response.success:
            return state

        data = response.data
        report = data.get("report", "")

        # Update plan task status
        plan = state.get("plan")
        if plan:
            for task in plan["dag_nodes"]:
                if task["agent_type"] == "reporter" and task["status"] == "pending":
                    task["status"] = "completed"
                    task["completed_at"] = timestamp_now()
                    task["outputs"] = {
                        "report_length": len(report),
                        "format": data.get("format"),
                        "citations_used": data.get("citations_used"),
                    }

        return {
            **state,
            "final_report": report,
            "current_phase": "complete",
            "should_terminate": True,
            "plan": plan,
        }


# =============================================================================
# Report Formatting Utilities
# =============================================================================


def format_citation_inline(citation: CitationRecord) -> str:
    """Format a citation for inline use."""
    ref_num = citation.get("reference_number", "?")
    return f"[{ref_num}]"


def format_citation_full(citation: CitationRecord) -> str:
    """Format a citation for the references section."""
    ref_num = citation.get("reference_number", "?")
    title = citation.get("title", "Untitled")
    url = citation.get("url", "")
    domain = citation.get("domain", "")

    return f"[{ref_num}] {title}. {domain}. {url}"


def create_executive_summary_prompt(
    synthesis: str,
    findings: list[Finding],
    tone: str,
) -> str:
    """Create a prompt for generating an executive summary."""
    finding_summaries = [f.get("claim", "")[:100] for f in findings[:5]]

    return f"""Create a concise executive summary (2-3 paragraphs) in {tone} tone.

Key findings to highlight:
{chr(10).join(f'- {s}' for s in finding_summaries)}

Full synthesis:
{synthesis[:2000]}

Write a compelling executive summary that captures the key insights."""


# =============================================================================
# Factory Function
# =============================================================================


def create_reporter_agent(
    llm_client: LLMClient | None = None,
    **kwargs,
) -> ReporterAgent:
    """
    Factory function to create a configured ReporterAgent.

    Args:
        llm_client: LLM client for report generation
        **kwargs: Additional configuration options

    Returns:
        Configured ReporterAgent instance
    """
    return ReporterAgent(llm_client=llm_client, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ReporterAgent",
    "create_reporter_agent",
    "format_citation_inline",
    "format_citation_full",
    "create_executive_summary_prompt",
    "REPORTER_SYSTEM_PROMPT",
]
