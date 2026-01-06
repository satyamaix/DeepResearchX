"""
Critic Agent for DRX Deep Research System.

Responsible for:
- Reviewing synthesis for gaps and weaknesses
- Identifying unsupported or weakly supported claims
- Checking source quality and diversity
- Validating factual consistency
- Suggesting additional research directions
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
from src.tools.bias_detector import BiasDetector, BiasReport, create_bias_detector

if TYPE_CHECKING:
    from ..orchestrator.state import (
        AgentState,
        AgentType,
        CitationRecord,
        Finding,
        QualityMetrics,
        ResearchPlan,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Critic Agent System Prompt
# =============================================================================

CRITIC_SYSTEM_PROMPT = """You are an expert research critic for a deep research system. Your role is to rigorously evaluate research quality and identify gaps.

## Your Capabilities
1. Identify knowledge gaps in the research
2. Detect unsupported or weakly supported claims
3. Evaluate source quality and credibility
4. Check for logical consistency
5. Assess coverage of the original query
6. Recommend additional research directions

## Output Format
Respond with a JSON object containing your critique:
```json
{
  "gaps": [
    {
      "description": "Description of the knowledge gap",
      "severity": "critical|high|medium|low",
      "suggested_action": "How to address this gap",
      "related_to_query": true
    }
  ],
  "unsupported_claims": [
    {
      "claim": "The unsupported claim text",
      "finding_id": "related finding ID if known",
      "issue": "Why it's unsupported",
      "recommendation": "How to verify or remove"
    }
  ],
  "source_quality_issues": [
    {
      "issue": "Description of source quality concern",
      "affected_findings": ["finding_id1", "finding_id2"],
      "severity": "high|medium|low"
    }
  ],
  "logical_issues": [
    {
      "issue": "Description of logical problem",
      "type": "contradiction|circular|non_sequitur|other",
      "affected_areas": ["area1"]
    }
  ],
  "coverage_assessment": {
    "score": 0.75,
    "well_covered": ["aspect1", "aspect2"],
    "poorly_covered": ["aspect3"],
    "not_covered": ["aspect4"]
  },
  "overall_quality_score": 0.7,
  "ready_for_report": false,
  "requires_iteration": true,
  "priority_actions": [
    "Most important action to take",
    "Second priority action"
  ],
  "critique_notes": "Additional observations"
}
```

## Evaluation Criteria
1. **Completeness**: Does research cover all aspects of the query?
2. **Accuracy**: Are claims properly supported by evidence?
3. **Source Quality**: Are sources credible and diverse?
4. **Consistency**: Are there contradictions or logical issues?
5. **Relevance**: Do findings address the actual query?
6. **Depth**: Is the analysis sufficiently detailed?

## Gap Severity Levels
- **critical**: Must be addressed before finalizing
- **high**: Should be addressed if possible
- **medium**: Would improve quality if addressed
- **low**: Minor improvement opportunity

## Quality Score Guidelines
- 0.9-1.0: Excellent, ready for final report
- 0.7-0.89: Good, minor gaps to address
- 0.5-0.69: Adequate, notable gaps remain
- 0.3-0.49: Weak, significant gaps
- 0.0-0.29: Poor, major issues require attention

Be thorough but fair. Focus on actionable feedback.

Respond ONLY with the JSON object."""


# =============================================================================
# Critique Prompt Template
# =============================================================================

CRITIQUE_TEMPLATE = """## Original Research Query
{user_query}

## Current Synthesis
{synthesis}

## Findings Summary
Total findings: {finding_count}
High confidence: {high_confidence_count}
Verified findings: {verified_count}

## Findings Details
{findings_formatted}

## Source Information
Total citations: {citation_count}
Unique domains: {unique_domains}
Domain distribution: {domain_distribution}

## Current Iteration
Iteration {iteration} of {max_iterations}

## Previous Gaps (if any)
{previous_gaps}

## Instructions
Critically evaluate the research quality:
1. Identify any gaps in addressing the original query
2. Find claims that lack sufficient evidence
3. Assess source quality and diversity
4. Check for logical consistency
5. Determine if research is complete or needs more iteration"""


# =============================================================================
# Critic Agent Implementation
# =============================================================================


class CriticAgent(BaseAgent):
    """
    Critique agent that evaluates research quality and identifies gaps.

    The critic reviews synthesis results, checks source quality,
    identifies unsupported claims, and determines whether additional
    research iterations are needed.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        temperature: float = 0.2,  # Low for consistent evaluation
        min_coverage_threshold: float = 0.7,
        min_quality_threshold: float = 0.6,
        require_source_diversity: bool = True,
        min_unique_sources: int = 3,
        high_bias_threshold: float = 0.7,
    ):
        """
        Initialize the critic agent.

        Args:
            llm_client: LLM client for critique
            model: Model identifier
            temperature: Sampling temperature
            min_coverage_threshold: Minimum coverage score for completion
            min_quality_threshold: Minimum quality score for completion
            require_source_diversity: Whether to require diverse sources
            min_unique_sources: Minimum unique source domains required
            high_bias_threshold: Threshold above which bias is considered high (0.0-1.0)
        """
        super().__init__(
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            max_output_tokens=4096,
        )
        self._min_coverage = min_coverage_threshold
        self._min_quality = min_quality_threshold
        self._require_diversity = require_source_diversity
        self._min_unique_sources = min_unique_sources
        self._high_bias_threshold = high_bias_threshold
        self._bias_detector = create_bias_detector()

    # =========================================================================
    # Required Abstract Properties
    # =========================================================================

    @property
    def name(self) -> str:
        return "critic"

    @property
    def description(self) -> str:
        return (
            "Evaluates research quality, identifies gaps and unsupported claims, "
            "and determines readiness for final report generation."
        )

    @property
    def agent_type(self) -> AgentType:
        return "critic"

    @property
    def system_prompt(self) -> str:
        return CRITIC_SYSTEM_PROMPT

    # =========================================================================
    # Core Processing
    # =========================================================================

    async def _process(self, state: AgentState) -> AgentResponse:
        """
        Critique the current research state.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with critique results
        """
        synthesis = state.get("synthesis", "")
        findings = state.get("findings", [])
        citations = state.get("citations", [])

        # If no synthesis, nothing to critique
        if not synthesis:
            return AgentResponse.success_response(
                data={
                    "gaps": [{"description": "No synthesis to evaluate", "severity": "critical"}],
                    "coverage_assessment": {"score": 0.0},
                    "overall_quality_score": 0.0,
                    "ready_for_report": False,
                    "requires_iteration": True,
                    "bias_report": None,
                },
                agent_name=self.name,
                tokens_used=0,
            )

        # Perform pre-analysis
        pre_analysis = self._pre_analyze(findings, citations, state)

        # Run bias analysis
        bias_report = self._analyze_bias(citations, findings, synthesis)
        pre_analysis["bias_report"] = bias_report

        # Format prompt
        prompt = self._format_critique_prompt(state, pre_analysis)

        response = await self._call_llm(prompt, temperature=0.2)

        if not response.success:
            return response

        try:
            data = self._extract_json(response.data)
            if not data:
                data = self._create_fallback_critique(pre_analysis)

            # Merge with pre-analysis results
            data = self._merge_with_pre_analysis(data, pre_analysis)

            # Add bias-related gaps if high bias detected
            data = self._add_bias_gaps(data, bias_report)

            # Include bias_report in response data
            data["bias_report"] = bias_report

            # Determine if ready for report
            data["ready_for_report"] = self._check_ready_for_report(data, state)
            data["requires_iteration"] = not data["ready_for_report"]

            return AgentResponse.success_response(
                data={
                    **data,
                    "critiqued_at": timestamp_now(),
                },
                agent_name=self.name,
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.exception(f"Critique parsing failed: {e}")
            return AgentResponse.error_response(str(e), self.name)

    # =========================================================================
    # Pre-Analysis
    # =========================================================================

    def _pre_analyze(
        self,
        findings: list[Finding],
        citations: list[CitationRecord],
        state: AgentState,
    ) -> dict[str, Any]:
        """Perform automated pre-analysis before LLM critique."""
        analysis = {
            "finding_stats": {},
            "source_stats": {},
            "automated_issues": [],
        }

        # Finding statistics
        total = len(findings)
        high_conf = len([f for f in findings if f.get("confidence_score", 0) >= 0.7])
        verified = len([f for f in findings if f.get("verified", False)])

        analysis["finding_stats"] = {
            "total": total,
            "high_confidence": high_conf,
            "verified": verified,
            "low_confidence": total - high_conf,
            "confidence_ratio": high_conf / total if total > 0 else 0,
        }

        # Source statistics
        domains = [c.get("domain", "") for c in citations]
        unique_domains = set(domains)
        domain_counts = {}
        for d in domains:
            domain_counts[d] = domain_counts.get(d, 0) + 1

        analysis["source_stats"] = {
            "total_citations": len(citations),
            "unique_domains": len(unique_domains),
            "domain_distribution": domain_counts,
            "has_diversity": len(unique_domains) >= self._min_unique_sources,
        }

        # Automated issue detection
        issues = []

        # Check for low source diversity
        if not analysis["source_stats"]["has_diversity"]:
            issues.append({
                "type": "source_diversity",
                "description": f"Only {len(unique_domains)} unique source domains (minimum: {self._min_unique_sources})",
                "severity": "medium",
            })

        # Check for too many low-confidence findings
        if analysis["finding_stats"]["confidence_ratio"] < 0.5:
            issues.append({
                "type": "low_confidence",
                "description": "Majority of findings have low confidence",
                "severity": "high",
            })

        # Check for no verified findings
        if verified == 0 and total > 0:
            issues.append({
                "type": "no_verification",
                "description": "No findings have been verified",
                "severity": "low",
            })

        # Check for single-source findings
        single_source = [f for f in findings if len(f.get("source_urls", [])) == 1]
        if len(single_source) > total * 0.8:
            issues.append({
                "type": "single_source",
                "description": "Most findings rely on single sources",
                "severity": "medium",
            })

        analysis["automated_issues"] = issues

        return analysis

    # =========================================================================
    # Prompt Formatting
    # =========================================================================

    def _format_critique_prompt(
        self,
        state: AgentState,
        pre_analysis: dict[str, Any],
    ) -> str:
        """Format the critique prompt."""
        findings = state.get("findings", [])
        citations = state.get("citations", [])

        # Format findings summary
        findings_formatted = []
        for i, f in enumerate(findings[:15], 1):  # Limit to first 15
            findings_formatted.append(
                f"{i}. {f.get('claim', 'N/A')[:100]}... "
                f"(conf: {f.get('confidence_score', 'N/A')}, "
                f"sources: {len(f.get('source_urls', []))})"
            )

        # Format domain distribution
        domain_dist = pre_analysis["source_stats"]["domain_distribution"]
        top_domains = sorted(domain_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        domain_str = ", ".join(f"{d}: {c}" for d, c in top_domains)

        # Previous gaps
        prev_gaps = state.get("gaps", [])
        prev_gaps_str = "\n".join(f"- {g}" for g in prev_gaps) if prev_gaps else "None identified"

        return CRITIQUE_TEMPLATE.format(
            user_query=state.get("user_query", ""),
            synthesis=state.get("synthesis", "")[:2000],  # Truncate
            finding_count=pre_analysis["finding_stats"]["total"],
            high_confidence_count=pre_analysis["finding_stats"]["high_confidence"],
            verified_count=pre_analysis["finding_stats"]["verified"],
            findings_formatted="\n".join(findings_formatted),
            citation_count=pre_analysis["source_stats"]["total_citations"],
            unique_domains=pre_analysis["source_stats"]["unique_domains"],
            domain_distribution=domain_str,
            iteration=state.get("iteration_count", 1),
            max_iterations=state.get("max_iterations", 5),
            previous_gaps=prev_gaps_str,
        )

    # =========================================================================
    # Result Processing
    # =========================================================================

    def _merge_with_pre_analysis(
        self,
        llm_critique: dict[str, Any],
        pre_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge LLM critique with automated pre-analysis."""
        # Add source quality issues from pre-analysis
        source_issues = llm_critique.get("source_quality_issues", [])
        for issue in pre_analysis["automated_issues"]:
            if issue["type"] in ["source_diversity", "single_source"]:
                source_issues.append({
                    "issue": issue["description"],
                    "severity": issue["severity"],
                    "automated": True,
                })

        llm_critique["source_quality_issues"] = source_issues

        # Add pre-analysis stats
        llm_critique["finding_stats"] = pre_analysis["finding_stats"]
        llm_critique["source_stats"] = pre_analysis["source_stats"]

        return llm_critique

    def _analyze_bias(
        self,
        citations: list[CitationRecord],
        findings: list[Finding],
        synthesis: str,
    ) -> BiasReport:
        """
        Run bias analysis on citations and findings.

        Args:
            citations: All citations from the research
            findings: All findings from the research
            synthesis: Current synthesis text for content analysis

        Returns:
            BiasReport with diversity metrics and bias indicators
        """
        # Prepare content samples from synthesis for bias indicator detection
        content_samples = [synthesis] if synthesis else []

        bias_report = self._bias_detector.analyze(
            citations=citations,
            findings=findings,
            content_samples=content_samples,
        )

        logger.info(
            f"Bias analysis complete: "
            f"overall_bias={bias_report['overall_bias_score']:.3f}, "
            f"risk_level={bias_report['risk_level']}, "
            f"diversity={bias_report['diversity']['overall_score']:.3f}"
        )

        return bias_report

    def _add_bias_gaps(
        self,
        critique_data: dict[str, Any],
        bias_report: BiasReport,
    ) -> dict[str, Any]:
        """
        Add bias-related gaps to the critique if high bias detected.

        Args:
            critique_data: Current critique data dict
            bias_report: BiasReport from bias analysis

        Returns:
            Updated critique_data with bias-related gaps added
        """
        gaps = critique_data.get("gaps", [])

        overall_bias = bias_report["overall_bias_score"]
        risk_level = bias_report["risk_level"]
        diversity = bias_report["diversity"]

        # Add gap for high overall bias
        if overall_bias >= self._high_bias_threshold:
            gaps.append({
                "description": (
                    f"High bias detected in sources (score: {overall_bias:.2f}). "
                    "Consider diversifying sources to reduce potential bias."
                ),
                "severity": "high" if risk_level == "high" else "medium",
                "suggested_action": "Search for alternative perspectives and authoritative sources",
                "related_to_query": True,
                "bias_related": True,
            })
            logger.warning(
                f"High bias detected: overall_bias={overall_bias:.3f} >= threshold={self._high_bias_threshold}"
            )

        # Add gaps from diversity recommendations
        for recommendation in diversity.get("recommendations", []):
            gaps.append({
                "description": recommendation,
                "severity": "medium",
                "suggested_action": "Address source diversity",
                "related_to_query": True,
                "bias_related": True,
            })

        # Add gap for low viewpoint balance
        viewpoint_balance = bias_report["viewpoint_assessment"]["balance_score"]
        if viewpoint_balance < 0.3:
            coverage_gaps = bias_report["viewpoint_assessment"].get("coverage_gaps", [])
            gaps.append({
                "description": (
                    f"Low viewpoint diversity (balance: {viewpoint_balance:.2f}). "
                    f"Missing perspectives: {', '.join(coverage_gaps[:3]) if coverage_gaps else 'various'}"
                ),
                "severity": "medium",
                "suggested_action": "Search for alternative viewpoints on the topic",
                "related_to_query": True,
                "bias_related": True,
            })

        critique_data["gaps"] = gaps
        return critique_data

    def _create_fallback_critique(
        self,
        pre_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Create critique from pre-analysis if LLM parsing fails."""
        gaps = []
        for issue in pre_analysis["automated_issues"]:
            gaps.append({
                "description": issue["description"],
                "severity": issue["severity"],
                "suggested_action": "Address this issue",
                "automated": True,
            })

        # Estimate quality based on pre-analysis
        quality_score = 0.5
        if pre_analysis["finding_stats"]["confidence_ratio"] >= 0.7:
            quality_score += 0.2
        if pre_analysis["source_stats"]["has_diversity"]:
            quality_score += 0.15
        if not pre_analysis["automated_issues"]:
            quality_score += 0.15

        return {
            "gaps": gaps,
            "unsupported_claims": [],
            "source_quality_issues": [],
            "logical_issues": [],
            "coverage_assessment": {
                "score": quality_score,
                "well_covered": [],
                "poorly_covered": [],
                "not_covered": [],
            },
            "overall_quality_score": quality_score,
            "ready_for_report": quality_score >= self._min_quality,
            "requires_iteration": quality_score < self._min_quality,
            "priority_actions": [issue["description"] for issue in pre_analysis["automated_issues"][:3]],
        }

    def _check_ready_for_report(
        self,
        critique: dict[str, Any],
        state: AgentState,
    ) -> bool:
        """Determine if research is ready for final report."""
        # Check iteration limit
        iteration = state.get("iteration_count", 1)
        max_iterations = state.get("max_iterations", 5)
        if iteration >= max_iterations:
            logger.info(f"Max iterations ({max_iterations}) reached, forcing completion")
            return True

        # Check quality thresholds
        quality_score = critique.get("overall_quality_score", 0)
        coverage_score = critique.get("coverage_assessment", {}).get("score", 0)

        if quality_score < self._min_quality:
            return False

        if coverage_score < self._min_coverage:
            return False

        # Check for critical gaps
        gaps = critique.get("gaps", [])
        critical_gaps = [g for g in gaps if g.get("severity") == "critical"]
        if critical_gaps:
            return False

        return True

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
        """Update state with critique results."""
        if not response.success:
            return state

        data = response.data

        # Extract gaps
        gaps = []
        for gap in data.get("gaps", []):
            if isinstance(gap, dict):
                gaps.append(gap.get("description", str(gap)))
            else:
                gaps.append(str(gap))

        # Extract bias report and score
        bias_report = data.get("bias_report")
        bias_score = 0.0
        if bias_report:
            bias_score = bias_report.get("overall_bias_score", 0.0)

        # Build quality metrics
        quality_metrics: QualityMetrics = {
            "coverage_score": data.get("coverage_assessment", {}).get("score", 0),
            "avg_confidence": data.get("finding_stats", {}).get("confidence_ratio", 0),
            "verified_findings": data.get("finding_stats", {}).get("verified", 0),
            "total_findings": data.get("finding_stats", {}).get("total", 0),
            "unique_sources": data.get("source_stats", {}).get("unique_domains", 0),
            "citation_density": 0,  # Would need to calculate from synthesis
            "consistency_score": data.get("overall_quality_score", 0),
            "bias_score": bias_score,
            "updated_at": timestamp_now(),
        }

        # Determine next phase
        ready_for_report = data.get("ready_for_report", False)
        next_phase = "reporting" if ready_for_report else "researching"

        # Update plan coverage score
        plan = state.get("plan")
        if plan:
            plan["coverage_score"] = quality_metrics["coverage_score"]

            # Mark critic task as complete
            for task in plan["dag_nodes"]:
                if task["agent_type"] == "critic" and task["status"] == "pending":
                    task["status"] = "completed"
                    task["completed_at"] = timestamp_now()
                    task["quality_score"] = data.get("overall_quality_score")
                    task["outputs"] = {
                        "gaps_count": len(gaps),
                        "ready_for_report": ready_for_report,
                    }

        return {
            **state,
            "gaps": gaps,
            "quality_metrics": quality_metrics,
            "bias_report": bias_report,
            "plan": plan,
            "current_phase": next_phase,
            "should_terminate": ready_for_report,
        }


# =============================================================================
# Verification Utilities
# =============================================================================


def verify_finding_sources(
    finding: Finding,
    citations: list[CitationRecord],
) -> dict[str, Any]:
    """
    Verify that a finding's sources exist and are relevant.

    Args:
        finding: Finding to verify
        citations: Available citations

    Returns:
        Verification result dict
    """
    citation_ids = finding.get("citation_ids", [])
    source_urls = finding.get("source_urls", [])

    # Build lookup
    citation_lookup = {c["id"]: c for c in citations}
    url_lookup = {c["url"]: c for c in citations}

    verified_citations = []
    missing_citations = []

    for cid in citation_ids:
        if cid in citation_lookup:
            verified_citations.append(citation_lookup[cid])
        else:
            missing_citations.append(cid)

    for url in source_urls:
        if url in url_lookup and url_lookup[url] not in verified_citations:
            verified_citations.append(url_lookup[url])

    return {
        "finding_id": finding.get("id"),
        "verified_citations": len(verified_citations),
        "missing_citations": len(missing_citations),
        "source_quality": _assess_source_quality(verified_citations),
        "is_verified": len(verified_citations) > 0 and len(missing_citations) == 0,
    }


def _assess_source_quality(citations: list[CitationRecord]) -> str:
    """Assess overall quality of citation sources."""
    if not citations:
        return "none"

    avg_relevance = sum(c.get("relevance_score", 0.5) for c in citations) / len(citations)

    if avg_relevance >= 0.8:
        return "high"
    elif avg_relevance >= 0.5:
        return "medium"
    else:
        return "low"


# =============================================================================
# Factory Function
# =============================================================================


def create_critic_agent(
    llm_client: LLMClient | None = None,
    **kwargs,
) -> CriticAgent:
    """
    Factory function to create a configured CriticAgent.

    Args:
        llm_client: LLM client for critique
        **kwargs: Additional configuration options

    Returns:
        Configured CriticAgent instance
    """
    return CriticAgent(llm_client=llm_client, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CriticAgent",
    "create_critic_agent",
    "verify_finding_sources",
    "CRITIC_SYSTEM_PROMPT",
]
