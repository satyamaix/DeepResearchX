"""
Bias Detection Tool for DRX Deep Research System.

Provides analysis of source diversity, bias indicators, and viewpoint coverage
to improve research quality and identify potential blind spots.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from urllib.parse import urlparse

if TYPE_CHECKING:
    from src.orchestrator.state import CitationRecord, Finding

# =============================================================================
# Type Definitions
# =============================================================================

BiasType = Literal["political", "commercial", "sensational", "selective", "temporal", "geographic"]
SeverityLevel = Literal["low", "medium", "high"]


class BiasIndicator(TypedDict):
    """A detected bias indicator in content."""
    indicator_type: BiasType
    description: str
    severity: SeverityLevel
    evidence: str
    confidence: float


class DiversityReport(TypedDict):
    """Report on source diversity metrics."""
    domain_diversity: float  # 0-1, entropy-based
    domain_count: int
    top_domains: list[tuple[str, int]]
    source_type_diversity: float
    source_types: dict[str, int]
    geographic_diversity: float  # Estimated
    temporal_diversity: float
    date_range_days: int | None
    overall_score: float
    recommendations: list[str]


class ViewpointAssessment(TypedDict):
    """Assessment of viewpoint coverage in findings."""
    viewpoint_count: int
    dominant_viewpoint: str | None
    underrepresented: list[str]
    balance_score: float  # 0-1
    coverage_gaps: list[str]


class BiasReport(TypedDict):
    """Complete bias analysis report."""
    diversity: DiversityReport
    indicators: list[BiasIndicator]
    viewpoint_assessment: ViewpointAssessment
    overall_bias_score: float  # 0-1, higher means more bias detected
    risk_level: SeverityLevel
    summary: str
    analyzed_at: str


# =============================================================================
# Source Type Detection
# =============================================================================

# Domain patterns for source type classification
SOURCE_TYPE_PATTERNS: dict[str, list[str]] = {
    "academic": [
        r"\.edu$", r"\.ac\.[a-z]{2}$", r"arxiv\.org", r"scholar\.google",
        r"pubmed", r"researchgate", r"academia\.edu", r"springer\.com",
        r"nature\.com", r"sciencedirect", r"ieee\.org", r"acm\.org",
    ],
    "news": [
        r"reuters\.", r"apnews\.", r"bbc\.", r"cnn\.", r"nytimes\.",
        r"washingtonpost\.", r"theguardian\.", r"news\.", r"\.news$",
    ],
    "government": [
        r"\.gov$", r"\.gov\.[a-z]{2}$", r"\.mil$", r"who\.int",
        r"un\.org", r"europa\.eu",
    ],
    "corporate": [
        r"blog\.", r"\.medium\.com", r"company", r"corporate",
        r"press-release", r"newsroom",
    ],
    "social": [
        r"twitter\.com", r"x\.com", r"facebook\.com", r"linkedin\.com",
        r"reddit\.com", r"youtube\.com",
    ],
    "wiki": [
        r"wikipedia\.org", r"wikimedia\.org", r"wiki\.",
    ],
}

# Geographic indicators in domains
# NOTE: .com, .org, .net, .io are global TLDs and should NOT be attributed to any region
GEO_INDICATORS: dict[str, list[str]] = {
    "us": [".us", ".edu", ".gov"],  # US-specific TLDs only
    "uk": [".co.uk", ".ac.uk", ".gov.uk", ".uk"],
    "eu": [".eu", ".de", ".fr", ".nl", ".es", ".it", ".be", ".at", ".ch", ".pl"],
    "asia": [".cn", ".jp", ".kr", ".in", ".sg", ".hk", ".tw", ".th", ".vn"],
    "international": [".com", ".org", ".net", ".io", ".info", ".biz"],  # Global TLDs
    "other": [],
}

# Bias indicator patterns
BIAS_PATTERNS: dict[str, list[str]] = {
    "political_left": [
        r"\bliberal\b", r"\bprogressive\b", r"\bsocialist\b",
        r"\bleft-wing\b", r"\bdemocrat\b",
    ],
    "political_right": [
        r"\bconservative\b", r"\btraditional\b", r"\bright-wing\b",
        r"\brepublican\b", r"\blibertarian\b",
    ],
    "sensational": [
        r"\bbreaking\b", r"\bshocking\b", r"\bunbelievable\b",
        r"\byou won't believe\b", r"\bexclusive\b", r"\b!!!+",
        r"\bOMG\b", r"\bWOW\b",
    ],
    "commercial": [
        r"\bbuy now\b", r"\blimited time\b", r"\bspecial offer\b",
        r"\bsponsored\b", r"\bad\b", r"\baffiliate\b",
    ],
}


# =============================================================================
# Bias Detector Implementation
# =============================================================================


class BiasDetector:
    """
    Detect bias and assess source diversity in research.

    Analyzes:
    - Source domain diversity
    - Source type distribution
    - Geographic diversity
    - Temporal diversity
    - Content bias indicators
    - Viewpoint coverage
    """

    def __init__(
        self,
        min_domain_diversity: float = 0.5,
        min_source_types: int = 2,
        commercial_threshold: float = 0.3,
    ):
        """
        Initialize the bias detector.

        Args:
            min_domain_diversity: Minimum acceptable domain diversity score
            min_source_types: Minimum number of source types expected
            commercial_threshold: Max acceptable ratio of commercial sources
        """
        self._min_domain_diversity = min_domain_diversity
        self._min_source_types = min_source_types
        self._commercial_threshold = commercial_threshold

    def analyze(
        self,
        citations: list[CitationRecord],
        findings: list[Finding],
        content_samples: list[str] | None = None,
    ) -> BiasReport:
        """
        Perform comprehensive bias analysis.

        Args:
            citations: Citations to analyze
            findings: Findings to assess
            content_samples: Optional content for bias indicators

        Returns:
            Complete BiasReport
        """
        diversity = self.analyze_diversity(citations)
        indicators: list[BiasIndicator] = []

        if content_samples:
            for content in content_samples[:10]:  # Limit analysis
                indicators.extend(self.detect_indicators(content))

        viewpoints = self.assess_viewpoints(findings)

        # Calculate overall bias score
        bias_factors = [
            1 - diversity["overall_score"],  # Lower diversity = higher bias
            1 - viewpoints["balance_score"],  # Lower balance = higher bias
            min(1.0, len(indicators) / 10),  # More indicators = higher bias
        ]
        overall_bias = sum(bias_factors) / len(bias_factors)

        # Determine risk level
        if overall_bias > 0.6:
            risk_level: SeverityLevel = "high"
        elif overall_bias > 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Generate summary
        summary = self._generate_summary(diversity, indicators, viewpoints, risk_level)

        return BiasReport(
            diversity=diversity,
            indicators=indicators,
            viewpoint_assessment=viewpoints,
            overall_bias_score=round(overall_bias, 3),
            risk_level=risk_level,
            summary=summary,
            analyzed_at=datetime.utcnow().isoformat() + "Z",
        )

    def analyze_diversity(self, citations: list[CitationRecord]) -> DiversityReport:
        """
        Analyze source diversity metrics.

        Args:
            citations: Citations to analyze

        Returns:
            DiversityReport with metrics and recommendations
        """
        if not citations:
            return DiversityReport(
                domain_diversity=0.0,
                domain_count=0,
                top_domains=[],
                source_type_diversity=0.0,
                source_types={},
                geographic_diversity=0.0,
                temporal_diversity=0.0,
                date_range_days=None,
                overall_score=0.0,
                recommendations=["No citations to analyze"],
            )

        # Extract domains
        domains: list[str] = []
        for c in citations:
            url = c.get("url", "")
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                if domain:
                    # Normalize to root domain
                    parts = domain.split(".")
                    if len(parts) > 2:
                        domain = ".".join(parts[-2:])
                    domains.append(domain)
            except Exception:
                pass

        # Domain diversity (entropy-based)
        domain_counts = Counter(domains)
        domain_diversity = self._calculate_entropy(list(domain_counts.values()))

        # Source type distribution
        source_types = self._classify_source_types(citations)
        source_type_diversity = self._calculate_entropy(list(source_types.values()))

        # Geographic diversity
        geo_distribution = self._analyze_geographic_diversity(citations)
        geo_diversity = self._calculate_entropy(list(geo_distribution.values()))

        # Temporal diversity
        temporal_diversity, date_range = self._analyze_temporal_diversity(citations)

        # Calculate overall score
        overall_score = (
            domain_diversity * 0.3 +
            source_type_diversity * 0.3 +
            geo_diversity * 0.2 +
            temporal_diversity * 0.2
        )

        # Generate recommendations
        recommendations = self._generate_diversity_recommendations(
            domain_diversity,
            source_types,
            geo_distribution,
            temporal_diversity,
        )

        return DiversityReport(
            domain_diversity=round(domain_diversity, 3),
            domain_count=len(domain_counts),
            top_domains=domain_counts.most_common(5),
            source_type_diversity=round(source_type_diversity, 3),
            source_types=dict(source_types),
            geographic_diversity=round(geo_diversity, 3),
            temporal_diversity=round(temporal_diversity, 3),
            date_range_days=date_range,
            overall_score=round(overall_score, 3),
            recommendations=recommendations,
        )

    def detect_indicators(self, content: str) -> list[BiasIndicator]:
        """
        Detect bias indicators in content.

        Args:
            content: Text content to analyze

        Returns:
            List of detected bias indicators
        """
        indicators: list[BiasIndicator] = []
        content_lower = content.lower()

        # Check for political bias
        left_matches = sum(
            len(re.findall(p, content_lower))
            for p in BIAS_PATTERNS["political_left"]
        )
        right_matches = sum(
            len(re.findall(p, content_lower))
            for p in BIAS_PATTERNS["political_right"]
        )

        if left_matches > 3 and left_matches > right_matches * 2:
            indicators.append(BiasIndicator(
                indicator_type="political",
                description="Content shows left-leaning political bias",
                severity="medium",
                evidence=f"Found {left_matches} left-leaning terms vs {right_matches} right-leaning",
                confidence=min(0.9, left_matches / 10),
            ))
        elif right_matches > 3 and right_matches > left_matches * 2:
            indicators.append(BiasIndicator(
                indicator_type="political",
                description="Content shows right-leaning political bias",
                severity="medium",
                evidence=f"Found {right_matches} right-leaning terms vs {left_matches} left-leaning",
                confidence=min(0.9, right_matches / 10),
            ))

        # Check for sensational language
        sensational_matches = sum(
            len(re.findall(p, content, re.IGNORECASE))
            for p in BIAS_PATTERNS["sensational"]
        )
        if sensational_matches > 2:
            indicators.append(BiasIndicator(
                indicator_type="sensational",
                description="Content uses sensational language",
                severity="low" if sensational_matches < 5 else "medium",
                evidence=f"Found {sensational_matches} sensational terms",
                confidence=min(0.8, sensational_matches / 5),
            ))

        # Check for commercial bias
        commercial_matches = sum(
            len(re.findall(p, content_lower))
            for p in BIAS_PATTERNS["commercial"]
        )
        if commercial_matches > 1:
            indicators.append(BiasIndicator(
                indicator_type="commercial",
                description="Content may have commercial bias",
                severity="medium" if commercial_matches > 3 else "low",
                evidence=f"Found {commercial_matches} commercial indicators",
                confidence=min(0.7, commercial_matches / 4),
            ))

        return indicators

    def assess_viewpoints(self, findings: list[Finding]) -> ViewpointAssessment:
        """
        Assess viewpoint diversity in findings.

        Args:
            findings: Findings to assess

        Returns:
            ViewpointAssessment with coverage analysis
        """
        if not findings:
            return ViewpointAssessment(
                viewpoint_count=0,
                dominant_viewpoint=None,
                underrepresented=[],
                balance_score=0.0,
                coverage_gaps=["No findings to assess"],
            )

        # Extract tags and themes from findings
        all_tags: list[str] = []
        for f in findings:
            all_tags.extend(f.get("tags", []))

        tag_counts = Counter(all_tags)

        # Identify dominant viewpoint
        viewpoint_count = len(tag_counts)
        dominant = tag_counts.most_common(1)[0] if tag_counts else None
        dominant_viewpoint = dominant[0] if dominant else None

        # Calculate balance score
        if viewpoint_count <= 1:
            balance_score = 0.0
        else:
            # Entropy-based balance
            balance_score = self._calculate_entropy(list(tag_counts.values()))

        # Identify underrepresented viewpoints
        underrepresented: list[str] = []
        if dominant:
            threshold = dominant[1] * 0.2  # 20% of dominant count
            for tag, count in tag_counts.items():
                if count < threshold and count < 2:
                    underrepresented.append(tag)

        # Identify coverage gaps (based on common research categories)
        expected_categories = {
            "pros", "cons", "risks", "benefits", "alternatives",
            "historical", "current", "future", "expert", "case_study"
        }
        covered = set(t.lower() for t in all_tags)
        coverage_gaps = list(expected_categories - covered)[:5]

        return ViewpointAssessment(
            viewpoint_count=viewpoint_count,
            dominant_viewpoint=dominant_viewpoint,
            underrepresented=underrepresented[:5],
            balance_score=round(balance_score, 3),
            coverage_gaps=coverage_gaps,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_entropy(self, counts: list[int]) -> float:
        """Calculate normalized Shannon entropy (0-1)."""
        if not counts or sum(counts) == 0:
            return 0.0

        total = sum(counts)
        probabilities = [c / total for c in counts if c > 0]

        if len(probabilities) <= 1:
            return 0.0

        entropy = -sum(p * math.log2(p) for p in probabilities)
        max_entropy = math.log2(len(probabilities))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _classify_source_types(
        self, citations: list[CitationRecord]
    ) -> Counter[str]:
        """Classify citations by source type."""
        type_counts: Counter[str] = Counter()

        for c in citations:
            url = c.get("url", "").lower()
            domain = c.get("domain", "").lower()

            classified = False
            for source_type, patterns in SOURCE_TYPE_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, url) or re.search(pattern, domain):
                        type_counts[source_type] += 1
                        classified = True
                        break
                if classified:
                    break

            if not classified:
                type_counts["other"] += 1

        return type_counts

    def _analyze_geographic_diversity(
        self, citations: list[CitationRecord]
    ) -> Counter[str]:
        """
        Analyze geographic distribution of sources.

        Note: International TLDs (.com, .org, .net, .io) are tracked separately
        and treated as geographically neutral. They contribute to diversity but
        are not counted as belonging to any specific region for bias detection.
        """
        geo_counts: Counter[str] = Counter()

        for c in citations:
            url = c.get("url", "").lower()
            domain = c.get("domain", "").lower()

            classified = False
            # Check regional TLDs first (more specific), then international
            # Order matters: check country-specific TLDs before generic ones
            check_order = ["uk", "eu", "asia", "us", "international", "other"]
            for region in check_order:
                indicators = GEO_INDICATORS.get(region, [])
                for ind in indicators:
                    if ind in domain or ind in url:
                        geo_counts[region] += 1
                        classified = True
                        break
                if classified:
                    break

            if not classified:
                geo_counts["unknown"] += 1

        return geo_counts

    def _analyze_temporal_diversity(
        self, citations: list[CitationRecord]
    ) -> tuple[float, int | None]:
        """Analyze temporal distribution of sources."""
        dates: list[datetime] = []

        for c in citations:
            retrieved_at = c.get("retrieved_at")
            if retrieved_at:
                try:
                    dt = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
                    dates.append(dt)
                except Exception:
                    pass

        if len(dates) < 2:
            return 0.0, None

        # Calculate date range
        date_range = (max(dates) - min(dates)).days

        # Score based on spread (higher is better diversity)
        if date_range == 0:
            return 0.0, 0
        elif date_range < 7:
            return 0.3, date_range
        elif date_range < 30:
            return 0.6, date_range
        elif date_range < 365:
            return 0.8, date_range
        else:
            return 1.0, date_range

    def _generate_diversity_recommendations(
        self,
        domain_diversity: float,
        source_types: Counter[str],
        geo_distribution: Counter[str],
        temporal_diversity: float,
    ) -> list[str]:
        """Generate recommendations for improving diversity."""
        recommendations: list[str] = []

        if domain_diversity < self._min_domain_diversity:
            recommendations.append(
                f"Domain diversity is low ({domain_diversity:.0%}). "
                "Consider adding sources from different websites."
            )

        if len(source_types) < self._min_source_types:
            recommendations.append(
                f"Only {len(source_types)} source type(s). "
                "Consider adding academic, news, or official government sources."
            )

        total_sources = sum(source_types.values())
        commercial_ratio = source_types.get("corporate", 0) / max(total_sources, 1)
        if commercial_ratio > self._commercial_threshold:
            recommendations.append(
                f"High ratio of corporate sources ({commercial_ratio:.0%}). "
                "Consider adding independent sources."
            )

        if "academic" not in source_types:
            recommendations.append(
                "No academic sources found. Consider adding peer-reviewed research."
            )

        if temporal_diversity < 0.5:
            recommendations.append(
                "Sources lack temporal diversity. "
                "Consider including both recent and historical sources."
            )

        # Calculate regional bias excluding international TLDs (which are geographically neutral)
        # Only count truly region-specific sources for geographic bias assessment
        regional_sources = {
            k: v for k, v in geo_distribution.items()
            if k not in ("international", "unknown", "other")
        }
        total_regional = sum(regional_sources.values())
        international_count = geo_distribution.get("international", 0)

        if total_regional > 0:
            us_ratio = regional_sources.get("us", 0) / total_regional
            if us_ratio > 0.8 and total_regional >= 3:
                recommendations.append(
                    f"Regional sources are heavily US-centric ({us_ratio:.0%} of {total_regional} regional sources). "
                    "Consider adding sources from other regions."
                )

        # Also check if there's very low regional diversity (all international TLDs)
        total_geo = sum(geo_distribution.values())
        if total_geo > 5 and total_regional == 0 and international_count > 0:
            recommendations.append(
                "All sources use international TLDs (.com, .org, etc.). "
                "Consider adding region-specific sources for geographic perspective diversity."
            )

        return recommendations

    def _generate_summary(
        self,
        diversity: DiversityReport,
        indicators: list[BiasIndicator],
        viewpoints: ViewpointAssessment,
        risk_level: SeverityLevel,
    ) -> str:
        """Generate a human-readable summary."""
        parts: list[str] = []

        # Overall assessment
        risk_descriptions = {
            "low": "Low bias risk detected",
            "medium": "Moderate bias concerns identified",
            "high": "Significant bias issues found",
        }
        parts.append(f"{risk_descriptions[risk_level]}.")

        # Diversity summary
        parts.append(
            f"Source diversity score: {diversity['overall_score']:.0%} "
            f"({diversity['domain_count']} unique domains)."
        )

        # Indicator summary
        if indicators:
            types = set(i["indicator_type"] for i in indicators)
            parts.append(f"Detected {len(indicators)} bias indicators: {', '.join(types)}.")
        else:
            parts.append("No explicit bias indicators detected in content.")

        # Viewpoint summary
        if viewpoints["viewpoint_count"] > 0:
            parts.append(
                f"Viewpoint balance: {viewpoints['balance_score']:.0%} "
                f"({viewpoints['viewpoint_count']} distinct viewpoints)."
            )

        # Top recommendation
        if diversity["recommendations"]:
            parts.append(f"Key recommendation: {diversity['recommendations'][0]}")

        return " ".join(parts)


# =============================================================================
# Factory Function
# =============================================================================


def create_bias_detector(**kwargs: Any) -> BiasDetector:
    """Create a configured BiasDetector instance."""
    return BiasDetector(**kwargs)


__all__ = [
    "BiasDetector",
    "BiasIndicator",
    "DiversityReport",
    "ViewpointAssessment",
    "BiasReport",
    "BiasType",
    "SeverityLevel",
    "create_bias_detector",
]
