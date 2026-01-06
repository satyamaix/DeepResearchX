"""
Citation Verification Tool for DRX Deep Research System.

Provides verification of citations including:
- URL accessibility checking
- Quote/snippet verification against source content
- Fuzzy matching for approximate quote finding
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, TypedDict

import httpx
from rapidfuzz import fuzz

if TYPE_CHECKING:
    from ..orchestrator.state import CitationRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================


class URLStatus(TypedDict):
    """Result of URL accessibility check."""

    url: str
    accessible: bool
    status_code: int | None
    error: str | None
    response_time_ms: float
    checked_at: str


class QuoteVerification(TypedDict):
    """Result of quote verification against source content."""

    quote: str
    found: bool
    similarity: float  # 0-1, fuzzy match score
    best_match: str | None
    match_position: int | None  # Character offset in source
    verification_method: str


class VerificationResult(TypedDict):
    """Complete verification result for a citation."""

    citation_id: str
    url: str
    url_accessible: bool
    url_status_code: int | None
    quote_found: bool
    quote_similarity: float
    best_match: str | None
    issues: list[str]
    verified_at: str
    overall_valid: bool


# =============================================================================
# Citation Verifier
# =============================================================================


class CitationVerifier:
    """
    Verify citations by checking URL accessibility and quote matching.

    Uses fuzzy string matching to handle minor variations in quotes
    (whitespace, punctuation, etc.).
    """

    def __init__(
        self,
        timeout: float = 10.0,
        similarity_threshold: float = 0.85,
        user_agent: str = "DRX-Research/1.0 (Citation Verifier)",
        max_concurrent: int = 5,
    ):
        """
        Initialize the citation verifier.

        Args:
            timeout: HTTP request timeout in seconds
            similarity_threshold: Minimum similarity score to consider quote found
            user_agent: User-Agent header for requests
            max_concurrent: Maximum concurrent HTTP requests
        """
        self._timeout = timeout
        self._similarity_threshold = similarity_threshold
        self._user_agent = user_agent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def verify_url(self, url: str) -> URLStatus:
        """
        Check if a URL is accessible.

        Args:
            url: The URL to check

        Returns:
            URLStatus with accessibility information
        """
        start = time.perf_counter()

        try:
            async with self._semaphore:
                async with httpx.AsyncClient(
                    timeout=self._timeout,
                    follow_redirects=True,
                ) as client:
                    response = await client.head(
                        url,
                        headers={"User-Agent": self._user_agent},
                    )

                    # Some servers don't support HEAD, try GET
                    if response.status_code == 405:
                        response = await client.get(
                            url,
                            headers={"User-Agent": self._user_agent},
                        )

                    elapsed_ms = (time.perf_counter() - start) * 1000

                    return URLStatus(
                        url=url,
                        accessible=response.status_code < 400,
                        status_code=response.status_code,
                        error=None,
                        response_time_ms=round(elapsed_ms, 2),
                        checked_at=datetime.utcnow().isoformat() + "Z",
                    )

        except httpx.TimeoutException:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return URLStatus(
                url=url,
                accessible=False,
                status_code=None,
                error="Request timed out",
                response_time_ms=round(elapsed_ms, 2),
                checked_at=datetime.utcnow().isoformat() + "Z",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return URLStatus(
                url=url,
                accessible=False,
                status_code=None,
                error=str(e),
                response_time_ms=round(elapsed_ms, 2),
                checked_at=datetime.utcnow().isoformat() + "Z",
            )

    def verify_quote(
        self,
        quote: str,
        source_content: str,
        threshold: float | None = None,
    ) -> QuoteVerification:
        """
        Verify if a quote exists in source content using fuzzy matching.

        Args:
            quote: The quote to find
            source_content: The content to search in
            threshold: Override default similarity threshold

        Returns:
            QuoteVerification with match results
        """
        if not quote or not source_content:
            return QuoteVerification(
                quote=quote,
                found=False,
                similarity=0.0,
                best_match=None,
                match_position=None,
                verification_method="none",
            )

        threshold = threshold or self._similarity_threshold

        # Normalize text for comparison
        normalized_quote = self._normalize_text(quote)
        normalized_source = self._normalize_text(source_content)

        # First try exact substring match
        if normalized_quote in normalized_source:
            position = source_content.lower().find(quote.lower()[:50])
            return QuoteVerification(
                quote=quote,
                found=True,
                similarity=1.0,
                best_match=quote,
                match_position=position if position >= 0 else None,
                verification_method="exact",
            )

        # Sliding window fuzzy match for longer quotes
        quote_len = len(normalized_quote)
        source_len = len(normalized_source)

        if quote_len > 20 and source_len > quote_len:
            best_score = 0.0
            best_match: str | None = None
            best_position: int | None = None
            window_size = min(quote_len * 2, source_len)
            step = max(1, quote_len // 4)

            for i in range(0, source_len - window_size + 1, step):
                window = normalized_source[i : i + window_size]
                # Use partial ratio for substring matching
                score = fuzz.partial_ratio(normalized_quote, window) / 100.0

                if score > best_score:
                    best_score = score
                    best_position = i
                    # Extract the matching portion
                    best_match = source_content[i : i + len(quote) + 50][: len(quote) + 20]

            return QuoteVerification(
                quote=quote,
                found=best_score >= threshold,
                similarity=round(best_score, 3),
                best_match=best_match if best_score >= threshold * 0.8 else None,
                match_position=best_position if best_score >= threshold else None,
                verification_method="fuzzy_window",
            )

        # For shorter quotes, use simple fuzzy ratio
        score = fuzz.ratio(normalized_quote, normalized_source[:500]) / 100.0

        return QuoteVerification(
            quote=quote,
            found=score >= threshold,
            similarity=round(score, 3),
            best_match=source_content[:100] if score >= threshold * 0.8 else None,
            match_position=0 if score >= threshold else None,
            verification_method="fuzzy_ratio",
        )

    async def verify_citation(
        self,
        citation: CitationRecord,
        source_content: str | None = None,
    ) -> VerificationResult:
        """
        Fully verify a citation (URL + quote).

        Args:
            citation: The citation to verify
            source_content: Optional pre-fetched source content

        Returns:
            Complete VerificationResult
        """
        issues: list[str] = []

        # Check URL
        url_status = await self.verify_url(citation.get("url", ""))

        if not url_status["accessible"]:
            issues.append(f"URL not accessible: {url_status.get('error', 'Unknown error')}")

        # Check quote if we have source content
        quote_result = QuoteVerification(
            quote=citation.get("snippet", ""),
            found=False,
            similarity=0.0,
            best_match=None,
            match_position=None,
            verification_method="skipped",
        )

        if source_content and citation.get("snippet"):
            quote_result = self.verify_quote(
                citation.get("snippet", ""),
                source_content,
            )

            if not quote_result["found"]:
                issues.append(
                    f"Quote not found in source (similarity: {quote_result['similarity']:.0%})"
                )

        # Determine overall validity
        overall_valid = url_status["accessible"] and (
            not source_content or quote_result["found"]
        )

        return VerificationResult(
            citation_id=citation.get("id", ""),
            url=citation.get("url", ""),
            url_accessible=url_status["accessible"],
            url_status_code=url_status["status_code"],
            quote_found=quote_result["found"],
            quote_similarity=quote_result["similarity"],
            best_match=quote_result["best_match"],
            issues=issues,
            verified_at=datetime.utcnow().isoformat() + "Z",
            overall_valid=overall_valid,
        )

    async def batch_verify(
        self,
        citations: list[CitationRecord],
        fetch_content: bool = False,
    ) -> list[VerificationResult]:
        """
        Verify multiple citations concurrently.

        Args:
            citations: List of citations to verify
            fetch_content: Whether to fetch source content for quote verification

        Returns:
            List of VerificationResults
        """

        async def verify_one(citation: CitationRecord) -> VerificationResult:
            source_content = None
            if fetch_content:
                try:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        response = await client.get(
                            citation.get("url", ""),
                            headers={"User-Agent": self._user_agent},
                            follow_redirects=True,
                        )
                        if response.status_code == 200:
                            source_content = response.text
                except Exception:
                    pass  # Will be noted as URL not accessible

            return await self.verify_citation(citation, source_content)

        tasks = [verify_one(c) for c in citations]
        return await asyncio.gather(*tasks)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove punctuation variations
        text = re.sub(r'[""''`]', '"', text)
        text = re.sub(r"[–—]", "-", text)
        return text.strip()


# =============================================================================
# Factory Function
# =============================================================================


def create_citation_verifier(**kwargs: float | str | int) -> CitationVerifier:
    """Create a configured CitationVerifier instance."""
    return CitationVerifier(**kwargs)  # type: ignore[arg-type]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CitationVerifier",
    "URLStatus",
    "QuoteVerification",
    "VerificationResult",
    "create_citation_verifier",
]
