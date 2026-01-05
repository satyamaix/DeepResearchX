"""
Tavily Search Tool for DRX Deep Research system.

Provides web search capabilities using the Tavily Search API.
Supports:
- Basic search queries
- Context-aware search
- Rate limiting for free tier (1000 searches/month)
- Response parsing into standardized SearchResult format
"""

from __future__ import annotations

import logging
import os
from typing import Any

from tavily import TavilyClient, AsyncTavilyClient

from .base import (
    SearchTool,
    SearchResult,
    ToolResult,
    RateLimiter,
    tool_with_retry,
)

logger = logging.getLogger(__name__)

# Tavily free tier: 1000 searches per month
TAVILY_FREE_TIER_MONTHLY_LIMIT = 1000
TAVILY_RATE_LIMIT_PER_SECOND = 5.0  # Conservative estimate


class TavilySearchTool(SearchTool):
    """
    Tavily Search API integration for DRX.

    Features:
    - Async search operations
    - Context-aware search for refined queries
    - Automatic rate limiting for free tier
    - Response parsing into SearchResult format
    - Supports search depth (basic/advanced) modes
    """

    def __init__(
        self,
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 30.0,
        default_max_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False,
        include_images: bool = False,
    ):
        """
        Initialize Tavily Search Tool.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
            rate_limiter: Custom rate limiter (auto-created if None)
            timeout: Request timeout in seconds
            default_max_results: Default number of results to return
            search_depth: "basic" or "advanced" (advanced costs more)
            include_raw_content: Include raw page content in results
            include_images: Include image results
        """
        # Create rate limiter for free tier if not provided
        if rate_limiter is None:
            rate_limiter = RateLimiter(
                requests_per_second=TAVILY_RATE_LIMIT_PER_SECOND,
                burst_size=10,
                max_requests_per_period=TAVILY_FREE_TIER_MONTHLY_LIMIT,
                period_seconds=30 * 24 * 60 * 60  # 30 days
            )

        super().__init__(
            rate_limiter=rate_limiter,
            timeout=timeout,
            default_max_results=default_max_results
        )

        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Tavily API key required. Set TAVILY_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._search_depth = search_depth
        self._include_raw_content = include_raw_content
        self._include_images = include_images

        # Initialize async client
        self._client: AsyncTavilyClient | None = None

    @property
    def name(self) -> str:
        return "tavily_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using Tavily API. Returns relevant web pages with "
            "titles, URLs, and content snippets. Best for finding current "
            "information, news, documentation, and factual content."
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (1-20)",
                    "minimum": 1,
                    "maximum": 20,
                    "default": self.default_max_results
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "Search depth - advanced provides more thorough results",
                    "default": "basic"
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit search to specific domains"
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude specific domains from search"
                }
            },
            "required": ["query"]
        }

    async def _get_client(self) -> AsyncTavilyClient:
        """Get or create async Tavily client."""
        if self._client is None:
            self._client = AsyncTavilyClient(api_key=self._api_key)
        return self._client

    @tool_with_retry(max_retries=3, base_delay=1.0)
    async def _search(
        self,
        query: str,
        max_results: int,
        **kwargs
    ) -> list[SearchResult]:
        """
        Execute Tavily search.

        Args:
            query: Search query
            max_results: Maximum results to return
            **kwargs: Additional search parameters

        Returns:
            List of SearchResult objects
        """
        client = await self._get_client()

        # Build search parameters
        search_params = {
            "query": query,
            "max_results": min(max_results, 20),  # Tavily max is 20
            "search_depth": kwargs.get("search_depth", self._search_depth),
            "include_raw_content": kwargs.get(
                "include_raw_content", self._include_raw_content
            ),
            "include_images": kwargs.get("include_images", self._include_images),
        }

        # Optional domain filtering
        if "include_domains" in kwargs:
            search_params["include_domains"] = kwargs["include_domains"]
        if "exclude_domains" in kwargs:
            search_params["exclude_domains"] = kwargs["exclude_domains"]

        logger.debug(f"Tavily search params: {search_params}")

        # Execute search
        response = await client.search(**search_params)

        # Parse response into SearchResult objects
        return self._parse_response(response, query)

    def _parse_response(
        self,
        response: dict[str, Any],
        query: str
    ) -> list[SearchResult]:
        """
        Parse Tavily API response into SearchResult objects.

        Args:
            response: Raw Tavily API response
            query: Original query for metadata

        Returns:
            List of SearchResult objects
        """
        results = []

        # Tavily returns results in 'results' key
        raw_results = response.get("results", [])

        for idx, item in enumerate(raw_results):
            # Calculate normalized score (Tavily scores are 0-1)
            score = item.get("score", 0.0)

            # Build metadata
            metadata = {
                "source": "tavily",
                "query": query,
                "rank": idx + 1,
                "search_depth": response.get("search_depth", "basic"),
            }

            # Include raw content if available
            if "raw_content" in item and item["raw_content"]:
                metadata["raw_content"] = item["raw_content"][:5000]  # Truncate

            # Include publish date if available
            if "published_date" in item:
                metadata["published_date"] = item["published_date"]

            result = SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                score=score,
                metadata=metadata
            )
            results.append(result)

        logger.info(f"Tavily search returned {len(results)} results for: {query[:50]}")
        return results

    async def search_with_context(
        self,
        query: str,
        context: str,
        max_results: int | None = None
    ) -> list[SearchResult]:
        """
        Execute context-aware search.

        Combines the query with additional context to refine search results.
        Useful for follow-up queries in a research session.

        Args:
            query: Base search query
            context: Additional context to refine the search
            max_results: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        max_results = max_results or self.default_max_results

        # Combine query with context
        # Truncate context to avoid overly long queries
        context_snippet = context[:500] if len(context) > 500 else context
        enhanced_query = f"{query} (context: {context_snippet})"

        logger.info(f"Context-aware search: {query[:50]}...")

        result = await self.invoke(
            enhanced_query,
            max_results=max_results,
            search_depth="advanced"  # Use advanced for context-aware
        )

        if result.success and isinstance(result.data, list):
            # Add context flag to metadata
            for r in result.data:
                r.metadata["context_enhanced"] = True
            return result.data
        return []

    async def get_search_context(
        self,
        query: str,
        max_results: int = 5
    ) -> str:
        """
        Get a text context from search results.

        Uses Tavily's get_search_context method for summarized content.
        Useful for RAG pipelines.

        Args:
            query: Search query
            max_results: Number of results to include in context

        Returns:
            Concatenated text context from search results
        """
        client = await self._get_client()

        try:
            context = await client.get_search_context(
                query=query,
                max_results=max_results,
                search_depth="advanced"
            )
            return context
        except Exception as e:
            logger.error(f"Failed to get search context: {e}")
            return ""

    async def qna_search(
        self,
        query: str
    ) -> str:
        """
        Execute Q&A search for direct answers.

        Uses Tavily's QnA endpoint for direct question answering.

        Args:
            query: Question to answer

        Returns:
            Direct answer string
        """
        client = await self._get_client()

        try:
            answer = await client.qna_search(query=query)
            return answer
        except Exception as e:
            logger.error(f"QnA search failed: {e}")
            return ""

    def get_remaining_searches(self) -> int | None:
        """Get remaining searches in the current period."""
        if self._rate_limiter:
            return self._rate_limiter.remaining_in_period
        return None

    async def close(self):
        """Clean up resources."""
        self._client = None


# Factory function
_tavily_tool_instance: TavilySearchTool | None = None


def get_tavily_tool(
    api_key: str | None = None,
    **kwargs
) -> TavilySearchTool:
    """
    Factory function to get TavilySearchTool instance.

    Uses singleton pattern for efficiency but allows custom instances.

    Args:
        api_key: Optional API key override
        **kwargs: Additional configuration parameters

    Returns:
        TavilySearchTool instance
    """
    global _tavily_tool_instance

    # Return singleton if no custom config
    if api_key is None and not kwargs:
        if _tavily_tool_instance is None:
            _tavily_tool_instance = TavilySearchTool()
        return _tavily_tool_instance

    # Create custom instance
    return TavilySearchTool(api_key=api_key, **kwargs)


async def create_tavily_tool(
    api_key: str | None = None,
    **kwargs
) -> TavilySearchTool:
    """
    Async factory function for TavilySearchTool.

    Useful when you need to await initialization.

    Args:
        api_key: Optional API key override
        **kwargs: Additional configuration parameters

    Returns:
        Initialized TavilySearchTool instance
    """
    tool = get_tavily_tool(api_key=api_key, **kwargs)
    # Pre-initialize client
    await tool._get_client()
    return tool
