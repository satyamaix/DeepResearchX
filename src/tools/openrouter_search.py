"""
OpenRouter Web Search Tool for DRX Deep Research system.

Provides web search capabilities using OpenRouter's web search plugin
(Exa-powered). Extracts citations from model responses with annotations.

Features:
- Web search via OpenRouter chat completion with :online plugin
- Citation extraction from annotations
- Cost tracking ($0.02 per search)
- Response parsing into standardized SearchResult format
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from .base import (
    SearchTool,
    SearchResult,
    ToolResult,
    RateLimiter,
    tool_with_retry,
)

logger = logging.getLogger(__name__)

# OpenRouter web search costs ~$0.02 per search
OPENROUTER_SEARCH_COST_USD = 0.02
OPENROUTER_RATE_LIMIT_PER_SECOND = 10.0


class OpenRouterSearchTool(SearchTool):
    """
    OpenRouter Web Search integration for DRX.

    Uses OpenRouter's web search plugin (Exa-powered) to search the web
    and extract citations from model responses.

    Features:
    - Async search via chat completion with :online suffix
    - Citation extraction from annotations
    - Cost tracking
    - Support for various models
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "google/gemini-2.0-flash-001",
        rate_limiter: RateLimiter | None = None,
        timeout: float = 60.0,
        default_max_results: int = 5,
        track_costs: bool = True,
    ):
        """
        Initialize OpenRouter Search Tool.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use for search (must support :online plugin)
            rate_limiter: Custom rate limiter
            timeout: Request timeout in seconds
            default_max_results: Default number of results to return
            track_costs: Whether to track search costs
        """
        if rate_limiter is None:
            rate_limiter = RateLimiter(
                requests_per_second=OPENROUTER_RATE_LIMIT_PER_SECOND,
                burst_size=20
            )

        super().__init__(
            rate_limiter=rate_limiter,
            timeout=timeout,
            default_max_results=default_max_results
        )

        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._model = model
        self._track_costs = track_costs
        self._total_cost_usd = 0.0
        self._search_count = 0

        # HTTP client
        self._client: httpx.AsyncClient | None = None

        # OpenRouter API base URL
        self._base_url = "https://openrouter.ai/api/v1"

    @property
    def name(self) -> str:
        return "openrouter_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using OpenRouter's Exa-powered search plugin. "
            "Returns web results with citations extracted from AI-processed "
            "search results. Best for comprehensive research queries."
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
                    "description": "Maximum number of results (1-10)",
                    "minimum": 1,
                    "maximum": 10,
                    "default": self.default_max_results
                },
                "search_prompt": {
                    "type": "string",
                    "description": "Custom prompt for search context"
                }
            },
            "required": ["query"]
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "HTTP-Referer": "https://drx.research.ai",
                    "X-Title": "DRX Deep Research"
                },
                timeout=self._timeout
            )
        return self._client

    @tool_with_retry(max_retries=3, base_delay=2.0)
    async def _search(
        self,
        query: str,
        max_results: int,
        **kwargs
    ) -> list[SearchResult]:
        """
        Execute OpenRouter web search.

        Args:
            query: Search query
            max_results: Maximum results to return
            **kwargs: Additional parameters

        Returns:
            List of SearchResult objects
        """
        client = await self._get_client()

        # Build search prompt
        search_prompt = kwargs.get("search_prompt") or self._build_search_prompt(
            query, max_results
        )

        # Use :online suffix for web search plugin
        model_with_search = f"{self._model}:online"

        # Build request payload
        payload = {
            "model": model_with_search,
            "messages": [
                {
                    "role": "user",
                    "content": search_prompt
                }
            ],
            "max_tokens": 2048,
            "plugins": [
                {
                    "id": "web",
                    "max_results": max_results
                }
            ]
        }

        logger.debug(f"OpenRouter search request: {payload}")

        # Execute request
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()

        # Track costs
        if self._track_costs:
            self._search_count += 1
            self._total_cost_usd += OPENROUTER_SEARCH_COST_USD

        # Parse response and extract citations
        return self._parse_response(data, query)

    def _build_search_prompt(self, query: str, max_results: int) -> str:
        """Build the search prompt for OpenRouter."""
        return f"""Search the web for: {query}

Provide {max_results} relevant results with:
1. The source URL
2. The page title
3. A brief summary of the relevant content

Format each result clearly."""

    def _parse_response(
        self,
        response: dict[str, Any],
        query: str
    ) -> list[SearchResult]:
        """
        Parse OpenRouter response and extract citations.

        Args:
            response: Raw OpenRouter API response
            query: Original query for metadata

        Returns:
            List of SearchResult objects
        """
        results = []

        # Get message content
        choices = response.get("choices", [])
        if not choices:
            logger.warning("No choices in OpenRouter response")
            return results

        message = choices[0].get("message", {})
        content = message.get("content", "")

        # Extract citations from annotations (OpenRouter web search format)
        annotations = message.get("annotations", [])

        if annotations:
            # Parse structured annotations
            results = self._parse_annotations(annotations, query)
        else:
            # Fallback: extract URLs and context from content
            results = self._extract_from_content(content, query)

        logger.info(
            f"OpenRouter search returned {len(results)} results for: {query[:50]}"
        )
        return results

    def _parse_annotations(
        self,
        annotations: list[dict[str, Any]],
        query: str
    ) -> list[SearchResult]:
        """
        Parse OpenRouter annotations into SearchResults.

        Args:
            annotations: List of annotation objects from response
            query: Original query

        Returns:
            List of SearchResult objects
        """
        results = []
        seen_urls = set()

        for idx, annotation in enumerate(annotations):
            # Handle different annotation formats
            url = ""
            title = ""
            snippet = ""

            if annotation.get("type") == "url_citation":
                url = annotation.get("url", "")
                title = annotation.get("title", annotation.get("text", ""))
                snippet = annotation.get("snippet", "")
            elif "url" in annotation:
                url = annotation.get("url", "")
                title = annotation.get("title", "")
                snippet = annotation.get("content", annotation.get("snippet", ""))

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            # Calculate score based on position
            score = max(0.1, 1.0 - (idx * 0.1))

            result = SearchResult(
                url=url,
                title=title or self._extract_title_from_url(url),
                snippet=snippet[:500] if snippet else "",
                score=score,
                metadata={
                    "source": "openrouter",
                    "query": query,
                    "rank": idx + 1,
                    "model": self._model,
                    "annotation_type": annotation.get("type", "unknown")
                }
            )
            results.append(result)

        return results

    def _extract_from_content(
        self,
        content: str,
        query: str
    ) -> list[SearchResult]:
        """
        Extract URLs and context from response content.

        Fallback method when annotations aren't available.

        Args:
            content: Response content text
            query: Original query

        Returns:
            List of SearchResult objects
        """
        results = []
        seen_urls = set()

        # URL pattern
        url_pattern = r'https?://[^\s\)\]<>"\']+[^\s\)\]<>"\',.]'
        urls = re.findall(url_pattern, content)

        for idx, url in enumerate(urls):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Try to extract context around URL
            snippet = self._extract_context_around_url(content, url)

            # Calculate score based on position
            score = max(0.1, 1.0 - (idx * 0.1))

            result = SearchResult(
                url=url,
                title=self._extract_title_from_url(url),
                snippet=snippet,
                score=score,
                metadata={
                    "source": "openrouter",
                    "query": query,
                    "rank": idx + 1,
                    "model": self._model,
                    "extracted_from_content": True
                }
            )
            results.append(result)

            if len(results) >= 10:  # Limit results
                break

        return results

    def _extract_context_around_url(self, content: str, url: str) -> str:
        """Extract text context around a URL."""
        try:
            idx = content.find(url)
            if idx == -1:
                return ""

            # Get surrounding text
            start = max(0, idx - 200)
            end = min(len(content), idx + len(url) + 200)

            context = content[start:end]

            # Clean up
            context = context.replace(url, "").strip()
            context = re.sub(r'\s+', ' ', context)

            return context[:300]
        except Exception:
            return ""

    def _extract_title_from_url(self, url: str) -> str:
        """Extract a readable title from URL."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]

            if path_parts:
                # Use last path segment
                title = path_parts[-1].replace('-', ' ').replace('_', ' ')
                # Remove file extension
                if '.' in title:
                    title = title.rsplit('.', 1)[0]
                return title.title()

            return parsed.netloc
        except Exception:
            return url[:50]

    async def search_with_system_prompt(
        self,
        query: str,
        system_prompt: str,
        max_results: int | None = None
    ) -> list[SearchResult]:
        """
        Execute search with a custom system prompt.

        Allows customizing the search behavior and result format.

        Args:
            query: Search query
            system_prompt: Custom system prompt
            max_results: Maximum results

        Returns:
            List of SearchResult objects
        """
        max_results = max_results or self.default_max_results
        client = await self._get_client()

        model_with_search = f"{self._model}:online"

        payload = {
            "model": model_with_search,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "max_tokens": 2048,
            "plugins": [{"id": "web", "max_results": max_results}]
        }

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()

        if self._track_costs:
            self._search_count += 1
            self._total_cost_usd += OPENROUTER_SEARCH_COST_USD

        return self._parse_response(data, query)

    def get_cost_stats(self) -> dict[str, Any]:
        """Get cost tracking statistics."""
        return {
            "search_count": self._search_count,
            "total_cost_usd": round(self._total_cost_usd, 4),
            "cost_per_search_usd": OPENROUTER_SEARCH_COST_USD
        }

    async def close(self):
        """Clean up resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


class OpenRouterClient:
    """
    OpenRouter client for chat completions with web search.

    This is a standalone client class that can be used directly
    or through the OpenRouterSearchTool.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """Initialize OpenRouter client."""
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OpenRouter API key required")

        self._base_url = base_url
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "HTTP-Referer": "https://drx.research.ai",
                    "X-Title": "DRX Deep Research"
                },
                timeout=60.0
            )
        return self._client

    async def chat_completion_with_search(
        self,
        messages: list[dict],
        model: str = "google/gemini-2.0-flash-001",
        max_results: int = 5
    ) -> dict:
        """
        Execute chat completion with web search.

        Args:
            messages: List of message dicts with role and content
            model: Model to use (will add :online suffix)
            max_results: Maximum search results

        Returns:
            OpenRouter API response dict
        """
        client = await self._get_client()

        # Add :online suffix if not present
        if not model.endswith(":online"):
            model = f"{model}:online"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "plugins": [{"id": "web", "max_results": max_results}]
        }

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()

        return response.json()

    async def close(self):
        """Clean up resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


# Factory functions
_openrouter_tool_instance: OpenRouterSearchTool | None = None


def get_openrouter_search_tool(
    api_key: str | None = None,
    **kwargs
) -> OpenRouterSearchTool:
    """
    Factory function to get OpenRouterSearchTool instance.

    Uses singleton pattern for efficiency but allows custom instances.

    Args:
        api_key: Optional API key override
        **kwargs: Additional configuration parameters

    Returns:
        OpenRouterSearchTool instance
    """
    global _openrouter_tool_instance

    if api_key is None and not kwargs:
        if _openrouter_tool_instance is None:
            _openrouter_tool_instance = OpenRouterSearchTool()
        return _openrouter_tool_instance

    return OpenRouterSearchTool(api_key=api_key, **kwargs)


async def create_openrouter_search_tool(
    api_key: str | None = None,
    **kwargs
) -> OpenRouterSearchTool:
    """
    Async factory function for OpenRouterSearchTool.

    Args:
        api_key: Optional API key override
        **kwargs: Additional configuration parameters

    Returns:
        Initialized OpenRouterSearchTool instance
    """
    tool = get_openrouter_search_tool(api_key=api_key, **kwargs)
    await tool._get_client()  # Pre-initialize client
    return tool
