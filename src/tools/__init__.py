"""
DRX Tools Package - Search and retrieval tools for deep research.

This package provides:
- TavilySearchTool: Web search via Tavily API (1000 free/month)
- OpenRouterSearchTool: Web search via OpenRouter (Exa-powered)
- RAGRetriever: Vector similarity search with pgvector

All tools follow a consistent interface defined in base.py.
"""

from .base import (
    # Core types
    BaseTool,
    SearchTool,
    ToolResult,
    ToolStatus,
    SearchResult,
    RateLimiter,
    # Decorators
    tool_with_retry,
    # Type aliases
    ToolFactory,
    SearchResults,
)

from .tavily_search import (
    TavilySearchTool,
    get_tavily_tool,
    create_tavily_tool,
    TAVILY_FREE_TIER_MONTHLY_LIMIT,
)

from .openrouter_search import (
    OpenRouterSearchTool,
    OpenRouterClient,
    get_openrouter_search_tool,
    create_openrouter_search_tool,
    OPENROUTER_SEARCH_COST_USD,
)

from .rag_retriever import (
    RAGRetriever,
    Document,
    RetrievalResult,
    EmbeddingProvider,
    OpenRouterEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_rag_retriever,
    create_rag_retriever,
    EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
)

__all__ = [
    # Base classes and types
    "BaseTool",
    "SearchTool",
    "ToolResult",
    "ToolStatus",
    "SearchResult",
    "RateLimiter",
    "tool_with_retry",
    "ToolFactory",
    "SearchResults",
    # Tavily search
    "TavilySearchTool",
    "get_tavily_tool",
    "create_tavily_tool",
    "TAVILY_FREE_TIER_MONTHLY_LIMIT",
    # OpenRouter search
    "OpenRouterSearchTool",
    "OpenRouterClient",
    "get_openrouter_search_tool",
    "create_openrouter_search_tool",
    "OPENROUTER_SEARCH_COST_USD",
    # RAG retriever
    "RAGRetriever",
    "Document",
    "RetrievalResult",
    "EmbeddingProvider",
    "OpenRouterEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_rag_retriever",
    "create_rag_retriever",
    "EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
]
