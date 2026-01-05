"""
DRX Tools Package - Search and retrieval tools for deep research.

This package provides:
- OpenRouterSearchTool: Native web search via OpenRouter (preferred, no extra cost)
- TavilySearchTool: Web search via Tavily API (fallback, 1000 free/month)
- RAGRetriever: Vector similarity search with pgvector

OpenRouter native search is the primary search method, available for
Anthropic, OpenAI, Perplexity, and xAI models with no additional per-search cost.

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
    OPENROUTER_EXA_COST_PER_RESULT_USD,
    OPENROUTER_NATIVE_COST_USD,
    SEARCH_ENGINE_NATIVE,
    SEARCH_ENGINE_EXA,
    NATIVE_SEARCH_MODELS,
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
    # OpenRouter search (preferred - native search, no extra cost)
    "OpenRouterSearchTool",
    "OpenRouterClient",
    "get_openrouter_search_tool",
    "create_openrouter_search_tool",
    "OPENROUTER_EXA_COST_PER_RESULT_USD",
    "OPENROUTER_NATIVE_COST_USD",
    "SEARCH_ENGINE_NATIVE",
    "SEARCH_ENGINE_EXA",
    "NATIVE_SEARCH_MODELS",
    # Tavily search (fallback)
    "TavilySearchTool",
    "get_tavily_tool",
    "create_tavily_tool",
    "TAVILY_FREE_TIER_MONTHLY_LIMIT",
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
