"""DRX Services Module.

Provides client services for external APIs and infrastructure.

Core Components:
- RedisClient: Async Redis operations (caching, pub/sub, rate limiting)
- OpenRouterClient: LLM API client with retry logic and streaming

Example:
    ```python
    from src.services import get_redis_client, get_openrouter_client

    # Redis operations
    redis = await get_redis_client()
    await redis.set("key", "value", ttl=3600)
    value = await redis.get("key")

    # LLM completions
    openrouter = await get_openrouter_client()
    response = await openrouter.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}],
        model="google/gemini-2.5-flash-preview"
    )
    print(response.content)
    ```
"""

from src.services.redis_client import (
    # Main client class
    RedisClient,
    # Singleton factory
    get_redis_client,
    close_redis_client,
    # Types
    CircuitState,
    CircuitBreakerConfig,
    RateLimitResult,
    # Utilities
    cache_key,
    session_cache_key,
    # Exceptions
    RedisError,
    RedisConnectionError,
)

from src.services.openrouter_client import (
    # Main client class
    OpenRouterClient,
    # Singleton factory
    get_openrouter_client,
    close_openrouter_client,
    # Data classes
    LLMModelConfig,
    ModelConfig,  # Backwards compat alias for LLMModelConfig
    ChatMessage,
    CompletionResponse,
    StreamChunk,
    MessageRole,
    # Model configurations
    MODELS,
    # Token utilities
    estimate_token_count,
    estimate_messages_tokens,
    # Convenience functions
    quick_completion,
    # Exceptions
    OpenRouterError,
    OpenRouterRateLimitError,
    OpenRouterAuthError,
    OpenRouterAPIError,
    OpenRouterTimeoutError,
)


__all__ = [
    # Redis
    "RedisClient",
    "get_redis_client",
    "close_redis_client",
    "CircuitState",
    "CircuitBreakerConfig",
    "RateLimitResult",
    "cache_key",
    "session_cache_key",
    "RedisError",
    "RedisConnectionError",
    # OpenRouter
    "OpenRouterClient",
    "get_openrouter_client",
    "close_openrouter_client",
    "LLMModelConfig",
    "ModelConfig",  # Alias for LLMModelConfig
    "ChatMessage",
    "CompletionResponse",
    "StreamChunk",
    "MessageRole",
    "MODELS",
    "estimate_token_count",
    "estimate_messages_tokens",
    "quick_completion",
    "OpenRouterError",
    "OpenRouterRateLimitError",
    "OpenRouterAuthError",
    "OpenRouterAPIError",
    "OpenRouterTimeoutError",
]
