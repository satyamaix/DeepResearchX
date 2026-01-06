"""FastAPI Dependency Injection for DRX API.

Provides reusable dependencies for database sessions, Redis clients,
orchestrator instances, authentication, and rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, AsyncGenerator

from fastapi import Depends, Header, HTTPException, Request, status
from redis.asyncio import ConnectionPool as RedisConnectionPool
from redis.asyncio import Redis

from psycopg import AsyncConnection

from src.config import Settings, get_settings
from src.db.connection import get_async_connection, get_async_pool
from src.orchestrator.workflow import ResearchOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Database Dependencies
# =============================================================================


async def get_db() -> AsyncGenerator[AsyncConnection[dict[str, Any]], None]:
    """Provide an async database connection from the pool.

    This dependency yields a connection that is automatically returned
    to the pool when the request completes.

    Yields:
        AsyncConnection: Database connection with dict row factory.

    Raises:
        HTTPException: If database connection fails.
    """
    try:
        async with get_async_connection() as conn:
            yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable",
        ) from e


# Type alias for dependency injection
DatabaseDep = Annotated[AsyncConnection[dict[str, Any]], Depends(get_db)]


# =============================================================================
# Redis Dependencies
# =============================================================================

# Module-level Redis connection pool singleton
_redis_pool: RedisConnectionPool | None = None
_redis_lock = asyncio.Lock()


async def get_redis_pool() -> RedisConnectionPool:
    """Get or create Redis connection pool singleton.

    Returns:
        RedisConnectionPool: Redis async connection pool.
    """
    global _redis_pool

    if _redis_pool is not None:
        return _redis_pool

    async with _redis_lock:
        if _redis_pool is not None:
            return _redis_pool

        settings = get_settings()
        _redis_pool = RedisConnectionPool.from_url(
            settings.redis_url_str,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            retry_on_timeout=settings.REDIS_RETRY_ON_TIMEOUT,
            decode_responses=True,
        )
        logger.info("Redis connection pool created")
        return _redis_pool


async def get_redis() -> AsyncGenerator[Redis, None]:
    """Provide a Redis client from the connection pool.

    Yields:
        Redis: Async Redis client instance.

    Raises:
        HTTPException: If Redis connection fails.
    """
    try:
        pool = await get_redis_pool()
        client = Redis(connection_pool=pool)
        try:
            yield client
        finally:
            await client.aclose()
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis connection unavailable",
        ) from e


async def close_redis_pool() -> None:
    """Close the Redis connection pool during shutdown."""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.disconnect()
        _redis_pool = None
        logger.info("Redis connection pool closed")


# Type alias for dependency injection
RedisDep = Annotated[Redis, Depends(get_redis)]


# =============================================================================
# Orchestrator Dependencies
# =============================================================================

# Module-level orchestrator instance
_orchestrator: ResearchOrchestrator | None = None
_orchestrator_lock = asyncio.Lock()


async def get_orchestrator(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ResearchOrchestrator:
    """Provide the research orchestrator instance.

    Creates a singleton orchestrator with database connection for checkpointing.

    Args:
        settings: Application settings.

    Returns:
        ResearchOrchestrator: The orchestrator instance.
    """
    global _orchestrator

    if _orchestrator is not None:
        return _orchestrator

    async with _orchestrator_lock:
        if _orchestrator is not None:
            return _orchestrator

        _orchestrator = ResearchOrchestrator(
            db_uri=settings.database_url_str,
        )
        await _orchestrator.initialize()
        logger.info("Research orchestrator initialized with LangGraph workflow")
        return _orchestrator


OrchestratorDep = Annotated[ResearchOrchestrator, Depends(get_orchestrator)]


# =============================================================================
# Authentication Dependencies
# =============================================================================


@dataclass
class User:
    """Authenticated user model."""

    id: str
    email: str | None = None
    is_active: bool = True
    is_admin: bool = False
    rate_limit_tier: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Settings = Depends(get_settings),
) -> User:
    """Validate authentication and return current user.

    Supports both Bearer token and API key authentication.
    This is a placeholder implementation - replace with actual
    authentication logic (JWT validation, API key lookup, etc.).

    Args:
        authorization: Bearer token from Authorization header.
        x_api_key: API key from X-API-Key header.
        settings: Application settings.

    Returns:
        User: Authenticated user object.

    Raises:
        HTTPException: If authentication fails.
    """
    # Development mode bypass
    if settings.is_development and not authorization and not x_api_key:
        return User(
            id="dev-user",
            email="dev@example.com",
            is_admin=True,
            rate_limit_tier="unlimited",
        )

    # API Key authentication
    if x_api_key:
        # TODO: Implement actual API key validation
        # Look up API key in database, validate, return associated user
        if x_api_key.startswith("drx_"):
            return User(
                id=f"apikey-{x_api_key[:12]}",
                rate_limit_tier="standard",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Bearer token authentication
    if authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = authorization[7:]  # Strip "Bearer "

        # TODO: Implement actual JWT validation
        # Decode token, verify signature, check expiration, return user
        if token:
            return User(
                id=f"bearer-{token[:12]}",
                rate_limit_tier="standard",
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer, ApiKey"},
    )


async def get_optional_user(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Settings = Depends(get_settings),
) -> User | None:
    """Get current user if authenticated, None otherwise.

    Useful for endpoints that support both authenticated and
    anonymous access with different rate limits.

    Returns:
        User if authenticated, None otherwise.
    """
    try:
        return await get_current_user(authorization, x_api_key, settings)
    except HTTPException:
        return None


CurrentUserDep = Annotated[User, Depends(get_current_user)]
OptionalUserDep = Annotated[User | None, Depends(get_optional_user)]


# =============================================================================
# Rate Limiting Dependencies
# =============================================================================


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    window_seconds: int = 60


class RateLimiter:
    """Token bucket rate limiter with Redis backend.

    Implements a sliding window rate limiter using Redis sorted sets
    for distributed rate limiting across multiple API instances.
    """

    # Rate limit tiers
    TIERS: dict[str, RateLimitConfig] = {
        "standard": RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_limit=10,
        ),
        "premium": RateLimitConfig(
            requests_per_minute=300,
            requests_per_hour=5000,
            burst_limit=50,
        ),
        "unlimited": RateLimitConfig(
            requests_per_minute=10000,
            requests_per_hour=100000,
            burst_limit=1000,
        ),
    }

    def __init__(
        self,
        redis: Redis,
        tier: str = "standard",
        key_prefix: str = "ratelimit",
    ) -> None:
        """Initialize rate limiter.

        Args:
            redis: Redis client.
            tier: Rate limit tier name.
            key_prefix: Redis key prefix for rate limit data.
        """
        self._redis = redis
        self._config = self.TIERS.get(tier, self.TIERS["standard"])
        self._key_prefix = key_prefix

    async def check_rate_limit(
        self,
        identifier: str,
        cost: int = 1,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request is within rate limits.

        Uses Redis sorted sets for sliding window counting.

        Args:
            identifier: Unique identifier (user ID, IP, API key).
            cost: Cost of this request (default 1).

        Returns:
            Tuple of (allowed, info_dict).
            info_dict contains remaining, limit, reset_at.
        """
        now = time.time()
        window_start = now - self._config.window_seconds
        minute_key = f"{self._key_prefix}:{identifier}:minute"

        pipe = self._redis.pipeline()

        # Remove old entries outside window
        pipe.zremrangebyscore(minute_key, 0, window_start)

        # Count current requests in window
        pipe.zcard(minute_key)

        # Add current request
        pipe.zadd(minute_key, {f"{now}:{cost}": now})

        # Set expiry on key
        pipe.expire(minute_key, self._config.window_seconds * 2)

        results = await pipe.execute()
        current_count = results[1]

        remaining = max(0, self._config.requests_per_minute - current_count - cost)
        allowed = current_count + cost <= self._config.requests_per_minute

        info = {
            "allowed": allowed,
            "remaining": remaining,
            "limit": self._config.requests_per_minute,
            "reset_at": int(now + self._config.window_seconds),
            "retry_after": self._config.window_seconds if not allowed else None,
        }

        if not allowed:
            # Remove the request we just added since it's not allowed
            await self._redis.zrem(minute_key, f"{now}:{cost}")

        return allowed, info

    async def get_usage(self, identifier: str) -> dict[str, Any]:
        """Get current rate limit usage for an identifier.

        Args:
            identifier: Unique identifier.

        Returns:
            Dict with usage statistics.
        """
        now = time.time()
        window_start = now - self._config.window_seconds
        minute_key = f"{self._key_prefix}:{identifier}:minute"

        # Count requests in current window
        count = await self._redis.zcount(minute_key, window_start, now)

        return {
            "used": count,
            "remaining": max(0, self._config.requests_per_minute - count),
            "limit": self._config.requests_per_minute,
            "window_seconds": self._config.window_seconds,
        }


class RateLimitDependency:
    """FastAPI dependency for rate limiting.

    Usage:
        @app.get("/endpoint")
        async def endpoint(
            rate_limit: Annotated[None, Depends(RateLimitDependency(cost=1))]
        ):
            ...
    """

    def __init__(self, cost: int = 1, use_ip_fallback: bool = True) -> None:
        """Initialize rate limit dependency.

        Args:
            cost: Cost of this endpoint (default 1).
            use_ip_fallback: Use client IP if no user authenticated.
        """
        self.cost = cost
        self.use_ip_fallback = use_ip_fallback

    async def __call__(
        self,
        request: Request,
        redis: RedisDep,
        user: OptionalUserDep,
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> None:
        """Check rate limit and raise HTTPException if exceeded.

        Args:
            request: FastAPI request object.
            redis: Redis client.
            user: Optional authenticated user.
            settings: Application settings.

        Raises:
            HTTPException: If rate limit exceeded (429).
        """
        # Skip rate limiting in development
        if settings.is_development:
            return

        # Determine identifier and tier
        if user:
            identifier = f"user:{user.id}"
            tier = user.rate_limit_tier
        elif self.use_ip_fallback:
            # Use client IP
            client_ip = request.client.host if request.client else "unknown"
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            identifier = f"ip:{client_ip}"
            tier = "standard"
        else:
            # No identifier available, skip rate limiting
            return

        limiter = RateLimiter(redis, tier=tier)
        allowed, info = await limiter.check_rate_limit(identifier, self.cost)

        # Add rate limit headers to response
        request.state.rate_limit_info = info

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset_at"]),
                    "Retry-After": str(info["retry_after"]),
                },
            )


# Convenience rate limit dependencies
rate_limit_standard = RateLimitDependency(cost=1)
rate_limit_heavy = RateLimitDependency(cost=5)
rate_limit_streaming = RateLimitDependency(cost=10)


# =============================================================================
# Settings Dependency
# =============================================================================

SettingsDep = Annotated[Settings, Depends(get_settings)]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Database
    "get_db",
    "DatabaseDep",
    # Redis
    "get_redis",
    "get_redis_pool",
    "close_redis_pool",
    "RedisDep",
    # Orchestrator
    "ResearchOrchestrator",
    "get_orchestrator",
    "OrchestratorDep",
    # Authentication
    "User",
    "get_current_user",
    "get_optional_user",
    "CurrentUserDep",
    "OptionalUserDep",
    # Rate Limiting
    "RateLimitConfig",
    "RateLimiter",
    "RateLimitDependency",
    "rate_limit_standard",
    "rate_limit_heavy",
    "rate_limit_streaming",
    # Settings
    "SettingsDep",
]
