"""Redis Client for DRX.

Provides async Redis operations for:
- Caching with TTL support
- Pub/Sub for SSE streaming channels
- Rate limiting counters
- Circuit breaker state management
- Distributed locking
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from src.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton
_redis_client: RedisClient | None = None
_client_lock = asyncio.Lock()


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    current_count: int
    limit: int
    remaining: int
    reset_at: float
    retry_after: float | None = None


class RedisError(Exception):
    """Base exception for Redis operations."""

    pass


class RedisConnectionError(RedisError):
    """Exception raised when Redis connection fails."""

    pass


class RedisClient:
    """Async Redis client with utilities for DRX.

    Provides high-level methods for common Redis operations including
    caching, pub/sub, rate limiting, and circuit breaker management.
    """

    def __init__(
        self,
        url: str | None = None,
        max_connections: int | None = None,
        socket_timeout: float | None = None,
        retry_on_timeout: bool = True,
        decode_responses: bool = True,
    ):
        """Initialize Redis client.

        Args:
            url: Redis connection URL. Uses REDIS_URL from config if not provided.
            max_connections: Maximum connections in pool.
            socket_timeout: Socket timeout in seconds.
            retry_on_timeout: Whether to retry on timeout.
            decode_responses: Whether to decode responses to strings.
        """
        settings = get_settings()

        self.url = url or settings.redis_url_str
        self.max_connections = max_connections or settings.REDIS_MAX_CONNECTIONS
        self.socket_timeout = socket_timeout or settings.REDIS_SOCKET_TIMEOUT
        self.retry_on_timeout = retry_on_timeout
        self.decode_responses = decode_responses

        self._pool: ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._pubsub_clients: dict[str, redis.client.PubSub] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

        # Circuit breaker config from settings
        self._circuit_config = CircuitBreakerConfig(
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            half_open_max_calls=settings.CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
        )

    async def initialize(self) -> None:
        """Initialize the Redis connection pool and client."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                self._pool = ConnectionPool.from_url(
                    self.url,
                    max_connections=self.max_connections,
                    socket_timeout=self.socket_timeout,
                    retry_on_timeout=self.retry_on_timeout,
                    decode_responses=self.decode_responses,
                )

                self._client = redis.Redis(connection_pool=self._pool)

                # Test connection
                await self._client.ping()

                self._initialized = True
                logger.info(
                    "Redis client initialized",
                    extra={"url": self._mask_url(self.url), "max_connections": self.max_connections},
                )

            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise RedisConnectionError(f"Redis connection failed: {e}") from e
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
                raise RedisError(f"Redis initialization failed: {e}") from e

    def _mask_url(self, url: str) -> str:
        """Mask password in Redis URL for logging."""
        if "@" in url:
            parts = url.split("@")
            return f"redis://***@{parts[-1]}"
        return url

    async def _ensure_initialized(self) -> redis.Redis:
        """Ensure client is initialized and return it."""
        if not self._initialized or self._client is None:
            await self.initialize()
        return self._client  # type: ignore

    # ==================== Basic Operations ====================

    async def get(self, key: str) -> str | None:
        """Get a value from Redis.

        Args:
            key: The key to retrieve.

        Returns:
            The value if found, None otherwise.
        """
        client = await self._ensure_initialized()
        return await client.get(key)

    async def set(
        self,
        key: str,
        value: str | bytes | int | float,
        ttl: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set a value in Redis.

        Args:
            key: The key to set.
            value: The value to store.
            ttl: Time to live in seconds.
            nx: Only set if key does not exist.
            xx: Only set if key already exists.

        Returns:
            bool: True if set was successful.
        """
        client = await self._ensure_initialized()
        result = await client.set(key, value, ex=ttl, nx=nx, xx=xx)
        return result is not None and result is not False

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys.

        Args:
            *keys: Keys to delete.

        Returns:
            int: Number of keys deleted.
        """
        client = await self._ensure_initialized()
        return await client.delete(*keys)

    async def exists(self, *keys: str) -> int:
        """Check if keys exist.

        Args:
            *keys: Keys to check.

        Returns:
            int: Number of keys that exist.
        """
        client = await self._ensure_initialized()
        return await client.exists(*keys)

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on a key.

        Args:
            key: The key.
            ttl: Time to live in seconds.

        Returns:
            bool: True if expiration was set.
        """
        client = await self._ensure_initialized()
        return await client.expire(key, ttl)

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for a key.

        Args:
            key: The key.

        Returns:
            int: Seconds remaining, -1 if no expiry, -2 if key doesn't exist.
        """
        client = await self._ensure_initialized()
        return await client.ttl(key)

    # ==================== JSON Operations ====================

    async def get_json(self, key: str) -> Any | None:
        """Get and deserialize JSON value.

        Args:
            key: The key.

        Returns:
            Deserialized JSON value or None.
        """
        value = await self.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON for key: {key}")
            return None

    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Serialize and set JSON value.

        Args:
            key: The key.
            value: Value to serialize.
            ttl: Time to live in seconds.

        Returns:
            bool: True if set was successful.
        """
        try:
            serialized = json.dumps(value, default=str)
            return await self.set(key, serialized, ttl=ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize JSON for key {key}: {e}")
            return False

    # ==================== Counter Operations ====================

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter.

        Args:
            key: The counter key.
            amount: Amount to increment by.

        Returns:
            int: New counter value.
        """
        client = await self._ensure_initialized()
        return await client.incrby(key, amount)

    async def decr(self, key: str, amount: int = 1) -> int:
        """Decrement a counter.

        Args:
            key: The counter key.
            amount: Amount to decrement by.

        Returns:
            int: New counter value.
        """
        client = await self._ensure_initialized()
        return await client.decrby(key, amount)

    # ==================== Rate Limiting ====================

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """Check and update rate limit using sliding window.

        Args:
            key: Rate limit key (e.g., "ratelimit:user:123").
            limit: Maximum requests allowed.
            window_seconds: Time window in seconds.

        Returns:
            RateLimitResult: Result with allowed status and metadata.
        """
        client = await self._ensure_initialized()

        # Use Redis pipeline for atomic operation
        now = time.time()
        window_start = now - window_seconds

        async with client.pipeline(transaction=True) as pipe:
            # Remove old entries outside window
            await pipe.zremrangebyscore(key, 0, window_start)
            # Count current entries
            await pipe.zcard(key)
            # Add current request
            await pipe.zadd(key, {str(now): now})
            # Set expiration
            await pipe.expire(key, window_seconds)

            results = await pipe.execute()

        current_count = results[1]
        reset_at = now + window_seconds

        if current_count >= limit:
            # Get oldest entry to calculate retry_after
            oldest = await client.zrange(key, 0, 0, withscores=True)
            retry_after = None
            if oldest:
                oldest_time = oldest[0][1]
                retry_after = oldest_time + window_seconds - now

            return RateLimitResult(
                allowed=False,
                current_count=current_count,
                limit=limit,
                remaining=0,
                reset_at=reset_at,
                retry_after=retry_after,
            )

        return RateLimitResult(
            allowed=True,
            current_count=current_count + 1,  # +1 for the just-added request
            limit=limit,
            remaining=limit - current_count - 1,
            reset_at=reset_at,
        )

    async def simple_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> bool:
        """Simple rate limit check using INCR.

        Faster but less accurate than sliding window.

        Args:
            key: Rate limit key.
            limit: Maximum requests.
            window_seconds: Time window.

        Returns:
            bool: True if request is allowed.
        """
        client = await self._ensure_initialized()

        current = await client.incr(key)

        if current == 1:
            await client.expire(key, window_seconds)

        return current <= limit

    # ==================== Pub/Sub Operations ====================

    async def publish(self, channel: str, message: str | dict[str, Any]) -> int:
        """Publish a message to a channel.

        Args:
            channel: Channel name.
            message: Message to publish (string or dict).

        Returns:
            int: Number of subscribers that received the message.
        """
        client = await self._ensure_initialized()

        if isinstance(message, dict):
            message = json.dumps(message, default=str)

        return await client.publish(channel, message)

    async def subscribe(
        self,
        *channels: str,
    ) -> redis.client.PubSub:
        """Subscribe to one or more channels.

        Args:
            *channels: Channel names to subscribe to.

        Returns:
            PubSub: PubSub client for receiving messages.
        """
        client = await self._ensure_initialized()
        pubsub = client.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub

    @asynccontextmanager
    async def subscribe_context(
        self,
        *channels: str,
    ) -> AsyncGenerator[redis.client.PubSub, None]:
        """Context manager for pub/sub subscription.

        Automatically unsubscribes and closes on exit.

        Args:
            *channels: Channels to subscribe to.

        Yields:
            PubSub: PubSub client.
        """
        pubsub = await self.subscribe(*channels)
        try:
            yield pubsub
        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.close()

    async def listen(
        self,
        pubsub: redis.client.PubSub,
        timeout: float | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Listen for messages on a PubSub client.

        Args:
            pubsub: PubSub client from subscribe().
            timeout: Optional timeout in seconds.

        Yields:
            dict: Message dictionaries with type, channel, data.
        """
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message["data"]
                # Try to parse JSON
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        pass
                yield {
                    "type": message["type"],
                    "channel": message["channel"],
                    "data": data,
                }

    # ==================== SSE Channel Operations ====================

    async def publish_sse_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> int:
        """Publish an SSE event to a session channel.

        Args:
            session_id: Research session ID.
            event_type: Event type (e.g., "progress", "result", "error").
            data: Event data.

        Returns:
            int: Number of subscribers.
        """
        channel = f"sse:{session_id}"
        message = {
            "event": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        return await self.publish(channel, message)

    async def subscribe_sse(self, session_id: str) -> redis.client.PubSub:
        """Subscribe to SSE channel for a session.

        Args:
            session_id: Research session ID.

        Returns:
            PubSub: PubSub client for the session channel.
        """
        channel = f"sse:{session_id}"
        return await self.subscribe(channel)

    # ==================== Circuit Breaker Operations ====================

    async def get_circuit_state(self, service_name: str) -> CircuitState:
        """Get current circuit breaker state for a service.

        Args:
            service_name: Name of the service.

        Returns:
            CircuitState: Current circuit state.
        """
        key = f"circuit:{service_name}:state"
        state = await self.get(key)

        if state is None:
            return CircuitState.CLOSED

        try:
            return CircuitState(state)
        except ValueError:
            return CircuitState.CLOSED

    async def set_circuit_state(
        self,
        service_name: str,
        state: CircuitState,
        ttl: int | None = None,
    ) -> None:
        """Set circuit breaker state for a service.

        Args:
            service_name: Name of the service.
            state: New circuit state.
            ttl: Optional TTL (used for OPEN state auto-recovery).
        """
        key = f"circuit:{service_name}:state"
        ttl = ttl or self._circuit_config.recovery_timeout
        await self.set(key, state.value, ttl=ttl)

    async def record_circuit_failure(self, service_name: str) -> CircuitState:
        """Record a failure and potentially open the circuit.

        Args:
            service_name: Name of the service.

        Returns:
            CircuitState: New circuit state after recording failure.
        """
        failure_key = f"circuit:{service_name}:failures"
        window = self._circuit_config.recovery_timeout

        failures = await self.incr(failure_key)
        if failures == 1:
            await self.expire(failure_key, window)

        if failures >= self._circuit_config.failure_threshold:
            await self.set_circuit_state(
                service_name,
                CircuitState.OPEN,
                ttl=self._circuit_config.recovery_timeout,
            )
            logger.warning(f"Circuit opened for service: {service_name}")
            return CircuitState.OPEN

        return await self.get_circuit_state(service_name)

    async def record_circuit_success(self, service_name: str) -> None:
        """Record a success and potentially close the circuit.

        Args:
            service_name: Name of the service.
        """
        failure_key = f"circuit:{service_name}:failures"
        await self.delete(failure_key)

        current_state = await self.get_circuit_state(service_name)
        if current_state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
            await self.set_circuit_state(service_name, CircuitState.CLOSED)
            logger.info(f"Circuit closed for service: {service_name}")

    async def should_allow_request(self, service_name: str) -> bool:
        """Check if a request should be allowed based on circuit state.

        Args:
            service_name: Name of the service.

        Returns:
            bool: True if request should be allowed.
        """
        state = await self.get_circuit_state(service_name)

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            state_key = f"circuit:{service_name}:state"
            ttl = await self.ttl(state_key)

            if ttl == -2:  # Key expired, transition to half-open
                await self.set_circuit_state(service_name, CircuitState.HALF_OPEN)
                return True
            return False

        if state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            half_open_key = f"circuit:{service_name}:half_open_count"
            count = await self.incr(half_open_key)

            if count == 1:
                await self.expire(half_open_key, self._circuit_config.recovery_timeout)

            return count <= self._circuit_config.half_open_max_calls

        return True

    # ==================== Distributed Lock ====================

    @asynccontextmanager
    async def lock(
        self,
        name: str,
        timeout: int = 30,
        blocking: bool = True,
        blocking_timeout: float | None = None,
    ) -> AsyncGenerator[bool, None]:
        """Acquire a distributed lock.

        Args:
            name: Lock name.
            timeout: Lock timeout in seconds.
            blocking: Whether to block waiting for lock.
            blocking_timeout: Max time to wait for lock.

        Yields:
            bool: True if lock was acquired.
        """
        client = await self._ensure_initialized()
        lock = client.lock(
            f"lock:{name}",
            timeout=timeout,
            blocking=blocking,
            blocking_timeout=blocking_timeout,
        )

        acquired = await lock.acquire()
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    await lock.release()
                except redis.exceptions.LockNotOwnedError:
                    logger.warning(f"Lock {name} was already released or expired")

    # ==================== Health Check ====================

    async def health_check(self) -> dict[str, Any]:
        """Check Redis connection health.

        Returns:
            dict: Health status with latency info.
        """
        try:
            client = await self._ensure_initialized()
            start = time.time()
            await client.ping()
            latency = time.time() - start

            info = await client.info("server")

            return {
                "status": "healthy",
                "latency_ms": round(latency * 1000, 2),
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    # ==================== Cleanup ====================

    async def close(self) -> None:
        """Close the Redis client and connection pool."""
        if self._client is not None:
            await self._client.close()
            self._client = None

        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None

        self._initialized = False
        logger.info("Redis client closed")


# ==================== Singleton Factory ====================


async def get_redis_client() -> RedisClient:
    """Get or create the Redis client singleton.

    Returns:
        RedisClient: Configured Redis client instance.
    """
    global _redis_client

    if _redis_client is not None and _redis_client._initialized:
        return _redis_client

    async with _client_lock:
        if _redis_client is not None and _redis_client._initialized:
            return _redis_client

        _redis_client = RedisClient()
        await _redis_client.initialize()

        return _redis_client


async def close_redis_client() -> None:
    """Close the singleton Redis client."""
    global _redis_client

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


# ==================== Utility Functions ====================


def cache_key(*parts: str) -> str:
    """Generate a cache key from parts.

    Args:
        *parts: Key parts to join.

    Returns:
        str: Cache key with colon separators.
    """
    return ":".join(parts)


def session_cache_key(session_id: str, suffix: str) -> str:
    """Generate a session-specific cache key.

    Args:
        session_id: Research session ID.
        suffix: Key suffix.

    Returns:
        str: Session cache key.
    """
    return f"session:{session_id}:{suffix}"
