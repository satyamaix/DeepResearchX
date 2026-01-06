"""
Progress Publisher for streaming research events via Redis pub/sub.

Enables real-time progress updates from Celery workers to API clients.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.services.redis_client import get_redis_client

if TYPE_CHECKING:
    from src.services.redis_client import RedisClient
    from src.orchestrator.workflow import StreamEvent

logger = logging.getLogger(__name__)


class ProgressPublisher:
    """
    Publish research progress events to Redis pub/sub channels.

    Each session has its own channel for real-time updates.
    Events are also stored in a list for late subscribers.
    """

    CHANNEL_PREFIX = "drx:progress:"
    EVENTS_KEY_PREFIX = "drx:events:"
    EVENT_TTL = 3600  # 1 hour
    MAX_STORED_EVENTS = 1000

    def __init__(self, redis_client: RedisClient, session_id: str):
        self._redis = redis_client
        self._session_id = session_id
        self._channel = f"{self.CHANNEL_PREFIX}{session_id}"
        self._events_key = f"{self.EVENTS_KEY_PREFIX}{session_id}"

    @property
    def channel(self) -> str:
        """Get the pub/sub channel name for this session."""
        return self._channel

    async def publish(self, event: StreamEvent) -> int:
        """
        Publish an event to the session channel.

        Args:
            event: The StreamEvent to publish

        Returns:
            Number of subscribers that received the message
        """
        event_dict = event.to_dict()
        event_json = json.dumps(event_dict)

        # Publish to channel
        subscriber_count = await self._redis.publish(self._channel, event_json)

        # Store in event list for late subscribers
        # Access the underlying client for list operations
        client = await self._redis._ensure_initialized()
        await client.rpush(self._events_key, event_json)
        await client.ltrim(self._events_key, -self.MAX_STORED_EVENTS, -1)
        await client.expire(self._events_key, self.EVENT_TTL)

        logger.debug(
            f"Published event {event.event_type} to {subscriber_count} subscribers"
        )

        return subscriber_count

    async def get_stored_events(self, start: int = 0, end: int = -1) -> list[dict]:
        """
        Get stored events for late subscribers.

        Args:
            start: Start index
            end: End index (-1 for all)

        Returns:
            List of event dicts
        """
        client = await self._redis._ensure_initialized()
        events_json = await client.lrange(self._events_key, start, end)
        return [json.loads(e) for e in events_json]

    async def publish_status(
        self,
        status: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> int:
        """
        Publish a simple status update.

        Args:
            status: Status string (e.g., "running", "completed", "failed")
            message: Human-readable message
            data: Optional additional data

        Returns:
            Number of subscribers
        """
        event_dict = {
            "event_type": f"status_{status}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self._session_id,
            "data": {
                "status": status,
                "message": message,
                **(data or {}),
            },
        }
        event_json = json.dumps(event_dict)

        subscriber_count = await self._redis.publish(self._channel, event_json)

        # Store in event list
        client = await self._redis._ensure_initialized()
        await client.rpush(self._events_key, event_json)
        await client.expire(self._events_key, self.EVENT_TTL)

        return subscriber_count

    async def close(self) -> None:
        """Clean up resources."""
        pass  # Nothing to clean up for pub/sub


async def create_progress_publisher(session_id: str) -> ProgressPublisher:
    """
    Factory function to create a ProgressPublisher.

    Args:
        session_id: The research session ID

    Returns:
        Configured ProgressPublisher
    """
    redis_client = await get_redis_client()
    return ProgressPublisher(redis_client, session_id)


# =============================================================================
# Cancellation Support
# =============================================================================

CANCELLATION_KEY_PREFIX = "drx:cancel:"


async def request_cancellation(session_id: str) -> bool:
    """
    Request cancellation of a research session.

    Args:
        session_id: The session to cancel

    Returns:
        True if cancellation was requested
    """
    redis_client = await get_redis_client()
    key = f"{CANCELLATION_KEY_PREFIX}{session_id}"
    await redis_client.set(key, "1", ttl=3600)
    return True


async def is_cancelled(session_id: str) -> bool:
    """
    Check if a session has been cancelled.

    Args:
        session_id: The session to check

    Returns:
        True if the session is cancelled
    """
    redis_client = await get_redis_client()
    key = f"{CANCELLATION_KEY_PREFIX}{session_id}"
    result = await redis_client.get(key)
    return result is not None


async def clear_cancellation(session_id: str) -> None:
    """
    Clear cancellation flag.

    Args:
        session_id: The session to clear cancellation for
    """
    redis_client = await get_redis_client()
    key = f"{CANCELLATION_KEY_PREFIX}{session_id}"
    await redis_client.delete(key)


__all__ = [
    "ProgressPublisher",
    "create_progress_publisher",
    "request_cancellation",
    "is_cancelled",
    "clear_cancellation",
]
