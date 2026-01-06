"""Feedback Store - Async feedback storage with Redis caching.

This module implements async feedback storage with Redis caching for fast
access and PostgreSQL for persistence, enabling the Dataset Flywheel pattern.

WP-3A: Dataset Flywheel Implementation
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, TypedDict

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions (TypedDict for LangGraph compatibility)
# =============================================================================


class FeedbackRecord(TypedDict):
    """User feedback record.

    Stores user feedback for a research session, including rating,
    comments, and categorical labels.
    """
    feedback_id: str
    session_id: str
    user_id: str | None
    rating: int  # 1-5 scale
    comment: str | None
    labels: list[str]  # e.g., ["accurate", "comprehensive", "well-cited"]
    created_at: str
    metadata: dict[str, Any]


class AggregateMetrics(TypedDict):
    """Aggregate feedback metrics."""
    total_feedback: int
    avg_rating: float | None
    rating_distribution: dict[str, int]  # {"1": count, "2": count, ...}
    common_labels: list[tuple[str, int]]  # [(label, count), ...]
    sessions_with_feedback: int
    feedback_rate: float  # percentage of sessions with feedback


# =============================================================================
# Redis Key Patterns
# =============================================================================


class RedisKeys:
    """Redis key patterns for feedback storage."""

    @staticmethod
    def feedback_by_session(session_id: str) -> str:
        """Key for session feedback list."""
        return f"drx:feedback:session:{session_id}"

    @staticmethod
    def feedback_record(feedback_id: str) -> str:
        """Key for individual feedback record."""
        return f"drx:feedback:record:{feedback_id}"

    @staticmethod
    def aggregate_metrics() -> str:
        """Key for aggregate metrics cache."""
        return "drx:feedback:metrics:aggregate"

    @staticmethod
    def label_counts() -> str:
        """Key for label count tracking."""
        return "drx:feedback:metrics:labels"

    @staticmethod
    def rating_counts() -> str:
        """Key for rating count tracking."""
        return "drx:feedback:metrics:ratings"


# =============================================================================
# Feedback Store Implementation
# =============================================================================


class FeedbackStore:
    """Async feedback storage with caching.

    Provides async feedback submission and retrieval with Redis caching
    for fast access and optional PostgreSQL persistence.

    The store supports:
    - Async feedback submission with automatic caching
    - Retrieval of feedback by session ID
    - Aggregate metrics calculation
    - Integration with Dataset Flywheel for training data collection
    """

    # Cache TTL values (in seconds)
    FEEDBACK_CACHE_TTL = 86400 * 7  # 7 days
    METRICS_CACHE_TTL = 300  # 5 minutes
    SESSION_FEEDBACK_TTL = 86400 * 30  # 30 days

    def __init__(
        self,
        redis: Redis,
        db_pool: Any | None = None,
    ) -> None:
        """Initialize the feedback store.

        Args:
            redis: Async Redis client
            db_pool: Optional PostgreSQL connection pool for persistence
        """
        self.redis = redis
        self.db_pool = db_pool

    async def submit_feedback(
        self,
        session_id: str,
        rating: int,
        comment: str | None = None,
        labels: list[str] | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit feedback for a research session.

        Args:
            session_id: Research session ID
            rating: Rating from 1-5
            comment: Optional text comment
            labels: Optional list of feedback labels
            user_id: Optional user identifier
            metadata: Optional additional metadata

        Returns:
            Generated feedback ID

        Raises:
            ValueError: If rating is out of valid range (1-5)
        """
        # Validate rating
        if not 1 <= rating <= 5:
            raise ValueError(f"Rating must be between 1 and 5, got {rating}")

        # Generate feedback ID
        feedback_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"

        # Create feedback record
        record: FeedbackRecord = {
            "feedback_id": feedback_id,
            "session_id": session_id,
            "user_id": user_id,
            "rating": rating,
            "comment": comment,
            "labels": labels or [],
            "created_at": now,
            "metadata": metadata or {},
        }

        # Store in Redis
        await self._cache_feedback(record)

        # Update aggregate metrics
        await self._update_aggregate_metrics(rating, labels or [])

        # Persist to PostgreSQL if available
        if self.db_pool is not None:
            await self._persist_feedback(record)

        logger.info(
            f"Submitted feedback {feedback_id} for session {session_id}",
            extra={
                "feedback_id": feedback_id,
                "session_id": session_id,
                "rating": rating,
            },
        )

        return feedback_id

    async def _cache_feedback(self, record: FeedbackRecord) -> None:
        """Cache feedback record in Redis.

        Args:
            record: Feedback record to cache
        """
        feedback_id = record["feedback_id"]
        session_id = record["session_id"]
        record_json = json.dumps(record, default=str)

        # Store the feedback record
        await self.redis.set(
            RedisKeys.feedback_record(feedback_id),
            record_json,
            ex=self.FEEDBACK_CACHE_TTL,
        )

        # Add to session's feedback list
        await self.redis.lpush(
            RedisKeys.feedback_by_session(session_id),
            feedback_id,
        )
        await self.redis.expire(
            RedisKeys.feedback_by_session(session_id),
            self.SESSION_FEEDBACK_TTL,
        )

    async def _update_aggregate_metrics(
        self,
        rating: int,
        labels: list[str],
    ) -> None:
        """Update aggregate metrics in Redis.

        Args:
            rating: New rating to include
            labels: New labels to include
        """
        # Update rating counts
        await self.redis.hincrby(
            RedisKeys.rating_counts(),
            str(rating),
            1,
        )

        # Update label counts
        for label in labels:
            await self.redis.hincrby(
                RedisKeys.label_counts(),
                label,
                1,
            )

        # Invalidate cached aggregate metrics
        await self.redis.delete(RedisKeys.aggregate_metrics())

    async def _persist_feedback(self, record: FeedbackRecord) -> None:
        """Persist feedback to PostgreSQL.

        Args:
            record: Feedback record to persist
        """
        if self.db_pool is None:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO feedback (
                        feedback_id, session_id, user_id, rating,
                        comment, labels, created_at, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (feedback_id) DO UPDATE SET
                        rating = EXCLUDED.rating,
                        comment = EXCLUDED.comment,
                        labels = EXCLUDED.labels,
                        metadata = EXCLUDED.metadata
                    """,
                    record["feedback_id"],
                    record["session_id"],
                    record["user_id"],
                    record["rating"],
                    record["comment"],
                    json.dumps(record["labels"]),
                    record["created_at"],
                    json.dumps(record["metadata"]),
                )
        except Exception as e:
            logger.error(
                f"Failed to persist feedback {record['feedback_id']}: {e}",
                exc_info=True,
            )
            # Don't raise - we have Redis as primary storage

    async def get_feedback(self, session_id: str) -> list[FeedbackRecord]:
        """Get all feedback for a session.

        Args:
            session_id: Research session ID

        Returns:
            List of feedback records for the session
        """
        # Get feedback IDs for session
        feedback_ids = await self.redis.lrange(
            RedisKeys.feedback_by_session(session_id),
            0,
            -1,
        )

        if not feedback_ids:
            # Try PostgreSQL if no cache hit
            if self.db_pool is not None:
                return await self._get_feedback_from_db(session_id)
            return []

        # Fetch individual feedback records
        records: list[FeedbackRecord] = []
        for feedback_id in feedback_ids:
            record = await self._get_feedback_record(feedback_id)
            if record:
                records.append(record)

        # Sort by created_at descending
        records.sort(key=lambda r: r["created_at"], reverse=True)

        return records

    async def _get_feedback_record(
        self,
        feedback_id: str,
    ) -> FeedbackRecord | None:
        """Get a single feedback record from cache.

        Args:
            feedback_id: Feedback record ID

        Returns:
            Feedback record or None if not found
        """
        record_json = await self.redis.get(RedisKeys.feedback_record(feedback_id))

        if record_json:
            return json.loads(record_json)

        return None

    async def _get_feedback_from_db(
        self,
        session_id: str,
    ) -> list[FeedbackRecord]:
        """Get feedback from PostgreSQL.

        Args:
            session_id: Session ID to get feedback for

        Returns:
            List of feedback records
        """
        if self.db_pool is None:
            return []

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT feedback_id, session_id, user_id, rating,
                           comment, labels, created_at, metadata
                    FROM feedback
                    WHERE session_id = $1
                    ORDER BY created_at DESC
                    """,
                    session_id,
                )

                return [
                    FeedbackRecord(
                        feedback_id=row["feedback_id"],
                        session_id=row["session_id"],
                        user_id=row["user_id"],
                        rating=row["rating"],
                        comment=row["comment"],
                        labels=json.loads(row["labels"]) if row["labels"] else [],
                        created_at=row["created_at"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get feedback from DB for session {session_id}: {e}")
            return []

    async def get_aggregate_metrics(self) -> AggregateMetrics:
        """Get aggregate feedback metrics.

        Returns cached metrics if available, otherwise calculates from stored data.

        Returns:
            Aggregate metrics including average rating, distribution, and common labels
        """
        # Try cache first
        cached = await self.redis.get(RedisKeys.aggregate_metrics())
        if cached:
            return json.loads(cached)

        # Calculate from stored data
        metrics = await self._calculate_aggregate_metrics()

        # Cache the result
        await self.redis.set(
            RedisKeys.aggregate_metrics(),
            json.dumps(metrics),
            ex=self.METRICS_CACHE_TTL,
        )

        return metrics

    async def _calculate_aggregate_metrics(self) -> AggregateMetrics:
        """Calculate aggregate metrics from Redis data.

        Returns:
            Calculated aggregate metrics
        """
        # Get rating distribution
        rating_counts = await self.redis.hgetall(RedisKeys.rating_counts())

        rating_distribution: dict[str, int] = {}
        total_feedback = 0
        weighted_sum = 0

        for rating_str, count_str in rating_counts.items():
            rating = int(rating_str)
            count = int(count_str)
            rating_distribution[rating_str] = count
            total_feedback += count
            weighted_sum += rating * count

        # Calculate average rating
        avg_rating = weighted_sum / total_feedback if total_feedback > 0 else None

        # Get label distribution
        label_counts = await self.redis.hgetall(RedisKeys.label_counts())

        # Sort labels by count
        sorted_labels = sorted(
            [(label, int(count)) for label, count in label_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Estimate sessions with feedback (this is approximate from Redis)
        # In production, this would query the database
        sessions_with_feedback = total_feedback  # Approximate
        feedback_rate = 0.0  # Would need total session count

        return AggregateMetrics(
            total_feedback=total_feedback,
            avg_rating=round(avg_rating, 2) if avg_rating else None,
            rating_distribution=rating_distribution,
            common_labels=sorted_labels[:10],  # Top 10 labels
            sessions_with_feedback=sessions_with_feedback,
            feedback_rate=feedback_rate,
        )

    async def get_feedback_by_rating(
        self,
        min_rating: int,
        max_rating: int = 5,
        limit: int = 100,
    ) -> list[FeedbackRecord]:
        """Get feedback records within a rating range.

        This is useful for curating training data based on quality.

        Args:
            min_rating: Minimum rating (inclusive)
            max_rating: Maximum rating (inclusive)
            limit: Maximum records to return

        Returns:
            List of feedback records matching the criteria
        """
        if self.db_pool is None:
            logger.warning("No database pool available for rating-based query")
            return []

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT feedback_id, session_id, user_id, rating,
                           comment, labels, created_at, metadata
                    FROM feedback
                    WHERE rating >= $1 AND rating <= $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    min_rating,
                    max_rating,
                    limit,
                )

                return [
                    FeedbackRecord(
                        feedback_id=row["feedback_id"],
                        session_id=row["session_id"],
                        user_id=row["user_id"],
                        rating=row["rating"],
                        comment=row["comment"],
                        labels=json.loads(row["labels"]) if row["labels"] else [],
                        created_at=row["created_at"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get feedback by rating: {e}")
            return []

    async def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback record.

        Args:
            feedback_id: Feedback ID to delete

        Returns:
            True if deleted successfully
        """
        # Get record to find session_id
        record = await self._get_feedback_record(feedback_id)
        if not record:
            return False

        session_id = record["session_id"]

        # Remove from Redis
        await self.redis.delete(RedisKeys.feedback_record(feedback_id))
        await self.redis.lrem(
            RedisKeys.feedback_by_session(session_id),
            0,
            feedback_id,
        )

        # Update aggregate metrics (decrement counts)
        await self.redis.hincrby(
            RedisKeys.rating_counts(),
            str(record["rating"]),
            -1,
        )
        for label in record["labels"]:
            await self.redis.hincrby(RedisKeys.label_counts(), label, -1)

        # Invalidate cached metrics
        await self.redis.delete(RedisKeys.aggregate_metrics())

        # Delete from database if available
        if self.db_pool is not None:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        "DELETE FROM feedback WHERE feedback_id = $1",
                        feedback_id,
                    )
            except Exception as e:
                logger.error(f"Failed to delete feedback from DB: {e}")

        logger.info(f"Deleted feedback {feedback_id}")
        return True


# =============================================================================
# Factory Function
# =============================================================================


async def create_feedback_store(
    redis: Redis,
    db_pool: Any | None = None,
) -> FeedbackStore:
    """Create a FeedbackStore instance.

    Args:
        redis: Async Redis client
        db_pool: Optional PostgreSQL connection pool

    Returns:
        Configured FeedbackStore instance
    """
    return FeedbackStore(redis=redis, db_pool=db_pool)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "FeedbackRecord",
    "AggregateMetrics",
    "FeedbackStore",
    "RedisKeys",
    "create_feedback_store",
]
