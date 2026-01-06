"""Dataset Flywheel - Collect research sessions for training data.

This module implements the Dataset Flywheel pattern for continuous learning,
collecting user feedback and research sessions to create fine-tuning datasets.

WP-3A: Dataset Flywheel Implementation
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions (TypedDict for LangGraph compatibility)
# =============================================================================


class SessionRecord(TypedDict):
    """A research session record for training.

    Contains all the information needed to create a training example,
    including the query, response, quality metrics, and user feedback.
    """
    session_id: str
    query: str
    final_report: str
    citations: list[dict[str, Any]]
    user_rating: int | None  # 1-5 scale
    user_feedback: str | None
    quality_metrics: dict[str, float]
    bias_report: dict[str, Any] | None
    duration_seconds: float
    token_count: int
    model_used: str
    created_at: str
    feedback_labels: list[str]  # e.g., ["accurate", "comprehensive"]


class TrainingExample(TypedDict):
    """A formatted training example for fine-tuning."""
    id: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


class DatasetStatistics(TypedDict):
    """Statistics about the collected dataset."""
    total_sessions: int
    sessions_with_feedback: int
    sessions_by_rating: dict[str, int]
    avg_rating: float | None
    label_distribution: dict[str, int]
    total_tokens: int
    avg_tokens_per_session: float
    date_range: dict[str, str | None]


# =============================================================================
# Quality Tier Classification
# =============================================================================


QualityTier = Literal["gold", "silver", "bronze", "unrated"]


def classify_quality_tier(
    rating: int | None,
    quality_metrics: dict[str, float] | None,
) -> QualityTier:
    """Classify a session into a quality tier based on rating and metrics.

    Args:
        rating: User rating (1-5) if available
        quality_metrics: Quality metrics from the research session

    Returns:
        Quality tier classification
    """
    if rating is None:
        return "unrated"

    # Gold tier: Rating 5 or rating 4 with high quality metrics
    if rating == 5:
        return "gold"

    if rating == 4:
        if quality_metrics:
            coverage = quality_metrics.get("coverage_score", 0)
            confidence = quality_metrics.get("avg_confidence", 0)
            if coverage >= 0.8 and confidence >= 0.8:
                return "gold"
        return "silver"

    # Silver tier: Rating 3-4
    if rating >= 3:
        return "silver"

    # Bronze tier: Rating 1-2 (can be used for negative examples)
    return "bronze"


# =============================================================================
# Dataset Collector Implementation
# =============================================================================


class DatasetCollector:
    """Collect and curate research sessions for training datasets.

    This class manages the collection of research sessions, user feedback,
    and export of curated training datasets in various formats.

    The collector organizes data by quality tier:
    - gold/: High-quality sessions (rating 4-5 with good metrics)
    - silver/: Medium-quality sessions (rating 3-4)
    - bronze/: Low-quality sessions (rating 1-2, useful for negative examples)
    - unrated/: Sessions without user feedback
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """Initialize the dataset collector.

        Args:
            storage_path: Path to store training data. Defaults to data/training.
        """
        self.storage_path = storage_path or Path("data/training")
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directory structure."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create tier directories
        for tier in ["gold", "silver", "bronze", "unrated"]:
            (self.storage_path / tier).mkdir(exist_ok=True)

        # Create exports directory
        (self.storage_path / "exports").mkdir(exist_ok=True)

    def _generate_content_hash(self, query: str, report: str) -> str:
        """Generate a unique hash based on content.

        Args:
            query: Research query
            report: Final report content

        Returns:
            SHA-256 hash truncated to 16 characters
        """
        content = f"{query}::{report}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_session_path(self, session_id: str, tier: QualityTier) -> Path:
        """Get the storage path for a session.

        Args:
            session_id: Unique session identifier
            tier: Quality tier for the session

        Returns:
            Path to the session file
        """
        return self.storage_path / tier / f"{session_id}.json"

    def _load_session(self, session_id: str) -> tuple[SessionRecord | None, QualityTier | None]:
        """Load a session record by ID, searching all tiers.

        Args:
            session_id: Session identifier to find

        Returns:
            Tuple of (session record, tier) or (None, None) if not found
        """
        for tier in ["gold", "silver", "bronze", "unrated"]:
            path = self._get_session_path(session_id, tier)  # type: ignore[arg-type]
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    record: SessionRecord = json.load(f)
                    return record, tier  # type: ignore[return-value]
        return None, None

    def _save_session(self, record: SessionRecord, tier: QualityTier) -> None:
        """Save a session record to the appropriate tier.

        Args:
            record: Session record to save
            tier: Quality tier to save under
        """
        path = self._get_session_path(record["session_id"], tier)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)

    def _move_session(
        self,
        session_id: str,
        from_tier: QualityTier,
        to_tier: QualityTier,
    ) -> None:
        """Move a session between quality tiers.

        Args:
            session_id: Session to move
            from_tier: Current tier
            to_tier: Target tier
        """
        old_path = self._get_session_path(session_id, from_tier)
        new_path = self._get_session_path(session_id, to_tier)

        if old_path.exists():
            old_path.rename(new_path)

    def record_session(
        self,
        session_id: str,
        query: str,
        final_report: str,
        citations: list[dict[str, Any]],
        quality_metrics: dict[str, float],
        duration_seconds: float,
        token_count: int,
        model_used: str,
        bias_report: dict[str, Any] | None = None,
    ) -> str:
        """Record a completed research session.

        Args:
            session_id: Unique session identifier
            query: Research query
            final_report: Generated research report
            citations: List of citation records
            quality_metrics: Quality metrics from the session
            duration_seconds: Session duration
            token_count: Total tokens used
            model_used: Model identifier used
            bias_report: Optional bias analysis report

        Returns:
            Content hash for the recorded session
        """
        content_hash = self._generate_content_hash(query, final_report)
        now = datetime.utcnow().isoformat() + "Z"

        record: SessionRecord = {
            "session_id": session_id,
            "query": query,
            "final_report": final_report,
            "citations": citations,
            "user_rating": None,
            "user_feedback": None,
            "quality_metrics": quality_metrics,
            "bias_report": bias_report,
            "duration_seconds": duration_seconds,
            "token_count": token_count,
            "model_used": model_used,
            "created_at": now,
            "feedback_labels": [],
        }

        # Start in unrated tier
        self._save_session(record, "unrated")

        logger.info(
            f"Recorded session {session_id} with content hash {content_hash}",
            extra={"session_id": session_id, "content_hash": content_hash},
        )

        return content_hash

    def add_feedback(
        self,
        session_id: str,
        rating: int,
        feedback: str | None = None,
        labels: list[str] | None = None,
    ) -> bool:
        """Add user feedback to a session.

        Args:
            session_id: Session to add feedback to
            rating: User rating (1-5)
            feedback: Optional text feedback
            labels: Optional feedback labels (e.g., ["accurate", "comprehensive"])

        Returns:
            True if feedback was added successfully, False if session not found
        """
        record, current_tier = self._load_session(session_id)

        if record is None or current_tier is None:
            logger.warning(f"Session {session_id} not found for feedback")
            return False

        # Update record with feedback
        record["user_rating"] = rating
        record["user_feedback"] = feedback
        record["feedback_labels"] = labels or []

        # Determine new quality tier
        new_tier = classify_quality_tier(rating, record["quality_metrics"])

        # Save to new tier (or same tier if unchanged)
        if new_tier != current_tier:
            self._move_session(session_id, current_tier, new_tier)

        self._save_session(record, new_tier)

        logger.info(
            f"Added feedback to session {session_id}: rating={rating}, tier={new_tier}",
            extra={"session_id": session_id, "rating": rating, "tier": new_tier},
        )

        return True

    def _format_for_jsonl(self, record: SessionRecord) -> dict[str, Any]:
        """Format a session record for JSONL export.

        Args:
            record: Session record to format

        Returns:
            JSONL-compatible dictionary
        """
        return {
            "id": record["session_id"],
            "query": record["query"],
            "response": record["final_report"],
            "citations": record["citations"],
            "rating": record["user_rating"],
            "labels": record["feedback_labels"],
            "metrics": record["quality_metrics"],
            "model": record["model_used"],
            "tokens": record["token_count"],
        }

    def _format_for_hf(self, record: SessionRecord) -> dict[str, Any]:
        """Format a session record for HuggingFace datasets.

        Args:
            record: Session record to format

        Returns:
            HuggingFace-compatible dictionary with conversation format
        """
        return {
            "id": record["session_id"],
            "conversations": [
                {"role": "user", "content": record["query"]},
                {"role": "assistant", "content": record["final_report"]},
            ],
            "metadata": {
                "rating": record["user_rating"],
                "labels": record["feedback_labels"],
                "citations_count": len(record["citations"]),
                "model": record["model_used"],
            },
        }

    def _format_for_openai(self, record: SessionRecord) -> dict[str, Any]:
        """Format a session record for OpenAI fine-tuning.

        Args:
            record: Session record to format

        Returns:
            OpenAI fine-tuning compatible dictionary
        """
        system_prompt = (
            "You are DRX, an advanced deep research assistant. "
            "Provide comprehensive, well-cited research reports based on user queries. "
            "Include relevant sources and maintain objectivity."
        )

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": record["query"]},
                {"role": "assistant", "content": record["final_report"]},
            ],
        }

    def export_training_set(
        self,
        min_rating: int = 4,
        output_format: Literal["jsonl", "hf", "openai"] = "jsonl",
        output_path: Path | None = None,
        include_unrated: bool = False,
    ) -> Path:
        """Export curated sessions as training data.

        Args:
            min_rating: Minimum rating to include (1-5)
            output_format: Export format (jsonl, hf, openai)
            output_path: Custom output path (optional)
            include_unrated: Whether to include unrated sessions

        Returns:
            Path to the exported file
        """
        # Determine which tiers to include based on min_rating
        tiers_to_include: list[QualityTier] = []

        if min_rating <= 2:
            tiers_to_include.extend(["gold", "silver", "bronze"])
        elif min_rating <= 3:
            tiers_to_include.extend(["gold", "silver"])
        else:
            tiers_to_include.append("gold")
            if min_rating == 4:
                tiers_to_include.append("silver")

        if include_unrated:
            tiers_to_include.append("unrated")

        # Collect records from selected tiers
        records: list[SessionRecord] = []

        for tier in tiers_to_include:
            tier_path = self.storage_path / tier
            for file_path in tier_path.glob("*.json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    record: SessionRecord = json.load(f)

                    # Filter by rating if not unrated
                    if record["user_rating"] is not None:
                        if record["user_rating"] >= min_rating:
                            records.append(record)
                    elif include_unrated:
                        records.append(record)

        # Format records based on output format
        formatted_records: list[dict[str, Any]] = []

        for record in records:
            if output_format == "jsonl":
                formatted_records.append(self._format_for_jsonl(record))
            elif output_format == "hf":
                formatted_records.append(self._format_for_hf(record))
            elif output_format == "openai":
                formatted_records.append(self._format_for_openai(record))

        # Determine output path
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"training_set_{output_format}_{timestamp}.jsonl"

        if output_path is None:
            output_path = self.storage_path / "exports" / filename

        # Write output file
        with open(output_path, "w", encoding="utf-8") as f:
            for record in formatted_records:
                f.write(json.dumps(record, default=str) + "\n")

        logger.info(
            f"Exported {len(formatted_records)} records to {output_path}",
            extra={"count": len(formatted_records), "format": output_format},
        )

        return output_path

    def get_statistics(self) -> DatasetStatistics:
        """Get dataset statistics.

        Returns:
            Statistics about the collected dataset
        """
        total_sessions = 0
        sessions_with_feedback = 0
        sessions_by_rating: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        label_distribution: dict[str, int] = {}
        total_tokens = 0
        ratings_sum = 0
        ratings_count = 0

        earliest_date: str | None = None
        latest_date: str | None = None

        # Iterate through all tiers
        for tier in ["gold", "silver", "bronze", "unrated"]:
            tier_path = self.storage_path / tier

            for file_path in tier_path.glob("*.json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    record: SessionRecord = json.load(f)

                total_sessions += 1
                total_tokens += record["token_count"]

                # Track date range
                created_at = record["created_at"]
                if earliest_date is None or created_at < earliest_date:
                    earliest_date = created_at
                if latest_date is None or created_at > latest_date:
                    latest_date = created_at

                # Track ratings
                if record["user_rating"] is not None:
                    sessions_with_feedback += 1
                    rating_key = str(record["user_rating"])
                    sessions_by_rating[rating_key] = sessions_by_rating.get(rating_key, 0) + 1
                    ratings_sum += record["user_rating"]
                    ratings_count += 1

                # Track labels
                for label in record["feedback_labels"]:
                    label_distribution[label] = label_distribution.get(label, 0) + 1

        avg_rating = ratings_sum / ratings_count if ratings_count > 0 else None
        avg_tokens = total_tokens / total_sessions if total_sessions > 0 else 0.0

        return DatasetStatistics(
            total_sessions=total_sessions,
            sessions_with_feedback=sessions_with_feedback,
            sessions_by_rating=sessions_by_rating,
            avg_rating=avg_rating,
            label_distribution=label_distribution,
            total_tokens=total_tokens,
            avg_tokens_per_session=avg_tokens,
            date_range={"earliest": earliest_date, "latest": latest_date},
        )

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Get a session record by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session record or None if not found
        """
        record, _ = self._load_session(session_id)
        return record

    def list_sessions(
        self,
        tier: QualityTier | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionRecord]:
        """List session records with optional filtering.

        Args:
            tier: Filter by quality tier (optional)
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of session records
        """
        records: list[SessionRecord] = []
        tiers = [tier] if tier else ["gold", "silver", "bronze", "unrated"]

        count = 0
        skipped = 0

        for t in tiers:
            tier_path = self.storage_path / t

            for file_path in sorted(tier_path.glob("*.json"), reverse=True):
                if skipped < offset:
                    skipped += 1
                    continue

                if count >= limit:
                    break

                with open(file_path, "r", encoding="utf-8") as f:
                    record: SessionRecord = json.load(f)
                    records.append(record)
                    count += 1

            if count >= limit:
                break

        return records


# =============================================================================
# Factory Function
# =============================================================================


def create_dataset_collector(storage_path: str | None = None) -> DatasetCollector:
    """Create a DatasetCollector instance.

    Args:
        storage_path: Optional custom storage path

    Returns:
        Configured DatasetCollector instance
    """
    path = Path(storage_path) if storage_path else None
    return DatasetCollector(storage_path=path)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "SessionRecord",
    "TrainingExample",
    "DatasetStatistics",
    "QualityTier",
    "DatasetCollector",
    "classify_quality_tier",
    "create_dataset_collector",
]
