"""
Context Storage Abstraction for DRX Deep Research System.

This module provides the storage layer for ResearchContext objects,
supporting both Redis (for active contexts with TTL) and PostgreSQL
(for persistent storage and vector operations via pgvector).

CRITICAL: Uses TypedDict for LangGraph compatibility with serialization.

Part of WP-M3: Context Propagation System implementation.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

ContextStoreType = Literal["redis", "postgres", "hybrid"]


class ContextStatus(str, Enum):
    """Status of a research context in the system."""

    ACTIVE = "active"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    DELETED = "deleted"


# =============================================================================
# TypedDict Definitions (LangGraph compatible)
# =============================================================================


class ResearchContext(TypedDict):
    """
    Research context for propagation between agents and iterations.

    This TypedDict is the core data structure for context management,
    designed for LangGraph serialization compatibility.

    Attributes:
        context_id: Unique identifier for this context
        session_id: Session this context belongs to
        summary: Compressed context summary for efficient propagation
        key_entities: Extracted named entities and key terms
        relevance_vector: Embedding vector for similarity-based relevance checks
        chunk_refs: References to full content chunks in pgvector
        created_at: ISO 8601 timestamp of creation
        ttl_seconds: Time-to-live before context expires
    """

    context_id: str
    session_id: str
    summary: str
    key_entities: list[str]
    relevance_vector: list[float]
    chunk_refs: list[str]
    created_at: str
    ttl_seconds: int


class ContextMetadata(TypedDict, total=False):
    """
    Optional metadata for a research context.

    Provides additional information about context origin and usage.
    """

    source_agent: str
    iteration_number: int
    parent_context_id: str | None
    compression_ratio: float
    original_token_count: int
    compressed_token_count: int
    status: str
    updated_at: str


class ContextStoreConfig(TypedDict, total=False):
    """
    Configuration for context store initialization.

    Supports both Redis and PostgreSQL backends.
    """

    # Redis configuration
    redis_url: str
    redis_prefix: str
    redis_default_ttl: int

    # PostgreSQL configuration
    postgres_dsn: str
    postgres_pool_size: int
    pgvector_table: str

    # Hybrid configuration
    use_redis_cache: bool
    cache_ttl_seconds: int


# =============================================================================
# Abstract Base Class
# =============================================================================


class ContextStore(ABC):
    """
    Abstract base class for context storage implementations.

    Provides the interface for storing, retrieving, and managing
    ResearchContext objects with support for TTL-based expiration.

    Subclasses must implement all abstract methods to provide
    backend-specific storage functionality.
    """

    @abstractmethod
    async def store_context(
        self,
        context: ResearchContext,
        metadata: ContextMetadata | None = None,
    ) -> None:
        """
        Store a research context.

        Args:
            context: The ResearchContext to store
            metadata: Optional metadata to store alongside context

        Raises:
            ContextStoreError: If storage fails
        """
        pass

    @abstractmethod
    async def get_context(self, context_id: str) -> ResearchContext | None:
        """
        Retrieve a context by its ID.

        Args:
            context_id: Unique identifier of the context

        Returns:
            ResearchContext if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_context(self, context_id: str) -> None:
        """
        Delete a context by its ID.

        Args:
            context_id: Unique identifier of the context to delete

        Raises:
            ContextStoreError: If deletion fails
        """
        pass

    @abstractmethod
    async def get_contexts_by_session(
        self,
        session_id: str,
        limit: int = 100,
        include_expired: bool = False,
    ) -> list[ResearchContext]:
        """
        Get all contexts for a session.

        Args:
            session_id: Session identifier to filter by
            limit: Maximum number of contexts to return
            include_expired: Whether to include expired contexts

        Returns:
            List of ResearchContext objects for the session
        """
        pass

    @abstractmethod
    async def update_ttl(self, context_id: str, ttl_seconds: int) -> bool:
        """
        Update the TTL for an existing context.

        Args:
            context_id: Unique identifier of the context
            ttl_seconds: New TTL value in seconds

        Returns:
            True if updated, False if context not found
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close connections and cleanup resources.

        Should be called when the store is no longer needed.
        """
        pass


# =============================================================================
# Custom Exceptions
# =============================================================================


class ContextStoreError(Exception):
    """Base exception for context store operations."""

    pass


class ContextNotFoundError(ContextStoreError):
    """Raised when a requested context is not found."""

    pass


class ContextStorageError(ContextStoreError):
    """Raised when storage operations fail."""

    pass


# =============================================================================
# Redis Context Store Implementation
# =============================================================================


class RedisContextStore(ContextStore):
    """
    Redis-backed context store for active contexts.

    Uses Redis for fast access to active research contexts with
    automatic TTL-based expiration. Ideal for short-term context
    propagation between agents within a session.

    Features:
        - Automatic TTL expiration
        - Fast key-value access
        - Session-based indexing via sorted sets
        - Metadata storage via hashes

    Example:
        >>> store = RedisContextStore(redis_url="redis://localhost:6379")
        >>> await store.initialize()
        >>> await store.store_context(context)
        >>> retrieved = await store.get_context(context["context_id"])
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "drx:context:",
        default_ttl: int = 3600,
    ) -> None:
        """
        Initialize Redis context store.

        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for all context keys
            default_ttl: Default TTL in seconds for contexts
        """
        self._redis_url = redis_url
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._redis: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize Redis connection.

        Must be called before using the store.

        Raises:
            ContextStorageError: If connection fails
        """
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self._redis.ping()
            self._initialized = True
            logger.info(f"Redis context store initialized: {self._redis_url}")
        except ImportError:
            raise ContextStorageError(
                "redis package not installed. Install with: pip install redis"
            )
        except Exception as e:
            raise ContextStorageError(f"Failed to connect to Redis: {e}")

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized or self._redis is None:
            raise ContextStorageError(
                "Redis store not initialized. Call initialize() first."
            )

    def _context_key(self, context_id: str) -> str:
        """Generate Redis key for a context."""
        return f"{self._prefix}ctx:{context_id}"

    def _metadata_key(self, context_id: str) -> str:
        """Generate Redis key for context metadata."""
        return f"{self._prefix}meta:{context_id}"

    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session index."""
        return f"{self._prefix}session:{session_id}"

    async def store_context(
        self,
        context: ResearchContext,
        metadata: ContextMetadata | None = None,
    ) -> None:
        """
        Store a research context in Redis.

        Stores the context as a JSON string with TTL, and maintains
        a session index using a sorted set with timestamps.

        Args:
            context: The ResearchContext to store
            metadata: Optional metadata to store alongside context
        """
        self._ensure_initialized()

        context_id = context["context_id"]
        session_id = context["session_id"]
        ttl = context.get("ttl_seconds", self._default_ttl)

        try:
            # Store context as JSON
            context_key = self._context_key(context_id)
            await self._redis.setex(
                context_key,
                ttl,
                json.dumps(context),
            )

            # Store metadata if provided
            if metadata:
                metadata_key = self._metadata_key(context_id)
                await self._redis.hset(
                    metadata_key,
                    mapping={k: str(v) for k, v in metadata.items()},
                )
                await self._redis.expire(metadata_key, ttl)

            # Add to session index (sorted set with timestamp score)
            session_key = self._session_key(session_id)
            timestamp = datetime.fromisoformat(
                context["created_at"].replace("Z", "+00:00")
            ).timestamp()
            await self._redis.zadd(session_key, {context_id: timestamp})

            # Set session index TTL (extend if needed)
            current_ttl = await self._redis.ttl(session_key)
            if current_ttl < ttl:
                await self._redis.expire(session_key, ttl)

            logger.debug(f"Stored context {context_id} in Redis with TTL {ttl}s")

        except Exception as e:
            raise ContextStorageError(f"Failed to store context: {e}")

    async def get_context(self, context_id: str) -> ResearchContext | None:
        """
        Retrieve a context from Redis by ID.

        Args:
            context_id: Unique identifier of the context

        Returns:
            ResearchContext if found, None if not found or expired
        """
        self._ensure_initialized()

        try:
            context_key = self._context_key(context_id)
            data = await self._redis.get(context_key)

            if data is None:
                logger.debug(f"Context {context_id} not found in Redis")
                return None

            context: ResearchContext = json.loads(data)
            return context

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode context {context_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get context {context_id}: {e}")
            return None

    async def delete_context(self, context_id: str) -> None:
        """
        Delete a context from Redis.

        Removes the context, its metadata, and session index entry.

        Args:
            context_id: Unique identifier of the context to delete
        """
        self._ensure_initialized()

        try:
            # Get context to find session_id
            context = await self.get_context(context_id)

            # Delete context and metadata
            context_key = self._context_key(context_id)
            metadata_key = self._metadata_key(context_id)
            await self._redis.delete(context_key, metadata_key)

            # Remove from session index
            if context:
                session_key = self._session_key(context["session_id"])
                await self._redis.zrem(session_key, context_id)

            logger.debug(f"Deleted context {context_id} from Redis")

        except Exception as e:
            raise ContextStorageError(f"Failed to delete context: {e}")

    async def get_contexts_by_session(
        self,
        session_id: str,
        limit: int = 100,
        include_expired: bool = False,
    ) -> list[ResearchContext]:
        """
        Get all contexts for a session.

        Retrieves contexts from the session index, sorted by creation time.

        Args:
            session_id: Session identifier to filter by
            limit: Maximum number of contexts to return
            include_expired: Ignored for Redis (expired keys auto-removed)

        Returns:
            List of ResearchContext objects for the session
        """
        self._ensure_initialized()

        try:
            session_key = self._session_key(session_id)

            # Get context IDs from sorted set (most recent first)
            context_ids = await self._redis.zrevrange(
                session_key, 0, limit - 1
            )

            contexts: list[ResearchContext] = []
            for ctx_id in context_ids:
                context = await self.get_context(ctx_id)
                if context:
                    contexts.append(context)

            return contexts

        except Exception as e:
            logger.error(f"Failed to get session contexts: {e}")
            return []

    async def update_ttl(self, context_id: str, ttl_seconds: int) -> bool:
        """
        Update the TTL for an existing context.

        Args:
            context_id: Unique identifier of the context
            ttl_seconds: New TTL value in seconds

        Returns:
            True if updated, False if context not found
        """
        self._ensure_initialized()

        try:
            context_key = self._context_key(context_id)
            metadata_key = self._metadata_key(context_id)

            # Check if context exists
            exists = await self._redis.exists(context_key)
            if not exists:
                return False

            # Update TTL
            await self._redis.expire(context_key, ttl_seconds)
            await self._redis.expire(metadata_key, ttl_seconds)

            logger.debug(f"Updated TTL for context {context_id} to {ttl_seconds}s")
            return True

        except Exception as e:
            logger.error(f"Failed to update TTL for context {context_id}: {e}")
            return False

    async def get_metadata(self, context_id: str) -> ContextMetadata | None:
        """
        Get metadata for a context.

        Args:
            context_id: Unique identifier of the context

        Returns:
            ContextMetadata if found, None otherwise
        """
        self._ensure_initialized()

        try:
            metadata_key = self._metadata_key(context_id)
            data = await self._redis.hgetall(metadata_key)

            if not data:
                return None

            # Convert string values back to appropriate types
            metadata = ContextMetadata()
            if "source_agent" in data:
                metadata["source_agent"] = data["source_agent"]
            if "iteration_number" in data:
                metadata["iteration_number"] = int(data["iteration_number"])
            if "parent_context_id" in data:
                val = data["parent_context_id"]
                metadata["parent_context_id"] = None if val == "None" else val
            if "compression_ratio" in data:
                metadata["compression_ratio"] = float(data["compression_ratio"])
            if "original_token_count" in data:
                metadata["original_token_count"] = int(data["original_token_count"])
            if "compressed_token_count" in data:
                metadata["compressed_token_count"] = int(data["compressed_token_count"])
            if "status" in data:
                metadata["status"] = data["status"]
            if "updated_at" in data:
                metadata["updated_at"] = data["updated_at"]

            return metadata

        except Exception as e:
            logger.error(f"Failed to get metadata for context {context_id}: {e}")
            return None

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._initialized = False
            logger.info("Redis context store closed")


# =============================================================================
# PostgreSQL Context Store Implementation
# =============================================================================


class PostgresContextStore(ContextStore):
    """
    PostgreSQL-backed context store with pgvector support.

    Uses PostgreSQL for persistent context storage with vector similarity
    search capabilities via pgvector. Ideal for long-term context storage
    and semantic retrieval of related contexts.

    Features:
        - Persistent storage with ACID guarantees
        - Vector similarity search via pgvector
        - TTL-based expiration via scheduled cleanup
        - Session-based querying with indexes

    Requires:
        - PostgreSQL 12+
        - pgvector extension installed
        - asyncpg package

    Example:
        >>> store = PostgresContextStore(dsn="postgresql://user:pass@host/db")
        >>> await store.initialize()
        >>> await store.store_context(context)
        >>> contexts = await store.search_similar(embedding, limit=5)
    """

    # SQL for table creation
    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS research_contexts (
            context_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            key_entities TEXT[] NOT NULL DEFAULT '{}',
            relevance_vector vector(1536),
            chunk_refs TEXT[] NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ,
            ttl_seconds INTEGER NOT NULL DEFAULT 3600,
            status TEXT NOT NULL DEFAULT 'active',
            metadata JSONB DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_contexts_session_id
            ON research_contexts(session_id);
        CREATE INDEX IF NOT EXISTS idx_contexts_created_at
            ON research_contexts(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_contexts_status
            ON research_contexts(status);
        CREATE INDEX IF NOT EXISTS idx_contexts_expires_at
            ON research_contexts(expires_at);
    """

    # SQL for vector index (requires pgvector)
    CREATE_VECTOR_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_contexts_embedding
            ON research_contexts
            USING ivfflat (relevance_vector vector_cosine_ops)
            WITH (lists = 100);
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost/drx",
        pool_size: int = 10,
        vector_dimensions: int = 1536,
    ) -> None:
        """
        Initialize PostgreSQL context store.

        Args:
            dsn: PostgreSQL connection string
            pool_size: Maximum pool connections
            vector_dimensions: Dimensions of embedding vectors
        """
        self._dsn = dsn
        self._pool_size = pool_size
        self._vector_dimensions = vector_dimensions
        self._pool: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize PostgreSQL connection pool and create tables.

        Must be called before using the store.

        Raises:
            ContextStorageError: If connection or table creation fails
        """
        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=self._pool_size,
            )

            # Create tables if not exist
            async with self._pool.acquire() as conn:
                await conn.execute(self.CREATE_TABLE_SQL)

                # Try to create vector index (may fail if pgvector not installed)
                try:
                    await conn.execute(self.CREATE_VECTOR_INDEX_SQL)
                except Exception as ve:
                    logger.warning(
                        f"Could not create vector index (pgvector may not be installed): {ve}"
                    )

            self._initialized = True
            logger.info(f"PostgreSQL context store initialized: {self._dsn}")

        except ImportError:
            raise ContextStorageError(
                "asyncpg package not installed. Install with: pip install asyncpg"
            )
        except Exception as e:
            raise ContextStorageError(f"Failed to connect to PostgreSQL: {e}")

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized or self._pool is None:
            raise ContextStorageError(
                "PostgreSQL store not initialized. Call initialize() first."
            )

    async def store_context(
        self,
        context: ResearchContext,
        metadata: ContextMetadata | None = None,
    ) -> None:
        """
        Store a research context in PostgreSQL.

        Args:
            context: The ResearchContext to store
            metadata: Optional metadata to store alongside context
        """
        self._ensure_initialized()

        try:
            # Calculate expiration time
            created_at = datetime.fromisoformat(
                context["created_at"].replace("Z", "+00:00")
            )
            ttl_seconds = context.get("ttl_seconds", 3600)
            from datetime import timedelta
            expires_at = created_at + timedelta(seconds=ttl_seconds)

            # Convert vector to PostgreSQL format
            vector = context.get("relevance_vector", [])
            vector_str = f"[{','.join(str(v) for v in vector)}]" if vector else None

            query = """
                INSERT INTO research_contexts (
                    context_id, session_id, summary, key_entities,
                    relevance_vector, chunk_refs, created_at, expires_at,
                    ttl_seconds, status, metadata
                ) VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (context_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    key_entities = EXCLUDED.key_entities,
                    relevance_vector = EXCLUDED.relevance_vector,
                    chunk_refs = EXCLUDED.chunk_refs,
                    expires_at = EXCLUDED.expires_at,
                    ttl_seconds = EXCLUDED.ttl_seconds,
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata
            """

            async with self._pool.acquire() as conn:
                await conn.execute(
                    query,
                    context["context_id"],
                    context["session_id"],
                    context["summary"],
                    context.get("key_entities", []),
                    vector_str,
                    context.get("chunk_refs", []),
                    created_at,
                    expires_at,
                    ttl_seconds,
                    "active",
                    json.dumps(metadata or {}),
                )

            logger.debug(f"Stored context {context['context_id']} in PostgreSQL")

        except Exception as e:
            raise ContextStorageError(f"Failed to store context: {e}")

    async def get_context(self, context_id: str) -> ResearchContext | None:
        """
        Retrieve a context from PostgreSQL by ID.

        Args:
            context_id: Unique identifier of the context

        Returns:
            ResearchContext if found and not expired, None otherwise
        """
        self._ensure_initialized()

        try:
            query = """
                SELECT
                    context_id, session_id, summary, key_entities,
                    relevance_vector::text, chunk_refs, created_at, ttl_seconds
                FROM research_contexts
                WHERE context_id = $1
                  AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > NOW())
            """

            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, context_id)

            if row is None:
                logger.debug(f"Context {context_id} not found in PostgreSQL")
                return None

            # Parse vector from PostgreSQL format
            vector_str = row["relevance_vector"]
            if vector_str:
                # Format: [0.1,0.2,0.3,...]
                vector_str = vector_str.strip("[]")
                relevance_vector = [float(v) for v in vector_str.split(",")] if vector_str else []
            else:
                relevance_vector = []

            context: ResearchContext = {
                "context_id": row["context_id"],
                "session_id": row["session_id"],
                "summary": row["summary"],
                "key_entities": list(row["key_entities"]) if row["key_entities"] else [],
                "relevance_vector": relevance_vector,
                "chunk_refs": list(row["chunk_refs"]) if row["chunk_refs"] else [],
                "created_at": row["created_at"].isoformat() + "Z",
                "ttl_seconds": row["ttl_seconds"],
            }

            return context

        except Exception as e:
            logger.error(f"Failed to get context {context_id}: {e}")
            return None

    async def delete_context(self, context_id: str) -> None:
        """
        Delete a context from PostgreSQL (soft delete).

        Args:
            context_id: Unique identifier of the context to delete
        """
        self._ensure_initialized()

        try:
            query = """
                UPDATE research_contexts
                SET status = 'deleted'
                WHERE context_id = $1
            """

            async with self._pool.acquire() as conn:
                await conn.execute(query, context_id)

            logger.debug(f"Deleted context {context_id} from PostgreSQL")

        except Exception as e:
            raise ContextStorageError(f"Failed to delete context: {e}")

    async def get_contexts_by_session(
        self,
        session_id: str,
        limit: int = 100,
        include_expired: bool = False,
    ) -> list[ResearchContext]:
        """
        Get all contexts for a session.

        Args:
            session_id: Session identifier to filter by
            limit: Maximum number of contexts to return
            include_expired: Whether to include expired contexts

        Returns:
            List of ResearchContext objects for the session
        """
        self._ensure_initialized()

        try:
            if include_expired:
                query = """
                    SELECT
                        context_id, session_id, summary, key_entities,
                        relevance_vector::text, chunk_refs, created_at, ttl_seconds
                    FROM research_contexts
                    WHERE session_id = $1 AND status != 'deleted'
                    ORDER BY created_at DESC
                    LIMIT $2
                """
            else:
                query = """
                    SELECT
                        context_id, session_id, summary, key_entities,
                        relevance_vector::text, chunk_refs, created_at, ttl_seconds
                    FROM research_contexts
                    WHERE session_id = $1
                      AND status = 'active'
                      AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY created_at DESC
                    LIMIT $2
                """

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, session_id, limit)

            contexts: list[ResearchContext] = []
            for row in rows:
                # Parse vector
                vector_str = row["relevance_vector"]
                if vector_str:
                    vector_str = vector_str.strip("[]")
                    relevance_vector = [float(v) for v in vector_str.split(",")] if vector_str else []
                else:
                    relevance_vector = []

                context: ResearchContext = {
                    "context_id": row["context_id"],
                    "session_id": row["session_id"],
                    "summary": row["summary"],
                    "key_entities": list(row["key_entities"]) if row["key_entities"] else [],
                    "relevance_vector": relevance_vector,
                    "chunk_refs": list(row["chunk_refs"]) if row["chunk_refs"] else [],
                    "created_at": row["created_at"].isoformat() + "Z",
                    "ttl_seconds": row["ttl_seconds"],
                }
                contexts.append(context)

            return contexts

        except Exception as e:
            logger.error(f"Failed to get session contexts: {e}")
            return []

    async def update_ttl(self, context_id: str, ttl_seconds: int) -> bool:
        """
        Update the TTL for an existing context.

        Args:
            context_id: Unique identifier of the context
            ttl_seconds: New TTL value in seconds

        Returns:
            True if updated, False if context not found
        """
        self._ensure_initialized()

        try:
            query = """
                UPDATE research_contexts
                SET
                    ttl_seconds = $2,
                    expires_at = created_at + ($2 || ' seconds')::interval
                WHERE context_id = $1 AND status = 'active'
                RETURNING context_id
            """

            async with self._pool.acquire() as conn:
                result = await conn.fetchval(query, context_id, ttl_seconds)

            if result:
                logger.debug(f"Updated TTL for context {context_id} to {ttl_seconds}s")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to update TTL for context {context_id}: {e}")
            return False

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        session_id: str | None = None,
        threshold: float = 0.7,
    ) -> list[tuple[ResearchContext, float]]:
        """
        Search for similar contexts using vector similarity.

        Uses cosine similarity via pgvector for semantic matching.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            session_id: Optional session filter
            threshold: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of (context, similarity_score) tuples
        """
        self._ensure_initialized()

        try:
            vector_str = f"[{','.join(str(v) for v in embedding)}]"

            if session_id:
                query = """
                    SELECT
                        context_id, session_id, summary, key_entities,
                        relevance_vector::text, chunk_refs, created_at, ttl_seconds,
                        1 - (relevance_vector <=> $1::vector) as similarity
                    FROM research_contexts
                    WHERE session_id = $2
                      AND status = 'active'
                      AND (expires_at IS NULL OR expires_at > NOW())
                      AND relevance_vector IS NOT NULL
                      AND 1 - (relevance_vector <=> $1::vector) >= $3
                    ORDER BY relevance_vector <=> $1::vector
                    LIMIT $4
                """
                params = (vector_str, session_id, threshold, limit)
            else:
                query = """
                    SELECT
                        context_id, session_id, summary, key_entities,
                        relevance_vector::text, chunk_refs, created_at, ttl_seconds,
                        1 - (relevance_vector <=> $1::vector) as similarity
                    FROM research_contexts
                    WHERE status = 'active'
                      AND (expires_at IS NULL OR expires_at > NOW())
                      AND relevance_vector IS NOT NULL
                      AND 1 - (relevance_vector <=> $1::vector) >= $2
                    ORDER BY relevance_vector <=> $1::vector
                    LIMIT $3
                """
                params = (vector_str, threshold, limit)

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            results: list[tuple[ResearchContext, float]] = []
            for row in rows:
                # Parse vector
                vec_str = row["relevance_vector"]
                if vec_str:
                    vec_str = vec_str.strip("[]")
                    relevance_vector = [float(v) for v in vec_str.split(",")] if vec_str else []
                else:
                    relevance_vector = []

                context: ResearchContext = {
                    "context_id": row["context_id"],
                    "session_id": row["session_id"],
                    "summary": row["summary"],
                    "key_entities": list(row["key_entities"]) if row["key_entities"] else [],
                    "relevance_vector": relevance_vector,
                    "chunk_refs": list(row["chunk_refs"]) if row["chunk_refs"] else [],
                    "created_at": row["created_at"].isoformat() + "Z",
                    "ttl_seconds": row["ttl_seconds"],
                }
                results.append((context, float(row["similarity"])))

            return results

        except Exception as e:
            logger.error(f"Failed to search similar contexts: {e}")
            return []

    async def cleanup_expired(self) -> int:
        """
        Clean up expired contexts.

        Marks expired contexts as 'expired' status.

        Returns:
            Number of contexts cleaned up
        """
        self._ensure_initialized()

        try:
            query = """
                UPDATE research_contexts
                SET status = 'expired'
                WHERE status = 'active'
                  AND expires_at IS NOT NULL
                  AND expires_at <= NOW()
                RETURNING context_id
            """

            async with self._pool.acquire() as conn:
                result = await conn.fetch(query)

            count = len(result)
            if count > 0:
                logger.info(f"Cleaned up {count} expired contexts")
            return count

        except Exception as e:
            logger.error(f"Failed to cleanup expired contexts: {e}")
            return 0

    async def close(self) -> None:
        """Close PostgreSQL connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("PostgreSQL context store closed")


# =============================================================================
# Hybrid Context Store (Redis Cache + PostgreSQL Persistence)
# =============================================================================


class HybridContextStore(ContextStore):
    """
    Hybrid context store combining Redis caching with PostgreSQL persistence.

    Uses Redis for fast access to active contexts (read-through cache)
    and PostgreSQL for durable storage and vector search.

    Write operations go to both stores. Read operations check Redis first,
    falling back to PostgreSQL on cache miss.

    Example:
        >>> store = HybridContextStore(
        ...     redis_url="redis://localhost:6379",
        ...     postgres_dsn="postgresql://localhost/drx"
        ... )
        >>> await store.initialize()
        >>> await store.store_context(context)  # Writes to both
        >>> ctx = await store.get_context(id)   # Reads from Redis, fallback to PG
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        postgres_dsn: str = "postgresql://localhost/drx",
        cache_ttl_seconds: int = 300,
    ) -> None:
        """
        Initialize hybrid context store.

        Args:
            redis_url: Redis connection URL
            postgres_dsn: PostgreSQL connection string
            cache_ttl_seconds: TTL for Redis cache entries
        """
        self._redis_store = RedisContextStore(redis_url=redis_url)
        self._postgres_store = PostgresContextStore(dsn=postgres_dsn)
        self._cache_ttl = cache_ttl_seconds
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize both Redis and PostgreSQL stores."""
        await self._redis_store.initialize()
        await self._postgres_store.initialize()
        self._initialized = True
        logger.info("Hybrid context store initialized")

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized:
            raise ContextStorageError(
                "Hybrid store not initialized. Call initialize() first."
            )

    async def store_context(
        self,
        context: ResearchContext,
        metadata: ContextMetadata | None = None,
    ) -> None:
        """
        Store context in both Redis and PostgreSQL.

        Args:
            context: The ResearchContext to store
            metadata: Optional metadata to store alongside context
        """
        self._ensure_initialized()

        # Store in PostgreSQL first (primary store)
        await self._postgres_store.store_context(context, metadata)

        # Then cache in Redis
        try:
            await self._redis_store.store_context(context, metadata)
        except Exception as e:
            # Log but don't fail if Redis cache write fails
            logger.warning(f"Failed to cache context in Redis: {e}")

    async def get_context(self, context_id: str) -> ResearchContext | None:
        """
        Get context from Redis cache, fallback to PostgreSQL.

        Args:
            context_id: Unique identifier of the context

        Returns:
            ResearchContext if found, None otherwise
        """
        self._ensure_initialized()

        # Try Redis first
        try:
            context = await self._redis_store.get_context(context_id)
            if context:
                return context
        except Exception as e:
            logger.warning(f"Redis cache miss for {context_id}: {e}")

        # Fallback to PostgreSQL
        context = await self._postgres_store.get_context(context_id)

        # Re-cache in Redis if found
        if context:
            try:
                await self._redis_store.store_context(context)
            except Exception as e:
                logger.warning(f"Failed to re-cache context in Redis: {e}")

        return context

    async def delete_context(self, context_id: str) -> None:
        """
        Delete context from both stores.

        Args:
            context_id: Unique identifier of the context to delete
        """
        self._ensure_initialized()

        # Delete from both stores
        try:
            await self._redis_store.delete_context(context_id)
        except Exception as e:
            logger.warning(f"Failed to delete from Redis: {e}")

        await self._postgres_store.delete_context(context_id)

    async def get_contexts_by_session(
        self,
        session_id: str,
        limit: int = 100,
        include_expired: bool = False,
    ) -> list[ResearchContext]:
        """
        Get contexts from PostgreSQL (authoritative source).

        Args:
            session_id: Session identifier to filter by
            limit: Maximum number of contexts to return
            include_expired: Whether to include expired contexts

        Returns:
            List of ResearchContext objects for the session
        """
        self._ensure_initialized()

        # Use PostgreSQL as authoritative source for session queries
        return await self._postgres_store.get_contexts_by_session(
            session_id, limit, include_expired
        )

    async def update_ttl(self, context_id: str, ttl_seconds: int) -> bool:
        """
        Update TTL in both stores.

        Args:
            context_id: Unique identifier of the context
            ttl_seconds: New TTL value in seconds

        Returns:
            True if updated in PostgreSQL, False if not found
        """
        self._ensure_initialized()

        # Update in both stores
        try:
            await self._redis_store.update_ttl(context_id, ttl_seconds)
        except Exception as e:
            logger.warning(f"Failed to update TTL in Redis: {e}")

        return await self._postgres_store.update_ttl(context_id, ttl_seconds)

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        session_id: str | None = None,
        threshold: float = 0.7,
    ) -> list[tuple[ResearchContext, float]]:
        """
        Search for similar contexts using PostgreSQL pgvector.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            session_id: Optional session filter
            threshold: Minimum similarity threshold

        Returns:
            List of (context, similarity_score) tuples
        """
        self._ensure_initialized()

        return await self._postgres_store.search_similar(
            embedding, limit, session_id, threshold
        )

    async def close(self) -> None:
        """Close both stores."""
        await self._redis_store.close()
        await self._postgres_store.close()
        self._initialized = False
        logger.info("Hybrid context store closed")


# =============================================================================
# Factory Function
# =============================================================================


async def create_context_store(
    store_type: ContextStoreType = "hybrid",
    config: ContextStoreConfig | None = None,
) -> ContextStore:
    """
    Factory function to create and initialize a context store.

    Creates the appropriate store implementation based on store_type
    and initializes it with the provided configuration.

    Args:
        store_type: Type of store to create ("redis", "postgres", or "hybrid")
        config: Optional configuration dictionary

    Returns:
        Initialized ContextStore instance

    Raises:
        ValueError: If store_type is invalid
        ContextStorageError: If initialization fails

    Example:
        >>> store = await create_context_store("hybrid", {
        ...     "redis_url": "redis://localhost:6379",
        ...     "postgres_dsn": "postgresql://localhost/drx"
        ... })
        >>> await store.store_context(context)
    """
    config = config or {}

    if store_type == "redis":
        store = RedisContextStore(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            prefix=config.get("redis_prefix", "drx:context:"),
            default_ttl=config.get("redis_default_ttl", 3600),
        )
    elif store_type == "postgres":
        store = PostgresContextStore(
            dsn=config.get("postgres_dsn", "postgresql://localhost/drx"),
            pool_size=config.get("postgres_pool_size", 10),
        )
    elif store_type == "hybrid":
        store = HybridContextStore(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            postgres_dsn=config.get("postgres_dsn", "postgresql://localhost/drx"),
            cache_ttl_seconds=config.get("cache_ttl_seconds", 300),
        )
    else:
        raise ValueError(
            f"Invalid store_type: {store_type}. "
            f"Valid types: redis, postgres, hybrid"
        )

    await store.initialize()
    return store


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    # Type definitions
    "ContextStoreType",
    "ContextStatus",
    # TypedDicts
    "ResearchContext",
    "ContextMetadata",
    "ContextStoreConfig",
    # Abstract base class
    "ContextStore",
    # Exceptions
    "ContextStoreError",
    "ContextNotFoundError",
    "ContextStorageError",
    # Implementations
    "RedisContextStore",
    "PostgresContextStore",
    "HybridContextStore",
    # Factory function
    "create_context_store",
]
