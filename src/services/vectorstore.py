"""
Vector Store Service for DRX Deep Research System.

Provides pgvector-backed semantic search for internal documents and knowledge.
Supports hybrid search combining semantic similarity with keyword matching.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, TypedDict

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.config import get_settings
from src.db.connection import get_async_pool

logger = logging.getLogger(__name__)
settings = get_settings()


# =============================================================================
# Type Definitions
# =============================================================================


class Document(TypedDict):
    """A document to be stored in the vector store."""

    id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None  # Generated if None


class SearchResult(TypedDict):
    """A search result from the vector store."""

    id: str
    content: str
    metadata: dict[str, Any]
    score: float  # Similarity score (0-1, higher is more similar)
    distance: float  # Raw distance (lower is more similar)


class CollectionStats(TypedDict):
    """Statistics about a collection."""

    name: str
    document_count: int
    dimension: int
    created_at: str


# =============================================================================
# Embedding Provider
# =============================================================================


class EmbeddingProvider:
    """
    Provider for generating embeddings via OpenRouter API.

    Uses the embeddings endpoint to convert text to vectors.
    """

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        dimension: int = 1536,
    ):
        """
        Initialize the embedding provider.

        Args:
            model: Embedding model identifier
            dimension: Expected embedding dimension
        """
        self._model = model
        self._dimension = dimension
        self._client: Any = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    async def _get_client(self) -> Any:
        """Get or create HTTP client."""
        if self._client is None:
            import httpx

            settings = get_settings()
            self._client = httpx.AsyncClient(
                base_url=settings.OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key_value}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = await self._get_client()

        # Clean and truncate texts (max 8000 chars for most models)
        cleaned_texts = [t.strip()[:8000] for t in texts]

        try:
            response = await client.post(
                "/embeddings",
                json={
                    "model": self._model,
                    "input": cleaned_texts,
                },
            )
            response.raise_for_status()

            data = response.json()
            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self._dimension for _ in texts]

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * self._dimension

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Vector Store Implementation
# =============================================================================


class VectorStore:
    """
    pgvector-backed vector store for semantic search.

    Features:
    - Multiple collections with isolation
    - Automatic embedding generation via OpenRouter
    - Hybrid search (semantic + keyword)
    - Metadata filtering
    - Batch operations

    Example:
        ```python
        store = await create_vectorstore()
        await store.create_collection("research_docs", dimension=1536)

        docs = [Document(id="1", content="...", metadata={}, embedding=None)]
        await store.ingest(docs, collection="research_docs")

        results = await store.search("quantum computing", k=10)
        ```
    """

    DEFAULT_COLLECTION = "default"
    DEFAULT_DIMENSION = 1536  # OpenAI ada-002 / text-embedding-3-small

    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """
        Initialize the vector store.

        Args:
            pool: PostgreSQL connection pool
            embedding_provider: Provider for generating embeddings
        """
        self._pool = pool
        self._embedding_provider = embedding_provider or EmbeddingProvider()

    async def create_collection(
        self,
        name: str,
        dimension: int = DEFAULT_DIMENSION,
        if_not_exists: bool = True,
    ) -> None:
        """
        Create a new vector collection.

        Args:
            name: Collection name (alphanumeric and underscores only)
            dimension: Embedding dimension
            if_not_exists: Don't error if exists
        """
        # Sanitize collection name to prevent SQL injection
        safe_name = "".join(c for c in name if c.isalnum() or c == "_")
        if safe_name != name:
            logger.warning(f"Collection name sanitized from '{name}' to '{safe_name}'")
            name = safe_name

        async with self._pool.connection() as conn:
            # Ensure pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create collection table
            exists_clause = "IF NOT EXISTS" if if_not_exists else ""

            # Use SQL composition for safe table creation
            await conn.execute(f"""
                CREATE TABLE {exists_clause} vectorstore_{name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    embedding vector({dimension}),
                    content_hash TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes - these may fail if table already exists with indexes
            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{name}_embedding
                    ON vectorstore_{name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            except Exception as e:
                logger.debug(f"Index creation note (may already exist): {e}")

            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{name}_metadata
                    ON vectorstore_{name}
                    USING gin (metadata)
                """)
            except Exception as e:
                logger.debug(f"Index creation note (may already exist): {e}")

            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{name}_content_search
                    ON vectorstore_{name}
                    USING gin (to_tsvector('english', content))
                """)
            except Exception as e:
                logger.debug(f"Index creation note (may already exist): {e}")

            await conn.commit()
            logger.info(f"Created collection '{name}' with dimension {dimension}")

    async def drop_collection(self, name: str) -> None:
        """Drop a collection and all its data."""
        safe_name = "".join(c for c in name if c.isalnum() or c == "_")
        async with self._pool.connection() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS vectorstore_{safe_name}")
            await conn.commit()
            logger.info(f"Dropped collection '{safe_name}'")

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        safe_name = "".join(c for c in name if c.isalnum() or c == "_")
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                )
            """,
                (f"vectorstore_{safe_name}",),
            )
            row = await result.fetchone()
            return bool(row[0]) if row else False

    async def get_collection_stats(self, name: str) -> CollectionStats | None:
        """Get statistics about a collection."""
        safe_name = "".join(c for c in name if c.isalnum() or c == "_")

        if not await self.collection_exists(safe_name):
            return None

        async with self._pool.connection() as conn:
            # Get count
            result = await conn.execute(f"SELECT COUNT(*) FROM vectorstore_{safe_name}")
            row = await result.fetchone()
            count = int(row[0]) if row else 0

            # Get dimension from first embedding
            result = await conn.execute(f"""
                SELECT vector_dims(embedding)
                FROM vectorstore_{safe_name}
                WHERE embedding IS NOT NULL
                LIMIT 1
            """)
            row = await result.fetchone()
            dimension = int(row[0]) if row else self.DEFAULT_DIMENSION

            return CollectionStats(
                name=safe_name,
                document_count=count,
                dimension=dimension,
                created_at=datetime.utcnow().isoformat() + "Z",
            )

    async def ingest(
        self,
        documents: list[Document],
        collection: str = DEFAULT_COLLECTION,
        generate_embeddings: bool = True,
    ) -> int:
        """
        Ingest documents into the collection.

        Args:
            documents: Documents to ingest
            collection: Target collection
            generate_embeddings: Generate embeddings for docs without them

        Returns:
            Number of documents ingested
        """
        if not documents:
            return 0

        safe_collection = "".join(c for c in collection if c.isalnum() or c == "_")

        # Ensure collection exists
        if not await self.collection_exists(safe_collection):
            await self.create_collection(safe_collection)

        # Generate embeddings for documents that need them
        if generate_embeddings:
            docs_needing_embeddings = [d for d in documents if d.get("embedding") is None]
            if docs_needing_embeddings:
                embeddings = await self._embedding_provider.generate_embeddings(
                    [d["content"] for d in docs_needing_embeddings]
                )
                for doc, emb in zip(docs_needing_embeddings, embeddings):
                    doc["embedding"] = emb

        # Insert documents
        inserted = 0
        async with self._pool.connection() as conn:
            for doc in documents:
                content_hash = hashlib.md5(doc["content"].encode()).hexdigest()

                # Convert embedding to string format for pgvector
                embedding = doc.get("embedding")
                embedding_str = str(embedding) if embedding else None

                await conn.execute(
                    f"""
                    INSERT INTO vectorstore_{safe_collection}
                    (id, content, metadata, embedding, content_hash)
                    VALUES (%s, %s, %s, %s::vector, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        content_hash = EXCLUDED.content_hash,
                        updated_at = NOW()
                """,
                    (
                        doc["id"],
                        doc["content"],
                        json.dumps(doc.get("metadata", {})),
                        embedding_str,
                        content_hash,
                    ),
                )
                inserted += 1

            await conn.commit()

        logger.info(f"Ingested {inserted} documents into '{safe_collection}'")
        return inserted

    async def search(
        self,
        query: str,
        collection: str = DEFAULT_COLLECTION,
        k: int = 10,
        filters: dict[str, Any] | None = None,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Semantic search for similar documents.

        Args:
            query: Search query
            collection: Collection to search
            k: Number of results
            filters: Metadata filters (JSONB containment)
            threshold: Optional similarity threshold (0-1)

        Returns:
            List of search results sorted by similarity
        """
        safe_collection = "".join(c for c in collection if c.isalnum() or c == "_")

        if not await self.collection_exists(safe_collection):
            return []

        # Generate query embedding
        query_embedding = await self._embedding_provider.generate_embedding(query)
        query_embedding_str = str(query_embedding)

        # Build query with parameterized values
        params: list[Any] = [query_embedding_str, query_embedding_str]
        filter_clause = ""

        if filters:
            filter_clause = "AND metadata @> %s::jsonb"
            params.append(json.dumps(filters))

        if threshold is not None:
            # Convert threshold to distance (cosine distance = 1 - similarity)
            max_distance = 1 - threshold
            filter_clause += f" AND (embedding <=> %s::vector) < {max_distance}"
            params.append(query_embedding_str)

        params.extend([query_embedding_str, k])

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            result = await conn.execute(
                f"""
                SELECT
                    id,
                    content,
                    metadata,
                    (embedding <=> %s::vector) as distance,
                    1 - (embedding <=> %s::vector) as score
                FROM vectorstore_{safe_collection}
                WHERE embedding IS NOT NULL
                {filter_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """,
                params,
            )

            rows = await result.fetchall()

        return [
            SearchResult(
                id=row["id"],
                content=row["content"],
                metadata=row["metadata"] or {},
                score=float(row["score"]),
                distance=float(row["distance"]),
            )
            for row in rows
        ]

    async def hybrid_search(
        self,
        query: str,
        collection: str = DEFAULT_COLLECTION,
        k: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search combining semantic and keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to combine semantic similarity
        with full-text keyword search results.

        Args:
            query: Search query
            collection: Collection to search
            k: Number of results
            alpha: Weight for semantic search (0=keyword only, 1=semantic only)
            filters: Metadata filters

        Returns:
            Combined and re-ranked results
        """
        safe_collection = "".join(c for c in collection if c.isalnum() or c == "_")

        if not await self.collection_exists(safe_collection):
            return []

        # Get semantic results
        semantic_results = await self.search(query, collection, k=k * 2, filters=filters)

        # Get keyword results
        keyword_results = await self._keyword_search(query, collection, k=k * 2, filters=filters)

        # Combine scores using RRF (Reciprocal Rank Fusion)
        result_scores: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(semantic_results, 1):
            doc_id = result["id"]
            if doc_id not in result_scores:
                result_scores[doc_id] = {
                    "result": result,
                    "semantic_rank": rank,
                    "keyword_rank": None,
                }
            else:
                result_scores[doc_id]["semantic_rank"] = rank

        for rank, result in enumerate(keyword_results, 1):
            doc_id = result["id"]
            if doc_id not in result_scores:
                result_scores[doc_id] = {
                    "result": result,
                    "semantic_rank": None,
                    "keyword_rank": rank,
                }
            else:
                result_scores[doc_id]["keyword_rank"] = rank

        # Calculate RRF scores
        k_constant = 60  # RRF constant
        for doc_id, data in result_scores.items():
            semantic_score = 0.0
            keyword_score = 0.0

            if data["semantic_rank"]:
                semantic_score = 1 / (k_constant + data["semantic_rank"])
            if data["keyword_rank"]:
                keyword_score = 1 / (k_constant + data["keyword_rank"])

            # Weighted combination
            data["rrf_score"] = alpha * semantic_score + (1 - alpha) * keyword_score

        # Sort by RRF score and return top k
        sorted_results = sorted(
            result_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:k]

        return [
            SearchResult(
                id=item["result"]["id"],
                content=item["result"]["content"],
                metadata=item["result"]["metadata"],
                score=float(item["rrf_score"]),
                distance=item["result"].get("distance", 0.0),
            )
            for item in sorted_results
        ]

    async def _keyword_search(
        self,
        query: str,
        collection: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Full-text keyword search using PostgreSQL tsvector."""
        safe_collection = "".join(c for c in collection if c.isalnum() or c == "_")

        params: list[Any] = [query, query]
        filter_clause = ""

        if filters:
            filter_clause = "AND metadata @> %s::jsonb"
            params.append(json.dumps(filters))

        params.append(k)

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            result = await conn.execute(
                f"""
                SELECT
                    id,
                    content,
                    metadata,
                    ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as score
                FROM vectorstore_{safe_collection}
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                {filter_clause}
                ORDER BY score DESC
                LIMIT %s
            """,
                params,
            )

            rows = await result.fetchall()

        return [
            SearchResult(
                id=row["id"],
                content=row["content"],
                metadata=row["metadata"] or {},
                score=float(row["score"]),
                distance=0.0,
            )
            for row in rows
        ]

    async def delete(
        self,
        doc_ids: list[str],
        collection: str = DEFAULT_COLLECTION,
    ) -> int:
        """
        Delete documents by ID.

        Args:
            doc_ids: Document IDs to delete
            collection: Collection to delete from

        Returns:
            Number of documents deleted
        """
        safe_collection = "".join(c for c in collection if c.isalnum() or c == "_")

        if not doc_ids or not await self.collection_exists(safe_collection):
            return 0

        async with self._pool.connection() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM vectorstore_{safe_collection}
                WHERE id = ANY(%s)
            """,
                (doc_ids,),
            )

            await conn.commit()
            deleted = result.rowcount or 0

        logger.info(f"Deleted {deleted} documents from '{safe_collection}'")
        return deleted

    async def get_document(
        self,
        doc_id: str,
        collection: str = DEFAULT_COLLECTION,
    ) -> Document | None:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID
            collection: Collection to search

        Returns:
            Document if found, None otherwise
        """
        safe_collection = "".join(c for c in collection if c.isalnum() or c == "_")

        if not await self.collection_exists(safe_collection):
            return None

        async with self._pool.connection() as conn:
            conn.row_factory = dict_row
            result = await conn.execute(
                f"""
                SELECT id, content, metadata, embedding
                FROM vectorstore_{safe_collection}
                WHERE id = %s
            """,
                (doc_id,),
            )
            row = await result.fetchone()

        if not row:
            return None

        return Document(
            id=row["id"],
            content=row["content"],
            metadata=row["metadata"] or {},
            embedding=list(row["embedding"]) if row["embedding"] else None,
        )

    async def list_collections(self) -> list[str]:
        """
        List all vector store collections.

        Returns:
            List of collection names
        """
        async with self._pool.connection() as conn:
            result = await conn.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE 'vectorstore_%'
            """)
            rows = await result.fetchall()

        return [row[0].replace("vectorstore_", "") for row in rows]

    async def close(self) -> None:
        """Close the embedding provider client."""
        await self._embedding_provider.close()


# =============================================================================
# Factory Function
# =============================================================================


async def create_vectorstore(
    embedding_model: str = "openai/text-embedding-3-small",
) -> VectorStore:
    """
    Factory function to create a VectorStore.

    Args:
        embedding_model: Model for generating embeddings

    Returns:
        Configured VectorStore
    """
    pool = await get_async_pool()
    embedding_provider = EmbeddingProvider(model=embedding_model)
    return VectorStore(pool, embedding_provider)


# Alias for convenience
async def get_vectorstore(
    embedding_model: str = "openai/text-embedding-3-small",
) -> VectorStore:
    """Alias for create_vectorstore."""
    return await create_vectorstore(embedding_model)


__all__ = [
    "VectorStore",
    "Document",
    "SearchResult",
    "CollectionStats",
    "EmbeddingProvider",
    "create_vectorstore",
    "get_vectorstore",
]
