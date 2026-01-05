"""
RAG Retriever Tool for DRX Deep Research system.

Provides vector similarity search using pgvector for document retrieval.
Supports:
- Document embedding and storage
- Cosine similarity search
- Session-scoped document collections
- Multiple embedding providers (OpenRouter, local)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import httpx
import numpy as np

from .base import (
    BaseTool,
    ToolResult,
    RateLimiter,
    tool_with_retry,
)

logger = logging.getLogger(__name__)

# Default embedding model dimensions
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "voyage-large-2": 1536,
    "voyage-code-2": 1536,
    "nomic-embed-text": 768,
}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536


@dataclass
class Document:
    """
    Document for RAG storage and retrieval.

    Attributes:
        id: Unique document identifier
        content: Document text content
        embedding: Vector embedding (list of floats)
        metadata: Additional document metadata
        source_url: Original source URL if applicable
    """
    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_url: str | None = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            # Generate deterministic ID from content
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            self.id = f"doc_{content_hash}"

        # Add timestamp if not present
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "source_url": self.source_url
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            source_url=data.get("source_url")
        )

    @classmethod
    def from_search_result(cls, search_result: Any) -> Document:
        """Create Document from SearchResult."""
        from .base import SearchResult
        if isinstance(search_result, SearchResult):
            return cls(
                id="",  # Will be auto-generated
                content=f"{search_result.title}\n\n{search_result.snippet}",
                source_url=search_result.url,
                metadata={
                    "title": search_result.title,
                    "score": search_result.score,
                    **search_result.metadata
                }
            )
        raise ValueError(f"Cannot convert {type(search_result)} to Document")


@dataclass
class RetrievalResult:
    """Result from RAG retrieval with similarity score."""
    document: Document
    similarity_score: float
    rank: int


class EmbeddingProvider:
    """
    Base class for embedding providers.

    Supports OpenRouter, OpenAI, and local embedding models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str | None = None
    ):
        """Initialize embedding provider."""
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client: httpx.AsyncClient | None = None
        self._dimension = EMBEDDING_DIMENSIONS.get(model, DEFAULT_EMBEDDING_DIM)

    @property
    def dimension(self) -> int:
        """Get embedding dimension for current model."""
        return self._dimension

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=60.0
            )
        return self._client

    @tool_with_retry(max_retries=3, base_delay=1.0)
    async def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        raise NotImplementedError

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        # Default implementation: process sequentially
        # Subclasses can override for batch processing
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    async def close(self):
        """Clean up resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using OpenRouter API.

    Routes to various embedding models through OpenRouter.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/text-embedding-3-small"
    ):
        """Initialize OpenRouter embedding provider."""
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1"
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding via OpenRouter."""
        client = await self._get_client()

        # Clean and truncate text
        text = text.strip()[:8000]

        response = await client.post(
            f"{self._base_url}/embeddings",
            json={
                "model": self._model,
                "input": text
            }
        )
        response.raise_for_status()

        data = response.json()
        return data["data"][0]["embedding"]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings in batch."""
        client = await self._get_client()

        # Clean and truncate texts
        cleaned_texts = [t.strip()[:8000] for t in texts]

        # OpenRouter supports batch embeddings
        response = await client.post(
            f"{self._base_url}/embeddings",
            json={
                "model": self._model,
                "input": cleaned_texts
            }
        )
        response.raise_for_status()

        data = response.json()
        # Sort by index to maintain order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using OpenAI API directly.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small"
    ):
        """Initialize OpenAI embedding provider."""
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.openai.com/v1"
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding via OpenAI."""
        client = await self._get_client()

        text = text.strip()[:8000]

        response = await client.post(
            f"{self._base_url}/embeddings",
            json={
                "model": self._model,
                "input": text
            }
        )
        response.raise_for_status()

        data = response.json()
        return data["data"][0]["embedding"]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings in batch."""
        client = await self._get_client()

        cleaned_texts = [t.strip()[:8000] for t in texts]

        response = await client.post(
            f"{self._base_url}/embeddings",
            json={
                "model": self._model,
                "input": cleaned_texts
            }
        )
        response.raise_for_status()

        data = response.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


class RAGRetriever(BaseTool):
    """
    RAG Retriever using pgvector for similarity search.

    Features:
    - Document embedding and storage
    - Cosine similarity search
    - Session-scoped document collections
    - Support for multiple embedding providers
    - In-memory fallback when database is unavailable
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        connection_string: str | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 30.0,
        use_memory_fallback: bool = True
    ):
        """
        Initialize RAG Retriever.

        Args:
            embedding_provider: Provider for text embeddings
            connection_string: PostgreSQL connection string with pgvector
            rate_limiter: Rate limiter for embedding API calls
            timeout: Operation timeout in seconds
            use_memory_fallback: Use in-memory storage if DB unavailable
        """
        if rate_limiter is None:
            rate_limiter = RateLimiter(requests_per_second=50.0, burst_size=100)

        super().__init__(rate_limiter=rate_limiter, timeout=timeout)

        # Initialize embedding provider
        self._embedding_provider = embedding_provider or OpenRouterEmbeddingProvider()

        # Database connection
        self._connection_string = connection_string or os.environ.get(
            "DATABASE_URL",
            os.environ.get("POSTGRES_URL")
        )

        self._use_memory_fallback = use_memory_fallback
        self._memory_store: dict[str, dict[str, Document]] = {}  # session_id -> {doc_id -> doc}

        # Connection pool
        self._pool = None

    @property
    def name(self) -> str:
        return "rag_retriever"

    @property
    def description(self) -> str:
        return (
            "Retrieve relevant documents from the research knowledge base using "
            "vector similarity search. Use this to find previously collected "
            "information related to a query."
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding relevant documents"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID to scope the search"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum similarity threshold (0-1)",
                    "default": 0.5,
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["query", "session_id"]
        }

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None and self._connection_string:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    self._connection_string,
                    min_size=2,
                    max_size=10
                )
                # Ensure table exists
                await self._ensure_table()
            except Exception as e:
                logger.warning(f"Failed to create database pool: {e}")
                if not self._use_memory_fallback:
                    raise
        return self._pool

    async def _ensure_table(self):
        """Ensure the documents table exists."""
        pool = await self._get_pool()
        if pool is None:
            return

        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create documents table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self._embedding_provider.dimension}),
                    metadata JSONB DEFAULT '{{}}',
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_documents_session
                ON rag_documents(session_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding
                ON rag_documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

    async def _execute(self, input: str, **kwargs) -> ToolResult:
        """Execute retrieval operation."""
        session_id = kwargs.get("session_id", "default")
        k = kwargs.get("k", 10)
        threshold = kwargs.get("threshold", 0.5)

        try:
            results = await self.retrieve(
                query=input,
                session_id=session_id,
                k=k,
                threshold=threshold
            )

            return ToolResult.ok(
                [r.document.to_dict() for r in results],
                result_count=len(results),
                query=input,
                session_id=session_id
            )
        except Exception as e:
            return ToolResult.fail(str(e), query=input)

    async def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return await self._embedding_provider.get_embedding(text)

    async def retrieve(
        self,
        query: str,
        session_id: str,
        k: int = 10,
        threshold: float = 0.5
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant documents using vector similarity search.

        Args:
            query: Search query
            session_id: Session ID to scope the search
            k: Number of documents to retrieve
            threshold: Minimum similarity threshold

        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        # Get query embedding
        query_embedding = await self.get_embedding(query)

        # Try database first
        pool = await self._get_pool()
        if pool:
            return await self._retrieve_from_db(
                query_embedding, session_id, k, threshold
            )

        # Fallback to memory
        if self._use_memory_fallback:
            return await self._retrieve_from_memory(
                query_embedding, session_id, k, threshold
            )

        return []

    async def _retrieve_from_db(
        self,
        query_embedding: list[float],
        session_id: str,
        k: int,
        threshold: float
    ) -> list[RetrievalResult]:
        """Retrieve from PostgreSQL with pgvector."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Use cosine distance (1 - cosine_similarity)
            rows = await conn.fetch("""
                SELECT
                    id, content, metadata, source_url,
                    1 - (embedding <=> $1::vector) as similarity
                FROM rag_documents
                WHERE session_id = $2
                AND 1 - (embedding <=> $1::vector) >= $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
            """, query_embedding, session_id, threshold, k)

        results = []
        for rank, row in enumerate(rows, 1):
            doc = Document(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                source_url=row["source_url"]
            )
            results.append(RetrievalResult(
                document=doc,
                similarity_score=float(row["similarity"]),
                rank=rank
            ))

        return results

    async def _retrieve_from_memory(
        self,
        query_embedding: list[float],
        session_id: str,
        k: int,
        threshold: float
    ) -> list[RetrievalResult]:
        """Retrieve from in-memory store."""
        if session_id not in self._memory_store:
            return []

        documents = list(self._memory_store[session_id].values())

        # Calculate similarities
        query_vec = np.array(query_embedding)
        results = []

        for doc in documents:
            if doc.embedding is None:
                continue

            doc_vec = np.array(doc.embedding)
            # Cosine similarity
            similarity = float(
                np.dot(query_vec, doc_vec) /
                (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            )

            if similarity >= threshold:
                results.append((doc, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(document=doc, similarity_score=sim, rank=rank)
            for rank, (doc, sim) in enumerate(results[:k], 1)
        ]

    async def add_documents(
        self,
        documents: list[Document],
        session_id: str
    ) -> list[str]:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents to add
            session_id: Session ID to scope the documents

        Returns:
            List of document IDs
        """
        # Generate embeddings for documents without them
        texts_to_embed = []
        docs_needing_embedding = []

        for doc in documents:
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                docs_needing_embedding.append(doc)

        if texts_to_embed:
            embeddings = await self._embedding_provider.get_embeddings(texts_to_embed)
            for doc, embedding in zip(docs_needing_embedding, embeddings):
                doc.embedding = embedding

        # Try database first
        pool = await self._get_pool()
        if pool:
            return await self._add_to_db(documents, session_id)

        # Fallback to memory
        if self._use_memory_fallback:
            return await self._add_to_memory(documents, session_id)

        return []

    async def _add_to_db(
        self,
        documents: list[Document],
        session_id: str
    ) -> list[str]:
        """Add documents to PostgreSQL."""
        pool = await self._get_pool()
        doc_ids = []

        async with pool.acquire() as conn:
            for doc in documents:
                await conn.execute("""
                    INSERT INTO rag_documents
                    (id, session_id, content, embedding, metadata, source_url)
                    VALUES ($1, $2, $3, $4::vector, $5, $6)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        source_url = EXCLUDED.source_url,
                        updated_at = NOW()
                """,
                    doc.id,
                    session_id,
                    doc.content,
                    doc.embedding,
                    json.dumps(doc.metadata),
                    doc.source_url
                )
                doc_ids.append(doc.id)

        logger.info(f"Added {len(doc_ids)} documents to database for session {session_id}")
        return doc_ids

    async def _add_to_memory(
        self,
        documents: list[Document],
        session_id: str
    ) -> list[str]:
        """Add documents to in-memory store."""
        if session_id not in self._memory_store:
            self._memory_store[session_id] = {}

        doc_ids = []
        for doc in documents:
            self._memory_store[session_id][doc.id] = doc
            doc_ids.append(doc.id)

        logger.info(f"Added {len(doc_ids)} documents to memory for session {session_id}")
        return doc_ids

    async def delete_documents(
        self,
        document_ids: list[str],
        session_id: str
    ) -> int:
        """
        Delete documents from the knowledge base.

        Args:
            document_ids: List of document IDs to delete
            session_id: Session ID

        Returns:
            Number of documents deleted
        """
        pool = await self._get_pool()
        if pool:
            async with pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM rag_documents
                    WHERE id = ANY($1) AND session_id = $2
                """, document_ids, session_id)
                return int(result.split()[-1])

        # Memory fallback
        if session_id in self._memory_store:
            deleted = 0
            for doc_id in document_ids:
                if doc_id in self._memory_store[session_id]:
                    del self._memory_store[session_id][doc_id]
                    deleted += 1
            return deleted

        return 0

    async def clear_session(self, session_id: str) -> int:
        """
        Clear all documents for a session.

        Args:
            session_id: Session ID to clear

        Returns:
            Number of documents deleted
        """
        pool = await self._get_pool()
        if pool:
            async with pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM rag_documents WHERE session_id = $1
                """, session_id)
                return int(result.split()[-1])

        # Memory fallback
        if session_id in self._memory_store:
            count = len(self._memory_store[session_id])
            del self._memory_store[session_id]
            return count

        return 0

    async def get_document_count(self, session_id: str) -> int:
        """Get the number of documents in a session."""
        pool = await self._get_pool()
        if pool:
            async with pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT COUNT(*) FROM rag_documents WHERE session_id = $1
                """, session_id)
                return result or 0

        if session_id in self._memory_store:
            return len(self._memory_store[session_id])

        return 0

    async def close(self):
        """Clean up resources."""
        await self._embedding_provider.close()
        if self._pool:
            await self._pool.close()
        self._memory_store.clear()


# Factory functions
_rag_retriever_instance: RAGRetriever | None = None


def get_rag_retriever(
    embedding_provider: EmbeddingProvider | None = None,
    **kwargs
) -> RAGRetriever:
    """
    Factory function to get RAGRetriever instance.

    Uses singleton pattern for efficiency but allows custom instances.

    Args:
        embedding_provider: Custom embedding provider
        **kwargs: Additional configuration parameters

    Returns:
        RAGRetriever instance
    """
    global _rag_retriever_instance

    if embedding_provider is None and not kwargs:
        if _rag_retriever_instance is None:
            _rag_retriever_instance = RAGRetriever()
        return _rag_retriever_instance

    return RAGRetriever(embedding_provider=embedding_provider, **kwargs)


async def create_rag_retriever(
    embedding_provider: EmbeddingProvider | None = None,
    **kwargs
) -> RAGRetriever:
    """
    Async factory function for RAGRetriever.

    Initializes database connection pool.

    Args:
        embedding_provider: Custom embedding provider
        **kwargs: Additional configuration parameters

    Returns:
        Initialized RAGRetriever instance
    """
    retriever = get_rag_retriever(embedding_provider=embedding_provider, **kwargs)
    await retriever._get_pool()  # Initialize pool
    return retriever
