"""
Context Propagation System for DRX Deep Research.

This module provides the ContextPropagator class for managing research context
across agent interactions and iterations. It handles context creation, relevance
checking, chunk retrieval, and context compression.

Key Components:
- ResearchContext TypedDict (re-exported from context_store)
- ContextPropagator class for context lifecycle management
- Utility functions for embedding and compression

Part of WP-M3: Context Propagation System implementation.

Usage:
    from src.metadata.context import ContextPropagator, ResearchContext

    # Create propagator with embedding and LLM clients
    propagator = ContextPropagator(
        embedding_client=embedding_client,
        llm_client=llm_client,
        context_store=store,
    )

    # Create context from agent state
    context = await propagator.create_context(agent_state)

    # Check relevance to a task
    is_relevant = await propagator.is_relevant(context, subtask)

    # Fetch relevant chunks
    chunks = await propagator.fetch_relevant_chunks(context, limit=5)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# Re-export ResearchContext from context_store for convenience
from src.metadata.context_store import (
    ContextMetadata,
    ContextStore,
    ResearchContext,
)

if TYPE_CHECKING:
    from src.orchestrator.state import AgentState, SubTask

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Definitions for Dependency Injection
# =============================================================================


@runtime_checkable
class EmbeddingClient(Protocol):
    """
    Protocol for embedding generation clients.

    Any embedding service (OpenAI, Anthropic, local models) that implements
    this protocol can be used with ContextPropagator.
    """

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for LLM clients used for context compression.

    Any LLM service that implements this protocol can be used
    for context summarization and compression.
    """

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        ...


@runtime_checkable
class ChunkStore(Protocol):
    """
    Protocol for chunk storage (pgvector-based).

    Used for storing and retrieving full content chunks
    referenced by ResearchContext.chunk_refs.
    """

    async def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve chunks by their IDs.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            List of chunk dictionaries with 'id', 'content', 'metadata'
        """
        ...

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.

        Args:
            embedding: Query embedding
            limit: Maximum results
            threshold: Minimum similarity threshold

        Returns:
            List of chunk dictionaries with similarity scores
        """
        ...


# =============================================================================
# Configuration
# =============================================================================


class ContextPropagatorConfig:
    """
    Configuration for ContextPropagator behavior.

    Attributes:
        default_ttl_seconds: Default TTL for new contexts
        relevance_threshold: Minimum similarity for relevance
        max_entities: Maximum key entities to extract
        compression_target_ratio: Target compression ratio
        embedding_dimensions: Expected embedding vector size
    """

    def __init__(
        self,
        default_ttl_seconds: int = 3600,
        relevance_threshold: float = 0.7,
        max_entities: int = 50,
        compression_target_ratio: float = 0.3,
        embedding_dimensions: int = 1536,
    ) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self.relevance_threshold = relevance_threshold
        self.max_entities = max_entities
        self.compression_target_ratio = compression_target_ratio
        self.embedding_dimensions = embedding_dimensions


# =============================================================================
# Context Propagator Class
# =============================================================================


class ContextPropagator:
    """
    Manages research context propagation across agents and iterations.

    The ContextPropagator is responsible for:
    1. Creating contexts from agent state
    2. Checking relevance of contexts to tasks
    3. Retrieving relevant content chunks
    4. Compressing contexts to fit token limits

    It integrates with embedding services for semantic similarity,
    LLM services for summarization, and context stores for persistence.

    Example:
        >>> propagator = ContextPropagator(
        ...     embedding_client=openai_embeddings,
        ...     llm_client=claude_client,
        ...     context_store=hybrid_store,
        ... )
        >>>
        >>> # Create context from current agent state
        >>> context = await propagator.create_context(state)
        >>>
        >>> # Check if context is relevant to a subtask
        >>> if await propagator.is_relevant(context, subtask):
        ...     chunks = await propagator.fetch_relevant_chunks(context)
        ...     compressed = await propagator.compress_context(
        ...         "\\n".join(chunks), max_tokens=2000
        ...     )
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        llm_client: LLMClient | None = None,
        context_store: ContextStore | None = None,
        chunk_store: ChunkStore | None = None,
        config: ContextPropagatorConfig | None = None,
    ) -> None:
        """
        Initialize the context propagator.

        Args:
            embedding_client: Client for generating embeddings
            llm_client: Client for LLM operations (compression)
            context_store: Store for context persistence
            chunk_store: Store for content chunks
            config: Configuration options
        """
        self._embedding_client = embedding_client
        self._llm_client = llm_client
        self._context_store = context_store
        self._chunk_store = chunk_store
        self._config = config or ContextPropagatorConfig()

        # Cache for embeddings to avoid redundant API calls
        self._embedding_cache: dict[str, list[float]] = {}

    @property
    def config(self) -> ContextPropagatorConfig:
        """Get the propagator configuration."""
        return self._config

    def _generate_context_id(self, session_id: str, content_hash: str) -> str:
        """
        Generate a unique context ID.

        Args:
            session_id: Session identifier
            content_hash: Hash of context content

        Returns:
            Unique context identifier
        """
        unique_part = uuid.uuid4().hex[:8]
        return f"ctx_{session_id[:8]}_{content_hash[:8]}_{unique_part}"

    def _hash_content(self, content: str) -> str:
        """
        Generate a hash of content for deduplication.

        Args:
            content: Content to hash

        Returns:
            SHA-256 hash of content
        """
        return hashlib.sha256(content.encode()).hexdigest()

    async def _get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for text, using cache if available.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ValueError: If no embedding client is configured
        """
        if self._embedding_client is None:
            raise ValueError("No embedding client configured")

        # Check cache
        content_hash = self._hash_content(text)
        if content_hash in self._embedding_cache:
            return self._embedding_cache[content_hash]

        # Generate embedding
        embedding = await self._embedding_client.embed(text)

        # Cache result (limit cache size)
        if len(self._embedding_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._embedding_cache.keys())[:100]
            for key in keys_to_remove:
                del self._embedding_cache[key]

        self._embedding_cache[content_hash] = embedding
        return embedding

    def _extract_key_entities(self, state: "AgentState") -> list[str]:
        """
        Extract key entities from agent state.

        Extracts important terms, names, and concepts from the state
        for efficient context matching.

        Args:
            state: Current agent state

        Returns:
            List of key entity strings
        """
        entities: set[str] = set()

        # Add user query terms
        query = state.get("user_query", "")
        if query:
            # Simple extraction: split on whitespace, filter short words
            words = query.split()
            entities.update(w for w in words if len(w) > 3)

        # Add focus areas from steerability
        steerability = state.get("steerability", {})
        if steerability:
            focus_areas = steerability.get("focus_areas", [])
            entities.update(focus_areas)

        # Add entities from findings
        findings = state.get("findings", [])
        for finding in findings:
            # Extract from claim
            claim = finding.get("claim", "")
            if claim:
                words = claim.split()
                entities.update(w for w in words if len(w) > 4)

            # Add tags
            tags = finding.get("tags", [])
            entities.update(tags)

        # Add sub-questions from plan
        plan = state.get("plan")
        if plan:
            sub_questions = plan.get("sub_questions", [])
            for sq in sub_questions:
                words = sq.split()
                entities.update(w for w in words if len(w) > 4)

        # Limit to max entities
        entity_list = list(entities)
        return entity_list[: self._config.max_entities]

    def _build_context_summary(self, state: "AgentState") -> str:
        """
        Build a summary string from agent state.

        Creates a condensed representation of the current research
        state for context propagation.

        Args:
            state: Current agent state

        Returns:
            Summary string
        """
        parts: list[str] = []

        # Add query
        query = state.get("user_query", "")
        if query:
            parts.append(f"Query: {query}")

        # Add current phase
        phase = state.get("current_phase", "")
        if phase:
            parts.append(f"Phase: {phase}")

        # Add iteration info
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 5)
        parts.append(f"Iteration: {iteration}/{max_iter}")

        # Add synthesis summary (truncated)
        synthesis = state.get("synthesis", "")
        if synthesis:
            truncated = synthesis[:500] + "..." if len(synthesis) > 500 else synthesis
            parts.append(f"Synthesis: {truncated}")

        # Add findings summary
        findings = state.get("findings", [])
        if findings:
            finding_claims = [f.get("claim", "")[:100] for f in findings[:5]]
            parts.append(f"Key Findings ({len(findings)} total):")
            parts.extend(f"  - {claim}" for claim in finding_claims if claim)

        # Add gaps
        gaps = state.get("gaps", [])
        if gaps:
            parts.append(f"Knowledge Gaps: {', '.join(gaps[:5])}")

        return "\n".join(parts)

    async def create_context(
        self,
        state: "AgentState",
        ttl_seconds: int | None = None,
        metadata: ContextMetadata | None = None,
    ) -> ResearchContext:
        """
        Create a ResearchContext from agent state.

        Extracts key information from the current agent state,
        generates embeddings, and creates a context suitable
        for propagation and relevance checking.

        Args:
            state: Current agent state
            ttl_seconds: Optional custom TTL
            metadata: Optional additional metadata

        Returns:
            Created ResearchContext

        Example:
            >>> context = await propagator.create_context(state)
            >>> print(context["context_id"])
            'ctx_abc12345_def67890_12345678'
        """
        # Build summary and extract entities
        summary = self._build_context_summary(state)
        key_entities = self._extract_key_entities(state)

        # Generate embedding for relevance matching
        if self._embedding_client:
            embedding_text = f"{state.get('user_query', '')} {summary[:1000]}"
            relevance_vector = await self._get_embedding(embedding_text)
        else:
            # Fallback: empty vector if no embedding client
            relevance_vector = []
            logger.warning("No embedding client - context will have empty relevance vector")

        # Generate context ID
        session_id = state.get("session_id", str(uuid.uuid4()))
        content_hash = self._hash_content(summary)
        context_id = self._generate_context_id(session_id, content_hash)

        # Create timestamp
        created_at = datetime.utcnow().isoformat() + "Z"

        # Build context
        context: ResearchContext = {
            "context_id": context_id,
            "session_id": session_id,
            "summary": summary,
            "key_entities": key_entities,
            "relevance_vector": relevance_vector,
            "chunk_refs": [],  # Will be populated when chunks are stored
            "created_at": created_at,
            "ttl_seconds": ttl_seconds or self._config.default_ttl_seconds,
        }

        # Store if context store is available
        if self._context_store:
            await self._context_store.store_context(context, metadata)
            logger.debug(f"Created and stored context {context_id}")
        else:
            logger.debug(f"Created context {context_id} (not stored)")

        return context

    async def is_relevant(
        self,
        context: ResearchContext,
        task: "SubTask",
        threshold: float | None = None,
    ) -> bool:
        """
        Check if a context is relevant to a subtask.

        Uses embedding similarity to determine if the context
        contains information relevant to the given task.

        Args:
            context: ResearchContext to check
            task: SubTask to check relevance against
            threshold: Optional custom relevance threshold

        Returns:
            True if context is relevant, False otherwise

        Example:
            >>> if await propagator.is_relevant(context, task):
            ...     print("Context is relevant!")
        """
        threshold = threshold or self._config.relevance_threshold

        # If no embedding client, fall back to keyword matching
        if self._embedding_client is None:
            return self._keyword_relevance(context, task)

        # Get context embedding
        context_vector = context.get("relevance_vector", [])
        if not context_vector:
            logger.warning(f"Context {context['context_id']} has no relevance vector")
            return self._keyword_relevance(context, task)

        # Build task text for embedding
        task_text = self._build_task_text(task)

        # Get task embedding
        task_vector = await self._get_embedding(task_text)

        # Calculate cosine similarity
        similarity = self._cosine_similarity(context_vector, task_vector)

        logger.debug(
            f"Context {context['context_id']} relevance to task {task.get('id', 'unknown')}: "
            f"{similarity:.3f} (threshold: {threshold})"
        )

        return similarity >= threshold

    def _build_task_text(self, task: "SubTask") -> str:
        """
        Build text representation of a subtask for embedding.

        Args:
            task: SubTask to represent

        Returns:
            Text representation
        """
        parts: list[str] = []

        description = task.get("description", "")
        if description:
            parts.append(description)

        inputs = task.get("inputs", {})
        if inputs:
            # Include relevant input fields
            query = inputs.get("query", "")
            if query:
                parts.append(query)

            context_text = inputs.get("context", "")
            if context_text:
                parts.append(str(context_text)[:500])

        return " ".join(parts)

    def _keyword_relevance(self, context: ResearchContext, task: "SubTask") -> bool:
        """
        Fall back to keyword-based relevance when embeddings unavailable.

        Args:
            context: ResearchContext to check
            task: SubTask to check against

        Returns:
            True if keywords overlap significantly
        """
        context_entities = set(e.lower() for e in context.get("key_entities", []))
        task_text = self._build_task_text(task).lower()

        # Check how many context entities appear in task
        matches = sum(1 for entity in context_entities if entity in task_text)

        # Consider relevant if at least 2 entities match or 20% of entities
        min_matches = max(2, len(context_entities) * 0.2)
        return matches >= min_matches

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0.0-1.0)
        """
        if len(vec1) != len(vec2):
            logger.warning(
                f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}"
            )
            return 0.0

        if not vec1 or not vec2:
            return 0.0

        # Calculate dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def fetch_relevant_chunks(
        self,
        context: ResearchContext,
        limit: int = 10,
        include_similar: bool = True,
    ) -> list[str]:
        """
        Retrieve relevant content chunks for a context.

        Fetches chunks referenced by the context and optionally
        searches for similar chunks using vector similarity.

        Args:
            context: ResearchContext to fetch chunks for
            limit: Maximum number of chunks to return
            include_similar: Whether to include similar chunks

        Returns:
            List of chunk content strings

        Example:
            >>> chunks = await propagator.fetch_relevant_chunks(context, limit=5)
            >>> for chunk in chunks:
            ...     print(chunk[:100])
        """
        chunks: list[str] = []

        # First, fetch directly referenced chunks
        chunk_refs = context.get("chunk_refs", [])
        if chunk_refs and self._chunk_store:
            try:
                referenced_chunks = await self._chunk_store.get_chunks(chunk_refs[:limit])
                for chunk in referenced_chunks:
                    content = chunk.get("content", "")
                    if content:
                        chunks.append(content)
            except Exception as e:
                logger.error(f"Failed to fetch referenced chunks: {e}")

        # If we need more chunks and have embedding capability, search for similar
        remaining = limit - len(chunks)
        if remaining > 0 and include_similar and self._chunk_store:
            relevance_vector = context.get("relevance_vector", [])
            if relevance_vector:
                try:
                    similar_chunks = await self._chunk_store.search_similar(
                        relevance_vector,
                        limit=remaining,
                        threshold=self._config.relevance_threshold,
                    )
                    for chunk in similar_chunks:
                        content = chunk.get("content", "")
                        if content and content not in chunks:
                            chunks.append(content)
                except Exception as e:
                    logger.error(f"Failed to search similar chunks: {e}")

        # If no chunk store, return summary as single chunk
        if not chunks:
            summary = context.get("summary", "")
            if summary:
                chunks.append(summary)

        return chunks[:limit]

    async def compress_context(
        self,
        full_context: str,
        max_tokens: int = 2000,
        preserve_facts: bool = True,
    ) -> str:
        """
        Compress context to fit within token limit.

        Uses LLM to intelligently summarize context while
        preserving key information and facts.

        Args:
            full_context: Full context text to compress
            max_tokens: Maximum tokens in compressed output
            preserve_facts: Whether to prioritize fact preservation

        Returns:
            Compressed context string

        Example:
            >>> compressed = await propagator.compress_context(
            ...     long_context_text,
            ...     max_tokens=1000
            ... )
        """
        # Estimate input tokens (rough: 4 chars per token)
        estimated_input_tokens = len(full_context) // 4

        # If already within limit, return as-is
        if estimated_input_tokens <= max_tokens:
            return full_context

        # If no LLM client, fall back to truncation
        if self._llm_client is None:
            logger.warning("No LLM client - using truncation for compression")
            return self._truncate_context(full_context, max_tokens)

        # Build compression prompt
        prompt = self._build_compression_prompt(
            full_context, max_tokens, preserve_facts
        )

        try:
            compressed = await self._llm_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.3,  # Low temperature for factual summarization
            )

            logger.debug(
                f"Compressed context from ~{estimated_input_tokens} to "
                f"~{len(compressed) // 4} tokens"
            )

            return compressed.strip()

        except Exception as e:
            logger.error(f"LLM compression failed: {e}")
            return self._truncate_context(full_context, max_tokens)

    def _build_compression_prompt(
        self,
        content: str,
        max_tokens: int,
        preserve_facts: bool,
    ) -> str:
        """
        Build prompt for LLM-based compression.

        Args:
            content: Content to compress
            max_tokens: Target token limit
            preserve_facts: Whether to emphasize fact preservation

        Returns:
            Compression prompt
        """
        fact_instruction = ""
        if preserve_facts:
            fact_instruction = (
                "Prioritize preserving specific facts, numbers, names, and citations. "
                "Do not introduce any new information not present in the original."
            )

        return f"""Compress the following research context into approximately {max_tokens} tokens while maintaining all essential information.

{fact_instruction}

Guidelines:
- Maintain key findings, claims, and evidence
- Preserve important entity names and relationships
- Keep citation references intact
- Remove redundant or verbose explanations
- Use concise, information-dense language

CONTEXT TO COMPRESS:
{content}

COMPRESSED VERSION:"""

    def _truncate_context(self, content: str, max_tokens: int) -> str:
        """
        Simple truncation fallback for compression.

        Args:
            content: Content to truncate
            max_tokens: Target token limit

        Returns:
            Truncated content
        """
        # Estimate max chars (4 chars per token average)
        max_chars = max_tokens * 4

        if len(content) <= max_chars:
            return content

        # Truncate with ellipsis
        truncated = content[: max_chars - 3] + "..."

        # Try to truncate at sentence boundary
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.8:  # At least 80% of target
            truncated = truncated[: last_period + 1]

        return truncated

    async def merge_contexts(
        self,
        contexts: list[ResearchContext],
        session_id: str,
    ) -> ResearchContext:
        """
        Merge multiple contexts into a single context.

        Combines summaries, entities, and embeddings from multiple
        contexts into a unified context.

        Args:
            contexts: List of contexts to merge
            session_id: Session ID for the merged context

        Returns:
            Merged ResearchContext
        """
        if not contexts:
            raise ValueError("Cannot merge empty context list")

        if len(contexts) == 1:
            return contexts[0]

        # Merge summaries
        summaries = [c.get("summary", "") for c in contexts if c.get("summary")]
        merged_summary = "\n---\n".join(summaries)

        # Merge entities (deduplicated)
        all_entities: set[str] = set()
        for ctx in contexts:
            all_entities.update(ctx.get("key_entities", []))
        merged_entities = list(all_entities)[: self._config.max_entities]

        # Merge chunk refs (deduplicated)
        all_chunks: list[str] = []
        seen_chunks: set[str] = set()
        for ctx in contexts:
            for ref in ctx.get("chunk_refs", []):
                if ref not in seen_chunks:
                    seen_chunks.add(ref)
                    all_chunks.append(ref)

        # Average embeddings
        vectors = [c.get("relevance_vector", []) for c in contexts if c.get("relevance_vector")]
        if vectors:
            merged_vector = self._average_vectors(vectors)
        else:
            merged_vector = []

        # Generate new context ID
        content_hash = self._hash_content(merged_summary)
        context_id = self._generate_context_id(session_id, content_hash)

        # Use minimum TTL from source contexts
        min_ttl = min(c.get("ttl_seconds", 3600) for c in contexts)

        merged_context: ResearchContext = {
            "context_id": context_id,
            "session_id": session_id,
            "summary": merged_summary,
            "key_entities": merged_entities,
            "relevance_vector": merged_vector,
            "chunk_refs": all_chunks,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "ttl_seconds": min_ttl,
        }

        return merged_context

    def _average_vectors(self, vectors: list[list[float]]) -> list[float]:
        """
        Calculate average of multiple vectors.

        Args:
            vectors: List of vectors to average

        Returns:
            Averaged vector
        """
        if not vectors:
            return []

        # Ensure all vectors have same dimension
        dim = len(vectors[0])
        valid_vectors = [v for v in vectors if len(v) == dim]

        if not valid_vectors:
            return []

        # Calculate element-wise average
        n = len(valid_vectors)
        averaged = [sum(v[i] for v in valid_vectors) / n for i in range(dim)]

        return averaged

    async def get_session_context(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[ResearchContext]:
        """
        Get all contexts for a session.

        Args:
            session_id: Session identifier
            limit: Maximum contexts to return

        Returns:
            List of ResearchContext objects
        """
        if not self._context_store:
            logger.warning("No context store configured")
            return []

        return await self._context_store.get_contexts_by_session(session_id, limit)

    async def cleanup_expired_contexts(self) -> int:
        """
        Clean up expired contexts from the store.

        Returns:
            Number of contexts cleaned up
        """
        if not self._context_store:
            return 0

        # Check if store supports cleanup
        if hasattr(self._context_store, "cleanup_expired"):
            return await self._context_store.cleanup_expired()

        return 0


# =============================================================================
# Utility Functions
# =============================================================================


def create_empty_context(
    session_id: str,
    ttl_seconds: int = 3600,
) -> ResearchContext:
    """
    Create an empty ResearchContext.

    Useful for initializing context before populating.

    Args:
        session_id: Session identifier
        ttl_seconds: Time-to-live in seconds

    Returns:
        Empty ResearchContext
    """
    context_id = f"ctx_{session_id[:8]}_{uuid.uuid4().hex[:16]}"

    return ResearchContext(
        context_id=context_id,
        session_id=session_id,
        summary="",
        key_entities=[],
        relevance_vector=[],
        chunk_refs=[],
        created_at=datetime.utcnow().isoformat() + "Z",
        ttl_seconds=ttl_seconds,
    )


def context_to_dict(context: ResearchContext) -> dict[str, Any]:
    """
    Convert ResearchContext to plain dictionary.

    Args:
        context: Context to convert

    Returns:
        Dictionary representation
    """
    return dict(context)


def estimate_context_tokens(context: ResearchContext) -> int:
    """
    Estimate token count for a context.

    Args:
        context: Context to estimate

    Returns:
        Estimated token count
    """
    # Rough estimate: 4 characters per token
    total_chars = len(context.get("summary", ""))
    total_chars += sum(len(e) for e in context.get("key_entities", []))

    return total_chars // 4


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    # Re-exported from context_store
    "ResearchContext",
    "ContextMetadata",
    # Protocols
    "EmbeddingClient",
    "LLMClient",
    "ChunkStore",
    # Configuration
    "ContextPropagatorConfig",
    # Main class
    "ContextPropagator",
    # Utility functions
    "create_empty_context",
    "context_to_dict",
    "estimate_context_tokens",
]
