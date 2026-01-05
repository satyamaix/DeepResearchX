"""LangGraph Checkpointer Setup for DRX.

Provides async PostgreSQL-backed checkpointing for LangGraph workflows.
Uses AsyncPostgresSaver with critical configurations for production stability.

CRITICAL: prepare_threshold=0 is required to avoid prepared statement errors
when using pgbouncer or connection pooling in transaction mode.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row

if TYPE_CHECKING:
    from psycopg import AsyncConnection

from src.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton for checkpointer
_checkpointer: AsyncPostgresSaver | None = None
_checkpointer_lock = asyncio.Lock()
_checkpointer_setup_done: bool = False


class CheckpointerError(Exception):
    """Exception raised for checkpointer-related errors."""

    pass


async def get_checkpointer(
    db_uri: str | None = None,
    setup: bool = True,
) -> AsyncPostgresSaver:
    """Get or create the AsyncPostgresSaver singleton.

    Creates a LangGraph checkpointer backed by PostgreSQL for persisting
    workflow state across invocations.

    CRITICAL: Uses prepare_threshold=0 to prevent prepared statement errors
    that can occur with pgbouncer or when connections are reused.

    Args:
        db_uri: PostgreSQL connection URI. Uses DATABASE_URL from config if not provided.
        setup: Whether to run checkpointer.setup() on first creation.

    Returns:
        AsyncPostgresSaver: Configured checkpointer instance.

    Raises:
        CheckpointerError: If checkpointer creation or setup fails.

    Example:
        checkpointer = await get_checkpointer()
        graph = workflow.compile(checkpointer=checkpointer)
    """
    global _checkpointer, _checkpointer_setup_done

    if _checkpointer is not None:
        return _checkpointer

    async with _checkpointer_lock:
        # Double-check after acquiring lock
        if _checkpointer is not None:
            return _checkpointer

        settings = get_settings()
        connection_uri = db_uri or settings.database_url_str

        try:
            # Create connection with critical settings
            # CRITICAL: prepare_threshold=0 prevents prepared statement errors
            conn = await psycopg.AsyncConnection.connect(
                connection_uri,
                autocommit=True,
                row_factory=dict_row,
                prepare_threshold=0,  # CRITICAL: Disable prepared statements
            )

            # Create the checkpointer with the connection
            _checkpointer = AsyncPostgresSaver(conn)

            # Setup creates required tables if they don't exist
            if setup and not _checkpointer_setup_done:
                logger.info("Setting up LangGraph checkpointer tables...")
                await _checkpointer.setup()
                _checkpointer_setup_done = True
                logger.info("LangGraph checkpointer setup complete")

            logger.info("AsyncPostgresSaver checkpointer initialized")
            return _checkpointer

        except psycopg.OperationalError as e:
            logger.error(f"Failed to create checkpointer connection: {e}")
            raise CheckpointerError(f"Database connection failed: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}")
            raise CheckpointerError(f"Checkpointer initialization failed: {e}") from e


async def create_checkpointer(
    db_uri: str | None = None,
    setup: bool = True,
) -> AsyncPostgresSaver:
    """Create a new AsyncPostgresSaver instance (non-singleton).

    Use this when you need a separate checkpointer instance,
    for example for testing or isolated workflows.

    Args:
        db_uri: PostgreSQL connection URI.
        setup: Whether to run checkpointer.setup().

    Returns:
        AsyncPostgresSaver: New checkpointer instance.
    """
    settings = get_settings()
    connection_uri = db_uri or settings.database_url_str

    conn = await psycopg.AsyncConnection.connect(
        connection_uri,
        autocommit=True,
        row_factory=dict_row,
        prepare_threshold=0,
    )

    checkpointer = AsyncPostgresSaver(conn)

    if setup:
        await checkpointer.setup()

    return checkpointer


class CheckpointerManager:
    """Context manager for AsyncPostgresSaver lifecycle management.

    Provides a convenient way to manage checkpointer creation and cleanup,
    particularly useful for testing and long-running processes.

    Example:
        async with CheckpointerManager() as checkpointer:
            graph = workflow.compile(checkpointer=checkpointer)
            result = await graph.ainvoke({"input": "..."})
    """

    def __init__(
        self,
        db_uri: str | None = None,
        setup: bool = True,
        use_singleton: bool = True,
    ):
        """Initialize the CheckpointerManager.

        Args:
            db_uri: PostgreSQL connection URI.
            setup: Whether to run setup() on creation.
            use_singleton: Whether to use the global singleton or create new instance.
        """
        self.db_uri = db_uri
        self.setup = setup
        self.use_singleton = use_singleton
        self._checkpointer: AsyncPostgresSaver | None = None
        self._conn: AsyncConnection[dict[str, Any]] | None = None
        self._owns_connection: bool = False

    async def __aenter__(self) -> AsyncPostgresSaver:
        """Enter the async context and return the checkpointer."""
        if self.use_singleton:
            self._checkpointer = await get_checkpointer(self.db_uri, self.setup)
            self._owns_connection = False
        else:
            self._checkpointer = await create_checkpointer(self.db_uri, self.setup)
            self._owns_connection = True

        return self._checkpointer

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Exit the async context and cleanup if needed."""
        # Only close connection if we created it (non-singleton)
        if self._owns_connection and self._checkpointer is not None:
            try:
                # Access the underlying connection through the checkpointer
                if hasattr(self._checkpointer, "conn") and self._checkpointer.conn is not None:
                    await self._checkpointer.conn.close()
                    logger.debug("Checkpointer connection closed")
            except Exception as e:
                logger.warning(f"Error closing checkpointer connection: {e}")

        self._checkpointer = None


@asynccontextmanager
async def checkpointer_context(
    db_uri: str | None = None,
    setup: bool = True,
    use_singleton: bool = False,
) -> AsyncGenerator[AsyncPostgresSaver, None]:
    """Async context manager for checkpointer with automatic cleanup.

    Args:
        db_uri: PostgreSQL connection URI.
        setup: Whether to run setup().
        use_singleton: Whether to use singleton pattern.

    Yields:
        AsyncPostgresSaver: Configured checkpointer instance.

    Example:
        async with checkpointer_context() as cp:
            graph = workflow.compile(checkpointer=cp)
    """
    async with CheckpointerManager(db_uri, setup, use_singleton) as checkpointer:
        yield checkpointer


async def close_checkpointer() -> None:
    """Close the singleton checkpointer and its connection.

    Should be called during application shutdown.
    """
    global _checkpointer, _checkpointer_setup_done

    async with _checkpointer_lock:
        if _checkpointer is not None:
            try:
                if hasattr(_checkpointer, "conn") and _checkpointer.conn is not None:
                    await _checkpointer.conn.close()
                logger.info("Checkpointer connection closed")
            except Exception as e:
                logger.error(f"Error closing checkpointer: {e}")
            finally:
                _checkpointer = None
                _checkpointer_setup_done = False


async def get_thread_state(
    thread_id: str,
    checkpointer: AsyncPostgresSaver | None = None,
) -> dict[str, Any] | None:
    """Retrieve the latest state for a thread.

    Args:
        thread_id: The thread ID to look up.
        checkpointer: Checkpointer instance. Uses singleton if not provided.

    Returns:
        dict: The thread state if found, None otherwise.
    """
    cp = checkpointer or await get_checkpointer()

    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = await cp.aget_tuple(config)

        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            return checkpoint_tuple.checkpoint.get("channel_values", {})

        return None

    except Exception as e:
        logger.error(f"Error retrieving thread state: {e}")
        return None


async def list_thread_checkpoints(
    thread_id: str,
    limit: int = 10,
    checkpointer: AsyncPostgresSaver | None = None,
) -> list[dict[str, Any]]:
    """List checkpoints for a thread.

    Args:
        thread_id: The thread ID.
        limit: Maximum number of checkpoints to return.
        checkpointer: Checkpointer instance.

    Returns:
        list: List of checkpoint metadata.
    """
    cp = checkpointer or await get_checkpointer()

    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = []

        async for checkpoint_tuple in cp.alist(config, limit=limit):
            checkpoints.append({
                "thread_id": checkpoint_tuple.config.get("configurable", {}).get("thread_id"),
                "checkpoint_id": checkpoint_tuple.config.get("configurable", {}).get("checkpoint_id"),
                "checkpoint_ns": checkpoint_tuple.config.get("configurable", {}).get("checkpoint_ns", ""),
                "metadata": checkpoint_tuple.metadata,
            })

        return checkpoints

    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}")
        return []


async def delete_thread_checkpoints(
    thread_id: str,
    checkpointer: AsyncPostgresSaver | None = None,
) -> bool:
    """Delete all checkpoints for a thread.

    Args:
        thread_id: The thread ID to delete.
        checkpointer: Checkpointer instance.

    Returns:
        bool: True if deletion was successful.
    """
    cp = checkpointer or await get_checkpointer()

    try:
        # Get all checkpoints for this thread and delete them
        config = {"configurable": {"thread_id": thread_id}}

        async for checkpoint_tuple in cp.alist(config):
            checkpoint_config = checkpoint_tuple.config
            # Note: AsyncPostgresSaver may not have adelete, check implementation
            if hasattr(cp, "adelete"):
                await cp.adelete(checkpoint_config)

        logger.info(f"Deleted checkpoints for thread: {thread_id}")
        return True

    except Exception as e:
        logger.error(f"Error deleting checkpoints: {e}")
        return False
