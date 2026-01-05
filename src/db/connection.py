"""Database Connection Management for DRX.

Provides async PostgreSQL connection utilities using psycopg v3:
- Connection pooling with health checks
- Async context managers for safe connection handling
- Schema initialization from SQL files
- Connection lifecycle management
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

if TYPE_CHECKING:
    from psycopg import AsyncConnection

from src.config import get_settings

logger = logging.getLogger(__name__)

# Module-level pool instance for singleton pattern
_pool: AsyncConnectionPool | None = None
_pool_lock = asyncio.Lock()


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


class ConnectionError(DatabaseError):
    """Exception raised when connection fails."""

    pass


class SchemaInitializationError(DatabaseError):
    """Exception raised when schema initialization fails."""

    pass


async def get_async_pool(
    min_size: int | None = None,
    max_size: int | None = None,
    timeout: float | None = None,
    max_lifetime: float = 3600.0,
    max_idle: float = 600.0,
    num_workers: int = 3,
    check_on_connect: bool = True,
) -> AsyncConnectionPool:
    """Get or create the async connection pool singleton.

    Uses a singleton pattern with async lock to ensure only one pool
    is created across all concurrent requests.

    Args:
        min_size: Minimum number of connections to maintain. Defaults to config value.
        max_size: Maximum number of connections. Defaults to config value.
        timeout: Connection acquisition timeout. Defaults to config value.
        max_lifetime: Maximum connection lifetime in seconds before recycling.
        max_idle: Maximum time a connection can be idle before being closed.
        num_workers: Number of background workers for pool maintenance.
        check_on_connect: Whether to check connection health on acquisition.

    Returns:
        AsyncConnectionPool: The database connection pool.

    Raises:
        ConnectionError: If pool creation fails.
    """
    global _pool

    if _pool is not None and not _pool.closed:
        return _pool

    async with _pool_lock:
        # Double-check pattern after acquiring lock
        if _pool is not None and not _pool.closed:
            return _pool

        settings = get_settings()

        # Use provided values or fall back to config
        pool_min = min_size if min_size is not None else settings.DB_POOL_MIN_SIZE
        pool_max = max_size if max_size is not None else settings.DB_POOL_MAX_SIZE
        pool_timeout = timeout if timeout is not None else settings.DB_POOL_TIMEOUT

        try:
            # Connection arguments for psycopg
            # CRITICAL: prepare_threshold=0 prevents prepared statement errors
            # with pgbouncer and LangGraph checkpointing
            conn_kwargs: dict[str, Any] = {
                "autocommit": True,
                "row_factory": dict_row,
                "prepare_threshold": 0,  # Disable prepared statements
            }

            _pool = AsyncConnectionPool(
                conninfo=settings.database_url_str,
                min_size=pool_min,
                max_size=pool_max,
                timeout=pool_timeout,
                max_lifetime=max_lifetime,
                max_idle=max_idle,
                num_workers=num_workers,
                check=AsyncConnectionPool.check_connection if check_on_connect else None,
                kwargs=conn_kwargs,
                open=False,  # Don't open immediately, we'll do it explicitly
            )

            # Open the pool and wait for minimum connections
            await _pool.open(wait=True, timeout=pool_timeout)

            logger.info(
                "Database connection pool created",
                extra={
                    "min_size": pool_min,
                    "max_size": pool_max,
                    "timeout": pool_timeout,
                },
            )

            return _pool

        except psycopg.OperationalError as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise ConnectionError(f"Failed to create database connection pool: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error creating connection pool: {e}")
            raise ConnectionError(f"Unexpected database error: {e}") from e


@asynccontextmanager
async def get_async_connection(
    autocommit: bool = True,
) -> AsyncGenerator[AsyncConnection[dict[str, Any]], None]:
    """Get an async database connection from the pool.

    This is an async context manager that automatically returns the
    connection to the pool when the context exits.

    Args:
        autocommit: Whether to enable autocommit mode. Defaults to True.

    Yields:
        AsyncConnection: A psycopg async connection configured with dict_row factory.

    Raises:
        ConnectionError: If connection acquisition fails.

    Example:
        async with get_async_connection() as conn:
            result = await conn.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user = await result.fetchone()
    """
    pool = await get_async_pool()

    try:
        async with pool.connection() as conn:
            # Ensure consistent settings
            if conn.autocommit != autocommit:
                await conn.set_autocommit(autocommit)

            yield conn

    except psycopg.OperationalError as e:
        logger.error(f"Database connection error: {e}")
        raise ConnectionError(f"Failed to acquire database connection: {e}") from e
    except psycopg.Error as e:
        logger.error(f"Database error: {e}")
        raise DatabaseError(f"Database operation failed: {e}") from e


@asynccontextmanager
async def get_raw_connection(
    conninfo: str | None = None,
    autocommit: bool = True,
) -> AsyncGenerator[AsyncConnection[dict[str, Any]], None]:
    """Get a standalone async connection (not from pool).

    Useful for one-off operations or when pool is not available.
    The connection is closed when the context exits.

    Args:
        conninfo: Connection string. Uses DATABASE_URL from config if not provided.
        autocommit: Whether to enable autocommit mode.

    Yields:
        AsyncConnection: A psycopg async connection.
    """
    settings = get_settings()
    connection_string = conninfo or settings.database_url_str

    conn = await psycopg.AsyncConnection.connect(
        connection_string,
        autocommit=autocommit,
        row_factory=dict_row,
        prepare_threshold=0,
    )

    try:
        yield conn
    finally:
        await conn.close()


async def check_connection_health() -> bool:
    """Check if database connection is healthy.

    Returns:
        bool: True if connection is healthy, False otherwise.
    """
    try:
        async with get_async_connection() as conn:
            result = await conn.execute("SELECT 1")
            row = await result.fetchone()
            return row is not None and row.get("?column?") == 1
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return False


async def init_db(
    schema_path: str | Path | None = None,
    force_reinit: bool = False,
) -> bool:
    """Initialize database schema.

    Executes the PostgreSQL schema file to create tables, indexes,
    and other database objects if they don't already exist.

    Args:
        schema_path: Path to the SQL schema file. Defaults to schemas/postgres_schema.sql.
        force_reinit: If True, drop and recreate tables (DANGEROUS in production).

    Returns:
        bool: True if schema was initialized or already exists.

    Raises:
        SchemaInitializationError: If schema initialization fails.
        FileNotFoundError: If schema file is not found.
    """
    # Determine schema file path
    if schema_path is None:
        # Default to project root schemas directory
        project_root = Path(__file__).parent.parent.parent
        schema_path = project_root / "schemas" / "postgres_schema.sql"
    else:
        schema_path = Path(schema_path)

    if not schema_path.exists():
        logger.warning(f"Schema file not found at {schema_path}")
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    try:
        # Read schema SQL
        schema_sql = schema_path.read_text(encoding="utf-8")

        async with get_async_connection(autocommit=True) as conn:
            # Check if tables already exist
            result = await conn.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'research_sessions'
                )
                """
            )
            row = await result.fetchone()
            tables_exist = row and row.get("exists", False)

            if tables_exist and not force_reinit:
                logger.info("Database schema already exists, skipping initialization")
                return True

            if force_reinit:
                logger.warning("Force reinitializing database schema")
                # Drop existing tables (add your tables here)
                await conn.execute(
                    """
                    DROP TABLE IF EXISTS
                        research_sessions,
                        research_sources,
                        research_findings,
                        checkpoints,
                        checkpoint_blobs,
                        checkpoint_writes
                    CASCADE
                    """
                )

            # Execute schema
            logger.info(f"Initializing database schema from {schema_path}")
            await conn.execute(schema_sql)
            logger.info("Database schema initialized successfully")

            return True

    except FileNotFoundError:
        raise
    except psycopg.Error as e:
        logger.error(f"Schema initialization failed: {e}")
        raise SchemaInitializationError(f"Failed to initialize schema: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during schema initialization: {e}")
        raise SchemaInitializationError(f"Unexpected error: {e}") from e


async def close_pool() -> None:
    """Close the connection pool.

    Should be called during application shutdown to cleanly
    close all database connections.
    """
    global _pool

    if _pool is not None:
        try:
            await _pool.close()
            logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")
        finally:
            _pool = None


async def get_pool_stats() -> dict[str, Any]:
    """Get connection pool statistics.

    Returns:
        dict: Pool statistics including size, idle, and used connections.
    """
    if _pool is None:
        return {"status": "not_initialized"}

    return {
        "status": "active" if not _pool.closed else "closed",
        "min_size": _pool.min_size,
        "max_size": _pool.max_size,
        "size": _pool.get_stats().get("pool_size", 0),
        "available": _pool.get_stats().get("pool_available", 0),
        "requests_waiting": _pool.get_stats().get("requests_waiting", 0),
    }


# Transaction helper for operations requiring explicit transaction control
@asynccontextmanager
async def transaction() -> AsyncGenerator[AsyncConnection[dict[str, Any]], None]:
    """Execute operations within a database transaction.

    This context manager provides a connection with autocommit disabled
    and handles commit/rollback automatically.

    Yields:
        AsyncConnection: Connection within a transaction context.

    Raises:
        DatabaseError: If transaction operations fail.

    Example:
        async with transaction() as conn:
            await conn.execute("INSERT INTO users ...", (...))
            await conn.execute("INSERT INTO profiles ...", (...))
            # Commits automatically if no exception
            # Rolls back automatically on exception
    """
    pool = await get_async_pool()

    async with pool.connection() as conn:
        # Disable autocommit for transaction
        await conn.set_autocommit(False)

        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            # Restore autocommit for pool return
            await conn.set_autocommit(True)
