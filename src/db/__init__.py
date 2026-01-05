"""DRX Database Module.

Provides async PostgreSQL connection management and utilities.

Core Components:
- Connection pooling with psycopg v3
- Schema initialization
- Transaction management
- Health checks

Example:
    ```python
    from src.db import get_async_connection, init_db

    # Initialize schema on startup
    await init_db()

    # Use connection in async context
    async with get_async_connection() as conn:
        result = await conn.execute("SELECT * FROM research_sessions")
        sessions = await result.fetchall()
    ```
"""

from src.db.connection import (
    # Connection management
    get_async_connection,
    get_async_pool,
    get_raw_connection,
    close_pool,
    # Schema management
    init_db,
    # Health and monitoring
    check_connection_health,
    get_pool_stats,
    # Transaction helper
    transaction,
    # Exceptions
    DatabaseError,
    ConnectionError,
    SchemaInitializationError,
)


__all__ = [
    # Connection management
    "get_async_connection",
    "get_async_pool",
    "get_raw_connection",
    "close_pool",
    # Schema management
    "init_db",
    # Health and monitoring
    "check_connection_health",
    "get_pool_stats",
    # Transaction helper
    "transaction",
    # Exceptions
    "DatabaseError",
    "ConnectionError",
    "SchemaInitializationError",
]
