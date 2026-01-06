"""FastAPI Application Setup for DRX Deep Research API.

Provides the FastAPI application factory with middleware, exception handlers,
lifespan management, and Phoenix observability instrumentation.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.dependencies import close_redis_pool
from src.api.replay_routes import router as replay_router
from src.api.routes import ErrorResponse, router
from src.config import Settings, get_settings
from src.db.connection import close_pool as close_db_pool
from src.db.connection import get_async_pool, init_db

logger = logging.getLogger(__name__)


# =============================================================================
# Application Metadata
# =============================================================================

API_TITLE = "DRX Deep Research API"
API_DESCRIPTION = """
## DRX Deep Research Platform

A multi-agent research system providing:

- **Research Interactions**: Create and manage deep research tasks
- **Real-time Streaming**: SSE-based progress updates
- **Checkpoint/Resume**: Pause and resume long-running research
- **Quality Assurance**: Built-in verification and gap analysis

### Authentication

All endpoints except `/health` require authentication via:
- **Bearer Token**: `Authorization: Bearer <token>`
- **API Key**: `X-API-Key: drx_<key>`

### Rate Limits

| Tier | Requests/min | Burst |
|------|-------------|-------|
| Standard | 60 | 10 |
| Premium | 300 | 50 |

### SSE Events

The `/interactions/{id}/stream` endpoint emits:
- `interaction.start` - Processing started
- `thought_summary` - Agent reasoning
- `content.delta` - Incremental content
- `tool.use` - Tool invocations
- `interaction.complete` - Processing finished
"""

API_VERSION = "1.0.0"

TAGS_METADATA = [
    {
        "name": "research",
        "description": "Research interaction management",
    },
    {
        "name": "replay",
        "description": "Session replay and event recording",
    },
    {
        "name": "system",
        "description": "Health checks and system status",
    },
]


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    Handles:
    - Database connection pool initialization
    - Redis connection pool initialization
    - Phoenix/OpenTelemetry instrumentation
    - Graceful shutdown of all connections
    """
    settings = get_settings()
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"Environment: {settings.APP_ENV}")

    # Startup
    try:
        # Initialize database pool
        logger.info("Initializing database connection pool...")
        await get_async_pool()
        logger.info("Database pool initialized")

        # Initialize database schema (in development)
        if settings.is_development:
            try:
                await init_db()
                logger.info("Database schema verified")
            except FileNotFoundError:
                logger.warning("Schema file not found, skipping initialization")
            except Exception as e:
                logger.warning(f"Schema initialization skipped: {e}")

        # Initialize Phoenix/OpenTelemetry instrumentation using observability module
        if settings.PHOENIX_COLLECTOR_ENDPOINT:
            try:
                from src.observability import setup_phoenix
                setup_phoenix()
                logger.info("Phoenix instrumentation configured via observability module")
            except Exception as e:
                logger.warning(f"Phoenix instrumentation failed: {e}")

        logger.info(f"{API_TITLE} started successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")

    try:
        # Shutdown Phoenix tracing
        from src.observability import shutdown_phoenix
        shutdown_phoenix()
        logger.info("Phoenix tracing shutdown")
    except Exception as e:
        logger.warning(f"Phoenix shutdown error: {e}")

    try:
        # Close Redis connections
        await close_redis_pool()
        logger.info("Redis pool closed")
    except Exception as e:
        logger.warning(f"Redis shutdown error: {e}")

    try:
        # Close database connections
        await close_db_pool()
        logger.info("Database pool closed")
    except Exception as e:
        logger.warning(f"Database shutdown error: {e}")

    logger.info(f"{API_TITLE} shutdown complete")




# =============================================================================
# Exception Handlers
# =============================================================================


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Handle HTTP exceptions with consistent error format.

    Args:
        request: FastAPI request object.
        exc: HTTPException instance.

    Returns:
        JSONResponse with error details.
    """
    request_id = getattr(request.state, "request_id", None)

    error_response = ErrorResponse(
        error=exc.detail if isinstance(exc.detail, str) else "Error",
        detail=exc.detail if not isinstance(exc.detail, str) else None,
        code=f"HTTP_{exc.status_code}",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(error_response),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle request validation errors.

    Args:
        request: FastAPI request object.
        exc: RequestValidationError instance.

    Returns:
        JSONResponse with validation error details.
    """
    request_id = getattr(request.state, "request_id", None)

    # Format validation errors
    errors = []
    for error in exc.errors():
        loc = ".".join(str(l) for l in error["loc"])
        errors.append(f"{loc}: {error['msg']}")

    error_response = ErrorResponse(
        error="Validation error",
        detail="; ".join(errors),
        code="VALIDATION_ERROR",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(error_response),
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions.

    Args:
        request: FastAPI request object.
        exc: Exception instance.

    Returns:
        JSONResponse with error details (sanitized in production).
    """
    request_id = getattr(request.state, "request_id", None)
    settings = get_settings()

    logger.exception(
        f"Unhandled exception: {exc}",
        extra={"request_id": request_id},
    )

    # Don't leak internal errors in production
    if settings.is_production:
        detail = "An internal error occurred"
    else:
        detail = str(exc)

    error_response = ErrorResponse(
        error="Internal server error",
        detail=detail,
        code="INTERNAL_ERROR",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(error_response),
    )


# =============================================================================
# Middleware
# =============================================================================


async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    """Add unique request ID to each request.

    Args:
        request: FastAPI request object.
        call_next: Next middleware/handler.

    Returns:
        Response with X-Request-ID header.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


async def timing_middleware(request: Request, call_next: Callable) -> Response:
    """Add request timing information.

    Args:
        request: FastAPI request object.
        call_next: Next middleware/handler.

    Returns:
        Response with X-Response-Time header.
    """
    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

    # Log slow requests
    if duration_ms > 1000:
        logger.warning(
            f"Slow request: {request.method} {request.url.path} took {duration_ms:.2f}ms"
        )

    return response


async def rate_limit_header_middleware(
    request: Request, call_next: Callable
) -> Response:
    """Add rate limit headers from request state.

    Args:
        request: FastAPI request object.
        call_next: Next middleware/handler.

    Returns:
        Response with rate limit headers.
    """
    response = await call_next(request)

    # Add rate limit headers if available
    rate_limit_info = getattr(request.state, "rate_limit_info", None)
    if rate_limit_info:
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info.get("limit", 0))
        response.headers["X-RateLimit-Remaining"] = str(
            rate_limit_info.get("remaining", 0)
        )
        response.headers["X-RateLimit-Reset"] = str(rate_limit_info.get("reset_at", 0))

    return response


# =============================================================================
# Application Factory
# =============================================================================


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override. Uses get_settings() if not provided.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Create FastAPI app
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        openapi_tags=TAGS_METADATA,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
        debug=settings.DEBUG,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "Last-Event-ID",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
    )

    # Add custom middleware (order matters - first added is outermost)
    app.middleware("http")(rate_limit_header_middleware)
    app.middleware("http")(timing_middleware)
    app.middleware("http")(request_id_middleware)

    # Register exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Include routers
    app.include_router(router)
    app.include_router(replay_router)

    # Add root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root to API documentation."""
        return JSONResponse(
            content={
                "name": API_TITLE,
                "version": API_VERSION,
                "docs": "/docs" if settings.DEBUG else "Disabled in production",
                "health": "/api/v1/health",
            }
        )

    # Instrument with Phoenix/OpenTelemetry if available
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,healthz,ready,readyz",
        )
    except ImportError:
        pass  # OpenTelemetry not installed

    logger.info(f"Created {API_TITLE} application")

    return app


# =============================================================================
# Application Instance
# =============================================================================

# Create the default application instance for uvicorn
app = create_app()


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Run the API server using uvicorn.

    This is the main entry point for running the server directly.
    Configuration is loaded from environment variables.
    """
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG,
        workers=1 if settings.is_development else 4,
        timeout_keep_alive=30,
        limit_concurrency=1000,
        limit_max_requests=10000,
    )


if __name__ == "__main__":
    main()
