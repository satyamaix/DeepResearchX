"""
Celery Worker for DRX Deep Research System.

Handles asynchronous task processing for long-running research operations.
"""

import asyncio
import logging
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# =============================================================================
# Phoenix Tracing Initialization
# =============================================================================


def _init_phoenix_tracing():
    """Initialize Phoenix tracing for the worker process."""
    if settings.PHOENIX_COLLECTOR_ENDPOINT:
        try:
            from src.observability import setup_phoenix
            setup_phoenix()
            logger.info(
                f"Phoenix tracing initialized for worker: {settings.PHOENIX_COLLECTOR_ENDPOINT}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Phoenix tracing in worker: {e}")


def _shutdown_phoenix_tracing():
    """Shutdown Phoenix tracing for the worker process."""
    try:
        from src.observability import shutdown_phoenix
        shutdown_phoenix()
        logger.info("Phoenix tracing shutdown for worker")
    except Exception as e:
        logger.warning(f"Failed to shutdown Phoenix tracing in worker: {e}")


@worker_process_init.connect
def init_worker_process(**kwargs):
    """Called when a worker process starts."""
    _init_phoenix_tracing()


@worker_process_shutdown.connect
def shutdown_worker_process(**kwargs):
    """Called when a worker process shuts down."""
    _shutdown_phoenix_tracing()


# =============================================================================
# Celery Application
# =============================================================================

# Create Celery application
app = Celery(
    "drx_worker",
    broker=str(settings.CELERY_BROKER_URL),
    backend=str(settings.CELERY_RESULT_BACKEND),
)

# Configure Celery
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIMEOUT,
    task_soft_time_limit=settings.CELERY_TASK_TIMEOUT - 300,  # Soft limit 5 min before hard limit
    worker_prefetch_multiplier=1,
)


# =============================================================================
# Research Execution Task
# =============================================================================


@app.task(
    bind=True,
    name="research.execute",
    max_retries=3,
    soft_time_limit=3300,
    time_limit=3600,
)
def execute_research(
    self,
    session_id: str,
    query: str,
    config: dict | None = None,
):
    """
    Execute a research workflow asynchronously.

    Args:
        session_id: Unique identifier for the research session
        query: The research query to process
        config: Optional configuration overrides including:
            - max_iterations: Maximum research iterations
            - token_budget: Token limit for the session
            - cost_budget: Optional cost limit in USD
            - steerability: Steerability parameters

    Returns:
        Dict with session_id, status, and results summary
    """
    # Run the async execution in an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _execute_research_async(self, session_id, query, config)
        )
    finally:
        loop.close()


async def _execute_research_async(
    task,
    session_id: str,
    query: str,
    config: dict | None = None,
):
    """
    Async implementation of research execution.

    Handles the actual workflow execution with progress streaming,
    budget tracking, and cancellation support.
    """
    from src.orchestrator.workflow import ResearchOrchestrator, StreamEventType
    from src.orchestrator.budget import BudgetTracker, BudgetExceededError
    from src.services.progress_publisher import (
        create_progress_publisher,
        is_cancelled,
        clear_cancellation,
    )

    config = config or {}
    publisher = await create_progress_publisher(session_id)

    # Initialize budget tracker
    budget = BudgetTracker(
        token_budget=config.get("token_budget", settings.TOKEN_BUDGET_PER_SESSION),
        cost_budget=config.get("cost_budget"),
    )

    try:
        # Publish start
        await publisher.publish_status(
            "running",
            f"Starting research: {query[:100]}...",
        )

        # Create orchestrator
        orchestrator = ResearchOrchestrator()
        await orchestrator.initialize()

        # Stream execution events
        final_state = None
        async for event in orchestrator.run(
            query=query,
            config={
                "max_iterations": config.get(
                    "max_iterations",
                    settings.MAX_RESEARCH_ITERATIONS,
                ),
                "token_budget": budget.token_budget,
            },
            session_id=session_id,
            steerability=config.get("steerability"),
        ):
            # Check for cancellation
            if await is_cancelled(session_id):
                await publisher.publish_status(
                    "cancelled",
                    "Research cancelled by user",
                )
                await clear_cancellation(session_id)
                return {
                    "session_id": session_id,
                    "status": "cancelled",
                    "message": "Research cancelled by user",
                }

            # Publish event
            await publisher.publish(event)

            # Track completion
            if event.event_type == StreamEventType.WORKFLOW_COMPLETED:
                final_state = event.data
            elif event.event_type == StreamEventType.WORKFLOW_FAILED:
                raise Exception(event.error or "Workflow failed")

        # Success
        await publisher.publish_status(
            "completed",
            "Research completed successfully",
            {
                "findings_count": (
                    final_state.get("findings_count", 0) if final_state else 0
                ),
                "citations_count": (
                    final_state.get("citations_count", 0) if final_state else 0
                ),
                "tokens_used": (
                    final_state.get("tokens_used", 0) if final_state else 0
                ),
            },
        )

        return {
            "session_id": session_id,
            "status": "completed",
            "final_report": (
                final_state.get("final_report") if final_state else None
            ),
            "findings_count": (
                final_state.get("findings_count", 0) if final_state else 0
            ),
            "citations_count": (
                final_state.get("citations_count", 0) if final_state else 0
            ),
            "tokens_used": (
                final_state.get("tokens_used", 0) if final_state else 0
            ),
        }

    except BudgetExceededError as e:
        await publisher.publish_status("failed", f"Budget exceeded: {e}")
        return {
            "session_id": session_id,
            "status": "failed",
            "error": str(e),
            "error_type": "budget_exceeded",
        }

    except Exception as e:
        logger.exception(f"Research execution failed for session {session_id}")
        await publisher.publish_status("failed", f"Research failed: {e}")

        # Retry logic
        if task.request.retries < task.max_retries:
            raise task.retry(exc=e, countdown=60 * (task.request.retries + 1))

        return {
            "session_id": session_id,
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# Health Check Task
# =============================================================================


@app.task(bind=True, name="health.ping")
def ping(self):
    """Health check task."""
    return "pong"
