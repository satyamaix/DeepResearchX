"""DRX API Module.

Provides FastAPI routes, dependencies, and SSE streaming
for the deep research REST API.
"""

from src.api.dependencies import (
    # Database
    get_db,
    DatabaseDep,
    # Redis
    get_redis,
    RedisDep,
    # Orchestrator
    ResearchOrchestrator,
    get_orchestrator,
    OrchestratorDep,
    # Authentication
    User,
    get_current_user,
    get_optional_user,
    CurrentUserDep,
    OptionalUserDep,
    # Rate Limiting
    RateLimiter,
    RateLimitDependency,
    rate_limit_standard,
    rate_limit_heavy,
    rate_limit_streaming,
    # Settings
    SettingsDep,
)

from src.api.streaming import (
    # Enums
    StreamEventType,
    # Models
    StreamEvent,
    SSEConfig,
    # Manager
    SSEManager,
    # Factory
    create_sse_response,
    # Event helpers
    create_start_event,
    create_thought_event,
    create_content_delta_event,
    create_tool_event,
    create_complete_event,
    create_error_event,
)

from src.api.routes import (
    router,
    # Request/Response Models
    SteerabilityConfig,
    ResearchConfig,
    ResearchRequest,
    InteractionResponse,
    InteractionListResponse,
    HealthResponse,
    ErrorResponse,
)

from src.api.main import (
    create_app,
    app,
    main,
    API_TITLE,
    API_VERSION,
)

__all__ = [
    # Application
    "create_app",
    "app",
    "main",
    "API_TITLE",
    "API_VERSION",
    # Router
    "router",
    # Dependencies
    "get_db",
    "DatabaseDep",
    "get_redis",
    "RedisDep",
    "ResearchOrchestrator",
    "get_orchestrator",
    "OrchestratorDep",
    "User",
    "get_current_user",
    "get_optional_user",
    "CurrentUserDep",
    "OptionalUserDep",
    "RateLimiter",
    "RateLimitDependency",
    "rate_limit_standard",
    "rate_limit_heavy",
    "rate_limit_streaming",
    "SettingsDep",
    # Streaming
    "StreamEventType",
    "StreamEvent",
    "SSEConfig",
    "SSEManager",
    "create_sse_response",
    "create_start_event",
    "create_thought_event",
    "create_content_delta_event",
    "create_tool_event",
    "create_complete_event",
    "create_error_event",
    # Models
    "SteerabilityConfig",
    "ResearchConfig",
    "ResearchRequest",
    "InteractionResponse",
    "InteractionListResponse",
    "HealthResponse",
    "ErrorResponse",
]
