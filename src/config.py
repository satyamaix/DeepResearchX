"""
Application Configuration for DRX Deep Research System.

Uses pydantic-settings for type-safe configuration with environment
variable loading, validation, and sensible defaults.

Configuration is loaded from:
1. Environment variables (highest priority)
2. .env file in project root
3. Default values defined here (lowest priority)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables using
    the same name (case-insensitive). Nested settings use double
    underscore separation (e.g., POSTGRES__HOST).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Application Settings
    # =========================================================================

    APP_NAME: str = Field(
        default="DRX Deep Research",
        description="Application name for logging and identification",
    )

    APP_ENV: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Current application environment",
    )

    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, development features)",
    )

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for the application",
    )

    SECRET_KEY: SecretStr = Field(
        default=SecretStr("change-me-in-production-use-secure-random-key"),
        description="Secret key for cryptographic operations",
    )

    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="CORS allowed origins (comma-separated)",
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated origins string into list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    # =========================================================================
    # Database Settings (PostgreSQL)
    # =========================================================================

    DATABASE_URL: PostgresDsn = Field(
        default="postgresql://drx:drx_password@localhost:5432/drx",
        description="PostgreSQL connection URL",
    )

    DB_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Database connection pool size",
    )

    DB_POOL_MIN_SIZE: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Minimum database pool size for async connections",
    )

    DB_POOL_MAX_SIZE: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum database pool size for async connections",
    )

    DB_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum overflow connections beyond pool size",
    )

    DB_POOL_TIMEOUT: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Timeout in seconds for acquiring a connection from pool",
    )

    DB_COMMAND_TIMEOUT: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Command execution timeout in seconds",
    )

    DB_POOL_RECYCLE: int = Field(
        default=1800,
        ge=300,
        le=7200,
        description="Seconds before a connection is recycled",
    )

    DB_STATEMENT_CACHE_SIZE: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="Prepared statement cache size (0 to disable)",
    )

    DB_ECHO: bool = Field(
        default=False,
        description="Echo all SQL statements (for debugging)",
    )

    # =========================================================================
    # Redis Settings
    # =========================================================================

    REDIS_URL: RedisDsn = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    REDIS_MAX_CONNECTIONS: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum Redis connections in pool",
    )

    REDIS_SOCKET_TIMEOUT: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Redis socket timeout in seconds",
    )

    REDIS_RETRY_ON_TIMEOUT: bool = Field(
        default=True,
        description="Retry operations on timeout",
    )

    # =========================================================================
    # OpenRouter Settings
    # =========================================================================

    OPENROUTER_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        description="OpenRouter API key",
    )

    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    OPENROUTER_TIMEOUT: float = Field(
        default=120.0,
        ge=10.0,
        le=600.0,
        description="API request timeout in seconds",
    )

    OPENROUTER_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed requests",
    )

    # Model configuration
    DEFAULT_MODEL: str = Field(
        default="google/gemini-3-flash-preview",
        description="Default model for general tasks",
    )

    REASONING_MODEL: str = Field(
        default="deepseek/deepseek-r1",
        description="Model for complex reasoning tasks (planner, critic)",
    )

    SYNTHESIS_MODEL: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="Model for synthesis and report generation",
    )

    DEFAULT_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for model inference",
    )

    MAX_TOKENS_PER_REQUEST: int = Field(
        default=100000,
        ge=1000,
        le=500000,
        description="Maximum tokens per single API request",
    )

    # =========================================================================
    # Web Search Settings (OpenRouter Native - Primary)
    # =========================================================================

    SEARCH_ENGINE: Literal["native", "exa", "auto"] = Field(
        default="native",
        description=(
            "Web search engine: 'native' (Anthropic/OpenAI/Perplexity/xAI - no extra cost), "
            "'exa' ($0.004/result), or 'auto' (let OpenRouter decide)"
        ),
    )

    SEARCH_MODEL: str = Field(
        default="google/gemini-3-flash-preview",
        description=(
            "Model for web search. Add :online suffix for real-time web search. "
            "Any model can use :online variant for web search capability."
        ),
    )

    SEARCH_MAX_RESULTS: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum results per search query",
    )

    SEARCH_TIMEOUT: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Search request timeout in seconds",
    )

    # =========================================================================
    # Tavily Search Settings (Fallback - Optional)
    # =========================================================================

    TAVILY_ENABLED: bool = Field(
        default=False,
        description="Enable Tavily as fallback search (requires API key)",
    )

    TAVILY_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        description="Tavily search API key (optional, for fallback)",
    )

    TAVILY_SEARCH_DEPTH: Literal["basic", "advanced"] = Field(
        default="advanced",
        description="Tavily search depth (basic is faster, advanced is more thorough)",
    )

    TAVILY_MAX_RESULTS: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum results per Tavily search",
    )

    TAVILY_INCLUDE_ANSWER: bool = Field(
        default=True,
        description="Include AI-generated answer in Tavily results",
    )

    TAVILY_INCLUDE_RAW_CONTENT: bool = Field(
        default=False,
        description="Include raw HTML content in Tavily results",
    )

    # =========================================================================
    # Phoenix Observability Settings
    # =========================================================================

    PHOENIX_ENABLED: bool = Field(
        default=True,
        description="Enable Phoenix observability",
    )

    PHOENIX_COLLECTOR_ENDPOINT: str = Field(
        default="http://localhost:6006/v1/traces",
        description="Phoenix OTLP collector endpoint (HTTP format: http://host:port/v1/traces)",
    )

    PHOENIX_PROJECT_NAME: str = Field(
        default="drx-research",
        description="Project name in Phoenix",
    )

    PHOENIX_PROTOCOL: Literal["http/protobuf", "grpc"] = Field(
        default="http/protobuf",
        description="Phoenix OTLP protocol: 'http/protobuf' (recommended, no TLS issues) or 'grpc'",
    )

    # =========================================================================
    # Research Defaults
    # =========================================================================

    MAX_RESEARCH_ITERATIONS: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Default maximum research iterations",
    )

    TOKEN_BUDGET_PER_SESSION: int = Field(
        default=500000,
        ge=10000,
        le=2000000,
        description="Default token budget per research session",
    )

    MAX_SOURCES_PER_QUERY: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Default maximum sources to retrieve",
    )

    MIN_COVERAGE_SCORE: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum coverage score to complete research",
    )

    DEFAULT_RESEARCH_TIMEOUT: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Default timeout in seconds for research sessions",
    )

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting",
    )

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Maximum requests per minute per client",
    )

    RATE_LIMIT_TOKENS_PER_MINUTE: int = Field(
        default=1000000,
        ge=10000,
        le=10000000,
        description="Maximum tokens per minute per client",
    )

    # =========================================================================
    # API Server Settings
    # =========================================================================

    API_HOST: str = Field(
        default="0.0.0.0",
        description="API server host",
    )

    API_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port",
    )

    API_WORKERS: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of API worker processes",
    )

    API_PREFIX: str = Field(
        default="/api/v1",
        description="API route prefix",
    )

    # =========================================================================
    # Celery Worker Settings
    # =========================================================================

    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL",
    )

    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend URL",
    )

    CELERY_TASK_TIMEOUT: int = Field(
        default=3600,
        ge=60,
        le=7200,
        description="Maximum task execution time in seconds",
    )

    # =========================================================================
    # Circuit Breaker Settings
    # =========================================================================

    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of failures before circuit opens",
    )

    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Seconds before attempting recovery",
    )

    CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max calls in half-open state",
    )

    # =========================================================================
    # Guardrails Settings
    # =========================================================================

    GUARDRAILS_ENABLED: bool = Field(
        default=True,
        description="Enable NeMo Guardrails for content safety",
    )

    GUARDRAILS_CONFIG_PATH: str = Field(
        default="config/guardrails",
        description="Path to guardrails configuration directory",
    )

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.APP_ENV == "development"

    @property
    def database_url_str(self) -> str:
        """Get database URL as string."""
        return str(self.DATABASE_URL)

    @property
    def database_url_async(self) -> str:
        """Get async database URL for asyncpg."""
        url = str(self.DATABASE_URL)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    @property
    def database_url_sync(self) -> str:
        """Get sync database URL for psycopg."""
        url = str(self.DATABASE_URL)
        if "+asyncpg" in url:
            return url.replace("+asyncpg", "", 1)
        return url

    @property
    def redis_url_str(self) -> str:
        """Get Redis URL as string."""
        return str(self.REDIS_URL)

    @property
    def openrouter_api_key_value(self) -> str:
        """Get the actual OpenRouter API key value."""
        return self.OPENROUTER_API_KEY.get_secret_value()

    @property
    def tavily_api_key_value(self) -> str:
        """Get the actual Tavily API key value."""
        return self.TAVILY_API_KEY.get_secret_value()

    @property
    def secret_key_value(self) -> str:
        """Get the actual secret key value."""
        return self.SECRET_KEY.get_secret_value()


# =============================================================================
# Settings Singleton
# =============================================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached application settings.

    This function is cached to ensure settings are only loaded once
    from environment/files. Use dependency injection in FastAPI:

        from fastapi import Depends
        from src.config import Settings, get_settings

        @app.get("/")
        async def root(settings: Settings = Depends(get_settings)):
            return {"debug": settings.DEBUG}

    To refresh settings (e.g., in tests), clear the cache:

        get_settings.cache_clear()

    Returns:
        Settings instance with loaded configuration
    """
    return Settings()


def get_settings_uncached() -> Settings:
    """
    Get fresh settings without caching.

    Useful for testing or when dynamic reloading is needed.
    Note: This bypasses the cache and reads from env/files each time.

    Returns:
        Fresh Settings instance
    """
    return Settings()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_database_url() -> str:
    """Get the database connection URL as string."""
    return get_settings().database_url_str


def get_redis_url() -> str:
    """Get the Redis connection URL as string."""
    return get_settings().redis_url_str


# =============================================================================
# Model Configuration Helper
# =============================================================================


class ModelConfig:
    """Helper class for getting model configurations based on task type."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def get_model_for_task(
        self,
        task_type: Literal["general", "reasoning", "synthesis"],
    ) -> str:
        """
        Get the appropriate model for a task type.

        Args:
            task_type: Type of task (general, reasoning, synthesis)

        Returns:
            Model identifier string
        """
        match task_type:
            case "reasoning":
                return self.settings.REASONING_MODEL
            case "synthesis":
                return self.settings.SYNTHESIS_MODEL
            case _:
                return self.settings.DEFAULT_MODEL

    def get_temperature_for_task(
        self,
        task_type: Literal["general", "reasoning", "synthesis"],
    ) -> float:
        """
        Get appropriate temperature for a task type.

        Args:
            task_type: Type of task

        Returns:
            Temperature value
        """
        match task_type:
            case "reasoning":
                return 0.3  # Lower for more deterministic reasoning
            case "synthesis":
                return 0.7  # Moderate for creative synthesis
            case _:
                return self.settings.DEFAULT_TEMPERATURE


# =============================================================================
# Environment Validation
# =============================================================================


def validate_required_settings() -> list[str]:
    """
    Validate that required settings are properly configured.

    Returns:
        List of validation error messages (empty if all valid)
    """
    errors: list[str] = []
    settings = get_settings()

    # Check required API keys in production
    if settings.is_production:
        if not settings.openrouter_api_key_value:
            errors.append("OPENROUTER_API_KEY is required in production")

        # Tavily is optional - only validate if enabled
        if settings.TAVILY_ENABLED and not settings.tavily_api_key_value:
            errors.append("TAVILY_API_KEY is required when TAVILY_ENABLED=true")

        if settings.secret_key_value == "change-me-in-production-use-secure-random-key":
            errors.append("SECRET_KEY must be changed in production")

    # Validate database URL format
    try:
        _ = settings.database_url_async
    except Exception as e:
        errors.append(f"Invalid DATABASE_URL: {e}")

    return errors


def print_settings_summary() -> None:
    """Print a summary of current settings (for debugging/startup)."""
    settings = get_settings()

    # Extract host info from URLs safely
    db_info = str(settings.DATABASE_URL).split("@")[-1] if "@" in str(settings.DATABASE_URL) else str(settings.DATABASE_URL)

    print("\n" + "=" * 60)
    print("DRX Configuration Summary")
    print("=" * 60)
    print(f"  Environment: {settings.APP_ENV}")
    print(f"  Debug Mode:  {settings.DEBUG}")
    print(f"  Log Level:   {settings.LOG_LEVEL}")
    print(f"  API Server:  {settings.API_HOST}:{settings.API_PORT}")
    print("-" * 60)
    print(f"  Database:    {db_info}")
    print(f"  Redis:       {settings.redis_url_str}")
    print("-" * 60)
    print(f"  Default Model:   {settings.DEFAULT_MODEL}")
    print(f"  Reasoning Model: {settings.REASONING_MODEL}")
    print(f"  Synthesis Model: {settings.SYNTHESIS_MODEL}")
    print("-" * 60)
    print(f"  Max Iterations:  {settings.MAX_RESEARCH_ITERATIONS}")
    print(f"  Token Budget:    {settings.TOKEN_BUDGET_PER_SESSION:,}")
    print(f"  Phoenix:         {'Enabled' if settings.PHOENIX_ENABLED else 'Disabled'}")
    print(f"  Guardrails:      {'Enabled' if settings.GUARDRAILS_ENABLED else 'Disabled'}")
    print("=" * 60 + "\n")


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    "Settings",
    "get_settings",
    "get_settings_uncached",
    "get_database_url",
    "get_redis_url",
    "ModelConfig",
    "validate_required_settings",
    "print_settings_summary",
]
