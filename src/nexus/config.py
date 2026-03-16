"""Application configuration with environment variable support."""

from enum import StrEnum
from functools import lru_cache
from typing import Any

import structlog
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Environment(StrEnum):
    """Application environment."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMProvider(StrEnum):
    """Supported LLM providers."""
    MOCK = "mock"
    OPENAI = "openai"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden by setting the corresponding
    environment variable (case-insensitive).
    """

    # Database
    database_url: str = "postgresql+asyncpg://nexus:nexus@localhost:5432/nexus"

    @field_validator("database_url", mode="before")
    @classmethod
    def ensure_async_driver(cls, v: str) -> str:
        """Ensure the database URL uses the asyncpg driver.

        Railway set DATABASE_URL with the 'postgresql://' scheme,
        which makes SQLAlchemy default to psycopg2.
        'postgresql+asyncpg://' is needed for Nexus' async engine,
        so this validator rewrites the prefix automatically.
        """
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Application
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"

    # Worker configuration
    worker_count: int = 3
    job_timeout_seconds: int = 300
    max_retries: int = 3
    poll_interval_seconds: float = 1.0

    # LLM Provider configuration
    llm_provider: LLMProvider = LLMProvider.MOCK
    openai_api_key: str | None = None
    default_model: str = "gpt-4o-mini"

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Metrics
    metrics_port: int = 9090

    # Dead letter queue
    dlq_enabled: bool = True

    # OpenTelemetry
    otel_enabled: bool = True
    otel_service_name: str = "nexus"
    otel_exporter_endpoint: str = "http://localhost:4317"
    otel_insecure: bool = True
    otel_exporter_protocol: str = "grpc"  # "grpc" for Jaeger, "http" for Grafana Cloud
    otel_exporter_headers: str = ""       # "Authorization=Basic <token>" for Grafana Cloud

    @field_validator("otel_enabled", mode="before")
    @classmethod
    def disable_otel_in_testing(cls, v: bool, info: Any) -> bool:
        """Disable tracing in test environment unless explicitly enabled."""
        # If explicitly set via environment variable, respect that
        import os
        if os.environ.get("OTEL_ENABLED") is not None:
            return v

        # Auto-disable in testing
        env = info.data.get("environment", Environment.DEVELOPMENT)
        if env == Environment.TESTING:
            return False
        return v

    # Pydantic v2 configuration using model_config
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore", # Ignore environment variables not defined in Settings
    }

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return self.environment == Environment.TESTING


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()

def configure_logging() -> None:
    """Configure structured logging for worker and API processes."""
    import logging

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set log level
    settings = get_settings()
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper()),
    )

def disable_logging() -> None:
    """Disable all structlog logging for clean module output.

    Call this in __main__ blocks when the --quiet flag is passed.
    Routes all log output to /dev/null instead of the terminal,
    leaving only explicit print() statements visible.

    Example:
        if __name__ == "__main__":
            import asyncio
            import sys
            from nexus.config import disable_logging

            if "--quiet" in sys.argv:
                disable_logging()

            asyncio.run(_test_my_module())
    """
    import os

    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(open(os.devnull, "w")),
        cache_logger_on_first_use=True,
    )

if __name__ == "__main__":
    settings = get_settings()
    print(f"Environment: {settings.environment.value}")
    print(f"Database URL: {settings.database_url}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"LLM Provider: {settings.llm_provider.value}")
    print(f"Worker Count: {settings.worker_count}")
