"""Application configuration with environment variable support."""

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    MOCK = "mock"
    OPENAI = "openai"

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overriden by setting the corresponding
    environment variable (case-insensitive).
    """

    # Database
    database_url: str = "postgresql+asyncpg://nexus:nexus@localhost:5432/nexus"

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

if __name__ == "__main__":
    settings = get_settings()
    print(f"Environment: {settings.environment.value}")
    print(f"Database URL: {settings.database_url}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"LLM Provider: {settings.llm_provider.value}")
    print(f"Worker Count: {settings.worker_count}")
