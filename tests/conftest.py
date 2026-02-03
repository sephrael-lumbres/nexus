"""
Pytest configuration and fixtures for Nexus tests.

This module provides:
- Database fixtures with automatic cleanup
- Test data factories
- Async test support
"""

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.config import Settings, get_settings
from nexus.database import Database, JobRepository, reset_database
from nexus.models import JobRecord, JobStatus, JobType


# =============================================================================
# Pytest Configuration
# =============================================================================
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# =============================================================================
# Event Loop Fixture
# =============================================================================
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create an event loop for the test session.

    This is required for pytest-asyncio to work properly
    with session-scoped async fixtures.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Settings Fixtures
# =============================================================================
@pytest.fixture
def test_settings() -> Settings:
    """
    Get test settings.

    Uses the same database but could be configured
    to use a separate test database.
    """
    return get_settings()


# =============================================================================
# Database Fixtures
# =============================================================================
@pytest_asyncio.fixture
async def db() -> AsyncGenerator[Database, None]:
    """
    Get a database instance for testing.

    Resets the singleton to ensure clean state.
    """
    reset_database()
    database = Database()
    yield database
    await database.close()
    reset_database()


@pytest_asyncio.fixture
async def session(db: Database) -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session for testing.

    Each test gets its own session with automatic rollback.
    """
    async with db.session() as session:
        yield session


@pytest_asyncio.fixture
async def repo(session: AsyncSession) -> JobRepository:
    """Get a JobRepository instance for testing."""
    return JobRepository(session)


# =============================================================================
# Test Data Fixtures
# =============================================================================
@pytest.fixture
def completion_input() -> dict:
    """Sample input data for a completion job."""
    return {
        "prompt": "What is the meaning of life?",
        "model": "gpt-4o-mini",
        "max_tokens": 500,
        "temperature": 0.7,
    }


@pytest.fixture
def batch_input() -> dict:
    """Sample input data for a batch job."""
    return {
        "items": [
            {"id": "q1", "prompt": "What is Python?"},
            {"id": "q2", "prompt": "What is FastAPI?"},
            {"id": "q3", "prompt": "What is Redis?"},
        ],
        "model": "gpt-4o-mini",
        "max_tokens": 200,
        "temperature": 0.7,
    }


@pytest_asyncio.fixture
async def sample_job(repo: JobRepository, completion_input: dict) -> AsyncGenerator[JobRecord, None]:
    """
    Create a sample job for testing.

    The job is persisted to the database and automatically cleaned up by the cleanup_jobs fixture.
    """
    job = JobRecord(
        job_type=JobType.LLM_COMPLETION.value,
        input_data=completion_input,
    )
    job = await repo.create(job)
    yield job


@pytest_asyncio.fixture
async def multiple_jobs(
    repo: JobRepository,
    completion_input: dict,
    batch_input: dict,
) -> AsyncGenerator[list[JobRecord], None]:
    """Create multiple jobs with different statuses for testing."""
    jobs = []

    # Pending completion job
    job1 = JobRecord(
        job_type=JobType.LLM_COMPLETION.value,
        input_data=completion_input,
        status=JobStatus.PENDING.value,
    )
    jobs.append(await repo.create(job1))

    # Running completion job
    job2 = JobRecord(
        job_type=JobType.LLM_COMPLETION.value,
        input_data=completion_input,
        status=JobStatus.RUNNING.value,
        worker_id="test-worker-1",
    )
    jobs.append(await repo.create(job2))

    # Completed batch job
    job3 = JobRecord(
        job_type=JobType.LLM_BATCH.value,
        input_data=batch_input,
        status=JobStatus.COMPLETED.value,
        total_tokens=150,
        cost_usd=0.001,
        duration_ms=250,
    )
    jobs.append(await repo.create(job3))

    # Failed job
    job4 = JobRecord(
        job_type=JobType.LLM_COMPLETION.value,
        input_data=completion_input,
        status=JobStatus.FAILED.value,
        error={"type": "APIError", "message": "Rate limited"},
        attempt=3,
    )
    jobs.append(await repo.create(job4))

    # Dead job (in DLQ)
    job5 = JobRecord(
        job_type=JobType.LLM_COMPLETION.value,
        input_data=completion_input,
        status=JobStatus.DEAD.value,
        dlq_reason="Max retries exceeded",
    )
    jobs.append(await repo.create(job5))

    yield jobs


# =============================================================================
# Cleanup Fixtures
# =============================================================================
@pytest_asyncio.fixture(autouse=True)
async def cleanup_jobs(db: Database):
    """
    Clean up any jobs created during tests.

    Runs after each test to ensure clean state.
    """
    yield
    # Cleanup after test
    async with db.session() as session:
        await session.execute(text("DELETE FROM jobs"))


# =============================================================================
# Helper Functions
# =============================================================================
def make_job(
    job_type: JobType = JobType.LLM_COMPLETION,
    status: JobStatus = JobStatus.PENDING,
    **kwargs,
) -> JobRecord:
    """
    Factory function to create JobRecord instances for testing.

    Args:
        job_type: Type of job
        status: Initial status
        **kwargs: Additional fields to set

    Returns:
        JobRecord instance (not persisted)
    """
    defaults = {
        "job_type": job_type.value,
        "input_data": {"prompt": "Test", "model": "gpt-4o-mini"},
        "status": status.value,
    }
    defaults.update(kwargs)
    return JobRecord(**defaults)
