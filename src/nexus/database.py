"""
Database connection management and repository pattern.

This module provides:
- Async database engine and session management
- JobRepository with all CRUD operations
- Transaction support via context managers
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nexus.config import get_settings
from nexus.models import JobRecord, JobStatus


class Database:
    """
    Async database connection manager.

    Handles engine creation and session lifecycle.
    Use as a singleton via get_database().

    Example:
        db = get_database()
        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)
    """

    def __init__(self, database_url: str | None = None):
        """
        Initialize database with connection URL.

        Args:
            database_url: PostgreSQL connection string.
                          Defaults to settings if not provided.
        """
        settings = get_settings()
        self.database_url = database_url or settings.database_url

        # Create async engine with connection pooling
        self.engine = create_async_engine(
            self.database_url,
            echo=settings.environment.value == "development",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before use
        )

        # Session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Don't expire objects after commit
        )

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic commit/rollback.

        Usage:
            async with db.session() as session:
                # Do work with session
                pass
            # Auto-commits on success, rolls back on exception

        Yields:
            AsyncSession: SQLAlchemy async session
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def health_check(self) -> bool:
        """
        Check database connectivity.

        Returns:
            bool: True if database is accessible
        """
        try:
            async with self.session() as session:
                await session.execute(select(1))
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the database engine and all connections."""
        await self.engine.dispose()


class JobRepository:
    """
    Data access layer for Job records.

    Provides all CRUD operations and specialized queries
    for the job queue. Uses the repository pattern to
    separate data access from business logic.

    Example:
        async with db.session() as session:
            repo = JobRepository(session)

            # Create a job
            job = JobRecord(job_type="llm.completion", input_data={...})
            job = await repo.create(job)

            # Get a job
            job = await repo.get(job_id)

            # List jobs
            jobs = await repo.list_jobs(status=JobStatus.PENDING)
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with a session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    # =========================================================================
    # Basic CRUD Operations
    # =========================================================================
    async def create(self, job: JobRecord) -> JobRecord:
        """
        Create a new job record.

        Args:
            job: JobRecord instance to persist

        Returns:
            JobRecord: Persisted job with generated ID
        """
        self.session.add(job)
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def get(self, job_id: UUID) -> JobRecord | None:
        """
        Get a job by ID.

        Args:
            job_id: UUID of the job

        Returns:
            JobRecord if found, None otherwise
        """
        return await self.session.get(JobRecord, job_id)

    async def update(self, job: JobRecord) -> JobRecord:
        """
        Update an existing job record.

        Args:
            job: JobRecord with updated fields

        Returns:
            JobRecord: Updated job
        """
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def delete(self, job: JobRecord) -> None:
        """
        Delete a job record.

        Args:
            job: JobRecord to delete
        """
        await self.session.delete(job)
        await self.session.flush()

    # =========================================================================
    # Query Operations
    # =========================================================================
    async def list_jobs(
        self,
        status: JobStatus | None = None,
        job_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobRecord]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by job status
            job_type: Filter by job type
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of JobRecord matching criteria
        """
        query = (
            select(JobRecord)
            .order_by(JobRecord.created_at.desc())
        )

        if status:
            query = query.where(JobRecord.status == status.value)
        if job_type:
            query = query.where(JobRecord.job_type == job_type)

        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_by_status(self) -> dict[str, int]:
        """
        Count jobs grouped by status.

        Returns:
            Dict mapping status to count
        """
        query = (
            select(JobRecord.status, func.count(JobRecord.id))
            .group_by(JobRecord.status)
        )
        result = await self.session.execute(query)
        return dict(result.all())  # type: ignore[arg-type]

    async def count_by_type(self) -> dict[str, int]:
        """
        Count jobs grouped by type.

        Returns:
            Dict mapping job_type to count
        """
        query = (
            select(JobRecord.job_type, func.count(JobRecord.id))
            .group_by(JobRecord.job_type)
        )
        result = await self.session.execute(query)
        return dict(result.all())  # type: ignore[arg-type]

    # =========================================================================
    # Statistics (Resume Metrics!)
    # =========================================================================
    async def get_stats(self) -> dict:
        """
        Get aggregate statistics for the job queue.

        These metrics can be cited on your resume!

        Returns:
            Dict with total_jobs, jobs_by_status, jobs_by_type,
            total_tokens, total_cost_usd, avg_duration_ms, success_rate
        """
        # Get counts by status
        jobs_by_status = await self.count_by_status()

        # Get counts by type
        jobs_by_type = await self.count_by_type()

        # Get aggregate metrics
        agg_query = select(
            func.count(JobRecord.id).label("total"),
            func.sum(JobRecord.total_tokens).label("tokens"),
            func.sum(JobRecord.cost_usd).label("cost"),
            func.avg(JobRecord.duration_ms).label("avg_duration"),
        )
        agg_result = await self.session.execute(agg_query)
        agg = agg_result.one()

        # Calculate success rate
        completed = jobs_by_status.get(JobStatus.COMPLETED.value, 0)
        failed = jobs_by_status.get(JobStatus.FAILED.value, 0)
        dead = jobs_by_status.get(JobStatus.DEAD.value, 0)
        total_finished = completed + failed + dead

        success_rate = (
            (completed / total_finished * 100) if total_finished > 0 else 0.0
        )

        # DLQ count
        dlq_count = jobs_by_status.get(JobStatus.DEAD.value, 0)

        return {
            "total_jobs": agg.total or 0,
            "jobs_by_status": jobs_by_status,
            "jobs_by_type": jobs_by_type,
            "total_tokens": agg.tokens or 0,
            "total_cost_usd": float(agg.cost or 0),
            "avg_duration_ms": float(agg.avg_duration) if agg.avg_duration else None,
            "success_rate": round(success_rate, 2),
            "dlq_count": dlq_count,
        }

    # =========================================================================
    # Worker Operations
    # =========================================================================
    async def claim_pending_job(self, worker_id: str) -> JobRecord | None:
        """
        Atomically claim the next pending job for processing.

        Uses SELECT FOR UPDATE SKIP LOCKED to ensure only one
        worker can claim each job, even under high concurrency.

        Args:
            worker_id: Unique identifier for the worker

        Returns:
            JobRecord if a job was claimed, None if queue is empty
        """
        # Find oldest pending job with row-level lock
        query = (
            select(JobRecord)
            .where(JobRecord.status == JobStatus.PENDING.value)
            .order_by(JobRecord.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        result = await self.session.execute(query)
        job = result.scalar_one_or_none()

        if job:
            # Update to running state
            job.status = JobStatus.RUNNING.value  # type: ignore[assignment]
            job.worker_id = worker_id             # type: ignore[assignment]
            job.started_at = datetime.now(UTC)    # type: ignore[assignment]
            job.attempt += 1                      # type: ignore[assignment]
            await self.session.flush()

        return job

    async def get_stale_running_jobs(
        self,
        timeout_seconds: int = 300,
    ) -> list[JobRecord]:
        """
        Find jobs that have been running longer than the timeout.

        These jobs may be stuck due to worker crashes.

        Args:
            timeout_seconds: Consider jobs stale after this many seconds

        Returns:
            List of stale JobRecord
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=timeout_seconds)

        query = (
            select(JobRecord)
            .where(JobRecord.status == JobStatus.RUNNING.value)
            .where(JobRecord.started_at < cutoff)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    # =========================================================================
    # Dead Letter Queue Operations
    # =========================================================================
    async def move_to_dlq(self, job: JobRecord, reason: str) -> JobRecord:
        """
        Move a failed job to the dead letter queue.

        Args:
            job: JobRecord to move
            reason: Reason for moving to DLQ

        Returns:
            Updated JobRecord
        """
        job.status = JobStatus.DEAD.value        # type: ignore[assignment]
        job.moved_to_dlq_at = datetime.now(UTC)  # type: ignore[assignment]
        job.dlq_reason = reason                  # type: ignore[assignment]
        await self.session.flush()
        return job

    async def list_dlq_jobs(self, limit: int = 100) -> list[JobRecord]:
        """
        List jobs in the dead letter queue.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of JobRecord in DLQ
        """
        query = (
            select(JobRecord)
            .where(JobRecord.status == JobStatus.DEAD.value)
            .order_by(JobRecord.moved_to_dlq_at.desc())
            .limit(limit)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def replay_from_dlq(self, job: JobRecord) -> JobRecord:
        """
        Replay a job from the dead letter queue.

        Resets the job to pending state for reprocessing.

        Args:
            job: JobRecord to replay

        Returns:
            Updated JobRecord
        """
        job.status = JobStatus.PENDING.value  # type: ignore[assignment]
        job.attempt = 0                       # type: ignore[assignment]
        job.error = None                      # type: ignore[assignment]
        job.moved_to_dlq_at = None            # type: ignore[assignment]
        job.dlq_reason = None                 # type: ignore[assignment]
        await self.session.flush()
        return job


# =============================================================================
# Module-level Singleton
# =============================================================================
# Global database instance
_db: Database | None = None


def get_database() -> Database:
    """
    Get or create the global database instance.

    Returns:
        Database: Singleton database instance
    """
    global _db
    if _db is None:
        _db = Database()
    return _db


def reset_database() -> None:
    """
    Reset the global database instance.

    Useful for testing to ensure clean state.
    """
    global _db
    _db = None


# =============================================================================
# Quick Test
# =============================================================================
async def _test_database() -> None:
    """Quick test of database functionality."""
    from nexus.models import JobType

    db = get_database()

    # Test health check
    healthy = await db.health_check()
    print(f"Database healthy: {healthy}")

    # Test creating and reading a job
    async with db.session() as session:
        repo = JobRepository(session)

        # Create a job
        job = JobRecord(
            job_type=JobType.LLM_COMPLETION.value,
            input_data={"prompt": "Test prompt", "model": "gpt-4o-mini"},
        )
        job = await repo.create(job)
        print(f"Created job: {job.id}")

        # Read it back
        fetched = await repo.get(job.id)  # type: ignore[arg-type]
        if fetched:
            print(f"Fetched job: {fetched.id}, status: {fetched.status}")

        # List jobs
        jobs = await repo.list_jobs(limit=5)
        print(f"Total jobs in list: {len(jobs)}")

        # Get stats
        stats = await repo.get_stats()
        print(f"Stats: {stats}")

        # Clean up test job
        await repo.delete(job)
        print("Deleted test job")


if __name__ == "__main__":
    import asyncio
    import sys

    from nexus.config import disable_logging

    if "--quiet" in sys.argv:
        import logging
        logging.disable(logging.CRITICAL)  # Silences SQLAlchemy
        disable_logging()                  # Silences structlog

    asyncio.run(_test_database())
