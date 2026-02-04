"""
Tests for database layer and repository pattern.

These tests verify that our data access layer correctly
persists and retrieves job records.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from nexus.database import Database, JobRepository, get_database, reset_database
from nexus.models import JobRecord, JobStatus, JobType


class TestDatabase:
    """Tests for Database class."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, db: Database):
        """Test health check returns True when database is accessible."""
        result = await db.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_session_auto_commits(self, db: Database):
        """Test that session auto-commits on success."""
        job_id = None

        async with db.session() as session:
            repo = JobRepository(session)
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "test"},
            )
            job = await repo.create(job)
            job_id = job.id

        # Verify job was committed
        async with db.session() as session:
            repo = JobRepository(session)
            fetched = await repo.get(job_id)
            assert fetched is not None
            assert fetched.id == job_id

    @pytest.mark.asyncio
    async def test_session_rollback_on_error(self, db: Database):
        """Test that session rolls back on exception."""
        job_id = uuid4()

        try:
            async with db.session() as session:
                repo = JobRepository(session)
                job = JobRecord(
                    id=job_id,
                    job_type=JobType.LLM_COMPLETION.value,
                    input_data={"prompt": "test"},
                )
                await repo.create(job)
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify job was NOT committed
        async with db.session() as session:
            repo = JobRepository(session)
            fetched = await repo.get(job_id)
            assert fetched is None


class TestJobRepository:
    """Tests for JobRepository class."""

    # =========================================================================
    # CRUD Tests
    # =========================================================================
    @pytest.mark.asyncio
    async def test_create_job(self, repo: JobRepository, completion_input: dict):
        """Test creating a new job."""
        job = JobRecord(
            job_type=JobType.LLM_COMPLETION.value,
            input_data=completion_input,
        )

        result = await repo.create(job)

        assert result.id is not None
        assert result.job_type == JobType.LLM_COMPLETION.value
        assert result.input_data == completion_input
        assert result.status == JobStatus.PENDING.value
        assert result.attempt == 0
        assert result.created_at is not None

    @pytest.mark.asyncio
    async def test_get_existing_job(self, repo: JobRepository, sample_job: JobRecord):
        """Test getting an existing job by ID."""
        result = await repo.get(sample_job.id)

        assert result is not None
        assert result.id == sample_job.id
        assert result.job_type == sample_job.job_type

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, repo: JobRepository):
        """Test getting a job that doesn't exist."""
        result = await repo.get(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_update_job(self, repo: JobRepository, sample_job: JobRecord):
        """Test updating a job."""
        sample_job.status = JobStatus.RUNNING.value
        sample_job.worker_id = "test-worker"
        sample_job.started_at = datetime.now(UTC)

        result = await repo.update(sample_job)

        assert result.status == JobStatus.RUNNING.value
        assert result.worker_id == "test-worker"
        assert result.started_at is not None

    @pytest.mark.asyncio
    async def test_delete_job(self, repo: JobRepository, completion_input: dict):
        """Test deleting a job."""
        # Create a job
        job = JobRecord(
            job_type=JobType.LLM_COMPLETION.value,
            input_data=completion_input,
        )
        job = await repo.create(job)
        job_id = job.id

        # Delete it
        await repo.delete(job)

        # Verify it's gone
        result = await repo.get(job_id)
        assert result is None

    # =========================================================================
    # Query Tests
    # =========================================================================
    @pytest.mark.asyncio
    async def test_list_jobs_no_filter(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test listing jobs without filters."""
        result = await repo.list_jobs()

        assert len(result) == len(multiple_jobs)

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_status(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test listing jobs filtered by status."""
        result = await repo.list_jobs(status=JobStatus.PENDING)

        assert len(result) == 1
        assert all(j.status == JobStatus.PENDING.value for j in result)

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_type(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test listing jobs filtered by type."""
        result = await repo.list_jobs(job_type=JobType.LLM_BATCH.value)

        assert len(result) == 1
        assert all(j.job_type == JobType.LLM_BATCH.value for j in result)

    @pytest.mark.asyncio
    async def test_list_jobs_with_limit(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test listing jobs with limit."""
        result = await repo.list_jobs(limit=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_jobs_with_offset(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test listing jobs with offset."""
        all_jobs = await repo.list_jobs()
        offset_jobs = await repo.list_jobs(offset=2)

        assert len(offset_jobs) == len(all_jobs) - 2

    @pytest.mark.asyncio
    async def test_list_jobs_ordered_by_created_at_desc(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test that jobs are ordered by created_at descending."""
        result = await repo.list_jobs()

        for i in range(len(result) - 1):
            assert result[i].created_at >= result[i + 1].created_at

    # =========================================================================
    # Statistics Tests
    # =========================================================================
    @pytest.mark.asyncio
    async def test_count_by_status(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test counting jobs by status."""
        result = await repo.count_by_status()

        assert result[JobStatus.PENDING.value] == 1
        assert result[JobStatus.RUNNING.value] == 1
        assert result[JobStatus.COMPLETED.value] == 1
        assert result[JobStatus.FAILED.value] == 1
        assert result[JobStatus.DEAD.value] == 1

    @pytest.mark.asyncio
    async def test_count_by_type(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test counting jobs by type."""
        result = await repo.count_by_type()

        # 4 completion jobs, 1 batch job
        assert result[JobType.LLM_COMPLETION.value] == 4
        assert result[JobType.LLM_BATCH.value] == 1

    @pytest.mark.asyncio
    async def test_get_stats(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test getting aggregate statistics."""
        result = await repo.get_stats()

        assert result["total_jobs"] == 5
        assert result["jobs_by_status"][JobStatus.COMPLETED.value] == 1
        assert result["jobs_by_type"][JobType.LLM_BATCH.value] == 1
        assert result["total_tokens"] == 150  # From completed job
        assert result["total_cost_usd"] == 0.001
        # Success rate: 1 completed / (1 completed + 1 failed + 1 dead) = 33.33%
        assert result["success_rate"] == pytest.approx(33.33, rel=0.1)
        assert result["dlq_count"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_empty_database(self, repo: JobRepository):
        """Test stats with no jobs."""
        result = await repo.get_stats()

        assert result["total_jobs"] == 0
        assert result["total_tokens"] == 0
        assert result["success_rate"] == 0.0

    # =========================================================================
    # Worker Operations Tests
    # =========================================================================
    @pytest.mark.asyncio
    async def test_claim_pending_job(
        self,
        repo: JobRepository,
        completion_input: dict,
    ):
        """Test claiming a pending job."""
        # Create a pending job
        job = JobRecord(
            job_type=JobType.LLM_COMPLETION.value,
            input_data=completion_input,
        )
        await repo.create(job)

        # Claim it
        claimed = await repo.claim_pending_job("test-worker")

        assert claimed is not None
        assert claimed.status == JobStatus.RUNNING.value
        assert claimed.worker_id == "test-worker"
        assert claimed.started_at is not None
        assert claimed.attempt == 1

    @pytest.mark.asyncio
    async def test_claim_pending_job_empty_queue(self, repo: JobRepository):
        """Test claiming when no pending jobs exist."""
        result = await repo.claim_pending_job("test-worker")

        assert result is None

    @pytest.mark.asyncio
    async def test_claim_pending_job_fifo_order(
        self,
        repo: JobRepository,
        completion_input: dict,
    ):
        """Test that jobs are claimed in FIFO order."""
        # Create jobs with specific order
        job1 = JobRecord(
            job_type=JobType.LLM_COMPLETION.value,
            input_data=completion_input,
        )
        job1 = await repo.create(job1)

        job2 = JobRecord(
            job_type=JobType.LLM_COMPLETION.value,
            input_data=completion_input,
        )
        job2 = await repo.create(job2)

        # First claim should get job1 (created first)
        claimed = await repo.claim_pending_job("worker-1")
        assert claimed.id == job1.id

    # =========================================================================
    # Dead Letter Queue Tests
    # =========================================================================
    @pytest.mark.asyncio
    async def test_move_to_dlq(
        self,
        repo: JobRepository,
        sample_job: JobRecord,
    ):
        """Test moving a job to the dead letter queue."""
        result = await repo.move_to_dlq(sample_job, "Max retries exceeded")

        assert result.status == JobStatus.DEAD.value
        assert result.dlq_reason == "Max retries exceeded"
        assert result.moved_to_dlq_at is not None

    @pytest.mark.asyncio
    async def test_list_dlq_jobs(
        self,
        repo: JobRepository,
        multiple_jobs: list[JobRecord],
    ):
        """Test listing jobs in the dead letter queue."""
        result = await repo.list_dlq_jobs()

        assert len(result) == 1
        assert all(j.status == JobStatus.DEAD.value for j in result)

    @pytest.mark.asyncio
    async def test_replay_from_dlq(
        self,
        repo: JobRepository,
        completion_input: dict,
    ):
        """Test replaying a job from the dead letter queue."""
        # Create and move to DLQ
        job = JobRecord(
            job_type=JobType.LLM_COMPLETION.value,
            input_data=completion_input,
            status=JobStatus.DEAD.value,
            dlq_reason="Test",
            error={"message": "Test error"},
        )
        job = await repo.create(job)

        # Replay
        result = await repo.replay_from_dlq(job)

        assert result.status == JobStatus.PENDING.value
        assert result.attempt == 0
        assert result.error is None
        assert result.dlq_reason is None
        assert result.moved_to_dlq_at is None


class TestDatabaseSingleton:
    """Tests for database singleton pattern."""

    def test_get_database_returns_same_instance(self):
        """Test that get_database returns singleton."""
        reset_database()

        db1 = get_database()
        db2 = get_database()

        assert db1 is db2

        reset_database()

    def test_reset_database_clears_singleton(self):
        """Test that reset_database clears the singleton."""
        db1 = get_database()
        reset_database()
        db2 = get_database()

        assert db1 is not db2

        reset_database()
