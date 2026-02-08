"""Integration tests for database and queue working together.

These tests verify the full job lifecycle from submission
through processing and completion.
"""

import pytest
import pytest_asyncio

from nexus.database import Database, JobRepository
from nexus.models import JobRecord, JobStatus, JobType
from nexus.queue import JobQueue


class TestDatabaseQueueIntegration:
    """Tests for database and queue integration."""

    @pytest_asyncio.fixture
    async def db(self):
        """Get database instance."""
        database = Database()
        yield database
        await database.close()

    @pytest_asyncio.fixture
    async def queue(self):
        """Get queue instance."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_job_submission_flow(self, db: Database, queue: JobQueue):
        """Test submitting a job through database and queue."""
        async with db.session() as session:
            repo = JobRepository(session)

            # Create job in database
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "Test", "model": "gpt-4o-mini"},
            )
            job = await repo.create(job)

            # Enqueue job ID
            await queue.enqueue(job.id)

            # Verify
            assert await queue.pending_count() == 1
            assert job.status == JobStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_job_processing_flow(self, db: Database, queue: JobQueue):
        """Test processing a job from queue."""
        job_id = None

        # Submit job
        async with db.session() as session:
            repo = JobRepository(session)
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "Test", "model": "gpt-4o-mini"},
            )
            job = await repo.create(job)
            job_id = job.id
            await queue.enqueue(job.id)

        # Simulate worker: dequeue
        dequeued_id = await queue.dequeue_nonblocking()
        assert dequeued_id == job_id

        # Simulate worker: claim job in database
        async with db.session() as session:
            repo = JobRepository(session)
            claimed = await repo.claim_pending_job("test-worker")

            # Note: This might claim a different job if there are others
            # In real tests, we'd need isolation
            assert claimed is not None

    @pytest.mark.asyncio
    async def test_job_completion_flow(self, db: Database, queue: JobQueue):
        """Test completing a job updates both database and queue."""
        job_id = None

        # Submit
        async with db.session() as session:
            repo = JobRepository(session)
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "Test", "model": "gpt-4o-mini"},
            )
            job = await repo.create(job)
            job_id = job.id
            await queue.enqueue(job.id)

        # Process
        await queue.dequeue_nonblocking()

        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)
            job.status = JobStatus.RUNNING.value
            job.worker_id = "test-worker"
            await repo.update(job)

        # Complete
        await queue.complete(job_id)

        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)
            job.status = JobStatus.COMPLETED.value
            job.result = {"content": "Test response"}
            job.total_tokens = 100
            job.cost_usd = 0.001
            await repo.update(job)

        # Verify final state
        assert await queue.processing_count() == 0

        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)
            assert job.status == JobStatus.COMPLETED.value
            assert job.total_tokens == 100

    @pytest.mark.asyncio
    async def test_job_failure_to_dlq_flow(self, db: Database, queue: JobQueue):
        """Test failed job moves to DLQ in both database and queue."""
        job_id = None

        # Submit
        async with db.session() as session:
            repo = JobRepository(session)
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "Test", "model": "gpt-4o-mini"},
                max_attempts=1,  # Only 1 attempt
            )
            job = await repo.create(job)
            job_id = job.id
            await queue.enqueue(job.id)

        # Process and fail
        await queue.dequeue_nonblocking()

        error = {"type": "APIError", "message": "Rate limited"}
        reason = "Max retries exceeded"

        # Move to DLQ in queue
        await queue.move_to_dlq(job_id, error=error, reason=reason)

        # Move to DLQ in database
        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)
            await repo.move_to_dlq(job, reason)

        # Verify
        assert await queue.dlq_count() == 1

        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)
            assert job.status == JobStatus.DEAD.value
            assert job.dlq_reason == reason

    @pytest.mark.asyncio
    async def test_dlq_replay_flow(self, db: Database, queue: JobQueue):
        """Test replaying a job from DLQ."""
        job_id = None

        # Create job already in DLQ state
        async with db.session() as session:
            repo = JobRepository(session)
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "Test", "model": "gpt-4o-mini"},
                status=JobStatus.DEAD.value,
                dlq_reason="Previous failure",
            )
            job = await repo.create(job)
            job_id = job.id

        # Add to queue DLQ
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()
        await queue.move_to_dlq(job_id, reason="Previous failure")

        # Replay from queue
        replayed = await queue.replay_from_dlq(job_id)
        assert replayed is True

        # Replay from database
        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)
            await repo.replay_from_dlq(job)

        # Verify
        assert await queue.pending_count() == 1
        assert await queue.dlq_count() == 0

        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)
            assert job.status == JobStatus.PENDING.value
            assert job.dlq_reason is None


class TestStatisticsIntegration:
    """Tests for statistics across database and queue."""

    @pytest_asyncio.fixture
    async def db(self):
        """Get database instance."""
        database = Database()
        yield database
        await database.close()

    @pytest_asyncio.fixture
    async def queue(self):
        """Get queue instance."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_combined_statistics(self, db: Database, queue: JobQueue):
        """Test getting stats from both database and queue."""
        # Create some jobs
        async with db.session() as session:
            repo = JobRepository(session)

            for i in range(5):
                job = JobRecord(
                    job_type=JobType.LLM_COMPLETION.value,
                    input_data={"prompt": f"Test {i}", "model": "gpt-4o-mini"},
                )
                job = await repo.create(job)
                await queue.enqueue(job.id)

        # Get stats from both
        db_stats = None
        async with db.session() as session:
            repo = JobRepository(session)
            db_stats = await repo.get_stats()

        queue_stats = await queue.get_stats()

        # Verify consistency
        assert db_stats["total_jobs"] == 5
        assert queue_stats["pending"] == 5
        assert queue_stats["total_enqueued"] == 5
