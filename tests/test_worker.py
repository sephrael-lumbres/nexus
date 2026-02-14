"""Tests for worker and worker pool functionality.

These tests verify job processing, retry logic, and pool management.
"""

import asyncio
from uuid import uuid4

import pytest
import pytest_asyncio

from nexus.database import Database, JobRepository, reset_database
from nexus.handlers import CompletionHandler
from nexus.models import JobRecord, JobStatus, JobType
from nexus.providers import MockLLMProvider
from nexus.queue import JobQueue, reset_queue
from nexus.worker import Worker, WorkerPool


# =============================================================================
# Fixtures
# =============================================================================
@pytest_asyncio.fixture
async def db():
    """Get a clean database for testing."""
    reset_database()
    database = Database()
    yield database
    await database.close()
    reset_database()


@pytest_asyncio.fixture
async def queue():
    """Get a clean queue for testing."""
    reset_queue()
    q = JobQueue()
    await q.connect()
    await q.clear_all()
    yield q
    await q.clear_all()
    await q.disconnect()
    reset_queue()


@pytest_asyncio.fixture
async def worker(db: Database, queue: JobQueue):
    """Get a worker for testing."""
    w = Worker(worker_id="test-worker", db=db, queue=queue)
    yield w


@pytest.fixture
def completion_input():
    """Sample completion job input."""
    return {
        "prompt": "What is Python?",
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "temperature": 0.7,
    }


@pytest_asyncio.fixture
async def pending_job(db: Database, queue: JobQueue, completion_input):
    """Create a pending job in database and queue."""
    async with db.session() as session:
        repo = JobRepository(session)

        job = JobRecord(
            job_type=JobType.LLM_COMPLETION.value,
            input_data=completion_input,
            status=JobStatus.PENDING.value,
        )
        job = await repo.create(job)
        job_id = job.id

    await queue.enqueue(job_id)

    return job_id


# =============================================================================
# Worker Tests
# =============================================================================
class TestWorker:
    """Tests for Worker class."""

    @pytest.mark.asyncio
    async def test_worker_initialization(self, db: Database, queue: JobQueue):
        """Test worker initializes correctly."""
        worker = Worker(worker_id="test-worker", db=db, queue=queue)

        assert worker.worker_id == "test-worker"
        assert worker.running is False
        assert worker.jobs_processed == 0
        assert worker.jobs_failed == 0

    @pytest.mark.asyncio
    async def test_worker_auto_generates_id(self, db: Database, queue: JobQueue):
        """Test worker generates ID if not provided."""
        worker = Worker(db=db, queue=queue)

        assert worker.worker_id is not None
        assert worker.worker_id.startswith("worker-")

    @pytest.mark.asyncio
    async def test_process_job_success(
        self,
        worker: Worker,
        db: Database,
        pending_job: uuid4,
    ):
        """Test successful job processing."""
        # Process the job
        await worker._process_next_job()

        # Verify job completed
        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(pending_job)

            assert job.status == JobStatus.COMPLETED.value
            assert job.result is not None
            assert job.worker_id == worker.worker_id
            assert job.total_tokens > 0
            assert job.cost_usd > 0

    @pytest.mark.asyncio
    async def test_process_job_updates_statistics(
        self,
        worker: Worker,
        pending_job: uuid4,
    ):
        """Test that processing updates worker statistics."""
        assert worker.jobs_processed == 0

        await worker._process_next_job()

        assert worker.jobs_processed == 1

    @pytest.mark.asyncio
    async def test_process_job_removes_from_queue(
        self,
        worker: Worker,
        queue: JobQueue,
        pending_job: uuid4,
    ):
        """Test that processed job is removed from queue."""
        assert await queue.pending_count() == 1

        await worker._process_next_job()

        assert await queue.pending_count() == 0
        assert await queue.processing_count() == 0

    @pytest.mark.asyncio
    async def test_process_cancelled_job(
        self,
        worker: Worker,
        db: Database,
        queue: JobQueue,
        completion_input: dict,
    ):
        """Test that cancelled jobs are skipped."""
        # Create a cancelled job
        async with db.session() as session:
            repo = JobRepository(session)

            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data=completion_input,
                status=JobStatus.CANCELLED.value,
            )
            job = await repo.create(job)
            job_id = job.id

        await queue.enqueue(job_id)

        # Process - should skip
        await worker._process_next_job()

        # Job should still be cancelled
        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)

            assert job.status == JobStatus.CANCELLED.value

        # Worker stats should not increase
        assert worker.jobs_processed == 0

    @pytest.mark.asyncio
    async def test_process_nonexistent_job(
        self,
        worker: Worker,
        queue: JobQueue,
    ):
        """Test handling of job ID not in database."""
        fake_job_id = uuid4()
        await queue.enqueue(fake_job_id)

        # Should not raise, just log warning
        await worker._process_next_job()

        # Queue should be cleared
        assert await queue.pending_count() == 0


class TestWorkerRetry:
    """Tests for worker retry logic."""

    @pytest.mark.asyncio
    async def test_job_retried_on_failure(
        self,
        db: Database,
        queue: JobQueue,
    ):
        """Test that failed jobs are retried."""
        # Create job with max_attempts=2
        async with db.session() as session:
            repo = JobRepository(session)

            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={
                    "prompt": "Test",
                    "model": "gpt-4o-mini",
                },
                max_attempts=2,
            )
            job = await repo.create(job)
            job_id = job.id

        await queue.enqueue(job_id)

        # Create worker with failing provider
        from nexus.handlers import CompletionHandler
        from nexus.providers import MockLLMProvider

        failing_provider = MockLLMProvider(failure_rate=1.0)

        # Monkey-patch get_handler to use failing provider
        original_get_handler = __import__('nexus.worker', fromlist=['get_handler']).get_handler

        def mock_get_handler(job_type, provider=None):
            return CompletionHandler(provider=failing_provider)

        import nexus.worker
        nexus.worker.get_handler = mock_get_handler

        try:
            worker = Worker(worker_id="test-worker", db=db, queue=queue)

            # First attempt - should fail and requeue
            await worker._process_next_job()

            # Job should be requeued (back to pending)
            async with db.session() as session:
                repo = JobRepository(session)
                job = await repo.get(job_id)

                assert job.status == JobStatus.PENDING.value
                assert job.attempt == 1
                assert job.error is not None
        finally:
            nexus.worker.get_handler = original_get_handler

    @pytest.mark.asyncio
    async def test_job_moved_to_dlq_after_max_retries(
        self,
        db: Database,
        queue: JobQueue,
    ):
        """Test that exhausted jobs go to DLQ."""
        # Create job with max_attempts=1 (will DLQ on first failure)
        async with db.session() as session:
            repo = JobRepository(session)

            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={
                    "prompt": "Test",
                    "model": "gpt-4o-mini",
                },
                max_attempts=1,
            )
            job = await repo.create(job)
            job_id = job.id

        await queue.enqueue(job_id)

        # Create worker with failing provider
        failing_provider = MockLLMProvider(failure_rate=1.0)

        import nexus.worker
        original_get_handler = nexus.worker.get_handler

        def mock_get_handler(job_type, provider=None):
            return CompletionHandler(provider=failing_provider)

        nexus.worker.get_handler = mock_get_handler

        try:
            worker = Worker(worker_id="test-worker", db=db, queue=queue)

            # Process - should fail and go to DLQ
            await worker._process_next_job()

            # Job should be dead
            async with db.session() as session:
                repo = JobRepository(session)
                job = await repo.get(job_id)

                assert job.status == JobStatus.DEAD.value
                assert job.dlq_reason is not None
                assert "Max retries" in job.dlq_reason

            # Should be in queue DLQ
            assert await queue.dlq_count() == 1

            # Worker should track failure
            assert worker.jobs_failed == 1
        finally:
            nexus.worker.get_handler = original_get_handler

    def test_calculate_backoff(self, db: Database, queue: JobQueue):
        """Test exponential backoff calculation."""
        worker = Worker(db=db, queue=queue)

        # First attempt: 2^1 = 2 seconds base
        backoff1 = worker._calculate_backoff(1)
        assert 2.0 <= backoff1 <= 3.0  # 2 + up to 1 second jitter

        # Second attempt: 2^2 = 4 seconds base
        backoff2 = worker._calculate_backoff(2)
        assert 4.0 <= backoff2 <= 5.0

        # Third attempt: 2^3 = 8 seconds base
        backoff3 = worker._calculate_backoff(3)
        assert 8.0 <= backoff3 <= 9.0

        # Should cap at 60 seconds
        backoff10 = worker._calculate_backoff(10)
        assert 60.0 <= backoff10 <= 61.0


class TestWorkerPool:
    """Tests for WorkerPool class."""

    @pytest.mark.asyncio
    async def test_pool_initialization(self, db: Database, queue: JobQueue):
        """Test pool initializes correctly."""
        pool = WorkerPool(num_workers=3, db=db, queue=queue)

        assert pool.num_workers == 3
        assert len(pool.workers) == 0  # Workers created on start
        assert pool._started is False

    @pytest.mark.asyncio
    async def test_pool_start_creates_workers(self, db: Database, queue: JobQueue):
        """Test that starting pool creates workers."""
        pool = WorkerPool(num_workers=2, db=db, queue=queue)

        # Start in background
        start_task = asyncio.create_task(pool.start())

        # Give it time to start
        await asyncio.sleep(0.1)

        try:
            assert len(pool.workers) == 2
            assert len(pool.tasks) == 2

            for worker in pool.workers:
                assert worker.running is True
        finally:
            await pool.stop()
            await start_task

    @pytest.mark.asyncio
    async def test_pool_stop_stops_workers(self, db: Database, queue: JobQueue):
        """Test that stopping pool stops all workers."""
        pool = WorkerPool(num_workers=2, db=db, queue=queue)

        # Start in background
        start_task = asyncio.create_task(pool.start())
        await asyncio.sleep(0.1)

        # Stop
        await pool.stop()

        for worker in pool.workers:
            assert worker.running is False

        await start_task

    @pytest.mark.asyncio
    async def test_pool_processes_jobs(
        self,
        db: Database,
        queue: JobQueue,
        completion_input: dict,
    ):
        """Test that pool processes jobs."""
        # Create some jobs
        job_ids = []
        async with db.session() as session:
            repo = JobRepository(session)

            for i in range(3):
                job = JobRecord(
                    job_type=JobType.LLM_COMPLETION.value,
                    input_data=completion_input,
                )
                job = await repo.create(job)
                job_ids.append(job.id)
                await queue.enqueue(job.id)

        # Start pool
        pool = WorkerPool(num_workers=2, db=db, queue=queue)
        start_task = asyncio.create_task(pool.start())

        # Wait for jobs to be processed
        await asyncio.sleep(0.5)

        # Stop pool
        await pool.stop()
        await start_task

        # Verify jobs completed
        async with db.session() as session:
            repo = JobRepository(session)

            completed = 0
            for job_id in job_ids:
                job = await repo.get(job_id)
                if job.status == JobStatus.COMPLETED.value:
                    completed += 1

            assert completed == 3

    @pytest.mark.asyncio
    async def test_pool_get_stats(self, db: Database, queue: JobQueue):
        """Test getting pool statistics."""
        pool = WorkerPool(num_workers=2, db=db, queue=queue)

        start_task = asyncio.create_task(pool.start())
        await asyncio.sleep(0.1)

        try:
            stats = pool.get_stats()

            assert stats["num_workers"] == 2
            assert stats["active_workers"] == 2
            assert "total_processed" in stats
            assert "workers" in stats
            assert len(stats["workers"]) == 2
        finally:
            await pool.stop()
            await start_task

    @pytest.mark.asyncio
    async def test_pool_cannot_start_twice(self, db: Database, queue: JobQueue):
        """Test that pool cannot be started twice."""
        pool = WorkerPool(num_workers=1, db=db, queue=queue)

        start_task = asyncio.create_task(pool.start())
        await asyncio.sleep(0.1)

        try:
            with pytest.raises(RuntimeError, match="already started"):
                await pool.start()
        finally:
            await pool.stop()
            await start_task


class TestWorkerJobStates:
    """Tests for job state transitions."""

    @pytest.mark.asyncio
    async def test_job_transitions_to_running(
        self,
        worker: Worker,
        db: Database,
        pending_job: uuid4,
    ):
        """Test that job transitions to running during processing."""
        # Start processing in background
        task = asyncio.create_task(worker._process_next_job())

        # Give it a moment to start
        await asyncio.sleep(0.01)

        # Job should be running (or already completed if fast)
        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(pending_job)

            # Could be running or completed depending on timing
            assert job.status in [
                JobStatus.RUNNING.value,
                JobStatus.COMPLETED.value,
            ]

        await task

    @pytest.mark.asyncio
    async def test_job_records_worker_id(
        self,
        worker: Worker,
        db: Database,
        pending_job: uuid4,
    ):
        """Test that job records which worker processed it."""
        await worker._process_next_job()

        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(pending_job)

            assert job.worker_id == worker.worker_id

    @pytest.mark.asyncio
    async def test_job_records_timestamps(
        self,
        worker: Worker,
        db: Database,
        pending_job: uuid4,
    ):
        """Test that job records processing timestamps."""
        await worker._process_next_job()

        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(pending_job)

            assert job.started_at is not None
            assert job.completed_at is not None
            assert job.completed_at >= job.started_at

    @pytest.mark.asyncio
    async def test_job_increments_attempt(
        self,
        worker: Worker,
        db: Database,
        pending_job: uuid4,
    ):
        """Test that job attempt counter is incremented."""
        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(pending_job)
            assert job.attempt == 0

        await worker._process_next_job()

        async with db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(pending_job)
            assert job.attempt == 1
