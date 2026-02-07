"""Tests for Redis job queue.

These tests verify queue operations including
enqueue, dequeue, DLQ, and statistics tracking.
"""

from uuid import uuid4

import pytest
import pytest_asyncio

from nexus.queue import JobQueue, get_queue, reset_queue


class TestJobQueueConnection:
    """Tests for queue connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to Redis."""
        queue = JobQueue()

        try:
            await queue.connect()
            assert queue.redis is not None
        finally:
            await queue.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection from Redis."""
        queue = JobQueue()

        await queue.connect()
        await queue.disconnect()

        assert queue.redis is None

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check returns True when Redis is available."""
        queue = JobQueue()

        try:
            result = await queue.health_check()
            assert result is True
        finally:
            await queue.disconnect()

    @pytest.mark.asyncio
    async def test_auto_connect_on_operation(self):
        """Test that operations auto-connect if not connected."""
        queue = JobQueue()

        try:
            # Should auto-connect
            await queue.pending_count()
            assert queue.redis is not None
        finally:
            await queue.disconnect()


class TestJobQueueEnqueue:
    """Tests for enqueue operations."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Get a clean queue for testing."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_enqueue_single_job(self, queue: JobQueue):
        """Test enqueueing a single job."""
        job_id = uuid4()

        length = await queue.enqueue(job_id)

        assert length == 1
        assert await queue.pending_count() == 1

    @pytest.mark.asyncio
    async def test_enqueue_multiple_jobs(self, queue: JobQueue):
        """Test enqueueing multiple jobs."""
        job_ids = [uuid4() for _ in range(5)]

        for job_id in job_ids:
            await queue.enqueue(job_id)

        assert await queue.pending_count() == 5

    @pytest.mark.asyncio
    async def test_enqueue_fifo_order(self, queue: JobQueue):
        """Test that jobs are queued in FIFO order."""
        job1 = uuid4()
        job2 = uuid4()
        job3 = uuid4()

        await queue.enqueue(job1)
        await queue.enqueue(job2)
        await queue.enqueue(job3)

        # Peek should return jobs in FIFO order
        peeked = await queue.peek(3)

        assert peeked[0] == job1
        assert peeked[1] == job2
        assert peeked[2] == job3

    @pytest.mark.asyncio
    async def test_enqueue_updates_stats(self, queue: JobQueue):
        """Test that enqueue updates statistics."""
        job_id = uuid4()

        await queue.enqueue(job_id)

        stats = await queue.get_stats()
        assert stats["total_enqueued"] >= 1


class TestJobQueueDequeue:
    """Tests for dequeue operations."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Get a clean queue for testing."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_dequeue_nonblocking_with_job(self, queue: JobQueue):
        """Test non-blocking dequeue when job is available."""
        job_id = uuid4()
        await queue.enqueue(job_id)

        dequeued = await queue.dequeue_nonblocking()

        assert dequeued == job_id
        assert await queue.pending_count() == 0
        assert await queue.processing_count() == 1

    @pytest.mark.asyncio
    async def test_dequeue_nonblocking_empty_queue(self, queue: JobQueue):
        """Test non-blocking dequeue when queue is empty."""
        dequeued = await queue.dequeue_nonblocking()

        assert dequeued is None

    @pytest.mark.asyncio
    async def test_dequeue_blocking_with_job(self, queue: JobQueue):
        """Test blocking dequeue with short timeout."""
        job_id = uuid4()
        await queue.enqueue(job_id)

        # Short timeout since job is already there
        dequeued = await queue.dequeue(timeout=1.0)

        assert dequeued == job_id

    @pytest.mark.asyncio
    async def test_dequeue_blocking_timeout(self, queue: JobQueue):
        """Test blocking dequeue times out on empty queue."""
        # Very short timeout for test speed
        dequeued = await queue.dequeue(timeout=0.1)

        assert dequeued is None

    @pytest.mark.asyncio
    async def test_dequeue_fifo_order(self, queue: JobQueue):
        """Test that jobs are dequeued in FIFO order."""
        job1 = uuid4()
        job2 = uuid4()
        job3 = uuid4()

        await queue.enqueue(job1)
        await queue.enqueue(job2)
        await queue.enqueue(job3)

        assert await queue.dequeue_nonblocking() == job1
        assert await queue.dequeue_nonblocking() == job2
        assert await queue.dequeue_nonblocking() == job3

    @pytest.mark.asyncio
    async def test_dequeue_adds_to_processing(self, queue: JobQueue):
        """Test that dequeued jobs are tracked in processing."""
        job_id = uuid4()
        await queue.enqueue(job_id)

        await queue.dequeue_nonblocking()

        assert await queue.is_processing(job_id)

    @pytest.mark.asyncio
    async def test_dequeue_updates_stats(self, queue: JobQueue):
        """Test that dequeue updates statistics."""
        job_id = uuid4()
        await queue.enqueue(job_id)

        await queue.dequeue_nonblocking()

        stats = await queue.get_stats()
        assert stats["total_dequeued"] >= 1


class TestJobQueueComplete:
    """Tests for job completion operations."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Get a clean queue for testing."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_complete_job(self, queue: JobQueue):
        """Test completing a job."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()

        result = await queue.complete(job_id)

        assert result is True
        assert await queue.processing_count() == 0
        assert not await queue.is_processing(job_id)

    @pytest.mark.asyncio
    async def test_complete_updates_stats(self, queue: JobQueue):
        """Test that complete updates statistics."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()

        await queue.complete(job_id)

        stats = await queue.get_stats()
        assert stats["total_completed"] >= 1

    @pytest.mark.asyncio
    async def test_complete_nonexistent_job(self, queue: JobQueue):
        """Test completing a job not in processing."""
        job_id = uuid4()

        result = await queue.complete(job_id)

        assert result is False


class TestJobQueueFail:
    """Tests for job failure operations."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Get a clean queue for testing."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_fail_job(self, queue: JobQueue):
        """Test failing a job."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()

        result = await queue.fail(job_id)

        assert result is True
        assert await queue.processing_count() == 0

    @pytest.mark.asyncio
    async def test_fail_updates_stats(self, queue: JobQueue):
        """Test that fail updates statistics."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()

        await queue.fail(job_id)

        stats = await queue.get_stats()
        assert stats["total_failed"] >= 1


class TestJobQueueRequeue:
    """Tests for requeue operations."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Get a clean queue for testing."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_requeue_to_front(self, queue: JobQueue):
        """Test requeueing a job to front of queue."""
        job1 = uuid4()
        job2 = uuid4()

        # Enqueue job2
        await queue.enqueue(job2)

        # Enqueue and dequeue job1
        await queue.enqueue(job1)
        await queue.dequeue_nonblocking()  # Gets job2
        await queue.dequeue_nonblocking()  # Gets job1

        # Requeue job1 to front
        await queue.requeue(job1, to_front=True)

        # job1 should be first now
        peeked = await queue.peek(1)
        assert peeked[0] == job1

    @pytest.mark.asyncio
    async def test_requeue_to_back(self, queue: JobQueue):
        """Test requeueing a job to back of queue."""
        job1 = uuid4()
        job2 = uuid4()

        await queue.enqueue(job1)
        await queue.enqueue(job2)

        # Dequeue job1
        await queue.dequeue_nonblocking()

        # Requeue job1 to back
        await queue.requeue(job1, to_front=False)

        # job2 should still be first
        peeked = await queue.peek(2)
        assert peeked[0] == job2
        assert peeked[1] == job1

    @pytest.mark.asyncio
    async def test_requeue_removes_from_processing(self, queue: JobQueue):
        """Test that requeue removes job from processing."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()

        assert await queue.is_processing(job_id)

        await queue.requeue(job_id)

        assert not await queue.is_processing(job_id)

    @pytest.mark.asyncio
    async def test_requeue_updates_stats(self, queue: JobQueue):
        """Test that requeue updates statistics."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()

        await queue.requeue(job_id)

        stats = await queue.get_stats()
        assert stats["total_requeued"] >= 1


class TestJobQueueDLQ:
    """Tests for dead letter queue operations."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Get a clean queue for testing."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_move_to_dlq(self, queue: JobQueue):
        """Test moving a job to DLQ."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()

        dlq_length = await queue.move_to_dlq(
            job_id,
            error={"type": "TestError", "message": "Test"},
            reason="Max retries exceeded",
        )

        assert dlq_length == 1
        assert await queue.dlq_count() == 1
        assert await queue.processing_count() == 0

    @pytest.mark.asyncio
    async def test_move_to_dlq_stores_metadata(self, queue: JobQueue):
        """Test that DLQ stores metadata correctly."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()

        await queue.move_to_dlq(
            job_id,
            error={"code": 500},
            reason="Server error",
        )

        entries = await queue.get_dlq_entries()

        assert len(entries) == 1
        assert entries[0]["job_id"] == str(job_id)
        assert entries[0]["error"] == {"code": 500}
        assert entries[0]["reason"] == "Server error"
        assert "moved_at" in entries[0]

    @pytest.mark.asyncio
    async def test_get_dlq_entries_pagination(self, queue: JobQueue):
        """Test DLQ pagination."""
        # Add 5 jobs to DLQ
        for _ in range(5):
            job_id = uuid4()
            await queue.enqueue(job_id)
            await queue.dequeue_nonblocking()
            await queue.move_to_dlq(job_id)

        # Get first 2
        entries = await queue.get_dlq_entries(start=0, end=1)
        assert len(entries) == 2

        # Get all
        entries = await queue.get_dlq_entries()
        assert len(entries) == 5

    @pytest.mark.asyncio
    async def test_replay_from_dlq(self, queue: JobQueue):
        """Test replaying a job from DLQ."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()
        await queue.move_to_dlq(job_id, reason="Test")

        result = await queue.replay_from_dlq(job_id)

        assert result is True
        assert await queue.dlq_count() == 0
        assert await queue.pending_count() == 1

    @pytest.mark.asyncio
    async def test_replay_from_dlq_not_found(self, queue: JobQueue):
        """Test replaying a job not in DLQ."""
        job_id = uuid4()

        result = await queue.replay_from_dlq(job_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_replay_updates_stats(self, queue: JobQueue):
        """Test that replay updates statistics."""
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()
        await queue.move_to_dlq(job_id)

        await queue.replay_from_dlq(job_id)

        stats = await queue.get_stats()
        assert stats["total_replayed"] >= 1

    @pytest.mark.asyncio
    async def test_clear_dlq(self, queue: JobQueue):
        """Test clearing the DLQ."""
        # Add 3 jobs to DLQ
        for _ in range(3):
            job_id = uuid4()
            await queue.enqueue(job_id)
            await queue.dequeue_nonblocking()
            await queue.move_to_dlq(job_id)

        count = await queue.clear_dlq()

        assert count == 3
        assert await queue.dlq_count() == 0


class TestJobQueueInspection:
    """Tests for queue inspection operations."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Get a clean queue for testing."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_pending_count(self, queue: JobQueue):
        """Test pending count."""
        for _ in range(3):
            await queue.enqueue(uuid4())

        assert await queue.pending_count() == 3

    @pytest.mark.asyncio
    async def test_processing_count(self, queue: JobQueue):
        """Test processing count."""
        for _ in range(3):
            await queue.enqueue(uuid4())

        await queue.dequeue_nonblocking()
        await queue.dequeue_nonblocking()

        assert await queue.processing_count() == 2

    @pytest.mark.asyncio
    async def test_get_processing_jobs(self, queue: JobQueue):
        """Test getting processing job IDs."""
        job1 = uuid4()
        job2 = uuid4()

        await queue.enqueue(job1)
        await queue.enqueue(job2)
        await queue.dequeue_nonblocking()
        await queue.dequeue_nonblocking()

        processing = await queue.get_processing_jobs()

        assert len(processing) == 2
        assert job1 in processing
        assert job2 in processing

    @pytest.mark.asyncio
    async def test_is_processing(self, queue: JobQueue):
        """Test checking if job is processing."""
        job_id = uuid4()
        await queue.enqueue(job_id)

        assert not await queue.is_processing(job_id)

        await queue.dequeue_nonblocking()

        assert await queue.is_processing(job_id)

    @pytest.mark.asyncio
    async def test_peek(self, queue: JobQueue):
        """Test peeking at queue."""
        job1 = uuid4()
        job2 = uuid4()
        job3 = uuid4()

        await queue.enqueue(job1)
        await queue.enqueue(job2)
        await queue.enqueue(job3)

        peeked = await queue.peek(2)

        assert len(peeked) == 2
        assert peeked[0] == job1
        assert peeked[1] == job2

        # Verify peek didn't remove jobs
        assert await queue.pending_count() == 3


class TestJobQueueStats:
    """Tests for queue statistics."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Get a clean queue for testing."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, queue: JobQueue):
        """Test stats on empty queue."""
        stats = await queue.get_stats()

        assert stats["pending"] == 0
        assert stats["processing"] == 0
        assert stats["dlq"] == 0
        assert stats["total_enqueued"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_activity(self, queue: JobQueue):
        """Test stats after queue activity."""
        job1 = uuid4()
        job2 = uuid4()
        job3 = uuid4()

        # Enqueue 3
        await queue.enqueue(job1)
        await queue.enqueue(job2)
        await queue.enqueue(job3)

        # Dequeue and complete 1
        await queue.dequeue_nonblocking()
        await queue.complete(job1)

        # Dequeue and fail 1 (move to DLQ)
        await queue.dequeue_nonblocking()
        await queue.move_to_dlq(job2)

        stats = await queue.get_stats()

        assert stats["pending"] == 1
        assert stats["processing"] == 0
        assert stats["dlq"] == 1
        assert stats["total_enqueued"] == 3
        assert stats["total_dequeued"] == 2
        assert stats["total_completed"] == 1
        assert stats["total_dlq"] == 1

    @pytest.mark.asyncio
    async def test_reset_stats(self, queue: JobQueue):
        """Test resetting statistics."""
        await queue.enqueue(uuid4())

        stats_before = await queue.get_stats()
        assert stats_before["total_enqueued"] >= 1

        await queue.reset_stats()

        stats_after = await queue.get_stats()
        assert stats_after["total_enqueued"] == 0


class TestJobQueueMaintenance:
    """Tests for queue maintenance operations."""

    @pytest_asyncio.fixture
    async def queue(self):
        """Get a clean queue for testing."""
        q = JobQueue()
        await q.connect()
        await q.clear_all()
        yield q
        await q.clear_all()
        await q.disconnect()

    @pytest.mark.asyncio
    async def test_clear_all(self, queue: JobQueue):
        """Test clearing all queues."""
        # Add data to all queues
        job1 = uuid4()
        job2 = uuid4()

        await queue.enqueue(job1)
        await queue.enqueue(job2)
        await queue.dequeue_nonblocking()  # job1 to processing
        await queue.move_to_dlq(job1)       # job1 to DLQ

        result = await queue.clear_all()

        assert result["pending"] == 1
        assert result["processing"] == 0  # Already moved to DLQ
        assert result["dlq"] == 1

        assert await queue.pending_count() == 0
        assert await queue.processing_count() == 0
        assert await queue.dlq_count() == 0

    @pytest.mark.asyncio
    async def test_recover_stuck_jobs(self, queue: JobQueue):
        """Test recovering stuck jobs."""
        job1 = uuid4()
        job2 = uuid4()

        await queue.enqueue(job1)
        await queue.enqueue(job2)
        await queue.dequeue_nonblocking()
        await queue.dequeue_nonblocking()

        # Both jobs in processing, simulate worker crash
        assert await queue.processing_count() == 2

        # Recover them
        recovered = await queue.recover_stuck_jobs([job1, job2])

        assert recovered == 2
        assert await queue.processing_count() == 0
        assert await queue.pending_count() == 2

    @pytest.mark.asyncio
    async def test_recover_stuck_jobs_partial(self, queue: JobQueue):
        """Test recovering only jobs that are actually stuck."""
        job1 = uuid4()
        job2 = uuid4()  # Not in queue

        await queue.enqueue(job1)
        await queue.dequeue_nonblocking()

        recovered = await queue.recover_stuck_jobs([job1, job2])

        assert recovered == 1  # Only job1 was stuck


class TestJobQueueSingleton:
    """Tests for queue singleton pattern."""

    def test_get_queue_returns_same_instance(self):
        """Test that get_queue returns singleton."""
        reset_queue()

        q1 = get_queue()
        q2 = get_queue()

        assert q1 is q2

        reset_queue()

    def test_reset_queue_clears_singleton(self):
        """Test that reset_queue clears the singleton."""
        q1 = get_queue()
        reset_queue()
        q2 = get_queue()

        assert q1 is not q2

        reset_queue()
