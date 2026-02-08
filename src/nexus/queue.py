"""Redis-backed job queue with dead letter queue support.

This module provides:
- FIFO job queue using Redis lists
- Blocking dequeue for efficient polling
- Dead letter queue for failed jobs after max retries
- Queue statistics for monitoring
- Atomic operations for reliability under concurrency

Architecture:
    - Pending jobs: Redis list (FIFO via RPUSH/BLPOP)
    - Processing jobs: Redis set (for tracking)
    - Dead letter queue: Redis list with metadata
    - Statistics: Redis hash
"""

import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import redis.asyncio as redis
import structlog

from nexus.config import get_settings

logger = structlog.get_logger()


class JobQueue:
    """Redis-backed job queue with dead letter queue support.

    Example:
        queue = JobQueue()
        await queue.connect()

        # Enqueue a job
        await queue.enqueue(job_id)

        # Dequeue (blocking)
        job_id = await queue.dequeue(timeout=5.0)

        # Mark complete or failed
        await queue.complete(job_id)
        # or
        await queue.fail(job_id)

        await queue.disconnect()
    """

    # Redis key prefixes
    KEY_PREFIX = "nexus"

    def __init__(self, redis_url: str | None = None):
        """Initialize the job queue.

        Args:
            redis_url: Redis connection URL. Defaults to settings if not provided.
        """
        self.redis_url = redis_url or get_settings().redis_url
        self.redis: redis.Redis | None = None

        # Define Redis keys
        self.pending_key = f"{self.KEY_PREFIX}:queue:pending"
        self.processing_key = f"{self.KEY_PREFIX}:queue:processing"
        self.dlq_key = f"{self.KEY_PREFIX}:queue:dlq"
        self.stats_key = f"{self.KEY_PREFIX}:stats"


    # =========================================================================
    # Connection Management
    # =========================================================================
    async def connect(self) -> None:
        """Connect to Redis.

        Creates a connection pool for efficient connection reuse.
        Safe to call multiple times (idempotent).
        """
        if self.redis is None:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Verify connection
            await self.redis.ping()  # type: ignore[misc]
            logger.info("Connected to Redis", url=self._safe_url())

    async def disconnect(self) -> None:
        """Disconnect from Redis.

        Closes the connection pool and releases resources.
        """
        if self.redis is not None:
            await self.redis.aclose()
            self.redis = None
            logger.info("Disconnected from Redis")

    async def health_check(self) -> bool:
        """Check Redis connectivity.

        Returns:
            bool: True if Redis is accessible, False otherwise
        """
        try:
            await self.connect()
            assert self.redis is not None
            await self.redis.ping()  # type: ignore[misc]
            return True
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False

    def _safe_url(self) -> str:
        """Get URL with password masked for logging."""
        # Simple masking - in production use proper URL parsing
        if "@" in self.redis_url:
            parts = self.redis_url.split("@")
            return f"redis://***@{parts[-1]}"
        return self.redis_url

    async def _ensure_connected(self) -> None:
        """Ensure Redis connection is established."""
        if self.redis is None:
            await self.connect()


    # =========================================================================
    # Core Queue Operations
    # =========================================================================
    async def enqueue(self, job_id: UUID) -> int:
        """Add a job to the pending queue.

        Jobs are added to the tail of the list (RPUSH) for FIFO ordering.

        Args:
            job_id: UUID of the job to enqueue

        Returns:
            int: New length of the pending queue
        """
        await self._ensure_connected()
        assert self.redis is not None

        job_id_str = str(job_id)

        # Add to pending queue (FIFO: add to tail)
        queue_length = cast(int, await self.redis.rpush(self.pending_key, job_id_str))  # type: ignore[misc]

        # Update statistics
        await self.redis.hincrby(self.stats_key, "total_enqueued", 1)  # type: ignore[misc]

        logger.debug("Job enqueued", job_id=job_id_str, queue_length=queue_length)

        return queue_length

    async def dequeue(self, timeout: float = 5.0) -> UUID | None:
        """Get the next job from the queue (blocking).

        Uses BLPOP for efficient blocking wait. The job is moved
        to the processing set to track in-flight jobs.

        Args:
            timeout: Maximum seconds to wait for a job (0 = forever)

        Returns:
            UUID of the job if available, None if timeout reached
        """
        await self._ensure_connected()
        assert self.redis is not None

        # Blocking pop from head of queue (FIFO)
        result = await self.redis.blpop([self.pending_key], timeout=timeout)  # type: ignore[misc]

        if result is None:
            return None

        # result is (key, value) tuple, ignoring 'key'
        _, job_id_str = result
        job_id = UUID(job_id_str)

        # Track in processing set
        await self.redis.sadd(self.processing_key, job_id_str)  # type: ignore[misc]

        # Update statistics
        await self.redis.hincrby(self.stats_key, "total_dequeued", 1)  # type: ignore[misc]

        logger.debug("Job dequeued", job_id=job_id_str)

        return job_id

    async def dequeue_nonblocking(self) -> UUID | None:
        """Get the next job from the queue (non-blocking).

        Returns immediately if no job is available.

        Returns:
            UUID of the job if available, None if queue is empty
        """
        await self._ensure_connected()
        assert self.redis is not None

        # Non-blocking pop
        job_id_str = await self.redis.lpop(self.pending_key)  # type: ignore[misc]

        if job_id_str is None:
            return None

        job_id = UUID(job_id_str)

        # Track in processing set
        await self.redis.sadd(self.processing_key, job_id_str)  # type: ignore[misc]

        # Update statistics
        await self.redis.hincrby(self.stats_key, "total_dequeued", 1)  # type: ignore[misc]

        logger.debug("Job dequeued (non-blocking)", job_id=job_id_str)

        return job_id

    async def complete(self, job_id: UUID) -> bool:
        """Mark a job as successfully completed.

        Removes the job from the processing set.

        Args:
            job_id: UUID of the completed job

        Returns:
            bool: True if job was in processing set, False otherwise
        """
        await self._ensure_connected()
        assert self.redis is not None

        job_id_str = str(job_id)

        # Remove from processing set
        removed = await self.redis.srem(self.processing_key, job_id_str)  # type: ignore[misc]

        if removed:
            # Update statistics
            await self.redis.hincrby(self.stats_key, "total_completed", 1)  # type: ignore[misc]
            logger.debug("Job completed", job_id=job_id_str)
        else:
            logger.warning("Job not in processing set", job_id=job_id_str)

        return bool(removed)

    async def fail(self, job_id: UUID) -> bool:
        """Mark a job as failed.

        Removes the job from the processing set. The job may be
        requeued for retry or moved to DLQ by the worker.

        Args:
            job_id: UUID of the failed job

        Returns:
            bool: True if job was in processing set, False otherwise
        """
        await self._ensure_connected()
        assert self.redis is not None

        job_id_str = str(job_id)

        # Remove from processing set
        removed = await self.redis.srem(self.processing_key, job_id_str)  # type: ignore[misc]

        if removed:
            # Update statistics
            await self.redis.hincrby(self.stats_key, "total_failed", 1)  # type: ignore[misc]
            logger.debug("Job failed", job_id=job_id_str)
        else:
            logger.warning("Job not in processing set", job_id=job_id_str)

        return bool(removed)

    async def requeue(self, job_id: UUID, to_front: bool = True) -> int:
        """Return a job to the queue for retry.

        Args:
            job_id: UUID of the job to requeue
            to_front: If True, add to front of queue (priority retry)

        Returns:
            int: New length of the pending queue
        """
        await self._ensure_connected()
        assert self.redis is not None

        job_id_str = str(job_id)

        # Remove from processing set
        await self.redis.srem(self.processing_key, job_id_str)  # type: ignore[misc]

        # Add back to queue
        if to_front:
            # Add to front for faster retry
            queue_length = cast(int, await self.redis.lpush(self.pending_key, job_id_str))  # type: ignore[misc]
        else:
            # Add to back (normal FIFO)
            queue_length = cast(int, await self.redis.rpush(self.pending_key, job_id_str))  # type: ignore[misc]

        # Update statistics
        await self.redis.hincrby(self.stats_key, "total_requeued", 1)  # type: ignore[misc]

        logger.debug(
            "Job requeued",
            job_id=job_id_str,
            to_front=to_front,
            queue_length=queue_length,
        )

        return queue_length


    # =========================================================================
    # Dead Letter Queue Operations
    # =========================================================================
    async def move_to_dlq(
        self,
        job_id: UUID,
        error: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> int:
        """Move a failed job to the dead letter queue.

        Jobs in the DLQ are stored with metadata for debugging.

        Args:
            job_id: UUID of the job to move
            error: Error details (optional)
            reason: Human-readable reason (optional)

        Returns:
            int: New length of the DLQ
        """
        await self._ensure_connected()
        assert self.redis is not None

        job_id_str = str(job_id)

        # Remove from processing set
        await self.redis.srem(self.processing_key, job_id_str)  # type: ignore[misc]

        # Create DLQ entry with metadata
        dlq_entry = {
            "job_id": job_id_str,
            "error": error,
            "reason": reason,
            "moved_at": datetime.now(UTC).isoformat(),
        }

        # Add to DLQ
        dlq_length = cast(int, await self.redis.rpush(self.dlq_key, json.dumps(dlq_entry)))  # type: ignore[misc]

        # Update statistics
        await self.redis.hincrby(self.stats_key, "total_dlq", 1)  # type: ignore[misc]

        logger.warning(
            "Job moved to DLQ",
            job_id=job_id_str,
            reason=reason,
            dlq_length=dlq_length,
        )

        return dlq_length

    async def get_dlq_entries(
        self,
        start: int = 0,
        end: int = -1,
    ) -> list[dict[str, Any]]:
        """Get entries from the dead letter queue.

        Args:
            start: Start index (0-based)
            end: End index (-1 for all)

        Returns:
            List of DLQ entries with metadata
        """
        await self._ensure_connected()
        assert self.redis is not None

        # Get entries from DLQ
        entries = await self.redis.lrange(self.dlq_key, start, end)  # type: ignore[misc]

        # Parse JSON entries
        return [json.loads(entry) for entry in entries]

    async def replay_from_dlq(self, job_id: UUID) -> bool:
        """Replay a job from the dead letter queue.

        Removes the job from DLQ and adds it back to the pending queue.

        Args:
            job_id: UUID of the job to replay

        Returns:
            bool: True if job was found and replayed, False otherwise
        """
        await self._ensure_connected()
        assert self.redis is not None

        job_id_str = str(job_id)

        # Get all DLQ entries
        entries = await self.redis.lrange(self.dlq_key, 0, -1)  # type: ignore[misc]

        # Find and remove the matching entry
        for entry_json in entries:
            entry = json.loads(entry_json)
            if entry.get("job_id") == job_id_str:
                # Remove from DLQ (removes first occurrence)
                removed = await self.redis.lrem(self.dlq_key, 1, entry_json)  # type: ignore[misc]

                if removed:
                    # Add back to pending queue
                    await self.redis.rpush(self.pending_key, job_id_str)  # type: ignore[misc]

                    # Update statistics
                    await self.redis.hincrby(self.stats_key, "total_replayed", 1)  # type: ignore[misc]

                    logger.info("Job replayed from DLQ", job_id=job_id_str)
                    return True

        logger.warning("Job not found in DLQ", job_id=job_id_str)
        return False

    async def clear_dlq(self) -> int:
        """Clear all entries from the dead letter queue.

        Use with caution - this permanently removes all DLQ entries.

        Returns:
            int: Number of entries removed
        """
        await self._ensure_connected()
        assert self.redis is not None

        # Get count before clearing
        count = cast(int, await self.redis.llen(self.dlq_key))  # type: ignore[misc]

        # Delete the DLQ
        await self.redis.delete(self.dlq_key)

        logger.warning("DLQ cleared", entries_removed=count)

        return count


    # =========================================================================
    # Queue Inspection
    # =========================================================================
    async def pending_count(self) -> int:
        """Get the number of jobs in the pending queue.

        Returns:
            int: Number of pending jobs
        """
        await self._ensure_connected()
        assert self.redis is not None
        return cast(int, await self.redis.llen(self.pending_key))  # type: ignore[misc]

    async def processing_count(self) -> int:
        """Get the number of jobs currently being processed.

        Returns:
            int: Number of in-flight jobs
        """
        await self._ensure_connected()
        assert self.redis is not None
        return cast(int, await self.redis.scard(self.processing_key))  # type: ignore[misc]

    async def dlq_count(self) -> int:
        """Get the number of jobs in the dead letter queue.

        Returns:
            int: Number of DLQ entries
        """
        await self._ensure_connected()
        assert self.redis is not None
        return cast(int, await self.redis.llen(self.dlq_key))  # type: ignore[misc]

    async def get_processing_jobs(self) -> list[UUID]:
        """Get all jobs currently being processed.

        Useful for detecting stuck jobs.

        Returns:
            List of job UUIDs currently in processing
        """
        await self._ensure_connected()
        assert self.redis is not None

        job_ids = await self.redis.smembers(self.processing_key)  # type: ignore[misc]
        return [UUID(job_id) for job_id in job_ids]

    async def is_processing(self, job_id: UUID) -> bool:
        """Check if a job is currently being processed.

        Args:
            job_id: UUID of the job to check

        Returns:
            bool: True if job is in processing set
        """
        await self._ensure_connected()
        assert self.redis is not None
        return bool(await self.redis.sismember(self.processing_key, str(job_id)))  # type: ignore[misc]

    async def peek(self, count: int = 10) -> list[UUID]:
        """Peek at the next jobs in the queue without removing them.

        Args:
            count: Number of jobs to peek

        Returns:
            List of job UUIDs at the front of the queue
        """
        await self._ensure_connected()
        assert self.redis is not None

        job_ids = await self.redis.lrange(self.pending_key, 0, count - 1)  # type: ignore[misc]
        return [UUID(job_id) for job_id in job_ids]


    # =========================================================================
    # Statistics
    # =========================================================================
    async def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns comprehensive stats for monitoring and debugging.

        Returns:
            Dict with queue statistics
        """
        await self._ensure_connected()
        assert self.redis is not None

        # Get counters from hash
        stats = await self.redis.hgetall(self.stats_key)  # type: ignore[misc]

        # Get current queue depths
        pending = await self.pending_count()
        processing = await self.processing_count()
        dlq = await self.dlq_count()

        return {
            "pending": pending,
            "processing": processing,
            "dlq": dlq,
            "total_enqueued": int(stats.get("total_enqueued", 0)),
            "total_dequeued": int(stats.get("total_dequeued", 0)),
            "total_completed": int(stats.get("total_completed", 0)),
            "total_failed": int(stats.get("total_failed", 0)),
            "total_requeued": int(stats.get("total_requeued", 0)),
            "total_dlq": int(stats.get("total_dlq", 0)),
            "total_replayed": int(stats.get("total_replayed", 0)),
        }

    async def reset_stats(self) -> None:
        """Reset all statistics counters.

        Use with caution - typically only for testing.
        """
        await self._ensure_connected()
        assert self.redis is not None
        await self.redis.delete(self.stats_key)
        logger.info("Queue statistics reset")


    # =========================================================================
    # Maintenance Operations
    # =========================================================================
    async def clear_all(self) -> dict[str, int]:
        """Clear all queue data.

        WARNING: This removes all pending, processing, and DLQ jobs.
        Use only for testing or complete reset.

        Returns:
            Dict with counts of items removed from each queue
        """
        await self._ensure_connected()
        assert self.redis is not None

        # Get counts before clearing
        pending = await self.pending_count()
        processing = await self.processing_count()
        dlq = await self.dlq_count()

        # Clear all keys
        await self.redis.delete(
            self.pending_key,
            self.processing_key,
            self.dlq_key,
            self.stats_key,
        )

        logger.warning(
            "All queues cleared",
            pending_removed=pending,
            processing_removed=processing,
            dlq_removed=dlq,
        )

        return {
            "pending": pending,
            "processing": processing,
            "dlq": dlq,
        }

    async def recover_stuck_jobs(self, job_ids: list[UUID]) -> int:
        """Recover jobs that are stuck in processing.

        Moves jobs from processing back to pending queue.
        Use when workers crash without completing jobs.

        Args:
            job_ids: List of job UUIDs to recover

        Returns:
            int: Number of jobs recovered
        """
        await self._ensure_connected()
        assert self.redis is not None

        recovered = 0
        for job_id in job_ids:
            job_id_str = str(job_id)

            # Check if in processing set
            is_processing = await self.redis.sismember(  # type: ignore[misc]
                self.processing_key,
                job_id_str,
            )

            if is_processing:
                # Remove from processing
                await self.redis.srem(self.processing_key, job_id_str)  # type: ignore[misc]

                # Add back to pending (front of queue)
                await self.redis.lpush(self.pending_key, job_id_str)  # type: ignore[misc]

                recovered += 1
                logger.info("Recovered stuck job", job_id=job_id_str)

        return recovered


# =============================================================================
# Module-level Singleton
# =============================================================================
_queue: JobQueue | None = None

def get_queue() -> JobQueue:
    """Get or create the global queue instance.

    Returns:
        JobQueue: Singleton queue instance
    """
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue

def reset_queue() -> None:
    """Reset the global queue instance.

    Useful for testing to ensure clean state.
    """
    global _queue
    _queue = None


# =============================================================================
# Quick Test
# =============================================================================
async def _test_queue() -> None:
    """Quick test of queue functionality."""
    from uuid import uuid4

    queue = JobQueue()

    try:
        # Connect
        await queue.connect()
        print("[TEST] Connected to Redis")

        # Health check
        healthy = await queue.health_check()
        print(f"[TEST] Health check: {healthy}")

        # Clear for clean test
        await queue.clear_all()

        # Create test job IDs
        job1 = uuid4()
        job2 = uuid4()
        job3 = uuid4()

        # Enqueue jobs
        await queue.enqueue(job1)
        await queue.enqueue(job2)
        await queue.enqueue(job3)
        print("[TEST] Enqueued 3 jobs")

        # Check pending count
        pending = await queue.pending_count()
        print(f"[TEST] Pending count: {pending}")

        # Peek at queue
        peeked = await queue.peek(2)
        print(f"[TEST] Peeked jobs: {[str(j)[:8] for j in peeked]}")

        # Dequeue a job
        dequeued = await queue.dequeue_nonblocking()
        print(f"[TEST] Dequeued: {str(dequeued)[:8] if dequeued else None}")

        # Check processing
        processing = await queue.processing_count()
        print(f"[TEST] Processing count: {processing}")

        # Complete the job
        if dequeued:
            await queue.complete(dequeued)
            print("[TEST] Completed job")

        # Dequeue and fail a job
        dequeued = await queue.dequeue_nonblocking()
        if dequeued:
            await queue.move_to_dlq(dequeued, error={"msg": "Test error"}, reason="Testing DLQ")
            print("[TEST] Moved job to DLQ")

        # Check DLQ
        dlq_entries = await queue.get_dlq_entries()
        print(f"[TEST] DLQ entries: {len(dlq_entries)}")

        # Get stats
        stats = await queue.get_stats()
        print(f"[TEST] Stats: {stats}")

        # Cleanup
        await queue.clear_all()
        print("[TEST] Cleared all queues")

    finally:
        await queue.disconnect()
        print("[TEST] Disconnected")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test_queue())
