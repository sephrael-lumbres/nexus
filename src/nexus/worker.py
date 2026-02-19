"""Worker pool for processing jobs from the queue.

This module provides:
- Worker class for individual job processing
- WorkerPool for managing multiple concurrent workers
- Retry logic with exponential backoff
- Dead letter queue integration
- Graceful shutdown handling

Architecture:
    WorkerPool
        └── Worker 1 ──┐
        └── Worker 2 ──┼── Queue ── Database
        └── Worker 3 ──┘      │
                              └── Handlers ── Providers

Usage:
    # Run worker pool (typically from command line)
    python -m nexus.worker 3  # Start 3 workers

    # Or programmatically
    pool = WorkerPool(num_workers=3)
    await pool.start()
"""

import asyncio
import os
import random
import signal
import sys
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import structlog

from nexus.config import get_settings
from nexus.database import Database, JobRepository, get_database
from nexus.handlers import HandlerResult, get_handler
from nexus.metrics import get_metrics
from nexus.models import JobRecord, JobStatus, JobType
from nexus.queue import JobQueue, get_queue

logger = structlog.get_logger()


# =============================================================================
# Worker Class
# =============================================================================
class Worker:
    """Individual worker that processes jobs from the queue.

    Each worker:
    1. Polls the queue for jobs (blocking)
    2. Claims the job in the database
    3. Executes the appropriate handler
    4. Updates job status (completed/failed)
    5. Handles retries with exponential backoff
    6. Moves exhausted jobs to DLQ

    Workers are designed to be resilient:
    - Graceful shutdown on SIGTERM/SIGINT
    - Automatic reconnection on database/queue errors
    - Detailed logging for debugging
    """

    def __init__(
        self,
        worker_id: str | None = None,
        db: Database | None = None,
        queue: JobQueue | None = None,
    ):
        """Initialize worker.

        Args:
            worker_id: Unique identifier for this worker (auto-generated if not provided)
            db: Database instance (uses singleton if not provided)
            queue: Queue instance (uses singleton if not provided)
        """
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self.settings = get_settings()
        self.db = db or get_database()
        self.queue = queue or get_queue()
        self.metrics = get_metrics()

        # State
        self.running = False
        self.current_job_id: UUID | None = None
        self.jobs_processed = 0
        self.jobs_failed = 0

        # Bind worker_id to logger for all log messages
        self.logger = logger.bind(worker_id=self.worker_id)

    async def start(self) -> None:
        """Start the worker loop.

        Runs continuously until stop() is called or a fatal error occurs.
        """
        self.running = True
        self.logger.info("Worker started")

        # Ensure queue is connected
        await self.queue.connect()

        while self.running:
            try:
                await self._process_next_job()
            except asyncio.CancelledError:
                self.logger.info("Worker cancelled")
                break
            except Exception as e:
                self.logger.error(
                    "Worker loop error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Back off on unexpected errors
                await asyncio.sleep(1.0)

        self.logger.info(
            "Worker stopped",
            jobs_processed=self.jobs_processed,
            jobs_failed=self.jobs_failed,
        )

    async def stop(self) -> None:
        """Stop the worker gracefully.

        Allows current job to complete before stopping.
        """
        self.running = False
        self.logger.info("Worker stopping")

    async def _process_next_job(self) -> None:
        """Dequeue and process the next job.

        This is the main processing loop that:
        1. Waits for a job from the queue
        2. Claims and processes it
        3. Handles success/failure
        """
        # Wait for job from queue (blocking with timeout)
        job_id = await self.queue.dequeue(
            timeout=self.settings.poll_interval_seconds
        )

        if job_id is None:
            # No job available, loop will retry
            return

        self.current_job_id = job_id

        try:
            await self._execute_job(job_id)
        finally:
            self.current_job_id = None

    async def _execute_job(self, job_id: UUID) -> None:
        """Execute a single job.

        Args:
            job_id: UUID of the job to execute
        """
        start_time = time.time()

        # Session 1: Claim the job
        async with self.db.session() as session:
            repo = JobRepository(session)

            # Fetch job from database
            job = await repo.get(job_id)

            if job is None:
                self.logger.warning("Job not found in database", job_id=str(job_id))
                await self.queue.complete(job_id)
                return

            # Check if job was cancelled while in queue
            if job.status == JobStatus.CANCELLED.value:
                self.logger.info("Skipping cancelled job", job_id=str(job_id))
                await self.queue.complete(job_id)
                return

            # Calculate and record job wait time for metrics
            if job.created_at:
                wait_seconds = (datetime.now(UTC) - job.created_at).total_seconds()
                self.metrics.record_job_wait_time(str(job.job_type), wait_seconds)

            # Update job to running state
            job.status = JobStatus.RUNNING.value  # type: ignore[assignment]
            job.worker_id = self.worker_id        # type: ignore[assignment]
            job.started_at = datetime.now(UTC)    # type: ignore[assignment]
            job.attempt += 1                      # type: ignore[assignment]
            await repo.update(job)
            # Session commits here and RUNNING job status is now visible

            # Record that this worker has picked up a job
            self.metrics.record_job_started(
                job_type=str(job.job_type),
                worker_id=self.worker_id,
            )

            self.logger.info(
                "Processing job",
                job_id=str(job_id),
                job_type=job.job_type,
                attempt=job.attempt,
                max_attempts=job.max_attempts,
            )

        # Get and execute handler (outside any session)
        try:
            handler = get_handler(JobType(job.job_type))

            # Execute with timeout
            result = await asyncio.wait_for(
                handler.execute(job),
                timeout=self.settings.job_timeout_seconds,
            )
        except TimeoutError:
            await self._handle_timeout(repo, job)
            return
        except Exception as e:
            await self._handle_error(repo, job, e)
            return

        # Session 2: Record the result
        async with self.db.session() as session:
            repo = JobRepository(session)
            job = await repo.get(job_id)
            if job is None:
                self.logger.warning("Job not found when recording result", job_id=str(job_id))
                return
            await self._handle_result(repo, job, result)
            # Session commits here and COMPLETED/FAILED job status is now visible
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.debug(
                "Job processing complete",
                job_id=str(job_id),
                duration_ms=duration_ms,
            )

    async def _handle_result(
        self,
        repo: JobRepository,
        job: JobRecord,
        result: HandlerResult,
    ) -> None:
        """Handle the result of job execution.

        Args:
            repo: Job repository
            job: Job record
            result: Handler result
        """
        # Update metrics
        job.input_tokens = result.input_tokens    # type: ignore[assignment]
        job.output_tokens = result.output_tokens  # type: ignore[assignment]
        job.total_tokens = result.total_tokens    # type: ignore[assignment]
        job.cost_usd = result.cost_usd            # type: ignore[assignment]
        job.duration_ms = result.duration_ms      # type: ignore[assignment]
        job.completed_at = datetime.now(UTC)      # type: ignore[assignment]

        if result.success:
            # Job completed successfully
            job.status = JobStatus.COMPLETED.value  # type: ignore[assignment]
            job.result = result.result              # type: ignore[assignment]

            await self.queue.complete(job.id)  # type: ignore[arg-type]
            self.jobs_processed += 1

            # Record job completed metrics upon completion
            model = job.input_data.get("model", "unknown")
            duration_seconds = result.duration_ms / 1000.0 if result.duration_ms else 0.0
            self.metrics.record_job_completed(
                job_type=str(job.job_type),
                worker_id=self.worker_id,
                duration_seconds=duration_seconds,
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cost_usd=result.cost_usd or 0.0,
                model=model,
            )

            self.logger.info(
                "Job completed successfully",
                job_id=str(job.id),
                tokens=result.total_tokens,
                cost=result.cost_usd,
                duration_ms=result.duration_ms,
            )
        else:
            # Job failed
            job.error = result.error  # type: ignore[assignment]
            await self._handle_failure(repo, job, result.error)

    async def _handle_failure(
        self,
        repo: JobRepository,
        job: JobRecord,
        error: dict[str, Any] | None,
    ) -> None:
        """Handle job failure with retry logic.

        Args:
            repo: Job repository
            job: Job record
            error: Error details
        """
        will_retry = job.attempt < job.max_attempts
        error_type = error.get("type", "Unknown") if error else "Unknown"

        # Record failure metrics before branching on retry logic
        self.metrics.record_job_failed(
            job_type=str(job.job_type),
            worker_id=self.worker_id,
            error_type=error_type,
            will_retry=bool(will_retry),
        )

        if will_retry:
            # Retry with exponential backoff
            backoff = self._calculate_backoff(job.attempt)  # type: ignore[arg-type]

            job.status = JobStatus.PENDING.value  # type: ignore[assignment]
            job.error = error                     # type: ignore[assignment]

            self.logger.warning(
                "Job failed, will retry",
                job_id=str(job.id),
                attempt=job.attempt,
                max_attempts=job.max_attempts,
                backoff_seconds=backoff,
                error=error,
            )

            # Wait for backoff period
            await asyncio.sleep(backoff)

            # Requeue for retry
            await self.queue.requeue(job.id, to_front=True)  # type: ignore[arg-type]

        else:
            # Max retries exceeded, move to DLQ
            reason = f"Max retries ({job.max_attempts}) exceeded"

            job.status = JobStatus.DEAD.value        # type: ignore[assignment]
            job.moved_to_dlq_at = datetime.now(UTC)  # type: ignore[assignment]
            job.dlq_reason = reason                  # type: ignore[assignment]

            await self.queue.move_to_dlq(job.id, error=error, reason=reason)  # type: ignore[arg-type]

            self.jobs_failed += 1

            self.logger.error(
                "Job moved to DLQ",
                job_id=str(job.id),
                reason=reason,
                error=error,
            )

    async def _handle_timeout(
        self,
        repo: JobRepository,
        job: JobRecord,
    ) -> None:
        """Handle job timeout.

        Args:
            repo: Job repository
            job: Job record
        """
        error = {
            "type": "TimeoutError",
            "message": f"Job exceeded timeout of {self.settings.job_timeout_seconds} seconds",
        }

        job.error = error                     # type: ignore[assignment]
        job.completed_at = datetime.now(UTC)  # type: ignore[assignment]

        self.logger.error(
            "Job timed out",
            job_id=str(job.id),
            timeout_seconds=self.settings.job_timeout_seconds,
        )

        await self._handle_failure(repo, job, error)

    async def _handle_error(
        self,
        repo: JobRepository,
        job: JobRecord,
        exception: Exception,
    ) -> None:
        """Handle unexpected error during job execution.

        Args:
            repo: Job repository
            job: Job record
            exception: The exception that occurred
        """
        error = {
            "type": type(exception).__name__,
            "message": str(exception),
        }

        job.error = error                     # type: ignore[assignment]
        job.completed_at = datetime.now(UTC)  # type: ignore[assignment]

        self.logger.error(
            "Job execution error",
            job_id=str(job.id),
            error_type=type(exception).__name__,
            error=str(exception),
        )

        await self._handle_failure(repo, job, error)

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Backoff duration in seconds
        """
        # Base backoff: 2^attempt seconds
        base_backoff = 2 ** attempt

        # Cap at 60 seconds
        capped_backoff = min(base_backoff, 60)

        # Add random jitter (0-1 seconds)
        jitter = random.random()

        return float(capped_backoff + jitter)


# =============================================================================
# Worker Pool Class
# =============================================================================
class WorkerPool:
    """Manages a pool of workers for concurrent job processing.

    The pool:
    - Starts multiple workers
    - Handles graceful shutdown
    - Monitors worker health
    - Provides statistics

    Usage:
        pool = WorkerPool(num_workers=3)

        # In an async context:
        await pool.start()  # Blocks until shutdown

        # Or with signal handling:
        asyncio.run(run_worker_pool(3))
    """

    def __init__(
        self,
        num_workers: int | None = None,
        db: Database | None = None,
        queue: JobQueue | None = None,
    ):
        """Initialize worker pool.

        Args:
            num_workers: Number of workers (defaults to settings)
            db: Database instance (shared across workers)
            queue: Queue instance (shared across workers)
        """
        self.num_workers = num_workers or get_settings().worker_count
        self.db = db or get_database()
        self.queue = queue or get_queue()

        self.workers: list[Worker] = []
        self.tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._started = False

    async def start(self) -> None:
        """Start the worker pool.

        Creates and starts all workers, then waits for shutdown signal.
        """
        if self._started:
            raise RuntimeError("Worker pool already started")

        self._started = True

        logger.info(
            "Starting worker pool",
            num_workers=self.num_workers,
        )

        # Ensure queue is connected
        await self.queue.connect()

        # Create and start workers
        for i in range(self.num_workers):
            worker = Worker(
                worker_id=f"worker-{i}",
                db=self.db,
                queue=self.queue,
            )
            self.workers.append(worker)

            task = asyncio.create_task(
                worker.start(),
                name=f"worker-{i}",
            )
            self.tasks.append(task)

        # Update active worker count metrics gauge on startup
        get_metrics().update_worker_count(len(self.workers))

        logger.info(
            "Worker pool started",
            num_workers=len(self.workers),
        )

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop all workers gracefully.

        Signals all workers to stop and waits for them to finish
        their current jobs.
        """
        logger.info("Stopping worker pool")

        # Signal all workers to stop
        for worker in self.workers:
            await worker.stop()

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete (with timeout)
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Disconnect from queue
        await self.queue.disconnect()

        # Update worker count metrics gauge to 0 on shutdown
        get_metrics().update_worker_count(0)

        # Set shutdown event
        self._shutdown_event.set()

        # Log final statistics
        total_processed = sum(w.jobs_processed for w in self.workers)
        total_failed = sum(w.jobs_failed for w in self.workers)

        logger.info(
            "Worker pool stopped",
            total_processed=total_processed,
            total_failed=total_failed,
        )

    def shutdown(self) -> None:
        """Trigger shutdown from signal handler.

        This is safe to call from a signal handler context.
        """
        asyncio.create_task(self.stop())

    def get_stats(self) -> dict[str, Any]:
        """Get worker pool statistics.

        Returns:
            Dict with pool statistics
        """
        return {
            "num_workers": self.num_workers,
            "active_workers": len([w for w in self.workers if w.running]),
            "total_processed": sum(w.jobs_processed for w in self.workers),
            "total_failed": sum(w.jobs_failed for w in self.workers),
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "running": w.running,
                    "jobs_processed": w.jobs_processed,
                    "jobs_failed": w.jobs_failed,
                    "current_job": str(w.current_job_id) if w.current_job_id else None,
                }
                for w in self.workers
            ],
        }


# =============================================================================
# Main Entry Point
# =============================================================================
async def run_worker_pool(num_workers: int | None = None) -> None:
    """Run the worker pool with signal handling.

    This is the main entry point for running workers.
    Handles SIGTERM and SIGINT for graceful shutdown.

    Args:
        num_workers: Number of workers (defaults to settings)
    """
    pool = WorkerPool(num_workers)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, pool.shutdown)

    try:
        await pool.start()
    except asyncio.CancelledError:
        pass
    finally:
        if pool._started and not pool._shutdown_event.is_set():
            await pool.stop()


def configure_logging() -> None:
    """Configure structured logging for workers."""
    import logging

    # Configure structlog
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


def main() -> None:
    """Command-line entry point for worker pool.

    Usage:
        python -m nexus.worker [num_workers]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Nexus Worker Pool - Process jobs from the queue",
    )
    parser.add_argument(
        "num_workers",
        type=int,
        nargs="?",
        default=None,
        help="Number of workers (default: from settings)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Override log level if specified
    if args.log_level:
        import logging
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Run the worker pool
    logger.info(
        "Starting Nexus Worker Pool",
        num_workers=args.num_workers or get_settings().worker_count,
        pid=os.getpid(),
    )

    try:
        asyncio.run(run_worker_pool(args.num_workers))
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")

    logger.info("Worker pool shutdown complete")


# =============================================================================
# Quick Test
# =============================================================================
async def _test_worker() -> None:
    """Quick test of worker functionality."""
    from nexus.database import reset_database
    from nexus.models import JobType
    from nexus.queue import reset_queue

    print("=" * 60)
    print("Testing Worker")
    print("=" * 60)

    # Reset singletons for clean test
    reset_database()
    reset_queue()

    db = get_database()
    queue = get_queue()

    # Connect queue
    await queue.connect()

    # Clear any existing data
    await queue.clear_all()

    # Create a test job
    async with db.session() as session:
        repo = JobRepository(session)

        job = JobRecord(
            job_type=JobType.LLM_COMPLETION.value,
            input_data={
                "prompt": "What is Python?",
                "model": "gpt-4o-mini",
                "max_tokens": 100,
            },
        )
        job = await repo.create(job)
        job_id = job.id
        print(f"Created job: {job_id}")

        # Enqueue job
        await queue.enqueue(job_id)  # type: ignore[arg-type]
        print("Enqueued job")

    # Create worker
    worker = Worker(worker_id="test-worker", db=db, queue=queue)

    # Process one job
    print("Processing job...")
    await worker._process_next_job()

    # Check result
    async with db.session() as session:
        repo = JobRepository(session)
        job = await repo.get(job_id)  # type: ignore[arg-type,assignment]

        if job is None:
            print(f"ERROR: Job {job_id} not found!")
            return

        print(f"Job status: {job.status}")
        print(f"Job result: {job.result}")
        print(f"Tokens: {job.total_tokens}")
        print(f"Cost: ${job.cost_usd:.6f}")

    # Cleanup
    await queue.clear_all()
    await queue.disconnect()

    print("\n" + "=" * 60)
    print("Worker test complete!")
    print("=" * 60)


if __name__ == "__main__":
    from nexus.config import disable_logging

    # If run directly, check for --test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if "--quiet" in sys.argv:
            import logging
            logging.disable(logging.CRITICAL)  # Silences SQLAlchemy
            disable_logging()                  # Silences structlog

        asyncio.run(_test_worker())
    else:
        main()
