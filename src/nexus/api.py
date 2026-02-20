"""FastAPI REST API for the Nexus job queue.

This module provides:
- Job submission and management endpoints
- Rate limiting with slowapi
- Health check endpoints
- Statistics and monitoring endpoints
- Dead letter queue management

Usage:
    # Run with uvicorn
    uvicorn nexus.api:app --reload

    # Or programmatically
    import uvicorn
    uvicorn.run("nexus.api:app", host="0.0.0.0", port=8000)
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import UUID

import structlog
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from nexus.config import get_settings
from nexus.database import Database, JobRepository, get_database
from nexus.metrics import (
    MetricsCollector,
    MetricsMiddleware,
    generate_metrics,
    get_content_type,
    get_metrics,
)
from nexus.models import (
    HealthResponse,
    JobCreate,
    JobRecord,
    JobResponse,
    JobStatus,
    JobType,
    StatsResponse,
)
from nexus.queue import JobQueue, get_queue

logger = structlog.get_logger()

# =============================================================================
# Rate Limiter Setup
# =============================================================================
limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# Application Lifespan
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Handles startup and shutdown events:
    - Startup: Connect to database and queue, start metrics collector
    - Shutdown: Clean up connections, stop metrics collector
    """
    # Startup
    logger.info("Starting Nexus API")

    settings = get_settings()

    # Connect to queue
    queue = get_queue()
    await queue.connect()

    # Verify database connection
    db = get_database()
    if await db.health_check():
        logger.info("Database connection verified")
    else:
        logger.error("Database connection failed")

    # Start metrics collector
    metrics_collector = MetricsCollector(queue)
    await metrics_collector.start()

    logger.info(
        "Nexus API started",
        environment=settings.environment.value,
        rate_limit=f"{settings.rate_limit_per_minute}/minute",
    )

    yield

    # Shutdown
    logger.info("Shutting down Nexus API")
    await metrics_collector.stop()
    await queue.disconnect()
    await db.close()
    logger.info("Nexus API shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="Nexus Job Queue API",
    description="AI-native distributed job queue for LLM workloads",
    version="0.1.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)


# =============================================================================
# Response Models
# =============================================================================
class JobSubmitResponse(BaseModel):
    """Response after submitting a job."""
    id: UUID
    job_type: JobType
    status: JobStatus
    message: str = "Job submitted successfully"

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    """Response for job listing."""
    jobs: list[JobResponse]
    total: int
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str | None = None


class DLQEntry(BaseModel):
    """Dead letter queue entry."""
    job_id: str
    error: dict[str, Any] | None
    reason: str | None
    moved_at: str


class DLQListResponse(BaseModel):
    """Response for DLQ listing."""
    entries: list[DLQEntry]
    total: int


class ReplayResponse(BaseModel):
    """Response after replaying a job from DLQ."""
    job_id: UUID
    status: JobStatus
    message: str


# =============================================================================
# Helper Functions
# =============================================================================
def get_db() -> Database:
    """Get database instance."""
    return get_database()


def get_q() -> JobQueue:
    """Get queue instance."""
    return get_queue()


def job_to_response(job: JobRecord) -> JobResponse:
    """Convert JobRecord to JobResponse."""
    return JobResponse.model_validate(job)


# =============================================================================
# Health Endpoints
# =============================================================================
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Check API and dependency health.

    Returns health status of:
    - API service
    - Database connection
    - Redis connection
    """
    db = get_db()
    queue = get_q()

    db_healthy = await db.health_check()
    redis_healthy = await queue.health_check()

    overall_status = "healthy" if (db_healthy and redis_healthy) else "unhealthy"

    return HealthResponse(
        status=overall_status,
        database="connected" if db_healthy else "disconnected",
        redis="connected" if redis_healthy else "disconnected",
    )


@app.get(
    "/",
    tags=["Health"],
    summary="Root endpoint",
)
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "service": "Nexus Job Queue API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# Metrics Endpoint
# =============================================================================
@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Prometheus metrics",
    response_class=Response,
)
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics.

    This endpoint is scraped by Prometheus to collect metrics.
    Returns metrics in Prometheus exposition format.
    """
    from fastapi.responses import Response

    return Response(
        content=generate_metrics(),
        media_type=get_content_type(),
    )


# =============================================================================
# Job Endpoints
# =============================================================================
@app.post(
    "/jobs",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Jobs"],
    summary="Submit a new job",
)
@limiter.limit(lambda: f"{get_settings().rate_limit_per_minute}/minute")
async def submit_job(request: Request, job_create: JobCreate) -> JobSubmitResponse:
    """Submit a new job to the queue.

    The job will be validated, stored in the database, and
    enqueued for processing by workers.

    **Job Types:**
    - `llm.completion`: Single prompt completion
    - `llm.batch`: Batch of prompts (concurrent processing)

    **Rate Limited:** {rate_limit}/minute per IP
    """
    db = get_db()
    queue = get_q()
    metrics = get_metrics()

    async with db.session() as session:
        repo = JobRepository(session)

        # Create job record
        job = JobRecord(
            job_type=job_create.job_type.value,
            input_data=job_create.input_data,
            max_attempts=job_create.max_attempts,
        )
        job = await repo.create(job)
        # Capture job ID, job type, and response before session closes
        job_id = job.id
        job_type = job.job_type
        response = JobSubmitResponse.model_validate(job)
        # session context manager exits here, DB commit happens now

    # Enqueue for processing AFTER commit to ensure DB row is visible to workers
    await queue.enqueue(cast(UUID, job_id))

    # Record metrics
    metrics.record_job_submitted(str(job_type))

    logger.info(
        "Job submitted",
        job_id=str(job_id),
        job_type=job_type,
    )

    return response


@app.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    tags=["Jobs"],
    summary="Get job details",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def get_job(job_id: UUID) -> JobResponse:
    """Get details of a specific job.

    Returns the full job record including:
    - Current status
    - Input data
    - Result (if completed)
    - Error (if failed)
    - Metrics (tokens, cost, duration)
    """
    db = get_db()

    async with db.session() as session:
        repo = JobRepository(session)
        job = await repo.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )

        return job_to_response(job)


@app.get(
    "/jobs",
    response_model=JobListResponse,
    tags=["Jobs"],
    summary="List jobs",
)
async def list_jobs(
    status_filter: JobStatus | None = Query(
        None,
        alias="status",
        description="Filter by job status",
    ),
    job_type: JobType | None = Query(
        None,
        alias="type",
        description="Filter by job type",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=100,
        description="Maximum number of jobs to return",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of jobs to skip",
    ),
) -> JobListResponse:
    """List jobs with optional filtering.

    Supports filtering by:
    - Status (pending, running, completed, failed, dead)
    - Job type (llm.completion, llm.batch)

    Results are paginated and ordered by creation time (newest first).
    """
    db = get_db()

    async with db.session() as session:
        repo = JobRepository(session)

        jobs = await repo.list_jobs(
            status=status_filter,
            job_type=job_type.value if job_type else None,
            limit=limit,
            offset=offset,
        )

        # Get total count (simplified - in production, add a count query)
        all_jobs = await repo.list_jobs(
            status=status_filter,
            job_type=job_type.value if job_type else None,
            limit=10000,
            offset=0,
        )

        return JobListResponse(
            jobs=[job_to_response(j) for j in jobs],
            total=len(all_jobs),
            limit=limit,
            offset=offset,
        )


@app.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Jobs"],
    summary="Cancel a job",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Job cannot be cancelled"},
    },
)
async def cancel_job(job_id: UUID) -> None:
    """Cancel a pending job.

    Only jobs in 'pending' status can be cancelled.
    Running jobs will complete normally.
    """
    db = get_db()

    async with db.session() as session:
        repo = JobRepository(session)
        job = await repo.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )

        if job.status != JobStatus.PENDING.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot cancel job in '{job.status}' status",
            )

        job.status = JobStatus.CANCELLED.value  # type: ignore[assignment]
        await repo.update(job)

        logger.info("Job cancelled", job_id=str(job_id))


# =============================================================================
# Statistics Endpoints
# =============================================================================
@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Statistics"],
    summary="Get queue statistics",
)
async def get_stats() -> StatsResponse:
    """Get comprehensive queue statistics.

    Returns:
    - Total jobs processed
    - Jobs by status
    - Jobs by type
    - Token usage and costs
    - Success rate
    - Queue depth
    - DLQ count

    These metrics can be used for monitoring dashboards
    and are great for your resume!
    """
    db = get_db()
    queue = get_q()

    async with db.session() as session:
        repo = JobRepository(session)
        db_stats = await repo.get_stats()

    queue_stats = await queue.get_stats()

    return StatsResponse(
        total_jobs=db_stats["total_jobs"],
        jobs_by_status=db_stats["jobs_by_status"],
        jobs_by_type=db_stats["jobs_by_type"],
        total_tokens=db_stats["total_tokens"],
        total_cost_usd=db_stats["total_cost_usd"],
        avg_duration_ms=db_stats["avg_duration_ms"],
        success_rate=db_stats["success_rate"],
        queue_depth=queue_stats["pending"],
        dlq_count=queue_stats["dlq"],
    )


# =============================================================================
# Dead Letter Queue Endpoints
# =============================================================================
@app.get(
    "/dlq",
    response_model=DLQListResponse,
    tags=["Dead Letter Queue"],
    summary="List DLQ entries",
)
async def list_dlq(
    limit: int = Query(
        100,
        ge=1,
        le=1000,
        description="Maximum number of entries to return",
    ),
) -> DLQListResponse:
    """List jobs in the dead letter queue.

    DLQ contains jobs that failed after exhausting all retry attempts.
    Each entry includes the error and reason for failure.
    """
    queue = get_q()

    entries = await queue.get_dlq_entries(start=0, end=limit - 1)
    dlq_count = await queue.dlq_count()

    return DLQListResponse(
        entries=[
            DLQEntry(
                job_id=e["job_id"],
                error=e.get("error"),
                reason=e.get("reason"),
                moved_at=e.get("moved_at", ""),
            )
            for e in entries
        ],
        total=dlq_count,
    )


@app.post(
    "/dlq/{job_id}/replay",
    response_model=ReplayResponse,
    tags=["Dead Letter Queue"],
    summary="Replay a job from DLQ",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found in DLQ"},
    },
)
async def replay_dlq_job(job_id: UUID) -> ReplayResponse:
    """Replay a failed job from the dead letter queue.

    The job will be:
    1. Removed from the DLQ
    2. Reset to pending status
    3. Re-enqueued for processing

    Use this to retry jobs after fixing underlying issues.
    """
    db = get_db()
    queue = get_q()

    # Replay from queue DLQ
    replayed = await queue.replay_from_dlq(job_id)

    if not replayed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found in DLQ",
        )

    # Update database status
    async with db.session() as session:
        repo = JobRepository(session)
        job = await repo.get(job_id)

        if job:
            await repo.replay_from_dlq(job)

    logger.info("Job replayed from DLQ", job_id=str(job_id))

    return ReplayResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job replayed successfully",
    )


@app.delete(
    "/dlq",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Dead Letter Queue"],
    summary="Clear the DLQ",
)
async def clear_dlq() -> None:
    """Clear all entries from the dead letter queue.

    **Warning:** This permanently removes all DLQ entries.
    Use with caution.
    """
    queue = get_q()

    count = await queue.clear_dlq()

    logger.warning("DLQ cleared", entries_removed=count)


# =============================================================================
# Queue Management Endpoints
# =============================================================================
@app.get(
    "/queue/stats",
    tags=["Queue"],
    summary="Get queue statistics",
)
async def get_queue_stats() -> dict[str, Any]:
    """Get detailed queue statistics.

    Returns Redis queue metrics including:
    - Pending job count
    - Processing job count
    - DLQ count
    - Total enqueued/dequeued/completed
    """
    queue = get_q()
    return await queue.get_stats()


@app.get(
    "/queue/pending",
    tags=["Queue"],
    summary="Peek at pending jobs",
)
async def peek_pending(
    count: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """
    Peek at jobs waiting in the pending queue.

    Returns job IDs without removing them from the queue.
    """
    queue = get_q()

    job_ids = await queue.peek(count)

    return {
        "pending_count": await queue.pending_count(),
        "peeked": [str(job_id) for job_id in job_ids],
    }


# =============================================================================
# Error Handlers
# =============================================================================
@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if get_settings().environment.value == "development" else None,
        },
    )


# =============================================================================
# Run Server
# =============================================================================
def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """
    Run the API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    import uvicorn

    uvicorn.run(
        "nexus.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nexus API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=args.reload)
