"""Prometheus metrics instrumentation for Nexus.

This module provides:
- Job processing metrics (submitted, completed, failed)
- Queue depth gauges
- Processing duration histograms
- Token usage counters
- Cost tracking
- Worker status gauges

Metrics are exposed at /metrics endpoint for Prometheus scraping.

Usage:
    from nexus.metrics import metrics

    # Record job submission
    metrics.jobs_submitted.labels(job_type="llm.completion").inc()

    # Record processing duration
    with metrics.job_duration.labels(job_type="llm.completion").time():
        await process_job()

    # Update queue depth
    metrics.queue_depth.labels(queue="pending").set(42)
"""

import asyncio
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

import structlog
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = structlog.get_logger()


# =============================================================================
# Metrics Registry
# =============================================================================
class NexusMetrics:
    """Centralized metrics registry for Nexus.

    All Prometheus metrics are defined here for consistency
    and easy discovery. Metrics follow naming conventions:
    - nexus_<component>_<metric>_<unit>

    Example metric names:
    - nexus_jobs_submitted_total
    - nexus_job_duration_seconds
    - nexus_queue_depth
    """

    def __init__(self, registry: CollectorRegistry = REGISTRY):
        """Initialize all metrics.

        Args:
            registry: Prometheus registry to use (default: global)
        """
        self.registry = registry

        # =====================================================================
        # Job Metrics
        # =====================================================================
        self.jobs_submitted = Counter(
            "nexus_jobs_submitted_total",
            "Total number of jobs submitted",
            labelnames=["job_type"],
            registry=registry,
        )

        self.jobs_completed = Counter(
            "nexus_jobs_completed_total",
            "Total number of jobs completed successfully",
            labelnames=["job_type"],
            registry=registry,
        )

        self.jobs_failed = Counter(
            "nexus_jobs_failed_total",
            "Total number of jobs that failed",
            labelnames=["job_type", "error_type"],
            registry=registry,
        )

        self.jobs_retried = Counter(
            "nexus_jobs_retried_total",
            "Total number of job retry attempts",
            labelnames=["job_type"],
            registry=registry,
        )

        self.jobs_dlq = Counter(
            "nexus_jobs_dlq_total",
            "Total number of jobs moved to dead letter queue",
            labelnames=["job_type"],
            registry=registry,
        )

        self.jobs_in_progress = Gauge(
            "nexus_jobs_in_progress",
            "Number of jobs currently being processed",
            labelnames=["job_type", "worker_id"],
            registry=registry,
        )

        # =====================================================================
        # Duration Metrics
        # =====================================================================
        self.job_duration_seconds = Histogram(
            "nexus_job_duration_seconds",
            "Job processing duration in seconds",
            labelnames=["job_type"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=registry,
        )

        self.job_wait_seconds = Histogram(
            "nexus_job_wait_seconds",
            "Time jobs spend waiting in queue before processing",
            labelnames=["job_type"],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0),
            registry=registry,
        )

        # =====================================================================
        # Queue Metrics
        # =====================================================================
        self.queue_depth = Gauge(
            "nexus_queue_depth",
            "Number of jobs in queue",
            labelnames=["queue"],
            registry=registry,
        )

        self.queue_operations = Counter(
            "nexus_queue_operations_total",
            "Total queue operations",
            labelnames=["operation"],  # enqueue, dequeue, complete, fail
            registry=registry,
        )

        # =====================================================================
        # Token Metrics
        # =====================================================================
        self.tokens_processed = Counter(
            "nexus_tokens_processed_total",
            "Total tokens processed",
            labelnames=["job_type", "direction"],  # direction: input, output
            registry=registry,
        )

        self.tokens_per_job = Histogram(
            "nexus_tokens_per_job",
            "Tokens used per job",
            labelnames=["job_type"],
            buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
            registry=registry,
        )

        # =====================================================================
        # Cost Metrics
        # =====================================================================
        self.cost_usd = Counter(
            "nexus_cost_usd_total",
            "Total cost in USD",
            labelnames=["job_type", "model"],
            registry=registry,
        )

        # =====================================================================
        # Worker Metrics
        # =====================================================================
        self.workers_active = Gauge(
            "nexus_workers_active",
            "Number of active workers",
            registry=registry,
        )

        self.worker_jobs_processed = Counter(
            "nexus_worker_jobs_processed_total",
            "Total jobs processed by worker",
            labelnames=["worker_id"],
            registry=registry,
        )

        self.worker_last_heartbeat = Gauge(
            "nexus_worker_last_heartbeat_timestamp",
            "Last heartbeat timestamp for worker",
            labelnames=["worker_id"],
            registry=registry,
        )

        # =====================================================================
        # API Metrics
        # =====================================================================
        self.http_requests = Counter(
            "nexus_http_requests_total",
            "Total HTTP requests",
            labelnames=["method", "endpoint", "status_code"],
            registry=registry,
        )

        self.http_request_duration_seconds = Histogram(
            "nexus_http_request_duration_seconds",
            "HTTP request duration in seconds",
            labelnames=["method", "endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=registry,
        )

        self.rate_limit_hits = Counter(
            "nexus_rate_limit_hits_total",
            "Total rate limit hits",
            labelnames=["endpoint"],
            registry=registry,
        )

        # =====================================================================
        # System Info
        # =====================================================================
        self.info = Info(
            "nexus",
            "Nexus application information",
            registry=registry,
        )
        self.info.info({
            "version": "0.1.0",
            "component": "nexus",
        })

    # =========================================================================
    # Helper Methods
    # =========================================================================
    def record_job_submitted(self, job_type: str) -> None:
        """Record a job submission."""
        self.jobs_submitted.labels(job_type=job_type).inc()
        self.queue_operations.labels(operation="enqueue").inc()

    def record_job_started(self, job_type: str, worker_id: str) -> None:
        """Record a job starting processing."""
        self.jobs_in_progress.labels(job_type=job_type, worker_id=worker_id).inc()
        self.queue_operations.labels(operation="dequeue").inc()

    def record_job_completed(
        self,
        job_type: str,
        worker_id: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        model: str,
    ) -> None:
        """Record a successful job completion."""
        self.jobs_completed.labels(job_type=job_type).inc()
        self.jobs_in_progress.labels(job_type=job_type, worker_id=worker_id).dec()
        self.job_duration_seconds.labels(job_type=job_type).observe(duration_seconds)
        self.tokens_processed.labels(job_type=job_type, direction="input").inc(input_tokens)
        self.tokens_processed.labels(job_type=job_type, direction="output").inc(output_tokens)
        self.tokens_per_job.labels(job_type=job_type).observe(input_tokens + output_tokens)
        self.cost_usd.labels(job_type=job_type, model=model).inc(cost_usd)
        self.worker_jobs_processed.labels(worker_id=worker_id).inc()
        self.queue_operations.labels(operation="complete").inc()

    def record_job_failed(
        self,
        job_type: str,
        worker_id: str,
        error_type: str,
        will_retry: bool,
    ) -> None:
        """Record a job failure."""
        self.jobs_in_progress.labels(job_type=job_type, worker_id=worker_id).dec()

        if will_retry:
            self.jobs_retried.labels(job_type=job_type).inc()
        else:
            self.jobs_failed.labels(job_type=job_type, error_type=error_type).inc()
            self.jobs_dlq.labels(job_type=job_type).inc()

        self.queue_operations.labels(operation="fail").inc()

    def record_job_wait_time(self, job_type: str, wait_seconds: float) -> None:
        """Record time a job spent waiting in queue."""
        self.job_wait_seconds.labels(job_type=job_type).observe(wait_seconds)

    def update_queue_depths(
        self,
        pending: int,
        processing: int,
        dlq: int,
    ) -> None:
        """Update queue depth gauges."""
        self.queue_depth.labels(queue="pending").set(pending)
        self.queue_depth.labels(queue="processing").set(processing)
        self.queue_depth.labels(queue="dlq").set(dlq)

    def update_worker_count(self, count: int) -> None:
        """Update active worker count."""
        self.workers_active.set(count)

    def record_worker_heartbeat(self, worker_id: str) -> None:
        """Record worker heartbeat."""
        self.worker_last_heartbeat.labels(worker_id=worker_id).set_to_current_time()

    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record an HTTP request."""
        self.http_requests.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration_seconds)

    def record_rate_limit_hit(self, endpoint: str) -> None:
        """Record a rate limit hit."""
        self.rate_limit_hits.labels(endpoint=endpoint).inc()

    @contextmanager
    def track_job_duration(self, job_type: str):
        """Context manager to track job duration.

        Usage:
            with metrics.track_job_duration("llm.completion"):
                await process_job()
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.job_duration_seconds.labels(job_type=job_type).observe(duration)

    def track_request_duration(self, method: str, endpoint: str):
        """Decorator to track HTTP request duration.

        Usage:
            @metrics.track_request_duration("POST", "/jobs")
            async def submit_job():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start
                    self.http_request_duration_seconds.labels(
                        method=method,
                        endpoint=endpoint,
                    ).observe(duration)
            return wrapper
        return decorator


# =============================================================================
# Global Metrics Instance
# =============================================================================
# Singleton metrics instance
metrics = NexusMetrics()

def get_metrics() -> NexusMetrics:
    """Get the global metrics instance."""
    return metrics


def generate_metrics() -> bytes:
    """Generate metrics in Prometheus format.

    Returns:
        Metrics data as bytes in Prometheus exposition format
    """
    return generate_latest(REGISTRY)


def get_content_type() -> str:
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST


# =============================================================================
# Metrics Middleware for FastAPI
# =============================================================================
class MetricsMiddleware:
    """ASGI middleware for tracking HTTP request metrics.

    Automatically records:
    - Request count by method, endpoint, status
    - Request duration histogram

    Usage:
        from nexus.metrics import MetricsMiddleware
        app.add_middleware(MetricsMiddleware)
    """

    def __init__(self, app):
        self.app = app
        self.metrics = get_metrics()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 500  # Default in case of error

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start_time

            # Extract endpoint (simplify path for metrics)
            path = scope.get("path", "/")
            method = scope.get("method", "GET")

            # Normalize paths with IDs to avoid high cardinality
            endpoint = self._normalize_path(path)

            self.metrics.record_http_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration_seconds=duration,
            )

    def _normalize_path(self, path: str) -> str:
        """Normalize path to avoid high cardinality metrics.

        Replaces UUIDs and numeric IDs with placeholders.
        """
        import re

        # Replace UUIDs
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path,
            flags=re.IGNORECASE,
        )

        # Replace numeric IDs
        path = re.sub(r"/\d+(/|$)", r"/{id}\1", path)

        return path


# =============================================================================
# Background Metrics Collector
# =============================================================================
class MetricsCollector:
    """Background task to periodically collect queue metrics.

    Updates queue depth gauges at regular intervals.

    Usage:
        collector = MetricsCollector(queue)
        await collector.start()
        # ... later ...
        await collector.stop()
    """

    def __init__(self, queue, interval_seconds: float = 15.0):
        """
        Initialize collector.

        Args:
            queue: JobQueue instance to collect metrics from
            interval_seconds: Collection interval
        """
        self.queue = queue
        self.interval = interval_seconds
        self.metrics = get_metrics()
        self._task = None
        self._running = False

    async def start(self) -> None:
        """Start the background collector."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info("Metrics collector started", interval=self.interval)

    async def stop(self) -> None:
        """Stop the background collector."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")

    async def _collect_loop(self) -> None:
        """Main collection loop."""
        import asyncio

        while self._running:
            try:
                await self._collect()
            except Exception as e:
                logger.error("Error collecting metrics", error=str(e))

            await asyncio.sleep(self.interval)

    async def _collect(self) -> None:
        """Collect metrics from queue."""
        try:
            pending = await self.queue.pending_count()
            processing = await self.queue.processing_count()
            dlq = await self.queue.dlq_count()

            self.metrics.update_queue_depths(
                pending=pending,
                processing=processing,
                dlq=dlq,
            )
        except Exception as e:
            logger.warning("Failed to collect queue metrics", error=str(e))


# =============================================================================
# Quick Test
# =============================================================================
def _test_metrics() -> None:
    """Quick test of metrics functionality."""
    print("=" * 60)
    print("Testing Metrics")
    print("=" * 60)

    m = get_metrics()

    # Record some metrics
    m.record_job_submitted("llm.completion")
    m.record_job_submitted("llm.completion")
    m.record_job_submitted("llm.batch")

    m.record_job_started("llm.completion", "worker-1")

    m.record_job_completed(
        job_type="llm.completion",
        worker_id="worker-1",
        duration_seconds=0.5,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        model="gpt-4o-mini",
    )

    m.record_job_failed(
        job_type="llm.completion",
        worker_id="worker-1",
        error_type="APIError",
        will_retry=True,
    )

    m.update_queue_depths(pending=10, processing=2, dlq=1)
    m.update_worker_count(3)

    # Generate output
    output = generate_metrics().decode("utf-8")

    print("\nGenerated Metrics:")
    print("-" * 60)

    # Print a subset of interesting metrics
    for line in output.split("\n"):
        if line and not line.startswith("#"):
            if any(x in line for x in ["submitted", "completed", "queue_depth", "workers"]):
                print(line)

    print("\n" + "=" * 60)
    print("Metrics test complete!")
    print("=" * 60)


if __name__ == "__main__":
    _test_metrics()
