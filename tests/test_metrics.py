"""Tests for Prometheus metrics instrumentation.

These tests verify metric recording and generation.
"""

import pytest
from prometheus_client import CollectorRegistry

from nexus.metrics import (
    MetricsMiddleware,
    NexusMetrics,
    generate_metrics,
    get_metrics,
)


class TestNexusMetrics:
    """Tests for NexusMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Get a fresh metrics instance with isolated registry."""
        registry = CollectorRegistry()
        return NexusMetrics(registry=registry)

    def test_record_job_submitted(self, metrics: NexusMetrics):
        """Test recording job submission."""
        metrics.record_job_submitted("llm.completion")
        metrics.record_job_submitted("llm.completion")
        metrics.record_job_submitted("llm.batch")

        # Check counter values
        assert metrics.jobs_submitted.labels(job_type="llm.completion")._value.get() == 2
        assert metrics.jobs_submitted.labels(job_type="llm.batch")._value.get() == 1

    def test_record_job_completed(self, metrics: NexusMetrics):
        """Test recording job completion."""
        metrics.record_job_completed(
            job_type="llm.completion",
            worker_id="worker-1",
            duration_seconds=0.5,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            model="gpt-4o-mini",
        )

        assert metrics.jobs_completed.labels(job_type="llm.completion")._value.get() == 1
        assert metrics.tokens_processed.labels(
            job_type="llm.completion",
            direction="input"
        )._value.get() == 100
        assert metrics.tokens_processed.labels(
            job_type="llm.completion",
            direction="output"
        )._value.get() == 50

    def test_record_job_failed_with_retry(self, metrics: NexusMetrics):
        """Test recording job failure with retry."""
        metrics.record_job_started("llm.completion", "worker-1")

        metrics.record_job_failed(
            job_type="llm.completion",
            worker_id="worker-1",
            error_type="APIError",
            will_retry=True,
        )

        assert metrics.jobs_retried.labels(job_type="llm.completion")._value.get() == 1
        assert metrics.jobs_failed.labels(
            job_type="llm.completion",
            error_type="APIError"
        )._value.get() == 0

    def test_record_job_failed_to_dlq(self, metrics: NexusMetrics):
        """Test recording job failure to DLQ."""
        metrics.record_job_started("llm.completion", "worker-1")

        metrics.record_job_failed(
            job_type="llm.completion",
            worker_id="worker-1",
            error_type="APIError",
            will_retry=False,
        )

        assert metrics.jobs_failed.labels(
            job_type="llm.completion",
            error_type="APIError"
        )._value.get() == 1
        assert metrics.jobs_dlq.labels(job_type="llm.completion")._value.get() == 1

    def test_update_queue_depths(self, metrics: NexusMetrics):
        """Test updating queue depth gauges."""
        metrics.update_queue_depths(pending=10, processing=5, dlq=2)

        assert metrics.queue_depth.labels(queue="pending")._value.get() == 10
        assert metrics.queue_depth.labels(queue="processing")._value.get() == 5
        assert metrics.queue_depth.labels(queue="dlq")._value.get() == 2

    def test_update_worker_count(self, metrics: NexusMetrics):
        """Test updating worker count."""
        metrics.update_worker_count(3)

        assert metrics.workers_active._value.get() == 3

    def test_record_http_request(self, metrics: NexusMetrics):
        """Test recording HTTP request."""
        metrics.record_http_request(
            method="POST",
            endpoint="/jobs",
            status_code=201,
            duration_seconds=0.05,
        )

        assert metrics.http_requests.labels(
            method="POST",
            endpoint="/jobs",
            status_code="201"
        )._value.get() == 1

    def test_record_rate_limit_hit(self, metrics: NexusMetrics):
        """Test recording rate limit hit."""
        metrics.record_rate_limit_hit("/jobs")
        metrics.record_rate_limit_hit("/jobs")

        assert metrics.rate_limit_hits.labels(endpoint="/jobs")._value.get() == 2


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    def test_normalize_path_with_uuid(self):
        """Test path normalization with UUID."""
        middleware = MetricsMiddleware(None)

        path = "/jobs/123e4567-e89b-12d3-a456-426614174000"
        normalized = middleware._normalize_path(path)

        assert normalized == "/jobs/{id}"

    def test_normalize_path_with_numeric_id(self):
        """Test path normalization with numeric ID."""
        middleware = MetricsMiddleware(None)

        path = "/users/12345/orders"
        normalized = middleware._normalize_path(path)

        assert normalized == "/users/{id}/orders"

    def test_normalize_path_no_id(self):
        """Test path normalization without ID."""
        middleware = MetricsMiddleware(None)

        path = "/health"
        normalized = middleware._normalize_path(path)

        assert normalized == "/health"


class TestGenerateMetrics:
    """Tests for metrics generation."""

    def test_generate_metrics_returns_bytes(self):
        """Test that generate_metrics returns bytes."""
        output = generate_metrics()

        assert isinstance(output, bytes)

    def test_generate_metrics_contains_nexus_metrics(self):
        """Test that output contains nexus metrics."""
        # Record some metrics first
        metrics = get_metrics()
        metrics.record_job_submitted("test")

        output = generate_metrics().decode("utf-8")

        assert "nexus_jobs_submitted_total" in output

    def test_generate_metrics_prometheus_format(self):
        """Test that output is valid Prometheus format."""
        output = generate_metrics().decode("utf-8")

        # Should have HELP and TYPE comments
        assert "# HELP" in output
        assert "# TYPE" in output
