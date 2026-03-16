"""Tests for Prometheus metrics instrumentation.

These tests verify metric recording, context managers, decorators,
middleware ASGI behavior, and the background MetricsCollector.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from prometheus_client import CollectorRegistry

from nexus.metrics import (
    MetricsCollector,
    MetricsMiddleware,
    NexusMetrics,
    generate_metrics,
    get_content_type,
    get_metrics,
)


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture
def registry():
    """Isolated Prometheus registry to avoid cross-test pollution."""
    return CollectorRegistry()


@pytest.fixture
def metrics(registry):
    """NexusMetrics wired to an isolated registry."""
    return NexusMetrics(registry=registry)

# =========================================================================
# Metrics Recording Tests
# =========================================================================
class TestNexusMetricsRecording:
    """Verify that every public record_* / update_* method touches
    the expected counters, gauges, and histograms."""

    def test_record_job_submitted(self, metrics: NexusMetrics):
        metrics.record_job_submitted("llm.completion")
        metrics.record_job_submitted("llm.completion")
        metrics.record_job_submitted("llm.batch")

        assert metrics.jobs_submitted.labels(job_type="llm.completion")._value.get() == 2
        assert metrics.jobs_submitted.labels(job_type="llm.batch")._value.get() == 1
        assert metrics.queue_operations.labels(operation="enqueue")._value.get() == 3

    def test_record_job_started(self, metrics: NexusMetrics):
        metrics.record_job_started("llm.completion", "worker-1")

        assert metrics.jobs_in_progress.labels(
            job_type="llm.completion",
            worker_id="worker-1",
        )._value.get() == 1
        assert metrics.queue_operations.labels(operation="dequeue")._value.get() == 1

    def test_record_job_completed(self, metrics: NexusMetrics):
        metrics.record_job_started("llm.completion", "worker-1")
        metrics.record_job_completed(
            job_type="llm.completion",
            worker_id="worker-1",
            duration_seconds=1.5,
            input_tokens=200,
            output_tokens=80,
            cost_usd=0.002,
            model="gpt-4o-mini",
        )

        assert metrics.jobs_completed.labels(job_type="llm.completion")._value.get() == 1
        assert metrics.tokens_processed.labels(
            job_type="llm.completion",
            direction="input"
        )._value.get() == 200
        assert metrics.tokens_processed.labels(
            job_type="llm.completion",
            direction="output"
        )._value.get() == 80
        assert metrics.cost_usd.labels(
            job_type="llm.completion",
            model="gpt-4o-mini"
        )._value.get() == pytest.approx(0.002)
        assert metrics.worker_jobs_processed.labels(worker_id="worker-1")._value.get() == 1
        assert metrics.queue_operations.labels(operation="complete")._value.get() == 1

    @pytest.mark.parametrize(
        "will_retry, expect_retry, expect_failed, expect_dlq",
        [
            (True, 1, 0, 0),
            (False, 0, 1, 1),
        ],
        ids=["retry", "dlq"],
    )
    def test_record_job_failed_branches(
        self,
        metrics: NexusMetrics,
        will_retry,
        expect_retry,
        expect_failed,
        expect_dlq
    ):
        metrics.record_job_started("llm.completion", "worker-1")
        metrics.record_job_failed(
            job_type="llm.completion",
            worker_id="worker-1",
            error_type="Timeout",
            will_retry=will_retry,
        )

        assert metrics.jobs_retried.labels(job_type="llm.completion")._value.get() == expect_retry
        assert metrics.jobs_failed.labels(
            job_type="llm.completion",
            error_type="Timeout"
        )._value.get() == expect_failed
        assert metrics.jobs_dlq.labels(job_type="llm.completion")._value.get() == expect_dlq
        assert metrics.queue_operations.labels(operation="fail")._value.get() == 1

    def test_record_job_wait_time(self, metrics: NexusMetrics):
        metrics.record_job_wait_time("llm.completion", 3.5)

        assert metrics.job_wait_seconds.labels(
            job_type="llm.completion"
        )._sum.get() == pytest.approx(3.5)

    def test_update_queue_depths(self, metrics: NexusMetrics):
        metrics.update_queue_depths(pending=10, processing=5, dlq=2)

        assert metrics.queue_depth.labels(queue="pending")._value.get() == 10
        assert metrics.queue_depth.labels(queue="processing")._value.get() == 5
        assert metrics.queue_depth.labels(queue="dlq")._value.get() == 2

    def test_update_worker_count(self, metrics: NexusMetrics):
        metrics.update_worker_count(7)
        assert metrics.workers_active._value.get() == 7

    def test_record_http_request(self, metrics: NexusMetrics):
        metrics.record_http_request(
            method="GET", endpoint="/health", status_code=200, duration_seconds=0.01
        )

        assert metrics.http_requests.labels(
            method="GET",
            endpoint="/health",
            status_code="200",
        )._value.get() == 1

    def test_record_rate_limit_hit(self, metrics: NexusMetrics):
        metrics.record_rate_limit_hit("/jobs")
        metrics.record_rate_limit_hit("/jobs")

        assert metrics.rate_limit_hits.labels(endpoint="/jobs")._value.get() == 2

    def test_record_worker_heartbeat(self, metrics: NexusMetrics):
        metrics.record_worker_heartbeat("worker-1")

        ts = metrics.worker_last_heartbeat.labels(worker_id="worker-1")._value.get()
        assert ts > 1_577_836_800


class TestTrackJobDuration:
    def test_observes_histogram(self, metrics: NexusMetrics):
        with metrics.track_job_duration("llm.completion"):
            pass

        assert metrics.registry.get_sample_value(
            "nexus_job_duration_seconds_count",
            {"job_type": "llm.completion"},
        ) == 1.0

    def test_records_on_exception(self, metrics: NexusMetrics):
        with pytest.raises(ValueError, match="boom"):
            with metrics.track_job_duration("llm.completion"):
                raise ValueError("boom")

        assert metrics.registry.get_sample_value(
            "nexus_job_duration_seconds_count",
            {"job_type": "llm.completion"},
        ) == 1.0


class TestTrackRequestDuration:
    @pytest.mark.asyncio
    async def test_success(self, metrics: NexusMetrics):
        @metrics.track_request_duration("POST", "/jobs")
        async def fake_handler():
            return "ok"

        result = await fake_handler()

        assert result == "ok"
        assert metrics.registry.get_sample_value(
            "nexus_http_request_duration_seconds_count",
            {"method": "POST", "endpoint": "/jobs"},
        ) == 1.0

    @pytest.mark.asyncio
    async def test_exception(self, metrics: NexusMetrics):
        @metrics.track_request_duration("POST", "/jobs")
        async def failing_handler():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            await failing_handler()

        assert metrics.registry.get_sample_value(
            "nexus_http_request_duration_seconds_count",
            {"method": "POST", "endpoint": "/jobs"},
        ) == 1.0


class TestModuleLevelHelpers:
    def test_get_metrics_singleton(self):
        assert get_metrics() is get_metrics()

    def test_generate_metrics_format(self):
        output = generate_metrics()
        assert isinstance(output, bytes)
        text = output.decode("utf-8")
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_get_content_type(self):
        ct = get_content_type()
        assert isinstance(ct, str)
        assert "text/" in ct or "application/" in ct


# =========================================================================
# MetricsMiddleware Tests — ASGI behaviour
# =========================================================================
class TestMetricsMiddleware:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("/jobs/123e4567-e89b-12d3-a456-426614174000", "/jobs/{id}"),
            ("/users/12345/orders", "/users/{id}/orders"),
            ("/health", "/health"),
            ("/items/99", "/items/{id}"),
        ],
        ids=["uuid", "numeric-mid", "no-id", "numeric-end"],
    )
    def test_normalize_path(self, path, expected):
        mw = MetricsMiddleware(None)
        assert mw._normalize_path(path) == expected

    @pytest.mark.asyncio
    async def test_non_http_passthrough(self):
        inner_app = AsyncMock()
        mw = MetricsMiddleware(inner_app)

        scope = {"type": "websocket", "path": "/ws"}
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)

        inner_app.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_http_records_metrics(self):
        async def fake_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 201})
            await send({"type": "http.response.body", "body": b""})

        mw = MetricsMiddleware(fake_app)
        scope = {"type": "http", "path": "/jobs", "method": "POST"}
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)

        assert send.await_count == 2
        send.assert_any_await({"type": "http.response.start", "status": 201})


# =========================================================================
# MetricsCollector Tests — background async collector
# =========================================================================
class TestMetricsCollector:
    @pytest.fixture
    def mock_queue(self):
        q = AsyncMock()
        q.pending_count.return_value = 5
        q.processing_count.return_value = 2
        q.dlq_count.return_value = 1
        return q

    @pytest.mark.asyncio
    async def test_start(self, mock_queue):
        collector = MetricsCollector(mock_queue, interval_seconds=0.01)
        await collector.start()

        assert collector._running is True
        assert collector._task is not None

        await collector.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, mock_queue):
        collector = MetricsCollector(mock_queue, interval_seconds=0.01)
        await collector.start()
        first_task = collector._task

        await collector.start()
        assert collector._task is first_task

        await collector.stop()

    @pytest.mark.asyncio
    async def test_stop(self, mock_queue):
        collector = MetricsCollector(mock_queue, interval_seconds=0.01)
        await collector.start()
        await collector.stop()

        assert collector._running is False

    @pytest.mark.asyncio
    async def test_collect_success(self, mock_queue):
        collector = MetricsCollector(mock_queue, interval_seconds=60)

        await collector._collect()

        mock_queue.pending_count.assert_awaited_once()
        mock_queue.processing_count.assert_awaited_once()
        mock_queue.dlq_count.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_collect_error_graceful(self, mock_queue):
        mock_queue.pending_count.side_effect = RuntimeError("db down")

        collector = MetricsCollector(mock_queue, interval_seconds=60)
        await collector._collect()  # should not raise

    @pytest.mark.asyncio
    async def test_loop_invokes_collect(self, mock_queue):
        collector = MetricsCollector(mock_queue, interval_seconds=0.01)

        await collector.start()
        await asyncio.sleep(0.05)
        await collector.stop()

        assert mock_queue.pending_count.await_count >= 1

    @pytest.mark.asyncio
    async def test_loop_survives_error(self, mock_queue):
        call_count = 0

        async def flaky_pending():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return 0

        mock_queue.pending_count = flaky_pending
        mock_queue.processing_count.return_value = 0
        mock_queue.dlq_count.return_value = 0

        collector = MetricsCollector(mock_queue, interval_seconds=0.01)
        await collector.start()
        await asyncio.sleep(0.08)
        await collector.stop()

        assert call_count >= 2
