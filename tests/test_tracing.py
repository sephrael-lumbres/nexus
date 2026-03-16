"""Tests for OpenTelemetry tracing module.

Covers:
- _parse_headers helper function
- get_tracer factory
- init_tracing disabled, gRPC, and HTTP paths
- init_tracing idempotency
- shutdown_tracing with real and proxy providers
- Trace context propagation through Redis (inject/extract round-trip)
- Backwards compatibility with legacy bare-UUID payloads
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace import TracerProvider

import nexus.tracing as tracing_module
from nexus.tracing import _parse_headers, get_tracer, init_tracing, shutdown_tracing


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(autouse=True)
def reset_tracing_state():
    """Reset the module-level _initialized flag before each test
    so init_tracing can run fresh every time."""
    tracing_module._initialized = False
    yield
    tracing_module._initialized = False


# =============================================================================
# _parse_headers
# =============================================================================
class TestParseHeaders:
    """Tests for the _parse_headers helper used by Grafana Cloud auth."""

    def test_empty_string_returns_empty_dict(self):
        assert _parse_headers("") == {}

    def test_single_key_value_pair(self):
        result = _parse_headers("Authorization=Basic abc123")
        assert result == {"Authorization": "Basic abc123"}

    def test_multiple_pairs(self):
        result = _parse_headers("Key1=Val1,Key2=Val2")
        assert result == {"Key1": "Val1", "Key2": "Val2"}

    def test_whitespace_is_stripped(self):
        result = _parse_headers("  Key1 = Val1 , Key2 = Val2 ")
        assert result == {"Key1": "Val1", "Key2": "Val2"}

    def test_pair_without_equals_is_skipped(self):
        """Entries without '=' are silently ignored."""
        result = _parse_headers("Good=Pair,NoEquals")
        assert result == {"Good": "Pair"}

    def test_base64_padding_preserved(self):
        """Only the first '=' splits; base64 padding stays in value."""
        result = _parse_headers("Auth=Basic eyJr==")
        assert result == {"Auth": "Basic eyJr=="}


# =============================================================================
# get_tracer
# =============================================================================
class TestGetTracer:
    """Tests for get_tracer factory."""

    def test_returns_tracer_with_span_method(self):
        tracer = get_tracer("test.module")
        assert hasattr(tracer, "start_as_current_span")

    def test_noop_tracer_creates_spans_without_error(self):
        """Before init_tracing, spans should be no-ops that don't crash."""
        tracer = get_tracer("test.module")
        with tracer.start_as_current_span("test_span"):
            pass  # Should not raise


# =============================================================================
# init_tracing – disabled path
# =============================================================================
class TestInitTracingDisabled:
    """Tests when otel_enabled is False."""

    @patch("nexus.tracing.get_settings")
    def test_sets_initialized_flag(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_settings.otel_enabled = False
        mock_get_settings.return_value = mock_settings

        init_tracing()

        assert tracing_module._initialized is True

    @patch("nexus.tracing.get_settings")
    def test_second_call_is_noop(self, mock_get_settings):
        """Calling init_tracing twice should only read settings once."""
        mock_settings = MagicMock()
        mock_settings.otel_enabled = False
        mock_get_settings.return_value = mock_settings

        init_tracing()
        init_tracing()

        assert mock_get_settings.call_count == 1


# =============================================================================
# init_tracing – gRPC exporter path
# =============================================================================
class TestInitTracingGRPC:
    """Tests for the gRPC exporter path (local Jaeger)."""

    @patch("nexus.tracing.trace")
    @patch("nexus.tracing.BatchSpanProcessor")
    @patch("nexus.tracing.get_settings")
    def test_grpc_exporter_registers_provider(
        self, mock_get_settings, mock_batch_processor, mock_trace
    ):
        mock_settings = MagicMock()
        mock_settings.otel_enabled = True
        mock_settings.otel_service_name = "nexus-test"
        mock_settings.otel_exporter_protocol = "grpc"
        mock_settings.otel_exporter_endpoint = "http://localhost:4317"
        mock_settings.otel_insecure = True
        mock_settings.environment.value = "testing"
        mock_get_settings.return_value = mock_settings

        with patch("nexus.tracing.TracerProvider") as mock_provider_cls:
            mock_provider_instance = MagicMock()
            mock_provider_cls.return_value = mock_provider_instance

            grpc_exporter_mock = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(
                        OTLPSpanExporter=grpc_exporter_mock
                    )
                },
            ):
                init_tracing(service_name="nexus-grpc-test")

            mock_trace.set_tracer_provider.assert_called_once_with(
                mock_provider_instance
            )
            mock_provider_instance.add_span_processor.assert_called_once()

        assert tracing_module._initialized is True


# =============================================================================
# init_tracing – HTTP exporter path
# =============================================================================
class TestInitTracingHTTP:
    """Tests for the HTTP exporter path (Grafana Cloud)."""

    @patch("nexus.tracing.trace")
    @patch("nexus.tracing.BatchSpanProcessor")
    @patch("nexus.tracing.get_settings")
    def test_http_exporter_registers_provider(
        self, mock_get_settings, mock_batch_processor, mock_trace
    ):
        mock_settings = MagicMock()
        mock_settings.otel_enabled = True
        mock_settings.otel_service_name = "nexus-test"
        mock_settings.otel_exporter_protocol = "http"
        mock_settings.otel_exporter_endpoint = "https://otlp.example.com"
        mock_settings.otel_exporter_headers = "Authorization=Basic abc123"
        mock_settings.environment.value = "testing"
        mock_get_settings.return_value = mock_settings

        with patch("nexus.tracing.TracerProvider") as mock_provider_cls:
            mock_provider_instance = MagicMock()
            mock_provider_cls.return_value = mock_provider_instance

            http_exporter_mock = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                        OTLPSpanExporter=http_exporter_mock
                    )
                },
            ):
                init_tracing()

            mock_trace.set_tracer_provider.assert_called_once()

        assert tracing_module._initialized is True


# =============================================================================
# shutdown_tracing
# =============================================================================
class TestShutdownTracing:
    """Tests for shutdown_tracing."""

    def test_shutdown_calls_provider_shutdown(self):
        """When the global provider is a real TracerProvider, shutdown is called."""
        mock_provider = MagicMock(spec=TracerProvider)

        with patch("nexus.tracing.trace") as mock_trace:
            mock_trace.get_tracer_provider.return_value = mock_provider
            # isinstance check passes because mock has TracerProvider spec
            shutdown_tracing()

        mock_provider.shutdown.assert_called_once()

    def test_shutdown_is_noop_when_never_initialized(self):
        """When tracing was never initialized, shutdown doesn't crash."""
        shutdown_tracing()  # Should not raise


# =============================================================================
# Trace Context Propagation
# =============================================================================
class TestTraceContextPropagation:
    """Tests for trace context inject/extract through Redis.

    These verify that trace context survives the cross-process
    Redis boundary — the core of distributed tracing in Nexus.
    """

    @pytest.mark.asyncio
    async def test_enqueue_dequeue_preserves_trace_context(self, queue):
        """Verify trace context survives the Redis round-trip."""
        from opentelemetry import trace

        job_id = uuid4()

        # Create a span so there's active trace context to inject
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("test_span"):
            await queue.enqueue(job_id)

        result = await queue.dequeue_nonblocking()
        assert result is not None
        dequeued_id, trace_ctx = result
        assert dequeued_id == job_id
        assert trace_ctx is not None

    @pytest.mark.asyncio
    async def test_dequeue_handles_legacy_bare_uuid(self, queue):
        """Verify dequeue handles pre-OTel bare-UUID payloads gracefully."""
        job_id = uuid4()

        # Manually push a bare UUID string (legacy format)
        await queue._ensure_connected()
        assert queue.redis is not None
        await queue.redis.rpush(queue.pending_key, str(job_id))

        result = await queue.dequeue_nonblocking()
        assert result is not None
        dequeued_id, trace_ctx = result
        assert dequeued_id == job_id
        assert trace_ctx is None

    @pytest.mark.asyncio
    async def test_peek_handles_json_payload(self, queue):
        """Verify peek correctly parses JSON payloads."""
        job1 = uuid4()
        job2 = uuid4()

        await queue.enqueue(job1)
        await queue.enqueue(job2)

        peeked = await queue.peek(2)
        assert peeked == [job1, job2]

    @pytest.mark.asyncio
    async def test_peek_handles_legacy_bare_uuid(self, queue):
        """Verify peek handles legacy bare-UUID payloads."""
        job_id = uuid4()

        await queue._ensure_connected()
        assert queue.redis is not None
        await queue.redis.rpush(queue.pending_key, str(job_id))

        peeked = await queue.peek(1)
        assert peeked == [job_id]
