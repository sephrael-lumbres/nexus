"""OpenTelemetry distributed tracing for Nexus.

Provides TracerProvider initialization, tracer factory, and shutdown.
Initialized ONCE at process startup (api.py or worker.py).
All other modules call get_tracer() to create spans.

Usage:
    # At process startup (once)
    from nexus.tracing import init_tracing
    init_tracing(service_name="nexus-api")

    # In any module
    from nexus.tracing import get_tracer
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
        do_work()
"""

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from nexus.config import get_settings

logger = structlog.get_logger()

_initialized: bool = False


def init_tracing(service_name: str | None = None) -> None:
    """Initialize OpenTelemetry tracing for this process.

    Must be called ONCE at startup. Safe to call multiple times
    (subsequent calls are no-ops).

    Args:
        service_name: Override service name from settings.
                      Use "nexus-api" for API, "nexus-worker" for workers.
    """
    global _initialized

    if _initialized:
        return

    settings = get_settings()

    if not settings.otel_enabled:
        logger.info("OpenTelemetry tracing is disabled")
        _initialized = True
        return

    resolved_name = service_name or settings.otel_service_name

    # Resource: metadata attached to all spans (service name, version, env)
    resource = Resource.create(
        {
            "service.name": resolved_name,
            "service.version": "0.1.0",
            "deployment.environment": settings.environment.value,
        }
    )

    # Create OTLP span exporter based on protocol setting.
    # gRPC for local Jaeger, HTTP for Grafana Cloud.
    exporter: SpanExporter
    if settings.otel_exporter_protocol == "http":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPSpanExporter,
        )

        # Parse "Key=Value" header format into dict
        headers = _parse_headers(settings.otel_exporter_headers)

        exporter = HTTPSpanExporter(
            endpoint=f"{settings.otel_exporter_endpoint}/v1/traces",
            headers=headers,
        )
    else:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GRPCSpanExporter,
        )

        exporter = GRPCSpanExporter(
            endpoint=settings.otel_exporter_endpoint,
            insecure=settings.otel_insecure,
        )

    # BatchSpanProcessor: buffers spans and flushes async on a background thread
    processor = BatchSpanProcessor(exporter)

    # Register as the global TracerProvider
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    _initialized = True

    logger.info(
        "OpenTelemetry tracing initialized",
        service_name=resolved_name,
        endpoint=settings.otel_exporter_endpoint,
        protocol=settings.otel_exporter_protocol,
        environment=settings.environment.value,
    )


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance for creating spans.

    Args:
        name: Tracer name (convention: use __name__). Appears in
              Jaeger as the instrumentation library for each span.

    Returns:
        A Tracer instance. Returns a zero-overhead no-op tracer
        if tracing is disabled or not yet initialized.
    """
    return trace.get_tracer(name)


def _parse_headers(headers_str: str) -> dict[str, str]:
    """Parse 'Key=Value,Key2=Value2' format into a dict.

    Grafana Cloud provides headers in this format:
        Authorization=Basic <base64 token>

    Args:
        headers_str: Comma-separated key=value pairs

    Returns:
        Dict of header name to value
    """
    if not headers_str:
        return {}

    headers = {}
    for pair in headers_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()
    return headers


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the tracer provider.

    Call during shutdown to ensure buffered spans are exported
    before the process exits.
    """
    provider = trace.get_tracer_provider()

    # Only real TracerProvider has shutdown() — ProxyTracerProvider
    # (the default when init was never called) does not.
    if isinstance(provider, TracerProvider):
        provider.shutdown()
        logger.info("OpenTelemetry tracing shut down")
