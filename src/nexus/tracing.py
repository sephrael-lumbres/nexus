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
from opentelemetry.sdk.trace.export import BatchSpanProcessor

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

    # Resource: metadata attached to all spans (service name, version, env).
    # Jaeger uses service.name to group traces in its UI.
    resource = Resource.create(
        {
            "service.name": resolved_name,
            "service.version": "0.1.0",
            "deployment.environment": settings.environment.value,
        }
    )

    # OTLP exporter: sends spans to Jaeger (local) or Grafana Tempo (prod).
    # Lazy import so grpc is only loaded when tracing is enabled.
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_endpoint,
        insecure=settings.otel_insecure,
    )

    # BatchSpanProcessor: buffers spans and flushes async on a background thread.
    # App code never blocks on Jaeger — span creation is microseconds.
    processor = BatchSpanProcessor(exporter)

    # Register as the global TracerProvider so all get_tracer() calls use it
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    _initialized = True

    logger.info(
        "OpenTelemetry tracing initialized",
        service_name=resolved_name,
        endpoint=settings.otel_exporter_endpoint,
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
