"""OpenTelemetry tracing.

Tracing is opt-in (``OTEL_ENABLED=true``). When disabled, :func:`span` is a no-op
context manager, so instrumentation can be sprinkled through the codebase without
paying for it in local dev or tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from theragraph.core.logging import get_logger, set_trace_id

log = get_logger(__name__)

_tracer: Any | None = None
_enabled = False


def setup_tracing(
    *,
    enabled: bool,
    service_name: str,
    endpoint: str | None,
    environment: str,
    version: str,
) -> None:
    """Initialise the global tracer provider."""
    global _tracer, _enabled
    if not enabled:
        _enabled = False
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": version,
                "deployment.environment": environment,
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("theragraph")
        _enabled = True
        log.info("tracing.enabled", exporter="otlp" if endpoint else "console")
    except Exception as exc:  # pragma: no cover - optional dependency path
        _enabled = False
        log.warning("tracing.setup_failed", error=str(exc))


def instrument_app(app: Any) -> None:
    """Attach FastAPI/SQLAlchemy/HTTPX auto-instrumentation when tracing is on."""
    if not _enabled:
        return
    try:  # pragma: no cover - exercised in deployed environments
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="/healthz,/readyz,/metrics")
        HTTPXClientInstrumentor().instrument()
    except Exception as exc:
        log.warning("tracing.instrumentation_failed", error=str(exc))


def instrument_engine(engine: Any) -> None:
    if not _enabled:
        return
    try:  # pragma: no cover
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    except Exception as exc:
        log.warning("tracing.engine_instrumentation_failed", error=str(exc))


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span. No-op when tracing is disabled."""
    if not _enabled or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:  # pragma: no cover
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(
                    key, value if isinstance(value, (str, int, float, bool)) else str(value)
                )
        yield current


def current_trace_id() -> str | None:
    """W3C trace id of the active span, for log correlation."""
    if not _enabled:
        return None
    try:  # pragma: no cover
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if ctx.trace_id == 0:
            return None
        return format(ctx.trace_id, "032x")
    except Exception:
        return None


def bind_trace_id() -> None:
    """Copy the active trace id into the logging context."""
    trace_id = current_trace_id()
    if trace_id:
        set_trace_id(trace_id)


def record_exception(exc: BaseException) -> None:
    if not _enabled:
        return
    try:  # pragma: no cover
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        current = trace.get_current_span()
        current.record_exception(exc)
        current.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:
        pass
