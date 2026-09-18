"""Structured logging.

Every log line is JSON in non-local environments and carries the ambient request /
run / job context via :mod:`contextvars`, so a single ``request_id`` (or ``run_id``)
threads through HTTP handlers, background jobs, agent tool calls and SQL timing logs
without being passed around explicitly.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

import structlog

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_job_id: ContextVar[str | None] = ContextVar("job_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)

_CONTEXT_VARS: dict[str, ContextVar[str | None]] = {
    "request_id": _request_id,
    "run_id": _run_id,
    "job_id": _job_id,
    "user_id": _user_id,
    "trace_id": _trace_id,
}

_configured = False


def _inject_context(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, var in _CONTEXT_VARS.items():
        value = var.get()
        if value is not None:
            event_dict.setdefault(key, value)
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json", service: str = "theragraph") -> None:
    """Configure structlog + stdlib logging once per process."""
    global _configured
    if _configured:
        return

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.EventRenamer("message"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelNamesMapping().get(level.upper(), logging.INFO),
        force=True,
    )
    for noisy in ("uvicorn.access", "httpx", "httpcore", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.contextvars.bind_contextvars(service=service)
    _configured = True


def reset_logging() -> None:
    """Allow re-configuration (tests only)."""
    global _configured
    _configured = False


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger tagged with the module name.

    The module name is bound into the event context rather than added by
    ``structlog.stdlib.add_logger_name``: that processor reads ``logger.name``, which
    only exists on stdlib loggers, and this configuration uses ``PrintLoggerFactory``
    for a direct, allocation-light write path.
    """
    logger = structlog.get_logger()
    if name:
        return logger.bind(logger=name)  # type: ignore[no-any-return]
    return logger  # type: ignore[no-any-return]


# ------------------------------------------------------------------ context api
def set_request_id(value: str | None) -> Token[str | None]:
    return _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()


def set_trace_id(value: str | None) -> Token[str | None]:
    return _trace_id.set(value)


def set_user_id(value: str | None) -> Token[str | None]:
    return _user_id.set(value)


def get_run_id() -> str | None:
    return _run_id.get()


@contextmanager
def log_context(**values: str | None) -> Iterator[None]:
    """Temporarily bind known context variables (``run_id``, ``job_id``, ...)."""
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    extras = {k: v for k, v in values.items() if k not in _CONTEXT_VARS}
    for key, value in values.items():
        var = _CONTEXT_VARS.get(key)
        if var is not None:
            tokens.append((var, var.set(value)))
    if extras:
        structlog.contextvars.bind_contextvars(**extras)
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
        if extras:
            structlog.contextvars.unbind_contextvars(*extras)
