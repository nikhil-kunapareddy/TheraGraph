"""HTTP middleware: request identity, structured access logs, metrics, limits."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from theragraph.config import Settings
from theragraph.core.errors import RateLimitedError
from theragraph.core.ids import new_request_id
from theragraph.core.logging import get_logger, set_request_id
from theragraph.core.metrics import (
    HTTP_IN_FLIGHT,
    HTTP_LATENCY,
    HTTP_REQUESTS,
    status_class,
)
from theragraph.core.telemetry import bind_trace_id

log = get_logger(__name__)

#: Paths excluded from access logging and rate limiting: probes and metrics scrapes are
#: high-frequency and carry no user intent.
QUIET_PATHS = frozenset({"/healthz", "/readyz", "/metrics", "/favicon.ico"})


def route_template(request: Request) -> str:
    """Route pattern (``/api/v1/targets/{identifier}``), never the raw path.

    Using the raw path as a metric label would create one time series per target id.
    """
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind it to logs, and echo it back to the caller."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id")
        request_id = incoming if incoming and len(incoming) <= 128 else new_request_id()
        token = set_request_id(request_id)
        request.state.request_id = request_id
        bind_trace_id()
        try:
            response = await call_next(request)
        finally:
            set_request_id(None) if token is None else None
        response.headers["X-Request-ID"] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """One structured line per request, plus latency and error metrics."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in QUIET_PATHS:
            return await call_next(request)

        started = time.perf_counter()
        HTTP_IN_FLIGHT.inc()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            HTTP_IN_FLIGHT.dec()
            elapsed = time.perf_counter() - started
            template = route_template(request)
            HTTP_REQUESTS.labels(
                method=request.method, route=template, status=status_class(status_code)
            ).inc()
            HTTP_LATENCY.labels(method=request.method, route=template).observe(elapsed)
            log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                route=template,
                status=status_code,
                duration_ms=round(elapsed * 1000, 2),
                client=request.client.host if request.client else None,
            )


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies before they are buffered."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "payload_too_large",
                        "message": f"request body exceeds {self.max_bytes} bytes",
                        "retryable": False,
                    }
                },
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Global per-caller rate limit.

    Applied in middleware (not a dependency) so it also covers unmatched routes and
    runs before any database work. Agent endpoints add a second, tighter bucket.
    """

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self.settings.rate_limit_enabled or request.url.path in QUIET_PATHS:
            return await call_next(request)

        limiter = getattr(request.app.state, "rate_limiter", None)
        if limiter is None:
            return await call_next(request)

        # The client address, never a request header: this runs before authentication,
        # so a header value is unverified and a caller could rotate it to get a fresh
        # bucket on every request. Behind a proxy, uvicorn resolves the real address
        # from X-Forwarded-For only for peers listed in FORWARDED_ALLOW_IPS.
        identity = request.client.host if request.client else "unknown"
        decision = await limiter.check(
            identity,
            bucket="global",
            limit=self.settings.rate_limit_requests,
            window_s=self.settings.rate_limit_window_s,
        )
        if not decision.allowed:
            error = RateLimitedError("rate limit exceeded", retry_after_s=decision.reset_in_s)
            return JSONResponse(
                status_code=429,
                content=error.to_payload(getattr(request.state, "request_id", None)),
                headers=decision.headers(),
            )

        response = await call_next(request)
        for header, value in decision.headers().items():
            response.headers.setdefault(header, value)
        return response


def install_middleware(app: FastAPI, settings: Settings) -> None:
    """Install middleware. Order matters: outermost first."""
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware

    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        # Credentials travel in headers (X-API-Key, Authorization), never cookies, so
        # there is nothing for a cross-origin page to ride on.
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "Retry-After"],
    )
