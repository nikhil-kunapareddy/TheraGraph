"""Exception handlers.

Every error leaves the API in the same envelope, with a stable code and the request id,
so a client (or the agent) can branch on ``error.code`` and an operator can find the
matching log line.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from theragraph.core.errors import RateLimitedError, TheraGraphError
from theragraph.core.logging import get_logger
from theragraph.core.telemetry import record_exception

log = get_logger(__name__)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(TheraGraphError)
    async def _domain_error(request: Request, exc: TheraGraphError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        level = log.warning if exc.status_code < 500 else log.error
        level(
            "api.domain_error",
            code=exc.code,
            status=exc.status_code,
            message=exc.message,
            path=request.url.path,
            details=exc.details or None,
        )
        if exc.status_code >= 500:
            record_exception(exc)
        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitedError):
            headers["Retry-After"] = str(exc.retry_after_s)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_payload(request_id),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Flatten Pydantic errors into a compact, actionable list."""
        issues: list[dict[str, Any]] = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())[1:]) or "body",
                "message": error.get("msg", "invalid value"),
                "type": error.get("type"),
            }
            for error in exc.errors()[:20]
        ]
        log.warning("api.validation_error", path=request.url.path, issues=issues[:5])
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "the request body or query parameters are invalid",
                    "retryable": False,
                    "details": {"issues": issues},
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": _code_for_status(exc.status_code),
                    "message": str(exc.detail),
                    "retryable": exc.status_code in (429, 502, 503, 504),
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Last resort. Logs the traceback, returns no internal detail to the client."""
        record_exception(exc)
        log.error(
            "api.unhandled_error",
            path=request.url.path,
            error=f"{type(exc).__name__}: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "an unexpected error occurred",
                    "retryable": True,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )


def _code_for_status(status_code: int) -> str:
    return {
        400: "bad_request",
        401: "unauthenticated",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        413: "payload_too_large",
        422: "validation_error",
        429: "rate_limited",
        500: "internal_error",
        502: "upstream_error",
        503: "unavailable",
    }.get(status_code, "error")
