"""Domain and application error taxonomy.

Every error carries a stable machine-readable ``code`` so that clients (and the research
agent) can branch on failures without string matching, plus an HTTP status used by the
API exception handlers.
"""

from __future__ import annotations

from typing import Any


class TheraGraphError(Exception):
    """Base class for all application errors."""

    code: str = "internal_error"
    status_code: int = 500
    retryable: bool = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self, request_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            }
        }
        if self.details:
            payload["error"]["details"] = self.details
        if request_id:
            payload["error"]["request_id"] = request_id
        return payload


class NotFoundError(TheraGraphError):
    code = "not_found"
    status_code = 404


class ValidationError(TheraGraphError):
    code = "validation_error"
    status_code = 422


class ConflictError(TheraGraphError):
    code = "conflict"
    status_code = 409


class AuthenticationError(TheraGraphError):
    code = "unauthenticated"
    status_code = 401


class AuthorizationError(TheraGraphError):
    code = "forbidden"
    status_code = 403


class RateLimitedError(TheraGraphError):
    code = "rate_limited"
    status_code = 429
    retryable = True

    def __init__(self, message: str, *, retry_after_s: int, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after_s = retry_after_s


class UpstreamError(TheraGraphError):
    """A public data source failed or returned an unusable response."""

    code = "upstream_error"
    status_code = 502
    retryable = True

    def __init__(self, message: str, *, source: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.source = source
        self.details.setdefault("source", source)


class UpstreamUnavailableError(UpstreamError):
    code = "upstream_unavailable"
    status_code = 503


class LiveIngestionDisabledError(TheraGraphError):
    code = "live_ingestion_disabled"
    status_code = 409


class ChemistryError(TheraGraphError):
    """Invalid/unparseable chemical input."""

    code = "invalid_chemistry"
    status_code = 422


class ModelNotAvailableError(TheraGraphError):
    code = "model_unavailable"
    status_code = 503


class AgentError(TheraGraphError):
    code = "agent_error"
    status_code = 500


class ToolExecutionError(AgentError):
    code = "tool_execution_error"

    def __init__(self, message: str, *, tool: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.tool = tool
        self.details.setdefault("tool", tool)


class EvidenceValidationError(AgentError):
    """Raised when a generated claim cannot be traced back to a retrieved record."""

    code = "evidence_validation_failed"
    status_code = 422


class JobError(TheraGraphError):
    code = "job_error"
