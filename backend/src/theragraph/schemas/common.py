"""Shared response envelopes."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int | None = None
    limit: int
    offset: int = 0
    next_cursor: str | None = None


class ProvenanceInfo(BaseModel):
    """Where a record came from. Present on every scientific entity."""

    model_config = ConfigDict(extra="allow")

    source: str
    source_identifier: str | None = None
    source_url: str | None = None
    retrieved_at: str
    dataset_version: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(description="`ok` or `degraded`.")
    version: str
    environment: str
    checks: dict[str, Any] = Field(default_factory=dict)


class JobAccepted(BaseModel):
    """Returned by endpoints that enqueue background work."""

    job_id: str
    task: str
    status: str
    idempotent_hit: bool = Field(
        default=False,
        description="True when an identical job already existed and was returned instead.",
    )
    poll: str = Field(description="URL to poll for completion.")


class RunAccepted(BaseModel):
    run_id: str
    status: str
    job_id: str | None = None
    poll: str
    inline: bool = False
