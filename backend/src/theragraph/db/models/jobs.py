"""Durable background job records.

The queue is Redis-backed for dispatch speed, but every job is *also* a database row.
That is deliberate: Redis is the transport, PostgreSQL is the truth. It means a Redis
flush loses throughput, not work, and operators can query job history with SQL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from theragraph.db.base import Base, TimestampMixin, ULIDPrimaryKey
from theragraph.db.types import JSONColumn, UTCDateTime


class Job(Base, ULIDPrimaryKey, TimestampMixin):
    """A unit of background work with at-least-once delivery and an idempotency key."""

    __tablename__ = "job"
    __table_args__ = (
        sa.UniqueConstraint("idempotency_key"),
        sa.Index("ix_job_claimable", "status", "run_after", "priority"),
        sa.Index("ix_job_task_status", "task", "status"),
        sa.CheckConstraint("attempts >= 0", name="attempts_non_negative"),
    )

    task: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    queue: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="default")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    idempotency_key: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    """Derived from the task + payload by default, so a double-submitted workflow
    returns the original job instead of doing the work twice."""

    status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="pending")
    """``pending`` | ``running`` | ``succeeded`` | ``failed`` | ``dead``."""

    priority: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=100)
    attempts: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=3)
    run_after: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)
    locked_by: Mapped[str | None] = mapped_column(sa.String(128))
    locked_until: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)
    """Visibility timeout: a worker that dies mid-job has its lease expire and the job
    is re-claimed by the reaper."""

    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    runtime_ms: Mapped[float | None] = mapped_column(sa.Float())
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONColumn)
    error: Mapped[str | None] = mapped_column(sa.Text())
    error_history: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    request_id: Mapped[str | None] = mapped_column(sa.String(64))

    @property
    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts
