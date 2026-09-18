"""Time access behind a seam so tests can freeze it and runs stay reproducible."""

from __future__ import annotations

import time
from datetime import UTC, datetime


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


def utcnow_iso() -> str:
    return utcnow().isoformat().replace("+00:00", "Z")


def ensure_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to a naive datetime.

    PostgreSQL ``timestamptz`` round-trips timezone-aware values; SQLite has no
    timezone type and hands back naive ones. Subtracting a naive from an aware datetime
    raises, so every duration computation on a persisted timestamp goes through here.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def elapsed_ms_since(
    started_at: datetime | None, finished_at: datetime | None = None
) -> float | None:
    """Milliseconds between two persisted timestamps, dialect-independent."""
    start = ensure_utc(started_at)
    if start is None:
        return None
    end = ensure_utc(finished_at) or utcnow()
    return (end - start).total_seconds() * 1000


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0


class Stopwatch:
    """Context manager measuring wall-clock duration in milliseconds."""

    __slots__ = ("_start", "elapsed_ms")

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = monotonic_ms()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = round(monotonic_ms() - self._start, 3)
