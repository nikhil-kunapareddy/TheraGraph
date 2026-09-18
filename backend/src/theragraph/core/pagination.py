"""Cursor pagination helpers.

Offset pagination breaks when rows are inserted mid-scroll (ingestion runs constantly),
so list endpoints use opaque keyset cursors built from the sort key plus the primary
key tie-breaker.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from theragraph.core.errors import ValidationError

T = TypeVar("T")

MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 25


def encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> dict[str, Any]:
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.urlsafe_b64decode(cursor + padding)
        payload = json.loads(decoded)
    except Exception as exc:
        raise ValidationError("malformed pagination cursor") from exc
    if not isinstance(payload, dict):
        raise ValidationError("malformed pagination cursor")
    return payload


def clamp_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_PAGE_SIZE
    return max(1, min(limit, MAX_PAGE_SIZE))


@dataclass
class Page(Generic[T]):
    items: list[T]
    next_cursor: str | None = None
    total: int | None = None

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None
