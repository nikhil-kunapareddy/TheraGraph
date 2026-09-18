"""Human-readable, sortable, collision-resistant identifiers.

Research runs get identifiers such as ``RUN-2026-00183``: a year-scoped monotonically
increasing counter that is easy for a scientist to quote in a lab notebook or a paper.
The counter is allocated inside the same database transaction that creates the row, so
it never skips or duplicates under concurrency (see
``theragraph.db.repositories.runs.allocate_run_id``).

Everything else uses ULIDs: 26-character, lexicographically sortable, URL-safe, and
generated client-side so a caller can pre-declare an idempotency key.
"""

from __future__ import annotations

import os
import re
import secrets
import time
from typing import Final

_CROCKFORD: Final = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE: Final = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")

RUN_ID_RE: Final = re.compile(r"^RUN-(?P<year>\d{4})-(?P<seq>\d{5,})$")
EXPERIMENT_ID_RE: Final = re.compile(r"^EXP-(?P<year>\d{4})-(?P<seq>\d{5,})$")


def _encode_crockford(value: int, length: int) -> str:
    chars: list[str] = []
    for _ in range(length):
        value, rem = divmod(value, 32)
        chars.append(_CROCKFORD[rem])
    return "".join(reversed(chars))


def new_ulid(now_ms: int | None = None) -> str:
    """Return a new ULID string (48-bit timestamp + 80 bits of randomness)."""
    timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
    randomness = int.from_bytes(secrets.token_bytes(10), "big")
    return _encode_crockford(timestamp, 10) + _encode_crockford(randomness, 16)


def is_ulid(value: str) -> bool:
    return bool(_ULID_RE.match(value))


def format_run_id(year: int, sequence: int) -> str:
    """``format_run_id(2026, 183) -> 'RUN-2026-00183'``."""
    return f"RUN-{year:04d}-{sequence:05d}"


def format_experiment_id(year: int, sequence: int) -> str:
    return f"EXP-{year:04d}-{sequence:05d}"


def parse_run_id(run_id: str) -> tuple[int, int]:
    match = RUN_ID_RE.match(run_id)
    if not match:
        raise ValueError(f"malformed run id: {run_id!r}")
    return int(match.group("year")), int(match.group("seq"))


def new_request_id() -> str:
    return f"req_{new_ulid().lower()}"


def new_idempotency_key() -> str:
    return f"idk_{new_ulid().lower()}"


def worker_name() -> str:
    """Stable-ish identity for a worker process, used for job lease ownership."""
    return f"{os.uname().nodename.split('.')[0]}:{os.getpid()}"
