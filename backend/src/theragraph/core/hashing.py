"""Content hashing used for dataset versions, provenance and idempotency keys.

Hashes must be *stable across processes and Python versions*, so dict ordering is
normalised and floats are serialised with a fixed repr before hashing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, NaN rejected."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_fallback,
    )


def _fallback(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_payload(payload: Any, *, length: int = 64) -> str:
    """Hash of an arbitrary JSON-serialisable payload."""
    return sha256_hex(canonical_json(payload))[:length]


def hash_records(records: Iterable[Mapping[str, Any]], *, length: int = 16) -> str:
    """Order-independent hash of a record collection (dataset content hash).

    Each record is hashed individually, the digests are sorted and then folded, so the
    result is stable regardless of the order in which an upstream API paginated.
    """
    digests = sorted(hash_payload(record) for record in records)
    return sha256_hex("\n".join(digests))[:length]


def hash_file(path: Path, *, length: int = 64) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def short_hash(payload: Any) -> str:
    """12-char hash, for display (e.g. ``dataset v2026.09/ab12cd34ef56``)."""
    return hash_payload(payload, length=12)
