"""Ingestion primitives: modes, record/replay cache, normalised batches."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from theragraph.core.clock import utcnow
from theragraph.core.errors import LiveIngestionDisabledError
from theragraph.core.hashing import hash_payload, hash_records
from theragraph.core.logging import get_logger

log = get_logger(__name__)


class IngestionMode(StrEnum):
    """How a connector obtains upstream data.

    ``REPLAY`` is what makes the demo and the test suite deterministic: every upstream
    response used to build the shipped dataset is stored as a fixture, so a fresh clone
    reproduces the exact same database without network access or an API key.
    """

    LIVE = "live"
    """Always call the upstream API; do not write fixtures."""

    RECORD = "record"
    """Call the upstream API and persist the raw response as a fixture."""

    REPLAY = "replay"
    """Serve only from fixtures; refuse network egress."""

    AUTO = "auto"
    """Use a fixture when present, otherwise call upstream and record it."""


@dataclass
class NormalizedBatch:
    """Normalised records of one entity type, ready for a repository upsert."""

    entity_type: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)
    raw_payloads: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def content_hash(self) -> str:
        return hash_records(self.rows)

    def extend(self, other: NormalizedBatch) -> None:
        if other.entity_type != self.entity_type:
            raise ValueError(
                f"cannot merge {other.entity_type} batch into {self.entity_type} batch"
            )
        self.rows.extend(other.rows)
        self.rejects.extend(other.rejects)
        self.raw_payloads.extend(other.raw_payloads)


@dataclass
class IngestionOutcome:
    """Result of one pipeline execution."""

    source_id: str
    dataset_name: str
    version: str
    mode: IngestionMode
    retrieved_at: datetime
    content_hash: str
    records_fetched: int = 0
    records_written: int = 0
    records_rejected: int = 0
    dataset_version_id: str | None = None
    entity_counts: dict[str, int] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    rejects: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "dataset": self.dataset_name,
            "version": self.version,
            "mode": str(self.mode),
            "retrieved_at": self.retrieved_at.isoformat(),
            "content_hash": self.content_hash,
            "fetched": self.records_fetched,
            "written": self.records_written,
            "rejected": self.records_rejected,
            "entities": self.entity_counts,
            "quality_score": self.quality.get("score"),
        }


class ResponseRecorder:
    """A VCR-style cache of upstream responses.

    Fixtures are keyed by a hash of (source, operation, parameters) and stored as
    pretty-printed JSON under ``data/seed/fixtures/<source>/``. Committing them means:

    * the shipped demo database is byte-reproducible from a clean clone,
    * ingestion tests exercise the *real* response shapes rather than hand-written
      mocks that drift from reality, and
    * a reviewer can read exactly what the upstream API said on a given date.
    """

    def __init__(self, root: Path, *, mode: IngestionMode = IngestionMode.AUTO) -> None:
        self.root = Path(root)
        self.mode = mode
        self.hits = 0
        self.misses = 0

    def path_for(self, source: str, operation: str, params: dict[str, Any]) -> Path:
        digest = hash_payload(params, length=16)
        safe_operation = operation.replace("/", "_").replace(":", "_")
        return self.root / source / f"{safe_operation}.{digest}.json"

    async def fetch(
        self,
        *,
        source: str,
        operation: str,
        params: dict[str, Any],
        loader: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Return a response from fixture or upstream, honouring the mode."""
        path = self.path_for(source, operation, params)

        if self.mode in (IngestionMode.REPLAY, IngestionMode.AUTO) and path.exists():
            self.hits += 1
            payload = json.loads(path.read_text())
            log.debug("ingestion.fixture_hit", source=source, operation=operation)
            return payload["response"]

        if self.mode is IngestionMode.REPLAY:
            raise LiveIngestionDisabledError(
                f"no fixture for {source}/{operation}; live ingestion is disabled",
                details={"expected_fixture": str(path), "params": params},
            )

        self.misses += 1
        response = await loader()

        if self.mode in (IngestionMode.RECORD, IngestionMode.AUTO):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "source": source,
                        "operation": operation,
                        "params": params,
                        "recorded_at": utcnow().isoformat(),
                        "response": response,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            log.info("ingestion.fixture_recorded", source=source, operation=operation)
        return response

    def stats(self) -> dict[str, int]:
        return {"fixture_hits": self.hits, "upstream_calls": self.misses}


def version_label(retrieved_at: datetime, upstream_version: str | None = None) -> str:
    """Dataset version label: the upstream release if published, else a date stamp.

    Preferring the upstream label matters scientifically -- "Open Targets 26.06" is
    citable and comparable across groups, whereas a retrieval date is only meaningful
    inside this deployment.
    """
    if upstream_version:
        return upstream_version
    return retrieved_at.strftime("%Y.%m.%d")
