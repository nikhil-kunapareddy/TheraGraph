"""Provenance repository: data sources, dataset versions, ingestion runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.core.clock import elapsed_ms_since, utcnow
from theragraph.db.models import (
    DatasetVersion,
    DataSource,
    IngestionRun,
    ProvenanceRecord,
)
from theragraph.db.repositories.base import upsert_many


async def upsert_data_source(session: AsyncSession, **fields: Any) -> DataSource:
    """Register or refresh a data source definition."""
    await upsert_many(
        session,
        DataSource,
        [fields],
        conflict_columns=["id"],
        update_columns=[
            "name",
            "description",
            "homepage_url",
            "api_base_url",
            "license",
            "citation",
            "entity_types",
        ],
    )
    source = await session.get(DataSource, fields["id"], populate_existing=True)
    assert source is not None
    return source


async def list_data_sources(session: AsyncSession) -> list[DataSource]:
    result = await session.execute(sa.select(DataSource).order_by(DataSource.id))
    return list(result.scalars())


async def create_dataset_version(
    session: AsyncSession,
    *,
    source_id: str,
    dataset_name: str,
    version: str,
    content_hash: str,
    record_count: int,
    retrieved_at: datetime | None = None,
    source_url: str | None = None,
    retrieval_params: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    notes: str | None = None,
    make_current: bool = True,
) -> DatasetVersion:
    """Create (or refresh) a dataset snapshot and optionally make it the current one.

    Re-ingesting identical upstream data under the same version label updates the
    existing row rather than creating a duplicate, which keeps ``content_hash``
    comparisons meaningful.
    """
    if make_current:
        await session.execute(
            sa.update(DatasetVersion)
            .where(
                DatasetVersion.source_id == source_id,
                DatasetVersion.dataset_name == dataset_name,
                DatasetVersion.is_current.is_(True),
            )
            .values(is_current=False)
            .execution_options(synchronize_session=False)
        )

    rows = await upsert_many(
        session,
        DatasetVersion,
        [
            {
                "source_id": source_id,
                "dataset_name": dataset_name,
                "version": version,
                "content_hash": content_hash,
                "record_count": record_count,
                "retrieved_at": retrieved_at or utcnow(),
                "source_url": source_url,
                "retrieval_params": retrieval_params or {},
                "quality": quality or {},
                "notes": notes,
                "is_current": make_current,
            }
        ],
        conflict_columns=["source_id", "dataset_name", "version"],
    )
    version_row = await session.get(DatasetVersion, rows[0][0], populate_existing=True)
    assert version_row is not None
    return version_row


async def get_current_dataset_version(
    session: AsyncSession, source_id: str, dataset_name: str
) -> DatasetVersion | None:
    result = await session.execute(
        sa.select(DatasetVersion)
        .where(
            DatasetVersion.source_id == source_id,
            DatasetVersion.dataset_name == dataset_name,
            DatasetVersion.is_current.is_(True),
        )
        .order_by(DatasetVersion.retrieved_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def list_dataset_versions(
    session: AsyncSession,
    *,
    source_id: str | None = None,
    dataset_name: str | None = None,
    current_only: bool = False,
    limit: int = 100,
) -> list[DatasetVersion]:
    stmt = sa.select(DatasetVersion)
    if source_id:
        stmt = stmt.where(DatasetVersion.source_id == source_id)
    if dataset_name:
        stmt = stmt.where(DatasetVersion.dataset_name == dataset_name)
    if current_only:
        stmt = stmt.where(DatasetVersion.is_current.is_(True))
    stmt = stmt.order_by(DatasetVersion.retrieved_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars())


async def get_dataset_versions_by_ids(
    session: AsyncSession, ids: list[str]
) -> list[DatasetVersion]:
    if not ids:
        return []
    result = await session.execute(sa.select(DatasetVersion).where(DatasetVersion.id.in_(ids)))
    return list(result.scalars())


async def start_ingestion_run(
    session: AsyncSession,
    *,
    source_id: str,
    dataset_name: str,
    mode: str,
    params: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> IngestionRun:
    run = IngestionRun(
        source_id=source_id,
        dataset_name=dataset_name,
        mode=mode,
        params=params or {},
        idempotency_key=idempotency_key,
        status="running",
        started_at=utcnow(),
    )
    session.add(run)
    await session.flush()
    return run


async def finish_ingestion_run(
    session: AsyncSession,
    run: IngestionRun,
    *,
    status: str,
    dataset_version_id: str | None = None,
    records_fetched: int = 0,
    records_written: int = 0,
    records_rejected: int = 0,
    rejects: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> IngestionRun:
    finished = utcnow()
    run.status = status
    run.finished_at = finished
    run.runtime_ms = elapsed_ms_since(run.started_at, finished)
    run.dataset_version_id = dataset_version_id
    run.records_fetched = records_fetched
    run.records_written = records_written
    run.records_rejected = records_rejected
    run.rejects = (rejects or [])[:50]
    run.error = error
    await session.flush()
    return run


async def list_ingestion_runs(
    session: AsyncSession, *, source_id: str | None = None, limit: int = 50
) -> list[IngestionRun]:
    stmt = sa.select(IngestionRun)
    if source_id:
        stmt = stmt.where(IngestionRun.source_id == source_id)
    stmt = stmt.order_by(IngestionRun.started_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars())


async def record_raw_payloads(session: AsyncSession, rows: list[dict[str, Any]]) -> int:
    """Archive upstream payloads for audit (best-effort, deduplicated by hash)."""
    if not rows:
        return 0
    session.add_all([ProvenanceRecord(**row) for row in rows])
    await session.flush()
    return len(rows)
