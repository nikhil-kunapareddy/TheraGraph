"""Ingestion endpoints.

All ingestion is enqueued, never run inline: a single disease ingestion makes a dozen
upstream calls and can take a minute, and a request timeout mid-ingestion would leave a
half-written dataset version. The job is idempotent, so a retried request collapses onto
the original.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request

from theragraph.api.deps import ContextDep, PrincipalDep, queue_dep
from theragraph.core.errors import LiveIngestionDisabledError
from theragraph.db.repositories import provenance as prov_repo
from theragraph.schemas.common import JobAccepted
from theragraph.schemas.requests import (
    IngestBioactivityRequest,
    IngestDiseaseRequest,
    IngestEvidenceRequest,
    IngestMoleculeNetRequest,
    IngestSimilarRequest,
    SeedRequest,
)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def _accepted(job: Any, task: str, prefix: str) -> JobAccepted:
    return JobAccepted(
        job_id=job.id,
        task=task,
        status=job.status,
        idempotent_hit=job.attempts > 0 or job.status in ("succeeded", "running"),
        poll=f"{prefix}/jobs/{job.id}",
    )


async def _enqueue(
    request: Request,
    context: ContextDep,
    principal: PrincipalDep,
    *,
    task: str,
    payload: dict[str, Any],
    priority: int = 100,
) -> JobAccepted:
    principal.require_role("researcher")
    if not context.settings.allow_live_ingestion and payload.get("mode") != "replay":
        raise LiveIngestionDisabledError(
            "live ingestion is disabled on this deployment; only fixture replay is allowed",
            details={"requested_mode": payload.get("mode")},
        )
    queue = queue_dep(request)
    job = await queue.enqueue(context.session, task=task, payload=payload, priority=priority)
    return _accepted(job, task, context.settings.api_prefix)


@router.get("/sources", summary="Registered data sources and dataset versions")
async def sources(
    context: ContextDep,
    current_only: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    data_sources = await prov_repo.list_data_sources(context.session)
    versions = await prov_repo.list_dataset_versions(
        context.session, current_only=current_only, limit=limit
    )
    return {
        "sources": [
            {
                "id": source.id,
                "name": source.name,
                "description": source.description,
                "homepage_url": source.homepage_url,
                "api_base_url": source.api_base_url,
                "license": source.license,
                "citation": source.citation,
                "entity_types": source.entity_types,
            }
            for source in data_sources
        ],
        "dataset_versions": [
            {
                "id": version.id,
                "label": version.label,
                "source": version.source_id,
                "dataset": version.dataset_name,
                "version": version.version,
                "record_count": version.record_count,
                "content_hash": version.content_hash,
                "retrieved_at": version.retrieved_at.isoformat(),
                "is_current": version.is_current,
                "quality": version.quality,
                "retrieval_params": version.retrieval_params,
                "notes": version.notes,
            }
            for version in versions
        ],
    }


@router.get("/runs", summary="Ingestion run history")
async def ingestion_runs(
    context: ContextDep,
    source_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    runs = await prov_repo.list_ingestion_runs(context.session, source_id=source_id, limit=limit)
    return {
        "runs": [
            {
                "id": run.id,
                "source": run.source_id,
                "dataset": run.dataset_name,
                "status": run.status,
                "mode": run.mode,
                "started_at": run.started_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "runtime_ms": run.runtime_ms,
                "records_fetched": run.records_fetched,
                "records_written": run.records_written,
                "records_rejected": run.records_rejected,
                "rejects": run.rejects[:10],
                "params": run.params,
                "dataset_version_id": run.dataset_version_id,
                "error": run.error,
            }
            for run in runs
        ],
        "count": len(runs),
    }


@router.post("/disease", response_model=JobAccepted, summary="Ingest a disease")
async def ingest_disease(
    request: Request,
    context: ContextDep,
    principal: PrincipalDep,
    payload: IngestDiseaseRequest,
) -> JobAccepted:
    return await _enqueue(
        request,
        context,
        principal,
        task="ingest_disease",
        payload=payload.model_dump(),
        priority=60,
    )


@router.post("/evidence", response_model=JobAccepted, summary="Ingest evidence")
async def ingest_evidence(
    request: Request,
    context: ContextDep,
    principal: PrincipalDep,
    payload: IngestEvidenceRequest,
) -> JobAccepted:
    return await _enqueue(
        request, context, principal, task="ingest_evidence", payload=payload.model_dump()
    )


@router.post("/bioactivity", response_model=JobAccepted, summary="Ingest ChEMBL bioactivity")
async def ingest_bioactivity(
    request: Request,
    context: ContextDep,
    principal: PrincipalDep,
    payload: IngestBioactivityRequest,
) -> JobAccepted:
    return await _enqueue(
        request,
        context,
        principal,
        task="ingest_bioactivity",
        payload=payload.model_dump(),
    )


@router.post("/similar-compounds", response_model=JobAccepted, summary="Ingest ChEMBL neighbours")
async def ingest_similar(
    request: Request,
    context: ContextDep,
    principal: PrincipalDep,
    payload: IngestSimilarRequest,
) -> JobAccepted:
    return await _enqueue(
        request,
        context,
        principal,
        task="ingest_similar_compounds",
        payload=payload.model_dump(),
    )


@router.post("/moleculenet", response_model=JobAccepted, summary="Snapshot a MoleculeNet set")
async def ingest_moleculenet(
    request: Request,
    context: ContextDep,
    principal: PrincipalDep,
    payload: IngestMoleculeNetRequest,
) -> JobAccepted:
    return await _enqueue(
        request,
        context,
        principal,
        task="ingest_moleculenet",
        payload=payload.model_dump(),
    )


@router.post("/seed", response_model=JobAccepted, summary="Build the demo corpus")
async def seed(
    request: Request,
    context: ContextDep,
    principal: PrincipalDep,
    payload: SeedRequest,
) -> JobAccepted:
    """Ingest the curated demo corpus (three disease areas + MoleculeNet benchmarks)."""
    principal.require_role("admin" if context.settings.is_production else "researcher")
    return await _enqueue(
        request,
        context,
        principal,
        task="seed_demo_corpus",
        payload=payload.model_dump(),
        priority=200,
    )


@router.post("/features/backfill", response_model=JobAccepted, summary="Backfill descriptors")
async def backfill_features(
    request: Request,
    context: ContextDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
) -> JobAccepted:
    return await _enqueue(
        request,
        context,
        principal,
        task="backfill_features",
        payload={"limit": limit, "mode": "replay"},
    )
