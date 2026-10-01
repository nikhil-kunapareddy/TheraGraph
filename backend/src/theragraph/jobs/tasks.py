"""Task handlers.

Each handler is thin: validate the payload, call a service, return a JSON-serialisable
result. All the logic lives in services, so the same operation behaves identically
whether it was triggered by an HTTP request, the CLI, or a background job.
"""

from __future__ import annotations

from typing import Any

from theragraph.core.errors import ValidationError
from theragraph.core.logging import get_logger
from theragraph.ingestion.base import IngestionMode
from theragraph.ingestion.pipelines import IngestionService, build_connectors
from theragraph.jobs.registry import JobContext, task

log = get_logger(__name__)


def _require(payload: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if not payload.get(key)]
    if missing:
        raise ValidationError(
            "job payload is missing required fields", details={"missing": missing}
        )


def _ingestion_service(context: JobContext, mode: str | None = None) -> IngestionService:
    resolved = IngestionMode(mode or IngestionMode.AUTO)
    connectors = build_connectors(context.settings, mode=resolved)
    return IngestionService(context.settings, connectors, mode=resolved)


@task("ingest_disease")
async def ingest_disease(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Ingest a disease, its associated targets and their association scores."""
    _require(payload, "efo_id")
    service = _ingestion_service(context, payload.get("mode"))
    try:
        outcome = await service.ingest_disease(
            context.session,
            payload["efo_id"],
            n_targets=int(payload.get("n_targets", 25)),
        )
        await context.beat()
        return outcome.summary()
    finally:
        await service.connectors.aclose()


@task("ingest_evidence")
async def ingest_evidence(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload, "efo_id", "target_keys")
    service = _ingestion_service(context, payload.get("mode"))
    try:
        outcome = await service.ingest_evidence(
            context.session,
            efo_id=payload["efo_id"],
            target_keys=list(payload["target_keys"]),
            datasource_ids=payload.get("datasource_ids"),
            per_datasource=int(payload.get("per_datasource", 8)),
        )
        return outcome.summary()
    finally:
        await service.connectors.aclose()


@task("ingest_bioactivity")
async def ingest_bioactivity(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    # The REST request names the field ``target``; older payloads used ``target_key``.
    target_key = payload.get("target_key") or payload.get("target")
    if not target_key:
        raise ValidationError(
            "job payload is missing required fields", details={"missing": ["target"]}
        )
    service = _ingestion_service(context, payload.get("mode"))
    try:
        outcome = await service.ingest_target_bioactivity(
            context.session,
            target_key=target_key,
            max_activities=int(payload.get("max_activities", 300)),
            activity_types=payload.get("activity_types"),
        )
        await context.beat()
        return {**outcome.summary(), "warnings": outcome.warnings}
    finally:
        await service.connectors.aclose()


@task("ingest_similar_compounds")
async def ingest_similar_compounds(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload, "smiles")
    service = _ingestion_service(context, payload.get("mode"))
    try:
        outcome = await service.ingest_similar_compounds(
            context.session,
            smiles=payload["smiles"],
            threshold_percent=int(payload.get("threshold_percent", 70)),
            limit=int(payload.get("limit", 30)),
        )
        return outcome.summary()
    finally:
        await service.connectors.aclose()


@task("ingest_moleculenet")
async def ingest_moleculenet(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload, "dataset_key")
    service = _ingestion_service(context, payload.get("mode"))
    try:
        outcome = await service.ingest_moleculenet(
            context.session,
            dataset_key=payload["dataset_key"],
            force_download=bool(payload.get("force_download", False)),
        )
        return outcome.summary()
    finally:
        await service.connectors.aclose()


@task("backfill_features")
async def backfill_features(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Compute descriptors and fingerprints for compounds that lack them."""
    service = _ingestion_service(context)
    try:
        return await service.backfill_features(
            context.session, limit=int(payload.get("limit", 1000))
        )
    finally:
        await service.connectors.aclose()


@task("enrich_pubchem")
async def enrich_pubchem(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload, "compound_ids")
    service = _ingestion_service(context, payload.get("mode"))
    try:
        return await service.enrich_with_pubchem(
            context.session,
            compound_ids=list(payload["compound_ids"]),
            limit=int(payload.get("limit", 50)),
        )
    finally:
        await service.connectors.aclose()


@task("seed_demo_corpus")
async def seed_demo_corpus(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the demo corpus from public sources."""
    from theragraph.ingestion.seed import DEMO_PLAN, SMOKE_PLAN, run_seed

    plan = SMOKE_PLAN if payload.get("plan") == "smoke" else DEMO_PLAN
    service = _ingestion_service(context, payload.get("mode"))
    try:
        return await run_seed(context.session, service, plan, commit=False)
    finally:
        await service.connectors.aclose()


@task("train_endpoint")
async def train_endpoint(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Train and register models for one prediction endpoint."""
    _require(payload, "endpoint", "dataset_key")
    from theragraph.ml.train import TrainingConfig
    from theragraph.ml.train import train_endpoint as run_training

    config = TrainingConfig(
        endpoint=payload["endpoint"],
        dataset_key=payload["dataset_key"],
        families=tuple(payload.get("families") or ("random_forest", "gin")),
        cv_folds=int(payload.get("cv_folds", 5)),
        run_cross_validation=bool(payload.get("run_cross_validation", True)),
        run_random_split_reference=bool(payload.get("run_random_split_reference", True)),
        baseline_hyperparameters=payload.get("baseline_hyperparameters") or {},
        graph_hyperparameters=payload.get("graph_hyperparameters") or {},
        seed=int(payload.get("seed", 1337)),
        experiment_name=payload.get("experiment_name"),
    )
    await context.beat()
    outcome = await run_training(context.session, context.settings, config, store=context.store)
    return outcome.summary()


@task("run_research_workflow", idempotent=False)
async def run_research_workflow(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a full agent research workflow for an already-created run."""
    _require(payload, "run_id")
    from theragraph.agent.orchestrator import execute_run

    await context.beat()
    return await execute_run(
        context.service_context(),
        run_id=payload["run_id"],
        heartbeat=context.beat,
    )


@task("replay_research_run", idempotent=False)
async def replay_research_run(context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload, "run_id")
    from theragraph.agent.orchestrator import execute_run

    await context.beat()
    return await execute_run(
        context.service_context(),
        run_id=payload["run_id"],
        heartbeat=context.beat,
    )
