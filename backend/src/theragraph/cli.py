"""TheraGraph command line.

Everything an operator or a scientist needs to run the platform without the UI:
bootstrap a database, ingest public data, train and evaluate models, run the research
agent, execute the eval suite, and start the API, worker or MCP server.

Every command that mutates data prints what it did, including dataset versions and
content hashes, so a terminal session doubles as a provenance record.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast, get_args

import typer

from theragraph.config import Settings, get_settings
from theragraph.core.logging import configure_logging, get_logger
from theragraph.schemas.requests import ModelFamily

app = typer.Typer(
    name="theragraph",
    help="Evidence-grounded computational drug-discovery workbench.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database schema and bootstrap.", no_args_is_help=True)
ingest_app = typer.Typer(help="Ingest public biomedical data.", no_args_is_help=True)
ml_app = typer.Typer(help="Train, evaluate and inspect models.", no_args_is_help=True)
agent_app = typer.Typer(help="Run and evaluate the research agent.", no_args_is_help=True)

app.add_typer(db_app, name="db")
app.add_typer(ingest_app, name="ingest")
app.add_typer(ml_app, name="ml")
app.add_typer(agent_app, name="agent")

log = get_logger(__name__)


def _setup(log_format: str = "console") -> Settings:
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=log_format, service="theragraph-cli")
    return settings


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


async def _with_session(coroutine: Any) -> Any:
    from theragraph.db.session import dispose_engine, init_engine, session_scope

    settings = get_settings()
    init_engine(settings)
    try:
        async with session_scope() as session:
            return await coroutine(settings, session)
    finally:
        await dispose_engine()


def _service_context(settings: Settings, session: Any) -> Any:
    from theragraph.core.cache import build_cache
    from theragraph.core.storage import build_object_store
    from theragraph.services.context import ServiceContext

    return ServiceContext(
        settings=settings,
        session=session,
        cache=build_cache(
            settings.redis_url,
            enabled=settings.cache_enabled,
            default_ttl_s=settings.cache_default_ttl_s,
        ),
        store=build_object_store(
            backend=settings.storage_backend,
            root=settings.storage_root,
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key_id,
            secret_key=settings.s3_secret_access_key,
            region=settings.s3_region,
        ),
    )


def _ingestion_service(settings: Settings, mode: str) -> Any:
    from theragraph.ingestion.base import IngestionMode
    from theragraph.ingestion.pipelines import IngestionService, build_connectors

    resolved = IngestionMode(mode)
    connectors = build_connectors(settings, mode=resolved)
    return IngestionService(settings, connectors, mode=resolved)


def _parse_families(raw: str) -> tuple[ModelFamily, ...]:
    """Validate ``--families`` up front so a typo fails with the allowed values."""
    allowed = get_args(ModelFamily)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise typer.BadParameter(
            f"unknown model family {', '.join(unknown)}; choose from {', '.join(allowed)}"
        )
    return cast("tuple[ModelFamily, ...]", tuple(names))


# ----------------------------------------------------------------------- info


@app.command()
def info() -> None:
    """Print the effective configuration (secrets redacted)."""
    settings = _setup()
    payload = settings.model_dump(mode="json")
    for key in ("anthropic_api_key", "jwt_secret", "s3_secret_access_key", "api_keys"):
        if payload.get(key):
            payload[key] = "[REDACTED]"
    payload["agent_backend_effective"] = (
        settings.agent_llm_backend if settings.anthropic_api_key else "deterministic"
    )
    _echo_json(payload)


@app.command()
def version() -> None:
    """Print the version."""
    from theragraph import __version__
    from theragraph.chem.molecule import rdkit_version

    typer.echo(f"theragraph {__version__} (rdkit {rdkit_version()})")


# ------------------------------------------------------------------------- db
@db_app.command("bootstrap")
def db_bootstrap(
    drop: bool = typer.Option(False, "--drop", help="Drop existing tables first."),
) -> None:
    """Create the schema directly from the models.

    For throwaway local databases and tests. Deployments use ``alembic upgrade head``,
    which is the only path that preserves data.
    """
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.db.session import create_all, drop_all, get_engine

        engine = get_engine()
        if drop:
            await drop_all(engine)
        await create_all(engine)
        from theragraph.db.base import Base

        return {"tables": len(Base.metadata.tables), "dropped_first": drop}

    _echo_json(asyncio.run(_with_session(run)))


@db_app.command("check")
def db_check() -> None:
    """Verify database connectivity."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.db.session import check_database, get_engine

        return await check_database(get_engine())

    result = asyncio.run(_with_session(run))
    _echo_json(result)
    raise typer.Exit(0 if result.get("ok") else 1)


@db_app.command("demo-corpus")
def db_demo_corpus() -> None:
    """Load the small offline demo corpus: real molecules, real identifiers, no network.

    This is the same in-process corpus the test suite builds -- one disease, two
    targets, four approved drugs with their measured ChEMBL potencies and three
    Open Targets evidence rows, all with their real accessions and PMIDs. It exists so
    the agent, the ranking engine and the eval suite can be exercised end to end on a
    machine with no internet access, and so CI does not depend on public APIs being up.
    Use ``theragraph ingest seed`` for the full corpus.
    """
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.ingestion.pipelines import ensure_data_sources
        from theragraph.tests_support.factories import build_corpus

        await ensure_data_sources(session)
        summary = await build_corpus(session)
        await session.commit()
        return {"loaded": {key: value for key, value in summary.items() if key != "now"}}

    _echo_json(asyncio.run(_with_session(run)))


@db_app.command("sources")
def db_sources() -> None:
    """Register the public data source definitions."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.ingestion.pipelines import ensure_data_sources

        count = await ensure_data_sources(session)
        return {"registered_sources": count}

    _echo_json(asyncio.run(_with_session(run)))


# --------------------------------------------------------------------- ingest
@ingest_app.command("disease")
def ingest_disease(
    efo_id: str = typer.Argument(..., help="EFO or MONDO id, e.g. MONDO_0004975."),
    n_targets: int = typer.Option(25, help="Associated targets to ingest."),
    mode: str = typer.Option("auto", help="live | record | replay | auto."),
) -> None:
    """Ingest a disease with its associated targets and association scores."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.ingestion.pipelines import ensure_data_sources

        await ensure_data_sources(session)
        service = _ingestion_service(settings, mode)
        try:
            outcome = await service.ingest_disease(session, efo_id, n_targets=n_targets)
            return outcome.summary()
        finally:
            await service.connectors.aclose()

    _echo_json(asyncio.run(_with_session(run)))


@ingest_app.command("bioactivity")
def ingest_bioactivity(
    target: str = typer.Argument(..., help="Target symbol or Ensembl id."),
    max_activities: int = typer.Option(300),
    mode: str = typer.Option("auto"),
) -> None:
    """Ingest ChEMBL compounds, assays and measured activities for a target."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        service = _ingestion_service(settings, mode)
        try:
            outcome = await service.ingest_target_bioactivity(
                session, target_key=target, max_activities=max_activities
            )
            return {**outcome.summary(), "warnings": outcome.warnings}
        finally:
            await service.connectors.aclose()

    _echo_json(asyncio.run(_with_session(run)))


@ingest_app.command("moleculenet")
def ingest_moleculenet(
    dataset: str = typer.Argument(..., help="esol | bbbp | lipophilicity | clintox."),
    force: bool = typer.Option(False, "--force", help="Re-download the raw file."),
    mode: str = typer.Option("auto"),
) -> None:
    """Snapshot a MoleculeNet benchmark and register its dataset version."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.ingestion.pipelines import ensure_data_sources

        await ensure_data_sources(session)
        service = _ingestion_service(settings, mode)
        try:
            outcome = await service.ingest_moleculenet(
                session, dataset_key=dataset, force_download=force
            )
            return {**outcome.summary(), "quality": outcome.quality}
        finally:
            await service.connectors.aclose()

    _echo_json(asyncio.run(_with_session(run)))


@ingest_app.command("seed")
def ingest_seed(
    plan: str = typer.Option("demo", help="demo | smoke."),
    mode: str = typer.Option("auto", help="Use 'replay' for a network-free rebuild."),
) -> None:
    """Build the demo corpus from public sources (three disease areas + benchmarks)."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.ingestion.seed import DEMO_PLAN, SMOKE_PLAN, run_seed

        service = _ingestion_service(settings, mode)
        try:
            return await run_seed(
                session, service, SMOKE_PLAN if plan == "smoke" else DEMO_PLAN, commit=True
            )
        finally:
            await service.connectors.aclose()

    _echo_json(asyncio.run(_with_session(run)))


@ingest_app.command("features")
def ingest_features(limit: int = typer.Option(2000)) -> None:
    """Compute descriptors and fingerprints for compounds that lack them."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        service = _ingestion_service(settings, "replay")
        try:
            return await service.backfill_features(session, limit=limit)
        finally:
            await service.connectors.aclose()

    _echo_json(asyncio.run(_with_session(run)))


# ------------------------------------------------------------------------- ml
@ml_app.command("train")
def ml_train(
    endpoint: str = typer.Option(
        ..., help="solubility | bbb_permeability | clinical_toxicity | lipophilicity."
    ),
    dataset: str = typer.Option(..., help="esol | bbbp | clintox | lipophilicity."),
    families: str = typer.Option("random_forest,gin", help="Comma-separated model families."),
    cv_folds: int = typer.Option(5),
    no_cv: bool = typer.Option(False, "--no-cv", help="Skip cross-validation (faster)."),
    no_random_split: bool = typer.Option(
        False, "--no-random-split", help="Skip the random-split optimism reference."
    ),
    epochs: int = typer.Option(0, help="Override GNN max epochs (0 = default)."),
    seed: int = typer.Option(1337),
) -> None:
    """Train, evaluate and register models for one endpoint.

    Always evaluates on a scaffold split; the random split is reported alongside purely
    to show how optimistic it is.
    """
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.ml.train import TrainingConfig, train_endpoint

        graph_hyperparameters = {"max_epochs": epochs} if epochs else {}
        config = TrainingConfig(
            endpoint=endpoint,
            dataset_key=dataset,
            families=_parse_families(families),
            cv_folds=cv_folds,
            run_cross_validation=not no_cv,
            run_random_split_reference=not no_random_split,
            graph_hyperparameters=graph_hyperparameters,
            seed=seed,
        )
        context = _service_context(settings, session)
        outcome = await train_endpoint(session, settings, config, store=context.store)
        return outcome.summary()

    _echo_json(asyncio.run(_with_session(run)))


@ml_app.command("models")
def ml_models(endpoint: str = typer.Option(None), status: str = typer.Option(None)) -> None:
    """List registered models with their scaffold-split metrics."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.db.repositories import ml as ml_repo

        models = await ml_repo.list_models(session, endpoint=endpoint, status=status)
        return {
            "models": [
                {
                    "ref": model.ref,
                    "family": model.family,
                    "endpoint": model.endpoint,
                    "task_type": model.task_type,
                    "status": model.status,
                    "scaffold": (model.metrics or {}).get("scaffold"),
                    "random": (model.metrics or {}).get("random"),
                    "cross_validation": (model.metrics or {}).get("cross_validation"),
                }
                for model in models
            ]
        }

    _echo_json(asyncio.run(_with_session(run)))


@ml_app.command("predict")
def ml_predict(
    smiles: list[str] = typer.Argument(..., help="One or more SMILES strings."),
    endpoint: str = typer.Option(None, help="Restrict to one endpoint."),
) -> None:
    """Score molecules with the active models."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.ml.inference import summarize_predictions

        context = _service_context(settings, session)
        inference = context.require_inference()
        endpoints = (
            [endpoint]
            if endpoint
            else [item["endpoint"] for item in await inference.available_endpoints(session)]
        )
        result = await inference.predict(
            session,
            molecules=[(None, item) for item in smiles],
            endpoints=endpoints,
            persist=False,
        )
        return {
            "predictions": [prediction.to_dict() for prediction in result.predictions],
            "summary": summarize_predictions(result.predictions),
            "failures": result.failures,
        }

    _echo_json(asyncio.run(_with_session(run)))


# ---------------------------------------------------------------------- agent
@agent_app.command("research")
def agent_research(
    query: str = typer.Argument(..., help="The research question."),
    disease: str = typer.Option(None, help="Pin a disease."),
    target: str = typer.Option(None, help="Pin a target."),
    n_targets: int = typer.Option(10),
    show_report: bool = typer.Option(True, "--report/--no-report", help="Print the report."),
) -> None:
    """Run a research workflow end to end and print the trace."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.agent.orchestrator import start_research
        from theragraph.services import runs as runs_service

        context = _service_context(settings, session)
        parameters: dict[str, Any] = {"n_targets": n_targets}
        if disease:
            parameters["disease"] = disease
        if target:
            parameters["target"] = target

        result = await start_research(
            context, query=query, parameters=parameters, execute_inline=True
        )
        trace = await runs_service.get_run(context, result["run_id"])
        if show_report and trace.get("report"):
            typer.echo("\n" + "=" * 78)
            typer.echo(trace["report"]["markdown"])
            typer.echo("=" * 78 + "\n")
        return {
            "run_id": result["run_id"],
            "status": result["status"],
            "tool_calls": [
                {
                    "tool": call["tool_name"],
                    "status": call["status"],
                    "latency_ms": call["latency_ms"],
                    "summary": call["result_summary"],
                }
                for call in trace["tool_calls"]
            ],
            "readiness": result.get("readiness"),
            "usage": result.get("usage"),
            "reproducibility_hash": trace["reproducibility_hash"],
        }

    _echo_json(asyncio.run(_with_session(run)))


@agent_app.command("eval")
def agent_eval(
    output: Path = typer.Option(
        Path("artifacts/eval/agent-eval.json"), help="Where to write the JSON report."
    ),
    case: str = typer.Option(None, help="Run a single case by id."),
    fail_under: bool = typer.Option(
        True, "--gate/--no-gate", help="Exit non-zero when a threshold is breached."
    ),
) -> None:
    """Run the agent evaluation suite and check the CI thresholds."""
    _setup()

    async def run(settings: Settings, session: Any) -> dict[str, Any]:
        from theragraph.agent.evals import EVAL_CASES, check_thresholds, run_suite

        cases = [item for item in EVAL_CASES if item.id == case] if case else None
        if case and not cases:
            raise typer.BadParameter(
                f"unknown case {case!r}; available: {[item.id for item in EVAL_CASES]}"
            )
        context = _service_context(settings, session)
        suite = await run_suite(context, cases=cases, output_path=output)
        ok, breaches = check_thresholds(suite)
        return {
            "aggregate": suite.aggregate(),
            "thresholds_met": ok,
            "breaches": breaches,
            "report": str(output),
            "cases": [
                {"id": item.case_id, "passed": item.passed, "failures": item.failures}
                for item in suite.cases
            ],
        }

    result = asyncio.run(_with_session(run))
    _echo_json(result)
    if fail_under and not result["thresholds_met"]:
        raise typer.Exit(1)


@agent_app.command("tools")
def agent_tools() -> None:
    """List the agent's tools and their schemas."""
    _setup()
    from theragraph.agent.tools import TOOL_REGISTRY

    _echo_json(
        {
            "tools": [
                {
                    "name": spec.name,
                    "category": spec.category,
                    "mutates": spec.mutates,
                    "description": spec.description,
                    "input_schema": spec.schema()["input_schema"],
                }
                for spec in sorted(TOOL_REGISTRY.values(), key=lambda item: item.name)
            ]
        }
    )


# --------------------------------------------------------------------- serving
@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (development)."),
    workers: int = typer.Option(1),
) -> None:
    """Start the API server."""
    import uvicorn

    settings = _setup(log_format=get_settings().log_format)
    uvicorn.run(
        "theragraph.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=settings.log_level.lower(),
        access_log=False,  # the access-log middleware emits structured lines instead
    )


@app.command()
def worker(
    max_jobs: int = typer.Option(0, help="Exit after this many jobs (0 = run forever)."),
    queue: str = typer.Option("default"),
) -> None:
    """Start a background job worker."""
    from theragraph.jobs.worker import run_worker

    processed = asyncio.run(run_worker(max_jobs=max_jobs or None, queue_name=queue))
    typer.echo(json.dumps({"processed_jobs": processed}))


@app.command("mcp")
def mcp_server() -> None:
    """Start the MCP server (stdio by default)."""
    from theragraph.mcp_server.server import main as mcp_main

    mcp_main()


@app.command()
def openapi(output: Path = typer.Option(Path("docs/openapi.json"))) -> None:
    """Write the OpenAPI schema to a file (used by CI and the docs)."""
    _setup()
    from theragraph.main import create_app

    schema = create_app().openapi()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2))
    typer.echo(json.dumps({"written": str(output), "paths": len(schema.get("paths", {}))}))


if __name__ == "__main__":  # pragma: no cover
    app()
