"""Application factory and lifespan.

The lifespan is where the process-wide expensive things are created once: the database
engine and pool, the cache client, the object store, the job queue, the rate limiter and
the model cache. Handlers get them from ``app.state`` rather than constructing their own,
which is the difference between a connection pool and a connection leak.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from theragraph.api.errors import install_error_handlers
from theragraph.api.middleware import install_middleware
from theragraph.api.routers import ALL_ROUTERS
from theragraph.config import Settings, get_settings
from theragraph.core.cache import build_cache
from theragraph.core.logging import configure_logging, get_logger
from theragraph.core.ratelimit import RateLimiter
from theragraph.core.security import assert_secrets_are_safe
from theragraph.core.storage import build_object_store
from theragraph.core.telemetry import instrument_app, instrument_engine, setup_tracing
from theragraph.db.session import dispose_engine, init_engine
from theragraph.jobs.queue import build_queue
from theragraph.ml.inference import InferenceService
from theragraph.ml.registry import ModelCache

log = get_logger(__name__)

DESCRIPTION = """
**TheraGraph** is an evidence-grounded computational drug-discovery workbench.

It takes a researcher from a disease to prioritised compounds through retrieved public
evidence and versioned molecular models, and it keeps the two apart:

* **Retrieved facts** come from Open Targets, ChEMBL, PubChem and Europe PMC. Every
  record carries its source, source identifier, retrieval timestamp and dataset
  version, and any score shown is the score the source published.
* **Model predictions** come from TheraGraph's own models. Every prediction carries its
  model version, uncertainty, calibrated confidence and applicability-domain status, and
  is labelled as a prediction wherever it appears.

Nothing here validates a target or a drug. Candidate prioritisation orders compounds for
experimental follow-up, with the contribution of every factor shown.

### Conventions

* Errors share one envelope with a stable `error.code`.
* Expensive operations (ingestion, training, research runs) return a job or run id to
  poll.
* `X-Request-ID` is echoed on every response and appears in the structured logs.
"""

TAGS_METADATA = [
    {"name": "system", "description": "Health probes and metrics."},
    {"name": "search", "description": "Find diseases, targets and compounds."},
    {"name": "diseases", "description": "Disease detail and target associations."},
    {"name": "targets", "description": "Target profile, evidence and tractability."},
    {"name": "compounds", "description": "Structures, descriptors, similarity, activity."},
    {"name": "predictions", "description": "Molecular property prediction."},
    {"name": "ranking", "description": "Transparent multi-objective prioritisation."},
    {"name": "graph", "description": "The evidence graph."},
    {"name": "agent", "description": "The tool-calling research assistant."},
    {"name": "runs", "description": "Reproducible research runs and reports."},
    {"name": "models", "description": "Model registry, evaluations, experiments."},
    {"name": "ingestion", "description": "Public-data ingestion and provenance."},
    {"name": "jobs", "description": "Background jobs, including training."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    assert_secrets_are_safe(
        environment=settings.environment,
        auth_enabled=settings.auth_enabled,
        jwt_secret=settings.jwt_secret,
    )

    setup_tracing(
        enabled=settings.otel_enabled,
        service_name=settings.otel_service_name,
        endpoint=settings.otel_exporter_endpoint,
        environment=settings.environment,
        version=settings.version,
    )

    engine = init_engine(settings)
    instrument_engine(engine)

    app.state.cache = build_cache(
        settings.redis_url,
        enabled=settings.cache_enabled,
        default_ttl_s=settings.cache_default_ttl_s,
    )
    app.state.store = build_object_store(
        backend=settings.storage_backend,
        root=settings.storage_root,
        bucket=settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        access_key=settings.s3_access_key_id,
        secret_key=settings.s3_secret_access_key,
        region=settings.s3_region,
    )
    app.state.model_cache = ModelCache()
    app.state.inference = InferenceService(
        settings,
        cache=app.state.cache,
        store=app.state.store,
        model_cache=app.state.model_cache,
    )
    app.state.rate_limiter = RateLimiter(
        app.state.cache.backend, enabled=settings.rate_limit_enabled
    )

    # Importing the tasks module registers every handler, which the inline queue needs.
    from theragraph.jobs import tasks  # noqa: F401

    app.state.queue = build_queue(settings, cache=app.state.cache, store=app.state.store)

    log.info(
        "app.started",
        version=settings.version,
        environment=settings.environment,
        auth_enabled=settings.auth_enabled,
        queue_backend=settings.job_queue_backend,
        storage_backend=settings.storage_backend,
        agent_backend=settings.agent_llm_backend if settings.anthropic_api_key else "deterministic",
    )

    try:
        yield
    finally:
        await app.state.queue.close()
        await app.state.cache.backend.close()
        await dispose_engine()
        log.info("app.stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts settings so tests can inject their own."""
    resolved = settings or get_settings()
    configure_logging(
        level=resolved.log_level,
        fmt=resolved.log_format,
        service=resolved.otel_service_name,
    )

    app = FastAPI(
        title=resolved.app_name,
        version=resolved.version,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "TheraGraph", "url": "https://github.com/theragraph"},
        license_info={"name": "Apache-2.0"},
    )
    app.state.settings = resolved

    install_middleware(app, resolved)
    install_error_handlers(app)

    for router in ALL_ROUTERS:
        prefix = "" if router.tags == ["system"] else resolved.api_prefix
        app.include_router(router, prefix=prefix)

    instrument_app(app)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "name": resolved.app_name,
            "version": resolved.version,
            "environment": resolved.environment,
            "docs": "/docs",
            "api": resolved.api_prefix,
            "description": (
                "Evidence-grounded computational drug-discovery workbench. "
                "Retrieved facts and model predictions are reported separately."
            ),
        }

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=TAGS_METADATA,
        )
        schema["components"] = schema.get("components", {})
        schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
        }
        if resolved.auth_enabled:
            schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
    return app


app = create_app()
