"""Liveness, readiness and metrics.

``/healthz`` answers "is this process alive" and must never touch a dependency, or a
database blip would cause an orchestrator to kill healthy pods. ``/readyz`` answers
"can this process serve traffic" and does check dependencies.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response

from theragraph.api.deps import SettingsDep
from theragraph.core.metrics import render_metrics
from theragraph.db.session import check_database
from theragraph.schemas.common import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.version, environment=settings.environment)


@router.get("/readyz", response_model=HealthResponse, summary="Readiness probe")
async def readyz(request: Request, settings: SettingsDep, response: Response) -> HealthResponse:
    """Check every dependency needed to serve a request.

    The cache is reported but not required: it degrades to cache-miss, so a Redis
    outage should not remove the API from the load balancer.
    """
    checks: dict[str, Any] = {"database": await check_database()}

    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        checks["cache"] = {"ok": await cache.backend.ping(), "required": False}

    store = getattr(request.app.state, "store", None)
    if store is not None:
        checks["object_storage"] = {
            "backend": type(store).__name__,
            "ok": True,
            "required": False,
        }

    inference = getattr(request.app.state, "inference", None)
    if inference is not None:
        checks["model_cache"] = inference.model_cache.stats()

    checks["agent_backend"] = {
        "backend": settings.agent_llm_backend if settings.anthropic_api_key else "deterministic",
        "model": settings.agent_model,
        "api_key_configured": bool(settings.anthropic_api_key),
    }

    required_ok = bool(checks["database"].get("ok"))
    if not required_ok:
        response.status_code = 503

    return HealthResponse(
        status="ok" if required_ok else "degraded",
        version=settings.version,
        environment=settings.environment,
        checks=checks,
    )


@router.get("/metrics", include_in_schema=False, summary="Prometheus metrics")
async def metrics(settings: SettingsDep) -> Response:
    if not settings.metrics_enabled:
        return Response(status_code=404)
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
