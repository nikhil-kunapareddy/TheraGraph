"""Research run endpoints: inspect, list, compare, replay."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request

from theragraph.api.deps import ContextDep, agent_rate_limit, queue_dep
from theragraph.schemas.requests import CompareRunsRequest, ReplayRequest
from theragraph.services import reports as reports_service
from theragraph.services import runs as runs_service

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", summary="List research runs")
async def list_runs(
    context: ContextDep,
    status: Annotated[str | None, Query()] = None,
    workflow: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query(description="Substring match on the query text.")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    return await runs_service.list_runs(
        context,
        status=status,
        workflow=workflow,
        query_contains=q,
        limit=limit,
        offset=offset,
    )


@router.get("/usage", summary="Aggregate run and tool observability")
async def usage(context: ContextDep) -> dict[str, Any]:
    """Latency, token, cost and per-tool error/cache statistics over recent runs."""
    return await runs_service.usage_overview(context)


@router.post("/compare", summary="Compare runs")
async def compare_runs(context: ContextDep, payload: CompareRunsRequest) -> dict[str, Any]:
    """Diff runs across pinned inputs, execution trace and outputs."""
    return await runs_service.compare_runs(context, payload.run_ids)


@router.get("/{run_id}", summary="Full reproducible execution trace")
async def get_run(
    context: ContextDep, run_id: Annotated[str, Path(description="e.g. RUN-2026-00183")]
) -> dict[str, Any]:
    return await runs_service.get_run(context, run_id)


@router.get("/{run_id}/report", summary="The run's evidence-backed report")
async def get_report(context: ContextDep, run_id: Annotated[str, Path()]) -> dict[str, Any]:
    return await reports_service.get_report(context, run_id)


@router.get("/{run_id}/replay-plan", summary="What a replay of this run would use")
async def replay_plan(
    context: ContextDep,
    run_id: Annotated[str, Path()],
    allow_version_drift: Annotated[bool, Query()] = False,
) -> dict[str, Any]:
    """Preview the pins a replay would run against, and any drift since the original."""
    return await runs_service.prepare_replay(
        context, run_id, allow_version_drift=allow_version_drift
    )


@router.post(
    "/{run_id}/replay",
    summary="Re-execute a run against its original configuration",
    dependencies=[Depends(agent_rate_limit)],
)
async def replay(
    request: Request,
    context: ContextDep,
    run_id: Annotated[str, Path()],
    payload: ReplayRequest,
) -> dict[str, Any]:
    """Create a new run with the original's query, parameters, config and seed.

    Refuses by default if the pinned dataset or model versions have changed, because a
    "replay" that silently used different inputs would undermine the guarantee.
    """
    from theragraph.agent.orchestrator import start_research

    plan = await runs_service.prepare_replay(
        context, run_id, allow_version_drift=payload.allow_version_drift
    )
    result = await start_research(
        context,
        # The replay lineage is recorded on the run's own columns below, not in
        # `parameters`: parameters feed the reproducibility hash, and a provenance
        # marker is not a scientific input.
        query=plan["query"],
        parameters=plan["parameters"],
        seed=plan["seed"],
        execute_inline=payload.execute_inline,
        queue=None if payload.execute_inline else queue_dep(request),
    )
    from theragraph.db.repositories import runs as runs_repo

    new_run = await runs_repo.get_run(context.session, result["run_id"], with_trace=False)
    if new_run is not None:
        await runs_repo.update_run(
            context.session,
            new_run,
            replay_of_run_id=run_id,
            parent_run_id=run_id,
            ranking_config=plan["ranking_config"],
        )
    return {
        **result,
        "replay_of": run_id,
        "drift": plan["drift"],
        "poll": f"{context.settings.api_prefix}/runs/{result['run_id']}",
    }
