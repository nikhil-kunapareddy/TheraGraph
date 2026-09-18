"""Background job endpoints, including model training."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query, Request

from theragraph.api.deps import ContextDep, PrincipalDep, queue_dep
from theragraph.core.errors import NotFoundError
from theragraph.db.repositories import jobs as jobs_repo
from theragraph.jobs.registry import TASKS
from theragraph.schemas.common import JobAccepted
from theragraph.schemas.requests import TrainRequest

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_payload(job: Any) -> dict[str, Any]:
    return {
        "id": job.id,
        "task": job.task,
        "queue": job.queue,
        "status": job.status,
        "priority": job.priority,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "payload": job.payload,
        "result": job.result,
        "error": job.error,
        "error_history": job.error_history,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "runtime_ms": job.runtime_ms,
        "locked_by": job.locked_by,
        "request_id": job.request_id,
    }


@router.get("", summary="List jobs")
async def list_jobs(
    context: ContextDep,
    status: Annotated[str | None, Query()] = None,
    task: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    jobs = await jobs_repo.list_jobs(context.session, status=status, task=task, limit=limit)
    return {
        "jobs": [_job_payload(job) for job in jobs],
        "queue_stats": await jobs_repo.queue_stats(context.session),
        "registered_tasks": sorted(TASKS),
    }


@router.get("/{job_id}", summary="Job status")
async def get_job(context: ContextDep, job_id: Annotated[str, Path()]) -> dict[str, Any]:
    job = await jobs_repo.get_job(context.session, job_id)
    if job is None:
        raise NotFoundError(f"job {job_id!r} not found")
    return _job_payload(job)


@router.post("/train", response_model=JobAccepted, summary="Train models for an endpoint")
async def train(
    request: Request,
    context: ContextDep,
    principal: PrincipalDep,
    payload: TrainRequest,
) -> JobAccepted:
    """Enqueue a training run.

    Training takes minutes (a GNN with cross-validation can take much longer), so it is
    always a background job. The resulting models, evaluations and experiment are
    queryable through ``/models`` and ``/experiments``.
    """
    principal.require_role("researcher")
    queue = queue_dep(request)
    job = await queue.enqueue(
        context.session,
        task="train_endpoint",
        payload=payload.model_dump(),
        priority=150,
    )
    return JobAccepted(
        job_id=job.id,
        task="train_endpoint",
        status=job.status,
        poll=f"{context.settings.api_prefix}/jobs/{job.id}",
    )
