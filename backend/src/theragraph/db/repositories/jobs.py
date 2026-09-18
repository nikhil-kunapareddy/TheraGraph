"""Job repository: durable state for the background queue."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.core.clock import elapsed_ms_since, utcnow
from theragraph.db.models import Job


async def get_by_idempotency_key(session: AsyncSession, key: str) -> Job | None:
    result = await session.execute(sa.select(Job).where(Job.idempotency_key == key).limit(1))
    return result.scalar_one_or_none()


async def create_job(session: AsyncSession, **fields: Any) -> Job:
    job = Job(**fields)
    session.add(job)
    await session.flush()
    return job


async def get_job(session: AsyncSession, job_id: str) -> Job | None:
    return await session.get(Job, job_id, populate_existing=True)


async def claim_job(session: AsyncSession, job_id: str, *, worker: str, lease_s: int) -> Job | None:
    """Take ownership of a job iff it is still claimable.

    The guard is in the ``WHERE`` clause, so two workers racing on the same queue
    message cannot both win: the second ``UPDATE`` matches zero rows.
    """
    now = utcnow()
    stmt = (
        sa.update(Job)
        .where(
            Job.id == job_id,
            Job.status.in_(["pending", "running"]),
            sa.or_(Job.locked_until.is_(None), Job.locked_until <= now),
        )
        .values(
            status="running",
            locked_by=worker,
            locked_until=now + timedelta(seconds=lease_s),
            attempts=Job.attempts + 1,
            started_at=sa.case((Job.started_at.is_(None), now), else_=Job.started_at),
        )
        .returning(Job.id)
        # Without this, the ORM tries to evaluate the WHERE clause in Python to
        # synchronize the session, which compares the driver's naive datetimes against
        # aware ones. The claimed row is re-read below, so no synchronisation is needed.
        .execution_options(synchronize_session=False)
    )
    result = await session.execute(stmt)
    if result.first() is None:
        return None
    # The claim was a Core UPDATE, so re-populate rather than returning the identity
    # map's pre-claim copy.
    return await session.get(Job, job_id, populate_existing=True)


async def extend_lease(session: AsyncSession, job_id: str, *, lease_s: int) -> None:
    """Heartbeat for long-running jobs so the reaper does not steal them."""
    await session.execute(
        sa.update(Job)
        .where(Job.id == job_id)
        .values(locked_until=utcnow() + timedelta(seconds=lease_s))
        .execution_options(synchronize_session=False)
    )


async def complete_job(
    session: AsyncSession, job_id: str, *, result: dict[str, Any] | None
) -> None:
    """Mark a job succeeded. Runtime is computed in Python: interval arithmetic differs
    between PostgreSQL and SQLite and is not worth a dialect branch here."""
    job = await session.get(Job, job_id, populate_existing=True)
    if job is None:
        return
    now = utcnow()
    job.status = "succeeded"
    job.result = result
    job.finished_at = now
    job.locked_by = None
    job.locked_until = None
    job.runtime_ms = elapsed_ms_since(job.started_at, now)
    await session.flush()


async def fail_job(
    session: AsyncSession,
    job_id: str,
    *,
    error: str,
    retry_in_s: int | None,
    dead: bool = False,
) -> Job | None:
    """Record a failure, scheduling a retry unless the attempt budget is exhausted."""
    job = await session.get(Job, job_id, populate_existing=True)
    if job is None:
        return None
    now = utcnow()
    history = list(job.error_history or [])
    history.append({"at": now.isoformat(), "attempt": job.attempts, "error": error[:1000]})
    job.error = error[:4000]
    job.error_history = history[-10:]
    job.locked_by = None
    job.locked_until = None
    if dead or not job.can_retry:
        job.status = "dead" if dead else "failed"
        job.finished_at = now
    else:
        job.status = "pending"
        job.run_after = now + timedelta(seconds=retry_in_s or 0)
    await session.flush()
    return job


async def reap_expired_leases(session: AsyncSession, *, limit: int = 100) -> list[str]:
    """Return jobs whose worker died mid-flight, so they can be re-queued."""
    now = utcnow()
    result = await session.execute(
        sa.select(Job.id)
        .where(Job.status == "running", Job.locked_until.is_not(None), Job.locked_until <= now)
        .limit(limit)
    )
    job_ids = [row[0] for row in result.all()]
    if job_ids:
        await session.execute(
            sa.update(Job)
            .where(Job.id.in_(job_ids))
            .values(status="pending", locked_by=None, locked_until=None)
            .execution_options(synchronize_session=False)
        )
    return job_ids


async def claimable_jobs(
    session: AsyncSession, *, queue: str = "default", limit: int = 50
) -> list[Job]:
    """Database-backed fallback poll, used when the Redis queue is empty or lost."""
    now = utcnow()
    result = await session.execute(
        sa.select(Job)
        .where(
            Job.queue == queue,
            Job.status == "pending",
            sa.or_(Job.run_after.is_(None), Job.run_after <= now),
        )
        .order_by(Job.priority, Job.created_at)
        .limit(limit)
    )
    return list(result.scalars())


async def list_jobs(
    session: AsyncSession,
    *,
    status: str | None = None,
    task: str | None = None,
    limit: int = 50,
) -> list[Job]:
    stmt = sa.select(Job)
    if status:
        stmt = stmt.where(Job.status == status)
    if task:
        stmt = stmt.where(Job.task == task)
    result = await session.execute(stmt.order_by(Job.created_at.desc()).limit(limit))
    return list(result.scalars())


async def queue_stats(session: AsyncSession) -> dict[str, int]:
    result = await session.execute(sa.select(Job.status, sa.func.count()).group_by(Job.status))
    return {row[0]: int(row[1]) for row in result.all()}
