"""The job queue.

**Delivery semantics: at-least-once with a database-backed lease.**

Redis is the transport; PostgreSQL is the truth. Enqueue writes the ``job`` row first,
then pushes the id onto a Redis list. If Redis loses the message, the worker's periodic
database sweep finds the pending row and runs it anyway -- throughput degrades, work
does not vanish.

A worker claims a job with a conditional ``UPDATE`` that sets a lease expiry, so two
workers racing on the same id cannot both win. A worker that dies mid-job has its lease
expire and the job is re-queued by the reaper. Handlers must therefore be idempotent,
which they are: ingestion upserts on natural keys, training derives its version from its
inputs, and research runs carry an idempotency key.

**Why not Celery/arq?** The requirements here are narrow (one queue, a handful of task
types, a single deployment) and the requirements that matter are unusual: the job's
durable state must live in the same PostgreSQL transaction as the domain rows it
creates, so an ingestion run and its job status can never disagree. Delegating to a
broker-owned state machine would have meant reconciling two sources of truth. ~250 lines
of explicit queue keeps that invariant and stays fully testable without a broker.
"""

from __future__ import annotations

import contextlib
from datetime import timedelta
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.config import Settings
from theragraph.core.clock import utcnow
from theragraph.core.hashing import hash_payload
from theragraph.core.ids import new_ulid
from theragraph.core.logging import get_logger, get_request_id
from theragraph.core.metrics import JOB_QUEUE_DEPTH, JOBS_ENQUEUED
from theragraph.db.models import Job
from theragraph.db.repositories import jobs as jobs_repo

log = get_logger(__name__)

DEFAULT_QUEUE = "default"
QUEUE_KEY_PREFIX = "tg:jobs"


def derive_idempotency_key(task: str, payload: dict[str, Any]) -> str:
    """Default idempotency key: the task plus a hash of its payload."""
    return f"{task}:{hash_payload(payload, length=24)}"


class JobQueue(Protocol):
    async def enqueue(
        self,
        session: AsyncSession,
        *,
        task: str,
        payload: dict[str, Any],
        idempotency_key: str | None = ...,
        priority: int = ...,
        delay_s: int = ...,
        max_attempts: int | None = ...,
        queue: str = ...,
    ) -> Job: ...

    async def next_job_id(
        self, *, queue: str = DEFAULT_QUEUE, timeout_s: float = 1.0
    ) -> str | None: ...
    async def depth(self, *, queue: str = DEFAULT_QUEUE) -> int: ...
    async def close(self) -> None: ...


class _BaseQueue:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def _create_row(
        self,
        session: AsyncSession,
        *,
        task: str,
        payload: dict[str, Any],
        idempotency_key: str | None,
        priority: int,
        delay_s: int,
        max_attempts: int | None,
        queue: str,
    ) -> tuple[Job, bool]:
        """Create (or return the existing) job row. Second element is ``created``."""
        key = idempotency_key or derive_idempotency_key(task, payload)
        existing = await jobs_repo.get_by_idempotency_key(session, key)
        if existing is not None:
            log.info(
                "job.idempotent_hit",
                job_id=existing.id,
                task=task,
                status=existing.status,
            )
            return existing, False

        job = await jobs_repo.create_job(
            session,
            id=new_ulid(),
            task=task,
            queue=queue,
            payload=payload,
            idempotency_key=key,
            status="pending",
            priority=priority,
            attempts=0,
            max_attempts=max_attempts or self.settings.job_max_attempts,
            run_after=utcnow() + timedelta(seconds=delay_s) if delay_s else None,
            request_id=get_request_id(),
            error_history=[],
        )
        JOBS_ENQUEUED.labels(task=task).inc()
        log.info("job.enqueued", job_id=job.id, task=task, queue=queue, priority=priority)
        return job, True


class RedisQueue(_BaseQueue):
    """Redis list transport with a database-backed durable record."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        from redis.asyncio import Redis

        self._redis: Any = Redis.from_url(settings.redis_url, decode_responses=True)

    def _key(self, queue: str) -> str:
        return f"{QUEUE_KEY_PREFIX}:{queue}"

    async def enqueue(
        self,
        session: AsyncSession,
        *,
        task: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        priority: int = 100,
        delay_s: int = 0,
        max_attempts: int | None = None,
        queue: str = DEFAULT_QUEUE,
    ) -> Job:
        job, created = await self._create_row(
            session,
            task=task,
            payload=payload,
            idempotency_key=idempotency_key,
            priority=priority,
            delay_s=delay_s,
            max_attempts=max_attempts,
            queue=queue,
        )
        if not created:
            return job

        # The row is committed by the caller's transaction; the push happens after, so a
        # rolled-back request never leaves an orphan message pointing at no row.
        async def push() -> None:
            try:
                if delay_s:
                    await self._redis.zadd(
                        f"{self._key(queue)}:delayed",
                        {job.id: (utcnow() + timedelta(seconds=delay_s)).timestamp()},
                    )
                else:
                    await self._redis.lpush(self._key(queue), job.id)
            except Exception as exc:  # pragma: no cover - redis outage
                log.warning(
                    "job.enqueue_transport_failed",
                    job_id=job.id,
                    error=str(exc),
                    note="job row persisted; the worker's database sweep will pick it up",
                )

        await push()
        return job

    async def next_job_id(
        self, *, queue: str = DEFAULT_QUEUE, timeout_s: float = 1.0
    ) -> str | None:
        try:
            await self._promote_delayed(queue)
            result = await self._redis.brpop(self._key(queue), timeout=int(max(1, timeout_s)))
        except Exception as exc:  # pragma: no cover - redis outage
            log.warning("job.dequeue_failed", error=str(exc))
            return None
        if not result:
            return None
        return str(result[1])

    async def _promote_delayed(self, queue: str) -> None:
        """Move due delayed jobs onto the ready list."""
        delayed_key = f"{self._key(queue)}:delayed"
        now = utcnow().timestamp()
        try:
            due = await self._redis.zrangebyscore(delayed_key, 0, now, start=0, num=50)
            for job_id in due:
                await self._redis.zrem(delayed_key, job_id)
                await self._redis.lpush(self._key(queue), job_id)
        except Exception:  # pragma: no cover
            return

    async def depth(self, *, queue: str = DEFAULT_QUEUE) -> int:
        try:
            depth = int(await self._redis.llen(self._key(queue)))
        except Exception:  # pragma: no cover
            return 0
        JOB_QUEUE_DEPTH.labels(queue=queue).set(depth)
        return depth

    async def close(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover
            await self._redis.aclose()


class InlineQueue(_BaseQueue):
    """Executes jobs immediately in the calling task.

    Used in tests and single-process local development. Keeps the same durable ``job``
    row so assertions about job state work identically to production.
    """

    def __init__(self, settings: Settings, *, cache: Any = None, store: Any = None) -> None:
        super().__init__(settings)
        self.cache = cache
        self.store = store
        self.executed: list[str] = []

    async def enqueue(
        self,
        session: AsyncSession,
        *,
        task: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        priority: int = 100,
        delay_s: int = 0,
        max_attempts: int | None = None,
        queue: str = DEFAULT_QUEUE,
    ) -> Job:
        job, created = await self._create_row(
            session,
            task=task,
            payload=payload,
            idempotency_key=idempotency_key,
            priority=priority,
            delay_s=delay_s,
            max_attempts=max_attempts,
            queue=queue,
        )
        if not created:
            return job

        # Commit the job row before running the handler. In production the enqueuing
        # request commits and a worker claims the row in a separate transaction; doing
        # the same here means a handler failure cannot roll back the record of the job
        # having existed.
        await session.commit()

        from theragraph.jobs.worker import execute_job

        try:
            await execute_job(
                session,
                job,
                settings=self.settings,
                cache=self.cache,
                store=self.store,
                worker="inline",
            )
        except Exception:
            # Submission and execution are separate concerns: a job that fails is
            # recorded as failed on its row (execute_job already did that), and the
            # submitter still gets its job id. Propagating here would turn a background
            # failure into a 500 for whoever enqueued it, which production -- where a
            # worker runs the job minutes later -- would never do.
            log.warning("job.inline_execution_failed", job_id=job.id, task=task)
        self.executed.append(job.id)
        return job

    async def next_job_id(
        self, *, queue: str = DEFAULT_QUEUE, timeout_s: float = 1.0
    ) -> str | None:
        return None

    async def depth(self, *, queue: str = DEFAULT_QUEUE) -> int:
        return 0

    async def close(self) -> None:
        return None


def build_queue(settings: Settings, *, cache: Any = None, store: Any = None) -> JobQueue:
    """Select the queue transport.

    ``memory://`` falls back to the inline queue rather than crashing. The cache already
    treats that URL as "no Redis here" (see ``build_cache``), so a single ``memory://``
    must mean the same thing to both -- otherwise a local setup with no Redis configures
    a working cache and an API that dies during startup on ``Redis.from_url``.
    """
    if settings.job_queue_backend == "inline" or not str(settings.redis_url).startswith(
        ("redis://", "rediss://", "unix://")
    ):
        if settings.job_queue_backend != "inline":
            log.info("jobs.inline_fallback", reason="redis_url is not a redis scheme")
        return InlineQueue(settings, cache=cache, store=store)
    return RedisQueue(settings)
