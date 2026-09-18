"""Job worker: claim, execute with a heartbeat, complete or retry with backoff."""

from __future__ import annotations

import asyncio
import contextlib
import random
import signal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.config import Settings, get_settings
from theragraph.core.cache import Cache, build_cache
from theragraph.core.clock import Stopwatch
from theragraph.core.ids import worker_name
from theragraph.core.logging import configure_logging, get_logger, log_context
from theragraph.core.metrics import JOB_LATENCY, JOBS_COMPLETED
from theragraph.core.storage import ObjectStore, build_object_store
from theragraph.core.telemetry import span
from theragraph.db.models import Job
from theragraph.db.repositories import jobs as jobs_repo
from theragraph.db.session import dispose_engine, init_engine, session_scope
from theragraph.jobs.queue import DEFAULT_QUEUE, JobQueue, build_queue
from theragraph.jobs.registry import JobContext, get_handler

log = get_logger(__name__)

#: Retry backoff: 15s, 60s, 240s, ... with jitter, capped. Upstream APIs rate-limit and
#: transient failures cluster, so an immediate retry usually fails the same way.
BASE_BACKOFF_S = 15
MAX_BACKOFF_S = 900


def retry_delay(attempt: int) -> int:
    """Jittered exponential backoff, capped after jitter so the cap actually holds."""
    delay = BASE_BACKOFF_S * (4 ** max(0, attempt - 1))
    jittered = delay * random.uniform(0.7, 1.3)
    return int(min(jittered, MAX_BACKOFF_S))


async def execute_job(
    session: AsyncSession,
    job: Job,
    *,
    settings: Settings,
    cache: Cache | None,
    store: ObjectStore | None,
    worker: str,
) -> dict[str, Any]:
    """Run one job's handler and record the outcome.

    The handler shares the caller's session, so the domain rows it writes and the job's
    terminal state commit together. A handler that raises leaves neither behind.
    """
    lease_s = settings.job_visibility_timeout_s

    # Snapshot the identity and bookkeeping fields before the handler runs. A failure
    # path calls ``rollback()``, which expires every ORM instance, so reading
    # ``job.attempts`` there would trigger lazy IO and raise ``MissingGreenlet``.
    job_id = job.id
    task = job.task
    attempt = job.attempts
    payload = job.payload or {}
    request_id = job.request_id

    async def heartbeat() -> None:
        await jobs_repo.extend_lease(session, job_id, lease_s=lease_s)

    context = JobContext(
        settings=settings,
        session=session,
        job_id=job_id,
        attempt=attempt,
        cache=cache,
        store=store,
        heartbeat=heartbeat,
    )

    stopwatch = Stopwatch()
    with log_context(job_id=job_id, request_id=request_id):
        log.info("job.started", task=task, attempt=attempt, worker=worker)
        try:
            handler = get_handler(task)
            with span("job.execute", task=task, job_id=job_id), stopwatch:
                result = await handler(context, payload)
            await jobs_repo.complete_job(session, job_id, result=result)
            JOBS_COMPLETED.labels(task=task, status="succeeded").inc()
            JOB_LATENCY.labels(task=task).observe(stopwatch.elapsed_ms / 1000)
            log.info(
                "job.succeeded",
                task=task,
                runtime_ms=round(stopwatch.elapsed_ms, 1),
            )
            return result
        except Exception as exc:
            # Discard whatever the handler half-wrote, then record the failure and
            # COMMIT it. Without the explicit commit the caller's transaction scope
            # would roll the failure record back too, losing the error history and the
            # incremented attempt count -- so a permanently failing job would retry for
            # ever and no operator could see why.
            await session.rollback()
            delay = retry_delay(attempt)
            refreshed = await jobs_repo.get_job(session, job_id)
            can_retry = bool(refreshed and refreshed.can_retry)
            await jobs_repo.fail_job(
                session,
                job_id,
                error=f"{type(exc).__name__}: {exc}",
                retry_in_s=delay if can_retry else None,
            )
            await session.commit()
            JOBS_COMPLETED.labels(task=task, status="retrying" if can_retry else "failed").inc()
            log.error(
                "job.failed",
                task=task,
                attempt=attempt,
                will_retry=can_retry,
                retry_in_s=delay if can_retry else None,
                error=str(exc)[:500],
                exc_info=True,
            )
            raise


class Worker:
    """Polls the queue, claims jobs and executes them with bounded concurrency."""

    def __init__(
        self,
        settings: Settings,
        *,
        queue: JobQueue | None = None,
        cache: Cache | None = None,
        store: ObjectStore | None = None,
        queue_name: str = DEFAULT_QUEUE,
    ) -> None:
        self.settings = settings
        self.name = worker_name()
        self.queue_name = queue_name
        self.cache = cache
        self.store = store
        self.queue = queue or build_queue(settings, cache=cache, store=store)
        self._stopping = asyncio.Event()
        self._semaphore = asyncio.Semaphore(settings.job_worker_concurrency)
        self._in_flight: set[asyncio.Task[Any]] = set()
        self.processed = 0

    def request_stop(self) -> None:
        log.info("worker.stop_requested", worker=self.name)
        self._stopping.set()

    async def run(self, *, max_jobs: int | None = None, sweep_interval_s: float = 10.0) -> int:
        """Main loop. Returns the number of jobs processed."""
        log.info(
            "worker.started",
            worker=self.name,
            queue=self.queue_name,
            concurrency=self.settings.job_worker_concurrency,
        )
        last_sweep = 0.0
        loop = asyncio.get_running_loop()

        while not self._stopping.is_set():
            if max_jobs is not None and self.processed >= max_jobs:
                break

            job_id = await self.queue.next_job_id(
                queue=self.queue_name, timeout_s=self.settings.job_poll_interval_s
            )

            # Periodic database sweep: re-queues expired leases and catches rows whose
            # Redis message was lost.
            now = loop.time()
            if job_id is None and now - last_sweep >= sweep_interval_s:
                last_sweep = now
                job_id = await self._sweep()

            if job_id is None:
                await asyncio.sleep(self.settings.job_poll_interval_s)
                continue

            await self._semaphore.acquire()
            task = asyncio.create_task(self._process(job_id))
            self._in_flight.add(task)
            task.add_done_callback(self._in_flight.discard)
            task.add_done_callback(lambda _: self._semaphore.release())

        if self._in_flight:
            log.info("worker.draining", in_flight=len(self._in_flight))
            await asyncio.gather(*self._in_flight, return_exceptions=True)

        log.info("worker.stopped", worker=self.name, processed=self.processed)
        return self.processed

    async def _sweep(self) -> str | None:
        async with session_scope() as session:
            reaped = await jobs_repo.reap_expired_leases(session)
            if reaped:
                log.warning("worker.reaped_expired_leases", count=len(reaped), ids=reaped[:5])
            claimable = await jobs_repo.claimable_jobs(session, queue=self.queue_name, limit=1)
            return claimable[0].id if claimable else None

    async def _process(self, job_id: str) -> None:
        self.processed += 1
        try:
            async with session_scope() as session:
                job = await jobs_repo.claim_job(
                    session,
                    job_id,
                    worker=self.name,
                    lease_s=self.settings.job_visibility_timeout_s,
                )
                if job is None:
                    log.debug("worker.job_not_claimable", job_id=job_id)
                    return
                await execute_job(
                    session,
                    job,
                    settings=self.settings,
                    cache=self.cache,
                    store=self.store,
                    worker=self.name,
                )
        except Exception:
            # Already logged and recorded in execute_job; the loop keeps going.
            pass


async def run_worker(*, max_jobs: int | None = None, queue_name: str = DEFAULT_QUEUE) -> int:
    """Entry point for ``theragraph worker``."""
    settings = get_settings()
    configure_logging(
        level=settings.log_level, fmt=settings.log_format, service="theragraph-worker"
    )
    init_engine(settings)

    cache = build_cache(
        settings.redis_url,
        enabled=settings.cache_enabled,
        default_ttl_s=settings.cache_default_ttl_s,
    )
    store = build_object_store(
        backend=settings.storage_backend,
        root=settings.storage_root,
        bucket=settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        access_key=settings.s3_access_key_id,
        secret_key=settings.s3_secret_access_key,
        region=settings.s3_region,
    )

    # Import for side effects: registers every task handler.
    from theragraph.jobs import tasks  # noqa: F401

    worker = Worker(settings, cache=cache, store=store, queue_name=queue_name)

    loop = asyncio.get_running_loop()
    for signal_name in ("SIGTERM", "SIGINT"):
        with contextlib.suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(getattr(signal, signal_name), worker.request_stop)

    try:
        return await worker.run(max_jobs=max_jobs)
    finally:
        await worker.queue.close()
        await cache.backend.close()
        await dispose_engine()
