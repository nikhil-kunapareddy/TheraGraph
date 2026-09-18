"""Task registry and execution context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.config import Settings
from theragraph.core.cache import Cache
from theragraph.core.errors import JobError
from theragraph.core.logging import get_logger
from theragraph.core.storage import ObjectStore

log = get_logger(__name__)


@dataclass
class JobContext:
    """What a task handler gets. Mirrors ``ServiceContext`` plus job bookkeeping."""

    settings: Settings
    session: AsyncSession
    job_id: str
    attempt: int
    cache: Cache | None = None
    store: ObjectStore | None = None
    heartbeat: Callable[[], Awaitable[None]] | None = None

    async def beat(self) -> None:
        """Extend the job lease during a long step."""
        if self.heartbeat is not None:
            await self.heartbeat()

    def service_context(self) -> Any:
        from theragraph.services.context import ServiceContext

        return ServiceContext(
            settings=self.settings,
            session=self.session,
            cache=self.cache,
            store=self.store,
            request_id=f"job:{self.job_id}",
        )


TaskHandler = Callable[[JobContext, dict[str, Any]], Awaitable[dict[str, Any]]]

TASKS: dict[str, TaskHandler] = {}

#: Tasks whose payload alone determines the outcome, so a duplicate submission can be
#: safely collapsed onto the existing job rather than re-executed.
IDEMPOTENT_TASKS: set[str] = set()


def task(name: str, *, idempotent: bool = True) -> Callable[[TaskHandler], TaskHandler]:
    """Register an async task handler."""

    def decorator(handler: TaskHandler) -> TaskHandler:
        if name in TASKS:
            raise JobError(f"task {name!r} is already registered")
        TASKS[name] = handler
        if idempotent:
            IDEMPOTENT_TASKS.add(name)
        return handler

    return decorator


def get_handler(name: str) -> TaskHandler:
    handler = TASKS.get(name)
    if handler is None:
        raise JobError(f"unknown task {name!r}", details={"registered": sorted(TASKS)})
    return handler
