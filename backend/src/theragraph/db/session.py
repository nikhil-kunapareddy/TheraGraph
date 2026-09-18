"""Async engine, session factory and query instrumentation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from theragraph.config import Settings
from theragraph.core.logging import get_logger
from theragraph.core.metrics import DB_QUERIES, DB_QUERY_LATENCY, DB_SLOW_QUERIES

log = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _operation_of(statement: str) -> str:
    """First keyword of a statement, used as a bounded metric label."""
    head = statement.lstrip().split(maxsplit=1)
    return head[0].lower() if head else "unknown"


def _instrument(engine: AsyncEngine, *, slow_threshold_ms: float) -> None:
    sync_engine = engine.sync_engine

    @event.listens_for(sync_engine, "before_cursor_execute")
    def _before(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        conn.info.setdefault("_tg_query_start", []).append(time.perf_counter())

    @event.listens_for(sync_engine, "after_cursor_execute")
    def _after(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        stack = conn.info.get("_tg_query_start")
        if not stack:
            return
        elapsed = time.perf_counter() - stack.pop()
        operation = _operation_of(statement)
        DB_QUERIES.labels(operation=operation).inc()
        DB_QUERY_LATENCY.labels(operation=operation).observe(elapsed)
        if elapsed * 1000 >= slow_threshold_ms:
            DB_SLOW_QUERIES.labels(operation=operation).inc()
            log.warning(
                "db.slow_query",
                operation=operation,
                duration_ms=round(elapsed * 1000, 2),
                statement=" ".join(statement.split())[:300],
            )


def create_engine(settings: Settings) -> AsyncEngine:
    """Build the async engine with driver-appropriate pooling."""
    is_sqlite = settings.database_url.startswith("sqlite")
    kwargs: dict[str, Any] = {
        "echo": settings.database_echo,
        "future": True,
        "pool_pre_ping": True,
    }
    if is_sqlite:
        # SQLite (tests) has no real pool; a static pool keeps one in-memory DB alive
        # across sessions in the same test.
        from sqlalchemy.pool import StaticPool

        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = settings.database_pool_size
        kwargs["max_overflow"] = settings.database_max_overflow
        kwargs["pool_recycle"] = 1800
        kwargs["connect_args"] = {
            "server_settings": {
                "application_name": f"{settings.app_name}/{settings.version}",
                # Bound runaway analytical queries instead of holding a connection open.
                "statement_timeout": str(settings.database_statement_timeout_ms),
                "jit": "off",
            }
        }

    engine = create_async_engine(settings.database_url, **kwargs)

    if is_sqlite:

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    _instrument(engine, slow_threshold_ms=settings.slow_query_threshold_ms)
    return engine


def init_engine(settings: Settings) -> AsyncEngine:
    """Create the process-wide engine and session factory."""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_engine(settings)
        _session_factory = async_sessionmaker(
            _engine,
            expire_on_commit=False,
            autoflush=False,
            class_=AsyncSession,
        )
        log.info(
            "db.engine_initialised",
            dialect=_engine.dialect.name,
            pool_size=getattr(_engine.pool, "size", lambda: None)(),
        )
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("database engine not initialised; call init_engine() first")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("session factory not initialised; call init_engine() first")
    return _session_factory


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope for background work and scripts.

    Commits on success, rolls back on failure. HTTP handlers use the request-scoped
    dependency in ``api.deps`` instead, which shares this behaviour.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def check_database(engine: AsyncEngine | None = None) -> dict[str, Any]:
    """Readiness probe: can we round-trip a trivial query?"""
    target = engine or get_engine()
    started = time.perf_counter()
    try:
        async with target.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
        return {
            "ok": True,
            "dialect": target.dialect.name,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "dialect": target.dialect.name}


async def create_all(engine: AsyncEngine | None = None) -> None:
    """Create the schema directly from the models.

    Used by the test suite and by ``theragraph db bootstrap`` for a throwaway local
    database. Real deployments always go through Alembic migrations.
    """
    from theragraph.db import models  # noqa: F401  (registers mappers)
    from theragraph.db.base import Base

    target = engine or get_engine()
    async with target.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all(engine: AsyncEngine | None = None) -> None:
    from theragraph.db import models  # noqa: F401
    from theragraph.db.base import Base

    target = engine or get_engine()
    async with target.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
