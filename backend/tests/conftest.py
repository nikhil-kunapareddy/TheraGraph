"""Shared test fixtures.

Two rules the suite enforces on itself:

* **No network.** Ingestion connectors are exercised against recorded fixtures or
  in-process fakes. A test that needs the internet is a test that fails in CI.
* **No shared state.** Each test gets its own SQLite database, created from the same
  models the migrations create, so tests cannot leak rows into each other.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from theragraph.config import Settings, reset_settings_cache
from theragraph.core.cache import build_cache
from theragraph.core.storage import FilesystemObjectStore
from theragraph.db import session as session_module
from theragraph.db.session import create_all, create_engine, drop_all
from theragraph.services.context import ServiceContext


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_database_url() -> str:
    """SQLite in memory by default; ``TEST_DATABASE_URL`` switches to real PostgreSQL.

    The dialect is not an implementation detail here: ``upsert_many`` compiles a
    different ON CONFLICT statement per backend, JSON columns become JSONB, and
    timestamps come back aware rather than naive. CI runs the suite both ways so a
    SQLite-only pass cannot hide a PostgreSQL-only failure.
    """
    return os.environ.get("TEST_DATABASE_URL") or "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def settings(tmp_path: Any) -> Settings:
    """Test settings: throwaway database, no cache, no network, inline jobs."""
    reset_settings_cache()
    return Settings(
        environment="test",
        database_url=test_database_url(),
        redis_url="memory://",
        cache_enabled=False,
        rate_limit_enabled=False,
        auth_enabled=False,
        job_queue_backend="inline",
        storage_backend="filesystem",
        storage_root=tmp_path / "artifacts",
        model_registry_dir=tmp_path / "artifacts" / "models",
        data_dir=tmp_path / "data",
        seed_dir=tmp_path / "data" / "seed",
        allow_live_ingestion=False,
        agent_llm_backend="deterministic",
        anthropic_api_key=None,
        log_level="WARNING",
        otel_enabled=False,
        metrics_enabled=True,
    )


@pytest_asyncio.fixture
async def engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    created = create_engine(settings)
    await create_all(created)
    yield created
    await drop_all(created)
    await created.dispose()


@pytest_asyncio.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as opened:
        yield opened


@pytest.fixture
def context(settings: Settings, session: AsyncSession, tmp_path: Any) -> ServiceContext:
    return ServiceContext(
        settings=settings,
        session=session,
        cache=build_cache("memory://", enabled=False),
        store=FilesystemObjectStore(tmp_path / "store"),
    )


@pytest_asyncio.fixture
async def api_client(settings: Settings, engine: AsyncEngine) -> AsyncIterator[AsyncClient]:
    """An ASGI client bound to an app that shares the test engine.

    The engine is injected into the session module rather than created by the app's
    lifespan, so the client and the fixtures see the same in-memory database.
    """
    from theragraph.core.ratelimit import RateLimiter
    from theragraph.jobs import tasks  # noqa: F401  (registers task handlers)
    from theragraph.jobs.queue import InlineQueue
    from theragraph.main import create_app
    from theragraph.ml.inference import InferenceService

    app = create_app(settings)
    cache = build_cache("memory://", enabled=False)

    # Replace the lifespan so it reuses the test engine instead of building its own.
    app.router.lifespan_context = _noop_lifespan  # type: ignore[assignment]
    app.state.cache = cache
    app.state.store = FilesystemObjectStore(settings.storage_root / "test")
    app.state.inference = InferenceService(settings, cache=cache, store=app.state.store)
    app.state.rate_limiter = RateLimiter(cache.backend, enabled=False)
    app.state.queue = InlineQueue(settings, cache=cache, store=app.state.store)

    session_module._engine = engine
    session_module._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


async def _noop_lifespan(app: Any) -> AsyncIterator[None]:  # pragma: no cover - plumbing
    yield


@pytest_asyncio.fixture
async def secured_client_factory(settings: Settings, engine: AsyncEngine) -> AsyncIterator[Any]:
    """Build an API client for a *modified* Settings -- auth or rate limiting enabled.

    The default ``api_client`` runs with both disabled, which is right for testing the
    science but leaves the security gates unexercised. This factory exists so the
    enforcement path itself can be asserted: an unauthenticated request must be refused,
    and an over-quota one must be throttled.
    """
    from theragraph.core.ratelimit import RateLimiter
    from theragraph.jobs.queue import InlineQueue
    from theragraph.main import create_app
    from theragraph.ml.inference import InferenceService

    clients: list[AsyncClient] = []

    async def build(**overrides: Any) -> AsyncClient:
        secured = settings.model_copy(update=overrides)
        app = create_app(secured)
        cache = build_cache("memory://", enabled=False)
        app.router.lifespan_context = _noop_lifespan  # type: ignore[assignment]
        app.state.settings = secured
        app.state.cache = cache
        app.state.store = FilesystemObjectStore(secured.storage_root / "secured")
        app.state.inference = InferenceService(secured, cache=cache, store=app.state.store)
        app.state.rate_limiter = RateLimiter(cache.backend, enabled=secured.rate_limit_enabled)
        app.state.queue = InlineQueue(secured, cache=cache, store=app.state.store)

        session_module._engine = engine
        session_module._session_factory = async_sessionmaker(engine, expire_on_commit=False)

        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
        clients.append(client)
        return client

    yield build
    for client in clients:
        await client.aclose()


# --------------------------------------------------------------------- factories
@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def seeded(session: AsyncSession, now: datetime) -> dict[str, Any]:
    """A small, deterministic corpus: one disease, two targets, four compounds.

    Built from real structures and plausible values but entirely in-process, so every
    test that needs data gets the same corpus with no network and no fixtures to load.
    """
    from theragraph.tests_support.factories import build_corpus

    return await build_corpus(session, now=now)
