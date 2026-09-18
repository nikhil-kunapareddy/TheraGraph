"""FastAPI dependencies.

Process-wide singletons (engine, cache, object store, queue, model cache) are created
once in the lifespan and handed to requests through ``app.state``, so a request never
constructs a connection pool. Per-request objects (session, principal, service context)
are created here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.config import Settings, get_settings
from theragraph.core.cache import Cache
from theragraph.core.errors import AuthenticationError, RateLimitedError
from theragraph.core.logging import get_request_id, set_user_id
from theragraph.core.ratelimit import RateLimiter
from theragraph.core.security import ANONYMOUS, Principal, verify_api_key, verify_token
from theragraph.core.storage import ObjectStore
from theragraph.db.session import get_session_factory
from theragraph.ml.inference import InferenceService
from theragraph.services.context import ServiceContext


def settings_dep(request: Request) -> Settings:
    """Return the settings the *application* was built with.

    Reading the global singleton here would silently ignore settings injected into
    ``create_app``, so a test (or a second app in one process) would run handlers
    against different configuration than the app was constructed with.
    """
    injected = getattr(request.app.state, "settings", None)
    return injected if injected is not None else get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dep)]


async def db_session() -> AsyncIterator[AsyncSession]:
    """Request-scoped session: commit on success, roll back on any exception."""
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


SessionDep = Annotated[AsyncSession, Depends(db_session)]


def cache_dep(request: Request) -> Cache:
    return request.app.state.cache  # type: ignore[no-any-return]


def store_dep(request: Request) -> ObjectStore:
    return request.app.state.store  # type: ignore[no-any-return]


def inference_dep(request: Request) -> InferenceService:
    return request.app.state.inference  # type: ignore[no-any-return]


def queue_dep(request: Request) -> Any:
    return request.app.state.queue


def rate_limiter_dep(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter  # type: ignore[no-any-return]


async def current_principal(
    request: Request,
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the caller.

    With auth disabled (local development) every caller is an anonymous researcher.
    With auth enabled, a valid API key or bearer token is required -- there is no
    partial-credit path, because a half-authenticated request is a security bug.
    """
    if not settings.auth_enabled:
        return ANONYMOUS

    if x_api_key:
        principal = verify_api_key(x_api_key, settings.api_keys)
    elif authorization and authorization.lower().startswith("bearer "):
        principal = verify_token(
            authorization.split(" ", 1)[1],
            secret=settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    else:
        raise AuthenticationError(
            "authentication required: send X-API-Key or Authorization: Bearer <token>"
        )

    set_user_id(principal.subject)
    request.state.principal = principal
    return principal


PrincipalDep = Annotated[Principal, Depends(current_principal)]


async def service_context(
    session: SessionDep,
    settings: SettingsDep,
    principal: PrincipalDep,
    request: Request,
) -> ServiceContext:
    return ServiceContext(
        settings=settings,
        session=session,
        cache=getattr(request.app.state, "cache", None),
        store=getattr(request.app.state, "store", None),
        inference=getattr(request.app.state, "inference", None),
        principal=principal,
        request_id=get_request_id(),
    )


ContextDep = Annotated[ServiceContext, Depends(service_context)]


async def agent_rate_limit(
    request: Request, settings: SettingsDep, principal: PrincipalDep
) -> None:
    """Tighter limit for endpoints that spend LLM tokens."""
    if not settings.rate_limit_enabled:
        return
    limiter: RateLimiter = request.app.state.rate_limiter
    identity = principal.subject if not principal.is_anonymous else _client_ip(request)
    decision = await limiter.check(
        identity,
        bucket="agent",
        limit=settings.rate_limit_agent_requests,
        window_s=settings.rate_limit_window_s,
    )
    if not decision.allowed:
        raise RateLimitedError(
            "agent request limit reached; these requests consume model tokens",
            retry_after_s=decision.reset_in_s,
            details={"limit": decision.limit, "window_s": settings.rate_limit_window_s},
        )


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
