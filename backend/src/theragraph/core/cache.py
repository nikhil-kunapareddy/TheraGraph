"""Caching layer.

Two backends implement the same small protocol:

* :class:`RedisCache` for real deployments (shared across API + worker processes).
* :class:`MemoryCache` for tests and single-process local dev, with TTL honoured so
  behaviour matches Redis.

Cache keys are namespaced and versioned (``tg:v1:<namespace>:<hash>``) so that a
deploy which changes a payload shape does not read stale entries, and every cached
value stores its own ``cached_at`` so callers can surface data freshness to users --
important for scientific provenance.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from theragraph.core.hashing import hash_payload
from theragraph.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

CACHE_SCHEMA_VERSION = "v1"


def cache_key(namespace: str, *parts: Any, prefix: str = "tg") -> str:
    """Build a deterministic, namespaced cache key."""
    digest = hash_payload(list(parts), length=32) if parts else "none"
    return f"{prefix}:{CACHE_SCHEMA_VERSION}:{namespace}:{digest}"


class CacheBackend(Protocol):
    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: bytes, ttl_s: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def incr(self, key: str, ttl_s: int | None = None) -> int: ...
    async def clear_namespace(self, namespace: str) -> int: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


@dataclass
class _Entry:
    value: bytes
    expires_at: float | None


class MemoryCache:
    """Process-local cache. Not shared between workers; fine for tests/dev."""

    def __init__(self) -> None:
        self._data: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> bytes | None:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at is not None and entry.expires_at <= time.monotonic():
                self._data.pop(key, None)
                return None
            return entry.value

    async def set(self, key: str, value: bytes, ttl_s: int | None = None) -> None:
        # ``ttl_s is None`` means "no expiry"; a non-positive TTL means "already
        # expired". Treating 0 as "cache forever" would silently pin stale data when a
        # caller computes a TTL that rounds down to zero.
        if ttl_s is not None and ttl_s <= 0:
            async with self._lock:
                self._data.pop(key, None)
            return
        async with self._lock:
            expires = time.monotonic() + ttl_s if ttl_s is not None else None
            self._data[key] = _Entry(value=value, expires_at=expires)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def incr(self, key: str, ttl_s: int | None = None) -> int:
        async with self._lock:
            entry = self._data.get(key)
            if entry is not None and (
                entry.expires_at is None or entry.expires_at > time.monotonic()
            ):
                current = int(entry.value)
                expires = entry.expires_at
            else:
                current = 0
                expires = time.monotonic() + ttl_s if ttl_s else None
            current += 1
            self._data[key] = _Entry(value=str(current).encode(), expires_at=expires)
            return current

    async def clear_namespace(self, namespace: str) -> int:
        marker = f":{namespace}:"
        async with self._lock:
            keys = [k for k in self._data if marker in k]
            for key in keys:
                del self._data[key]
            return len(keys)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self._data.clear()


class RedisCache:
    """Redis-backed cache. Failures degrade to cache-miss rather than erroring."""

    def __init__(self, url: str) -> None:
        from redis.asyncio import Redis

        self._redis: Any = Redis.from_url(url, decode_responses=False)
        self._healthy = True

    @property
    def client(self) -> Any:
        return self._redis

    async def get(self, key: str) -> bytes | None:
        try:
            return await self._redis.get(key)  # type: ignore[no-any-return]
        except Exception as exc:  # pragma: no cover - requires redis outage
            self._degrade("get", exc)
            return None

    async def set(self, key: str, value: bytes, ttl_s: int | None = None) -> None:
        if ttl_s is not None and ttl_s <= 0:
            # Redis rejects ``ex=0``; a non-positive TTL means do not cache.
            await self.delete(key)
            return
        try:
            await self._redis.set(key, value, ex=ttl_s)
        except Exception as exc:  # pragma: no cover
            self._degrade("set", exc)

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(key)
        except Exception as exc:  # pragma: no cover
            self._degrade("delete", exc)

    async def incr(self, key: str, ttl_s: int | None = None) -> int:
        try:
            pipe = self._redis.pipeline()
            pipe.incr(key)
            if ttl_s:
                pipe.expire(key, ttl_s, nx=True)
            result = await pipe.execute()
            return int(result[0])
        except Exception as exc:  # pragma: no cover
            self._degrade("incr", exc)
            return 0

    async def clear_namespace(self, namespace: str) -> int:
        pattern = f"tg:{CACHE_SCHEMA_VERSION}:{namespace}:*"
        removed = 0
        try:
            async for key in self._redis.scan_iter(match=pattern, count=500):
                await self._redis.delete(key)
                removed += 1
        except Exception as exc:  # pragma: no cover
            self._degrade("clear_namespace", exc)
        return removed

    async def ping(self) -> bool:
        try:
            await self._redis.ping()
            self._healthy = True
        except Exception:
            self._healthy = False
        return self._healthy

    async def close(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover
            await self._redis.aclose()

    def _degrade(self, op: str, exc: Exception) -> None:
        if self._healthy:
            log.warning("cache.degraded", operation=op, error=str(exc))
        self._healthy = False


class Cache:
    """High-level typed cache facade with JSON (de)serialisation and metrics."""

    def __init__(self, backend: CacheBackend, *, enabled: bool = True, default_ttl_s: int = 3600):
        self.backend = backend
        self.enabled = enabled
        self.default_ttl_s = default_ttl_s

    async def get_json(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        raw = await self.backend.get(key)
        if raw is None:
            _record(hit=False)
            return None
        _record(hit=True)
        import json

        try:
            return json.loads(raw)
        except ValueError:  # pragma: no cover - corrupted entry
            await self.backend.delete(key)
            return None

    async def set_json(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        if not self.enabled:
            return
        import json

        payload = json.dumps(value, separators=(",", ":"), default=str).encode()
        await self.backend.set(key, payload, ttl_s or self.default_ttl_s)

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        ttl_s: int | None = None,
    ) -> T:
        """Read-through cache. The factory runs only on a miss."""
        cached = await self.get_json(key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]
        value = await factory()
        if value is not None:
            await self.set_json(key, value, ttl_s)
        return value

    async def invalidate(self, namespace: str) -> int:
        return await self.backend.clear_namespace(namespace)


def _record(*, hit: bool) -> None:
    from theragraph.core.metrics import CACHE_EVENTS

    CACHE_EVENTS.labels(result="hit" if hit else "miss").inc()


def build_cache(url: str, *, enabled: bool = True, default_ttl_s: int = 3600) -> Cache:
    """Create a cache from a URL. ``memory://`` selects the in-process backend."""
    backend: CacheBackend = (
        MemoryCache() if url.startswith("memory://") or not enabled else RedisCache(url)
    )
    return Cache(backend, enabled=enabled, default_ttl_s=default_ttl_s)
