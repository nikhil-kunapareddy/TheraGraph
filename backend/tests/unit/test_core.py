"""Cross-cutting infrastructure: ids, hashing, cache, rate limiting, security, HTTP."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from theragraph.core.cache import Cache, MemoryCache, cache_key
from theragraph.core.clock import Stopwatch, elapsed_ms_since, ensure_utc
from theragraph.core.errors import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    UpstreamError,
    UpstreamUnavailableError,
)
from theragraph.core.hashing import canonical_json, hash_payload, hash_records, sha256_hex
from theragraph.core.http_client import CircuitBreaker, SourceHttpClient
from theragraph.core.ids import (
    format_run_id,
    is_ulid,
    new_ulid,
    parse_run_id,
)
from theragraph.core.ratelimit import RateLimiter
from theragraph.core.security import (
    ANONYMOUS,
    hash_api_key,
    issue_token,
    redact,
    redact_mapping,
    verify_api_key,
    verify_token,
)
from theragraph.core.storage import FilesystemObjectStore


class TestIdentifiers:
    def test_ulids_are_unique_and_well_formed(self) -> None:
        ulids = {new_ulid() for _ in range(2000)}
        assert len(ulids) == 2000
        assert all(is_ulid(value) for value in ulids)

    def test_ulids_sort_chronologically(self) -> None:
        """Lexicographic order must match creation order, so ids are sortable in SQL."""
        first = new_ulid(now_ms=1_700_000_000_000)
        second = new_ulid(now_ms=1_700_000_001_000)
        assert first < second

    def test_run_id_round_trip(self) -> None:
        assert format_run_id(2026, 183) == "RUN-2026-00183"
        assert parse_run_id("RUN-2026-00183") == (2026, 183)

    def test_run_id_zero_pads_to_five_digits(self) -> None:
        assert format_run_id(2026, 7) == "RUN-2026-00007"
        assert format_run_id(2026, 123456) == "RUN-2026-123456"

    @pytest.mark.parametrize("bad", ["RUN-26-1", "run-2026-00001", "RUN-2026", "2026-00001"])
    def test_malformed_run_id_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="malformed"):
            parse_run_id(bad)


class TestHashing:
    def test_hash_is_order_independent_for_dicts(self) -> None:
        """Dataset content hashes must not depend on key order."""
        assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})

    def test_record_hash_is_order_independent(self) -> None:
        """Upstream pagination order must not change a dataset's content hash."""
        records = [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "c", "v": 3}]
        assert hash_records(records) == hash_records(list(reversed(records)))

    def test_hash_changes_when_content_changes(self) -> None:
        assert hash_records([{"id": "a"}]) != hash_records([{"id": "a", "extra": 1}])

    def test_canonical_json_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            canonical_json({"value": float("nan")})

    def test_datetimes_serialise_deterministically(self) -> None:
        moment = datetime(2026, 6, 1, tzinfo=UTC)
        assert hash_payload({"at": moment}) == hash_payload({"at": moment})

    def test_sha256_accepts_str_and_bytes(self) -> None:
        assert sha256_hex("abc") == sha256_hex(b"abc")


class TestClock:
    def test_ensure_utc_attaches_timezone(self) -> None:
        """SQLite returns naive datetimes; arithmetic must still work."""
        naive = datetime(2026, 1, 1, 12, 0, 0)
        assert ensure_utc(naive).tzinfo is UTC

    def test_elapsed_across_mixed_awareness(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0, 0)
        aware = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
        assert elapsed_ms_since(naive, aware) == pytest.approx(1000.0)

    def test_elapsed_handles_missing_start(self) -> None:
        assert elapsed_ms_since(None) is None

    def test_stopwatch_measures(self) -> None:
        with Stopwatch() as watch:
            pass
        assert watch.elapsed_ms >= 0


class TestCache:
    async def test_round_trip_and_ttl_expiry(self) -> None:
        backend = MemoryCache()
        cache = Cache(backend, enabled=True)
        await cache.set_json("k", {"value": 1})
        assert await cache.get_json("k") == {"value": 1}

        await backend.set("expiring", b"1", ttl_s=0)
        await asyncio.sleep(0.01)
        assert await backend.get("expiring") is None

    async def test_disabled_cache_never_stores(self) -> None:
        cache = Cache(MemoryCache(), enabled=False)
        await cache.set_json("k", {"value": 1})
        assert await cache.get_json("k") is None

    async def test_read_through_calls_factory_once(self) -> None:
        cache = Cache(MemoryCache(), enabled=True)
        calls = 0

        async def factory() -> dict[str, int]:
            nonlocal calls
            calls += 1
            return {"value": calls}

        assert await cache.get_or_set("k", factory) == {"value": 1}
        assert await cache.get_or_set("k", factory) == {"value": 1}
        assert calls == 1

    async def test_namespace_invalidation(self) -> None:
        cache = Cache(MemoryCache(), enabled=True)
        await cache.set_json(cache_key("evidence", "a"), {"v": 1})
        await cache.set_json(cache_key("compound", "b"), {"v": 2})
        removed = await cache.invalidate("evidence")
        assert removed == 1
        assert await cache.get_json(cache_key("compound", "b")) == {"v": 2}

    def test_key_is_namespaced_and_versioned(self) -> None:
        key = cache_key("evidence", "target-1", 5)
        assert key.startswith("tg:v1:evidence:")
        assert cache_key("evidence", "target-1", 5) == key
        assert cache_key("evidence", "target-2", 5) != key


class TestRateLimiter:
    async def test_allows_up_to_the_limit_then_blocks(self) -> None:
        limiter = RateLimiter(MemoryCache(), enabled=True)
        decisions = [
            await limiter.check("user", bucket="test", limit=3, window_s=60) for _ in range(4)
        ]
        assert [decision.allowed for decision in decisions] == [True, True, True, False]
        assert decisions[-1].remaining == 0
        assert "Retry-After" in decisions[-1].headers()

    async def test_identities_are_independent(self) -> None:
        limiter = RateLimiter(MemoryCache(), enabled=True)
        await limiter.check("a", bucket="t", limit=1, window_s=60)
        second = await limiter.check("b", bucket="t", limit=1, window_s=60)
        assert second.allowed

    async def test_disabled_limiter_always_allows(self) -> None:
        limiter = RateLimiter(MemoryCache(), enabled=False)
        for _ in range(10):
            assert (await limiter.check("u", bucket="t", limit=1, window_s=60)).allowed


class TestSecurity:
    def test_api_key_verification_is_constant_time_and_correct(self) -> None:
        principal = verify_api_key("secret-key", ["other", "secret-key"])
        assert principal.auth_method == "api_key"
        assert principal.subject.startswith("tgk_")
        with pytest.raises(AuthenticationError):
            verify_api_key("wrong", ["secret-key"])

    def test_api_key_hash_is_stable_and_not_the_key(self) -> None:
        assert hash_api_key("k") == hash_api_key("k")
        assert "k" not in hash_api_key("k").removeprefix("tgk_")

    def test_jwt_round_trip(self) -> None:
        token = issue_token("scientist", secret="s3cret", role="researcher")
        principal = verify_token(token, secret="s3cret")
        assert principal.subject == "scientist"
        assert principal.role == "researcher"

    def test_jwt_rejects_wrong_secret(self) -> None:
        token = issue_token("scientist", secret="right")
        with pytest.raises(AuthenticationError):
            verify_token(token, secret="wrong")

    def test_expired_jwt_rejected(self) -> None:
        token = issue_token("scientist", secret="s", ttl_s=-10)
        with pytest.raises(AuthenticationError, match="expired"):
            verify_token(token, secret="s")

    def test_role_hierarchy(self) -> None:
        ANONYMOUS.require_role("viewer")
        ANONYMOUS.require_role("researcher")
        with pytest.raises(AuthorizationError):
            ANONYMOUS.require_role("admin")

    def test_secrets_are_redacted_from_text(self) -> None:
        text = "call with sk-ant-api03-ABCDEFGHIJKLMNOP and password: hunter2"
        cleaned = redact(text)
        assert "sk-ant-api03-ABCDEFGHIJKLMNOP" not in cleaned
        assert "hunter2" not in cleaned

    def test_sensitive_keys_are_redacted_recursively(self) -> None:
        cleaned = redact_mapping({"api_key": "abc", "nested": {"token": "xyz", "ok": 1}})
        assert cleaned["api_key"] == "[REDACTED]"
        assert cleaned["nested"]["token"] == "[REDACTED]"
        assert cleaned["nested"]["ok"] == 1


class TestCircuitBreaker:
    def test_opens_after_threshold_and_recovers(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, recovery_time_s=0.01)
        assert breaker.allow()
        breaker.record_failure()
        assert breaker.allow()
        breaker.record_failure()
        assert breaker.state == "open"
        assert not breaker.allow()

    def test_success_resets(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure()
        breaker.record_success()
        assert breaker.state == "closed"


class TestHttpClient:
    async def test_retries_then_succeeds(self) -> None:
        """A transient 503 must be retried, not surfaced."""
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with SourceHttpClient(
            "test",
            max_retries=4,
            backoff_base_s=0.001,
            client=httpx.AsyncClient(transport=transport, base_url="http://x"),
        ) as client:
            assert await client.get_json("/thing") == {"ok": True}
        assert attempts == 3

    async def test_client_error_is_not_retried(self) -> None:
        """A 400 means our request was wrong; retrying just wastes the upstream's quota."""
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(400, text="bad request")

        transport = httpx.MockTransport(handler)
        async with SourceHttpClient(
            "test", client=httpx.AsyncClient(transport=transport, base_url="http://x")
        ) as client:
            with pytest.raises(UpstreamError):
                await client.get_json("/thing")
        assert attempts == 1

    async def test_exhausted_retries_raise_unavailable(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(503))
        async with SourceHttpClient(
            "test",
            max_retries=2,
            backoff_base_s=0.001,
            client=httpx.AsyncClient(transport=transport, base_url="http://x"),
        ) as client:
            with pytest.raises(UpstreamError):
                await client.get_json("/thing")

    async def test_non_json_response_raises_typed_error(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>"))
        async with SourceHttpClient(
            "test", client=httpx.AsyncClient(transport=transport, base_url="http://x")
        ) as client:
            with pytest.raises(UpstreamError, match="non-JSON"):
                await client.get_json("/thing")

    async def test_open_breaker_fails_fast(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        async with SourceHttpClient(
            "test", client=httpx.AsyncClient(transport=transport, base_url="http://x")
        ) as client:
            client.breaker.failure_threshold = 1
            client.breaker.record_failure()
            with pytest.raises(UpstreamUnavailableError):
                await client.get_json("/thing")


class TestObjectStore:
    async def test_round_trip(self, tmp_path) -> None:
        store = FilesystemObjectStore(tmp_path)
        stored = await store.put_bytes("models/a/model.bin", b"weights")
        assert stored.size_bytes == 7
        assert await store.get_bytes("models/a/model.bin") == b"weights"
        assert await store.exists("models/a/model.bin")
        assert "models/a/model.bin" in await store.list_prefix("models")

    async def test_missing_object_raises_not_found(self, tmp_path) -> None:
        store = FilesystemObjectStore(tmp_path)
        with pytest.raises(NotFoundError):
            await store.get_bytes("nope")

    @pytest.mark.parametrize("key", ["../escape", "a/../../etc/passwd", "", "~/secrets"])
    async def test_traversal_keys_rejected(self, tmp_path, key: str) -> None:
        """Object keys become filesystem paths; traversal must be impossible."""
        store = FilesystemObjectStore(tmp_path)
        with pytest.raises(ValueError):
            await store.put_bytes(key, b"x")
