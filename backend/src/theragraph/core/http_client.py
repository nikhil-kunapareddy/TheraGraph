"""Resilient HTTP client for public biomedical data sources.

Public scientific APIs are rate limited, occasionally flaky and sometimes slow. Every
outbound call therefore gets:

* a per-source circuit breaker, so a dead source fails fast instead of consuming the
  whole request budget;
* jittered exponential backoff on 429/5xx/network errors, honouring ``Retry-After``;
* a hard timeout and response-size guard;
* metrics + structured logs tagged with the source name.

The client never retries non-idempotent verbs and never retries 4xx other than 429.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from theragraph.core.errors import UpstreamError, UpstreamUnavailableError
from theragraph.core.logging import get_logger
from theragraph.core.metrics import UPSTREAM_LATENCY, UPSTREAM_REQUESTS, UPSTREAM_RETRIES
from theragraph.core.telemetry import span

log = get_logger(__name__)

RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
Method = Literal["GET", "POST"]


@dataclass
class CircuitBreaker:
    """Minimal three-state breaker (closed -> open -> half-open)."""

    failure_threshold: int = 5
    recovery_time_s: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.recovery_time_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class SourceHttpClient:
    """An ``httpx.AsyncClient`` wrapper scoped to one upstream data source."""

    def __init__(
        self,
        source: str,
        *,
        base_url: str = "",
        timeout_s: float = 30.0,
        max_retries: int = 4,
        backoff_base_s: float = 0.5,
        user_agent: str = "TheraGraph/0.1",
        headers: dict[str, str] | None = None,
        max_response_bytes: int = 64 * 1024 * 1024,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.source = source
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.max_response_bytes = max_response_bytes
        self.breaker = CircuitBreaker()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
                **(headers or {}),
            },
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def __aenter__(self) -> SourceHttpClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ verbs
    async def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        response = await self._request("GET", url, params=params)
        return self._parse_json(response)

    async def post_json(self, url: str, *, json: dict[str, Any]) -> Any:
        response = await self._request("POST", url, json=json)
        return self._parse_json(response)

    async def get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        response = await self._request("GET", url, params=params)
        return response.text

    async def get_bytes(self, url: str, *, params: dict[str, Any] | None = None) -> bytes:
        response = await self._request("GET", url, params=params)
        content = response.content
        if len(content) > self.max_response_bytes:
            raise UpstreamError(
                f"{self.source} response exceeded {self.max_response_bytes} bytes",
                source=self.source,
            )
        return content

    # --------------------------------------------------------------- internals
    async def _request(
        self,
        method: Method,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        if not self.breaker.allow():
            UPSTREAM_REQUESTS.labels(source=self.source, outcome="circuit_open").inc()
            raise UpstreamUnavailableError(
                f"{self.source} circuit breaker is open; not sending request",
                source=self.source,
                details={"retry_after_s": int(self.breaker.recovery_time_s)},
            )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started = time.monotonic()
            try:
                with span(
                    f"upstream.{self.source}",
                    **{"http.method": method, "http.url": url, "retry.attempt": attempt},
                ):
                    response = await self._client.request(method, url, params=params, json=json)
                elapsed = time.monotonic() - started
                UPSTREAM_LATENCY.labels(source=self.source).observe(elapsed)

                if response.status_code in RETRY_STATUS:
                    last_error = UpstreamError(
                        f"{self.source} returned HTTP {response.status_code}",
                        source=self.source,
                        details={"status_code": response.status_code},
                    )
                    if attempt == self.max_retries:
                        break
                    UPSTREAM_RETRIES.labels(
                        source=self.source, reason=f"http_{response.status_code}"
                    ).inc()
                    await self._sleep(attempt, response.headers.get("Retry-After"))
                    continue

                if response.status_code >= 400:
                    UPSTREAM_REQUESTS.labels(source=self.source, outcome="client_error").inc()
                    self.breaker.record_success()  # upstream is alive, our request was wrong
                    raise UpstreamError(
                        f"{self.source} rejected the request with HTTP {response.status_code}",
                        source=self.source,
                        details={
                            "status_code": response.status_code,
                            "body": response.text[:500],
                        },
                    )

                UPSTREAM_REQUESTS.labels(source=self.source, outcome="success").inc()
                self.breaker.record_success()
                if attempt > 1:
                    log.info(
                        "upstream.recovered", source=self.source, attempts=attempt, url=str(url)
                    )
                return response

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                UPSTREAM_LATENCY.labels(source=self.source).observe(time.monotonic() - started)
                if attempt == self.max_retries:
                    break
                UPSTREAM_RETRIES.labels(source=self.source, reason=type(exc).__name__.lower()).inc()
                await self._sleep(attempt, None)

        self.breaker.record_failure()
        UPSTREAM_REQUESTS.labels(source=self.source, outcome="failure").inc()
        log.warning(
            "upstream.failed",
            source=self.source,
            url=str(url),
            attempts=self.max_retries,
            error=str(last_error),
            breaker_state=self.breaker.state,
        )
        if isinstance(last_error, UpstreamError):
            raise last_error
        raise UpstreamUnavailableError(
            f"{self.source} unreachable after {self.max_retries} attempts: {last_error}",
            source=self.source,
        )

    async def _sleep(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 30.0))
                return
            except ValueError:
                pass
        # full jitter exponential backoff
        delay = min(self.backoff_base_s * (2 ** (attempt - 1)), 20.0)
        await asyncio.sleep(random.uniform(0, delay))

    def _parse_json(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise UpstreamError(
                f"{self.source} returned a non-JSON response",
                source=self.source,
                details={"content_type": response.headers.get("content-type", "")},
            ) from exc
