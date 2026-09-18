"""Distributed rate limiting.

A fixed-window counter in Redis (or the in-memory cache for single-process dev). Fixed
windows are chosen over a token bucket because they need one atomic INCR, are trivial
to reason about in an incident, and the burst behaviour at window edges is acceptable
for a research workbench. Limits are per-principal, with a separate, much tighter
bucket for agent endpoints because those spend LLM tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from theragraph.core.cache import CacheBackend
from theragraph.core.clock import utcnow


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_in_s: int

    def __post_init__(self) -> None:
        # A negative "remaining" is meaningless to a client, and it would leak how far
        # over the limit a caller is. Clamp at the boundary.
        if self.remaining < 0:
            object.__setattr__(self, "remaining", 0)

    def headers(self) -> dict[str, str]:
        base = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_in_s),
        }
        if not self.allowed:
            base["Retry-After"] = str(self.reset_in_s)
        return base


class RateLimiter:
    def __init__(self, backend: CacheBackend, *, enabled: bool = True) -> None:
        self.backend = backend
        self.enabled = enabled

    async def check(
        self, identity: str, *, bucket: str, limit: int, window_s: int
    ) -> RateLimitDecision:
        if not self.enabled or limit <= 0:
            return RateLimitDecision(True, limit, limit, window_s)

        now = utcnow().timestamp()
        window_index = int(now // window_s)
        key = f"tg:rl:{bucket}:{window_index}:{identity}"
        count = await self.backend.incr(key, ttl_s=window_s + 1)
        reset_in = int((window_index + 1) * window_s - now) or 1

        # A Redis outage returns 0 from incr(); fail *open* so a cache blip cannot take
        # the API down, and rely on upstream infrastructure limits as the backstop.
        if count == 0:
            return RateLimitDecision(True, limit, limit, reset_in)

        return RateLimitDecision(
            allowed=count <= limit,
            limit=limit,
            remaining=limit - count,
            reset_in_s=reset_in,
        )
