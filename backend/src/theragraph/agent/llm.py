"""LLM access for the agent.

Two interchangeable backends behind one interface:

* :class:`AnthropicBackend` -- the real thing, via the official ``anthropic`` SDK.
* :class:`DeterministicBackend` -- a rule-based stand-in that produces the same
  structured outputs from the same inputs, with no network and no API key.

The deterministic backend is not a mock for tests alone. It is a first-class product
mode: the whole workflow (planning, tool selection, report assembly, claim validation)
runs without an LLM, because every *fact* in a TheraGraph report comes from a tool
result, not from the model. That makes the demo runnable by anyone who clones the repo,
and it makes the eval suite able to isolate agent-harness defects from model variance.

Model IDs, pricing and API shapes here follow the Anthropic API as of 2026-06: adaptive
thinking (no ``budget_tokens``), ``output_config.effort``, structured outputs via
``messages.parse``, and server-side refusal fallbacks on Opus 5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel, ValidationError

from theragraph.config import Settings
from theragraph.core.clock import Stopwatch
from theragraph.core.errors import AgentError
from theragraph.core.logging import get_logger
from theragraph.core.metrics import AGENT_COST_USD, AGENT_TOKENS

log = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

#: USD per million tokens (input, output), Anthropic first-party API rates.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

#: Cached input tokens bill at roughly a tenth of the uncached rate.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

#: Models where thinking is on by default and ``budget_tokens`` is rejected.
ADAPTIVE_THINKING_MODELS = frozenset(
    {
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    }
)

#: Models that support server-side refusal fallbacks.
REFUSAL_FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-fable-5", "claude-opus-5"})
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ModelUsage:
    """Token and cost accounting for one or more LLM calls."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens

    @property
    def cost_usd(self) -> float:
        return estimate_cost_usd(
            self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
        )

    def add(self, other: ModelUsage) -> None:
        self.model = self.model or other.model
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.calls += other.calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Estimated spend for one call.

    Unknown model ids return 0.0 rather than guessing a price: a silently wrong cost
    figure is worse than an obviously absent one.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    input_rate, output_rate = pricing
    return (
        input_tokens * input_rate
        + cache_read_tokens * input_rate * CACHE_READ_MULTIPLIER
        + cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER
        + output_tokens * output_rate
    ) / 1_000_000


@dataclass
class ToolInvocation:
    """A tool call the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalised model response."""

    text: str = ""
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    parsed: Any = None
    stop_reason: str | None = None
    refusal_category: str | None = None
    usage: ModelUsage = field(default_factory=ModelUsage)
    latency_ms: float = 0.0
    backend: Literal["anthropic", "deterministic"] = "anthropic"
    raw_blocks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


class LLMBackend(Protocol):
    name: Literal["anthropic", "deterministic"]

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[BaseModel],
        max_tokens: int | None = ...,
    ) -> LLMResponse: ...

    async def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = ...,
        max_tokens: int | None = ...,
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...


class AnthropicBackend:
    """Anthropic Messages API backend."""

    name: Literal["anthropic", "deterministic"] = "anthropic"

    def __init__(self, settings: Settings) -> None:
        import anthropic

        if not settings.anthropic_api_key:
            raise AgentError(
                "ANTHROPIC_API_KEY is not set; set it or use AGENT_LLM_BACKEND=deterministic",
                details={"backend": "anthropic"},
            )
        self.settings = settings
        self.model = settings.agent_model
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            max_retries=3,
            timeout=float(settings.agent_tool_timeout_s * 2),
        )

    def _request_kwargs(self, max_tokens: int | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.settings.agent_max_tokens,
            # Adaptive thinking: the current API shape for Opus 5 / Sonnet 5 and the
            # 4.6+ family. `budget_tokens` is rejected by these models.
            "output_config": {"effort": self.settings.agent_effort},
        }
        if self.model in ADAPTIVE_THINKING_MODELS:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.model in REFUSAL_FALLBACK_MODELS:
            # Server-side fallback: if a safety classifier declines, the request is
            # re-routed rather than failing the whole research run.
            kwargs["betas"] = [REFUSAL_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _usage(self, response: Any) -> ModelUsage:
        usage = getattr(response, "usage", None)
        model_usage = ModelUsage(
            model=getattr(response, "model", self.model) or self.model, calls=1
        )
        if usage is not None:
            model_usage.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            model_usage.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            model_usage.cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            model_usage.cache_write_tokens = int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            )
        AGENT_TOKENS.labels(model=model_usage.model, kind="input").inc(model_usage.input_tokens)
        AGENT_TOKENS.labels(model=model_usage.model, kind="output").inc(model_usage.output_tokens)
        AGENT_TOKENS.labels(model=model_usage.model, kind="cache_read").inc(
            model_usage.cache_read_tokens
        )
        AGENT_COST_USD.labels(model=model_usage.model).inc(model_usage.cost_usd)
        return model_usage

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[BaseModel],
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Request a response validated against a Pydantic model."""
        import anthropic

        stopwatch = Stopwatch()
        kwargs = self._request_kwargs(max_tokens)
        # `messages.parse` is not on the beta path, so the beta-only fallback
        # parameters do not apply to it.
        kwargs.pop("betas", None)
        kwargs.pop("fallbacks", None)

        try:
            with stopwatch:
                response = await self._client.messages.parse(
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    output_format=schema_model,
                    **kwargs,
                )
        except anthropic.APIStatusError as exc:
            raise AgentError(
                f"Anthropic API error {exc.status_code}: {exc}",
                details={"status_code": exc.status_code},
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AgentError(f"could not reach the Anthropic API: {exc}") from exc

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return LLMResponse(
                stop_reason="refusal",
                refusal_category=getattr(details, "category", None),
                usage=self._usage(response),
                latency_ms=stopwatch.elapsed_ms,
                backend="anthropic",
            )

        return LLMResponse(
            text=_first_text(response),
            parsed=getattr(response, "parsed_output", None),
            stop_reason=stop_reason,
            usage=self._usage(response),
            latency_ms=stopwatch.elapsed_ms,
            backend="anthropic",
        )

    async def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One turn of a tool-using conversation.

        A manual loop rather than the SDK's tool runner: every call must be persisted
        with its retrieved record ids and validation verdict inside the caller's
        database transaction, and the same loop has to drive the deterministic backend.
        Owning the loop keeps those two requirements in one place.
        """
        import anthropic

        stopwatch = Stopwatch()
        kwargs = self._request_kwargs(max_tokens)
        if tools:
            kwargs["tools"] = tools
        # Cache the system prompt + tool definitions: they are identical across every
        # turn of a run, and they are the bulk of the input tokens.
        kwargs["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

        create = (
            self._client.beta.messages.create if "betas" in kwargs else self._client.messages.create
        )

        try:
            with stopwatch:
                # Both overloads want their own MessageParam TypedDict for what is
                # the same wire payload; the shape is asserted by the eval suite.
                response = await create(messages=cast(Any, messages), **kwargs)
        except anthropic.RateLimitError as exc:
            raise AgentError(
                "Anthropic rate limit reached; retry shortly", details={"retryable": True}
            ) from exc
        except anthropic.APIStatusError as exc:
            raise AgentError(
                f"Anthropic API error {exc.status_code}: {exc}",
                details={"status_code": exc.status_code},
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AgentError(f"could not reach the Anthropic API: {exc}") from exc

        stop_reason = getattr(response, "stop_reason", None)
        blocks = [
            block.model_dump() if hasattr(block, "model_dump") else dict(block)
            for block in getattr(response, "content", []) or []
        ]
        tool_calls = [
            ToolInvocation(
                id=str(block.get("id")),
                name=str(block.get("name")),
                arguments=dict(block.get("input") or {}),
            )
            for block in blocks
            if block.get("type") == "tool_use"
        ]

        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return LLMResponse(
                stop_reason="refusal",
                refusal_category=getattr(details, "category", None),
                usage=self._usage(response),
                latency_ms=stopwatch.elapsed_ms,
                backend="anthropic",
                raw_blocks=blocks,
            )

        return LLMResponse(
            text="\n".join(
                str(block.get("text", "")) for block in blocks if block.get("type") == "text"
            ).strip(),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=self._usage(response),
            latency_ms=stopwatch.elapsed_ms,
            backend="anthropic",
            raw_blocks=blocks,
        )

    async def aclose(self) -> None:
        await self._client.close()


def _first_text(response: Any) -> str:
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type == "text":
            return str(getattr(block, "text", None) or block.get("text", ""))
    return ""


class DeterministicBackend:
    """Rule-based planner/reporter used when no LLM is configured.

    Responsibilities are identical to the Anthropic backend's: return a valid structured
    object for a planning request, and return tool calls or a final answer for a
    conversation turn. It cannot invent scientific content because it has none -- the
    facts still come from the tools.
    """

    name: Literal["anthropic", "deterministic"] = "deterministic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = "deterministic-planner-v1"

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[BaseModel],
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from theragraph.agent.planner import deterministic_structured

        stopwatch = Stopwatch()
        with stopwatch:
            payload = deterministic_structured(schema_model, user)
        try:
            parsed = schema_model.model_validate(payload)
        except ValidationError as exc:  # pragma: no cover - guards planner changes
            raise AgentError(
                "deterministic backend produced an invalid structured payload",
                details={"errors": exc.errors()[:3], "schema": schema_model.__name__},
            ) from exc
        return LLMResponse(
            text=json.dumps(payload, default=str),
            parsed=parsed,
            stop_reason="end_turn",
            usage=ModelUsage(model=self.model, calls=1),
            latency_ms=stopwatch.elapsed_ms,
            backend="deterministic",
        )

    async def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """No conversational tool loop in this backend.

        The deterministic mode drives tools from the *plan* rather than from turn-by-turn
        model decisions, so the orchestrator never calls this. Returning an explicit
        end-of-turn (instead of raising) keeps the interface honest for callers that do.
        """
        return LLMResponse(
            text="",
            stop_reason="end_turn",
            usage=ModelUsage(model=self.model, calls=1),
            backend="deterministic",
        )

    async def aclose(self) -> None:
        return None


def build_llm(settings: Settings) -> LLMBackend:
    """Select a backend, falling back to deterministic when no key is configured."""
    if settings.agent_llm_backend == "deterministic":
        log.info("agent.llm_backend", backend="deterministic", reason="configured")
        return DeterministicBackend(settings)
    if not settings.anthropic_api_key:
        log.warning(
            "agent.llm_backend",
            backend="deterministic",
            reason="ANTHROPIC_API_KEY not set",
        )
        return DeterministicBackend(settings)
    log.info("agent.llm_backend", backend="anthropic", model=settings.agent_model)
    return AnthropicBackend(settings)
