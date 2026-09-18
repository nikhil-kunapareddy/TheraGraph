"""AI research assistant endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from theragraph.agent.orchestrator import start_research
from theragraph.agent.tools import TOOL_REGISTRY, tool_schemas
from theragraph.api.deps import ContextDep, agent_rate_limit, queue_dep
from theragraph.schemas.requests import AgentRunRequest

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/tools", summary="The agent's tool surface")
async def tools() -> dict[str, Any]:
    """Tool names, categories, descriptions and JSON schemas.

    Exposed so the UI can explain what the assistant is able to do, and so the same
    contract is visible to MCP clients.
    """
    return {
        "tools": [
            {
                "name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "mutates": spec.mutates,
                "expensive": spec.expensive,
                "input_schema": spec.schema()["input_schema"],
            }
            for spec in sorted(TOOL_REGISTRY.values(), key=lambda item: item.name)
        ],
        "default_exposed": len(tool_schemas()),
        "guarantees": [
            "Tools are the agent's only source of facts.",
            "Every tool call is schema-validated, persisted and auditable.",
            "Report claims citing records not retrieved in the run are removed.",
        ],
    }


@router.post(
    "/research",
    summary="Start a research run",
    dependencies=[Depends(agent_rate_limit)],
)
async def research(
    request: Request, context: ContextDep, payload: AgentRunRequest
) -> dict[str, Any]:
    """Create a reproducible research run and execute it.

    By default the run is enqueued and the response returns immediately with a run id
    to poll: a full workflow takes tens of seconds and spends model tokens, which does
    not belong in a request/response cycle.
    """
    queue = queue_dep(request)
    result = await start_research(
        context,
        query=payload.query,
        parameters=payload.parameters,
        idempotency_key=payload.idempotency_key,
        seed=payload.seed,
        execute_inline=payload.execute_inline,
        queue=None if payload.execute_inline else queue,
    )
    return {
        **result,
        "poll": f"{context.settings.api_prefix}/runs/{result['run_id']}",
    }
