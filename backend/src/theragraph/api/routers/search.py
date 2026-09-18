"""Unified search and corpus description."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query

from theragraph.api.deps import ContextDep
from theragraph.services import discovery

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", summary="Search diseases, targets and compounds")
async def search(
    context: ContextDep,
    q: Annotated[str, Query(min_length=1, max_length=500, description="Free-text query.")],
    entity_types: Annotated[
        list[Literal["disease", "target", "compound"]] | None,
        Query(description="Restrict to these entity types."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    include_upstream: Annotated[
        bool,
        Query(
            description=(
                "When nothing matches locally, also query Open Targets and return "
                "discoverable (not yet ingested) entities."
            )
        ),
    ] = False,
) -> dict[str, Any]:
    return await discovery.search(
        context,
        q,
        entity_types=entity_types,
        limit=limit,
        include_upstream=include_upstream,
    )


@router.get("/corpus", summary="What data is currently ingested")
async def corpus(context: ContextDep) -> dict[str, Any]:
    """Entity counts, sources with licences, and current dataset versions."""
    return await discovery.corpus_summary(context)
