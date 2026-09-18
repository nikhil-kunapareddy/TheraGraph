"""Disease endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query

from theragraph.api.deps import ContextDep
from theragraph.services import discovery

router = APIRouter(prefix="/diseases", tags=["diseases"])


@router.get("/{identifier}", summary="Disease detail with associated targets")
async def get_disease(
    context: ContextDep,
    identifier: Annotated[str, Path(description="Internal id or EFO/MONDO id.")],
    n_targets: Annotated[int, Query(ge=1, le=100)] = 25,
    min_score: Annotated[
        float, Query(ge=0.0, le=1.0, description="Minimum association score.")
    ] = 0.0,
) -> dict[str, Any]:
    return await discovery.get_disease_overview(
        context, identifier, n_targets=n_targets, min_score=min_score
    )
