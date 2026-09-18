"""Candidate prioritisation and comparison endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path

from theragraph.api.deps import ContextDep
from theragraph.ranking.config import list_presets, preset
from theragraph.schemas.requests import CompareRequest, PrioritizeRequest
from theragraph.services import prioritization

router = APIRouter(prefix="/ranking", tags=["ranking"])


@router.get("/presets", summary="Available ranking configurations")
async def presets() -> dict[str, Any]:
    """Every preset with its full objective list, curves and explanations.

    The UI renders the weight sliders and the "why" text from this, so the scoring
    model shown to a scientist is always the one the backend actually uses.
    """
    return {
        "presets": list_presets(),
        "configs": {item["name"]: preset(item["name"]).to_dict() for item in list_presets()},
    }


@router.post("", summary="Prioritise candidate compounds for a target")
async def prioritize(context: ContextDep, payload: PrioritizeRequest) -> dict[str, Any]:
    return await prioritization.prioritize(
        context,
        target_identifier=payload.target,
        disease_identifier=payload.disease,
        compound_ids=payload.compound_ids,
        config_name=payload.config_name,
        weights=payload.weights or None,
        min_pchembl=payload.min_pchembl,
        limit=payload.limit,
        include_similar_to=payload.include_similar_to,
        similarity_threshold=payload.similarity_threshold,
        run_predictions=payload.run_predictions,
        persist=payload.persist,
    )


@router.post("/compare", summary="Compare compounds side by side")
async def compare(context: ContextDep, payload: CompareRequest) -> dict[str, Any]:
    return await prioritization.compare(
        context,
        compound_ids=payload.compound_ids,
        target_identifier=payload.target,
        disease_identifier=payload.disease,
        config_name=payload.config_name,
        weights=payload.weights or None,
    )


@router.get("/{ranking_id}", summary="Retrieve a persisted ranking")
async def get_ranking(context: ContextDep, ranking_id: Annotated[str, Path()]) -> dict[str, Any]:
    return await prioritization.get_ranking(context, ranking_id)
