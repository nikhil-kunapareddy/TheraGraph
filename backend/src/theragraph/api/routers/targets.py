"""Target explorer endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query

from theragraph.api.deps import ContextDep
from theragraph.services import targets as targets_service

router = APIRouter(prefix="/targets", tags=["targets"])


@router.get("/{identifier}", summary="Target detail, evidence, tractability, chemistry")
async def get_target(
    context: ContextDep,
    identifier: Annotated[str, Path(description="Symbol, Ensembl id, or internal id.")],
    disease: Annotated[str | None, Query(description="Disease context.")] = None,
    evidence_limit: Annotated[int, Query(ge=1, le=200)] = 50,
    compound_limit: Annotated[int, Query(ge=1, le=100)] = 25,
    min_pchembl: Annotated[float, Query(ge=0.0, le=15.0)] = 6.0,
) -> dict[str, Any]:
    return await targets_service.get_target_overview(
        context,
        identifier,
        disease_identifier=disease,
        evidence_limit=evidence_limit,
        compound_limit=compound_limit,
        min_pchembl=min_pchembl,
    )


@router.get("/{identifier}/evidence", summary="Filtered evidence for a target")
async def get_target_evidence(
    context: ContextDep,
    identifier: Annotated[str, Path()],
    disease: Annotated[str | None, Query()] = None,
    evidence_types: Annotated[list[str] | None, Query()] = None,
    sources: Annotated[list[str] | None, Query()] = None,
    min_score: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    return await targets_service.get_target_evidence(
        context,
        identifier,
        disease_identifier=disease,
        evidence_types=evidence_types,
        sources=sources,
        min_score=min_score,
        limit=limit,
        offset=offset,
    )
