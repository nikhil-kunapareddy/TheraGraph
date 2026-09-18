"""Evidence graph endpoint."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from theragraph.api.deps import ContextDep
from theragraph.services import graph as graph_service

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("", summary="Build the evidence graph")
async def evidence_graph(
    context: ContextDep,
    disease: Annotated[str | None, Query(description="Root disease.")] = None,
    target: Annotated[str | None, Query(description="Root or focus target.")] = None,
    compound: Annotated[str | None, Query(description="Root compound.")] = None,
    max_targets: Annotated[int, Query(ge=1, le=25)] = 8,
    max_evidence_per_target: Annotated[int, Query(ge=1, le=25)] = 6,
    max_compounds_per_target: Annotated[int, Query(ge=1, le=25)] = 6,
    include_predictions: Annotated[bool, Query()] = True,
) -> dict[str, Any]:
    """Nodes and edges for the disease -> target -> evidence -> compound -> prediction graph.

    Every edge names its source and retrieval date; prediction nodes are marked as
    model output so the renderer can keep them visually distinct from measurements.
    """
    return await graph_service.build_evidence_graph(
        context,
        disease_identifier=disease,
        target_identifier=target,
        compound_identifier=compound,
        max_targets=max_targets,
        max_evidence_per_target=max_evidence_per_target,
        max_compounds_per_target=max_compounds_per_target,
        include_predictions=include_predictions,
    )
