"""Compound explorer, similarity, substructure and descriptor endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query

from theragraph.api.deps import ContextDep
from theragraph.chem.descriptors import descriptor_dictionary
from theragraph.chem.fingerprints import FINGERPRINT_KINDS
from theragraph.schemas.requests import (
    PropertiesRequest,
    SimilarityRequest,
    SubstructureRequest,
)
from theragraph.services import compounds as compounds_service

router = APIRouter(prefix="/compounds", tags=["compounds"])


@router.get("/descriptors", summary="Descriptor dictionary")
async def descriptors() -> dict[str, Any]:
    """Every curated descriptor with its label, unit and meaning.

    Served so the UI never hard-codes scientific explanations of the numbers it shows.
    """
    return {
        "descriptors": descriptor_dictionary(),
        "fingerprints": [{"kind": kind, **spec} for kind, spec in FINGERPRINT_KINDS.items()],
    }


@router.post("/properties", summary="Compute descriptors for a structure")
async def compute_properties(payload: PropertiesRequest) -> dict[str, Any]:
    """Ad-hoc computation for a pasted SMILES. Nothing is persisted."""
    return await compounds_service.compute_properties_for_smiles(payload.smiles)


@router.post("/similarity", summary="Tanimoto similarity search")
async def similarity(context: ContextDep, payload: SimilarityRequest) -> dict[str, Any]:
    return await compounds_service.find_similar(
        context,
        smiles=payload.smiles,
        threshold=payload.threshold,
        limit=payload.limit,
        fingerprint_kind=payload.fingerprint_kind,
    )


@router.post("/substructure", summary="SMARTS substructure search")
async def substructure(context: ContextDep, payload: SubstructureRequest) -> dict[str, Any]:
    return await compounds_service.substructure_search(
        context, smarts=payload.smarts, limit=payload.limit
    )


@router.get("/{identifier}", summary="Compound detail with activity and neighbours")
async def get_compound(
    context: ContextDep,
    identifier: Annotated[str, Path(description="Internal id, ChEMBL id, CID or InChIKey.")],
    include_bioactivity: Annotated[bool, Query()] = True,
    include_similar: Annotated[bool, Query()] = True,
    similarity_threshold: Annotated[float, Query(ge=0.3, le=1.0)] = 0.7,
    include_structure: Annotated[
        bool, Query(description="Include a 2D molblock for rendering.")
    ] = False,
) -> dict[str, Any]:
    return await compounds_service.get_compound_overview(
        context,
        identifier,
        include_bioactivity=include_bioactivity,
        include_similar=include_similar,
        similarity_threshold=similarity_threshold,
        include_structure=include_structure,
    )
