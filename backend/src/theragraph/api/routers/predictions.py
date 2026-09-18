"""ML inference endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from theragraph.api.deps import ContextDep
from theragraph.core.errors import ModelNotAvailableError, ValidationError
from theragraph.db.repositories import chemistry as chem_repo
from theragraph.ml.inference import summarize_predictions
from theragraph.schemas.requests import PredictRequest

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/endpoints", summary="Available prediction endpoints")
async def endpoints(context: ContextDep) -> dict[str, Any]:
    """Which endpoints have an active model, and what the model reports about itself."""
    inference = context.require_inference()
    return {"endpoints": await inference.available_endpoints(context.session)}


@router.post("", summary="Predict molecular properties")
async def predict(context: ContextDep, payload: PredictRequest) -> dict[str, Any]:
    """Score compounds and/or raw SMILES across one or more endpoints.

    Results carry uncertainty, calibrated confidence and applicability-domain status.
    Anything outside the training chemistry is flagged rather than quietly returned as
    equivalent to an in-domain prediction.
    """
    if not payload.has_input():
        raise ValidationError("provide compound_ids and/or smiles")

    inference = context.require_inference()
    molecules: list[tuple[str | None, str]] = []
    unresolved: list[str] = []

    for identifier in payload.compound_ids:
        compound = await chem_repo.get_compound(context.session, identifier)
        if compound is None:
            unresolved.append(identifier)
            continue
        molecules.append((compound.id, compound.canonical_smiles))
    molecules.extend((None, smiles) for smiles in payload.smiles)

    if not molecules:
        raise ValidationError(
            "none of the requested compounds are in the local corpus",
            details={"unresolved": unresolved},
        )

    endpoint_names = payload.endpoints or [
        item["endpoint"] for item in await inference.available_endpoints(context.session)
    ]
    if not endpoint_names:
        # "you did not request an endpoint" would misdescribe the problem: there are no
        # endpoints to request.
        raise ModelNotAvailableError(
            "no prediction models are registered on this deployment",
            details={
                "hint": "train one with `theragraph ml train --endpoint solubility --dataset esol`"
            },
        )
    result = await inference.predict(
        context.session,
        molecules=molecules,
        endpoints=endpoint_names,
        model_pins=payload.model_pins,
        persist=payload.persist,
    )
    return {
        **result.to_dict(),
        "summary_by_endpoint": summarize_predictions(result.predictions),
        "unresolved_compounds": unresolved,
    }
