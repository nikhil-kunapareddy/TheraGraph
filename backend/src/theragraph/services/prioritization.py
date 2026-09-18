"""Candidate prioritisation service.

Assembles candidates from the database (measured potency, target evidence, computed
descriptors), scores them with the ML models, computes structural novelty, runs the
ranking engine and persists the result with its exact configuration.

Novelty deserves a note. It is defined as ``1 - max(similarity to approved drugs,
scaffold crowding)``: a compound that is nearly identical to an approved drug is not
novel chemistry, and a compound built on the most over-represented scaffold in the
corpus is not novel either. Both components are computed from data, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from theragraph.chem.fingerprints import compute_fingerprint, tanimoto_bulk
from theragraph.core.errors import NotFoundError, ValidationError
from theragraph.core.logging import get_logger
from theragraph.db.repositories import biology as biology_repo
from theragraph.db.repositories import chemistry as chem_repo
from theragraph.db.repositories import ml as ml_repo
from theragraph.ml.inference import summarize_predictions
from theragraph.ranking import (
    CandidateInput,
    RankingConfig,
    preset,
    rank_candidates,
)
from theragraph.ranking.config import DEFAULT_CONFIG
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

#: Endpoints scored during prioritisation when a model is registered for them.
PRIORITIZATION_ENDPOINTS = (
    "solubility",
    "clinical_toxicity",
    "bbb_permeability",
    "lipophilicity",
)

MAX_CANDIDATES = 200


@dataclass
class NoveltyReference:
    """Reference sets used to judge structural novelty."""

    approved_drug_bits: list[bytes]
    common_scaffolds: dict[str, int]
    corpus_size: int


async def build_novelty_reference(
    context: ServiceContext, *, max_approved: int = 400
) -> NoveltyReference:
    approved = await chem_repo.filter_compounds(
        context.session, max_phase_min=4.0, limit=max_approved
    )
    bits: list[bytes] = []
    for compound in approved:
        try:
            bits.append(compute_fingerprint(compound.canonical_smiles).bits)
        except Exception:
            continue
    scaffolds = dict(await chem_repo.scaffold_counts(context.session, limit=25))
    return NoveltyReference(
        approved_drug_bits=bits,
        common_scaffolds=scaffolds,
        corpus_size=await chem_repo.count_compounds(context.session),
    )


def compute_novelty(
    smiles: str,
    scaffold_inchikey: str | None,
    reference: NoveltyReference,
) -> tuple[float | None, dict[str, Any]]:
    """Structural novelty in [0, 1] plus the components that produced it."""
    try:
        query_bits = compute_fingerprint(smiles).bits
    except Exception:
        return None, {"reason": "fingerprint failed"}

    similarity_to_approved = 0.0
    if reference.approved_drug_bits:
        scores = tanimoto_bulk(query_bits, reference.approved_drug_bits)
        similarity_to_approved = float(scores.max()) if scores.size else 0.0

    scaffold_count = reference.common_scaffolds.get(scaffold_inchikey or "", 0)
    max_scaffold_count = max(reference.common_scaffolds.values(), default=1) or 1
    scaffold_crowding = min(1.0, scaffold_count / max_scaffold_count)

    novelty = 1.0 - max(similarity_to_approved, scaffold_crowding)
    return round(max(0.0, novelty), 4), {
        "max_similarity_to_approved_drug": round(similarity_to_approved, 4),
        "scaffold_occurrences_in_corpus": scaffold_count,
        "scaffold_crowding": round(scaffold_crowding, 4),
        "definition": (
            "1 - max(Tanimoto to nearest approved drug, scaffold crowding in the local corpus)"
        ),
    }


async def assemble_candidates(
    context: ServiceContext,
    *,
    target_identifier: str,
    disease_identifier: str | None = None,
    compound_ids: list[str] | None = None,
    min_pchembl: float = 5.0,
    limit: int = 50,
    include_similar_to: str | None = None,
    similarity_threshold: float = 0.7,
    run_predictions: bool = True,
    research_run_id: str | None = None,
) -> tuple[list[CandidateInput], dict[str, Any]]:
    """Gather candidates and everything known about them."""
    target = await biology_repo.get_target(context.session, target_identifier)
    if target is None:
        raise NotFoundError(f"target {target_identifier!r} is not in the local corpus")

    disease = (
        await biology_repo.get_disease(context.session, disease_identifier)
        if disease_identifier
        else None
    )
    association = (
        await biology_repo.get_association(context.session, target.id, disease.id)
        if disease is not None
        else None
    )
    evidence_score = association.overall_score if association is not None else None
    evidence_rows = await biology_repo.list_evidence(
        context.session,
        target_id=target.id,
        disease_id=disease.id if disease else None,
        limit=25,
    )
    evidence_ids = [row.id for row in evidence_rows]

    potency_by_compound: dict[str, tuple[float, int]] = {}
    if compound_ids:
        compounds = await chem_repo.get_compounds_by_ids(context.session, compound_ids[:limit])
        activities = await chem_repo.bioactivities_for_target(
            context.session, target.id, limit=2000
        )
        for activity in activities:
            if activity.pchembl_value is None:
                continue
            best, count = potency_by_compound.get(activity.compound_id, (0.0, 0))
            potency_by_compound[activity.compound_id] = (
                max(best, activity.pchembl_value),
                count + 1,
            )
    else:
        known = await chem_repo.known_compounds_for_target(
            context.session, target.id, min_pchembl=min_pchembl, limit=limit
        )
        compounds = [compound for compound, _, _ in known]
        potency_by_compound = {compound.id: (best, count) for compound, best, count in known}

    # Optionally widen the candidate set with structural neighbours of a chosen hit.
    if include_similar_to:
        from theragraph.services.compounds import find_similar

        similar = await find_similar(
            context,
            smiles=include_similar_to,
            threshold=similarity_threshold,
            limit=limit,
        )
        extra_ids = [hit["compound_id"] for hit in similar["hits"]]
        existing = {compound.id for compound in compounds}
        extra = await chem_repo.get_compounds_by_ids(
            context.session, [cid for cid in extra_ids if cid not in existing]
        )
        compounds = [*compounds, *extra]

    compounds = compounds[:MAX_CANDIDATES]
    if not compounds:
        return [], {
            "target": target.approved_symbol,
            "n_candidates": 0,
            "warning": (
                "No candidate compounds found for this target in the local corpus. "
                "Ingest its ChEMBL bioactivity first, or lower the potency threshold."
            ),
        }

    properties = await chem_repo.get_properties_for_compounds(
        context.session, [compound.id for compound in compounds]
    )
    novelty_reference = await build_novelty_reference(context)

    predictions_by_compound: dict[str, dict[str, dict[str, Any]]] = {}
    prediction_ids_by_compound: dict[str, list[str]] = {}
    models_used: list[dict[str, Any]] = []
    prediction_summary: dict[str, Any] = {}

    if run_predictions:
        inference = context.require_inference()
        endpoints = [
            endpoint["endpoint"]
            for endpoint in await inference.available_endpoints(context.session)
            if endpoint["endpoint"] in PRIORITIZATION_ENDPOINTS
        ]
        if endpoints:
            result = await inference.predict(
                context.session,
                molecules=[(compound.id, compound.canonical_smiles) for compound in compounds],
                endpoints=endpoints,
                research_run_id=research_run_id,
            )
            models_used = result.models_used
            prediction_summary = summarize_predictions(result.predictions)
            for prediction in result.predictions:
                if prediction.compound_id is None:
                    continue
                predictions_by_compound.setdefault(prediction.compound_id, {})[
                    prediction.endpoint
                ] = prediction.to_dict()
                if prediction.prediction_id:
                    prediction_ids_by_compound.setdefault(prediction.compound_id, []).append(
                        prediction.prediction_id
                    )

    candidates: list[CandidateInput] = []
    for compound in compounds:
        property_row = properties.get(compound.id)
        best_pchembl, n_measurements = potency_by_compound.get(compound.id, (None, 0))
        novelty, novelty_detail = compute_novelty(
            compound.canonical_smiles, compound.murcko_scaffold_inchikey, novelty_reference
        )
        candidates.append(
            CandidateInput(
                compound_id=compound.id,
                name=compound.display_name,
                smiles=compound.canonical_smiles,
                chembl_id=compound.chembl_id,
                best_pchembl=best_pchembl,
                n_measurements=n_measurements,
                evidence_score=evidence_score,
                evidence_count=len(evidence_ids),
                max_phase=compound.max_phase,
                withdrawn_flag=compound.withdrawn_flag,
                molecular_weight=property_row.molecular_weight if property_row else None,
                logp=property_row.logp if property_row else None,
                tpsa=property_row.tpsa if property_row else None,
                qed=property_row.qed if property_row else None,
                heavy_atom_count=(
                    property_row.heavy_atom_count if property_row else compound.heavy_atom_count
                ),
                hbd=property_row.hbd if property_row else None,
                hba=property_row.hba if property_row else None,
                rotatable_bonds=property_row.rotatable_bonds if property_row else None,
                passes_lipinski=property_row.passes_lipinski if property_row else None,
                novelty_score=novelty,
                scaffold_frequency=novelty_detail.get("scaffold_occurrences_in_corpus"),
                predictions=predictions_by_compound.get(compound.id, {}),
                evidence_ids=evidence_ids,
                prediction_ids=prediction_ids_by_compound.get(compound.id, []),
            )
        )

    metadata = {
        "target": {
            "id": target.id,
            "symbol": target.approved_symbol,
            "target_key": target.target_key,
        },
        "disease": (
            {"id": disease.id, "name": disease.name, "efo_id": disease.efo_id} if disease else None
        ),
        "association_score": evidence_score,
        "n_candidates": len(candidates),
        "n_with_measured_potency": sum(
            1 for candidate in candidates if candidate.best_pchembl is not None
        ),
        "models_used": models_used,
        "prediction_summary": prediction_summary,
        "novelty_reference": {
            "approved_drugs_compared": len(novelty_reference.approved_drug_bits),
            "corpus_size": novelty_reference.corpus_size,
        },
        "evidence_ids": evidence_ids,
    }
    return candidates, metadata


async def prioritize(
    context: ServiceContext,
    *,
    target_identifier: str,
    disease_identifier: str | None = None,
    compound_ids: list[str] | None = None,
    config_name: str | None = None,
    config_override: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    min_pchembl: float = 5.0,
    limit: int = 50,
    include_similar_to: str | None = None,
    similarity_threshold: float = 0.7,
    run_predictions: bool = True,
    persist: bool = True,
    research_run_id: str | None = None,
) -> dict[str, Any]:
    """Full prioritisation workflow, persisted as a reproducible ranking."""
    config = _resolve_config(config_name, config_override, weights)

    candidates, metadata = await assemble_candidates(
        context,
        target_identifier=target_identifier,
        disease_identifier=disease_identifier,
        compound_ids=compound_ids,
        min_pchembl=min_pchembl,
        limit=limit,
        include_similar_to=include_similar_to,
        similarity_threshold=similarity_threshold,
        run_predictions=run_predictions,
        research_run_id=research_run_id,
    )
    if not candidates:
        return {
            "ranking_id": None,
            "config": config.to_dict(),
            "candidates": [],
            "n_ranked": 0,
            "metadata": metadata,
            "warnings": [metadata.get("warning", "no candidates available")],
        }

    outcome = rank_candidates(candidates, config)
    payload = outcome.to_dict()
    payload["metadata"] = metadata

    if persist:
        ranking = await ml_repo.create_ranking(
            context.session,
            scores=outcome.score_rows(),
            name=(
                f"{metadata['target']['symbol']} candidates"
                + (f" for {metadata['disease']['name']}" if metadata.get("disease") else "")
            ),
            target_id=metadata["target"]["id"],
            disease_id=(metadata.get("disease") or {}).get("id"),
            research_run_id=research_run_id,
            config=config.to_dict(),
            model_versions={
                model["endpoint"]: model["ref"] for model in metadata.get("models_used", [])
            },
            n_candidates=len(outcome.ranked),
            n_filtered_out=len(outcome.filtered_out),
            config_hash=config.config_hash,
        )
        payload["ranking_id"] = ranking.id
    else:
        payload["ranking_id"] = None

    return payload


def _resolve_config(
    config_name: str | None,
    config_override: dict[str, Any] | None,
    weights: dict[str, float] | None,
) -> RankingConfig:
    if config_override:
        config = RankingConfig.from_dict(config_override)
    elif config_name:
        config = preset(config_name)
    else:
        config = DEFAULT_CONFIG
    if weights:
        unknown = set(weights) - {objective.key for objective in config.objectives}
        if unknown:
            raise ValidationError(
                "unknown objective keys in weights",
                details={
                    "unknown": sorted(unknown),
                    "available": [objective.key for objective in config.objectives],
                },
            )
        config = config.with_weights(weights)
    return config


async def get_ranking(context: ServiceContext, ranking_id: str) -> dict[str, Any]:
    ranking = await ml_repo.get_ranking(context.session, ranking_id)
    if ranking is None:
        raise NotFoundError(f"ranking {ranking_id!r} not found")
    return {
        "ranking_id": ranking.id,
        "name": ranking.name,
        "created_at": ranking.created_at.isoformat(),
        "config": ranking.config,
        "config_hash": ranking.config_hash,
        "model_versions": ranking.model_versions,
        "n_candidates": ranking.n_candidates,
        "n_filtered_out": ranking.n_filtered_out,
        "research_run_id": ranking.research_run_id,
        "candidates": [
            {
                "compound_id": score.compound_id,
                "rank": score.rank,
                "total_score": round(score.total_score, 4),
                "factors": score.factors,
                "flags": score.flags,
                "evidence_ids": score.evidence_ids,
                "prediction_ids": score.prediction_ids,
                "compound": (
                    {
                        "name": score.compound.display_name,
                        "chembl_id": score.compound.chembl_id,
                        "smiles": score.compound.canonical_smiles,
                        "max_phase": score.compound.max_phase,
                    }
                    if score.compound is not None
                    else None
                ),
            }
            for score in sorted(ranking.scores, key=lambda item: item.rank)
        ],
    }


async def compare(
    context: ServiceContext,
    *,
    compound_ids: list[str],
    target_identifier: str | None = None,
    disease_identifier: str | None = None,
    config_name: str | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Side-by-side comparison of chosen compounds.

    When a target is supplied the comparison includes measured potency against it;
    without one, only structure-derived and predicted properties are compared, and the
    response says so rather than leaving a suspicious column of nulls unexplained.
    """
    if not compound_ids:
        raise ValidationError("at least one compound id is required")
    if len(compound_ids) > 12:
        raise ValidationError("compare at most 12 compounds at a time")

    from theragraph.ranking.engine import compare_candidates

    if target_identifier:
        payload = await prioritize(
            context,
            target_identifier=target_identifier,
            disease_identifier=disease_identifier,
            compound_ids=compound_ids,
            config_name=config_name,
            weights=weights,
            persist=False,
        )
        ranked_lookup = {item["compound_id"]: item for item in payload["candidates"]}
        outcome_config = payload["config"]
        metadata = payload["metadata"]
        note = None
    else:
        _, metadata, outcome_config, ranked_lookup = await _compare_without_target(
            context, compound_ids, config_name, weights
        )
        note = (
            "No target supplied: measured potency and target evidence are excluded from "
            "the comparison."
        )

    ordered = [
        ranked_lookup[compound_id] for compound_id in compound_ids if compound_id in ranked_lookup
    ]
    missing = [compound_id for compound_id in compound_ids if compound_id not in ranked_lookup]

    from theragraph.ranking.engine import FactorScore, RankedCandidate

    reconstructed = [
        RankedCandidate(
            compound_id=item["compound_id"],
            rank=item["rank"],
            total_score=item["total_score"],
            factors=[
                FactorScore(
                    key=factor["key"],
                    label=factor["label"],
                    basis=factor["basis"],
                    raw_value=factor["raw_value"],
                    desirability=factor["desirability"],
                    weight=factor["weight"],
                    normalized_weight=factor["normalized_weight"],
                    contribution=factor["contribution"],
                    explanation=factor["explanation"],
                    curve=factor["curve"],
                    context=factor.get("context") or {},
                    discounted=factor.get("discounted_for_low_confidence", False),
                    missing_reason=factor.get("missing_reason"),
                )
                for factor in item["factors"]
            ],
            flags=item.get("flags") or [],
            name=item.get("name"),
            smiles=item.get("smiles"),
            chembl_id=item.get("chembl_id"),
        )
        for item in ordered
    ]

    comparison = compare_candidates(reconstructed)
    comparison["config"] = outcome_config
    comparison["metadata"] = metadata
    comparison["excluded"] = missing
    if note:
        comparison["note"] = note
    return comparison


async def _compare_without_target(
    context: ServiceContext,
    compound_ids: list[str],
    config_name: str | None,
    weights: dict[str, float] | None,
) -> tuple[list[CandidateInput], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Comparison path with no target: drop the target-dependent objectives."""
    base = _resolve_config(config_name, None, weights)
    config = RankingConfig(
        name=f"{base.name}+no_target",
        description=base.description + " (target-dependent objectives removed)",
        objectives=[
            objective
            for objective in base.objectives
            if objective.key not in ("measured_potency", "evidence_support")
        ],
        filters=base.filters,
        min_objectives_present=2,
        confidence_penalty=base.confidence_penalty,
        notes=[*base.notes, "compared without a target context"],
    )

    compounds = await chem_repo.get_compounds_by_ids(context.session, compound_ids)
    if not compounds:
        raise NotFoundError("none of the requested compounds are in the local corpus")

    properties = await chem_repo.get_properties_for_compounds(
        context.session, [compound.id for compound in compounds]
    )
    reference = await build_novelty_reference(context)

    inference = context.require_inference()
    endpoints = [
        endpoint["endpoint"]
        for endpoint in await inference.available_endpoints(context.session)
        if endpoint["endpoint"] in PRIORITIZATION_ENDPOINTS
    ]
    predictions_by_compound: dict[str, dict[str, dict[str, Any]]] = {}
    models_used: list[dict[str, Any]] = []
    if endpoints:
        result = await inference.predict(
            context.session,
            molecules=[(compound.id, compound.canonical_smiles) for compound in compounds],
            endpoints=endpoints,
        )
        models_used = result.models_used
        for prediction in result.predictions:
            if prediction.compound_id:
                predictions_by_compound.setdefault(prediction.compound_id, {})[
                    prediction.endpoint
                ] = prediction.to_dict()

    candidates = []
    for compound in compounds:
        property_row = properties.get(compound.id)
        novelty, _ = compute_novelty(
            compound.canonical_smiles, compound.murcko_scaffold_inchikey, reference
        )
        candidates.append(
            CandidateInput(
                compound_id=compound.id,
                name=compound.display_name,
                smiles=compound.canonical_smiles,
                chembl_id=compound.chembl_id,
                max_phase=compound.max_phase,
                withdrawn_flag=compound.withdrawn_flag,
                molecular_weight=property_row.molecular_weight if property_row else None,
                logp=property_row.logp if property_row else None,
                tpsa=property_row.tpsa if property_row else None,
                qed=property_row.qed if property_row else None,
                heavy_atom_count=(
                    property_row.heavy_atom_count if property_row else compound.heavy_atom_count
                ),
                novelty_score=novelty,
                predictions=predictions_by_compound.get(compound.id, {}),
            )
        )

    outcome = rank_candidates(candidates, config)
    lookup = {item.compound_id: item.to_dict() for item in outcome.ranked}
    metadata = {
        "target": None,
        "disease": None,
        "n_candidates": len(candidates),
        "models_used": models_used,
    }
    return candidates, metadata, config.to_dict(), lookup
