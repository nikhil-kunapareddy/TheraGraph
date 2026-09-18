"""Compound explorer service: structure, properties, activity, similarity, predictions."""

from __future__ import annotations

from typing import Any

from theragraph.chem.descriptors import describe_descriptor
from theragraph.chem.fingerprints import DEFAULT_KIND, compute_fingerprint
from theragraph.chem.molecule import safe_standardize, standardize_smiles, to_molblock
from theragraph.chem.similarity import (
    matches_substructure,
    popcount_window,
    rank_by_tanimoto,
    screen_substructure_candidates,
    substructure_query_fingerprint,
)
from theragraph.core.errors import ChemistryError, NotFoundError, ValidationError
from theragraph.core.logging import get_logger
from theragraph.db.models import Compound, MolecularProperty
from theragraph.db.repositories import chemistry as chem_repo
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

CURATED_PROPERTY_FIELDS = (
    "molecular_weight",
    "exact_mass",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_rings",
    "ring_count",
    "heavy_atom_count",
    "fraction_csp3",
    "formal_charge",
    "stereocentres",
    "qed",
    "molar_refractivity",
    "heteroatom_count",
)


def _properties_dict(properties: MolecularProperty | None) -> dict[str, Any] | None:
    if properties is None:
        return None
    values = {
        field: getattr(properties, field)
        for field in CURATED_PROPERTY_FIELDS
        if getattr(properties, field) is not None
    }
    return {
        "provider": properties.provider,
        "descriptor_set_version": properties.descriptor_set_version,
        "rdkit_version": properties.rdkit_version,
        "values": values,
        "dictionary": [describe_descriptor(field) for field in values],
        "rule_filters": {
            "lipinski_violations": properties.lipinski_violations,
            "passes_lipinski": properties.passes_lipinski,
            "passes_veber": properties.passes_veber,
            "cns_mpo_score": properties.cns_mpo_score,
            "cns_mpo_caveat": (
                "LogD and pKa approximated from LogP; indicative only."
                if properties.cns_mpo_score is not None
                else None
            ),
        },
        "n_full_descriptors": len(properties.descriptors or {}),
        "basis": "computed from the standardised structure with RDKit",
    }


def _compound_dict(compound: Compound) -> dict[str, Any]:
    return {
        "id": compound.id,
        "inchikey": compound.inchikey,
        "smiles": compound.canonical_smiles,
        "inchi": compound.inchi,
        "name": compound.display_name,
        "synonyms": (compound.synonyms or [])[:10],
        "chembl_id": compound.chembl_id,
        "pubchem_cid": compound.pubchem_cid,
        "drugbank_id": compound.drugbank_id,
        "max_phase": compound.max_phase,
        "max_phase_meaning": _phase_meaning(compound.max_phase),
        "molecule_type": compound.molecule_type,
        "first_approval": compound.first_approval,
        "is_approved_drug": compound.is_approved_drug,
        "withdrawn_flag": compound.withdrawn_flag,
        "oral": compound.oral,
        "parenteral": compound.parenteral,
        "murcko_scaffold": compound.murcko_scaffold,
        "heavy_atom_count": compound.heavy_atom_count,
        "provenance": {
            "source": compound.source,
            "source_identifier": compound.source_identifier,
            "source_url": compound.source_url,
            "retrieved_at": compound.retrieved_at.isoformat(),
        },
    }


def _phase_meaning(max_phase: float | None) -> str | None:
    if max_phase is None:
        return None
    return {
        0.0: "Preclinical / research compound",
        0.5: "Early phase 1",
        1.0: "Phase 1 (safety in humans)",
        2.0: "Phase 2 (efficacy signal)",
        3.0: "Phase 3 (large-scale efficacy)",
        4.0: "Approved",
    }.get(float(max_phase), f"ChEMBL max phase {max_phase}")


async def get_compound_overview(
    context: ServiceContext,
    identifier: str,
    *,
    include_bioactivity: bool = True,
    include_similar: bool = True,
    similarity_threshold: float = 0.7,
    similar_limit: int = 10,
    include_structure: bool = False,
) -> dict[str, Any]:
    compound = await chem_repo.get_compound(context.session, identifier)
    if compound is None:
        raise NotFoundError(
            f"compound {identifier!r} is not in the local corpus",
            details={"identifier": identifier},
        )

    properties = await chem_repo.get_properties(context.session, compound.id)
    pubchem_properties = await chem_repo.get_properties(
        context.session, compound.id, provider="pubchem"
    )

    payload: dict[str, Any] = {
        "compound": _compound_dict(compound),
        "properties": _properties_dict(properties),
        "external_properties": _properties_dict(pubchem_properties),
    }

    if include_structure:
        try:
            payload["molblock"] = to_molblock(compound.canonical_smiles)
        except ChemistryError:
            payload["molblock"] = None

    if include_bioactivity:
        activities = await chem_repo.bioactivities_for_compound(
            context.session, compound.id, limit=50
        )
        targets = await chem_repo.targets_for_compound(context.session, compound.id)
        payload["bioactivity"] = {
            "measurements": [
                {
                    "id": activity.id,
                    "target": (
                        {
                            "id": activity.target.id,
                            "symbol": activity.target.approved_symbol,
                            "name": activity.target.approved_name,
                        }
                        if activity.target is not None
                        else None
                    ),
                    "activity_type": activity.activity_type,
                    "standard_value": activity.standard_value,
                    "standard_units": activity.standard_units,
                    "standard_relation": activity.standard_relation,
                    "pchembl_value": activity.pchembl_value,
                    "is_active": activity.is_active,
                    "assay": (
                        {
                            "chembl_assay_id": activity.assay.chembl_assay_id,
                            "description": activity.assay.description,
                            "assay_type": activity.assay.assay_type,
                            "organism": activity.assay.organism,
                            "confidence_score": activity.assay.confidence_score,
                            "confidence_meaning": (
                                "ChEMBL assay confidence 0-9; >=8 means the assay maps "
                                "to a single molecular target."
                            ),
                        }
                        if activity.assay is not None
                        else None
                    ),
                    "data_validity_comment": activity.data_validity_comment,
                    "quality_warning": bool(activity.data_validity_comment)
                    or bool(activity.potential_duplicate),
                    "provenance": {
                        "source": activity.source,
                        "source_identifier": activity.source_identifier,
                        "source_url": activity.source_url,
                        "retrieved_at": activity.retrieved_at.isoformat(),
                    },
                    "is_experimental_measurement": True,
                }
                for activity in activities
            ],
            "targets": [
                {
                    "target_id": target.id,
                    "symbol": target.approved_symbol,
                    "name": target.approved_name,
                    "best_pchembl": round(best, 3),
                    "n_measurements": count,
                }
                for target, best, count in targets
            ],
            "count": len(activities),
        }

    if include_similar:
        payload["similar_compounds"] = await find_similar(
            context,
            smiles=compound.canonical_smiles,
            threshold=similarity_threshold,
            limit=similar_limit,
            exclude_compound_id=compound.id,
        )

    return payload


async def find_similar(
    context: ServiceContext,
    *,
    smiles: str,
    threshold: float = 0.7,
    limit: int = 25,
    fingerprint_kind: str = DEFAULT_KIND,
    exclude_compound_id: str | None = None,
    candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Local Tanimoto similarity search over the ingested corpus.

    The popcount window narrows the candidate set in SQL before any similarity is
    computed; the returned ``search_stats`` report how effective that was, which is the
    difference between a search that scales and one that scans.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValidationError("similarity threshold must be in (0, 1]")

    info = standardize_smiles(smiles)
    query = compute_fingerprint(info.canonical_smiles, fingerprint_kind)
    low, high = popcount_window(query.popcount, threshold)

    candidates = await chem_repo.iter_fingerprints_for_similarity(
        context.session,
        kind=fingerprint_kind,
        min_popcount=low,
        max_popcount=high,
        exclude_compound_id=exclude_compound_id,
        candidate_ids=candidate_ids,
    )
    hits = rank_by_tanimoto(query.bits, candidates, threshold=threshold, limit=limit)

    compounds = await chem_repo.get_compounds_by_ids(
        context.session, [hit.compound_id for hit in hits]
    )
    by_id = {compound.id: compound for compound in compounds}
    properties = await chem_repo.get_properties_for_compounds(
        context.session, [hit.compound_id for hit in hits]
    )

    total_fingerprints = await chem_repo.count_compounds(context.session)
    return {
        "query": {
            "input_smiles": smiles,
            "canonical_smiles": info.canonical_smiles,
            "inchikey": info.inchikey,
            "fingerprint_kind": fingerprint_kind,
            "popcount": query.popcount,
            "standardization_notes": info.standardization_notes,
        },
        "threshold": threshold,
        "hits": [
            {
                "compound_id": hit.compound_id,
                "similarity": round(hit.similarity, 4),
                "name": by_id[hit.compound_id].display_name if hit.compound_id in by_id else None,
                "chembl_id": by_id[hit.compound_id].chembl_id if hit.compound_id in by_id else None,
                "smiles": by_id[hit.compound_id].canonical_smiles
                if hit.compound_id in by_id
                else None,
                "max_phase": by_id[hit.compound_id].max_phase if hit.compound_id in by_id else None,
                "shares_scaffold": (
                    by_id[hit.compound_id].murcko_scaffold_inchikey == info.murcko_scaffold_inchikey
                    if hit.compound_id in by_id and info.murcko_scaffold_inchikey
                    else None
                ),
                "properties": {
                    key: getattr(properties[hit.compound_id], key)
                    for key in ("molecular_weight", "logp", "tpsa", "qed")
                    if hit.compound_id in properties
                    and getattr(properties[hit.compound_id], key) is not None
                },
            }
            for hit in hits
        ],
        "count": len(hits),
        "search_stats": {
            "corpus_compounds": total_fingerprints,
            "popcount_window": [low, high],
            "candidates_examined": len(candidates),
            "screened_out_fraction": round(1 - (len(candidates) / total_fingerprints), 4)
            if total_fingerprints
            else 0.0,
            "method": "exact Tanimoto over Morgan fingerprints, popcount-bounded",
        },
    }


async def substructure_search(
    context: ServiceContext, *, smarts: str, limit: int = 25
) -> dict[str, Any]:
    """SMARTS substructure search: fingerprint screen, then exact graph matching.

    The screen is a necessary condition, not a sufficient one, so every survivor is
    confirmed with a real substructure match before being returned.
    """
    query_bits = substructure_query_fingerprint(smarts)
    all_fingerprints = await chem_repo.all_fingerprints(context.session, kind="rdkit_pattern_2048")
    screened = screen_substructure_candidates(query_bits, all_fingerprints)

    compounds = await chem_repo.get_compounds_by_ids(context.session, screened)
    matches: list[dict[str, Any]] = []
    for compound in compounds:
        try:
            if matches_substructure(compound.canonical_smiles, smarts):
                matches.append(
                    {
                        "compound_id": compound.id,
                        "name": compound.display_name,
                        "chembl_id": compound.chembl_id,
                        "smiles": compound.canonical_smiles,
                        "max_phase": compound.max_phase,
                    }
                )
        except ChemistryError:
            continue
        if len(matches) >= limit:
            break

    return {
        "smarts": smarts,
        "matches": matches,
        "count": len(matches),
        "search_stats": {
            "corpus_compounds": len(all_fingerprints),
            "passed_fingerprint_screen": len(screened),
            "confirmed_by_graph_match": len(matches),
            "screen_efficiency": round(1 - len(screened) / len(all_fingerprints), 4)
            if all_fingerprints
            else 0.0,
        },
    }


async def compute_properties_for_smiles(smiles: str) -> dict[str, Any]:
    """Ad-hoc property computation for a pasted structure (nothing persisted)."""
    from theragraph.chem.descriptors import compute_descriptors

    info, error = safe_standardize(smiles)
    if info is None:
        raise ChemistryError(error or "could not parse SMILES", details={"smiles": smiles[:200]})

    descriptors = compute_descriptors(info.canonical_smiles)
    return {
        "input_smiles": smiles,
        "canonical_smiles": info.canonical_smiles,
        "inchikey": info.inchikey,
        "formula": info.formula,
        "murcko_scaffold": info.murcko_scaffold,
        "standardization_notes": info.standardization_notes,
        "properties": {
            "provider": "rdkit",
            "descriptor_set_version": descriptors.descriptor_set_version,
            "rdkit_version": descriptors.rdkit_version,
            "values": {
                key: value for key, value in descriptors.curated.items() if value is not None
            },
            "dictionary": [
                describe_descriptor(key)
                for key, value in descriptors.curated.items()
                if value is not None
            ],
            "rule_filters": descriptors.rule_filters,
            "basis": "computed from the standardised structure with RDKit",
        },
        "persisted": False,
    }
