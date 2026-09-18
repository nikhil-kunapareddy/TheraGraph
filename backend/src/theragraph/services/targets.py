"""Target explorer service: target detail, evidence, tractability, known chemistry."""

from __future__ import annotations

from typing import Any

from theragraph.core.errors import NotFoundError
from theragraph.core.logging import get_logger
from theragraph.db.models import Evidence
from theragraph.db.repositories import biology as biology_repo
from theragraph.db.repositories import chemistry as chem_repo
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

#: Order in which evidence types are presented. Genetics first because human genetic
#: evidence is the strongest available support for a target hypothesis; literature
#: last because co-mention is the weakest.
EVIDENCE_TYPE_ORDER = [
    "genetic_association",
    "known_drug",
    "somatic_mutation",
    "animal_model",
    "functional_genomics",
    "pathway",
    "expression",
    "biomarker",
    "literature",
    "other",
]

EVIDENCE_TYPE_MEANING = {
    "genetic_association": (
        "Human genetic variation linking the gene to the disease. The strongest "
        "available class of target-validation evidence."
    ),
    "known_drug": (
        "A drug or clinical candidate acting on this target is in development or "
        "approved for this indication."
    ),
    "somatic_mutation": "Somatic alterations in the gene observed in the disease.",
    "animal_model": "Phenotype in a model organism consistent with the disease.",
    "functional_genomics": (
        "CRISPR/RNAi screens implicating the gene in disease-relevant cell states."
    ),
    "pathway": "The gene participates in a pathway implicated in the disease.",
    "expression": "Differential expression of the gene in disease-relevant tissue.",
    "biomarker": "The gene or its product is used as a biomarker in this disease.",
    "literature": (
        "Co-mention in the scientific literature, extracted by text mining. Weak on its "
        "own: co-occurrence is not causation."
    ),
    "other": "Evidence from a source without a specific interpretation in TheraGraph.",
}


def _evidence_dict(evidence: Evidence) -> dict[str, Any]:
    """Serialise one evidence row with its full provenance.

    Every field shown to a user or handed to the agent comes from the source. There is
    no field here that a language model can write.
    """
    publication = evidence.publication
    return {
        "id": evidence.id,
        "evidence_type": evidence.evidence_type,
        "type_meaning": EVIDENCE_TYPE_MEANING.get(evidence.evidence_type),
        "relation": evidence.relation,
        "title": evidence.title,
        "summary": evidence.summary,
        "score": evidence.score,
        "score_type": evidence.score_type,
        "score_provided_by_source": evidence.score is not None,
        "citation_key": evidence.citation_key,
        "provenance": {
            "source": evidence.source,
            "source_identifier": evidence.source_identifier,
            "source_url": evidence.source_url,
            "retrieved_at": evidence.retrieved_at.isoformat(),
            "datasource_id": (evidence.extra or {}).get("datasource_id"),
        },
        "publication": (
            {
                "pmid": publication.pmid,
                "doi": publication.doi,
                "title": publication.title,
                "journal": publication.journal,
                "year": publication.year,
                "authors": (publication.authors or [])[:5],
                "citation_count": publication.citation_count,
                "is_open_access": publication.is_open_access,
                "url": publication.source_url,
            }
            if publication is not None
            else None
        ),
        "metadata": {
            key: value
            for key, value in (evidence.extra or {}).items()
            if key not in ("urls",) and value is not None
        },
        "urls": (evidence.extra or {}).get("urls") or [],
        "is_retrieved_fact": True,
    }


async def get_target_overview(
    context: ServiceContext,
    identifier: str,
    *,
    disease_identifier: str | None = None,
    evidence_limit: int = 50,
    compound_limit: int = 25,
    min_pchembl: float = 6.0,
) -> dict[str, Any]:
    """Everything the Target Explorer needs in one call."""
    target = await biology_repo.get_target(context.session, identifier)
    if target is None:
        raise NotFoundError(
            f"target {identifier!r} is not in the local corpus",
            details={"identifier": identifier},
        )

    disease = None
    association = None
    if disease_identifier:
        disease = await biology_repo.get_disease(context.session, disease_identifier)
        if disease is not None:
            association = await biology_repo.get_association(context.session, target.id, disease.id)

    evidence = await biology_repo.list_evidence(
        context.session,
        target_id=target.id,
        disease_id=disease.id if disease else None,
        limit=evidence_limit,
    )
    evidence_counts = await biology_repo.evidence_type_counts(
        context.session, target_id=target.id, disease_id=disease.id if disease else None
    )
    diseases = await biology_repo.diseases_for_target(context.session, target.id, limit=15)
    known_compounds = await chem_repo.known_compounds_for_target(
        context.session, target.id, min_pchembl=min_pchembl, limit=compound_limit
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        grouped.setdefault(item.evidence_type, []).append(_evidence_dict(item))

    ordered_groups = [
        {
            "evidence_type": evidence_type,
            "meaning": EVIDENCE_TYPE_MEANING.get(evidence_type),
            "count": evidence_counts.get(evidence_type, len(grouped[evidence_type])),
            "items": grouped[evidence_type],
        }
        for evidence_type in EVIDENCE_TYPE_ORDER
        if evidence_type in grouped
    ]

    return {
        "target": {
            "id": target.id,
            "target_key": target.target_key,
            "symbol": target.approved_symbol,
            "name": target.approved_name,
            "biotype": target.biotype,
            "description": target.description,
            "target_class": target.target_class or [],
            "synonyms": (target.synonyms or [])[:15],
            "uniprot_ids": target.uniprot_ids or [],
            "chembl_target_ids": target.chembl_target_ids or [],
            "subcellular_locations": target.subcellular_locations or [],
            "gene": (
                {
                    "ensembl_id": target.gene.ensembl_id,
                    "symbol": target.gene.symbol,
                    "chromosome": target.gene.chromosome,
                    "genomic_start": target.gene.genomic_start,
                    "genomic_end": target.gene.genomic_end,
                }
                if target.gene is not None
                else None
            ),
            "provenance": {
                "source": target.source,
                "source_identifier": target.source_identifier,
                "source_url": target.source_url,
                "retrieved_at": target.retrieved_at.isoformat(),
            },
        },
        "tractability": {
            "buckets": {
                key: value
                for key, value in (target.tractability or {}).items()
                if not key.startswith("_")
            },
            "all_buckets": (target.tractability or {}).get("_all_buckets", {}),
            "source": target.source,
            "meaning": (
                "Open Targets tractability assessment per modality, retrieved verbatim. "
                "Indicates whether a target is amenable to a modality, not whether a "
                "drug will work."
            ),
        },
        "safety_liabilities": target.safety_liabilities or [],
        "disease_context": (
            {
                "disease": {
                    "id": disease.id,
                    "efo_id": disease.efo_id,
                    "name": disease.name,
                },
                "association": (
                    {
                        "overall_score": round(association.overall_score, 5),
                        "datatype_scores": association.datatype_scores or {},
                        "source": association.source,
                        "source_url": association.source_url,
                        "retrieved_at": association.retrieved_at.isoformat(),
                    }
                    if association is not None
                    else None
                ),
            }
            if disease is not None
            else None
        ),
        "associated_diseases": [
            {
                "id": item.id,
                "efo_id": item.efo_id,
                "name": item.name,
                "overall_score": round(link.overall_score, 5),
                "datatype_scores": link.datatype_scores or {},
            }
            for item, link in diseases
        ],
        "evidence": {
            "groups": ordered_groups,
            "counts_by_type": evidence_counts,
            "total": sum(evidence_counts.values()),
            "returned": len(evidence),
        },
        "known_compounds": [
            {
                "compound_id": compound.id,
                "chembl_id": compound.chembl_id,
                "name": compound.display_name,
                "smiles": compound.canonical_smiles,
                "max_phase": compound.max_phase,
                "is_approved_drug": compound.is_approved_drug,
                "best_pchembl": round(best_pchembl, 3),
                "n_measurements": n_measurements,
                "potency_nm": round(10 ** (9 - best_pchembl), 2) if best_pchembl else None,
                "basis": "experimental measurement (ChEMBL)",
            }
            for compound, best_pchembl, n_measurements in known_compounds
        ],
        "chemistry_summary": {
            "n_known_compounds": len(known_compounds),
            "min_pchembl_filter": min_pchembl,
            "has_small_molecule_chemistry": bool(known_compounds),
            "note": (
                "No compounds above the potency threshold in the local corpus. This may "
                "mean the target is not small-molecule tractable, or simply that its "
                "chemistry has not been ingested yet."
            )
            if not known_compounds
            else None,
        },
    }


async def get_target_evidence(
    context: ServiceContext,
    identifier: str,
    *,
    disease_identifier: str | None = None,
    evidence_types: list[str] | None = None,
    sources: list[str] | None = None,
    min_score: float | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Filtered evidence retrieval -- the agent's evidence tool calls this."""
    target = await biology_repo.get_target(context.session, identifier)
    if target is None:
        raise NotFoundError(f"target {identifier!r} is not in the local corpus")

    disease = (
        await biology_repo.get_disease(context.session, disease_identifier)
        if disease_identifier
        else None
    )
    if disease_identifier and disease is None:
        raise NotFoundError(f"disease {disease_identifier!r} is not in the local corpus")

    evidence = await biology_repo.list_evidence(
        context.session,
        target_id=target.id,
        disease_id=disease.id if disease else None,
        evidence_types=evidence_types,
        sources=sources,
        min_score=min_score,
        limit=limit,
        offset=offset,
    )
    return {
        "target": {
            "id": target.id,
            "symbol": target.approved_symbol,
            "target_key": target.target_key,
        },
        "disease": (
            {"id": disease.id, "name": disease.name, "efo_id": disease.efo_id} if disease else None
        ),
        "filters": {
            "evidence_types": evidence_types,
            "sources": sources,
            "min_score": min_score,
            "limit": limit,
            "offset": offset,
        },
        "items": [_evidence_dict(item) for item in evidence],
        "count": len(evidence),
        "citation_keys": [item.citation_key for item in evidence],
    }
