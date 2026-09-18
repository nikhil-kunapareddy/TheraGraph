"""Search and discovery across diseases, targets and compounds.

Search is local-first: the ingested corpus is queried before any upstream call, so the
common case is fast and offline-capable. When a local query comes up empty (or the
caller explicitly asks), Open Targets' own search is consulted and the result is
returned as a *discoverable* entity the user can choose to ingest -- clearly
distinguished from entities already in the local corpus with full provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from theragraph.core.cache import cache_key
from theragraph.core.errors import NotFoundError
from theragraph.core.logging import get_logger
from theragraph.db.repositories import biology as biology_repo
from theragraph.db.repositories import chemistry as chem_repo
from theragraph.db.repositories import provenance as prov_repo
from theragraph.db.repositories.base import normalize_text
from theragraph.db.repositories.biology import _PREFIX
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

EntityType = Literal["disease", "target", "compound"]


@dataclass
class SearchHit:
    entity_type: EntityType
    id: str
    external_id: str
    name: str
    relevance: float
    description: str | None = None
    source: str = "local"
    availability: Literal["ingested", "discoverable"] = "ingested"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "id": self.id,
            "external_id": self.external_id,
            "name": self.name,
            "relevance": round(self.relevance, 3),
            "description": self.description,
            "source": self.source,
            "availability": self.availability,
            "metadata": self.metadata,
        }


#: Tokens too generic to identify an entity on their own. "disease" would otherwise
#: match *Alzheimer disease* for a question about Kawasaki disease -- the exact failure
#: mode that lets an agent answer confidently about the wrong entity.
GENERIC_TOKENS = frozenset(
    {
        "disease",
        "diseases",
        "disorder",
        "disorders",
        "syndrome",
        "syndromes",
        "condition",
        "conditions",
        "deficiency",
        "type",
        "human",
        "protein",
        "gene",
        "receptor",
        "kinase",
        "inhibitor",
        "inhibitors",
        "agonist",
        "antagonist",
        "acid",
        "cell",
        "cells",
        "factor",
        "chronic",
        "acute",
        "primary",
        "secondary",
        "familial",
        "idiopathic",
        "congenital",
        "malignant",
        "benign",
        "and",
        "with",
    }
)


def _search_tiers(query: str, *, max_terms: int = 10) -> tuple[str, list[str]]:
    """Split a query into ``(exact_phrase, fallback_terms)``.

    Corpus entity names are short; user questions are not. The exact phrase is tried
    first because it is the most specific. If it matches nothing, every contiguous
    bigram and every token of four or more characters is tried and the results merged --
    which is what makes "BACE1 have genetic Alzheimer disease" resolve to *both* BACE1
    and Alzheimer disease, rather than to nothing.

    Each fallback term is one indexed LIKE query, so the fan-out is cheap and bounded.
    """
    normalized = normalize_text(query)
    tokens = [token for token in normalized.split() if token]

    fallbacks: list[str] = []
    for index in range(len(tokens) - 1):
        bigram = f"{tokens[index]} {tokens[index + 1]}"
        if bigram != normalized:
            fallbacks.append(bigram)
    for token in tokens:
        if (
            len(token) >= 4
            and token not in GENERIC_TOKENS
            and token not in fallbacks
            and token != normalized
        ):
            fallbacks.append(token)

    return normalized, fallbacks[:max_terms]


async def search(
    context: ServiceContext,
    query: str,
    *,
    entity_types: list[EntityType] | None = None,
    limit: int = 10,
    include_upstream: bool = False,
) -> dict[str, Any]:
    """Unified search entry point used by the dashboard and the agent.

    Runs progressively narrower variants of the query until something matches, so a
    conversational phrase still resolves to the entity it names. The variant that
    produced each hit is reported, so a surprising result is explainable.
    """
    types = entity_types or ["disease", "target", "compound"]
    exact, fallbacks = _search_tiers(query)

    hits = await _search_once(context, exact, types=types, limit=limit)
    terms_tried = [exact]
    matched_term: str | None = exact if hits else None

    if not hits and fallbacks:
        merged: dict[tuple[str, str], SearchHit] = {}
        for term in fallbacks:
            terms_tried.append(term)
            # A single-token fallback must match a name exactly or as a prefix. Allowing
            # substring matches here is how "kawasaki" style queries end up returning
            # an unrelated disease that merely shares a word.
            min_relevance = _PREFIX if " " not in term else 0.0
            for hit in await _search_once(
                context, term, types=types, limit=limit, min_relevance=min_relevance
            ):
                key = (hit.entity_type, hit.id)
                existing = merged.get(key)
                # A hit found by several terms keeps its best relevance, so an entity
                # matched by both a bigram and a token is not penalised.
                if existing is None or hit.relevance > existing.relevance:
                    merged[key] = hit
                    if matched_term is None:
                        matched_term = term
        hits = list(merged.values())

    hits.sort(key=lambda hit: (-hit.relevance, len(hit.name), hit.name))
    result: dict[str, Any] = {
        "query": query,
        "matched_term": matched_term,
        "terms_tried": terms_tried,
        "hits": [hit.to_dict() for hit in hits[: limit * len(types)]],
        "counts": {
            entity_type: sum(1 for hit in hits if hit.entity_type == entity_type)
            for entity_type in types
        },
        "searched": "local_corpus",
    }

    if include_upstream and not hits:
        result["upstream"] = await _search_upstream(context, exact, limit=limit)
        result["searched"] = "local_corpus+open_targets"

    return result


async def _search_once(
    context: ServiceContext,
    query: str,
    *,
    types: list[EntityType],
    limit: int,
    min_relevance: float = 0.0,
) -> list[SearchHit]:
    """One pass of the local search for a single term."""
    hits: list[SearchHit] = []

    if "disease" in types:
        for disease, relevance in await biology_repo.search_diseases(
            context.session, query, limit=limit
        ):
            hits.append(
                SearchHit(
                    entity_type="disease",
                    id=disease.id,
                    external_id=disease.efo_id,
                    name=disease.name,
                    relevance=relevance,
                    description=(disease.description or "")[:400] or None,
                    source=disease.source,
                    metadata={
                        "therapeutic_areas": [
                            area.get("name") for area in disease.therapeutic_areas or []
                        ],
                        "synonyms": (disease.synonyms or [])[:5],
                        "retrieved_at": disease.retrieved_at.isoformat(),
                    },
                )
            )

    if "target" in types:
        for target, relevance in await biology_repo.search_targets(
            context.session, query, limit=limit
        ):
            hits.append(
                SearchHit(
                    entity_type="target",
                    id=target.id,
                    external_id=target.target_key,
                    name=f"{target.approved_symbol} - {target.approved_name or ''}".strip(" -"),
                    relevance=relevance,
                    description=(target.description or "")[:400] or None,
                    source=target.source,
                    metadata={
                        "symbol": target.approved_symbol,
                        "biotype": target.biotype,
                        "target_class": target.target_class or [],
                        "tractability": {
                            key: value
                            for key, value in (target.tractability or {}).items()
                            if not key.startswith("_")
                        },
                        "retrieved_at": target.retrieved_at.isoformat(),
                    },
                )
            )

    if "compound" in types:
        for compound, relevance in await chem_repo.search_compounds(
            context.session, query, limit=limit
        ):
            hits.append(
                SearchHit(
                    entity_type="compound",
                    id=compound.id,
                    external_id=compound.chembl_id or compound.inchikey,
                    name=compound.display_name,
                    relevance=relevance,
                    source=compound.source,
                    metadata={
                        "smiles": compound.canonical_smiles,
                        "max_phase": compound.max_phase,
                        "inchikey": compound.inchikey,
                        "is_approved_drug": compound.is_approved_drug,
                        "retrieved_at": compound.retrieved_at.isoformat(),
                    },
                )
            )

    return hits


async def _search_upstream(context: ServiceContext, query: str, *, limit: int) -> dict[str, Any]:
    """Ask Open Targets what exists, without ingesting anything.

    Returned entities are marked ``discoverable``: the UI offers an explicit "ingest"
    action rather than silently pulling data, so the user always knows when a network
    call and a new dataset version happened.
    """
    from theragraph.ingestion.base import IngestionMode
    from theragraph.ingestion.pipelines import build_connectors

    if not context.settings.allow_live_ingestion:
        return {"available": False, "reason": "live ingestion is disabled"}

    key = cache_key("upstream_search", query, limit)
    if context.cache is not None:
        cached = await context.cache.get_json(key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

    connectors = build_connectors(context.settings, mode=IngestionMode.LIVE)
    try:
        raw = await connectors.open_targets.search(query, size=limit)
        release = await connectors.open_targets.release()
    except Exception as exc:
        log.warning("discovery.upstream_search_failed", error=str(exc))
        return {"available": False, "reason": str(exc)[:200]}
    finally:
        await connectors.aclose()

    payload = {
        "available": True,
        "source": "open_targets",
        "data_version": release.data_version,
        "hits": [
            {
                "entity_type": hit["entity"],
                "external_id": hit["id"],
                "name": hit["name"],
                "description": (hit.get("description") or "")[:300] or None,
                "availability": "discoverable",
                "ingest_hint": (
                    f"POST /api/v1/ingestion/disease with efo_id={hit['id']}"
                    if hit["entity"] == "disease"
                    else None
                ),
            }
            for hit in raw
        ],
    }
    if context.cache is not None:
        await context.cache.set_json(key, payload, ttl_s=900)
    return payload


async def get_disease_overview(
    context: ServiceContext, identifier: str, *, n_targets: int = 25, min_score: float = 0.0
) -> dict[str, Any]:
    """Disease detail plus its associated targets, ordered by the source's score."""
    disease = await biology_repo.get_disease(context.session, identifier)
    if disease is None:
        raise NotFoundError(
            f"disease {identifier!r} is not in the local corpus",
            details={
                "identifier": identifier,
                "hint": "search with include_upstream=true, then ingest it",
            },
        )

    associations = await biology_repo.targets_for_disease(
        context.session, disease.id, min_score=min_score, limit=n_targets
    )
    evidence_counts = await biology_repo.evidence_type_counts(
        context.session, disease_id=disease.id
    )
    dataset_version = (
        await prov_repo.get_dataset_versions_by_ids(context.session, [disease.dataset_version_id])
        if disease.dataset_version_id
        else []
    )

    return {
        "disease": {
            "id": disease.id,
            "efo_id": disease.efo_id,
            "name": disease.name,
            "description": disease.description,
            "synonyms": disease.synonyms or [],
            "therapeutic_areas": disease.therapeutic_areas or [],
            "parent_ids": disease.parent_ids or [],
            "provenance": {
                "source": disease.source,
                "source_identifier": disease.source_identifier,
                "source_url": disease.source_url,
                "retrieved_at": disease.retrieved_at.isoformat(),
                "dataset_version": dataset_version[0].label if dataset_version else None,
            },
        },
        "associated_targets": [
            {
                "target": {
                    "id": target.id,
                    "target_key": target.target_key,
                    "symbol": target.approved_symbol,
                    "name": target.approved_name,
                    "biotype": target.biotype,
                    "target_class": target.target_class or [],
                    "tractability": {
                        key: value
                        for key, value in (target.tractability or {}).items()
                        if not key.startswith("_")
                    },
                },
                "association": {
                    "overall_score": round(association.overall_score, 5),
                    "datatype_scores": association.datatype_scores or {},
                    "source": association.source,
                    "source_url": association.source_url,
                    "retrieved_at": association.retrieved_at.isoformat(),
                    "score_meaning": (
                        "Open Targets association score (0-1), retrieved verbatim; "
                        "aggregates evidence across genetics, drugs, literature, "
                        "animal models, expression and pathways."
                    ),
                },
            }
            for target, association in associations
        ],
        "evidence_summary": {
            "by_type": evidence_counts,
            "total": sum(evidence_counts.values()),
        },
        "counts": {"associated_targets": len(associations)},
    }


async def corpus_summary(context: ServiceContext) -> dict[str, Any]:
    """What is actually in the local corpus -- shown on the dashboard."""
    sources = await prov_repo.list_data_sources(context.session)
    current_versions = await prov_repo.list_dataset_versions(
        context.session, current_only=True, limit=50
    )
    return {
        "counts": {
            "diseases": await biology_repo.count_diseases(context.session),
            "targets": await biology_repo.count_targets(context.session),
            "compounds": await chem_repo.count_compounds(context.session),
            "bioactivities": await chem_repo.count_bioactivities(context.session),
        },
        "sources": [
            {
                "id": source.id,
                "name": source.name,
                "license": source.license,
                "citation": source.citation,
                "homepage_url": source.homepage_url,
                "entity_types": source.entity_types or [],
            }
            for source in sources
        ],
        "dataset_versions": [
            {
                "label": version.label,
                "source": version.source_id,
                "dataset": version.dataset_name,
                "version": version.version,
                "record_count": version.record_count,
                "content_hash": version.content_hash,
                "retrieved_at": version.retrieved_at.isoformat(),
                "quality_score": (version.quality or {}).get("score"),
            }
            for version in current_versions
        ],
    }
