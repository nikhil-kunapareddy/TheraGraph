"""Open Targets Platform connector (GraphQL API v4).

Open Targets is the source of truth for disease/target associations and the aggregated
evidence behind them. Association scores, evidence scores and tractability buckets are
copied verbatim: TheraGraph re-scores nothing, because a number shown next to
"Open Targets" must be the number Open Targets published.

Data licence: Open Targets Platform is available under CC0 1.0; individual evidence
records inherit the licence of their originating datasource.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from theragraph.core.clock import utcnow
from theragraph.core.errors import UpstreamError
from theragraph.core.hashing import hash_payload
from theragraph.core.http_client import SourceHttpClient
from theragraph.core.logging import get_logger
from theragraph.ingestion.base import IngestionMode, NormalizedBatch, ResponseRecorder

log = get_logger(__name__)

SOURCE_ID = "open_targets"

#: Datasource id -> internal evidence type. Open Targets exposes ~20 datasources; these
#: are the ones with a meaningful interpretation for target triage.
DATASOURCE_TO_EVIDENCE_TYPE: dict[str, str] = {
    "ot_genetics_portal": "genetic_association",
    "gwas_credible_sets": "genetic_association",
    "eva": "genetic_association",
    "eva_somatic": "somatic_mutation",
    "gene_burden": "genetic_association",
    "genomics_england": "genetic_association",
    "clingen": "genetic_association",
    "gene2phenotype": "genetic_association",
    "orphanet": "genetic_association",
    "uniprot_literature": "genetic_association",
    "uniprot_variants": "genetic_association",
    "chembl": "known_drug",
    "europepmc": "literature",
    "impc": "animal_model",
    "expression_atlas": "expression",
    "reactome": "pathway",
    "slapenrich": "pathway",
    "progeny": "pathway",
    "crispr": "functional_genomics",
    "crispr_screen": "functional_genomics",
    "cancer_gene_census": "somatic_mutation",
    "intogen": "somatic_mutation",
    "cancer_biomarkers": "biomarker",
    "sysbio": "expression",
}

DEFAULT_EVIDENCE_DATASOURCES = [
    "ot_genetics_portal",
    "gwas_credible_sets",
    "eva",
    "gene_burden",
    "chembl",
    "europepmc",
    "impc",
    "expression_atlas",
    "reactome",
]

# --------------------------------------------------------------------- queries
META_QUERY = """
query Meta {
  meta {
    apiVersion { x y z }
    dataVersion { year month iteration }
  }
}
"""

SEARCH_QUERY = """
query Search($queryString: String!, $entityNames: [String!], $size: Int!) {
  search(queryString: $queryString, entityNames: $entityNames, page: {index: 0, size: $size}) {
    total
    hits {
      id
      entity
      name
      description
      score
    }
  }
}
"""

DISEASE_QUERY = """
query DiseaseWithTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    description
    synonyms { relation terms }
    therapeuticAreas { id name }
    parents { id name }
    associatedTargets(page: {index: 0, size: $size}, orderByScore: "score") {
      count
      rows {
        score
        datatypeScores { id score }
        target {
          id
          approvedSymbol
          approvedName
          biotype
        }
      }
    }
  }
}
"""

TARGET_QUERY = """
query TargetDetail($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    synonyms { label source }
    proteinIds { id source }
    dbXrefs { id source }
    tractability { label modality value }
    subcellularLocations { location termSL }
    safetyLiabilities { event eventId datasource literature }
    targetClass { id label level }
    genomicLocation { chromosome start end strand }
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug {
          id
          name
          drugType
          maximumClinicalStage
          mechanismsOfAction { rows { mechanismOfAction actionType } }
        }
      }
    }
  }
}
"""

EVIDENCE_QUERY = """
query DiseaseEvidence(
  $efoId: String!
  $ensemblIds: [String!]!
  $datasourceIds: [String!]
  $size: Int!
  $cursor: String
) {
  disease(efoId: $efoId) {
    id
    name
    evidences(
      ensemblIds: $ensemblIds
      datasourceIds: $datasourceIds
      size: $size
      cursor: $cursor
    ) {
      count
      cursor
      rows {
        id
        score
        datasourceId
        datatypeId
        targetFromSourceId
        diseaseFromSource
        diseaseFromSourceMappedId
        literature
        publicationYear
        publicationFirstAuthor
        studyId
        variantRsId
        clinicalStage
        clinicalSignificances
        targetModulation
        drugFromSource
        resourceScore
        confidence
        studyOverview
        oddsRatio
        pValueMantissa
        pValueExponent
        beta
        target { id approvedSymbol }
        disease { id name }
        drug { id name drugType maximumClinicalStage }
        urls { niceName url }
        textMiningSentences { text section }
      }
    }
  }
}
"""


@dataclass
class OpenTargetsRelease:
    api_version: str
    data_version: str


class OpenTargetsConnector:
    """Typed wrapper over the Open Targets GraphQL API."""

    source_id = SOURCE_ID

    def __init__(
        self,
        client: SourceHttpClient,
        *,
        graphql_url: str,
        recorder: ResponseRecorder | None = None,
    ) -> None:
        self._client = client
        self._url = graphql_url
        self._recorder = recorder
        self._release: OpenTargetsRelease | None = None

    async def _graphql(
        self, operation: str, query: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        async def loader() -> Any:
            payload = await self._client.post_json(
                self._url, json={"query": query, "variables": variables}
            )
            if isinstance(payload, dict) and payload.get("errors"):
                raise UpstreamError(
                    "Open Targets GraphQL returned errors",
                    source=SOURCE_ID,
                    details={
                        "operation": operation,
                        "errors": [error.get("message") for error in payload["errors"][:3]],
                    },
                )
            return payload

        if self._recorder is not None:
            payload = await self._recorder.fetch(
                source=SOURCE_ID, operation=operation, params=variables, loader=loader
            )
        else:
            payload = await loader()

        data = (payload or {}).get("data")
        if data is None:
            raise UpstreamError(
                "Open Targets response contained no data",
                source=SOURCE_ID,
                details={"operation": operation},
            )
        return data  # type: ignore[no-any-return]

    # -------------------------------------------------------------- release info
    async def release(self) -> OpenTargetsRelease:
        """Fetch and cache the upstream release, used as the dataset version label."""
        if self._release is not None:
            return self._release
        data = await self._graphql("meta", META_QUERY, {})
        meta = data.get("meta", {})
        api = meta.get("apiVersion", {}) or {}
        version = meta.get("dataVersion", {}) or {}
        iteration = version.get("iteration")
        data_version = f"{version.get('year', '??')}.{version.get('month', '??')}"
        if iteration:
            data_version = f"{data_version}.{iteration}"
        self._release = OpenTargetsRelease(
            api_version=f"{api.get('x', '?')}.{api.get('y', '?')}.{api.get('z', '?')}",
            data_version=data_version,
        )
        return self._release

    # -------------------------------------------------------------------- search
    async def search(
        self, query_string: str, *, entities: list[str] | None = None, size: int = 10
    ) -> list[dict[str, Any]]:
        data = await self._graphql(
            "search",
            SEARCH_QUERY,
            {
                "queryString": query_string,
                "entityNames": entities or ["disease", "target"],
                "size": size,
            },
        )
        hits = ((data.get("search") or {}).get("hits")) or []
        return [
            {
                "id": hit.get("id"),
                "entity": hit.get("entity"),
                "name": hit.get("name"),
                "description": hit.get("description"),
                "score": hit.get("score"),
            }
            for hit in hits
            if hit.get("id")
        ]

    # ------------------------------------------------------------------- disease
    async def fetch_disease(self, efo_id: str, *, associated_targets: int = 25) -> dict[str, Any]:
        data = await self._graphql(
            "disease", DISEASE_QUERY, {"efoId": efo_id, "size": associated_targets}
        )
        disease = data.get("disease")
        if not disease:
            raise UpstreamError(
                f"Open Targets has no disease {efo_id!r}",
                source=SOURCE_ID,
                details={"efo_id": efo_id},
            )
        return disease  # type: ignore[no-any-return]

    async def fetch_target(self, ensembl_id: str) -> dict[str, Any]:
        data = await self._graphql("target", TARGET_QUERY, {"ensemblId": ensembl_id})
        target = data.get("target")
        if not target:
            raise UpstreamError(
                f"Open Targets has no target {ensembl_id!r}",
                source=SOURCE_ID,
                details={"ensembl_id": ensembl_id},
            )
        return target  # type: ignore[no-any-return]

    async def fetch_evidence(
        self,
        efo_id: str,
        *,
        ensembl_ids: list[str],
        datasource_ids: list[str] | None = None,
        size: int = 50,
        max_pages: int = 3,
    ) -> list[dict[str, Any]]:
        """Paginate evidence rows for a disease, scoped to specific targets.

        Open Targets requires ``ensemblIds`` on ``disease.evidences``, which suits the
        product: evidence is always retrieved for a concrete target-disease pair so it
        can be cited precisely, rather than pulled in bulk and filtered client-side.
        """
        if not ensembl_ids:
            raise UpstreamError(
                "evidence retrieval requires at least one Ensembl target id",
                source=SOURCE_ID,
            )
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        for page in range(max_pages):
            variables: dict[str, Any] = {
                "efoId": efo_id,
                "ensemblIds": ensembl_ids,
                "datasourceIds": datasource_ids or DEFAULT_EVIDENCE_DATASOURCES,
                "size": size,
                "cursor": cursor,
            }
            data = await self._graphql(f"evidence.page{page}", EVIDENCE_QUERY, variables)
            evidences = ((data.get("disease") or {}).get("evidences")) or {}
            page_rows = evidences.get("rows") or []
            rows.extend(page_rows)
            cursor = evidences.get("cursor")
            if not cursor or not page_rows:
                break
        return rows


# --------------------------------------------------------------- normalisation
def normalize_disease(
    payload: dict[str, Any], *, retrieved_at: datetime, dataset_version_id: str | None
) -> dict[str, Any]:
    """Map an Open Targets disease onto the internal ``disease`` row shape."""
    synonym_terms: list[str] = []
    for group in payload.get("synonyms") or []:
        if group.get("relation") in ("hasExactSynonym", "hasNarrowSynonym"):
            synonym_terms.extend(group.get("terms") or [])

    return {
        "efo_id": payload["id"],
        "name": payload.get("name") or payload["id"],
        "description": payload.get("description"),
        "synonyms": sorted({term for term in synonym_terms if term})[:50],
        "therapeutic_areas": [
            {"id": area.get("id"), "name": area.get("name")}
            for area in payload.get("therapeuticAreas") or []
        ],
        "parent_ids": [
            parent.get("id") for parent in payload.get("parents") or [] if parent.get("id")
        ],
        "source": SOURCE_ID,
        "source_identifier": payload["id"],
        "source_url": f"https://platform.opentargets.org/disease/{payload['id']}",
        "retrieved_at": retrieved_at,
        "dataset_version_id": dataset_version_id,
    }


def normalize_target(
    payload: dict[str, Any], *, retrieved_at: datetime, dataset_version_id: str | None
) -> dict[str, Any]:
    """Map an Open Targets target onto the internal ``target`` row shape.

    Tractability is collapsed from a flat list of ``(modality, label, value)`` triples
    into ``{modality: highest_achieved_bucket}``, keeping only buckets the source marked
    as true. The raw list stays available in ``tractability['_raw']`` so nothing is lost.
    """
    tractability_raw = payload.get("tractability") or []
    by_modality: dict[str, list[str]] = {}
    for entry in tractability_raw:
        if not entry.get("value"):
            continue
        modality = str(entry.get("modality") or "unknown")
        by_modality.setdefault(modality, []).append(str(entry.get("label")))

    tractability: dict[str, Any] = {
        modality: labels[0] for modality, labels in by_modality.items() if labels
    }
    tractability["_all_buckets"] = by_modality
    tractability["_source"] = SOURCE_ID

    protein_ids = payload.get("proteinIds") or []
    uniprot_ids = [
        entry["id"]
        for entry in protein_ids
        if entry.get("id") and str(entry.get("source", "")).startswith("uniprot")
    ]
    chembl_ids = [
        entry["id"]
        for entry in (payload.get("dbXrefs") or [])
        if entry.get("id") and entry.get("source") == "ChEMBL"
    ]

    function_descriptions = payload.get("functionDescriptions") or []

    return {
        "target_key": payload["id"],
        "approved_symbol": payload.get("approvedSymbol") or payload["id"],
        "approved_name": payload.get("approvedName"),
        "biotype": payload.get("biotype"),
        "target_class": sorted(
            {
                str(entry.get("label"))
                for entry in payload.get("targetClass") or []
                if entry.get("label")
            }
        ),
        "synonyms": sorted(
            {
                str(entry.get("label"))
                for entry in payload.get("synonyms") or []
                if entry.get("label")
            }
        )[:50],
        "uniprot_ids": uniprot_ids,
        "chembl_target_ids": chembl_ids,
        "subcellular_locations": sorted(
            {
                str(entry.get("location"))
                for entry in payload.get("subcellularLocations") or []
                if entry.get("location")
            }
        )[:25],
        "tractability": tractability,
        "safety_liabilities": [
            {
                "event": entry.get("event"),
                "event_id": entry.get("eventId"),
                "datasource": entry.get("datasource"),
                "literature": entry.get("literature"),
            }
            for entry in payload.get("safetyLiabilities") or []
        ][:50],
        "description": function_descriptions[0] if function_descriptions else None,
        "source": SOURCE_ID,
        "source_identifier": payload["id"],
        "source_url": f"https://platform.opentargets.org/target/{payload['id']}",
        "retrieved_at": retrieved_at,
        "dataset_version_id": dataset_version_id,
    }


def normalize_gene(
    payload: dict[str, Any], *, retrieved_at: datetime, dataset_version_id: str | None
) -> dict[str, Any]:
    location = payload.get("genomicLocation") or {}
    protein_ids = payload.get("proteinIds") or []
    return {
        "ensembl_id": payload["id"],
        "symbol": payload.get("approvedSymbol") or payload["id"],
        "name": payload.get("approvedName"),
        "biotype": payload.get("biotype"),
        "chromosome": str(location.get("chromosome")) if location.get("chromosome") else None,
        "genomic_start": location.get("start"),
        "genomic_end": location.get("end"),
        "uniprot_ids": [
            entry["id"]
            for entry in protein_ids
            if entry.get("id") and str(entry.get("source", "")).startswith("uniprot")
        ],
        "source": SOURCE_ID,
        "source_identifier": payload["id"],
        "source_url": f"https://platform.opentargets.org/target/{payload['id']}",
        "retrieved_at": retrieved_at,
        "dataset_version_id": dataset_version_id,
    }


def normalize_associations(
    disease_payload: dict[str, Any],
    *,
    disease_id: str,
    target_id_by_key: dict[str, str],
    retrieved_at: datetime,
    dataset_version_id: str | None,
) -> NormalizedBatch:
    """Build association rows, skipping targets that were not resolved locally."""
    batch = NormalizedBatch(entity_type="target_disease_association")
    associated = (disease_payload.get("associatedTargets") or {}).get("rows") or []
    for row in associated:
        target = row.get("target") or {}
        target_key = str(target.get("id") or "")
        internal_target_id = target_id_by_key.get(target_key)
        if internal_target_id is None:
            batch.rejects.append(
                {"target_key": target_key, "_quality_errors": ["unresolved_target"]}
            )
            continue
        datatype_scores = {
            str(entry.get("id")): entry.get("score")
            for entry in row.get("datatypeScores") or []
            if entry.get("id") is not None
        }
        batch.rows.append(
            {
                "target_id": internal_target_id,
                "disease_id": disease_id,
                "overall_score": float(row.get("score") or 0.0),
                "datatype_scores": datatype_scores,
                "evidence_count": 0,
                "source": SOURCE_ID,
                "source_identifier": f"{target_key}-{disease_payload['id']}",
                "source_url": (
                    "https://platform.opentargets.org/evidence/"
                    f"{target_key}/{disease_payload['id']}"
                ),
                "retrieved_at": retrieved_at,
                "dataset_version_id": dataset_version_id,
            }
        )
    return batch


def normalize_evidence(
    rows: list[dict[str, Any]],
    *,
    disease_id: str,
    target_id_by_key: dict[str, str],
    retrieved_at: datetime,
    dataset_version_id: str | None,
) -> NormalizedBatch:
    """Map Open Targets evidence rows into internal ``evidence`` rows.

    Each row keeps its datasource, source identifier, score and a URL back to the
    upstream record. ``summary`` is assembled only from fields the source provided --
    no interpretation, no generated prose.
    """
    batch = NormalizedBatch(entity_type="evidence")
    for row in rows:
        datasource = str(row.get("datasourceId") or "unknown")
        evidence_type = DATASOURCE_TO_EVIDENCE_TYPE.get(datasource, "other")
        target_key = (row.get("target") or {}).get("id") or row.get("targetFromSourceId")
        internal_target_id = target_id_by_key.get(str(target_key))
        if internal_target_id is None:
            batch.rejects.append(
                {
                    "source_identifier": row.get("id"),
                    "_quality_errors": ["unresolved_target"],
                }
            )
            continue

        literature = row.get("literature") or []
        pmid = str(literature[0]) if literature else None
        urls = [entry.get("url") for entry in row.get("urls") or [] if entry.get("url")]

        summary_parts: list[str] = []
        if row.get("diseaseFromSource"):
            summary_parts.append(f"Reported disease: {row['diseaseFromSource']}")
        if row.get("targetModulation"):
            summary_parts.append(f"Target modulation: {row['targetModulation']}")
        if row.get("clinicalStage") is not None:
            summary_parts.append(f"Clinical stage: {row['clinicalStage']}")
        if row.get("clinicalSignificances"):
            summary_parts.append(
                "Clinical significance: " + ", ".join(row["clinicalSignificances"])
            )
        if row.get("studyOverview"):
            summary_parts.append(str(row["studyOverview"])[:400])
        if row.get("oddsRatio") is not None:
            summary_parts.append(f"Odds ratio: {row['oddsRatio']}")
        if row.get("pValueMantissa") is not None and row.get("pValueExponent") is not None:
            summary_parts.append(f"p-value: {row['pValueMantissa']}e{row['pValueExponent']}")
        for sentence in (row.get("textMiningSentences") or [])[:2]:
            text = str(sentence.get("text") or "").strip()
            if text:
                section = sentence.get("section") or "text"
                summary_parts.append(f"[{section}] {text[:300]}")
        if row.get("drugFromSource") or (row.get("drug") or {}).get("name"):
            drug_name = row.get("drugFromSource") or (row.get("drug") or {})["name"]
            summary_parts.append(f"Drug: {drug_name}")

        source_identifier = str(
            row.get("id") or hash_payload({"datasource": datasource, "row": row}, length=24)
        )

        batch.rows.append(
            {
                "evidence_type": evidence_type,
                "relation": "disease_target",
                "target_id": internal_target_id,
                "disease_id": disease_id,
                "compound_id": None,
                "publication_id": None,
                "score": float(row["score"]) if row.get("score") is not None else None,
                "score_type": "open_targets_evidence_score"
                if row.get("score") is not None
                else None,
                "title": _evidence_title(datasource, row),
                "summary": " | ".join(summary_parts) or None,
                "extra": {
                    "datasource_id": datasource,
                    "datatype_id": row.get("datatypeId"),
                    "resource_score": row.get("resourceScore"),
                    "confidence": row.get("confidence"),
                    "pmid": pmid,
                    "publication_year": row.get("publicationYear"),
                    "first_author": row.get("publicationFirstAuthor"),
                    "study_id": row.get("studyId"),
                    "variant_rs_id": row.get("variantRsId"),
                    "drug_chembl_id": (row.get("drug") or {}).get("id"),
                    "drug_name": row.get("drugFromSource") or (row.get("drug") or {}).get("name"),
                    "max_clinical_stage": (row.get("drug") or {}).get("maximumClinicalStage"),
                    "urls": urls[:5],
                    "beta": row.get("beta"),
                },
                "source": SOURCE_ID,
                "source_identifier": source_identifier,
                "source_url": urls[0]
                if urls
                else (
                    f"https://europepmc.org/article/MED/{pmid}"
                    if pmid
                    else f"https://platform.opentargets.org/evidence/{target_key}/{disease_id}"
                ),
                "retrieved_at": retrieved_at,
                "dataset_version_id": dataset_version_id,
            }
        )
    return batch


def _evidence_title(datasource: str, row: dict[str, Any]) -> str:
    symbol = (row.get("target") or {}).get("approvedSymbol") or row.get(
        "targetFromSourceId", "target"
    )
    disease_name = (row.get("disease") or {}).get("name") or "disease"
    label = DATASOURCE_TO_EVIDENCE_TYPE.get(datasource, datasource).replace("_", " ")
    return f"{label.capitalize()} evidence linking {symbol} to {disease_name} ({datasource})"


def normalize_known_drugs(
    target_payload: dict[str, Any],
    *,
    target_id: str,
    retrieved_at: datetime,
    dataset_version_id: str | None,
) -> NormalizedBatch:
    """Known drugs / clinical candidates reported against a target.

    Stored as ``known_drug`` evidence on the target-compound relation. The compound
    link is resolved later by the pipeline once the ChEMBL molecule has been ingested.
    """
    batch = NormalizedBatch(entity_type="known_drug")
    candidates = (target_payload.get("drugAndClinicalCandidates") or {}).get("rows") or []
    for row in candidates:
        drug = row.get("drug") or {}
        chembl_id = drug.get("id")
        if not chembl_id:
            continue
        batch.rows.append(
            {
                "chembl_id": chembl_id,
                "name": drug.get("name"),
                "drug_type": drug.get("drugType"),
                "max_clinical_stage": drug.get("maximumClinicalStage")
                or row.get("maxClinicalStage"),
                "mechanisms_of_action": sorted(
                    {
                        str(mechanism.get("mechanismOfAction"))
                        for mechanism in ((drug.get("mechanismsOfAction") or {}).get("rows") or [])
                        if mechanism.get("mechanismOfAction")
                    }
                ),
                "target_id": target_id,
                "source": SOURCE_ID,
                "source_identifier": f"{target_payload['id']}-{chembl_id}",
                "source_url": f"https://platform.opentargets.org/drug/{chembl_id}",
                "retrieved_at": retrieved_at,
                "dataset_version_id": dataset_version_id,
            }
        )
    return batch


def build_connector(
    *,
    graphql_url: str,
    timeout_s: float,
    max_retries: int,
    user_agent: str,
    recorder: ResponseRecorder | None = None,
) -> OpenTargetsConnector:
    client = SourceHttpClient(
        SOURCE_ID,
        timeout_s=timeout_s,
        max_retries=max_retries,
        user_agent=user_agent,
        headers={"Content-Type": "application/json"},
    )
    return OpenTargetsConnector(client, graphql_url=graphql_url, recorder=recorder)


DATA_SOURCE_DEFINITION = {
    "id": SOURCE_ID,
    "name": "Open Targets Platform",
    "description": (
        "Integrated evidence for target-disease associations, aggregating genetics, "
        "known drugs, literature, animal models, expression and pathway data."
    ),
    "homepage_url": "https://platform.opentargets.org",
    "api_base_url": "https://api.platform.opentargets.org/api/v4/graphql",
    "license": "CC0 1.0 (platform); individual evidence inherits its datasource licence",
    "citation": (
        "Ochoa D, et al. The next-generation Open Targets Platform: reimagined, "
        "redesigned, rebuilt. Nucleic Acids Research (2023)."
    ),
    "entity_types": ["disease", "target", "association", "evidence", "known_drug"],
}


async def probe(connector: OpenTargetsConnector) -> dict[str, Any]:
    """Cheap connectivity/schema check used by the readiness endpoint and CI."""
    release = await connector.release()
    return {
        "source": SOURCE_ID,
        "api_version": release.api_version,
        "data_version": release.data_version,
        "checked_at": utcnow().isoformat(),
    }


__all__ = [
    "DATASOURCE_TO_EVIDENCE_TYPE",
    "DATA_SOURCE_DEFINITION",
    "DEFAULT_EVIDENCE_DATASOURCES",
    "SOURCE_ID",
    "IngestionMode",
    "OpenTargetsConnector",
    "OpenTargetsRelease",
    "build_connector",
    "normalize_associations",
    "normalize_disease",
    "normalize_evidence",
    "normalize_gene",
    "normalize_known_drugs",
    "normalize_target",
    "probe",
]
