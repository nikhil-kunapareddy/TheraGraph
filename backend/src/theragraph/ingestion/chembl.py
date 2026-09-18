"""ChEMBL connector (EBI REST API).

ChEMBL supplies the chemistry: molecule structures, curated bioactivity measurements,
assay metadata and clinical phase. Three ChEMBL-specific realities shape this module:

* **Activity data needs curation before use.** Rows carrying a
  ``data_validity_comment`` (e.g. "Outside typical range") or flagged as
  ``potential_duplicate`` are kept in the database for transparency but excluded from
  model training and from "known actives" listings.
* **pChEMBL is the comparable scale.** Raw ``standard_value`` mixes nM/uM/%; pChEMBL
  (-log10 molar) is what can be averaged or thresholded, so it drives activity views.
* **Pagination is cursor-ish.** Responses carry ``page_meta.next``; the client follows
  it with a hard page cap so one ingestion cannot walk 24M activities.

Data licence: ChEMBL is released under CC BY-SA 3.0.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from theragraph.chem.molecule import safe_standardize
from theragraph.core.errors import UpstreamError
from theragraph.core.http_client import SourceHttpClient
from theragraph.core.logging import get_logger
from theragraph.ingestion.base import NormalizedBatch, ResponseRecorder
from theragraph.ingestion.quality import (
    QualityRule,
    evaluate_quality,
    in_range,
    required,
)

log = get_logger(__name__)

SOURCE_ID = "chembl"

#: Activity types that map onto a comparable potency scale.
POTENCY_TYPES = ("IC50", "Ki", "Kd", "EC50", "AC50", "Potency", "XC50")

#: ChEMBL assay confidence >= 8 means the assay maps to a single molecular target;
#: lower values are cell/organism level and are not equivalent evidence for a target.
MIN_ASSAY_CONFIDENCE = 8

MOLECULE_FIELDS = ",".join(
    [
        "molecule_chembl_id",
        "pref_name",
        "molecule_structures",
        "molecule_properties",
        "molecule_synonyms",
        "max_phase",
        "molecule_type",
        "first_approval",
        "oral",
        "parenteral",
        "topical",
        "withdrawn_flag",
        "structure_type",
        "prodrug",
        "cross_references",
    ]
)

ACTIVITY_FIELDS = ",".join(
    [
        "activity_id",
        "assay_chembl_id",
        "assay_description",
        "assay_type",
        "molecule_chembl_id",
        "canonical_smiles",
        "target_chembl_id",
        "target_organism",
        "target_pref_name",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "activity_comment",
        "data_validity_comment",
        "potential_duplicate",
        "document_chembl_id",
        "bao_format",
        "bao_label",
        "assay_variant_accession",
    ]
)


class ChEMBLConnector:
    """Typed wrapper over the ChEMBL REST API."""

    source_id = SOURCE_ID

    def __init__(
        self,
        client: SourceHttpClient,
        *,
        base_url: str,
        recorder: ResponseRecorder | None = None,
        max_pages: int = 20,
    ) -> None:
        self._client = client
        self._base = base_url.rstrip("/")
        self._recorder = recorder
        self._max_pages = max_pages
        self._version: str | None = None

    async def _get(
        self, path: str, params: dict[str, Any] | None = None, *, operation: str | None = None
    ) -> dict[str, Any]:
        url = f"{self._base}/{path.lstrip('/')}"
        params = params or {}

        async def loader() -> Any:
            return await self._client.get_json(url, params=params)

        op = operation or path.strip("/").replace("/", "_")
        if self._recorder is not None:
            payload = await self._recorder.fetch(
                source=SOURCE_ID, operation=op, params={"path": path, **params}, loader=loader
            )
        else:
            payload = await loader()
        if not isinstance(payload, dict):
            raise UpstreamError(
                "ChEMBL returned an unexpected payload type",
                source=SOURCE_ID,
                details={"path": path, "type": type(payload).__name__},
            )
        return payload

    async def _paginate(
        self,
        path: str,
        *,
        collection: str,
        params: dict[str, Any],
        limit: int,
        operation: str | None = None,
    ) -> list[dict[str, Any]]:
        """Collect up to ``limit`` records, following ``page_meta.next``."""
        page_size = min(limit, 1000)
        records: list[dict[str, Any]] = []
        offset = 0
        for page in range(self._max_pages):
            payload = await self._get(
                path,
                {**params, "limit": page_size, "offset": offset, "format": "json"},
                operation=f"{operation or collection}.page{page}",
            )
            batch = payload.get(collection) or []
            records.extend(batch)
            meta = payload.get("page_meta") or {}
            if len(records) >= limit or not meta.get("next") or not batch:
                break
            offset += page_size
        return records[:limit]

    # --------------------------------------------------------------- release info
    async def version(self) -> str:
        """ChEMBL release label (e.g. ``ChEMBL_37``) used as the dataset version."""
        if self._version is not None:
            return self._version
        payload = await self._get("status.json", operation="status")
        self._version = str(payload.get("chembl_db_version") or "unknown")
        return self._version

    async def status(self) -> dict[str, Any]:
        return await self._get("status.json", operation="status")

    # ----------------------------------------------------------------- molecules
    async def fetch_molecule(self, chembl_id: str) -> dict[str, Any]:
        payload = await self._get(
            f"molecule/{chembl_id.upper()}.json", operation=f"molecule.{chembl_id.upper()}"
        )
        if not payload.get("molecule_chembl_id"):
            raise UpstreamError(
                f"ChEMBL has no molecule {chembl_id!r}",
                source=SOURCE_ID,
                details={"chembl_id": chembl_id},
            )
        return payload

    async def fetch_molecules(self, chembl_ids: list[str]) -> list[dict[str, Any]]:
        """Batch molecule fetch using ChEMBL's ``__in`` filter (one request per 50)."""
        results: list[dict[str, Any]] = []
        unique = sorted({cid.upper() for cid in chembl_ids if cid})
        for start in range(0, len(unique), 50):
            chunk = unique[start : start + 50]
            records = await self._paginate(
                "molecule.json",
                collection="molecules",
                params={
                    "molecule_chembl_id__in": ",".join(chunk),
                    "only": MOLECULE_FIELDS,
                },
                limit=len(chunk),
                operation=f"molecule.batch.{start}",
            )
            results.extend(records)
        return results

    async def search_molecules(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return await self._paginate(
            "molecule/search.json",
            collection="molecules",
            params={"q": query},
            limit=limit,
            operation=f"molecule.search.{query[:40]}",
        )

    async def similar_molecules(
        self, smiles: str, *, threshold_percent: int = 70, limit: int = 50
    ) -> list[dict[str, Any]]:
        """ChEMBL's own similarity search -- used to *discover* compounds not yet local.

        Local Tanimoto search (``chem.similarity``) covers everything already ingested;
        this covers the rest of ChEMBL, letting a researcher pull in neighbours of a hit
        on demand.
        """
        threshold = max(40, min(100, threshold_percent))
        from urllib.parse import quote

        return await self._paginate(
            f"similarity/{quote(smiles, safe='')}/{threshold}.json",
            collection="molecules",
            params={"only": MOLECULE_FIELDS},
            limit=limit,
            operation=f"similarity.{threshold}",
        )

    async def substructure_molecules(self, smiles: str, *, limit: int = 50) -> list[dict[str, Any]]:
        from urllib.parse import quote

        return await self._paginate(
            f"substructure/{quote(smiles, safe='')}.json",
            collection="molecules",
            params={"only": MOLECULE_FIELDS},
            limit=limit,
            operation="substructure",
        )

    # ------------------------------------------------------------------- targets
    async def find_targets_by_uniprot(self, accession: str) -> list[dict[str, Any]]:
        """Map a UniProt accession to ChEMBL target(s) -- the Open Targets bridge."""
        return await self._paginate(
            "target.json",
            collection="targets",
            params={
                "target_components__accession": accession,
                "only": "target_chembl_id,pref_name,target_type,organism,tax_id",
            },
            limit=25,
            operation=f"target.uniprot.{accession}",
        )

    async def fetch_target(self, target_chembl_id: str) -> dict[str, Any]:
        return await self._get(
            f"target/{target_chembl_id.upper()}.json",
            operation=f"target.{target_chembl_id.upper()}",
        )

    # ---------------------------------------------------------------- activities
    async def fetch_activities_for_target(
        self,
        target_chembl_id: str,
        *,
        activity_types: list[str] | None = None,
        require_pchembl: bool = True,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "target_chembl_id": target_chembl_id.upper(),
            "only": ACTIVITY_FIELDS,
        }
        if require_pchembl:
            params["pchembl_value__isnull"] = "false"
        if activity_types:
            params["standard_type__in"] = ",".join(activity_types)
        return await self._paginate(
            "activity.json",
            collection="activities",
            params=params,
            limit=limit,
            operation=f"activity.target.{target_chembl_id.upper()}",
        )

    async def fetch_activities_for_molecule(
        self, chembl_id: str, *, limit: int = 200
    ) -> list[dict[str, Any]]:
        return await self._paginate(
            "activity.json",
            collection="activities",
            params={
                "molecule_chembl_id": chembl_id.upper(),
                "pchembl_value__isnull": "false",
                "only": ACTIVITY_FIELDS,
            },
            limit=limit,
            operation=f"activity.molecule.{chembl_id.upper()}",
        )

    async def fetch_assays(self, assay_ids: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        unique = sorted({aid for aid in assay_ids if aid})
        for start in range(0, len(unique), 50):
            chunk = unique[start : start + 50]
            records = await self._paginate(
                "assay.json",
                collection="assays",
                params={
                    "assay_chembl_id__in": ",".join(chunk),
                    "only": (
                        "assay_chembl_id,description,assay_type,assay_category,"
                        "assay_organism,assay_tissue,assay_cell_type,confidence_score,"
                        "bao_format,bao_label,document_chembl_id,target_chembl_id"
                    ),
                },
                limit=len(chunk),
                operation=f"assay.batch.{start}",
            )
            results.extend(records)
        return results


# --------------------------------------------------------------- normalisation
def normalize_molecule(
    payload: dict[str, Any],
    *,
    retrieved_at: datetime,
    dataset_version_id: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Normalise a ChEMBL molecule into ``(compound_row, reject)``.

    Structures are re-standardised with RDKit rather than trusting the upstream SMILES,
    because the InChIKey produced here is the deduplication key across *all* sources.
    """
    chembl_id = payload.get("molecule_chembl_id")
    structures = payload.get("molecule_structures") or {}
    smiles = structures.get("canonical_smiles")

    if not chembl_id:
        return None, {"payload": str(payload)[:200], "_quality_errors": ["missing_chembl_id"]}
    if not smiles:
        return None, {
            "chembl_id": chembl_id,
            "_quality_errors": ["missing_structure"],
        }

    info, error = safe_standardize(smiles)
    if info is None:
        return None, {
            "chembl_id": chembl_id,
            "smiles": smiles[:120],
            "_quality_errors": [f"unparseable_structure: {error}"],
        }

    synonyms = sorted(
        {
            str(entry.get("molecule_synonym"))
            for entry in payload.get("molecule_synonyms") or []
            if entry.get("molecule_synonym")
        }
    )[:30]

    cross_references = payload.get("cross_references") or []
    pubchem_cid = next(
        (
            str(ref.get("xref_id"))
            for ref in cross_references
            if str(ref.get("xref_src", "")).lower().startswith("pubchem")
        ),
        None,
    )
    drugbank_id = next(
        (
            str(ref.get("xref_id"))
            for ref in cross_references
            if str(ref.get("xref_src", "")).lower().startswith("drugbank")
        ),
        None,
    )

    max_phase = payload.get("max_phase")
    try:
        max_phase_value = float(max_phase) if max_phase is not None else None
    except (TypeError, ValueError):
        max_phase_value = None

    row = {
        "inchikey": info.inchikey,
        "canonical_smiles": info.canonical_smiles,
        "inchi": info.inchi or structures.get("standard_inchi"),
        "name": payload.get("pref_name"),
        "synonyms": synonyms,
        "chembl_id": chembl_id,
        "pubchem_cid": pubchem_cid,
        "drugbank_id": drugbank_id,
        "max_phase": max_phase_value,
        "molecule_type": payload.get("molecule_type"),
        "first_approval": payload.get("first_approval"),
        "is_approved_drug": bool(max_phase_value is not None and max_phase_value >= 4),
        "withdrawn_flag": bool(payload.get("withdrawn_flag")),
        "oral": payload.get("oral"),
        "parenteral": payload.get("parenteral"),
        "murcko_scaffold": info.murcko_scaffold,
        "murcko_scaffold_inchikey": info.murcko_scaffold_inchikey,
        "heavy_atom_count": info.heavy_atom_count,
        "source": SOURCE_ID,
        "source_identifier": chembl_id,
        "source_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/",
        "retrieved_at": retrieved_at,
        "dataset_version_id": dataset_version_id,
        "_standardization_notes": info.standardization_notes,
    }
    return row, None


def normalize_molecules(
    payloads: list[dict[str, Any]],
    *,
    retrieved_at: datetime,
    dataset_version_id: str | None,
) -> NormalizedBatch:
    batch = NormalizedBatch(entity_type="compound")
    seen_inchikeys: set[str] = set()
    for payload in payloads:
        row, reject = normalize_molecule(
            payload, retrieved_at=retrieved_at, dataset_version_id=dataset_version_id
        )
        if reject is not None:
            batch.rejects.append(reject)
            continue
        assert row is not None
        if row["inchikey"] in seen_inchikeys:
            continue
        seen_inchikeys.add(row["inchikey"])
        notes = row.pop("_standardization_notes", [])
        if notes:
            batch.raw_payloads.append(
                {"chembl_id": row["chembl_id"], "standardization_notes": notes}
            )
        batch.rows.append(row)
    return batch


def normalize_assay(
    payload: dict[str, Any],
    *,
    target_id: str | None,
    retrieved_at: datetime,
    dataset_version_id: str | None,
) -> dict[str, Any] | None:
    assay_id = payload.get("assay_chembl_id")
    if not assay_id:
        return None
    return {
        "chembl_assay_id": assay_id,
        "description": payload.get("description"),
        "assay_type": payload.get("assay_type"),
        "assay_category": payload.get("assay_category"),
        "organism": payload.get("assay_organism"),
        "tissue": payload.get("assay_tissue"),
        "cell_type": payload.get("assay_cell_type"),
        "confidence_score": payload.get("confidence_score"),
        "bao_format": payload.get("bao_format"),
        "bao_label": payload.get("bao_label"),
        "document_chembl_id": payload.get("document_chembl_id"),
        "target_id": target_id,
        "chembl_target_id": payload.get("target_chembl_id"),
        "source": SOURCE_ID,
        "source_identifier": assay_id,
        "source_url": f"https://www.ebi.ac.uk/chembl/assay_report_card/{assay_id}/",
        "retrieved_at": retrieved_at,
        "dataset_version_id": dataset_version_id,
    }


def normalize_activities(
    payloads: list[dict[str, Any]],
    *,
    compound_id_by_chembl: dict[str, str],
    assay_id_by_chembl: dict[str, str],
    target_id_by_chembl: dict[str, str],
    retrieved_at: datetime,
    dataset_version_id: str | None,
) -> NormalizedBatch:
    """Normalise activity rows, resolving foreign keys and flagging quality issues.

    ``is_active`` is derived only from pChEMBL >= 6 (1 uM) *and* an exact ``=``
    relation: a ``>10000 nM`` measurement means "inactive up to this concentration",
    not "active", and conflating the two is a classic way to poison a training set.
    """
    batch = NormalizedBatch(entity_type="bioactivity")
    for payload in payloads:
        activity_id = payload.get("activity_id")
        molecule_chembl = payload.get("molecule_chembl_id")
        compound_id = compound_id_by_chembl.get(str(molecule_chembl))
        if activity_id is None or compound_id is None:
            batch.rejects.append(
                {
                    "activity_id": activity_id,
                    "molecule_chembl_id": molecule_chembl,
                    "_quality_errors": ["unresolved_compound"],
                }
            )
            continue

        pchembl = payload.get("pchembl_value")
        try:
            pchembl_value = float(pchembl) if pchembl is not None else None
        except (TypeError, ValueError):
            pchembl_value = None

        relation = payload.get("standard_relation")
        is_active: bool | None = None
        if pchembl_value is not None:
            is_active = pchembl_value >= 6.0 and relation in ("=", "~", None)

        try:
            standard_value = (
                float(payload["standard_value"])
                if payload.get("standard_value") is not None
                else None
            )
        except (TypeError, ValueError):
            standard_value = None

        batch.rows.append(
            {
                "compound_id": compound_id,
                "target_id": target_id_by_chembl.get(str(payload.get("target_chembl_id"))),
                "assay_id": assay_id_by_chembl.get(str(payload.get("assay_chembl_id"))),
                "activity_type": payload.get("standard_type"),
                "standard_value": standard_value,
                "standard_units": payload.get("standard_units"),
                "standard_relation": relation,
                "pchembl_value": pchembl_value,
                "is_active": is_active,
                "activity_comment": payload.get("activity_comment"),
                "data_validity_comment": payload.get("data_validity_comment"),
                "potential_duplicate": bool(payload.get("potential_duplicate")),
                "document_chembl_id": payload.get("document_chembl_id"),
                "source": SOURCE_ID,
                "source_identifier": str(activity_id),
                "source_url": (
                    "https://www.ebi.ac.uk/chembl/assay_report_card/"
                    f"{payload.get('assay_chembl_id')}/"
                ),
                "retrieved_at": retrieved_at,
                "dataset_version_id": dataset_version_id,
            }
        )
    return batch


# ------------------------------------------------------------------- quality rules
COMPOUND_RULES: list[QualityRule] = [
    required("inchikey"),
    required("canonical_smiles"),
    required("chembl_id"),
    in_range("max_phase", 0.0, 4.0),
]

ACTIVITY_RULES: list[QualityRule] = [
    required("compound_id"),
    required("source_identifier"),
    in_range("pchembl_value", 0.0, 15.0),
    QualityRule(
        name="units_present_for_value",
        description="a numeric activity value must carry units",
        predicate=lambda row: row.get("standard_value") is None or bool(row.get("standard_units")),
        severity="warning",
    ),
    QualityRule(
        name="no_validity_comment",
        description="ChEMBL curators flagged this measurement",
        predicate=lambda row: not row.get("data_validity_comment"),
        severity="warning",
    ),
]


def apply_compound_quality(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted, rejected, report = evaluate_quality(
        rows,
        COMPOUND_RULES,
        completeness_fields=["name", "max_phase", "pubchem_cid", "murcko_scaffold"],
        dedup_key="inchikey",
    )
    return accepted, rejected, report.to_dict()


def apply_activity_quality(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted, rejected, report = evaluate_quality(
        rows,
        ACTIVITY_RULES,
        completeness_fields=["pchembl_value", "standard_units", "assay_id", "target_id"],
        dedup_key="source_identifier",
    )
    return accepted, rejected, report.to_dict()


def build_connector(
    *,
    base_url: str,
    timeout_s: float,
    max_retries: int,
    user_agent: str,
    recorder: ResponseRecorder | None = None,
) -> ChEMBLConnector:
    client = SourceHttpClient(
        SOURCE_ID, timeout_s=timeout_s, max_retries=max_retries, user_agent=user_agent
    )
    return ChEMBLConnector(client, base_url=base_url, recorder=recorder)


DATA_SOURCE_DEFINITION = {
    "id": SOURCE_ID,
    "name": "ChEMBL",
    "description": (
        "Manually curated database of bioactive molecules with drug-like properties, "
        "including structures, assay metadata and quantitative bioactivity."
    ),
    "homepage_url": "https://www.ebi.ac.uk/chembl/",
    "api_base_url": "https://www.ebi.ac.uk/chembl/api/data",
    "license": "CC BY-SA 3.0",
    "citation": (
        "Zdrazil B, et al. The ChEMBL Database in 2023: a drug discovery platform "
        "spanning multiple bioactivity data types and time periods. "
        "Nucleic Acids Research (2024)."
    ),
    "entity_types": ["compound", "assay", "bioactivity", "target"],
}

__all__ = [
    "ACTIVITY_RULES",
    "COMPOUND_RULES",
    "DATA_SOURCE_DEFINITION",
    "MIN_ASSAY_CONFIDENCE",
    "POTENCY_TYPES",
    "SOURCE_ID",
    "ChEMBLConnector",
    "apply_activity_quality",
    "apply_compound_quality",
    "build_connector",
    "normalize_activities",
    "normalize_assay",
    "normalize_molecule",
    "normalize_molecules",
]
