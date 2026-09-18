"""PubChem PUG-REST connector.

PubChem is used as a *cross-reference and enrichment* source rather than a primary one:
it resolves trivial names to structures, supplies CIDs for compounds that arrived from
ChEMBL without one, and provides an independent read on experimental-ish properties
(XLogP3, complexity) for cross-checking.

Two operational notes:

* PubChem asks for <= 5 requests/second; the connector batches CIDs into single calls
  and the shared HTTP client backs off on 503 "server busy" responses.
* The property key for connectivity SMILES was renamed (``CanonicalSMILES`` ->
  ``ConnectivitySMILES``); both spellings are accepted so the connector survives the
  transition.

Data licence: PubChem data is in the public domain; individual depositor records may
carry their own terms.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from theragraph.chem.molecule import safe_standardize
from theragraph.core.errors import UpstreamError
from theragraph.core.http_client import SourceHttpClient
from theragraph.core.logging import get_logger
from theragraph.ingestion.base import ResponseRecorder

log = get_logger(__name__)

SOURCE_ID = "pubchem"

PROPERTY_FIELDS = [
    "MolecularFormula",
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "Complexity",
    "CanonicalSMILES",
    "InChIKey",
    "IUPACName",
    "Title",
]

#: Upstream renamed this property; accept either spelling.
_SMILES_KEYS = ("ConnectivitySMILES", "CanonicalSMILES", "SMILES", "IsomericSMILES")


class PubChemConnector:
    source_id = SOURCE_ID

    def __init__(
        self,
        client: SourceHttpClient,
        *,
        base_url: str,
        recorder: ResponseRecorder | None = None,
    ) -> None:
        self._client = client
        self._base = base_url.rstrip("/")
        self._recorder = recorder

    async def _get(self, path: str, *, operation: str) -> Any:
        url = f"{self._base}/{path.lstrip('/')}"

        async def loader() -> Any:
            return await self._client.get_json(url)

        if self._recorder is not None:
            return await self._recorder.fetch(
                source=SOURCE_ID, operation=operation, params={"path": path}, loader=loader
            )
        return await loader()

    async def cids_for_name(self, name: str, *, limit: int = 5) -> list[str]:
        from urllib.parse import quote

        try:
            payload = await self._get(
                f"compound/name/{quote(name, safe='')}/cids/JSON",
                operation=f"cids.name.{name[:40]}",
            )
        except UpstreamError as exc:
            if exc.details.get("status_code") == 404:
                return []
            raise
        cids = ((payload or {}).get("IdentifierList") or {}).get("CID") or []
        return [str(cid) for cid in cids[:limit]]

    async def cid_for_inchikey(self, inchikey: str) -> str | None:
        try:
            payload = await self._get(
                f"compound/inchikey/{inchikey}/cids/JSON",
                operation=f"cids.inchikey.{inchikey}",
            )
        except UpstreamError as exc:
            if exc.details.get("status_code") == 404:
                return None
            raise
        cids = ((payload or {}).get("IdentifierList") or {}).get("CID") or []
        return str(cids[0]) if cids else None

    async def properties(self, cids: list[str]) -> list[dict[str, Any]]:
        """Batch property lookup (one request per 100 CIDs)."""
        results: list[dict[str, Any]] = []
        unique = [cid for cid in dict.fromkeys(cids) if cid]
        for start in range(0, len(unique), 100):
            chunk = unique[start : start + 100]
            payload = await self._get(
                f"compound/cid/{','.join(chunk)}/property/{','.join(PROPERTY_FIELDS)}/JSON",
                operation=f"properties.{start}.{len(chunk)}",
            )
            rows = ((payload or {}).get("PropertyTable") or {}).get("Properties") or []
            results.extend(rows)
        return results

    async def synonyms(self, cid: str, *, limit: int = 20) -> list[str]:
        try:
            payload = await self._get(
                f"compound/cid/{cid}/synonyms/JSON", operation=f"synonyms.{cid}"
            )
        except UpstreamError as exc:
            if exc.details.get("status_code") == 404:
                return []
            raise
        information = ((payload or {}).get("InformationList") or {}).get("Information") or []
        if not information:
            return []
        return [str(name) for name in (information[0].get("Synonym") or [])[:limit]]


def _smiles_of(payload: dict[str, Any]) -> str | None:
    for key in _SMILES_KEYS:
        value = payload.get(key)
        if value:
            return str(value)
    return None


def normalize_compound(
    payload: dict[str, Any],
    *,
    retrieved_at: datetime,
    dataset_version_id: str | None,
    synonyms: list[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Normalise a PubChem property record into a compound row."""
    cid = payload.get("CID")
    smiles = _smiles_of(payload)
    if cid is None:
        return None, {"_quality_errors": ["missing_cid"]}
    if not smiles:
        return None, {"pubchem_cid": str(cid), "_quality_errors": ["missing_structure"]}

    info, error = safe_standardize(smiles)
    if info is None:
        return None, {
            "pubchem_cid": str(cid),
            "smiles": smiles[:120],
            "_quality_errors": [f"unparseable_structure: {error}"],
        }

    return {
        "inchikey": info.inchikey,
        "canonical_smiles": info.canonical_smiles,
        "inchi": info.inchi,
        "name": payload.get("Title") or payload.get("IUPACName"),
        "synonyms": (synonyms or [])[:30],
        "pubchem_cid": str(cid),
        "murcko_scaffold": info.murcko_scaffold,
        "murcko_scaffold_inchikey": info.murcko_scaffold_inchikey,
        "heavy_atom_count": info.heavy_atom_count,
        "source": SOURCE_ID,
        "source_identifier": str(cid),
        "source_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        "retrieved_at": retrieved_at,
        "dataset_version_id": dataset_version_id,
    }, None


def normalize_reported_properties(
    payload: dict[str, Any], *, compound_id: str
) -> dict[str, Any] | None:
    """Store PubChem's own computed descriptors as a second, attributed provider.

    Keeping them under ``provider='pubchem'`` rather than merging into the RDKit row is
    what lets the UI show "RDKit LogP 1.31 / PubChem XLogP3 1.2" side by side instead of
    silently picking one.
    """
    if not payload:
        return None

    def number(key: str) -> float | None:
        value = payload.get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def integer(key: str) -> int | None:
        value = number(key)
        return int(value) if value is not None else None

    return {
        "compound_id": compound_id,
        "provider": SOURCE_ID,
        "descriptor_set_version": "pug-rest-1",
        "molecular_weight": number("MolecularWeight"),
        "logp": number("XLogP"),
        "tpsa": number("TPSA"),
        "hbd": integer("HBondDonorCount"),
        "hba": integer("HBondAcceptorCount"),
        "rotatable_bonds": integer("RotatableBondCount"),
        "heavy_atom_count": integer("HeavyAtomCount"),
        "descriptors": {
            "complexity": number("Complexity"),
            "molecular_formula": payload.get("MolecularFormula"),
            "iupac_name": payload.get("IUPACName"),
        },
    }


def build_connector(
    *,
    base_url: str,
    timeout_s: float,
    max_retries: int,
    user_agent: str,
    recorder: ResponseRecorder | None = None,
) -> PubChemConnector:
    client = SourceHttpClient(
        SOURCE_ID, timeout_s=timeout_s, max_retries=max_retries, user_agent=user_agent
    )
    return PubChemConnector(client, base_url=base_url, recorder=recorder)


DATA_SOURCE_DEFINITION = {
    "id": SOURCE_ID,
    "name": "PubChem",
    "description": (
        "NIH chemical information repository; used for name-to-structure resolution, "
        "cross-references and independent computed properties."
    ),
    "homepage_url": "https://pubchem.ncbi.nlm.nih.gov/",
    "api_base_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
    "license": "Public domain (individual depositor records may vary)",
    "citation": ("Kim S, et al. PubChem 2023 update. Nucleic Acids Research (2023)."),
    "entity_types": ["compound", "molecular_property"],
}

__all__ = [
    "DATA_SOURCE_DEFINITION",
    "PROPERTY_FIELDS",
    "SOURCE_ID",
    "PubChemConnector",
    "build_connector",
    "normalize_compound",
    "normalize_reported_properties",
]
