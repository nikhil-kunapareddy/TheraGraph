"""Europe PMC connector -- literature *metadata* only.

TheraGraph stores bibliographic metadata (title, journal, year, authors, abstract
snippet, open-access flag) and links out. It deliberately does not store or serve full
text, and the research agent may only cite publications that exist as rows here, which
is what prevents invented references.

Data licence: Europe PMC REST API metadata is free to use; full text carries
article-specific licences.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from theragraph.core.http_client import SourceHttpClient
from theragraph.core.logging import get_logger
from theragraph.ingestion.base import ResponseRecorder

log = get_logger(__name__)

SOURCE_ID = "europepmc"
MAX_ABSTRACT_SNIPPET = 1200


class EuropePMCConnector:
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

    async def search(
        self, query: str, *, limit: int = 25, result_type: str = "core"
    ) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "format": "json",
            "resultType": result_type,
            "pageSize": min(limit, 100),
            "sort": "CITED desc",
        }

        async def loader() -> Any:
            return await self._client.get_json(f"{self._base}/search", params=params)

        if self._recorder is not None:
            payload = await self._recorder.fetch(
                source=SOURCE_ID,
                operation=f"search.{query[:60]}",
                params=params,
                loader=loader,
            )
        else:
            payload = await loader()
        return ((payload or {}).get("resultList") or {}).get("result") or []

    async def by_pmids(self, pmids: list[str], *, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch metadata for specific PMIDs (the ones cited by Open Targets evidence)."""
        unique = [pmid for pmid in dict.fromkeys(pmids) if pmid][:limit]
        if not unique:
            return []
        query = " OR ".join(f"EXT_ID:{pmid}" for pmid in unique)
        return await self.search(query, limit=len(unique))


def normalize_publication(
    payload: dict[str, Any], *, retrieved_at: datetime, dataset_version_id: str | None
) -> dict[str, Any] | None:
    external_id = payload.get("id") or payload.get("pmid")
    if not external_id:
        return None
    title = payload.get("title")
    if not title:
        return None

    authors = [
        str(author.get("fullName"))
        for author in ((payload.get("authorList") or {}).get("author") or [])
        if author.get("fullName")
    ]
    if not authors and payload.get("authorString"):
        authors = [
            name.strip() for name in str(payload["authorString"]).split(",") if name.strip()
        ][:20]

    year = payload.get("pubYear")
    try:
        year_value = int(year) if year else None
    except (TypeError, ValueError):
        year_value = None

    abstract = payload.get("abstractText")
    pmid = str(payload["pmid"]) if payload.get("pmid") else None

    return {
        "external_id": str(external_id),
        "pmid": pmid,
        "doi": payload.get("doi"),
        "title": str(title).strip(),
        "journal": payload.get("journalTitle")
        or ((payload.get("journalInfo") or {}).get("journal") or {}).get("title"),
        "year": year_value,
        "authors": authors[:20],
        "abstract_snippet": str(abstract)[:MAX_ABSTRACT_SNIPPET] if abstract else None,
        "citation_count": payload.get("citedByCount"),
        "is_open_access": str(payload.get("isOpenAccess", "")).upper() == "Y",
        "source": SOURCE_ID,
        "source_identifier": str(external_id),
        "source_url": (
            f"https://europepmc.org/article/{payload.get('source', 'MED')}/{external_id}"
        ),
        "retrieved_at": retrieved_at,
        "dataset_version_id": dataset_version_id,
    }


def build_connector(
    *,
    base_url: str,
    timeout_s: float,
    max_retries: int,
    user_agent: str,
    recorder: ResponseRecorder | None = None,
) -> EuropePMCConnector:
    client = SourceHttpClient(
        SOURCE_ID, timeout_s=timeout_s, max_retries=max_retries, user_agent=user_agent
    )
    return EuropePMCConnector(client, base_url=base_url, recorder=recorder)


DATA_SOURCE_DEFINITION = {
    "id": SOURCE_ID,
    "name": "Europe PMC",
    "description": (
        "Life-sciences literature database; used for publication metadata and citations."
    ),
    "homepage_url": "https://europepmc.org/",
    "api_base_url": "https://www.ebi.ac.uk/europepmc/webservices/rest",
    "license": "Metadata free to use; article licences vary",
    "citation": ("Ferguson C, et al. Europe PMC in 2020. Nucleic Acids Research (2021)."),
    "entity_types": ["publication"],
}

__all__ = [
    "DATA_SOURCE_DEFINITION",
    "SOURCE_ID",
    "EuropePMCConnector",
    "build_connector",
    "normalize_publication",
]
