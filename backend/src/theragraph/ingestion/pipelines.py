"""Ingestion pipelines: connectors + repositories + provenance, composed.

Every pipeline follows the same contract:

1. open an :class:`IngestionRun` row (so a crashed ingestion is visible, not invisible),
2. fetch from upstream through a connector,
3. normalise and run data-quality rules,
4. create an immutable :class:`DatasetVersion` (content hash + retrieval timestamp),
5. upsert the normalised rows with that version stamped on every one,
6. close the run with counts, rejects and the quality report.

Pipelines are idempotent: re-running one with unchanged upstream data produces the same
content hash and updates rather than duplicates rows, because every table has a natural
key (EFO id, Ensembl id, InChIKey, ChEMBL activity id).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.chem.featurize import featurize_many
from theragraph.config import Settings
from theragraph.core.clock import Stopwatch, utcnow
from theragraph.core.errors import UpstreamError
from theragraph.core.logging import get_logger, log_context
from theragraph.core.metrics import INGESTION_RECORDS, INGESTION_REJECTS
from theragraph.db.repositories import biology as biology_repo
from theragraph.db.repositories import chemistry as chem_repo
from theragraph.db.repositories import provenance as prov_repo
from theragraph.ingestion import chembl, europepmc, moleculenet, opentargets, pubchem
from theragraph.ingestion.base import (
    IngestionMode,
    IngestionOutcome,
    ResponseRecorder,
    version_label,
)

log = get_logger(__name__)

ALL_DATA_SOURCE_DEFINITIONS = [
    opentargets.DATA_SOURCE_DEFINITION,
    chembl.DATA_SOURCE_DEFINITION,
    pubchem.DATA_SOURCE_DEFINITION,
    europepmc.DATA_SOURCE_DEFINITION,
    moleculenet.DATA_SOURCE_DEFINITION,
]


@dataclass
class ConnectorBundle:
    """The set of connectors a pipeline run uses, built once and closed together."""

    open_targets: opentargets.OpenTargetsConnector
    chembl: chembl.ChEMBLConnector
    pubchem: pubchem.PubChemConnector
    europepmc: europepmc.EuropePMCConnector
    moleculenet: moleculenet.MoleculeNetConnector
    recorder: ResponseRecorder

    async def aclose(self) -> None:
        for connector in (
            self.open_targets,
            self.chembl,
            self.pubchem,
            self.europepmc,
            self.moleculenet,
        ):
            client = getattr(connector, "_client", None)
            if client is not None:
                await client.aclose()


class _ConnectorDefaults(TypedDict):
    """The kwargs every connector factory shares, typed so ``**common`` checks."""

    timeout_s: float
    max_retries: int
    user_agent: str
    recorder: ResponseRecorder | None


def build_connectors(settings: Settings, *, mode: IngestionMode) -> ConnectorBundle:
    """Construct all connectors sharing one fixture recorder."""
    recorder = ResponseRecorder(settings.seed_dir / "fixtures", mode=mode)
    common: _ConnectorDefaults = {
        "timeout_s": settings.http_timeout_s,
        "max_retries": settings.http_max_retries,
        "user_agent": settings.http_user_agent,
        "recorder": recorder,
    }
    return ConnectorBundle(
        open_targets=opentargets.build_connector(
            graphql_url=str(settings.opentargets_graphql_url), **common
        ),
        chembl=chembl.build_connector(base_url=str(settings.chembl_base_url), **common),
        pubchem=pubchem.build_connector(base_url=str(settings.pubchem_base_url), **common),
        europepmc=europepmc.build_connector(base_url=str(settings.europepmc_base_url), **common),
        moleculenet=moleculenet.build_connector(
            base_url=str(settings.moleculenet_base_url),
            cache_dir=settings.data_dir / "raw" / "moleculenet",
            **common,
        ),
        recorder=recorder,
    )


async def ensure_data_sources(session: AsyncSession) -> int:
    """Register every data source definition (idempotent)."""
    for definition in ALL_DATA_SOURCE_DEFINITIONS:
        await prov_repo.upsert_data_source(session, **definition)
    return len(ALL_DATA_SOURCE_DEFINITIONS)


class IngestionService:
    """Orchestrates the ingestion pipelines."""

    def __init__(
        self,
        settings: Settings,
        connectors: ConnectorBundle,
        *,
        mode: IngestionMode = IngestionMode.AUTO,
    ) -> None:
        self.settings = settings
        self.connectors = connectors
        self.mode = mode

    # ------------------------------------------------------------------ diseases
    async def ingest_disease(
        self,
        session: AsyncSession,
        efo_id: str,
        *,
        n_targets: int = 25,
    ) -> IngestionOutcome:
        """Ingest a disease, its top associated targets and the association scores.

        Targets are ingested in the same run because an association row is meaningless
        without both endpoints, and a half-ingested association would be a dangling
        edge in the evidence graph.
        """
        run = await prov_repo.start_ingestion_run(
            session,
            source_id=opentargets.SOURCE_ID,
            dataset_name="disease_associations",
            mode=str(self.mode),
            params={"efo_id": efo_id, "n_targets": n_targets},
        )
        stopwatch = Stopwatch()
        with log_context(job_id=run.id), stopwatch:
            try:
                release = await self.connectors.open_targets.release()
                retrieved_at = utcnow()
                payload = await self.connectors.open_targets.fetch_disease(
                    efo_id, associated_targets=n_targets
                )

                association_rows = (payload.get("associatedTargets") or {}).get("rows") or []
                target_payloads: list[dict[str, Any]] = []
                for row in association_rows:
                    ensembl_id = (row.get("target") or {}).get("id")
                    if not ensembl_id:
                        continue
                    try:
                        target_payloads.append(
                            await self.connectors.open_targets.fetch_target(ensembl_id)
                        )
                    except UpstreamError as exc:
                        log.warning(
                            "ingestion.target_fetch_failed",
                            ensembl_id=ensembl_id,
                            error=exc.message,
                        )

                version = version_label(retrieved_at, release.data_version)
                disease_row = opentargets.normalize_disease(
                    payload, retrieved_at=retrieved_at, dataset_version_id=None
                )
                gene_rows = [
                    opentargets.normalize_gene(
                        target, retrieved_at=retrieved_at, dataset_version_id=None
                    )
                    for target in target_payloads
                ]
                target_rows = [
                    opentargets.normalize_target(
                        target, retrieved_at=retrieved_at, dataset_version_id=None
                    )
                    for target in target_payloads
                ]

                dataset_version = await prov_repo.create_dataset_version(
                    session,
                    source_id=opentargets.SOURCE_ID,
                    dataset_name="disease_associations",
                    version=version,
                    content_hash=_hash_all([disease_row, *target_rows, *association_rows]),
                    record_count=1 + len(target_rows) + len(association_rows),
                    retrieved_at=retrieved_at,
                    source_url=f"https://platform.opentargets.org/disease/{efo_id}",
                    retrieval_params={
                        "efo_id": efo_id,
                        "n_targets": n_targets,
                        "api_version": release.api_version,
                    },
                    quality={
                        "targets_requested": len(association_rows),
                        "targets_resolved": len(target_rows),
                        "resolution_rate": round(len(target_rows) / len(association_rows), 4)
                        if association_rows
                        else 1.0,
                    },
                )
                for row in (disease_row, *gene_rows, *target_rows):
                    row["dataset_version_id"] = dataset_version.id

                disease_ids = await biology_repo.upsert_diseases(session, [disease_row])
                gene_ids = await biology_repo.upsert_genes(session, gene_rows)
                for target_row, gene_id in zip(target_rows, gene_ids, strict=False):
                    target_row["gene_id"] = gene_id
                target_ids = await biology_repo.upsert_targets(session, target_rows)

                target_id_by_key = {
                    row["target_key"]: target_id
                    for row, target_id in zip(target_rows, target_ids, strict=False)
                }
                association_batch = opentargets.normalize_associations(
                    payload,
                    disease_id=disease_ids[0],
                    target_id_by_key=target_id_by_key,
                    retrieved_at=retrieved_at,
                    dataset_version_id=dataset_version.id,
                )
                written_associations = await biology_repo.upsert_associations(
                    session, association_batch.rows
                )

                written = 1 + len(target_ids) + len(written_associations)
                INGESTION_RECORDS.labels(source=opentargets.SOURCE_ID, entity="disease").inc()
                INGESTION_RECORDS.labels(source=opentargets.SOURCE_ID, entity="target").inc(
                    len(target_ids)
                )
                INGESTION_RECORDS.labels(source=opentargets.SOURCE_ID, entity="association").inc(
                    len(written_associations)
                )
                if association_batch.rejects:
                    INGESTION_REJECTS.labels(
                        source=opentargets.SOURCE_ID, reason="unresolved_target"
                    ).inc(len(association_batch.rejects))

                await prov_repo.finish_ingestion_run(
                    session,
                    run,
                    status="succeeded",
                    dataset_version_id=dataset_version.id,
                    records_fetched=1 + len(association_rows) + len(target_payloads),
                    records_written=written,
                    records_rejected=len(association_batch.rejects),
                    rejects=association_batch.rejects,
                )
                return IngestionOutcome(
                    source_id=opentargets.SOURCE_ID,
                    dataset_name="disease_associations",
                    version=version,
                    mode=self.mode,
                    retrieved_at=retrieved_at,
                    content_hash=dataset_version.content_hash,
                    records_fetched=1 + len(association_rows) + len(target_payloads),
                    records_written=written,
                    records_rejected=len(association_batch.rejects),
                    dataset_version_id=dataset_version.id,
                    entity_counts={
                        "disease": 1,
                        "gene": len(gene_ids),
                        "target": len(target_ids),
                        "association": len(written_associations),
                    },
                    quality=dataset_version.quality,
                    rejects=association_batch.rejects[:20],
                )
            except Exception as exc:
                await prov_repo.finish_ingestion_run(
                    session, run, status="failed", error=str(exc)[:4000]
                )
                raise

    # ------------------------------------------------------------------ evidence
    async def ingest_evidence(
        self,
        session: AsyncSession,
        *,
        efo_id: str,
        target_keys: list[str],
        datasource_ids: list[str] | None = None,
        per_datasource: int = 10,
        with_publications: bool = True,
    ) -> IngestionOutcome:
        """Ingest evidence for target-disease pairs, one query per datasource.

        Querying per datasource (rather than once for all of them) guarantees a
        representative mix: a single combined query returns whichever datasource sorts
        first and buries the rest behind pagination.
        """
        datasources = datasource_ids or opentargets.DEFAULT_EVIDENCE_DATASOURCES
        run = await prov_repo.start_ingestion_run(
            session,
            source_id=opentargets.SOURCE_ID,
            dataset_name="evidence",
            mode=str(self.mode),
            params={
                "efo_id": efo_id,
                "target_keys": target_keys,
                "datasources": datasources,
            },
        )
        try:
            disease = await biology_repo.get_disease(session, efo_id)
            if disease is None:
                raise UpstreamError(
                    f"disease {efo_id} must be ingested before its evidence",
                    source=opentargets.SOURCE_ID,
                )
            targets = [
                target
                for target in [await biology_repo.get_target(session, key) for key in target_keys]
                if target is not None
            ]
            target_id_by_key = {target.target_key: target.id for target in targets}
            if not target_id_by_key:
                raise UpstreamError(
                    "no known targets for evidence ingestion",
                    source=opentargets.SOURCE_ID,
                    details={"target_keys": target_keys},
                )

            release = await self.connectors.open_targets.release()
            retrieved_at = utcnow()
            raw_rows: list[dict[str, Any]] = []
            for datasource in datasources:
                try:
                    rows = await self.connectors.open_targets.fetch_evidence(
                        efo_id,
                        ensembl_ids=list(target_id_by_key),
                        datasource_ids=[datasource],
                        size=per_datasource,
                        max_pages=1,
                    )
                except UpstreamError as exc:
                    log.warning(
                        "ingestion.evidence_datasource_failed",
                        datasource=datasource,
                        error=exc.message,
                    )
                    continue
                raw_rows.extend(rows)

            version = version_label(retrieved_at, release.data_version)
            batch = opentargets.normalize_evidence(
                raw_rows,
                disease_id=disease.id,
                target_id_by_key=target_id_by_key,
                retrieved_at=retrieved_at,
                dataset_version_id=None,
            )

            dataset_version = await prov_repo.create_dataset_version(
                session,
                source_id=opentargets.SOURCE_ID,
                dataset_name="evidence",
                version=version,
                content_hash=batch.content_hash,
                record_count=len(batch.rows),
                retrieved_at=retrieved_at,
                source_url=f"https://platform.opentargets.org/disease/{efo_id}",
                retrieval_params={
                    "efo_id": efo_id,
                    "target_keys": sorted(target_id_by_key),
                    "datasources": datasources,
                    "per_datasource": per_datasource,
                },
                quality={
                    "evidence_types": _count_by(batch.rows, "evidence_type"),
                    "with_score": sum(1 for row in batch.rows if row.get("score") is not None),
                    "unresolved_targets": len(batch.rejects),
                },
            )
            for row in batch.rows:
                row["dataset_version_id"] = dataset_version.id

            publication_count = 0
            if with_publications:
                publication_count = await self._attach_publications(
                    session,
                    batch.rows,
                    retrieved_at=retrieved_at,
                    dataset_version_id=dataset_version.id,
                )

            written = await biology_repo.upsert_evidence(session, batch.rows)
            INGESTION_RECORDS.labels(source=opentargets.SOURCE_ID, entity="evidence").inc(
                len(written)
            )

            await prov_repo.finish_ingestion_run(
                session,
                run,
                status="succeeded",
                dataset_version_id=dataset_version.id,
                records_fetched=len(raw_rows),
                records_written=len(written),
                records_rejected=len(batch.rejects),
                rejects=batch.rejects,
            )
            return IngestionOutcome(
                source_id=opentargets.SOURCE_ID,
                dataset_name="evidence",
                version=version,
                mode=self.mode,
                retrieved_at=retrieved_at,
                content_hash=dataset_version.content_hash,
                records_fetched=len(raw_rows),
                records_written=len(written),
                records_rejected=len(batch.rejects),
                dataset_version_id=dataset_version.id,
                entity_counts={
                    "evidence": len(written),
                    "publication": publication_count,
                },
                quality=dataset_version.quality,
            )
        except Exception as exc:
            await prov_repo.finish_ingestion_run(
                session, run, status="failed", error=str(exc)[:4000]
            )
            raise

    async def _attach_publications(
        self,
        session: AsyncSession,
        evidence_rows: list[dict[str, Any]],
        *,
        retrieved_at: Any,
        dataset_version_id: str,
    ) -> int:
        """Fetch Europe PMC metadata for PMIDs cited by evidence and link them.

        This is what makes a literature citation in a report verifiable: the PMID came
        from Open Targets, and the title/journal/year came from Europe PMC. Neither was
        produced by a language model.
        """
        pmids = [
            str(row["extra"]["pmid"])
            for row in evidence_rows
            if (row.get("extra") or {}).get("pmid")
        ]
        if not pmids:
            return 0
        try:
            results = await self.connectors.europepmc.by_pmids(pmids, limit=40)
        except UpstreamError as exc:
            log.warning("ingestion.publication_fetch_failed", error=exc.message)
            return 0

        publication_rows = [
            row
            for row in (
                europepmc.normalize_publication(
                    result,
                    retrieved_at=retrieved_at,
                    dataset_version_id=dataset_version_id,
                )
                for result in results
            )
            if row is not None
        ]
        if not publication_rows:
            return 0

        ids = await biology_repo.upsert_publications(session, publication_rows)
        publication_id_by_pmid = {
            row.get("pmid"): publication_id
            for row, publication_id in zip(publication_rows, ids, strict=False)
            if row.get("pmid")
        }
        for row in evidence_rows:
            pmid = (row.get("extra") or {}).get("pmid")
            if pmid and str(pmid) in publication_id_by_pmid:
                row["publication_id"] = publication_id_by_pmid[str(pmid)]
        INGESTION_RECORDS.labels(source=europepmc.SOURCE_ID, entity="publication").inc(len(ids))
        return len(ids)

    # ----------------------------------------------------------------- chemistry
    async def ingest_target_bioactivity(
        self,
        session: AsyncSession,
        *,
        target_key: str,
        activity_types: list[str] | None = None,
        max_activities: int = 300,
        featurize: bool = True,
    ) -> IngestionOutcome:
        """Ingest ChEMBL compounds, assays and activities for an Open Targets target.

        The UniProt accession recorded on the target is the bridge into ChEMBL. Targets
        with no UniProt mapping (non-protein-coding, complexes) are reported as such
        rather than silently returning zero compounds.
        """
        run = await prov_repo.start_ingestion_run(
            session,
            source_id=chembl.SOURCE_ID,
            dataset_name="bioactivity",
            mode=str(self.mode),
            params={"target_key": target_key, "max_activities": max_activities},
        )
        try:
            target = await biology_repo.get_target(session, target_key)
            if target is None:
                raise UpstreamError(
                    f"target {target_key} must be ingested before its bioactivity",
                    source=chembl.SOURCE_ID,
                )

            chembl_target_ids = list(target.chembl_target_ids or [])
            if not chembl_target_ids:
                for accession in target.uniprot_ids or []:
                    mapped = await self.connectors.chembl.find_targets_by_uniprot(accession)
                    chembl_target_ids.extend(
                        str(row["target_chembl_id"])
                        for row in mapped
                        if row.get("target_chembl_id")
                        and row.get("target_type") == "SINGLE PROTEIN"
                    )
                    if chembl_target_ids:
                        break

            if not chembl_target_ids:
                await prov_repo.finish_ingestion_run(
                    session,
                    run,
                    status="succeeded",
                    records_fetched=0,
                    records_written=0,
                    error="no ChEMBL target mapping for this Open Targets target",
                )
                retrieved_at = utcnow()
                return IngestionOutcome(
                    source_id=chembl.SOURCE_ID,
                    dataset_name="bioactivity",
                    version=version_label(retrieved_at),
                    mode=self.mode,
                    retrieved_at=retrieved_at,
                    content_hash="empty",
                    warnings=[
                        f"{target.approved_symbol} has no ChEMBL single-protein target "
                        "mapping; no bioactivity available"
                    ],
                )

            chembl_target_id = chembl_target_ids[0]
            release = await self.connectors.chembl.version()
            retrieved_at = utcnow()

            activities = await self.connectors.chembl.fetch_activities_for_target(
                chembl_target_id,
                activity_types=activity_types or list(chembl.POTENCY_TYPES),
                limit=max_activities,
            )
            molecule_ids = sorted(
                {
                    str(activity["molecule_chembl_id"])
                    for activity in activities
                    if activity.get("molecule_chembl_id")
                }
            )
            molecules = await self.connectors.chembl.fetch_molecules(molecule_ids)
            assay_ids = sorted(
                {
                    str(activity["assay_chembl_id"])
                    for activity in activities
                    if activity.get("assay_chembl_id")
                }
            )
            assays = await self.connectors.chembl.fetch_assays(assay_ids)

            compound_batch = chembl.normalize_molecules(
                molecules, retrieved_at=retrieved_at, dataset_version_id=None
            )
            accepted_compounds, rejected_compounds, compound_quality = (
                chembl.apply_compound_quality(compound_batch.rows)
            )

            dataset_version = await prov_repo.create_dataset_version(
                session,
                source_id=chembl.SOURCE_ID,
                dataset_name="bioactivity",
                version=release,
                content_hash=_hash_all([*accepted_compounds, *activities]),
                record_count=len(accepted_compounds) + len(activities) + len(assays),
                retrieved_at=retrieved_at,
                source_url=(f"https://www.ebi.ac.uk/chembl/target_report_card/{chembl_target_id}/"),
                retrieval_params={
                    "chembl_target_id": chembl_target_id,
                    "open_targets_target": target.target_key,
                    "activity_types": activity_types or list(chembl.POTENCY_TYPES),
                    "max_activities": max_activities,
                },
                quality={"compounds": compound_quality},
            )
            for row in accepted_compounds:
                row["dataset_version_id"] = dataset_version.id

            compound_ids = await chem_repo.upsert_compounds(session, accepted_compounds)
            compound_id_by_chembl = {
                row["chembl_id"]: compound_id
                for row, compound_id in zip(accepted_compounds, compound_ids, strict=False)
            }

            assay_rows = [
                row
                for row in (
                    chembl.normalize_assay(
                        assay,
                        target_id=target.id,
                        retrieved_at=retrieved_at,
                        dataset_version_id=dataset_version.id,
                    )
                    for assay in assays
                )
                if row is not None
            ]
            written_assay_ids = await chem_repo.upsert_assays(session, assay_rows)
            assay_id_by_chembl = {
                row["chembl_assay_id"]: assay_id
                for row, assay_id in zip(assay_rows, written_assay_ids, strict=False)
            }

            activity_batch = chembl.normalize_activities(
                activities,
                compound_id_by_chembl=compound_id_by_chembl,
                assay_id_by_chembl=assay_id_by_chembl,
                target_id_by_chembl={chembl_target_id: target.id},
                retrieved_at=retrieved_at,
                dataset_version_id=dataset_version.id,
            )
            accepted_activities, rejected_activities, activity_quality = (
                chembl.apply_activity_quality(activity_batch.rows)
            )
            written_activities = await chem_repo.upsert_bioactivities(session, accepted_activities)

            dataset_version.quality = {
                **dataset_version.quality,
                "activities": activity_quality,
            }

            feature_count = 0
            if featurize and accepted_compounds:
                feature_count = await self._featurize_compounds(
                    session,
                    [
                        (compound_id, row["canonical_smiles"])
                        for row, compound_id in zip(accepted_compounds, compound_ids, strict=False)
                    ],
                )

            rejects = [
                *rejected_compounds,
                *rejected_activities,
                *activity_batch.rejects,
                *compound_batch.rejects,
            ]
            INGESTION_RECORDS.labels(source=chembl.SOURCE_ID, entity="compound").inc(
                len(compound_ids)
            )
            INGESTION_RECORDS.labels(source=chembl.SOURCE_ID, entity="bioactivity").inc(
                len(written_activities)
            )
            INGESTION_RECORDS.labels(source=chembl.SOURCE_ID, entity="assay").inc(
                len(written_assay_ids)
            )
            for reject in rejects:
                for reason in reject.get("_quality_errors", ["unknown"]):
                    INGESTION_REJECTS.labels(
                        source=chembl.SOURCE_ID, reason=str(reason).split(":")[0]
                    ).inc()

            await prov_repo.finish_ingestion_run(
                session,
                run,
                status="succeeded",
                dataset_version_id=dataset_version.id,
                records_fetched=len(activities) + len(molecules) + len(assays),
                records_written=len(compound_ids)
                + len(written_activities)
                + len(written_assay_ids),
                records_rejected=len(rejects),
                rejects=rejects,
            )
            return IngestionOutcome(
                source_id=chembl.SOURCE_ID,
                dataset_name="bioactivity",
                version=release,
                mode=self.mode,
                retrieved_at=retrieved_at,
                content_hash=dataset_version.content_hash,
                records_fetched=len(activities) + len(molecules) + len(assays),
                records_written=len(compound_ids) + len(written_activities),
                records_rejected=len(rejects),
                dataset_version_id=dataset_version.id,
                entity_counts={
                    "compound": len(compound_ids),
                    "assay": len(written_assay_ids),
                    "bioactivity": len(written_activities),
                    "featurized": feature_count,
                },
                quality=dataset_version.quality,
                rejects=rejects[:20],
            )
        except Exception as exc:
            await prov_repo.finish_ingestion_run(
                session, run, status="failed", error=str(exc)[:4000]
            )
            raise

    async def ingest_similar_compounds(
        self,
        session: AsyncSession,
        *,
        smiles: str,
        threshold_percent: int = 70,
        limit: int = 30,
        featurize: bool = True,
    ) -> IngestionOutcome:
        """Pull structural neighbours from ChEMBL that are not yet in the local store.

        Local similarity search only sees ingested compounds; this expands the local
        neighbourhood on demand so a researcher can explore beyond what was seeded.
        """
        run = await prov_repo.start_ingestion_run(
            session,
            source_id=chembl.SOURCE_ID,
            dataset_name="similar_compounds",
            mode=str(self.mode),
            params={"threshold_percent": threshold_percent, "limit": limit},
        )
        try:
            release = await self.connectors.chembl.version()
            retrieved_at = utcnow()
            molecules = await self.connectors.chembl.similar_molecules(
                smiles, threshold_percent=threshold_percent, limit=limit
            )
            batch = chembl.normalize_molecules(
                molecules, retrieved_at=retrieved_at, dataset_version_id=None
            )
            accepted, rejected, quality = chembl.apply_compound_quality(batch.rows)

            dataset_version = await prov_repo.create_dataset_version(
                session,
                source_id=chembl.SOURCE_ID,
                dataset_name="similar_compounds",
                version=release,
                content_hash=_hash_all(accepted),
                record_count=len(accepted),
                retrieved_at=retrieved_at,
                retrieval_params={
                    "query_smiles": smiles,
                    "threshold_percent": threshold_percent,
                    "limit": limit,
                },
                quality={"compounds": quality},
                make_current=False,
            )
            for row in accepted:
                row["dataset_version_id"] = dataset_version.id
            compound_ids = await chem_repo.upsert_compounds(session, accepted)

            feature_count = 0
            if featurize and accepted:
                feature_count = await self._featurize_compounds(
                    session,
                    [
                        (compound_id, row["canonical_smiles"])
                        for row, compound_id in zip(accepted, compound_ids, strict=False)
                    ],
                )

            await prov_repo.finish_ingestion_run(
                session,
                run,
                status="succeeded",
                dataset_version_id=dataset_version.id,
                records_fetched=len(molecules),
                records_written=len(compound_ids),
                records_rejected=len(rejected) + len(batch.rejects),
                rejects=[*rejected, *batch.rejects],
            )
            return IngestionOutcome(
                source_id=chembl.SOURCE_ID,
                dataset_name="similar_compounds",
                version=release,
                mode=self.mode,
                retrieved_at=retrieved_at,
                content_hash=dataset_version.content_hash,
                records_fetched=len(molecules),
                records_written=len(compound_ids),
                records_rejected=len(rejected) + len(batch.rejects),
                dataset_version_id=dataset_version.id,
                entity_counts={"compound": len(compound_ids), "featurized": feature_count},
                quality=dataset_version.quality,
            )
        except Exception as exc:
            await prov_repo.finish_ingestion_run(
                session, run, status="failed", error=str(exc)[:4000]
            )
            raise

    async def enrich_with_pubchem(
        self, session: AsyncSession, *, compound_ids: list[str], limit: int = 50
    ) -> dict[str, Any]:
        """Backfill PubChem CIDs and PubChem-reported properties for known compounds."""
        compounds = await chem_repo.get_compounds_by_ids(session, compound_ids[:limit])
        retrieved_at = utcnow()
        resolved = 0
        property_rows: list[dict[str, Any]] = []
        cid_by_compound: dict[str, str] = {}

        for compound in compounds:
            cid = compound.pubchem_cid
            if not cid:
                try:
                    cid = await self.connectors.pubchem.cid_for_inchikey(compound.inchikey)
                except UpstreamError:
                    cid = None
            if cid:
                cid_by_compound[compound.id] = cid
                resolved += 1

        if cid_by_compound:
            properties = await self.connectors.pubchem.properties(list(cid_by_compound.values()))
            by_cid = {str(row["CID"]): row for row in properties if row.get("CID")}
            for compound_id, cid in cid_by_compound.items():
                payload = by_cid.get(str(cid))
                if payload is None:
                    continue
                row = pubchem.normalize_reported_properties(payload, compound_id=compound_id)
                if row is not None:
                    property_rows.append(row)

        if property_rows:
            await chem_repo.upsert_properties(session, property_rows)
            INGESTION_RECORDS.labels(source=pubchem.SOURCE_ID, entity="molecular_property").inc(
                len(property_rows)
            )

        return {
            "compounds_examined": len(compounds),
            "cids_resolved": resolved,
            "properties_written": len(property_rows),
            "retrieved_at": retrieved_at.isoformat(),
        }

    async def _featurize_compounds(
        self, session: AsyncSession, compounds: list[tuple[str, str]]
    ) -> int:
        """Compute and persist descriptors + fingerprints for freshly ingested rows."""
        if not compounds:
            return 0
        result = featurize_many(compounds)
        if result.properties:
            await chem_repo.upsert_properties(session, result.properties)
        if result.fingerprints:
            await chem_repo.upsert_fingerprints(session, result.fingerprints)
        if result.failures:
            log.warning("ingestion.featurization_failures", count=len(result.failures))
        return result.succeeded

    async def backfill_features(self, session: AsyncSession, *, limit: int = 500) -> dict[str, Any]:
        """Featurise any compound missing descriptors or fingerprints."""
        missing_properties = await chem_repo.compounds_missing_properties(session, limit=limit)
        missing_fingerprints = await chem_repo.compounds_missing_fingerprint(
            session, kind="morgan_r2_2048", limit=limit
        )
        pending = {
            compound.id: compound.canonical_smiles
            for compound in [*missing_properties, *missing_fingerprints]
        }
        count = await self._featurize_compounds(session, list(pending.items()))
        return {"candidates": len(pending), "featurized": count}

    # --------------------------------------------------------------- moleculenet
    async def ingest_moleculenet(
        self, session: AsyncSession, *, dataset_key: str, force_download: bool = False
    ) -> IngestionOutcome:
        """Snapshot a MoleculeNet benchmark and register its dataset version."""
        spec = moleculenet.get_spec(dataset_key)
        run = await prov_repo.start_ingestion_run(
            session,
            source_id=moleculenet.SOURCE_ID,
            dataset_name=spec.dataset_name,
            mode=str(self.mode),
            params={"dataset": dataset_key},
        )
        try:
            retrieved_at = utcnow()
            raw = await self.connectors.moleculenet.download(spec, force=force_download)
            dataset = moleculenet.parse_dataset(
                spec,
                raw,
                retrieved_at=retrieved_at,
                source_url=f"{self.settings.moleculenet_base_url}/{spec.filename}",
            )
            version = f"{version_label(retrieved_at)}+{dataset.raw_bytes_hash}"
            snapshot = moleculenet.snapshot_path(self.settings.data_dir / "raw", spec, version)
            moleculenet.write_snapshot(dataset, snapshot)

            dataset_version = await prov_repo.create_dataset_version(
                session,
                source_id=moleculenet.SOURCE_ID,
                dataset_name=spec.dataset_name,
                version=version,
                content_hash=dataset.content_hash,
                record_count=len(dataset),
                retrieved_at=retrieved_at,
                source_url=dataset.source_url,
                retrieval_params={
                    "dataset": dataset_key,
                    "endpoint": spec.endpoint,
                    "task_type": spec.task_type,
                    "label_column": spec.label_column,
                    "snapshot_path": str(
                        snapshot.relative_to(self.settings.data_dir.parent)
                        if self.settings.data_dir.parent in snapshot.parents
                        else snapshot
                    ),
                    "raw_bytes_hash": dataset.raw_bytes_hash,
                },
                quality=dataset.quality(),
                notes=spec.description,
            )
            INGESTION_RECORDS.labels(source=moleculenet.SOURCE_ID, entity="training_row").inc(
                len(dataset)
            )

            await prov_repo.finish_ingestion_run(
                session,
                run,
                status="succeeded",
                dataset_version_id=dataset_version.id,
                records_fetched=len(dataset) + len(dataset.rejects),
                records_written=len(dataset),
                records_rejected=len(dataset.rejects),
                rejects=dataset.rejects,
            )
            return IngestionOutcome(
                source_id=moleculenet.SOURCE_ID,
                dataset_name=spec.dataset_name,
                version=version,
                mode=self.mode,
                retrieved_at=retrieved_at,
                content_hash=dataset.content_hash,
                records_fetched=len(dataset) + len(dataset.rejects),
                records_written=len(dataset),
                records_rejected=len(dataset.rejects),
                dataset_version_id=dataset_version.id,
                entity_counts={"training_row": len(dataset)},
                quality=dataset.quality(),
            )
        except Exception as exc:
            await prov_repo.finish_ingestion_run(
                session, run, status="failed", error=str(exc)[:4000]
            )
            raise


def _hash_all(records: list[Any]) -> str:
    from theragraph.core.hashing import hash_records

    normalized = [record if isinstance(record, dict) else {"value": record} for record in records]
    return hash_records(normalized)


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def resolve_snapshot_path(settings: Settings, retrieval_params: dict[str, Any]) -> Path:
    """Locate a MoleculeNet snapshot recorded on a dataset version."""
    raw = retrieval_params.get("snapshot_path")
    if not raw:
        raise ValueError("dataset version has no recorded snapshot path")
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (settings.data_dir.parent, settings.data_dir, Path.cwd()):
        resolved = base / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"snapshot not found: {raw}")
