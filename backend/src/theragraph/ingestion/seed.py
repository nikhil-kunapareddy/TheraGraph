"""Demo seeding: build a small but realistic database from public sources.

The seed is deliberately narrow and deep rather than broad and shallow. Three disease
areas with different chemistry profiles cover the interesting cases:

* **Alzheimer disease** -- heavily studied, strong genetics, mixed small-molecule and
  biologic modalities, and several targets with *no* tractable small-molecule chemistry.
  That last point matters: the product must handle "no compounds found" honestly.
* **Type 2 diabetes mellitus** -- classic small-molecule targets with rich ChEMBL
  bioactivity, so similarity search and ranking have something to chew on.
* **Non-small cell lung carcinoma** -- kinase targets with approved drugs, which
  exercises the known-drug evidence path end to end.

Seeding is idempotent: running it twice updates rows and produces the same content
hashes, so a re-seed is safe and diffable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.core.clock import Stopwatch
from theragraph.core.logging import get_logger
from theragraph.db.repositories import biology as biology_repo
from theragraph.ingestion.opentargets import DEFAULT_EVIDENCE_DATASOURCES
from theragraph.ingestion.pipelines import IngestionService, ensure_data_sources

log = get_logger(__name__)


@dataclass(frozen=True)
class SeedDisease:
    efo_id: str
    label: str
    n_targets: int = 10
    bioactivity_targets: int = 4
    max_activities: int = 250
    notes: str = ""


@dataclass(frozen=True)
class SeedPlan:
    diseases: list[SeedDisease]
    moleculenet_datasets: list[str] = field(default_factory=list)
    evidence_datasources: list[str] = field(default_factory=list)
    per_datasource_evidence: int = 8
    enrich_pubchem: bool = True

    @property
    def datasources(self) -> list[str]:
        return self.evidence_datasources or DEFAULT_EVIDENCE_DATASOURCES


DEMO_PLAN = SeedPlan(
    diseases=[
        SeedDisease(
            efo_id="MONDO_0004975",
            label="Alzheimer disease",
            n_targets=12,
            bioactivity_targets=5,
            max_activities=300,
            notes="Strong genetics; several targets lack tractable small-molecule chemistry.",
        ),
        SeedDisease(
            efo_id="MONDO_0005148",
            label="type 2 diabetes mellitus",
            n_targets=12,
            bioactivity_targets=5,
            max_activities=300,
            notes="Rich ChEMBL bioactivity for classic small-molecule targets.",
        ),
        SeedDisease(
            efo_id="MONDO_0005233",
            label="non-small cell lung carcinoma",
            n_targets=10,
            bioactivity_targets=4,
            max_activities=250,
            notes="Kinase targets with approved drugs; exercises known-drug evidence.",
        ),
    ],
    moleculenet_datasets=["esol", "bbbp", "lipophilicity", "clintox"],
    per_datasource_evidence=8,
)

SMOKE_PLAN = SeedPlan(
    diseases=[
        SeedDisease(
            efo_id="MONDO_0004975",
            label="Alzheimer disease",
            n_targets=3,
            bioactivity_targets=1,
            max_activities=40,
        )
    ],
    moleculenet_datasets=["esol"],
    evidence_datasources=["chembl", "europepmc", "eva"],
    per_datasource_evidence=3,
    enrich_pubchem=False,
)


async def run_seed(
    session: AsyncSession,
    service: IngestionService,
    plan: SeedPlan = DEMO_PLAN,
    *,
    commit: bool = True,
) -> dict[str, Any]:
    """Execute a seed plan, returning a per-step report."""
    report: dict[str, Any] = {"diseases": [], "moleculenet": [], "errors": []}
    stopwatch = Stopwatch()

    with stopwatch:
        await ensure_data_sources(session)
        if commit:
            await session.commit()

        for seed in plan.diseases:
            entry: dict[str, Any] = {"efo_id": seed.efo_id, "label": seed.label}
            try:
                disease_outcome = await service.ingest_disease(
                    session, seed.efo_id, n_targets=seed.n_targets
                )
                if commit:
                    await session.commit()
                entry["disease"] = disease_outcome.summary()

                disease = await biology_repo.get_disease(session, seed.efo_id)
                assert disease is not None
                associated = await biology_repo.targets_for_disease(
                    session, disease.id, limit=seed.n_targets
                )
                target_keys = [target.target_key for target, _ in associated]

                evidence_outcome = await service.ingest_evidence(
                    session,
                    efo_id=seed.efo_id,
                    target_keys=target_keys,
                    datasource_ids=plan.datasources,
                    per_datasource=plan.per_datasource_evidence,
                )
                if commit:
                    await session.commit()
                entry["evidence"] = evidence_outcome.summary()

                chemistry: list[dict[str, Any]] = []
                for target, _ in associated[: seed.bioactivity_targets]:
                    try:
                        outcome = await service.ingest_target_bioactivity(
                            session,
                            target_key=target.target_key,
                            max_activities=seed.max_activities,
                        )
                        if commit:
                            await session.commit()
                        chemistry.append(
                            {
                                "target": target.approved_symbol,
                                "counts": outcome.entity_counts,
                                "warnings": outcome.warnings,
                            }
                        )
                    except Exception as exc:  # one target must not fail the seed
                        log.warning(
                            "seed.bioactivity_failed",
                            target=target.approved_symbol,
                            error=str(exc),
                        )
                        report["errors"].append(
                            {
                                "step": "bioactivity",
                                "target": target.approved_symbol,
                                "error": str(exc)[:300],
                            }
                        )
                entry["chemistry"] = chemistry

            except Exception as exc:
                log.warning("seed.disease_failed", efo_id=seed.efo_id, error=str(exc))
                report["errors"].append(
                    {"step": "disease", "efo_id": seed.efo_id, "error": str(exc)[:300]}
                )
            report["diseases"].append(entry)

        for dataset_key in plan.moleculenet_datasets:
            try:
                outcome = await service.ingest_moleculenet(session, dataset_key=dataset_key)
                if commit:
                    await session.commit()
                report["moleculenet"].append(outcome.summary())
            except Exception as exc:
                log.warning("seed.moleculenet_failed", dataset=dataset_key, error=str(exc))
                report["errors"].append(
                    {"step": "moleculenet", "dataset": dataset_key, "error": str(exc)[:300]}
                )

        backfill = await service.backfill_features(session, limit=2000)
        if commit:
            await session.commit()
        report["feature_backfill"] = backfill

    report["runtime_ms"] = stopwatch.elapsed_ms
    report["fixtures"] = service.connectors.recorder.stats()
    return report
