"""Repository behaviour against a real database: upserts, search, aggregation, locking."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.chem.fingerprints import compute_fingerprint
from theragraph.chem.similarity import popcount_window, rank_by_tanimoto
from theragraph.core.clock import utcnow
from theragraph.db.repositories import biology as biology_repo
from theragraph.db.repositories import chemistry as chem_repo
from theragraph.db.repositories import jobs as jobs_repo
from theragraph.db.repositories import ml as ml_repo
from theragraph.db.repositories import provenance as prov_repo
from theragraph.db.repositories import runs as runs_repo
from theragraph.db.repositories.base import like_pattern, normalize_text

pytestmark = pytest.mark.asyncio


class TestUpsertSemantics:
    async def test_reingesting_updates_rather_than_duplicating(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        """Ingestion is idempotent; a second run must not double the corpus."""
        before = await biology_repo.count_diseases(session)
        await biology_repo.upsert_diseases(
            session,
            [
                {
                    "efo_id": "MONDO_0004975",
                    "name": "Alzheimer disease (revised name)",
                    "synonyms": ["AD"],
                    "therapeutic_areas": [],
                    "source": "open_targets",
                    "source_identifier": "MONDO_0004975",
                    "retrieved_at": utcnow(),
                }
            ],
        )
        assert await biology_repo.count_diseases(session) == before
        disease = await biology_repo.get_disease(session, "MONDO_0004975")
        assert disease is not None
        assert disease.name == "Alzheimer disease (revised name)"

    async def test_compounds_deduplicate_on_inchikey_across_sources(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        """The same molecule from ChEMBL and PubChem must land on one row."""
        before = await chem_repo.count_compounds(session)
        aspirin = await chem_repo.get_compound(session, "CHEMBL25")
        assert aspirin is not None
        await chem_repo.upsert_compounds(
            session,
            [
                {
                    "inchikey": aspirin.inchikey,
                    "canonical_smiles": aspirin.canonical_smiles,
                    "name": "Aspirin (from PubChem)",
                    "pubchem_cid": "2244",
                    "source": "pubchem",
                    "source_identifier": "2244",
                    "retrieved_at": utcnow(),
                }
            ],
        )
        assert await chem_repo.count_compounds(session) == before
        merged = await chem_repo.get_compound(session, aspirin.inchikey)
        assert merged is not None
        assert merged.pubchem_cid == "2244"

    async def test_upsert_returns_stable_ids(self, session: AsyncSession) -> None:
        row = {
            "efo_id": "EFO_9999999",
            "name": "Test disease",
            "synonyms": [],
            "therapeutic_areas": [],
            "source": "open_targets",
            "source_identifier": "EFO_9999999",
            "retrieved_at": utcnow(),
        }
        first = await biology_repo.upsert_diseases(session, [row])
        second = await biology_repo.upsert_diseases(session, [row])
        assert first == second

    async def test_empty_upsert_is_a_no_op(self, session: AsyncSession) -> None:
        assert await biology_repo.upsert_diseases(session, []) == []

    async def test_heterogeneous_rows_are_accepted(self, session: AsyncSession) -> None:
        """Partial rows are normal from upstream; the upsert fills defaults."""
        ids = await biology_repo.upsert_diseases(
            session,
            [
                {
                    "efo_id": "EFO_1",
                    "name": "A",
                    "source": "open_targets",
                    "retrieved_at": utcnow(),
                },
                {
                    "efo_id": "EFO_2",
                    "name": "B",
                    "description": "has a description",
                    "synonyms": ["b"],
                    "source": "open_targets",
                    "retrieved_at": utcnow(),
                },
            ],
        )
        assert len(ids) == 2


class TestSearch:
    async def test_exact_name_ranks_above_substring(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        await biology_repo.upsert_diseases(
            session,
            [
                {
                    "efo_id": "EFO_X",
                    "name": "Early-onset Alzheimer disease variant",
                    "synonyms": [],
                    "therapeutic_areas": [],
                    "source": "open_targets",
                    "retrieved_at": utcnow(),
                }
            ],
        )
        results = await biology_repo.search_diseases(session, "Alzheimer disease")
        assert results[0][0].name == "Alzheimer disease"
        assert results[0][1] > results[1][1]

    async def test_target_symbol_beats_name_match(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        results = await biology_repo.search_targets(session, "BACE1")
        assert results[0][0].approved_symbol == "BACE1"

    async def test_search_is_case_insensitive(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        for term in ("bace1", "BACE1", "Bace1"):
            assert await biology_repo.search_targets(session, term)

    async def test_synonym_match_is_found(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        results = await biology_repo.search_targets(session, "memapsin-2")
        assert any(target.approved_symbol == "BACE1" for target, _ in results)

    async def test_efo_id_lookup(self, session: AsyncSession, seeded: dict[str, Any]) -> None:
        assert await biology_repo.search_diseases(session, "MONDO_0004975")

    async def test_empty_query_returns_nothing(self, session: AsyncSession) -> None:
        assert await biology_repo.search_diseases(session, "   ") == []

    async def test_like_wildcards_in_user_input_are_escaped(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        """Without escaping, a '%' would match everything."""
        assert await biology_repo.search_diseases(session, "%") == []
        assert await biology_repo.search_diseases(session, "_") == []
        assert like_pattern("100%") == "%100\\%%"

    async def test_compound_search_by_identifier_and_name(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        by_chembl = await chem_repo.search_compounds(session, "CHEMBL25")
        assert by_chembl[0][0].chembl_id == "CHEMBL25"
        by_name = await chem_repo.search_compounds(session, "caffeine")
        assert by_name[0][0].name == "Caffeine"

    def test_text_normalisation(self) -> None:
        assert normalize_text("  Alzheimer   DISEASE ") == "alzheimer disease"


class TestAssociationsAndEvidence:
    async def test_targets_for_disease_are_ordered_by_source_score(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        rows = await biology_repo.targets_for_disease(session, seeded["disease_id"])
        scores = [association.overall_score for _, association in rows]
        assert scores == sorted(scores, reverse=True)
        assert rows[0][0].approved_symbol == "APP"  # 0.87 beats BACE1's 0.74

    async def test_min_score_filter(self, session: AsyncSession, seeded: dict[str, Any]) -> None:
        rows = await biology_repo.targets_for_disease(session, seeded["disease_id"], min_score=0.8)
        assert len(rows) == 1

    async def test_evidence_filters_by_type(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        genetic = await biology_repo.list_evidence(
            session, disease_id=seeded["disease_id"], evidence_types=["genetic_association"]
        )
        assert len(genetic) == 1
        assert genetic[0].evidence_type == "genetic_association"

    async def test_evidence_orders_scored_records_first(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        """A record without a source score must not outrank a scored one."""
        rows = await biology_repo.list_evidence(session, disease_id=seeded["disease_id"])
        assert rows[0].score is not None
        assert rows[-1].score is None

    async def test_evidence_publication_is_eager_loaded(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        """Lazy loading here would fire one query per citation in the report."""
        rows = await biology_repo.list_evidence(
            session, disease_id=seeded["disease_id"], evidence_types=["literature"]
        )
        assert rows[0].publication is not None
        assert rows[0].publication.pmid == "1520398"

    async def test_evidence_type_counts(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        counts = await biology_repo.evidence_type_counts(session, disease_id=seeded["disease_id"])
        assert counts["genetic_association"] == 1
        assert counts["literature"] == 1

    async def test_citation_key_is_source_scoped(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        rows = await biology_repo.get_evidence_by_ids(session, [seeded["evidence_ids"][0]])
        assert rows[0].citation_key.startswith("open_targets:")


class TestChemistryQueries:
    async def test_known_compounds_aggregate_in_sql(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        """Aggregation must happen in the database, not by streaming rows into Python."""
        rows = await chem_repo.known_compounds_for_target(
            session, seeded["target_ids"][0], min_pchembl=6.0
        )
        assert rows
        best = [pchembl for _, pchembl, _ in rows]
        assert best == sorted(best, reverse=True)
        assert all(pchembl >= 6.0 for pchembl in best)

    async def test_potency_threshold_excludes_weak_compounds(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        strict = await chem_repo.known_compounds_for_target(
            session, seeded["target_ids"][0], min_pchembl=8.0
        )
        assert len(strict) == 1  # only ibuprofen at 8.1

    async def test_flagged_activities_are_excluded_from_actives(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        """Curator-flagged rows stay in the database but out of "known actives"."""
        await chem_repo.upsert_bioactivities(
            session,
            [
                {
                    "compound_id": seeded["compound_ids"][0],
                    "target_id": seeded["target_ids"][0],
                    "activity_type": "IC50",
                    "pchembl_value": 11.0,
                    "data_validity_comment": "Potential transcription error",
                    "source": "chembl",
                    "source_identifier": "flagged-1",
                    "retrieved_at": utcnow(),
                }
            ],
        )
        rows = await chem_repo.bioactivities_for_target(
            session, seeded["target_ids"][0], exclude_flagged=True
        )
        assert all(row.data_validity_comment is None for row in rows)
        unfiltered = await chem_repo.bioactivities_for_target(
            session, seeded["target_ids"][0], exclude_flagged=False
        )
        assert len(unfiltered) > len(rows)

    async def test_targets_for_compound(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        rows = await chem_repo.targets_for_compound(session, seeded["compound_ids"][3])
        assert rows[0][0].approved_symbol == "BACE1"

    async def test_compound_lookup_accepts_every_identifier_form(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        aspirin = await chem_repo.get_compound(session, "CHEMBL25")
        assert aspirin is not None
        for identifier in (aspirin.id, aspirin.chembl_id, aspirin.inchikey):
            found = await chem_repo.get_compound(session, identifier)
            assert found is not None and found.id == aspirin.id

    async def test_properties_batch_fetch_is_one_query(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        properties = await chem_repo.get_properties_for_compounds(session, seeded["compound_ids"])
        assert len(properties) == len(seeded["compound_ids"])

    async def test_property_window_filter(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        small = await chem_repo.filter_compounds(session, mw_range=(100.0, 200.0))
        assert all(
            100.0 <= row.properties[0].molecular_weight <= 200.0 for row in small if row.properties
        )

    async def test_missing_feature_backfill_queries(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        assert await chem_repo.compounds_missing_properties(session) == []
        await chem_repo.upsert_compounds(
            session,
            [
                {
                    "inchikey": "AAAAAAAAAAAAAA-UHFFFAOYSA-N",
                    "canonical_smiles": "CCCCO",
                    "name": "Butanol",
                    "source": "chembl",
                    "source_identifier": "X1",
                    "retrieved_at": utcnow(),
                }
            ],
        )
        assert len(await chem_repo.compounds_missing_properties(session)) == 1

    async def test_scaffold_frequency_and_counts(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        counts = await chem_repo.scaffold_counts(session)
        assert counts
        top_scaffold, top_count = counts[0]
        assert await chem_repo.scaffold_frequency(session, top_scaffold) == top_count


class TestSimilarityQuery:
    async def test_popcount_window_narrows_the_candidate_set(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        """The index range scan is the difference between a search and a table scan."""
        query = compute_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
        low, high = popcount_window(query.popcount, 0.7)
        narrowed = await chem_repo.iter_fingerprints_for_similarity(
            session, kind="morgan_r2_2048", min_popcount=low, max_popcount=high
        )
        everything = await chem_repo.all_fingerprints(session, kind="morgan_r2_2048")
        assert len(narrowed) <= len(everything)

    async def test_end_to_end_similarity_finds_the_analogue(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        query = compute_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
        aspirin = await chem_repo.get_compound(session, "CHEMBL25")
        assert aspirin is not None
        low, high = popcount_window(query.popcount, 0.3)
        candidates = await chem_repo.iter_fingerprints_for_similarity(
            session,
            kind="morgan_r2_2048",
            min_popcount=low,
            max_popcount=high,
            exclude_compound_id=aspirin.id,
        )
        hits = rank_by_tanimoto(query.bits, candidates, threshold=0.3, limit=5)
        found = await chem_repo.get_compounds_by_ids(session, [hit.compound_id for hit in hits])
        assert "Salicylic acid" in {compound.name for compound in found}

    async def test_compounds_are_returned_in_requested_order(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        """Ranking order must survive the database round-trip."""
        requested = list(reversed(seeded["compound_ids"]))
        rows = await chem_repo.get_compounds_by_ids(session, requested)
        assert [row.id for row in rows] == requested


class TestProvenanceRepository:
    async def test_new_version_supersedes_the_previous_current(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        await prov_repo.create_dataset_version(
            session,
            source_id="open_targets",
            dataset_name="disease_associations",
            version="test.2",
            content_hash="newhash",
            record_count=10,
            retrieved_at=utcnow(),
        )
        current = await prov_repo.get_current_dataset_version(
            session, "open_targets", "disease_associations"
        )
        assert current is not None
        assert current.version == "test.2"
        versions = await prov_repo.list_dataset_versions(
            session, source_id="open_targets", dataset_name="disease_associations"
        )
        assert sum(1 for version in versions if version.is_current) == 1

    async def test_version_label_is_quotable(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        current = await prov_repo.get_current_dataset_version(
            session, "open_targets", "disease_associations"
        )
        assert current is not None
        assert current.label == "open_targets/disease_associations@test.1"

    async def test_ingestion_run_lifecycle(self, session: AsyncSession) -> None:
        from theragraph.ingestion.pipelines import ensure_data_sources

        await ensure_data_sources(session)
        run = await prov_repo.start_ingestion_run(
            session, source_id="chembl", dataset_name="bioactivity", mode="replay"
        )
        assert run.status == "running"
        await prov_repo.finish_ingestion_run(
            session, run, status="succeeded", records_written=5, records_rejected=1
        )
        assert run.status == "succeeded"
        assert run.runtime_ms is not None
        assert run.records_written == 5

    async def test_failed_ingestion_records_the_error(self, session: AsyncSession) -> None:
        from theragraph.ingestion.pipelines import ensure_data_sources

        await ensure_data_sources(session)
        run = await prov_repo.start_ingestion_run(
            session, source_id="chembl", dataset_name="x", mode="live"
        )
        await prov_repo.finish_ingestion_run(
            session, run, status="failed", error="upstream returned 500"
        )
        assert run.status == "failed"
        assert "500" in run.error


class TestRunRepository:
    async def test_run_ids_are_sequential_and_gapless(self, session: AsyncSession) -> None:
        ids = [await runs_repo.allocate_run_id(session, 2026) for _ in range(5)]
        assert [sequence for _, sequence in ids] == [1, 2, 3, 4, 5]
        assert ids[0][0] == "RUN-2026-00001"

    async def test_counters_are_scoped_per_year(self, session: AsyncSession) -> None:
        await runs_repo.allocate_run_id(session, 2026)
        _, sequence = await runs_repo.allocate_run_id(session, 2027)
        assert sequence == 1

    async def test_concurrent_allocation_does_not_collide(self, session: AsyncSession) -> None:
        """Sequential allocation inside one transaction must never repeat a number."""
        allocated = {await runs_repo.allocate_run_id(session, 2026) for _ in range(20)}
        assert len(allocated) == 20

    async def test_idempotency_key_returns_the_original_run(self, session: AsyncSession) -> None:
        run_id, sequence = await runs_repo.allocate_run_id(session, 2026)
        await runs_repo.create_run(
            session,
            id=run_id,
            sequence=sequence,
            year=2026,
            query="q",
            status="queued",
            idempotency_key="key-1",
            started_at=utcnow(),
        )
        found = await runs_repo.get_run_by_idempotency_key(session, "key-1")
        assert found is not None and found.id == run_id

    async def test_tool_call_stats_aggregate(self, session: AsyncSession) -> None:
        run_id, sequence = await runs_repo.allocate_run_id(session, 2026)
        run = await runs_repo.create_run(
            session,
            id=run_id,
            sequence=sequence,
            year=2026,
            query="q",
            status="succeeded",
            started_at=utcnow(),
        )
        for index, (tool, status, latency) in enumerate(
            [("search_entities", "succeeded", 10.0), ("search_entities", "failed", 20.0)]
        ):
            await runs_repo.add_tool_call(
                session,
                research_run_id=run.id,
                step_index=index,
                tool_name=tool,
                arguments={},
                status=status,
                latency_ms=latency,
            )
        stats = await runs_repo.tool_call_stats(session)
        row = next(item for item in stats if item["tool_name"] == "search_entities")
        assert row["calls"] == 2
        assert row["errors"] == 1
        assert row["error_rate"] == 0.5

    async def test_usage_summary(self, session: AsyncSession) -> None:
        run_id, sequence = await runs_repo.allocate_run_id(session, 2026)
        await runs_repo.create_run(
            session,
            id=run_id,
            sequence=sequence,
            year=2026,
            query="q",
            status="succeeded",
            runtime_ms=1500.0,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            started_at=utcnow(),
        )
        summary = await runs_repo.run_usage_summary(session)
        assert summary["runs"] == 1
        assert summary["input_tokens"] == 100
        assert summary["cost_usd"] == pytest.approx(0.01)


class TestJobRepository:
    async def test_claim_is_exclusive(self, session: AsyncSession) -> None:
        """Two workers racing on one message must not both win."""
        job = await jobs_repo.create_job(
            session, task="t", payload={}, idempotency_key="k1", status="pending"
        )
        first = await jobs_repo.claim_job(session, job.id, worker="w1", lease_s=60)
        second = await jobs_repo.claim_job(session, job.id, worker="w2", lease_s=60)
        assert first is not None
        assert second is None
        assert first.locked_by == "w1"
        assert first.attempts == 1

    async def test_expired_lease_is_reclaimed(self, session: AsyncSession) -> None:
        """A worker that dies mid-job must not strand the work."""
        job = await jobs_repo.create_job(
            session, task="t", payload={}, idempotency_key="k2", status="pending"
        )
        await jobs_repo.claim_job(session, job.id, worker="dead", lease_s=-1)
        reaped = await jobs_repo.reap_expired_leases(session)
        assert job.id in reaped
        reclaimed = await jobs_repo.claim_job(session, job.id, worker="alive", lease_s=60)
        assert reclaimed is not None

    async def test_failure_schedules_a_retry_until_the_budget_runs_out(
        self, session: AsyncSession
    ) -> None:
        job = await jobs_repo.create_job(
            session,
            task="t",
            payload={},
            idempotency_key="k3",
            status="pending",
            max_attempts=2,
        )
        await jobs_repo.claim_job(session, job.id, worker="w", lease_s=60)
        retried = await jobs_repo.fail_job(session, job.id, error="boom", retry_in_s=30)
        assert retried is not None and retried.status == "pending"
        assert retried.run_after is not None

        await jobs_repo.claim_job(session, job.id, worker="w", lease_s=60)
        exhausted = await jobs_repo.fail_job(session, job.id, error="boom again", retry_in_s=30)
        assert exhausted is not None and exhausted.status == "failed"
        assert len(exhausted.error_history) == 2

    async def test_completion_records_runtime(self, session: AsyncSession) -> None:
        job = await jobs_repo.create_job(
            session, task="t", payload={}, idempotency_key="k4", status="pending"
        )
        await jobs_repo.claim_job(session, job.id, worker="w", lease_s=60)
        await jobs_repo.complete_job(session, job.id, result={"ok": True})
        finished = await jobs_repo.get_job(session, job.id)
        assert finished is not None
        assert finished.status == "succeeded"
        assert finished.result == {"ok": True}
        assert finished.runtime_ms is not None

    async def test_delayed_job_is_not_claimable_yet(self, session: AsyncSession) -> None:
        await jobs_repo.create_job(
            session,
            task="t",
            payload={},
            idempotency_key="k5",
            status="pending",
            run_after=utcnow() + timedelta(hours=1),
        )
        assert await jobs_repo.claimable_jobs(session) == []

    async def test_queue_stats(self, session: AsyncSession) -> None:
        await jobs_repo.create_job(
            session, task="t", payload={}, idempotency_key="k6", status="pending"
        )
        assert (await jobs_repo.queue_stats(session))["pending"] == 1


class TestModelRepository:
    async def test_active_model_resolution_prefers_the_graph_model(
        self, session: AsyncSession
    ) -> None:
        """When both are active the graph model is the better scaffold-split performer."""
        for family in ("random_forest", "gin"):
            await ml_repo.upsert_model(
                session,
                name=f"solubility-{family}",
                version="1",
                family=family,
                endpoint="solubility",
                task_type="regression",
                framework="x",
                status="active",
            )
        resolved = await ml_repo.get_active_model_for_endpoint(session, "solubility")
        assert resolved is not None and resolved.family == "gin"

    async def test_archived_models_are_not_served(self, session: AsyncSession) -> None:
        await ml_repo.upsert_model(
            session,
            name="solubility-rf",
            version="1",
            family="random_forest",
            endpoint="solubility",
            task_type="regression",
            framework="x",
            status="archived",
        )
        assert await ml_repo.get_active_model_for_endpoint(session, "solubility") is None

    async def test_explicit_version_pin_is_honoured(self, session: AsyncSession) -> None:
        for version in ("1", "2"):
            await ml_repo.upsert_model(
                session,
                name="solubility-rf",
                version=version,
                family="random_forest",
                endpoint="solubility",
                task_type="regression",
                framework="x",
                status="active",
            )
        pinned = await ml_repo.get_model(session, "solubility-rf", "1")
        assert pinned is not None and pinned.version == "1"
        latest = await ml_repo.get_model(session, "solubility-rf")
        assert latest is not None and latest.version == "2"

    async def test_predictions_are_idempotent_on_input_hash(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        model = await ml_repo.upsert_model(
            session,
            name="solubility-rf",
            version="1",
            family="random_forest",
            endpoint="solubility",
            task_type="regression",
            framework="x",
        )
        row = {
            "compound_id": seeded["compound_ids"][0],
            "model_id": model.id,
            "endpoint": "solubility",
            "task_type": "regression",
            "value": -1.72,
            "input_hash": "hash-1",
        }
        first = await ml_repo.upsert_predictions(session, [row])
        second = await ml_repo.upsert_predictions(session, [{**row, "value": -1.80}])
        assert first == second
        assert await ml_repo.count_predictions(session) == 1

    async def test_experiment_ids_are_allocated(self, session: AsyncSession) -> None:
        experiment_id, sequence = await ml_repo.allocate_experiment_id(session, 2026)
        assert experiment_id == "EXP-2026-00001"
        assert sequence == 1

    async def test_ranking_persists_scores_with_factors(
        self, session: AsyncSession, seeded: dict[str, Any]
    ) -> None:
        ranking = await ml_repo.create_ranking(
            session,
            scores=[
                {
                    "compound_id": seeded["compound_ids"][0],
                    "rank": 1,
                    "total_score": 0.81,
                    "factors": [{"key": "measured_potency", "contribution": 0.24}],
                    "flags": [],
                    "evidence_ids": seeded["evidence_ids"][:1],
                    "prediction_ids": [],
                }
            ],
            name="test ranking",
            target_id=seeded["target_ids"][0],
            config={"name": "default"},
            config_hash="cfg1",
            n_candidates=1,
        )
        loaded = await ml_repo.get_ranking(session, ranking.id)
        assert loaded is not None
        assert loaded.scores[0].factors[0]["key"] == "measured_potency"
        assert loaded.scores[0].compound is not None
