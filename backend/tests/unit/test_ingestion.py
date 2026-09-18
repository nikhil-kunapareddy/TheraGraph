"""Ingestion normalisers and data-quality rules.

These tests run against *recorded shapes* of real upstream responses, so a change in
what the API returns is caught here rather than in production. No network.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from theragraph.core.errors import LiveIngestionDisabledError, ValidationError
from theragraph.ingestion.base import (
    IngestionMode,
    NormalizedBatch,
    ResponseRecorder,
    version_label,
)
from theragraph.ingestion.chembl import (
    apply_activity_quality,
    apply_compound_quality,
    normalize_activities,
    normalize_assay,
    normalize_molecule,
    normalize_molecules,
)
from theragraph.ingestion.europepmc import normalize_publication
from theragraph.ingestion.moleculenet import DATASETS, get_spec, parse_dataset
from theragraph.ingestion.opentargets import (
    DATASOURCE_TO_EVIDENCE_TYPE,
    normalize_associations,
    normalize_disease,
    normalize_evidence,
    normalize_gene,
    normalize_known_drugs,
    normalize_target,
)
from theragraph.ingestion.pubchem import normalize_compound, normalize_reported_properties
from theragraph.ingestion.quality import (
    QualityRule,
    evaluate_quality,
    in_range,
    required,
)

NOW = datetime(2026, 6, 1, tzinfo=UTC)

# Shapes recorded from the live Open Targets GraphQL API (release 26.06).
OT_DISEASE = {
    "id": "MONDO_0004975",
    "name": "Alzheimer disease",
    "description": "A progressive, neurodegenerative disease.",
    "synonyms": [
        {"relation": "hasExactSynonym", "terms": ["AD", "Alzheimer's disease"]},
        {"relation": "hasRelatedSynonym", "terms": []},
        {"relation": "hasNarrowSynonym", "terms": ["Alzheimer disease, familial"]},
    ],
    "therapeuticAreas": [{"id": "MONDO_0005071", "name": "nervous system disorder"}],
    "parents": [{"id": "MONDO_0001627", "name": "dementia"}],
    "associatedTargets": {
        "count": 13367,
        "rows": [
            {
                "score": 0.8722,
                "datatypeScores": [
                    {"id": "genetic_association", "score": 0.9229},
                    {"id": "literature", "score": 0.9972},
                ],
                "target": {
                    "id": "ENSG00000142192",
                    "approvedSymbol": "APP",
                    "approvedName": "amyloid beta precursor protein",
                    "biotype": "protein_coding",
                },
            }
        ],
    },
}

OT_TARGET = {
    "id": "ENSG00000142192",
    "approvedSymbol": "APP",
    "approvedName": "amyloid beta precursor protein",
    "biotype": "protein_coding",
    "functionDescriptions": ["Functions as a cell surface receptor."],
    "synonyms": [{"label": "A4", "source": "HGNC"}, {"label": "AD1", "source": "HGNC"}],
    "proteinIds": [
        {"id": "P05067", "source": "uniprot_swissprot"},
        {"id": "ENSP00000284981", "source": "ensembl_PRO"},
    ],
    "dbXrefs": [{"id": "CHEMBL2487", "source": "ChEMBL"}],
    "tractability": [
        {"label": "Approved Drug", "modality": "AB", "value": True},
        {"label": "Clinical Precedence", "modality": "SM", "value": False},
        {"label": "Predicted Tractable", "modality": "SM", "value": True},
    ],
    "subcellularLocations": [{"location": "Cell membrane", "termSL": "SL-0039"}],
    "safetyLiabilities": [
        {"event": "hepatotoxicity", "eventId": "EFO_0001", "datasource": "AOP", "literature": None}
    ],
    "targetClass": [{"id": 1, "label": "Other membrane protein", "level": "l1"}],
    "genomicLocation": {"chromosome": "21", "start": 25880550, "end": 26171128, "strand": -1},
    "drugAndClinicalCandidates": {
        "count": 2,
        "rows": [
            {
                "id": "row1",
                "maxClinicalStage": "APPROVAL",
                "drug": {
                    "id": "CHEMBL4297245",
                    "name": "DONANEMAB",
                    "drugType": "Antibody",
                    "maximumClinicalStage": "APPROVAL",
                    "mechanismsOfAction": {
                        "rows": [
                            {
                                "mechanismOfAction": "Amyloid-beta A4 protein disrupting agent",
                                "actionType": "DISRUPTER",
                            }
                        ]
                    },
                },
            }
        ],
    },
}

OT_EVIDENCE_ROWS = [
    {
        "id": "abc123",
        "score": 0.95,
        "datasourceId": "eva",
        "datatypeId": "genetic_association",
        "targetFromSourceId": "ENSG00000142192",
        "diseaseFromSource": "Alzheimer disease type 1",
        "literature": ["1520398"],
        "publicationYear": 1992,
        "clinicalSignificances": ["pathogenic"],
        "target": {"id": "ENSG00000142192", "approvedSymbol": "APP"},
        "disease": {"id": "MONDO_0004975", "name": "Alzheimer disease"},
        "urls": [{"niceName": "ClinVar", "url": "https://www.ncbi.nlm.nih.gov/clinvar/x"}],
        "textMiningSentences": None,
    },
    {
        "id": "def456",
        "score": 1.0,
        "datasourceId": "europepmc",
        "datatypeId": "literature",
        "targetFromSourceId": "ENSG00000142192",
        "literature": ["37569624"],
        "publicationYear": 2023,
        "target": {"id": "ENSG00000142192", "approvedSymbol": "APP"},
        "disease": {"id": "MONDO_0004975", "name": "Alzheimer disease"},
        "textMiningSentences": [
            {"text": "Amyloid precursor protein processing is altered.", "section": "results"}
        ],
        "urls": [],
    },
    {
        # An evidence row for a target the run did not resolve locally.
        "id": "ghi789",
        "score": 0.5,
        "datasourceId": "impc",
        "targetFromSourceId": "ENSG99999999999",
        "target": {"id": "ENSG99999999999", "approvedSymbol": "UNKNOWN"},
        "urls": [],
    },
]

# Shape recorded from the ChEMBL REST API (ChEMBL_37).
CHEMBL_MOLECULE = {
    "molecule_chembl_id": "CHEMBL25",
    "pref_name": "ASPIRIN",
    "molecule_structures": {
        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    },
    "molecule_synonyms": [{"molecule_synonym": "Aspirin"}, {"molecule_synonym": "ASA"}],
    "max_phase": "4.0",
    "molecule_type": "Small molecule",
    "first_approval": 1950,
    "oral": True,
    "parenteral": False,
    "withdrawn_flag": False,
    "cross_references": [{"xref_src": "PubChem", "xref_id": "2244"}],
}

CHEMBL_ACTIVITY = {
    "activity_id": 78857,
    "assay_chembl_id": "CHEMBL653511",
    "molecule_chembl_id": "CHEMBL25",
    "target_chembl_id": "CHEMBL4822",
    "standard_type": "IC50",
    "standard_relation": "=",
    "standard_value": "413.0",
    "standard_units": "nM",
    "pchembl_value": "6.38",
    "data_validity_comment": None,
    "potential_duplicate": False,
    "document_chembl_id": "CHEMBL1127323",
}


class TestOpenTargetsNormalisation:
    def test_disease_keeps_source_identifiers_and_provenance(self) -> None:
        row = normalize_disease(OT_DISEASE, retrieved_at=NOW, dataset_version_id="dv1")
        assert row["efo_id"] == "MONDO_0004975"
        assert row["source"] == "open_targets"
        assert row["source_identifier"] == "MONDO_0004975"
        assert row["retrieved_at"] == NOW
        assert row["dataset_version_id"] == "dv1"
        assert row["source_url"].endswith("MONDO_0004975")

    def test_only_exact_and_narrow_synonyms_are_kept(self) -> None:
        """Related synonyms are too loose to use for entity resolution."""
        row = normalize_disease(OT_DISEASE, retrieved_at=NOW, dataset_version_id=None)
        assert "AD" in row["synonyms"]
        assert "Alzheimer disease, familial" in row["synonyms"]

    def test_target_tractability_keeps_only_true_buckets(self) -> None:
        """A bucket with value=False means "not in this bucket"; keeping it would lie."""
        row = normalize_target(OT_TARGET, retrieved_at=NOW, dataset_version_id=None)
        assert row["tractability"]["AB"] == "Approved Drug"
        assert row["tractability"]["SM"] == "Predicted Tractable"
        assert "Clinical Precedence" not in row["tractability"].values()

    def test_target_extracts_uniprot_and_chembl_ids(self) -> None:
        row = normalize_target(OT_TARGET, retrieved_at=NOW, dataset_version_id=None)
        assert row["uniprot_ids"] == ["P05067"]
        assert row["chembl_target_ids"] == ["CHEMBL2487"]

    def test_target_raw_buckets_are_preserved(self) -> None:
        row = normalize_target(OT_TARGET, retrieved_at=NOW, dataset_version_id=None)
        assert "_all_buckets" in row["tractability"]

    def test_gene_normalisation_extracts_locus(self) -> None:
        row = normalize_gene(OT_TARGET, retrieved_at=NOW, dataset_version_id=None)
        assert row["chromosome"] == "21"
        assert row["genomic_start"] == 25880550

    def test_associations_skip_unresolved_targets(self) -> None:
        """A dangling association edge would corrupt the evidence graph."""
        batch = normalize_associations(
            OT_DISEASE,
            disease_id="d1",
            target_id_by_key={},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert batch.rows == []
        assert batch.rejects[0]["_quality_errors"] == ["unresolved_target"]

    def test_associations_copy_the_source_score_verbatim(self) -> None:
        batch = normalize_associations(
            OT_DISEASE,
            disease_id="d1",
            target_id_by_key={"ENSG00000142192": "t1"},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert batch.rows[0]["overall_score"] == 0.8722
        assert batch.rows[0]["datatype_scores"]["genetic_association"] == 0.9229

    def test_evidence_maps_datasources_to_types(self) -> None:
        batch = normalize_evidence(
            OT_EVIDENCE_ROWS,
            disease_id="d1",
            target_id_by_key={"ENSG00000142192": "t1"},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        types = {row["evidence_type"] for row in batch.rows}
        assert types == {"genetic_association", "literature"}
        assert len(batch.rejects) == 1  # the unresolved target

    def test_evidence_summary_only_contains_source_fields(self) -> None:
        """No generated prose may enter the summary."""
        batch = normalize_evidence(
            OT_EVIDENCE_ROWS[:1],
            disease_id="d1",
            target_id_by_key={"ENSG00000142192": "t1"},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        summary = batch.rows[0]["summary"]
        assert "Alzheimer disease type 1" in summary
        assert "pathogenic" in summary

    def test_evidence_captures_text_mined_sentences(self) -> None:
        batch = normalize_evidence(
            OT_EVIDENCE_ROWS[1:2],
            disease_id="d1",
            target_id_by_key={"ENSG00000142192": "t1"},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert "[results]" in batch.rows[0]["summary"]

    def test_absent_score_stays_absent(self) -> None:
        """A missing score must never become 0.0."""
        rows = [{**OT_EVIDENCE_ROWS[0], "score": None}]
        batch = normalize_evidence(
            rows,
            disease_id="d1",
            target_id_by_key={"ENSG00000142192": "t1"},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert batch.rows[0]["score"] is None
        assert batch.rows[0]["score_type"] is None

    def test_evidence_links_to_pubmed_when_a_pmid_is_present(self) -> None:
        batch = normalize_evidence(
            OT_EVIDENCE_ROWS[1:2],
            disease_id="d1",
            target_id_by_key={"ENSG00000142192": "t1"},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert batch.rows[0]["extra"]["pmid"] == "37569624"

    def test_known_drugs_are_extracted_with_mechanisms(self) -> None:
        batch = normalize_known_drugs(
            OT_TARGET, target_id="t1", retrieved_at=NOW, dataset_version_id=None
        )
        assert batch.rows[0]["chembl_id"] == "CHEMBL4297245"
        assert batch.rows[0]["max_clinical_stage"] == "APPROVAL"
        assert "Amyloid-beta A4 protein disrupting agent" in batch.rows[0]["mechanisms_of_action"]

    def test_every_supported_datasource_has_an_evidence_type(self) -> None:
        assert all(bool(value) for value in DATASOURCE_TO_EVIDENCE_TYPE.values())


class TestChEMBLNormalisation:
    def test_molecule_is_restandardised_not_trusted(self) -> None:
        """The InChIKey computed here is the cross-source dedup key."""
        row, reject = normalize_molecule(CHEMBL_MOLECULE, retrieved_at=NOW, dataset_version_id=None)
        assert reject is None
        assert row["inchikey"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        assert row["canonical_smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
        assert row["murcko_scaffold"] == "c1ccccc1"

    def test_max_phase_is_coerced_and_approval_derived(self) -> None:
        row, _ = normalize_molecule(CHEMBL_MOLECULE, retrieved_at=NOW, dataset_version_id=None)
        assert row["max_phase"] == 4.0
        assert row["is_approved_drug"] is True

    def test_cross_references_are_extracted(self) -> None:
        row, _ = normalize_molecule(CHEMBL_MOLECULE, retrieved_at=NOW, dataset_version_id=None)
        assert row["pubchem_cid"] == "2244"

    def test_biologic_without_a_structure_is_rejected_cleanly(self) -> None:
        """Antibodies have no SMILES; rejecting them is correct, crashing is not."""
        row, reject = normalize_molecule(
            {"molecule_chembl_id": "CHEMBL4297245", "molecule_structures": None},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert row is None
        assert reject["_quality_errors"] == ["missing_structure"]

    def test_unparseable_structure_is_rejected_with_the_reason(self) -> None:
        row, reject = normalize_molecule(
            {
                "molecule_chembl_id": "CHEMBLX",
                "molecule_structures": {"canonical_smiles": "not-a-molecule"},
            },
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert row is None
        assert "unparseable_structure" in reject["_quality_errors"][0]

    def test_batch_deduplicates_on_inchikey(self) -> None:
        batch = normalize_molecules(
            [CHEMBL_MOLECULE, {**CHEMBL_MOLECULE, "molecule_chembl_id": "CHEMBL25_DUP"}],
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert len(batch.rows) == 1

    def test_activity_is_active_only_for_exact_potent_measurements(self) -> None:
        """A '>10000 nM' result means inactive up to that concentration, not active."""
        censored = {**CHEMBL_ACTIVITY, "standard_relation": ">", "pchembl_value": "7.0"}
        batch = normalize_activities(
            [CHEMBL_ACTIVITY, censored],
            compound_id_by_chembl={"CHEMBL25": "c1"},
            assay_id_by_chembl={"CHEMBL653511": "a1"},
            target_id_by_chembl={"CHEMBL4822": "t1"},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert batch.rows[0]["is_active"] is True
        assert batch.rows[1]["is_active"] is False

    def test_activity_resolves_foreign_keys(self) -> None:
        batch = normalize_activities(
            [CHEMBL_ACTIVITY],
            compound_id_by_chembl={"CHEMBL25": "c1"},
            assay_id_by_chembl={"CHEMBL653511": "a1"},
            target_id_by_chembl={"CHEMBL4822": "t1"},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert batch.rows[0]["compound_id"] == "c1"
        assert batch.rows[0]["assay_id"] == "a1"
        assert batch.rows[0]["target_id"] == "t1"

    def test_activity_for_unknown_compound_is_rejected(self) -> None:
        batch = normalize_activities(
            [CHEMBL_ACTIVITY],
            compound_id_by_chembl={},
            assay_id_by_chembl={},
            target_id_by_chembl={},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert batch.rows == []
        assert batch.rejects[0]["_quality_errors"] == ["unresolved_compound"]

    def test_assay_confidence_is_preserved(self) -> None:
        row = normalize_assay(
            {
                "assay_chembl_id": "CHEMBL653511",
                "description": "Inhibitory activity against BACE1",
                "confidence_score": 8,
                "assay_type": "B",
            },
            target_id="t1",
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert row["confidence_score"] == 8

    def test_compound_quality_report_shape(self) -> None:
        batch = normalize_molecules([CHEMBL_MOLECULE], retrieved_at=NOW, dataset_version_id=None)
        accepted, _rejected, report = apply_compound_quality(batch.rows)
        assert len(accepted) == 1
        assert report["score"] == 1.0
        assert "name" in report["field_completeness"]

    def test_activity_quality_warns_but_keeps_flagged_rows(self) -> None:
        """Curator-flagged rows stay visible; they are excluded from training, not hidden."""
        flagged = {**CHEMBL_ACTIVITY, "data_validity_comment": "Potential transcription error"}
        batch = normalize_activities(
            [flagged],
            compound_id_by_chembl={"CHEMBL25": "c1"},
            assay_id_by_chembl={},
            target_id_by_chembl={},
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        accepted, _rejected, report = apply_activity_quality(batch.rows)
        assert len(accepted) == 1
        assert "no_validity_comment" in report["rule_failures"]


class TestPubChemNormalisation:
    def test_accepts_either_smiles_property_name(self) -> None:
        """PubChem renamed CanonicalSMILES to ConnectivitySMILES; both must work."""
        for key in ("CanonicalSMILES", "ConnectivitySMILES", "SMILES"):
            row, reject = normalize_compound(
                {"CID": 2244, key: "CC(=O)OC1=CC=CC=C1C(=O)O", "Title": "Aspirin"},
                retrieved_at=NOW,
                dataset_version_id=None,
            )
            assert reject is None, key
            assert row["pubchem_cid"] == "2244"

    def test_reported_properties_are_a_separate_provider(self) -> None:
        """Kept separate so the UI can show RDKit and PubChem values side by side."""
        row = normalize_reported_properties(
            {"CID": 2244, "MolecularWeight": "180.16", "XLogP": 1.2, "TPSA": 63.6},
            compound_id="c1",
        )
        assert row["provider"] == "pubchem"
        assert row["molecular_weight"] == pytest.approx(180.16)
        assert row["logp"] == pytest.approx(1.2)


class TestEuropePMCNormalisation:
    def test_publication_metadata_only(self) -> None:
        row = normalize_publication(
            {
                "id": "37569624",
                "source": "MED",
                "pmid": "37569624",
                "doi": "10.3390/ijms241512246",
                "title": "The Amyloid Cascade Hypothesis 2.0",
                "journalTitle": "International journal of molecular sciences",
                "pubYear": "2023",
                "authorString": "Kepp KP, Robakis NK",
                "abstractText": "A" * 2000,
                "citedByCount": 23,
                "isOpenAccess": "Y",
            },
            retrieved_at=NOW,
            dataset_version_id=None,
        )
        assert row["pmid"] == "37569624"
        assert row["year"] == 2023
        assert row["is_open_access"] is True
        assert len(row["abstract_snippet"]) <= 1200  # snippet only, never full text

    def test_untitled_record_is_skipped(self) -> None:
        assert normalize_publication({"id": "1"}, retrieved_at=NOW, dataset_version_id=None) is None


class TestQualityRules:
    def test_required_rule_rejects_empty_values(self) -> None:
        rule = required("inchikey")
        accepted, rejected, report = evaluate_quality(
            [{"inchikey": "ABC"}, {"inchikey": None}, {"inchikey": ""}], [rule]
        )
        assert len(accepted) == 1
        assert len(rejected) == 2
        assert report.rule_failures["required:inchikey"] == 2

    def test_range_rule_allows_nulls_by_default(self) -> None:
        rule = in_range("max_phase", 0, 4)
        accepted, rejected, _ = evaluate_quality(
            [{"max_phase": 4}, {"max_phase": None}, {"max_phase": 9}], [rule]
        )
        assert len(accepted) == 2
        assert len(rejected) == 1

    def test_warnings_do_not_reject_the_row(self) -> None:
        rule = QualityRule(
            name="soft", description="a warning", predicate=lambda row: False, severity="warning"
        )
        accepted, rejected, report = evaluate_quality([{"a": 1}], [rule])
        assert len(accepted) == 1
        assert len(rejected) == 0
        assert report.rule_failures["soft"] == 1

    def test_deduplication_is_counted(self) -> None:
        accepted, _, report = evaluate_quality(
            [{"id": "a"}, {"id": "a"}, {"id": "b"}], [], dedup_key="id"
        )
        assert len(accepted) == 2
        assert report.duplicate_count == 1

    def test_field_completeness_is_measured(self) -> None:
        _, _, report = evaluate_quality(
            [{"name": "x"}, {"name": None}], [], completeness_fields=["name"]
        )
        assert report.field_completeness["name"] == 0.5

    def test_score_is_the_acceptance_rate(self) -> None:
        _, _, report = evaluate_quality(
            [{"a": 1}, {"a": None}, {"a": 2}, {"a": 3}], [required("a")]
        )
        assert report.score == 0.75

    def test_predicate_exception_is_treated_as_a_failure(self) -> None:
        exploding = QualityRule(
            name="boom",
            description="",
            predicate=lambda row: 1 / 0 > 0,  # type: ignore[operator]
        )
        _, rejected, _ = evaluate_quality([{"a": 1}], [exploding])
        assert len(rejected) == 1


class TestRecorderAndVersioning:
    async def test_replay_mode_refuses_network(self, tmp_path) -> None:
        """Offline mode must fail loudly rather than silently reaching out."""
        recorder = ResponseRecorder(tmp_path, mode=IngestionMode.REPLAY)

        async def loader() -> dict[str, str]:
            raise AssertionError("the loader must not be called in replay mode")

        with pytest.raises(LiveIngestionDisabledError):
            await recorder.fetch(
                source="open_targets", operation="disease", params={"efoId": "x"}, loader=loader
            )

    async def test_record_then_replay_round_trip(self, tmp_path) -> None:
        recorder = ResponseRecorder(tmp_path, mode=IngestionMode.RECORD)
        calls = 0

        async def loader() -> dict[str, int]:
            nonlocal calls
            calls += 1
            return {"value": calls}

        first = await recorder.fetch(
            source="chembl", operation="molecule", params={"id": "CHEMBL25"}, loader=loader
        )
        replayer = ResponseRecorder(tmp_path, mode=IngestionMode.REPLAY)
        second = await replayer.fetch(
            source="chembl", operation="molecule", params={"id": "CHEMBL25"}, loader=loader
        )
        assert first == second
        assert calls == 1
        assert replayer.stats()["fixture_hits"] == 1

    async def test_fixture_key_depends_on_parameters(self, tmp_path) -> None:
        recorder = ResponseRecorder(tmp_path, mode=IngestionMode.AUTO)
        first = recorder.path_for("chembl", "molecule", {"id": "A"})
        second = recorder.path_for("chembl", "molecule", {"id": "B"})
        assert first != second

    def test_version_label_prefers_the_upstream_release(self) -> None:
        """ "Open Targets 26.06" is citable; a retrieval date is only local."""
        assert version_label(NOW, "26.06") == "26.06"
        assert version_label(NOW, None) == "2026.06.01"

    def test_batch_content_hash_is_order_independent(self) -> None:
        batch = NormalizedBatch(entity_type="compound", rows=[{"a": 1}, {"b": 2}])
        reversed_batch = NormalizedBatch(entity_type="compound", rows=[{"b": 2}, {"a": 1}])
        assert batch.content_hash == reversed_batch.content_hash

    def test_batches_of_different_types_cannot_merge(self) -> None:
        with pytest.raises(ValueError):
            NormalizedBatch(entity_type="compound").extend(NormalizedBatch(entity_type="evidence"))


class TestMoleculeNetParsing:
    def test_dataset_specs_are_complete(self) -> None:
        for spec in DATASETS.values():
            assert spec.endpoint and spec.label_column and spec.smiles_column
            assert spec.task_type in ("classification", "regression")
            assert spec.citation

    def test_unknown_dataset_raises_with_options(self) -> None:
        with pytest.raises(ValidationError, match="unknown MoleculeNet dataset"):
            get_spec("nope")

    def test_parsing_standardises_deduplicates_and_reports(self) -> None:
        csv = (
            b"smiles,measured log solubility in mols per litre\n"
            b"CC(=O)Oc1ccccc1C(=O)O,-1.72\n"
            b"CC(=O)OC1=CC=CC=C1C(=O)O,-1.72\n"  # same molecule, different SMILES
            b"not-a-molecule,-2.0\n"
            b"CCO,\n"  # missing label
            b"CN1C=NC2=C1C(=O)N(C)C(=O)N2C,-0.9\n"
        )
        dataset = parse_dataset(DATASETS["esol"], csv, retrieved_at=NOW)
        assert len(dataset) == 2  # duplicate, unparseable and unlabelled all dropped
        reasons = dataset.quality()["reject_reasons"]
        assert reasons["duplicate_structure"] == 1
        assert reasons["unparseable_structure"] == 1
        assert reasons["missing_label"] == 1

    def test_missing_expected_column_raises(self) -> None:
        with pytest.raises(ValidationError, match="expected columns missing"):
            parse_dataset(DATASETS["esol"], b"a,b\n1,2\n", retrieved_at=NOW)

    def test_classification_quality_reports_class_balance(self) -> None:
        csv = b"smiles,p_np\nCC(=O)Oc1ccccc1C(=O)O,1\nCCO,0\nCN1C=NC2=C1C(=O)N(C)C(=O)N2C,1\n"
        dataset = parse_dataset(DATASETS["bbbp"], csv, retrieved_at=NOW)
        balance = dataset.quality()["class_balance"]
        assert balance["positive"] == 2
        assert balance["negative"] == 1

    def test_scaffolds_are_computed_at_load_time(self) -> None:
        """Deterministic scaffolds are what make the split reproducible."""
        csv = (
            b"smiles,measured log solubility in mols per litre\n"
            b"CC(=O)Oc1ccccc1C(=O)O,-1.7\nCCCC,-2.0\n"
        )
        dataset = parse_dataset(DATASETS["esol"], csv, retrieved_at=NOW)
        scaffolds = dataset.scaffolds()
        assert scaffolds[0] == "c1ccccc1"
        assert scaffolds[1] is None  # acyclic
