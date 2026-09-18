"""Molecular processing: standardisation, descriptors, fingerprints, similarity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from theragraph.chem.descriptors import compute_descriptors, descriptor_names
from theragraph.chem.featurize import featurize_many
from theragraph.chem.fingerprints import (
    bits_to_array,
    compute_count_fingerprint,
    compute_fingerprint,
    dice_similarity,
    popcount,
    tanimoto,
    tanimoto_bulk,
)
from theragraph.chem.molecule import (
    murcko_scaffold,
    parse_smiles,
    safe_standardize,
    standardize_smiles,
    to_molblock,
)
from theragraph.chem.similarity import (
    diversity_score,
    matches_substructure,
    nearest_neighbour_similarity,
    popcount_window,
    rank_by_tanimoto,
    screen_substructure_candidates,
    substructure_query_fingerprint,
)
from theragraph.core.errors import ChemistryError

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC = "Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"


class TestStandardization:
    def test_canonical_smiles_is_stable_across_input_forms(self) -> None:
        """The same molecule written differently must canonicalise identically.

        This underpins deduplication: a compound arriving from ChEMBL and from PubChem
        has to land on one row.
        """
        forms = [ASPIRIN, "O=C(C)Oc1ccccc1C(=O)O", "CC(=O)OC1=CC=CC=C1C(O)=O"]
        keys = {standardize_smiles(form).inchikey for form in forms}
        assert len(keys) == 1

    def test_salt_is_reduced_to_parent_and_reported(self) -> None:
        info = standardize_smiles("CC(=O)Oc1ccccc1C(=O)O.Cl")
        assert info.num_fragments == 2
        assert "kept_largest_of_2_fragments" in info.standardization_notes
        assert info.inchikey == standardize_smiles(ASPIRIN).inchikey

    def test_charged_species_is_neutralised(self) -> None:
        info = standardize_smiles("c1ccccc1C(=O)[O-].[Na+]")
        assert info.was_modified
        assert "+" not in info.canonical_smiles or "[Na" not in info.canonical_smiles

    @pytest.mark.parametrize("bad", ["", "   ", "not-a-molecule", "C(((", "X" * 20])
    def test_invalid_input_raises_typed_error(self, bad: str) -> None:
        with pytest.raises(ChemistryError):
            standardize_smiles(bad)

    def test_safe_standardize_reports_instead_of_raising(self) -> None:
        info, error = safe_standardize("definitely not smiles")
        assert info is None
        assert error is not None

    def test_oversized_input_is_rejected(self) -> None:
        with pytest.raises(ChemistryError, match="exceeds"):
            standardize_smiles("C" * 1001)

    def test_acyclic_molecule_has_no_scaffold(self) -> None:
        """Acyclic molecules genuinely have no Bemis-Murcko scaffold.

        Returning None rather than an empty string matters for scaffold splitting: all
        acyclic molecules would otherwise collapse into a single fold.
        """
        scaffold, key = murcko_scaffold(parse_smiles("CCCCO"))
        assert scaffold is None
        assert key is None

    def test_ring_molecule_has_scaffold(self) -> None:
        scaffold, key = murcko_scaffold(parse_smiles(ASPIRIN))
        assert scaffold == "c1ccccc1"
        assert key is not None

    def test_molblock_renders(self) -> None:
        block = to_molblock(ASPIRIN)
        assert "V2000" in block or "V3000" in block


class TestDescriptors:
    def test_known_values_for_aspirin(self) -> None:
        """Anchored to published values so a silent RDKit change is caught."""
        result = compute_descriptors(ASPIRIN)
        assert result.curated["molecular_weight"] == pytest.approx(180.159, abs=0.01)
        assert result.curated["hbd"] == 1
        assert result.curated["hba"] == 3
        assert result.curated["aromatic_rings"] == 1
        assert result.curated["tpsa"] == pytest.approx(63.6, abs=0.5)

    def test_all_descriptors_are_finite(self) -> None:
        """Non-finite descriptors poison models and JSON; they must be zeroed centrally."""
        for smiles in (ASPIRIN, CAFFEINE, "C1=CC2=CC3=CC=CC4=C3C(=C2C=C1)C=CC=C4"):
            block = compute_descriptors(smiles).descriptors
            assert all(math.isfinite(value) for value in block.values())

    def test_lipinski_verdict(self) -> None:
        assert compute_descriptors(ASPIRIN).rule_filters["passes_lipinski"] is True
        # A deliberately greasy, heavy molecule should fail.
        heavy = compute_descriptors("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCc1ccccc1C(=O)O")
        assert heavy.rule_filters["lipinski_violations"] >= 1

    def test_descriptor_order_is_stable(self) -> None:
        assert descriptor_names() == sorted(descriptor_names())

    def test_row_shape_matches_property_columns(self) -> None:
        row = compute_descriptors(ASPIRIN).to_row()
        for field in ("molecular_weight", "logp", "tpsa", "qed", "descriptors", "provider"):
            assert field in row
        assert row["provider"] == "rdkit"


class TestFingerprints:
    def test_identical_molecules_have_similarity_one(self) -> None:
        bits = compute_fingerprint(ASPIRIN).bits
        assert tanimoto(bits, bits) == 1.0

    def test_related_molecules_are_more_similar_than_unrelated(self) -> None:
        aspirin = compute_fingerprint(ASPIRIN).bits
        salicylic = compute_fingerprint(SALICYLIC).bits
        caffeine = compute_fingerprint(CAFFEINE).bits
        assert tanimoto(aspirin, salicylic) > tanimoto(aspirin, caffeine)

    def test_packed_representation_is_compact(self) -> None:
        fingerprint = compute_fingerprint(ASPIRIN)
        assert fingerprint.n_bits == 2048
        assert len(fingerprint.bits) == 256  # 2048 bits packed into bytes

    def test_popcount_matches_unpacked_array(self) -> None:
        fingerprint = compute_fingerprint(ASPIRIN)
        assert popcount(fingerprint.bits) == fingerprint.popcount
        assert int(bits_to_array(fingerprint.bits, 2048).sum()) == fingerprint.popcount

    def test_bulk_matches_pairwise(self) -> None:
        query = compute_fingerprint(ASPIRIN).bits
        others = [compute_fingerprint(smiles).bits for smiles in (SALICYLIC, CAFFEINE)]
        bulk = tanimoto_bulk(query, others)
        pairwise = [tanimoto(query, other) for other in others]
        assert np.allclose(bulk, pairwise, atol=1e-6)

    def test_count_fingerprint_captures_multiplicity(self) -> None:
        counts = compute_count_fingerprint(CAFFEINE)
        assert counts.max() > 1  # caffeine has repeated N-methyl environments

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ChemistryError, match="unknown fingerprint"):
            compute_fingerprint(ASPIRIN, "not_a_kind")

    def test_dice_and_tanimoto_agree_on_identity(self) -> None:
        bits = compute_fingerprint(ASPIRIN).bits
        assert dice_similarity(bits, bits) == pytest.approx(1.0)


class TestSimilaritySearch:
    def test_popcount_window_bounds_are_correct(self) -> None:
        """The window is a mathematical bound, not a heuristic: verify it holds."""
        low, high = popcount_window(24, 0.7)
        assert low == 17  # ceil(0.7 * 24)
        assert high == 34  # floor(24 / 0.7)

    def test_popcount_window_cannot_exclude_a_true_hit(self) -> None:
        query = compute_fingerprint(ASPIRIN)
        threshold = 0.5
        low, high = popcount_window(query.popcount, threshold)
        for smiles in (SALICYLIC, CAFFEINE, "CC(C)Cc1ccc(cc1)C(C)C(=O)O"):
            other = compute_fingerprint(smiles)
            if tanimoto(query.bits, other.bits) >= threshold:
                assert low <= other.popcount <= high

    def test_ranking_respects_threshold_and_order(self) -> None:
        query = compute_fingerprint(ASPIRIN)
        candidates = [
            ("salicylic", *_fingerprint_tuple(SALICYLIC)),
            ("caffeine", *_fingerprint_tuple(CAFFEINE)),
        ]
        hits = rank_by_tanimoto(query.bits, candidates, threshold=0.2, limit=5)
        assert [hit.compound_id for hit in hits] == ["salicylic"]
        assert hits[0].similarity > 0.2

    def test_nearest_neighbour_similarity(self) -> None:
        query = compute_fingerprint(ASPIRIN).bits
        candidates = [("a", *_fingerprint_tuple(SALICYLIC)), ("b", *_fingerprint_tuple(CAFFEINE))]
        best, score = nearest_neighbour_similarity(query, candidates)
        assert best == "a"
        assert 0 < score < 1

    def test_substructure_screen_never_drops_a_real_match(self) -> None:
        """The screen is a necessary condition; a false negative would be a bug."""
        query_bits = substructure_query_fingerprint("c1ccccc1")
        candidates = [
            ("aspirin", *_pattern_tuple(ASPIRIN)),
            ("caffeine", *_pattern_tuple(CAFFEINE)),
        ]
        survivors = screen_substructure_candidates(query_bits, candidates)
        assert "aspirin" in survivors
        assert matches_substructure(ASPIRIN, "c1ccccc1")

    def test_substructure_exact_match_rejects_non_match(self) -> None:
        assert matches_substructure(CAFFEINE, "c1ccccc1") is False

    def test_diversity_score_is_bounded(self) -> None:
        fingerprints = [compute_fingerprint(s).bits for s in (ASPIRIN, SALICYLIC, CAFFEINE)]
        score = diversity_score(fingerprints)
        assert 0.0 <= score <= 1.0
        assert diversity_score(fingerprints[:1]) == 0.0


class TestFeaturization:
    def test_batch_failure_is_isolated(self) -> None:
        """One bad structure must not abort a batch backfill."""
        result = featurize_many([("a", ASPIRIN), ("b", "not-a-molecule"), ("c", CAFFEINE)])
        assert result.succeeded == 2
        assert len(result.failures) == 1
        assert result.failures[0]["compound_id"] == "b"

    def test_every_compound_gets_every_fingerprint_kind(self) -> None:
        result = featurize_many([("a", ASPIRIN)])
        kinds = {row["kind"] for row in result.fingerprints}
        assert kinds == {"morgan_r2_2048", "rdkit_pattern_2048", "maccs_167"}


def _fingerprint_tuple(smiles: str) -> tuple[bytes, int]:
    fingerprint = compute_fingerprint(smiles)
    return fingerprint.bits, fingerprint.popcount


def _pattern_tuple(smiles: str) -> tuple[bytes, int]:
    fingerprint = compute_fingerprint(smiles, "rdkit_pattern_2048")
    return fingerprint.bits, fingerprint.popcount
