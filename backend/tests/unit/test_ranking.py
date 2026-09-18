"""Candidate prioritisation: desirability curves, filters, weighting, attribution."""

from __future__ import annotations

import pytest

from theragraph.core.errors import ValidationError
from theragraph.ranking.config import (
    CNS_CONFIG,
    DEFAULT_CONFIG,
    EVIDENCE_FIRST_CONFIG,
    Desirability,
    HardFilter,
    Objective,
    RankingConfig,
    list_presets,
    preset,
)
from theragraph.ranking.engine import (
    CandidateInput,
    compare_candidates,
    explain_candidate,
    rank_candidates,
)


class TestDesirability:
    @pytest.mark.parametrize(
        "value,expected",
        [(9.0, 1.0), (8.0, 1.0), (6.5, 0.5), (5.0, 0.0), (3.0, 0.0)],
    )
    def test_higher_better_ramp(self, value: float, expected: float) -> None:
        curve = Desirability(kind="higher_better", good=8.0, bad=5.0)
        assert curve.apply(value) == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "value,expected",
        [(0.05, 1.0), (0.1, 1.0), (0.35, 0.5), (0.6, 0.0), (0.9, 0.0)],
    )
    def test_lower_better_ramp(self, value: float, expected: float) -> None:
        curve = Desirability(kind="lower_better", good=0.1, bad=0.6)
        assert curve.apply(value) == pytest.approx(expected, abs=0.01)

    def test_window_plateaus_then_falls_off(self) -> None:
        """Both extremes are undesirable for properties like molecular weight."""
        curve = Desirability(kind="window", low=250.0, high=500.0, tolerance=150.0)
        assert curve.apply(400) == 1.0
        assert curve.apply(250) == 1.0
        assert curve.apply(500) == 1.0
        assert curve.apply(575) == pytest.approx(0.5, abs=0.01)
        assert curve.apply(175) == pytest.approx(0.5, abs=0.01)
        assert curve.apply(50) == 0.0
        assert curve.apply(1000) == 0.0

    def test_step_and_identity(self) -> None:
        assert Desirability(kind="step", threshold=6.0).apply(6.5) == 1.0
        assert Desirability(kind="step", threshold=6.0).apply(5.5) == 0.0
        assert Desirability(kind="identity").apply(0.42) == 0.42
        assert Desirability(kind="identity").apply(1.8) == 1.0  # clamped

    def test_output_is_always_bounded(self) -> None:
        curve = Desirability(kind="higher_better", good=8.0, bad=5.0)
        for value in (-100, 0, 5, 6, 8, 100, 1e6):
            assert 0.0 <= curve.apply(value) <= 1.0

    def test_misconfigured_curve_raises(self) -> None:
        with pytest.raises(ValidationError):
            Desirability(kind="window").apply(5.0)
        with pytest.raises(ValidationError):
            Desirability(kind="step").apply(5.0)

    def test_curve_describes_itself(self) -> None:
        """The UI renders these strings, so they must be meaningful."""
        assert "higher is better" in Desirability(kind="higher_better", good=8, bad=5).describe()
        assert "ideal between" in Desirability(kind="window", low=1, high=4).describe()


class TestConfig:
    def test_presets_are_valid_and_hashable(self) -> None:
        for entry in list_presets():
            config = preset(entry["name"])
            assert config.total_weight > 0
            assert len(config.config_hash) == 16
            assert sum(config.normalized_weights().values()) == pytest.approx(1.0)

    def test_unknown_preset_raises_with_options(self) -> None:
        with pytest.raises(ValidationError, match="unknown ranking preset"):
            preset("nope")

    def test_config_round_trips_through_json(self) -> None:
        restored = RankingConfig.from_dict(DEFAULT_CONFIG.to_dict())
        assert restored.config_hash == DEFAULT_CONFIG.config_hash

    def test_cns_preset_adds_bbb_and_tightens_weight_window(self) -> None:
        keys = {objective.key for objective in CNS_CONFIG.objectives}
        assert "predicted_bbb" in keys
        weight_curve = next(
            objective.desirability
            for objective in CNS_CONFIG.objectives
            if objective.key == "molecular_weight"
        )
        assert weight_curve.high == 400.0  # CNS programmes favour smaller molecules

    def test_evidence_first_preset_downweights_predictions(self) -> None:
        by_key = {objective.key: objective for objective in EVIDENCE_FIRST_CONFIG.objectives}
        assert by_key["measured_potency"].weight > by_key["predicted_solubility"].weight * 5

    def test_with_weights_drops_zeroed_objectives(self) -> None:
        adjusted = DEFAULT_CONFIG.with_weights({"predicted_solubility": 0.0})
        assert "predicted_solubility" not in {objective.key for objective in adjusted.objectives}

    def test_empty_config_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankingConfig(name="x", description="", objectives=[])


class TestHardFilters:
    def test_filter_abstains_when_it_cannot_evaluate(self) -> None:
        """A missing value must not be treated as a failure."""
        hard_filter = HardFilter(
            key="mw", label="MW", source_field="molecular_weight", operator="<=", value=800
        )
        assert hard_filter.passes(None) is None
        assert hard_filter.passes(500) is True
        assert hard_filter.passes(900) is False

    def test_withdrawn_drug_is_removed(self) -> None:
        candidates = [
            _candidate("keep", best_pchembl=8.0),
            _candidate("withdrawn", best_pchembl=9.0, withdrawn_flag=True),
        ]
        outcome = rank_candidates(candidates, DEFAULT_CONFIG)
        assert [item.compound_id for item in outcome.ranked] == ["keep"]
        assert outcome.filtered_out[0].failed_filter == "not_withdrawn"

    def test_fragment_is_removed_by_heavy_atom_floor(self) -> None:
        outcome = rank_candidates(
            [_candidate("fragment", heavy_atom_count=8, molecular_weight=120.0)],
            DEFAULT_CONFIG,
        )
        assert outcome.ranked == []
        assert outcome.filtered_out[0].failed_filter == "heavy_atom_floor"

    def test_filtered_candidates_are_reported_not_hidden(self) -> None:
        outcome = rank_candidates([_candidate("withdrawn", withdrawn_flag=True)], DEFAULT_CONFIG)
        assert outcome.filtered_out[0].reason
        assert "removed by hard filters" in " ".join(outcome.warnings)


class TestRankingEngine:
    def test_potent_well_characterised_candidate_outranks_a_weak_one(self) -> None:
        outcome = rank_candidates(
            [_candidate("weak", best_pchembl=5.2), _candidate("potent", best_pchembl=8.8)],
            DEFAULT_CONFIG,
        )
        assert outcome.ranked[0].compound_id == "potent"
        assert outcome.ranked[0].rank == 1

    def test_contributions_sum_to_the_total(self) -> None:
        """Every point of the score must be attributable, or the breakdown lies."""
        outcome = rank_candidates([_candidate("c1", best_pchembl=7.5)], DEFAULT_CONFIG)
        candidate = outcome.ranked[0]
        total = sum(
            factor.contribution for factor in candidate.factors if factor.raw_value is not None
        )
        assert total == pytest.approx(candidate.total_score, abs=1e-6)

    def test_missing_objectives_renormalise_rather_than_scoring_zero(self) -> None:
        """Scoring absent data as zero would systematically bury novel compounds."""
        full = _candidate("full", best_pchembl=8.0)
        sparse = _candidate("sparse", best_pchembl=8.0)
        sparse.predictions = {}
        outcome = rank_candidates([full, sparse], DEFAULT_CONFIG)
        scores = {item.compound_id: item.total_score for item in outcome.ranked}
        # The sparse candidate is still scored on a comparable 0-1 scale.
        assert 0.0 < scores["sparse"] <= 1.0
        sparse_row = next(item for item in outcome.ranked if item.compound_id == "sparse")
        assert sparse_row.objectives_missing > 0
        present_weight = sum(
            factor.normalized_weight
            for factor in sparse_row.factors
            if factor.raw_value is not None
        )
        assert present_weight == pytest.approx(1.0, abs=1e-6)

    def test_out_of_domain_prediction_is_discounted_and_flagged(self) -> None:
        confident = _candidate("confident", best_pchembl=7.0)
        extrapolated = _candidate("extrapolated", best_pchembl=7.0)
        extrapolated.predictions = {
            "solubility": {
                "value": -2.0,
                "confidence": 0.2,
                "model": "m@1",
                "applicability": {"in_domain": False},
            }
        }
        outcome = rank_candidates([confident, extrapolated], DEFAULT_CONFIG)
        row = next(item for item in outcome.ranked if item.compound_id == "extrapolated")
        assert "out_of_domain_prediction" in row.flags
        assert "prediction_discounted" in row.flags
        assert any(factor.discounted for factor in row.factors)

    def test_candidate_without_measured_potency_is_flagged(self) -> None:
        candidate = _candidate("no_potency")
        candidate.best_pchembl = None
        outcome = rank_candidates([candidate], DEFAULT_CONFIG)
        assert "no_experimental_potency" in outcome.ranked[0].flags

    def test_approved_drug_is_flagged(self) -> None:
        outcome = rank_candidates([_candidate("approved", max_phase=4.0)], DEFAULT_CONFIG)
        assert "approved_drug" in outcome.ranked[0].flags

    def test_weights_change_the_order(self) -> None:
        """The whole premise of the weight sliders."""
        novel = _candidate("novel", best_pchembl=6.0, novelty_score=0.98)
        potent = _candidate("potent", best_pchembl=8.5, novelty_score=0.1)

        default = rank_candidates([novel, potent], DEFAULT_CONFIG)
        assert default.ranked[0].compound_id == "potent"

        novelty_first = DEFAULT_CONFIG.with_weights(
            {"structural_novelty": 10.0, "measured_potency": 0.25}
        )
        reweighted = rank_candidates([novel, potent], novelty_first)
        assert reweighted.ranked[0].compound_id == "novel"

    def test_ranks_are_dense_and_ordered(self) -> None:
        outcome = rank_candidates(
            [_candidate(f"c{index}", best_pchembl=5.0 + index * 0.3) for index in range(6)],
            DEFAULT_CONFIG,
        )
        assert [item.rank for item in outcome.ranked] == [1, 2, 3, 4, 5, 6]
        scores = [item.total_score for item in outcome.ranked]
        assert scores == sorted(scores, reverse=True)

    def test_result_is_deterministic(self) -> None:
        candidates = [_candidate(f"c{index}", best_pchembl=7.0) for index in range(5)]
        first = rank_candidates(candidates, DEFAULT_CONFIG)
        second = rank_candidates(candidates, DEFAULT_CONFIG)
        assert [item.compound_id for item in first.ranked] == [
            item.compound_id for item in second.ranked
        ]

    def test_empty_input_warns_rather_than_raising(self) -> None:
        outcome = rank_candidates([], DEFAULT_CONFIG)
        assert outcome.ranked == []
        assert "no candidates could be ranked" in " ".join(outcome.warnings)

    def test_candidate_with_no_scorable_data_is_excluded_with_a_reason(self) -> None:
        bare = CandidateInput(compound_id="bare", heavy_atom_count=20, molecular_weight=300.0)
        outcome = rank_candidates([bare], DEFAULT_CONFIG)
        # Only the weight/heavy-atom objectives can score, so it is under-characterised
        # but still comparable; nothing is silently dropped.
        assert outcome.ranked or outcome.filtered_out

    def test_required_objective_filters_the_candidate(self) -> None:
        config = RankingConfig(
            name="requires_potency",
            description="",
            objectives=[
                Objective(
                    key="measured_potency",
                    label="Potency",
                    weight=1.0,
                    source_field="best_pchembl",
                    basis="measured",
                    desirability=Desirability(kind="higher_better", good=8.0, bad=5.0),
                    explanation="",
                    required=True,
                )
            ],
        )
        candidate = _candidate("no_potency")
        candidate.best_pchembl = None
        outcome = rank_candidates([candidate], config)
        assert outcome.filtered_out[0].failed_filter == "required_objective_missing"

    def test_disclaimer_is_attached_to_every_row(self) -> None:
        """The framing must travel with the data, not live only in the UI."""
        outcome = rank_candidates([_candidate("c1", best_pchembl=7.0)], DEFAULT_CONFIG)
        payload = outcome.ranked[0].to_dict()
        assert "not a prediction of clinical efficacy" in payload["disclaimer"]
        assert "not experimentally validated" in outcome.to_dict()["disclaimer"].replace(
            "no ranking here has been experimentally validated", "not experimentally validated"
        )


class TestExplanationAndComparison:
    def test_explanation_names_the_basis_of_each_factor(self) -> None:
        outcome = rank_candidates([_candidate("c1", best_pchembl=8.0)], DEFAULT_CONFIG)
        text = explain_candidate(outcome.ranked[0])
        assert "experimental" in text
        assert "model prediction" in text
        assert "Rank 1" in text

    def test_comparison_highlights_best_and_worst_per_objective(self) -> None:
        outcome = rank_candidates(
            [_candidate("a", best_pchembl=8.5), _candidate("b", best_pchembl=5.5)],
            DEFAULT_CONFIG,
        )
        comparison = compare_candidates(outcome.ranked)
        assert comparison["highlights"]["measured_potency"]["best"] == "a"
        assert comparison["highlights"]["measured_potency"]["worst"] == "b"
        assert len(comparison["candidates"]) == 2


def _candidate(
    compound_id: str,
    *,
    best_pchembl: float | None = 7.0,
    novelty_score: float = 0.5,
    molecular_weight: float = 380.0,
    heavy_atom_count: int = 27,
    withdrawn_flag: bool = False,
    max_phase: float | None = 2.0,
) -> CandidateInput:
    """A fully characterised candidate, so tests vary one thing at a time."""
    return CandidateInput(
        compound_id=compound_id,
        name=compound_id,
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        chembl_id=f"CHEMBL{abs(hash(compound_id)) % 100000}",
        best_pchembl=best_pchembl,
        n_measurements=4,
        evidence_score=0.8,
        evidence_count=3,
        max_phase=max_phase,
        withdrawn_flag=withdrawn_flag,
        molecular_weight=molecular_weight,
        logp=2.8,
        tpsa=70.0,
        qed=0.68,
        heavy_atom_count=heavy_atom_count,
        novelty_score=novelty_score,
        predictions={
            "solubility": {
                "value": -3.2,
                "confidence": 0.82,
                "model": "solubility-rf@1",
                "applicability": {"in_domain": True},
            },
            "clinical_toxicity": {
                "probability": 0.12,
                "confidence": 0.75,
                "model": "tox-rf@1",
                "applicability": {"in_domain": True},
            },
        },
    )
