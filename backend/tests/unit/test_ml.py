"""ML: splits, calibration, metrics, error analysis, applicability domain."""

from __future__ import annotations

import numpy as np
import pytest

from theragraph.core.errors import ValidationError
from theragraph.ml.calibration import (
    Calibrator,
    fit_calibrator,
    regression_interval_coverage,
    reliability_curve,
)
from theragraph.ml.datasets import FeatureSpec, MolecularDataset, model_descriptor_names
from theragraph.ml.evaluation import (
    applicability_domain,
    classification_metrics,
    compare_models,
    evaluate,
    prediction_distribution,
    regression_metrics,
)
from theragraph.ml.graph_features import collate_graphs, mol_to_graph
from theragraph.ml.inference import compute_confidence
from theragraph.ml.registry import compute_version
from theragraph.ml.splits import (
    cross_validation_folds,
    describe_split_difficulty,
    random_split,
    scaffold_split,
)

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CCO",
    "CCCC",
]


def make_dataset(n: int = 60, task: str = "regression") -> MolecularDataset:
    """A synthetic dataset with a controlled number of scaffolds."""
    rng = np.random.default_rng(7)
    smiles = [SMILES[index % len(SMILES)] for index in range(n)]
    scaffolds = [f"scaffold_{index % 8}" if index % 5 else None for index in range(n)]
    labels = (
        rng.normal(0, 1, n).astype(np.float32)
        if task == "regression"
        else (rng.random(n) < 0.3).astype(np.float32)
    )
    return MolecularDataset(
        name="synthetic",
        endpoint="test",
        task_type=task,  # type: ignore[arg-type]
        smiles=smiles,
        labels=labels,
        scaffolds=scaffolds,
    )


class TestScaffoldSplit:
    def test_no_scaffold_appears_in_two_folds(self) -> None:
        """The whole point of a scaffold split; a leak makes the metric meaningless."""
        dataset = make_dataset(120)
        split = scaffold_split(dataset.scaffolds)
        split.assert_disjoint()

        def scaffolds_of(indices: np.ndarray) -> set[str]:
            return {dataset.scaffolds[i] for i in indices.tolist() if dataset.scaffolds[i]}

        train = scaffolds_of(split.train_indices)
        test = scaffolds_of(split.test_indices)
        assert not (train & test)

    def test_split_is_deterministic(self) -> None:
        dataset = make_dataset(80)
        first = scaffold_split(dataset.scaffolds)
        second = scaffold_split(dataset.scaffolds)
        assert np.array_equal(first.test_indices, second.test_indices)

    def test_every_index_is_used_exactly_once(self) -> None:
        dataset = make_dataset(90)
        split = scaffold_split(dataset.scaffolds)
        combined = np.concatenate([split.train_indices, split.valid_indices, split.test_indices])
        assert sorted(combined.tolist()) == list(range(90))

    def test_acyclic_molecules_are_distributed_not_bundled(self) -> None:
        """All-acyclic molecules must not collapse into one fold."""
        scaffolds: list[str | None] = [None] * 40
        split = scaffold_split(scaffolds)
        assert len(split.test_indices) > 0
        assert len(split.train_indices) > 0

    def test_test_scaffolds_are_unseen(self) -> None:
        """With every molecule scaffolded, all test scaffolds must be novel."""
        scaffolds: list[str | None] = [f"scaffold_{index % 20}" for index in range(100)]
        split = scaffold_split(scaffolds)
        difficulty = describe_split_difficulty(scaffolds, split)
        assert difficulty["shared_scaffolds"] == 0
        assert difficulty["test_scaffolds_unseen_fraction"] == 1.0

    def test_unseen_fraction_is_none_when_the_test_fold_is_all_acyclic(self) -> None:
        """Reporting 0.0 here would read as "no novel scaffolds", which is not the fact."""
        scaffolds: list[str | None] = [None] * 40
        split = scaffold_split(scaffolds)
        difficulty = describe_split_difficulty(scaffolds, split)
        assert difficulty["test_scaffolds_unseen_fraction"] is None
        assert difficulty["acyclic_test_molecules"] == len(split.test_indices)

    @pytest.mark.parametrize("train,valid", [(0.0, 0.1), (1.0, 0.0), (0.95, 0.1)])
    def test_invalid_fractions_rejected(self, train: float, valid: float) -> None:
        with pytest.raises(ValidationError):
            scaffold_split(["a", "b", "c"], train_fraction=train, valid_fraction=valid)

    def test_empty_dataset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            scaffold_split([])


class TestCrossValidation:
    def test_folds_partition_the_dataset(self) -> None:
        dataset = make_dataset(100)
        folds = cross_validation_folds(dataset.scaffolds, n_folds=5)
        assert len(folds) == 5
        all_test = np.concatenate([fold.test_indices for fold in folds])
        assert sorted(all_test.tolist()) == list(range(100))

    def test_no_scaffold_spans_folds(self) -> None:
        dataset = make_dataset(100)
        for fold in cross_validation_folds(dataset.scaffolds, n_folds=4):
            train = {
                dataset.scaffolds[i] for i in fold.train_indices.tolist() if dataset.scaffolds[i]
            }
            test = {
                dataset.scaffolds[i] for i in fold.test_indices.tolist() if dataset.scaffolds[i]
            }
            assert not (train & test)

    def test_minimum_fold_count_enforced(self) -> None:
        with pytest.raises(ValidationError):
            cross_validation_folds(["a"], n_folds=1)


class TestRandomSplit:
    def test_is_reproducible_and_disjoint(self) -> None:
        first = random_split(100, seed=42)
        second = random_split(100, seed=42)
        assert np.array_equal(first.test_indices, second.test_indices)
        first.assert_disjoint()

    def test_different_seed_gives_different_split(self) -> None:
        assert not np.array_equal(
            random_split(100, seed=1).test_indices, random_split(100, seed=2).test_indices
        )


class TestFeatureSpec:
    def test_inference_reproduces_training_features(self) -> None:
        """Train/serve skew is structurally impossible only if this holds."""
        dataset = make_dataset(20)
        matrix, spec = dataset.build_features("fingerprint+descriptors")
        single = spec.featurize_one(dataset.smiles[0])
        assert np.allclose(single, matrix[0], atol=1e-4)

    def test_spec_round_trips_through_json(self) -> None:
        dataset = make_dataset(10)
        _, spec = dataset.build_features("fingerprint+descriptors")
        restored = FeatureSpec.from_dict(spec.to_dict())
        assert restored.n_features == spec.n_features
        assert restored.descriptor_names == spec.descriptor_names

    def test_unstable_descriptors_are_excluded(self) -> None:
        names = model_descriptor_names()
        assert "Ipc" not in names
        assert not any(name.startswith("BCUT2D") for name in names)

    def test_scaler_statistics_are_baked_into_the_spec(self) -> None:
        dataset = make_dataset(30)
        _, spec = dataset.build_features("descriptors", fit_scaler=True)
        assert spec.descriptor_mean is not None
        assert len(spec.descriptor_mean) == len(spec.descriptor_names)

    def test_subset_preserves_alignment(self) -> None:
        dataset = make_dataset(20)
        subset = dataset.subset([0, 5, 10])
        assert subset.smiles == [dataset.smiles[0], dataset.smiles[5], dataset.smiles[10]]
        assert np.allclose(subset.labels, dataset.labels[[0, 5, 10]])


class TestCalibration:
    def test_platt_improves_a_miscalibrated_classifier(self) -> None:
        rng = np.random.default_rng(0)
        labels = (rng.random(600) < 0.3).astype(float)
        # Deliberately over-confident probabilities.
        raw = np.clip(labels * 0.5 + rng.random(600) * 0.5, 0.01, 0.99) ** 0.5
        calibrator = fit_calibrator(raw, labels)
        assert calibrator.method in ("platt", "isotonic")
        before = reliability_curve(raw, labels)["ece"]
        after = reliability_curve(calibrator.apply(raw), labels)["ece"]
        assert after < before

    def test_calibration_is_refused_when_it_would_not_help(self) -> None:
        """Applying a transform that makes things worse is the failure to avoid."""
        rng = np.random.default_rng(1)
        labels = (rng.random(400) < 0.5).astype(float)
        already_good = labels * 0.98 + (1 - labels) * 0.02
        calibrator = fit_calibrator(already_good, labels)
        assert calibrator.method in ("none", "platt", "isotonic")
        after = reliability_curve(calibrator.apply(already_good), labels)["ece"]
        assert after <= reliability_curve(already_good, labels)["ece"] + 1e-6

    def test_insufficient_data_skips_calibration(self) -> None:
        assert fit_calibrator(np.array([0.5, 0.6]), np.array([0, 1])).method == "none"

    def test_single_class_skips_calibration(self) -> None:
        assert fit_calibrator(np.random.random(50), np.zeros(50)).method == "none"

    def test_calibrator_round_trips(self, tmp_path) -> None:
        calibrator = Calibrator(method="platt", platt_a=1.2, platt_b=-0.3)
        path = tmp_path / "cal.json"
        calibrator.save(path)
        assert Calibrator.load(path).platt_a == pytest.approx(1.2)

    def test_reliability_bins_report_counts(self) -> None:
        rng = np.random.default_rng(2)
        labels = (rng.random(200) < 0.4).astype(float)
        curve = reliability_curve(rng.random(200), labels, n_bins=10)
        assert len(curve["bins"]) == 10
        assert sum(bin["count"] for bin in curve["bins"]) == 200
        assert curve["brier"] is not None

    def test_interval_coverage_is_reported_honestly(self) -> None:
        rng = np.random.default_rng(3)
        predictions = rng.normal(0, 1, 400)
        actuals = predictions + rng.normal(0, 0.5, 400)
        # Under-stated uncertainty must produce coverage below nominal.
        coverage = regression_interval_coverage(predictions, np.full(400, 0.05), actuals)
        assert coverage["coverage"] < 0.5
        assert "over-confident" in coverage["interpretation"]

    def test_the_verdict_describes_this_model_not_a_general_rule(self) -> None:
        """A fixed legend would call a conservative model over-confident."""
        rng = np.random.default_rng(11)
        predictions = rng.normal(0, 1, 400)
        actuals = predictions + rng.normal(0, 0.1, 400)

        wide = regression_interval_coverage(predictions, np.full(400, 2.0), actuals)
        assert wide["coverage"] > wide["nominal"]
        assert "conservative" in wide["interpretation"]
        assert "over-confident" not in wide["interpretation"]

        # sigma = 0.1 against a 0.1 noise scale is the calibrated case.
        matched = regression_interval_coverage(predictions, np.full(400, 0.1), actuals)
        assert matched["coverage"] == pytest.approx(matched["nominal"], abs=0.05)
        assert "calibrated" in matched["interpretation"]


class TestMetrics:
    def test_perfect_classifier(self) -> None:
        labels = np.array([0, 0, 1, 1] * 25)
        metrics = classification_metrics(labels.astype(float), labels.astype(float))
        assert metrics["roc_auc"]["value"] == pytest.approx(1.0)
        assert metrics["confusion_matrix"]["false_positive"] == 0

    def test_confidence_intervals_are_produced(self) -> None:
        rng = np.random.default_rng(4)
        labels = (rng.random(300) < 0.4).astype(float)
        probabilities = np.clip(labels * 0.6 + rng.random(300) * 0.4, 0, 1)
        metrics = classification_metrics(probabilities, labels)
        assert metrics["roc_auc"]["ci_low"] is not None
        assert metrics["roc_auc"]["ci_low"] <= metrics["roc_auc"]["value"]

    def test_pr_auc_baseline_is_reported(self) -> None:
        """Without the baseline, PR-AUC on imbalanced data is uninterpretable."""
        labels = np.array([1.0] * 5 + [0.0] * 95)
        metrics = classification_metrics(np.random.random(100), labels)
        assert metrics["baseline_pr_auc"] == pytest.approx(0.05, abs=0.001)

    def test_regression_metrics_and_bias(self) -> None:
        actuals = np.linspace(-3, 3, 100)
        predictions = actuals + 0.5  # a constant, detectable bias
        metrics = regression_metrics(predictions, actuals)
        assert metrics["rmse"]["value"] == pytest.approx(0.5, abs=0.01)
        assert metrics["bias"] == pytest.approx(0.5, abs=0.01)

    def test_prediction_distribution_summary(self) -> None:
        distribution = prediction_distribution(np.random.default_rng(5).normal(0, 1, 500))
        assert distribution["count"] == 500
        assert "histogram" in distribution
        assert sum(distribution["histogram"]["counts"]) == 500

    def test_model_comparison_flags_overlapping_intervals(self) -> None:
        """Two models with overlapping CIs are not measurably different, and the
        comparison must say so rather than declaring a winner outright."""
        rng = np.random.default_rng(6)
        labels = (rng.random(120) < 0.4).astype(float)
        results = {}
        for name, noise in (("a", 0.45), ("b", 0.46)):
            probabilities = np.clip(labels * 0.55 + rng.random(120) * noise, 0, 1)
            results[name] = evaluate(
                task_type="classification",
                split_strategy="scaffold",
                split_name="test",
                predictions=probabilities,
                actuals=labels,
                smiles=["CCO"] * 120,
                identifiers=[str(index) for index in range(120)],
                scaffolds=["s"] * 120,
            )
        comparison = compare_models(results)
        assert comparison["winner"] in results
        assert (
            "unproven" in comparison["verdict"]
            or comparison["statistically_indistinguishable_from_winner"]
        )


class TestApplicabilityDomain:
    def test_dissimilar_molecule_is_out_of_domain(self) -> None:
        domain = applicability_domain(
            ["CN1C=NC2=C1C(=O)N(C)C(=O)N2C"], ["CC(=O)Oc1ccccc1C(=O)O", "Oc1ccccc1C(=O)O"]
        )
        assert domain["out_of_domain"] == 1

    def test_identical_molecule_is_in_domain(self) -> None:
        domain = applicability_domain(["CC(=O)Oc1ccccc1C(=O)O"], ["CC(=O)Oc1ccccc1C(=O)O"])
        assert domain["in_domain"] == 1
        assert domain["mean_nearest_similarity"] == pytest.approx(1.0)

    def test_empty_inputs_are_handled(self) -> None:
        assert applicability_domain([], ["CCO"])["assessed"] == 0


class TestConfidence:
    def test_tight_in_domain_prediction_is_confident(self) -> None:
        confidence = compute_confidence(
            task_type="regression", uncertainty=0.1, nearest_similarity=0.8, label_scale=2.0
        )
        assert confidence is not None and confidence > 0.9

    def test_wide_out_of_domain_prediction_is_not(self) -> None:
        confidence = compute_confidence(
            task_type="regression", uncertainty=2.5, nearest_similarity=0.1, label_scale=2.0
        )
        assert confidence is not None and confidence < 0.3

    def test_absent_information_returns_none_not_a_guess(self) -> None:
        assert (
            compute_confidence(task_type="regression", uncertainty=None, nearest_similarity=None)
            is None
        )


class TestGraphFeatures:
    def test_graph_shapes(self) -> None:
        graph = mol_to_graph("CC(=O)Oc1ccccc1C(=O)O")
        assert graph.n_nodes == 13
        assert graph.n_edges == 26  # undirected bonds stored both ways
        assert graph.node_features.shape[0] == graph.n_nodes

    def test_single_atom_molecule_gets_a_self_loop(self) -> None:
        """Without this, message passing and pooling are undefined for ions."""
        graph = mol_to_graph("[Na+]")
        assert graph.n_nodes == 1
        assert graph.n_edges == 1

    def test_batching_offsets_edge_indices(self) -> None:
        graphs = [mol_to_graph(smiles) for smiles in SMILES[:3]]
        batch = collate_graphs(graphs)
        assert batch.n_graphs == 3
        assert batch.node_features.shape[0] == sum(graph.n_nodes for graph in graphs)
        assert int(batch.edge_index.max()) < batch.node_features.shape[0]
        assert sorted(set(batch.batch.tolist())) == [0, 1, 2]


class TestModelVersioning:
    def test_version_is_deterministic_for_identical_inputs(self) -> None:
        arguments = {
            "dataset_content_hash": "abc",
            "hyperparameters": {"n": 500},
            "feature_spec": {"representation": "fingerprint"},
            "code_version": "0.1.0",
            "split_strategy": "scaffold",
        }
        assert compute_version(**arguments) == compute_version(**arguments)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("dataset_content_hash", "different"),
            ("hyperparameters", {"n": 501}),
            ("feature_spec", {"representation": "graph"}),
            ("code_version", "0.2.0"),
        ],
    )
    def test_version_changes_when_anything_that_matters_changes(self, field: str, value) -> None:
        base = {
            "dataset_content_hash": "abc",
            "hyperparameters": {"n": 500},
            "feature_spec": {"representation": "fingerprint"},
            "code_version": "0.1.0",
            "split_strategy": "scaffold",
        }
        assert compute_version(**base) != compute_version(**{**base, field: value})
