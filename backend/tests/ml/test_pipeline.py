"""Training and serving pipeline, end to end on a small real dataset.

Marked ``slow`` because it genuinely trains models. The point is not accuracy on 90
molecules -- it is that the pipeline produces a registered, loadable, reproducible
artifact with honest metrics attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.config import Settings
from theragraph.core.clock import utcnow
from theragraph.db.repositories import ml as ml_repo
from theragraph.db.repositories import provenance as prov_repo
from theragraph.ingestion import moleculenet
from theragraph.ml.baseline import BaselineModel
from theragraph.ml.calibration import Calibrator
from theragraph.ml.datasets import load_dataset_from_snapshot
from theragraph.ml.gnn import GraphModel
from theragraph.ml.inference import InferenceService
from theragraph.ml.registry import (
    TrainingReference,
    compute_version,
    load_model,
    register_model,
    save_artifacts,
)
from theragraph.ml.splits import scaffold_split
from theragraph.ml.train import TrainingConfig, train_endpoint

pytestmark = pytest.mark.asyncio

# Real molecules with real measured solubilities (ESOL), enough scaffolds to split.
TRAINING_ROWS = [
    ("CC(=O)Oc1ccccc1C(=O)O", -1.72),
    ("Oc1ccccc1C(=O)O", -1.55),
    ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", -0.9),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", -3.61),
    ("CCO", 1.1),
    ("CCCCO", -0.4),
    ("CCCCCCCCO", -2.39),
    ("c1ccccc1", -1.64),
    ("Cc1ccccc1", -2.21),
    ("CCc1ccccc1", -2.77),
    ("Clc1ccccc1", -2.38),
    ("Oc1ccccc1", 0.0),
    ("Nc1ccccc1", -0.41),
    ("O=Cc1ccccc1", -1.19),
    ("CC(=O)c1ccccc1", -1.28),
    ("c1ccc2ccccc2c1", -3.6),
    ("c1ccc2c(c1)ccc1ccccc12", -5.2),
    ("CCCCCCCC", -5.24),
    ("CCCCCC", -3.84),
    ("CCCC", -2.55),
    ("C1CCCCC1", -3.1),
    ("C1CCCC1", -2.58),
    ("OCC(O)CO", 1.11),
    ("OCCO", 1.02),
    ("CC(=O)N", 1.58),
    ("CC(=O)O", 1.22),
    ("c1ccncc1", 0.76),
    ("c1cc[nH]c1", 0.5),
    ("c1ccsc1", -1.44),
    ("c1ccoc1", -0.82),
    ("CS(=O)(=O)O", 1.0),
    ("CCOC(=O)C", -0.04),
    ("CCOCC", -0.09),
    ("CC(C)O", 0.43),
    ("CC(C)(C)O", 0.62),
    ("ClCCl", -1.17),
    ("ClC(Cl)Cl", -1.96),
    ("BrCCBr", -1.68),
    ("NCCO", 1.2),
    ("NCCN", 1.3),
]


async def _register_dataset(session: AsyncSession, settings: Settings, tmp_path: Path) -> str:
    """Write a real snapshot and register its dataset version, as ingestion would."""
    from theragraph.ingestion.pipelines import ensure_data_sources

    await ensure_data_sources(session)
    spec = moleculenet.get_spec("esol")

    header = f"smiles,{spec.label_column}\n"
    body = "".join(f"{smiles},{label}\n" for smiles, label in TRAINING_ROWS)
    dataset = moleculenet.parse_dataset(spec, (header + body).encode(), retrieved_at=utcnow())

    snapshot = tmp_path / "esol-test.csv"
    moleculenet.write_snapshot(dataset, snapshot)
    version = await prov_repo.create_dataset_version(
        session,
        source_id="moleculenet",
        dataset_name=spec.dataset_name,
        version="test.1",
        content_hash=dataset.content_hash,
        record_count=len(dataset),
        retrieved_at=utcnow(),
        retrieval_params={"snapshot_path": str(snapshot), "dataset": "esol"},
        quality=dataset.quality(),
    )
    await session.commit()
    return version.id


class TestBaselineTraining:
    async def test_model_trains_scores_and_round_trips(self, tmp_path: Path) -> None:
        """Serving a saved artifact must reproduce the in-memory model exactly."""
        snapshot = tmp_path / "data.csv"
        snapshot.write_text(
            "smiles,label,scaffold,inchikey,original_smiles\n"
            + "".join(
                f"{smiles},{label},,{index},{smiles}\n"
                for index, (smiles, label) in enumerate(TRAINING_ROWS)
            )
        )
        dataset = load_dataset_from_snapshot(
            snapshot, name="esol", endpoint="solubility", task_type="regression", units="log mol/L"
        )
        split = scaffold_split(dataset.scaffolds, train_fraction=0.7, valid_fraction=0.15)

        model, report = BaselineModel.train(
            dataset,
            split.train_indices,
            family="random_forest",
            hyperparameters={"n_estimators": 40},
        )
        assert report.n_train == len(split.train_indices)
        assert report.feature_count and report.feature_count > 2000

        before = model.predict_smiles(["CC(=O)Oc1ccccc1C(=O)O"])
        assert before.values is not None
        assert before.uncertainty is not None  # a forest reports ensemble spread

        path = tmp_path / "model.joblib"
        model.save(path)
        reloaded = BaselineModel.load(path)
        after = reloaded.predict_smiles(["CC(=O)Oc1ccccc1C(=O)O"])
        assert np.allclose(before.values, after.values)

    async def test_feature_importances_are_named(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "data.csv"
        snapshot.write_text(
            "smiles,label,scaffold,inchikey,original_smiles\n"
            + "".join(
                f"{smiles},{label},,{index},{smiles}\n"
                for index, (smiles, label) in enumerate(TRAINING_ROWS)
            )
        )
        dataset = load_dataset_from_snapshot(
            snapshot, name="esol", endpoint="solubility", task_type="regression"
        )
        split = scaffold_split(dataset.scaffolds, train_fraction=0.7, valid_fraction=0.15)
        model, _ = BaselineModel.train(
            dataset, split.train_indices, hyperparameters={"n_estimators": 20}
        )
        importances = model.feature_importances(top_k=5)
        assert importances
        assert all(item["feature"].startswith(("morgan", "descriptor:")) for item in importances)


@pytest.mark.slow
class TestGraphModel:
    async def test_gnn_trains_and_round_trips(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "data.csv"
        snapshot.write_text(
            "smiles,label,scaffold,inchikey,original_smiles\n"
            + "".join(
                f"{smiles},{label},,{index},{smiles}\n"
                for index, (smiles, label) in enumerate(TRAINING_ROWS)
            )
        )
        dataset = load_dataset_from_snapshot(
            snapshot, name="esol", endpoint="solubility", task_type="regression"
        )
        split = scaffold_split(dataset.scaffolds, train_fraction=0.7, valid_fraction=0.15)

        model, report = GraphModel.train(
            dataset,
            split.train_indices,
            split.valid_indices,
            hyperparameters={"max_epochs": 6, "hidden_dim": 32, "n_layers": 2, "patience": 4},
        )
        assert report.history  # training actually ran epochs
        assert any("parameters=" in note for note in report.notes)

        batch = model.predict_smiles(["CCO", "c1ccccc1"], mc_dropout_passes=3)
        assert batch.values is not None and len(batch.values) == 2
        assert batch.uncertainty is not None  # MC dropout spread

        path = tmp_path / "model.pt"
        model.save(path)
        reloaded = GraphModel.load(path)
        again = reloaded.predict_smiles(["CCO", "c1ccccc1"], mc_dropout_passes=1)
        # The deterministic (eval-mode) pass must reproduce exactly.
        assert np.allclose(
            model.predict_smiles(["CCO", "c1ccccc1"], mc_dropout_passes=1).values,
            again.values,
            atol=1e-5,
        )

    async def test_checkpoint_with_a_stale_feature_version_is_refused(self, tmp_path: Path) -> None:
        """Loading weights trained on different graph features would silently mispredict."""
        import json

        import torch

        from theragraph.core.errors import ModelNotAvailableError

        path = tmp_path / "model.pt"
        torch.save(
            {
                "state_dict": {},
                "hyperparameters": {"hidden_dim": 32, "n_layers": 2, "dropout": 0.1},
                "label_mean": 0.0,
                "label_std": 1.0,
                "graph_feature_version": "0",
            },
            path,
        )
        path.with_suffix(".meta.json").write_text(
            json.dumps({"family": "gin", "task_type": "regression", "endpoint": "solubility"})
        )
        with pytest.raises(ModelNotAvailableError, match="feature version"):
            GraphModel.load(path)


class TestRegistry:
    async def test_artifact_hash_mismatch_is_detected(
        self, session: AsyncSession, settings: Settings, tmp_path: Path
    ) -> None:
        """A swapped artifact would invalidate every prediction already recorded."""
        from theragraph.core.errors import ModelNotAvailableError

        snapshot = tmp_path / "data.csv"
        snapshot.write_text(
            "smiles,label,scaffold,inchikey,original_smiles\n"
            + "".join(
                f"{smiles},{label},,{index},{smiles}\n"
                for index, (smiles, label) in enumerate(TRAINING_ROWS)
            )
        )
        dataset = load_dataset_from_snapshot(
            snapshot, name="esol", endpoint="solubility", task_type="regression"
        )
        split = scaffold_split(dataset.scaffolds, train_fraction=0.7, valid_fraction=0.15)
        model, _ = BaselineModel.train(
            dataset, split.train_indices, hyperparameters={"n_estimators": 10}
        )

        version = compute_version(
            dataset_content_hash="x",
            hyperparameters={},
            feature_spec=model.feature_spec_dict,
            code_version="test",
            split_strategy="scaffold",
        )
        artifact = await save_artifacts(
            model,
            Calibrator(method="none"),
            endpoint="solubility",
            family="random_forest",
            version=version,
            local_root=settings.model_registry_dir,
            training_reference=TrainingReference(smiles=dataset.smiles[:5], label_std=1.0),
        )
        row = await register_model(
            session,
            name="solubility-random_forest",
            version=version,
            family="random_forest",
            endpoint="solubility",
            task_type="regression",
            framework="scikit-learn",
            framework_version="1.0",
            hyperparameters={},
            feature_spec=model.feature_spec_dict,
            metrics={},
            calibration={},
            training_summary={},
            artifact=artifact,
            training_dataset_version_id=None,
            experiment_id=None,
            split_strategy="scaffold",
        )
        # Loading works while the artifact matches.
        loaded = await load_model(row, local_root=settings.model_registry_dir)
        assert loaded.training_reference is not None

        # Tamper with the file: loading must now refuse.
        Path(artifact["local_path"]).write_bytes(b"corrupted")
        with pytest.raises(ModelNotAvailableError, match="hash mismatch"):
            await load_model(row, local_root=settings.model_registry_dir)

    async def test_missing_artifact_is_reported_clearly(
        self, session: AsyncSession, settings: Settings
    ) -> None:
        from theragraph.core.errors import ModelNotAvailableError

        row = await ml_repo.upsert_model(
            session,
            name="ghost",
            version="1",
            family="random_forest",
            endpoint="solubility",
            task_type="regression",
            framework="scikit-learn",
            artifact_uri="file:///nonexistent/model.joblib",
        )
        with pytest.raises(ModelNotAvailableError, match="not found"):
            await load_model(row, local_root=settings.model_registry_dir)


@pytest.mark.slow
class TestFullTrainingPipeline:
    async def test_pipeline_registers_models_evaluations_and_an_experiment(
        self, session: AsyncSession, settings: Settings, tmp_path: Path
    ) -> None:
        dataset_version_id = await _register_dataset(session, settings, tmp_path)

        outcome = await train_endpoint(
            session,
            settings,
            TrainingConfig(
                endpoint="solubility",
                dataset_key="esol",
                families=("random_forest",),
                run_cross_validation=True,
                cv_folds=3,
                run_random_split_reference=True,
                baseline_hyperparameters={"n_estimators": 25},
            ),
        )
        await session.commit()

        assert outcome.experiment_id.startswith("EXP-")
        assert outcome.artifacts

        models = await ml_repo.list_models(session, endpoint="solubility")
        assert models
        model = models[0]
        assert model.training_dataset_version_id == dataset_version_id
        assert model.artifact_hash
        assert model.status == "active"

        # Scaffold metrics are the headline; the random split is the optimism reference.
        assert model.metrics["scaffold"]["rmse"] is not None
        assert model.metrics["random"]["rmse"] is not None
        assert model.metrics["cross_validation"]["folds"] == 3

        evaluations = await ml_repo.list_evaluations(session, model_id=model.id)
        strategies = {evaluation.split_strategy for evaluation in evaluations}
        assert {"scaffold", "random", "cv_scaffold"} <= strategies

        scaffold = next(e for e in evaluations if e.split_strategy == "scaffold")
        assert scaffold.error_analysis["worst_predictions"]
        assert scaffold.prediction_distribution["count"] > 0

        # A bootstrap interval is reported only when the fold is large enough for one to
        # mean anything. Below ten molecules it is correctly absent rather than a
        # meaningless number -- assert the rule, not a fixed outcome.
        rmse = scaffold.metrics["rmse"]
        assert rmse["value"] is not None
        if (scaffold.n_samples or 0) >= 10:
            assert rmse["ci_low"] is not None
        else:
            assert rmse["ci_low"] is None

        experiment = await ml_repo.get_experiment(session, outcome.experiment_id)
        assert experiment is not None
        assert experiment.status == "succeeded"
        assert experiment.environment["rdkit"]
        assert experiment.comparison["optimism_gap"]

    async def test_random_split_looks_better_than_scaffold(
        self, session: AsyncSession, settings: Settings, tmp_path: Path
    ) -> None:
        """The core scientific claim of the evaluation protocol, measured."""
        await _register_dataset(session, settings, tmp_path)
        outcome = await train_endpoint(
            session,
            settings,
            TrainingConfig(
                endpoint="solubility",
                dataset_key="esol",
                families=("random_forest",),
                run_cross_validation=False,
                baseline_hyperparameters={"n_estimators": 25},
            ),
        )
        gap = outcome.comparison["optimism_gap"]["random_forest"]
        assert gap["metric"] == "rmse"
        assert gap["absolute_gap"] >= 0
        assert gap["interpretation"]

    async def test_trained_model_serves_predictions_with_applicability(
        self, session: AsyncSession, settings: Settings, tmp_path: Path, seeded: dict[str, Any]
    ) -> None:
        await _register_dataset(session, settings, tmp_path)
        await train_endpoint(
            session,
            settings,
            TrainingConfig(
                endpoint="solubility",
                dataset_key="esol",
                families=("random_forest",),
                run_cross_validation=False,
                run_random_split_reference=False,
                baseline_hyperparameters={"n_estimators": 25},
            ),
        )
        await session.commit()

        inference = InferenceService(settings)
        endpoints = await inference.available_endpoints(session)
        assert [item["endpoint"] for item in endpoints] == ["solubility"]

        result = await inference.predict(
            session,
            molecules=[(seeded["compound_ids"][0], seeded["compound_smiles"][0])],
            endpoints=["solubility"],
        )
        prediction = result.predictions[0]
        assert prediction.value is not None
        assert prediction.uncertainty is not None
        assert prediction.confidence is not None
        assert prediction.in_domain is not None
        assert prediction.prediction_id  # persisted

        # A wildly different molecule must be flagged as extrapolation.
        exotic = await inference.predict(
            session,
            molecules=[(None, "C1=CC2=CC3=CC=CC4=C3C(=C2C=C1)C=CC=C4")],
            endpoints=["solubility"],
            persist=False,
        )
        assert exotic.predictions[0].in_domain is False
        assert "out_of_domain" in exotic.predictions[0].flags

    async def test_prediction_writes_are_idempotent(
        self, session: AsyncSession, settings: Settings, tmp_path: Path, seeded: dict[str, Any]
    ) -> None:
        await _register_dataset(session, settings, tmp_path)
        await train_endpoint(
            session,
            settings,
            TrainingConfig(
                endpoint="solubility",
                dataset_key="esol",
                families=("random_forest",),
                run_cross_validation=False,
                run_random_split_reference=False,
                baseline_hyperparameters={"n_estimators": 20},
            ),
        )
        await session.commit()

        inference = InferenceService(settings)
        molecules = [(seeded["compound_ids"][0], seeded["compound_smiles"][0])]
        await inference.predict(session, molecules=molecules, endpoints=["solubility"])
        first = await ml_repo.count_predictions(session)
        await inference.predict(
            session, molecules=molecules, endpoints=["solubility"], use_cache=False
        )
        assert await ml_repo.count_predictions(session) == first
