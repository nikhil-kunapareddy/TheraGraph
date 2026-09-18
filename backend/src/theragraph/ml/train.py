"""Training pipeline: dataset -> splits -> baseline + graph model -> evaluation -> registry.

One command trains and registers an endpoint. The pipeline always does the following,
in this order, because each step depends on the previous one being honest:

1. **Split first, featurise second.** Scalers and label statistics are fitted on the
   training fold only.
2. **Train both families on the identical split.** A comparison across different splits
   is not a comparison.
3. **Calibrate on validation, never on test.** Test data touches nothing but the final
   metric computation.
4. **Evaluate on scaffold split (headline) and random split (optimism reference).**
5. **Cross-validate the baseline** for a confidence interval that reflects split
   variance, not just sampling variance.
6. **Register** experiment, models and evaluations with dataset/model versions linked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.config import Settings
from theragraph.core.clock import Stopwatch, utcnow
from theragraph.core.logging import get_logger
from theragraph.core.storage import ObjectStore
from theragraph.db.repositories import ml as ml_repo
from theragraph.db.repositories import provenance as prov_repo
from theragraph.ingestion import moleculenet
from theragraph.ingestion.pipelines import resolve_snapshot_path
from theragraph.ml import registry
from theragraph.ml.baseline import BaselineModel, xgboost_available
from theragraph.ml.calibration import Calibrator, fit_calibrator, reliability_curve
from theragraph.ml.datasets import MolecularDataset, load_dataset_from_snapshot
from theragraph.ml.evaluation import EvaluationResult, compare_models, evaluate
from theragraph.ml.gnn import GraphModel
from theragraph.ml.splits import (
    SplitResult,
    cross_validation_folds,
    describe_split_difficulty,
    random_split,
    scaffold_split,
)

log = get_logger(__name__)

Family = Literal["random_forest", "xgboost", "gin"]

DEFAULT_FAMILIES: tuple[Family, ...] = ("random_forest", "gin")


@dataclass
class TrainingConfig:
    endpoint: str
    dataset_key: str
    families: tuple[Family, ...] = DEFAULT_FAMILIES
    train_fraction: float = 0.8
    valid_fraction: float = 0.1
    seed: int = 1337
    cv_folds: int = 5
    run_cross_validation: bool = True
    run_random_split_reference: bool = True
    baseline_hyperparameters: dict[str, Any] = field(default_factory=dict)
    graph_hyperparameters: dict[str, Any] = field(default_factory=dict)
    representation: str = "fingerprint+descriptors"
    fingerprint_kind: str = "morgan_r2_2048"
    max_training_rows: int | None = None
    experiment_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "dataset_key": self.dataset_key,
            "families": list(self.families),
            "train_fraction": self.train_fraction,
            "valid_fraction": self.valid_fraction,
            "seed": self.seed,
            "cv_folds": self.cv_folds,
            "run_cross_validation": self.run_cross_validation,
            "run_random_split_reference": self.run_random_split_reference,
            "baseline_hyperparameters": self.baseline_hyperparameters,
            "graph_hyperparameters": self.graph_hyperparameters,
            "representation": self.representation,
            "fingerprint_kind": self.fingerprint_kind,
            "max_training_rows": self.max_training_rows,
        }


@dataclass
class TrainedArtifact:
    family: Family
    model: BaselineModel | GraphModel
    calibrator: Calibrator
    training_report: dict[str, Any]
    evaluations: dict[str, EvaluationResult]
    cv_metrics: list[dict[str, Any]] = field(default_factory=list)
    version: str = ""
    model_id: str | None = None


@dataclass
class TrainingOutcome:
    experiment_id: str
    endpoint: str
    dataset_version_label: str | None
    artifacts: list[TrainedArtifact]
    comparison: dict[str, Any]
    runtime_s: float

    def summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "endpoint": self.endpoint,
            "dataset_version": self.dataset_version_label,
            "runtime_s": round(self.runtime_s, 2),
            "models": [
                {
                    "family": artifact.family,
                    "version": artifact.version,
                    "scaffold": artifact.evaluations["scaffold"].headline()
                    if "scaffold" in artifact.evaluations
                    else None,
                    "random": artifact.evaluations["random"].headline()
                    if "random" in artifact.evaluations
                    else None,
                }
                for artifact in self.artifacts
            ],
            "comparison": self.comparison,
        }


async def load_training_dataset(
    session: AsyncSession, settings: Settings, dataset_key: str
) -> MolecularDataset:
    """Load the current snapshot for a MoleculeNet dataset via its dataset version."""
    spec = moleculenet.get_spec(dataset_key)
    version = await prov_repo.get_current_dataset_version(
        session, moleculenet.SOURCE_ID, spec.dataset_name
    )
    if version is None:
        raise FileNotFoundError(
            f"dataset {dataset_key!r} has not been ingested; run ingestion first"
        )
    snapshot = resolve_snapshot_path(settings, version.retrieval_params or {})
    return load_dataset_from_snapshot(
        snapshot,
        name=spec.key,
        endpoint=spec.endpoint,
        task_type=spec.task_type,
        units=spec.units,
        dataset_version_id=version.id,
        dataset_version_label=version.label,
    )


def _train_family(
    family: Family,
    dataset: MolecularDataset,
    split: SplitResult,
    config: TrainingConfig,
) -> tuple[BaselineModel | GraphModel, dict[str, Any]]:
    if family == "gin":
        graph_model, graph_report = GraphModel.train(
            dataset,
            split.train_indices,
            split.valid_indices,
            hyperparameters=config.graph_hyperparameters,
        )
        return graph_model, graph_report.to_dict()

    model, report = BaselineModel.train(
        dataset,
        split.train_indices,
        family=family,
        representation=config.representation,  # type: ignore[arg-type]
        fingerprint_kind=config.fingerprint_kind,
        hyperparameters=config.baseline_hyperparameters,
        valid_indices=split.valid_indices,
    )
    return model, report.to_dict()


def _predict(model: BaselineModel | GraphModel, smiles: list[str]) -> Any:
    return model.predict_smiles(smiles)


def _evaluate_on_split(
    *,
    model: BaselineModel | GraphModel,
    calibrator: Calibrator,
    dataset: MolecularDataset,
    split: SplitResult,
    training_scaffold_counts: dict[str, int],
    training_smiles: list[str],
    split_label: str,
) -> EvaluationResult:
    test = dataset.subset(split.test_indices)
    batch = _predict(model, test.smiles)

    if dataset.task_type == "classification":
        probabilities = batch.probabilities
        assert probabilities is not None
        predictions = calibrator.apply(probabilities)
    else:
        assert batch.values is not None
        predictions = batch.values

    return evaluate(
        task_type=dataset.task_type,
        split_strategy=split_label,
        split_name="test",
        predictions=np.asarray(predictions),
        actuals=np.asarray(test.labels),
        smiles=test.smiles,
        identifiers=test.identifiers,
        scaffolds=test.scaffolds,
        uncertainties=batch.uncertainty,
        training_smiles=training_smiles,
        training_scaffold_counts=training_scaffold_counts,
    )


def _cross_validate(
    family: Family, dataset: MolecularDataset, config: TrainingConfig
) -> list[dict[str, Any]]:
    """Scaffold-grouped K-fold for the requested family.

    Only run for tree baselines by default: five GNN trainings would dominate the
    pipeline's runtime for a confidence interval the baseline already provides.
    """
    folds = cross_validation_folds(dataset.scaffolds, n_folds=config.cv_folds, seed=config.seed)
    results: list[dict[str, Any]] = []
    for fold in folds:
        model, _ = _train_family(family, dataset, fold, config)
        test = dataset.subset(fold.test_indices)
        batch = _predict(model, test.smiles)
        predictions = batch.probabilities if dataset.task_type == "classification" else batch.values
        assert predictions is not None
        result = evaluate(
            task_type=dataset.task_type,
            split_strategy="cv_scaffold",
            split_name=f"fold{fold.metadata['fold']}",
            predictions=np.asarray(predictions),
            actuals=np.asarray(test.labels),
            smiles=test.smiles,
            identifiers=test.identifiers,
            scaffolds=test.scaffolds,
            uncertainties=batch.uncertainty,
        )
        primary = "roc_auc" if dataset.task_type == "classification" else "rmse"
        results.append(
            {
                "fold": fold.metadata["fold"],
                "n_test": len(fold.test_indices),
                "primary_metric": primary,
                "value": (result.metrics.get(primary) or {}).get("value"),
                "metrics": {
                    key: (value or {}).get("value") if isinstance(value, dict) else value
                    for key, value in result.metrics.items()
                    if key not in ("confusion_matrix",)
                },
            }
        )
    return results


def _aggregate_cv(results: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["value"] for row in results if row.get("value") is not None]
    if not values:
        return {"folds": len(results)}
    array = np.asarray(values, dtype=np.float64)
    return {
        "folds": len(results),
        "primary_metric": results[0]["primary_metric"],
        "mean": round(float(array.mean()), 5),
        "std": round(float(array.std()), 5),
        "min": round(float(array.min()), 5),
        "max": round(float(array.max()), 5),
        "per_fold": [round(float(value), 5) for value in values],
    }


async def train_endpoint(
    session: AsyncSession,
    settings: Settings,
    config: TrainingConfig,
    *,
    store: ObjectStore | None = None,
) -> TrainingOutcome:
    """Train, evaluate and register every configured model family for one endpoint."""
    stopwatch = Stopwatch()
    dataset = await load_training_dataset(session, settings, config.dataset_key)

    if config.max_training_rows and len(dataset) > config.max_training_rows:
        rng = np.random.default_rng(config.seed)
        keep = rng.choice(len(dataset), config.max_training_rows, replace=False)
        dataset = dataset.subset(np.sort(keep))
        log.info("train.subsampled", rows=len(dataset))

    now = utcnow()
    experiment_id, sequence = await ml_repo.allocate_experiment_id(session, now.year)
    experiment = await ml_repo.create_experiment(
        session,
        id=experiment_id,
        sequence=sequence,
        year=now.year,
        name=config.experiment_name
        or f"{config.endpoint} baseline vs graph ({config.dataset_key})",
        description=(
            f"Trains {', '.join(config.families)} on {config.dataset_key} with a "
            "scaffold split; random split reported as an optimism reference."
        ),
        endpoint=config.endpoint,
        status="running",
        dataset_version_id=dataset.dataset_version_id,
        parameters=config.to_dict(),
        started_at=now,
        git_sha=registry.git_sha(),
        random_seed=config.seed,
        environment=_environment_snapshot(),
    )
    await session.flush()

    try:
        with stopwatch:
            scaffold = scaffold_split(
                dataset.scaffolds,
                train_fraction=config.train_fraction,
                valid_fraction=config.valid_fraction,
                seed=config.seed,
            )
            random = (
                random_split(
                    len(dataset),
                    train_fraction=config.train_fraction,
                    valid_fraction=config.valid_fraction,
                    seed=config.seed,
                )
                if config.run_random_split_reference
                else None
            )

            train_subset = dataset.subset(scaffold.train_indices)
            training_scaffold_counts: dict[str, int] = {}
            for scaffold_key in train_subset.scaffolds:
                if scaffold_key:
                    training_scaffold_counts[scaffold_key] = (
                        training_scaffold_counts.get(scaffold_key, 0) + 1
                    )

            artifacts: list[TrainedArtifact] = []
            for family in config.families:
                if family == "xgboost" and not xgboost_available():
                    log.warning("train.skipping_family", family=family, reason="not installed")
                    continue

                model, training_report = _train_family(family, dataset, scaffold, config)

                # Calibration is fitted on the validation fold only.
                calibrator = Calibrator(method="none")
                if dataset.task_type == "classification" and len(scaffold.valid_indices):
                    valid = dataset.subset(scaffold.valid_indices)
                    valid_batch = _predict(model, valid.smiles)
                    if valid_batch.probabilities is not None:
                        calibrator = fit_calibrator(
                            valid_batch.probabilities, np.asarray(valid.labels)
                        )

                evaluations: dict[str, EvaluationResult] = {
                    "scaffold": _evaluate_on_split(
                        model=model,
                        calibrator=calibrator,
                        dataset=dataset,
                        split=scaffold,
                        training_scaffold_counts=training_scaffold_counts,
                        training_smiles=train_subset.smiles,
                        split_label="scaffold",
                    )
                }

                if random is not None:
                    # Retrain on the random split so the reference is a real
                    # random-split model, not a scaffold-trained model scored on
                    # random test rows (which would leak training molecules).
                    random_model, _ = _train_family(family, dataset, random, config)
                    random_calibrator = Calibrator(method="none")
                    if dataset.task_type == "classification" and len(random.valid_indices):
                        random_valid = dataset.subset(random.valid_indices)
                        random_valid_batch = _predict(random_model, random_valid.smiles)
                        if random_valid_batch.probabilities is not None:
                            random_calibrator = fit_calibrator(
                                random_valid_batch.probabilities,
                                np.asarray(random_valid.labels),
                            )
                    evaluations["random"] = _evaluate_on_split(
                        model=random_model,
                        calibrator=random_calibrator,
                        dataset=dataset,
                        split=random,
                        training_scaffold_counts={},
                        training_smiles=dataset.subset(random.train_indices).smiles,
                        split_label="random",
                    )

                cv_metrics: list[dict[str, Any]] = []
                if config.run_cross_validation and family != "gin":
                    cv_metrics = _cross_validate(family, dataset, config)

                version = registry.compute_version(
                    dataset_content_hash=dataset.dataset_version_label or dataset.name,
                    hyperparameters=training_report.get("hyperparameters", {}),
                    feature_spec=model.feature_spec_dict,
                    code_version=registry.code_version(),
                    split_strategy="scaffold",
                )
                artifacts.append(
                    TrainedArtifact(
                        family=family,
                        model=model,
                        calibrator=calibrator,
                        training_report=training_report,
                        evaluations=evaluations,
                        cv_metrics=cv_metrics,
                        version=version,
                    )
                )

            comparison = compare_models(
                {
                    artifact.family: artifact.evaluations["scaffold"]
                    for artifact in artifacts
                    if "scaffold" in artifact.evaluations
                }
            )
            comparison["split_difficulty"] = describe_split_difficulty(dataset.scaffolds, scaffold)
            comparison["optimism_gap"] = _optimism_gap(artifacts)

            # ------------------------------------------------------- persistence
            spec = moleculenet.get_spec(config.dataset_key)
            for artifact in artifacts:
                stored = await registry.save_artifacts(
                    artifact.model,
                    artifact.calibrator,
                    endpoint=config.endpoint,
                    family=artifact.family,
                    version=artifact.version,
                    local_root=settings.model_registry_dir,
                    store=store,
                    training_reference=registry.TrainingReference(
                        smiles=train_subset.smiles,
                        label_std=float(np.asarray(train_subset.labels).std()),
                        label_mean=float(np.asarray(train_subset.labels).mean()),
                        n_training_total=len(train_subset),
                        scaffold_counts=training_scaffold_counts,
                    ),
                )
                scaffold_eval = artifact.evaluations["scaffold"]
                calibration_payload = {
                    **artifact.calibrator.to_dict(),
                    "reliability": scaffold_eval.calibration,
                }
                model_row = await registry.register_model(
                    session,
                    name=f"{config.endpoint}-{artifact.family}",
                    version=artifact.version,
                    family=artifact.family,
                    endpoint=config.endpoint,
                    task_type=dataset.task_type,
                    framework="pytorch"
                    if artifact.family == "gin"
                    else "scikit-learn"
                    if artifact.family == "random_forest"
                    else "xgboost",
                    framework_version=_framework_version(artifact.family),
                    hyperparameters=artifact.training_report.get("hyperparameters", {}),
                    feature_spec=artifact.model.feature_spec_dict,
                    metrics={
                        "scaffold": _flat_metrics(scaffold_eval),
                        "random": _flat_metrics(artifact.evaluations["random"])
                        if "random" in artifact.evaluations
                        else None,
                        "cross_validation": _aggregate_cv(artifact.cv_metrics)
                        if artifact.cv_metrics
                        else None,
                    },
                    calibration=calibration_payload,
                    training_summary=artifact.training_report,
                    artifact=stored,
                    training_dataset_version_id=dataset.dataset_version_id,
                    experiment_id=experiment_id,
                    split_strategy="scaffold",
                    units=dataset.units,
                    description=spec.description,
                    positive_class_meaning=spec.positive_class_meaning,
                    git_sha=registry.git_sha(),
                    random_seed=config.seed,
                    training_runtime_s=artifact.training_report.get("training_seconds"),
                )
                artifact.model_id = model_row.id

                for split_label, result in artifact.evaluations.items():
                    await ml_repo.upsert_evaluation(
                        session,
                        model_id=model_row.id,
                        dataset_version_id=dataset.dataset_version_id,
                        split_strategy=split_label,
                        split_name="test",
                        n_samples=result.n_samples,
                        metrics=result.metrics,
                        calibration=result.calibration,
                        error_analysis=result.error_analysis,
                        prediction_distribution=result.prediction_distribution,
                        per_fold_metrics=artifact.cv_metrics if split_label == "scaffold" else [],
                        notes=(
                            "Headline protocol: scaffold split, no scaffold shared between folds."
                            if split_label == "scaffold"
                            else "Random split shown only to quantify optimism."
                        ),
                    )
                if artifact.cv_metrics:
                    await ml_repo.upsert_evaluation(
                        session,
                        model_id=model_row.id,
                        dataset_version_id=dataset.dataset_version_id,
                        split_strategy="cv_scaffold",
                        split_name="aggregate",
                        n_samples=sum(row["n_test"] for row in artifact.cv_metrics),
                        metrics=_aggregate_cv(artifact.cv_metrics),
                        per_fold_metrics=artifact.cv_metrics,
                        notes=f"{config.cv_folds}-fold scaffold-grouped cross-validation.",
                    )

        finished = utcnow()
        experiment.status = "succeeded"
        experiment.finished_at = finished
        experiment.runtime_s = stopwatch.elapsed_ms / 1000
        experiment.metrics = {
            artifact.family: _flat_metrics(artifact.evaluations["scaffold"])
            for artifact in artifacts
            if "scaffold" in artifact.evaluations
        }
        experiment.comparison = comparison
        experiment.artifacts = {
            artifact.family: {
                "version": artifact.version,
                "model_id": artifact.model_id,
            }
            for artifact in artifacts
        }
        await session.flush()

        outcome = TrainingOutcome(
            experiment_id=experiment_id,
            endpoint=config.endpoint,
            dataset_version_label=dataset.dataset_version_label,
            artifacts=artifacts,
            comparison=comparison,
            runtime_s=stopwatch.elapsed_ms / 1000,
        )
        log.info("train.completed", **outcome.summary()["models"][0] if outcome.artifacts else {})
        return outcome

    except Exception as exc:
        experiment.status = "failed"
        experiment.error = str(exc)[:4000]
        experiment.finished_at = utcnow()
        await session.flush()
        raise


def _flat_metrics(result: EvaluationResult) -> dict[str, Any]:
    """Flatten ``{metric: {value, ci_low, ci_high}}`` for compact storage/display."""
    flat: dict[str, Any] = {"n_samples": result.n_samples}
    for key, value in result.metrics.items():
        if isinstance(value, dict) and "value" in value:
            flat[key] = value["value"]
            if value.get("ci_low") is not None:
                flat[f"{key}_ci"] = [value["ci_low"], value["ci_high"]]
        else:
            flat[key] = value
    if result.calibration:
        for key in ("ece", "brier", "coverage", "uncertainty_error_correlation"):
            if result.calibration.get(key) is not None:
                flat[key] = result.calibration[key]
    if result.applicability:
        flat["out_of_domain_fraction"] = result.applicability.get("out_of_domain_fraction")
    return flat


def _optimism_gap(artifacts: list[TrainedArtifact]) -> dict[str, Any]:
    """How much better does a random split make each model look?

    This number is the entire argument for scaffold splitting, so it is computed and
    displayed rather than left as a claim in the documentation.
    """
    gaps: dict[str, Any] = {}
    for artifact in artifacts:
        scaffold_eval = artifact.evaluations.get("scaffold")
        random_eval = artifact.evaluations.get("random")
        if scaffold_eval is None or random_eval is None:
            continue
        primary = "roc_auc" if scaffold_eval.task_type == "classification" else "rmse"
        scaffold_value = (scaffold_eval.metrics.get(primary) or {}).get("value")
        random_value = (random_eval.metrics.get(primary) or {}).get("value")
        if scaffold_value is None or random_value is None:
            continue
        gaps[artifact.family] = {
            "metric": primary,
            "scaffold": scaffold_value,
            "random": random_value,
            "absolute_gap": round(abs(random_value - scaffold_value), 5),
            "relative_gap_pct": round(
                abs(random_value - scaffold_value) / abs(scaffold_value) * 100, 2
            )
            if scaffold_value
            else None,
            "interpretation": (
                "random split looks better; scaffold split is the realistic estimate"
                if (
                    random_value > scaffold_value
                    if primary == "roc_auc"
                    else random_value < scaffold_value
                )
                else "no optimism observed on this dataset"
            ),
        }
    return gaps


def _framework_version(family: Family) -> str:
    if family == "gin":
        import torch

        return str(torch.__version__)
    if family == "xgboost":
        try:
            import xgboost

            return str(xgboost.__version__)
        except Exception:  # pragma: no cover
            return "unknown"
    import sklearn

    return str(sklearn.__version__)


def _environment_snapshot() -> dict[str, Any]:
    """Library versions recorded on the experiment -- part of reproducibility."""
    import platform
    import sys

    import numpy
    import sklearn
    import torch

    from theragraph.chem.molecule import rdkit_version

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "rdkit": rdkit_version(),
    }


def reliability_of(result: EvaluationResult) -> dict[str, Any]:
    if result.task_type != "classification":
        return {}
    return reliability_curve(
        np.asarray(result.prediction_distribution.get("histogram", {}).get("counts", [])),
        np.asarray([]),
    )
