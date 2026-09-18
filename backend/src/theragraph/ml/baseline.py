"""Baseline models: Random Forest and gradient boosting over molecular features.

These are the honest baseline for molecular property prediction, and they are hard to
beat: count-based Morgan fingerprints concatenated with physicochemical descriptors,
fed to a tree ensemble, routinely matches or beats graph networks on datasets of a few
thousand molecules. Shipping them as a first-class, evaluated model (rather than a
throwaway comparison) is the point.

Uncertainty comes free with a Random Forest: the spread of per-tree predictions is a
usable epistemic estimate, and for classification the spread of per-tree class votes
tells you when the ensemble is internally split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

from theragraph.core.clock import Stopwatch
from theragraph.core.errors import ModelNotAvailableError, ValidationError
from theragraph.core.logging import get_logger
from theragraph.ml.base import PredictionBatch, TaskType, TrainingReport
from theragraph.ml.datasets import FeatureSpec, MolecularDataset, Representation

log = get_logger(__name__)

Family = Literal["random_forest", "xgboost"]

DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 1337,
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 1337,
    "tree_method": "hist",
}


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:  # pragma: no cover - optional dependency
        return False


@dataclass
class BaselineModel:
    """A fitted tree-ensemble model plus everything needed to serve it."""

    family: Family
    task_type: TaskType
    endpoint: str
    feature_spec: FeatureSpec
    estimator: Any = None
    hyperparameters: dict[str, Any] = None  # type: ignore[assignment]
    training_labels: np.ndarray | None = None
    units: str | None = None

    def __post_init__(self) -> None:
        if self.hyperparameters is None:
            self.hyperparameters = {}

    # --------------------------------------------------------------------- fit
    @classmethod
    def train(
        cls,
        dataset: MolecularDataset,
        train_indices: np.ndarray,
        *,
        family: Family = "random_forest",
        representation: Representation = "fingerprint+descriptors",
        fingerprint_kind: str = "morgan_r2_2048",
        hyperparameters: dict[str, Any] | None = None,
        valid_indices: np.ndarray | None = None,
    ) -> tuple[BaselineModel, TrainingReport]:
        """Fit on the given training indices only.

        The feature scaler is fitted on the training subset, never on the full dataset:
        fitting on all rows leaks test-set statistics into the model and quietly
        inflates the reported score.
        """
        if family == "xgboost" and not xgboost_available():
            raise ModelNotAvailableError(
                "xgboost is not installed; install the 'xgb' extra or use random_forest"
            )

        train_subset = dataset.subset(train_indices)
        features, spec = train_subset.build_features(
            representation, fingerprint_kind=fingerprint_kind, fit_scaler=True
        )
        labels = np.asarray(train_subset.labels, dtype=np.float64)

        params = dict(DEFAULT_RF_PARAMS if family == "random_forest" else DEFAULT_XGB_PARAMS)
        params.update(hyperparameters or {})

        estimator = _build_estimator(family, dataset.task_type, params, labels)

        stopwatch = Stopwatch()
        with stopwatch:
            if dataset.task_type == "classification":
                estimator.fit(features, (labels >= 0.5).astype(np.int32))
            else:
                estimator.fit(features, labels)

        model = cls(
            family=family,
            task_type=dataset.task_type,
            endpoint=dataset.endpoint,
            feature_spec=spec,
            estimator=estimator,
            hyperparameters=params,
            training_labels=labels.astype(np.float32),
            units=dataset.units,
        )
        report = TrainingReport(
            family=family,
            task_type=dataset.task_type,
            n_train=len(train_indices),
            n_valid=len(valid_indices) if valid_indices is not None else 0,
            hyperparameters=params,
            training_seconds=stopwatch.elapsed_ms / 1000,
            feature_count=int(features.shape[1]),
            notes=[
                f"representation={representation}",
                f"fingerprint={fingerprint_kind}",
            ],
        )
        log.info(
            "baseline.trained",
            family=family,
            endpoint=dataset.endpoint,
            n_train=len(train_indices),
            features=int(features.shape[1]),
            seconds=round(stopwatch.elapsed_ms / 1000, 2),
        )
        return model, report

    # ----------------------------------------------------------------- predict
    def predict_features(self, features: np.ndarray) -> PredictionBatch:
        if self.estimator is None:
            raise ModelNotAvailableError("model has no fitted estimator")

        if self.task_type == "classification":
            probabilities = self.estimator.predict_proba(features)[:, 1]
            uncertainty = self._ensemble_spread(features, classification=True)
            return PredictionBatch(
                probabilities=probabilities.astype(np.float32),
                uncertainty=uncertainty,
                n_estimators=self._n_estimators(),
            )

        values = self.estimator.predict(features)
        uncertainty = self._ensemble_spread(features, classification=False)
        return PredictionBatch(
            values=values.astype(np.float32),
            uncertainty=uncertainty,
            n_estimators=self._n_estimators(),
        )

    def predict_smiles(self, smiles: list[str]) -> PredictionBatch:
        if not smiles:
            return PredictionBatch()
        features = self.feature_spec.featurize_many(smiles)
        return self.predict_features(features)

    def _n_estimators(self) -> int | None:
        return getattr(self.estimator, "n_estimators", None)

    def _ensemble_spread(self, features: np.ndarray, *, classification: bool) -> np.ndarray | None:
        """Standard deviation across ensemble members.

        Available for Random Forests (individual trees are exposed). Gradient boosting
        builds an additive sequence, not independent members, so per-member spread is
        not meaningful and ``None`` is returned rather than a fabricated number.
        """
        trees = getattr(self.estimator, "estimators_", None)
        if not trees:
            return None
        try:
            if classification:
                votes = np.stack([tree.predict_proba(features)[:, 1] for tree in trees], axis=0)
            else:
                votes = np.stack([tree.predict(features) for tree in trees], axis=0)
        except Exception:  # pragma: no cover - non-uniform estimator API
            return None
        return votes.std(axis=0).astype(np.float32)

    def feature_importances(self, *, top_k: int = 25) -> list[dict[str, Any]]:
        """Most influential features, named where possible.

        Fingerprint bits are opaque by nature ("bit 1043"), so they are labelled as
        such; descriptor features get their real names. This is honest attribution
        rather than a pretty but meaningless chart.
        """
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            return []
        names = self._feature_names()
        order = np.argsort(-np.asarray(importances))[:top_k]
        return [
            {
                "feature": names[index] if index < len(names) else f"feature_{index}",
                "importance": round(float(importances[index]), 6),
            }
            for index in order
        ]

    def _feature_names(self) -> list[str]:
        names: list[str] = []
        spec = self.feature_spec
        if spec.fingerprint_bits and "fingerprint" in spec.representation:
            names.extend(
                f"{spec.fingerprint_kind}:bit{index}" for index in range(spec.fingerprint_bits)
            )
        if "descriptors" in spec.representation:
            names.extend(f"descriptor:{name}" for name in spec.descriptor_names)
        return names

    # -------------------------------------------------------------- persistence
    @property
    def feature_spec_dict(self) -> dict[str, Any]:
        return self.feature_spec.to_dict()

    def save(self, path: Path) -> dict[str, Any]:
        """Persist the estimator and its metadata side by side."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "training_labels": self.training_labels,
            },
            path,
            compress=3,
        )
        metadata = {
            "family": self.family,
            "task_type": self.task_type,
            "endpoint": self.endpoint,
            "units": self.units,
            "hyperparameters": self.hyperparameters,
            "feature_spec": self.feature_spec.to_dict(),
        }
        path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
        return metadata

    @classmethod
    def load(cls, path: Path) -> BaselineModel:
        path = Path(path)
        metadata_path = path.with_suffix(".meta.json")
        if not path.exists() or not metadata_path.exists():
            raise ModelNotAvailableError(f"model artifact missing at {path}")
        payload = joblib.load(path)
        metadata = json.loads(metadata_path.read_text())
        return cls(
            family=metadata["family"],
            task_type=metadata["task_type"],
            endpoint=metadata["endpoint"],
            feature_spec=FeatureSpec.from_dict(metadata["feature_spec"]),
            estimator=payload["estimator"],
            hyperparameters=metadata.get("hyperparameters") or {},
            training_labels=payload.get("training_labels"),
            units=metadata.get("units"),
        )


def _build_estimator(
    family: Family, task_type: TaskType, params: dict[str, Any], labels: np.ndarray
) -> Any:
    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if task_type == "classification":
            # Molecular toxicity/permeability sets are strongly imbalanced (ClinTox is
            # ~5% positive), so class weights are on by default.
            return RandomForestClassifier(class_weight="balanced_subsample", **params)
        return RandomForestRegressor(**params)

    if family == "xgboost":
        import xgboost as xgb

        if task_type == "classification":
            positives = float((labels >= 0.5).sum())
            negatives = float(len(labels) - positives)
            scale = negatives / positives if positives else 1.0
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=round(scale, 4),
                **params,
            )
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **params)

    raise ValidationError(f"unknown baseline family: {family}")
