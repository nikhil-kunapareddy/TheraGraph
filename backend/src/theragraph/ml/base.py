"""Common model interface.

Every model -- tree ensemble or neural network -- exposes the same three things:
``fit``, ``predict`` and ``save``/``load``, plus a serialisable ``feature_spec``.
The inference service, the evaluation harness and the training CLI are written against
this interface only, so adding a model family touches one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

TaskType = Literal["classification", "regression"]


@dataclass
class PredictionBatch:
    """Model output for a batch of molecules.

    ``uncertainty`` is an epistemic estimate (ensemble spread or MC-dropout spread), not
    a calibrated error bar. It is reported as such everywhere it surfaces.
    """

    values: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    n_estimators: int | None = None

    def __len__(self) -> int:
        primary = self.values if self.values is not None else self.probabilities
        return 0 if primary is None else len(primary)

    def primary(self) -> np.ndarray:
        if self.probabilities is not None:
            return self.probabilities
        if self.values is not None:
            return self.values
        raise ValueError("prediction batch is empty")


@dataclass
class TrainingReport:
    """What happened during training -- persisted on the experiment row."""

    family: str
    task_type: TaskType
    n_train: int
    n_valid: int
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    training_seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    feature_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "task_type": self.task_type,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "hyperparameters": self.hyperparameters,
            "training_seconds": round(self.training_seconds, 3),
            "history": self.history[-200:],
            "notes": self.notes,
            "feature_count": self.feature_count,
        }


@runtime_checkable
class MolecularModel(Protocol):
    family: str
    task_type: TaskType

    def predict_smiles(self, smiles: list[str]) -> PredictionBatch: ...
    def save(self, path: Path) -> dict[str, Any]: ...
    @property
    def feature_spec_dict(self) -> dict[str, Any]: ...
