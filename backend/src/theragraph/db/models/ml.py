"""ML entities: models, experiments, evaluations and predictions.

Models are tracked *independently* of datasets: a :class:`MLModel` row points at the
:class:`~theragraph.db.models.provenance.DatasetVersion` it was trained on, so the same
architecture retrained on a newer snapshot is a new model version with its own metrics,
and an old research run can still say exactly which weights produced its numbers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from theragraph.db.base import Base, TimestampMixin, ULIDPrimaryKey
from theragraph.db.types import JSONColumn, UTCDateTime


class MLModel(Base, ULIDPrimaryKey, TimestampMixin):
    """A trained, versioned model artifact plus its headline metrics."""

    __tablename__ = "ml_model"
    __table_args__ = (
        sa.UniqueConstraint("name", "version"),
        sa.Index("ix_ml_model_endpoint_status", "endpoint", "status"),
        sa.CheckConstraint("task_type IN ('classification', 'regression')", name="task_type_valid"),
    )

    name: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    version: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    family: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    """``random_forest``, ``xgboost``, ``gin`` (graph neural network)."""

    endpoint: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    """What it predicts: ``solubility``, ``bbb_permeability``, ``toxicity``,
    ``bioactivity``."""

    task_type: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    framework: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    framework_version: Mapped[str | None] = mapped_column(sa.String(32))
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="active")
    """``training`` | ``active`` | ``shadow`` | ``archived``."""

    description: Mapped[str | None] = mapped_column(sa.Text())
    target_column: Mapped[str | None] = mapped_column(sa.String(64))
    units: Mapped[str | None] = mapped_column(sa.String(64))
    positive_class_meaning: Mapped[str | None] = mapped_column(sa.String(255))

    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    feature_spec: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Exactly how inputs are featurised (fingerprint kind, radius, bits, descriptor
    list and order). Inference re-derives features from this spec, so train/serve skew
    is structurally impossible."""

    split_strategy: Mapped[str | None] = mapped_column(sa.String(64))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    calibration: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    training_summary: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)

    artifact_uri: Mapped[str | None] = mapped_column(sa.String(1024))
    artifact_hash: Mapped[str | None] = mapped_column(sa.String(64))
    artifact_size_bytes: Mapped[int | None] = mapped_column(sa.BigInteger())

    training_dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )
    experiment_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("experiment.id", ondelete="SET NULL"), index=True
    )
    trained_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    training_runtime_s: Mapped[float | None] = mapped_column(sa.Float())
    git_sha: Mapped[str | None] = mapped_column(sa.String(40))
    random_seed: Mapped[int | None] = mapped_column(sa.Integer())

    evaluations: Mapped[list[ModelEvaluation]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )

    @property
    def ref(self) -> str:
        """``solubility-gin@2026.09.1`` -- how a model is quoted in a report."""
        return f"{self.name}@{self.version}"


class ModelEvaluation(Base, ULIDPrimaryKey, TimestampMixin):
    """Evaluation of one model under one split protocol.

    A model gets several of these -- random split, scaffold split, cross-validation --
    because a random split flatters molecular models badly. Scaffold-split numbers are
    the ones surfaced by default in the UI.
    """

    __tablename__ = "model_evaluation"
    __table_args__ = (
        sa.UniqueConstraint("model_id", "split_strategy", "split_name"),
        sa.Index("ix_model_evaluation_model", "model_id", "split_strategy"),
    )

    model_id: Mapped[str] = mapped_column(
        sa.ForeignKey("ml_model.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )
    split_strategy: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    """``scaffold``, ``random``, ``cv_scaffold``, ``temporal``."""

    split_name: Mapped[str] = mapped_column(sa.String(64), nullable=False, default="test")
    n_samples: Mapped[int | None] = mapped_column(sa.Integer())
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """ROC-AUC / PR-AUC / MCC for classification, RMSE / MAE / R2 / Spearman for
    regression, each with a bootstrap confidence interval."""

    calibration: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Reliability curve bins, Brier score, expected calibration error."""

    error_analysis: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Worst predictions, error by molecular-weight/logP bucket, error by scaffold
    frequency, and in/out-of-domain breakdown."""

    prediction_distribution: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    per_fold_metrics: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    notes: Mapped[str | None] = mapped_column(sa.Text())

    model: Mapped[MLModel] = relationship(back_populates="evaluations")


class Experiment(Base, TimestampMixin):
    """A tracked training/evaluation experiment (``EXP-2026-00007``)."""

    __tablename__ = "experiment"
    __table_args__ = (sa.Index("ix_experiment_status_created", "status", "created_at"),)

    id: Mapped[str] = mapped_column(sa.String(32), primary_key=True)
    sequence: Mapped[int] = mapped_column(sa.Integer(), nullable=False)
    year: Mapped[int] = mapped_column(sa.Integer(), nullable=False)

    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text())
    endpoint: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="running")
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    artifacts: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    comparison: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Baseline-vs-graph-model comparison table produced by the training pipeline."""

    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    runtime_s: Mapped[float | None] = mapped_column(sa.Float())
    git_sha: Mapped[str | None] = mapped_column(sa.String(40))
    random_seed: Mapped[int | None] = mapped_column(sa.Integer())
    environment: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    error: Mapped[str | None] = mapped_column(sa.Text())

    models: Mapped[list[MLModel]] = relationship(
        foreign_keys="MLModel.experiment_id", viewonly=True
    )


class Prediction(Base, ULIDPrimaryKey, TimestampMixin):
    """A model prediction for one compound.

    Predictions are first-class, persisted rows -- not transient API output -- because a
    research run must be able to show precisely which number it used, from which model
    version, with which uncertainty, months later.
    """

    __tablename__ = "prediction"
    __table_args__ = (
        sa.UniqueConstraint("compound_id", "model_id", "input_hash"),
        sa.Index("ix_prediction_endpoint", "endpoint", "created_at"),
        sa.Index("ix_prediction_run", "research_run_id"),
    )

    compound_id: Mapped[str] = mapped_column(
        sa.ForeignKey("compound.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_id: Mapped[str] = mapped_column(
        sa.ForeignKey("ml_model.id", ondelete="CASCADE"), nullable=False, index=True
    )
    endpoint: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(sa.String(32), nullable=False)

    value: Mapped[float | None] = mapped_column(sa.Float())
    """Regression output in the model's units, or ``NULL`` for classification."""

    probability: Mapped[float | None] = mapped_column(sa.Float())
    """Calibrated positive-class probability for classification."""

    predicted_class: Mapped[str | None] = mapped_column(sa.String(64))
    uncertainty: Mapped[float | None] = mapped_column(sa.Float())
    """Standard deviation across the ensemble / MC-dropout passes."""

    confidence: Mapped[float | None] = mapped_column(sa.Float())
    """[0, 1] confidence combining ensemble spread with applicability domain."""

    applicability: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Nearest-neighbour Tanimoto to the training set and an in/out-of-domain verdict.
    Out-of-domain predictions are flagged in the UI rather than silently shown."""

    input_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    """Hash of the featurised input; makes prediction writes idempotent and lets the
    cache prove a cached value belongs to the exact same input."""

    latency_ms: Mapped[float | None] = mapped_column(sa.Float())
    research_run_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("research_run.id", ondelete="SET NULL")
    )

    compound: Mapped[Any] = relationship("Compound")
    model: Mapped[MLModel] = relationship()


class CandidateRanking(Base, ULIDPrimaryKey, TimestampMixin):
    """One execution of the multi-objective prioritisation, with its exact config."""

    __tablename__ = "candidate_ranking"

    name: Mapped[str | None] = mapped_column(sa.String(255))
    target_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("target.id", ondelete="SET NULL"), index=True
    )
    disease_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("disease.id", ondelete="SET NULL"), index=True
    )
    research_run_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("research_run.id", ondelete="CASCADE"), index=True
    )
    config: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Objective weights, desirability curves and hard filters -- persisted so a ranking
    can be reproduced or re-run with modified weights."""

    model_versions: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    n_candidates: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    n_filtered_out: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    config_hash: Mapped[str | None] = mapped_column(sa.String(64), index=True)

    scores: Mapped[list[CandidateScore]] = relationship(
        back_populates="ranking", cascade="all, delete-orphan"
    )


class CandidateScore(Base, ULIDPrimaryKey, TimestampMixin):
    """A compound's position in a ranking, with the full factor breakdown."""

    __tablename__ = "candidate_score"
    __table_args__ = (
        sa.UniqueConstraint("ranking_id", "compound_id"),
        sa.Index("ix_candidate_score_rank", "ranking_id", "rank"),
    )

    ranking_id: Mapped[str] = mapped_column(
        sa.ForeignKey("candidate_ranking.id", ondelete="CASCADE"), nullable=False, index=True
    )
    compound_id: Mapped[str] = mapped_column(
        sa.ForeignKey("compound.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank: Mapped[int] = mapped_column(sa.Integer(), nullable=False)
    total_score: Mapped[float] = mapped_column(sa.Float(), nullable=False)
    factors: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    """Per-objective ``{name, raw_value, desirability, weight, contribution, basis,
    source}`` rows -- every point of the total score is attributable."""

    flags: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    """Caveats: ``out_of_domain``, ``no_experimental_evidence``, ``low_confidence``."""

    evidence_ids: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    prediction_ids: Mapped[list[str]] = mapped_column(JSONColumn, default=list)

    ranking: Mapped[CandidateRanking] = relationship(back_populates="scores")
    compound: Mapped[Any] = relationship("Compound")
