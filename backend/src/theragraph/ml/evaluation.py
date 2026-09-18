"""Scientific evaluation: metrics, confidence intervals, error analysis, applicability.

Design decisions that matter more than the metric list:

* **Every headline metric carries a bootstrap confidence interval.** A test set of 112
  molecules produces an ROC-AUC with roughly ±0.08 of sampling noise; reporting the
  point estimate alone invites reading a coin flip as an improvement.
* **PR-AUC is reported next to ROC-AUC.** With a 5% positive rate (ClinTox), ROC-AUC
  looks respectable while the model is useless at the operating point that matters.
* **Error analysis is part of evaluation, not an afterthought.** Error by molecular
  weight, by scaffold novelty and by predicted-uncertainty decile is what tells a
  chemist whether the model can be trusted for *their* series.
* **Applicability domain.** A prediction on a molecule with no close training analogue
  is extrapolation; it is flagged, not silently served.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from theragraph.chem.fingerprints import compute_fingerprint, tanimoto_bulk
from theragraph.core.logging import get_logger
from theragraph.ml.calibration import (
    regression_interval_coverage,
    reliability_curve,
)

log = get_logger(__name__)

TaskType = Literal["classification", "regression"]

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260918

#: Below this maximum Tanimoto to the training set, a prediction is extrapolation.
#: 0.35 on Morgan r2/2048 is the commonly used "no meaningful analogue" threshold.
AD_SIMILARITY_THRESHOLD = 0.35


@dataclass
class MetricWithCI:
    value: float
    ci_low: float | None = None
    ci_high: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 5),
            "ci_low": round(self.ci_low, 5) if self.ci_low is not None else None,
            "ci_high": round(self.ci_high, 5) if self.ci_high is not None else None,
        }


@dataclass
class EvaluationResult:
    task_type: TaskType
    split_strategy: str
    split_name: str
    n_samples: int
    metrics: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    error_analysis: dict[str, Any] = field(default_factory=dict)
    prediction_distribution: dict[str, Any] = field(default_factory=dict)
    applicability: dict[str, Any] = field(default_factory=dict)

    def headline(self) -> str:
        if self.task_type == "classification":
            roc = (self.metrics.get("roc_auc") or {}).get("value")
            pr = (self.metrics.get("pr_auc") or {}).get("value")
            return f"ROC-AUC {roc} / PR-AUC {pr}"
        rmse = (self.metrics.get("rmse") or {}).get("value")
        r2 = (self.metrics.get("r2") or {}).get("value")
        return f"RMSE {rmse} / R2 {r2}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "split_strategy": self.split_strategy,
            "split_name": self.split_name,
            "n_samples": self.n_samples,
            "metrics": self.metrics,
            "calibration": self.calibration,
            "error_analysis": self.error_analysis,
            "prediction_distribution": self.prediction_distribution,
            "applicability": self.applicability,
            "headline": self.headline(),
        }


# ------------------------------------------------------------------- bootstrap
def _bootstrap_ci(
    metric_fn: Any,
    *arrays: np.ndarray,
    n_samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI. Resamples that break the metric are skipped."""
    length = len(arrays[0])
    if length < 10:
        return None, None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_samples):
        indices = rng.integers(0, length, length)
        try:
            value = metric_fn(*(array[indices] for array in arrays))
        except Exception:
            continue
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if len(values) < n_samples // 10:
        return None, None
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def _metric(metric_fn: Any, *arrays: np.ndarray, with_ci: bool = True) -> dict[str, Any] | None:
    try:
        value = metric_fn(*arrays)
    except Exception as exc:
        log.warning(
            "evaluation.metric_failed", metric=getattr(metric_fn, "__name__", "?"), error=str(exc)
        )
        return None
    if value is None or not np.isfinite(value):
        return None
    low, high = _bootstrap_ci(metric_fn, *arrays) if with_ci else (None, None)
    return MetricWithCI(value=float(value), ci_low=low, ci_high=high).to_dict()


# -------------------------------------------------------------------- metrics
def classification_metrics(
    probabilities: np.ndarray, labels: np.ndarray, *, threshold: float = 0.5
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    binary = (np.asarray(labels, dtype=np.float64).reshape(-1) >= 0.5).astype(np.int32)
    predicted = (probabilities >= threshold).astype(np.int32)

    metrics: dict[str, Any] = {
        "roc_auc": _metric(roc_auc_score, binary, probabilities),
        "pr_auc": _metric(average_precision_score, binary, probabilities),
        "accuracy": _metric(accuracy_score, binary, predicted),
        "balanced_accuracy": _metric(balanced_accuracy_score, binary, predicted),
        "mcc": _metric(matthews_corrcoef, binary, predicted),
        "f1": _metric(lambda y, p: f1_score(y, p, zero_division=0), binary, predicted),
        "precision": _metric(
            lambda y, p: precision_score(y, p, zero_division=0), binary, predicted
        ),
        "recall": _metric(lambda y, p: recall_score(y, p, zero_division=0), binary, predicted),
    }
    true_positive = int(((predicted == 1) & (binary == 1)).sum())
    false_positive = int(((predicted == 1) & (binary == 0)).sum())
    true_negative = int(((predicted == 0) & (binary == 0)).sum())
    false_negative = int(((predicted == 0) & (binary == 1)).sum())
    metrics["confusion_matrix"] = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }
    metrics["threshold"] = threshold
    metrics["positive_rate"] = round(float(binary.mean()), 4)
    metrics["baseline_pr_auc"] = round(float(binary.mean()), 4)
    """Majority-class PR-AUC baseline; PR-AUC must beat this to mean anything."""
    return metrics


def regression_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict[str, Any]:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    actuals = np.asarray(actuals, dtype=np.float64).reshape(-1)

    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics: dict[str, Any] = {
        "rmse": _metric(rmse, actuals, predictions),
        "mae": _metric(mean_absolute_error, actuals, predictions),
        "r2": _metric(r2_score, actuals, predictions),
        "spearman": _metric(lambda a, b: float(spearmanr(a, b).statistic), actuals, predictions),
        "pearson": _metric(lambda a, b: float(pearsonr(a, b).statistic), actuals, predictions),
    }
    residuals = predictions - actuals
    metrics["max_absolute_error"] = (
        round(float(np.abs(residuals).max()), 5) if len(residuals) else None
    )
    metrics["median_absolute_error"] = (
        round(float(np.median(np.abs(residuals))), 5) if len(residuals) else None
    )
    metrics["bias"] = round(float(residuals.mean()), 5) if len(residuals) else None
    """Mean signed residual: a non-zero bias means the model is systematically off,
    which a symmetric error metric like RMSE hides."""
    return metrics


# ------------------------------------------------------------- error analysis
def error_analysis(
    *,
    task_type: TaskType,
    predictions: np.ndarray,
    actuals: np.ndarray,
    smiles: list[str],
    identifiers: list[str],
    scaffolds: list[str | None],
    uncertainties: np.ndarray | None = None,
    training_scaffold_counts: dict[str, int] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Where and why the model is wrong."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    actuals = np.asarray(actuals, dtype=np.float64).reshape(-1)
    if len(predictions) == 0:
        return {}

    if task_type == "classification":
        binary = (actuals >= 0.5).astype(np.int32)
        errors = np.abs(predictions - binary)
    else:
        errors = np.abs(predictions - actuals)

    order = np.argsort(-errors)[:top_k]
    worst = [
        {
            "identifier": identifiers[index] if index < len(identifiers) else str(index),
            "smiles": smiles[index] if index < len(smiles) else None,
            "predicted": round(float(predictions[index]), 4),
            "actual": round(float(actuals[index]), 4),
            "absolute_error": round(float(errors[index]), 4),
            "uncertainty": round(float(uncertainties[index]), 4)
            if uncertainties is not None and index < len(uncertainties)
            else None,
            "scaffold": scaffolds[index] if index < len(scaffolds) else None,
        }
        for index in order
    ]

    analysis: dict[str, Any] = {
        "worst_predictions": worst,
        "error_by_molecular_weight": _error_by_property(smiles, errors, "molecular_weight"),
        "error_by_logp": _error_by_property(smiles, errors, "logp"),
        "error_summary": {
            "mean_absolute_error": round(float(errors.mean()), 5),
            "p90_absolute_error": round(float(np.percentile(errors, 90)), 5),
            "worst_absolute_error": round(float(errors.max()), 5),
        },
    }

    if uncertainties is not None and len(uncertainties) == len(errors):
        analysis["error_by_uncertainty_decile"] = _error_by_uncertainty(
            np.asarray(uncertainties, dtype=np.float64), errors
        )

    if training_scaffold_counts is not None:
        analysis["error_by_scaffold_familiarity"] = _error_by_scaffold_familiarity(
            scaffolds, errors, training_scaffold_counts
        )

    return analysis


def _error_by_property(
    smiles: list[str], errors: np.ndarray, property_key: str, *, n_buckets: int = 4
) -> list[dict[str, Any]]:
    """Bucket error by a physicochemical property to expose systematic weakness."""
    from theragraph.chem.descriptors import compute_descriptors

    values: list[float] = []
    usable: list[int] = []
    for index, item in enumerate(smiles[: len(errors)]):
        try:
            result = compute_descriptors(item)
            value = result.get(property_key)
        except Exception:
            value = None
        if value is not None:
            values.append(float(value))
            usable.append(index)

    if len(values) < n_buckets * 4:
        return []

    array = np.asarray(values)
    edges = np.quantile(array, np.linspace(0, 1, n_buckets + 1))
    buckets: list[dict[str, Any]] = []
    for bucket in range(n_buckets):
        low, high = edges[bucket], edges[bucket + 1]
        mask = (
            (array >= low) & (array <= high)
            if bucket == n_buckets - 1
            else (array >= low) & (array < high)
        )
        if not mask.any():
            continue
        selected = [usable[i] for i in np.flatnonzero(mask)]
        buckets.append(
            {
                "property": property_key,
                "range": [round(float(low), 2), round(float(high), 2)],
                "count": len(selected),
                "mean_absolute_error": round(float(errors[selected].mean()), 5),
            }
        )
    return buckets


def _error_by_uncertainty(uncertainties: np.ndarray, errors: np.ndarray) -> list[dict[str, Any]]:
    """Does higher reported uncertainty actually mean higher error?

    If the top decile's error is not clearly above the bottom decile's, the uncertainty
    estimate is decoration and should not be used for triage.
    """
    if uncertainties.std() < 1e-9:
        return []
    deciles = np.quantile(uncertainties, np.linspace(0, 1, 11))
    rows: list[dict[str, Any]] = []
    for index in range(10):
        low, high = deciles[index], deciles[index + 1]
        mask = (
            (uncertainties >= low) & (uncertainties <= high)
            if index == 9
            else (uncertainties >= low) & (uncertainties < high)
        )
        if not mask.any():
            continue
        rows.append(
            {
                "decile": index + 1,
                "uncertainty_range": [round(float(low), 5), round(float(high), 5)],
                "count": int(mask.sum()),
                "mean_absolute_error": round(float(errors[mask].mean()), 5),
            }
        )
    return rows


def _error_by_scaffold_familiarity(
    scaffolds: list[str | None], errors: np.ndarray, training_counts: dict[str, int]
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {"unseen": [], "rare_1_2": [], "common_3_plus": []}
    for index, scaffold in enumerate(scaffolds[: len(errors)]):
        count = training_counts.get(scaffold or "", 0)
        if count == 0:
            groups["unseen"].append(index)
        elif count <= 2:
            groups["rare_1_2"].append(index)
        else:
            groups["common_3_plus"].append(index)

    return [
        {
            "familiarity": name,
            "count": len(indices),
            "mean_absolute_error": round(float(errors[indices].mean()), 5),
        }
        for name, indices in groups.items()
        if indices
    ]


# ------------------------------------------------------- applicability domain
def applicability_domain(
    query_smiles: list[str],
    training_smiles: list[str],
    *,
    fingerprint_kind: str = "morgan_r2_2048",
    threshold: float = AD_SIMILARITY_THRESHOLD,
    max_training: int = 5000,
) -> dict[str, Any]:
    """Nearest-neighbour Tanimoto from each query to the training set."""
    if not query_smiles or not training_smiles:
        return {"assessed": 0, "threshold": threshold}

    training_bits = [
        compute_fingerprint(smiles, fingerprint_kind).bits
        for smiles in training_smiles[:max_training]
    ]
    similarities: list[float] = []
    for smiles in query_smiles:
        try:
            query_bits = compute_fingerprint(smiles, fingerprint_kind).bits
        except Exception:
            similarities.append(0.0)
            continue
        scores = tanimoto_bulk(query_bits, training_bits)
        similarities.append(float(scores.max()) if scores.size else 0.0)

    array = np.asarray(similarities)
    in_domain = array >= threshold
    return {
        "assessed": len(array),
        "threshold": threshold,
        "in_domain": int(in_domain.sum()),
        "out_of_domain": int((~in_domain).sum()),
        "out_of_domain_fraction": round(float((~in_domain).mean()), 4),
        "mean_nearest_similarity": round(float(array.mean()), 4),
        "median_nearest_similarity": round(float(np.median(array)), 4),
        "per_query_nearest_similarity": [round(value, 4) for value in similarities],
    }


def nearest_training_similarity(
    query_smiles: str,
    training_bits: list[bytes],
    *,
    fingerprint_kind: str = "morgan_r2_2048",
) -> float:
    """Single-molecule applicability score, used at serving time."""
    if not training_bits:
        return 0.0
    try:
        query_bits = compute_fingerprint(query_smiles, fingerprint_kind).bits
    except Exception:
        return 0.0
    scores = tanimoto_bulk(query_bits, training_bits)
    return float(scores.max()) if scores.size else 0.0


# ------------------------------------------------------ distribution + wrapper
def prediction_distribution(predictions: np.ndarray, *, n_bins: int = 20) -> dict[str, Any]:
    array = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0}
    counts, edges = np.histogram(array, bins=n_bins)
    return {
        "count": int(array.size),
        "min": round(float(array.min()), 5),
        "max": round(float(array.max()), 5),
        "mean": round(float(array.mean()), 5),
        "std": round(float(array.std()), 5),
        "quantiles": {
            "p05": round(float(np.percentile(array, 5)), 5),
            "p25": round(float(np.percentile(array, 25)), 5),
            "p50": round(float(np.percentile(array, 50)), 5),
            "p75": round(float(np.percentile(array, 75)), 5),
            "p95": round(float(np.percentile(array, 95)), 5),
        },
        "histogram": {
            "bin_edges": [round(float(edge), 5) for edge in edges],
            "counts": [int(count) for count in counts],
        },
    }


def evaluate(
    *,
    task_type: TaskType,
    split_strategy: str,
    split_name: str,
    predictions: np.ndarray,
    actuals: np.ndarray,
    smiles: list[str],
    identifiers: list[str],
    scaffolds: list[str | None],
    uncertainties: np.ndarray | None = None,
    training_smiles: list[str] | None = None,
    training_scaffold_counts: dict[str, int] | None = None,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Full evaluation of one model on one split."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    actuals = np.asarray(actuals, dtype=np.float64).reshape(-1)

    if task_type == "classification":
        metrics = classification_metrics(predictions, actuals, threshold=threshold)
        calibration = reliability_curve(predictions, actuals)
    else:
        metrics = regression_metrics(predictions, actuals)
        calibration = (
            regression_interval_coverage(predictions, uncertainties, actuals)
            if uncertainties is not None
            else {}
        )

    result = EvaluationResult(
        task_type=task_type,
        split_strategy=split_strategy,
        split_name=split_name,
        n_samples=len(predictions),
        metrics=metrics,
        calibration=calibration,
        error_analysis=error_analysis(
            task_type=task_type,
            predictions=predictions,
            actuals=actuals,
            smiles=smiles,
            identifiers=identifiers,
            scaffolds=scaffolds,
            uncertainties=uncertainties,
            training_scaffold_counts=training_scaffold_counts,
        ),
        prediction_distribution=prediction_distribution(predictions),
    )

    if training_smiles:
        result.applicability = applicability_domain(smiles, training_smiles)

    return result


def compare_models(results: dict[str, EvaluationResult]) -> dict[str, Any]:
    """Side-by-side comparison table with an explicit, defensible winner rule.

    The winner is chosen on the primary metric's *point estimate*, but the comparison
    also reports whether the confidence intervals overlap. Overlapping intervals mean
    "no measurable difference", which is the honest conclusion far more often than
    leaderboards suggest.
    """
    if not results:
        return {}

    task_types = {result.task_type for result in results.values()}
    task_type = task_types.pop() if len(task_types) == 1 else "mixed"
    primary = "roc_auc" if task_type == "classification" else "rmse"
    higher_is_better = task_type == "classification"

    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        metric = result.metrics.get(primary) or {}
        rows.append(
            {
                "model": name,
                "split_strategy": result.split_strategy,
                "n_samples": result.n_samples,
                "primary_metric": primary,
                "value": metric.get("value"),
                "ci_low": metric.get("ci_low"),
                "ci_high": metric.get("ci_high"),
                "headline": result.headline(),
            }
        )

    scored = [row for row in rows if row["value"] is not None]
    if not scored:
        return {"primary_metric": primary, "rows": rows, "winner": None}

    best = (max if higher_is_better else min)(scored, key=lambda row: row["value"])
    others = [row for row in scored if row["model"] != best["model"]]
    overlapping = [
        row["model"]
        for row in others
        if row.get("ci_high") is not None
        and best.get("ci_low") is not None
        and row["ci_high"] >= best["ci_low"]
        and row.get("ci_low") is not None
        and best.get("ci_high") is not None
        and best["ci_high"] >= row["ci_low"]
    ]

    return {
        "primary_metric": primary,
        "higher_is_better": higher_is_better,
        "rows": sorted(
            rows,
            key=lambda row: (
                row["value"] is None,
                -(row["value"] or 0) if higher_is_better else (row["value"] or 0),
            ),
        ),
        "winner": best["model"],
        "statistically_indistinguishable_from_winner": overlapping,
        "verdict": (
            f"{best['model']} leads on {primary}"
            + (
                f", but its confidence interval overlaps {', '.join(overlapping)} -- "
                "treat the difference as unproven"
                if overlapping
                else " with non-overlapping confidence intervals"
            )
        ),
    }
