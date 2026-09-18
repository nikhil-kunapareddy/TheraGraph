"""Probability calibration and reliability measurement.

A classifier that reports 0.9 must be right about 90% of the time, or the number is
worse than useless in a triage workflow -- a scientist will deprioritise a compound on
the strength of it. Raw tree-ensemble and neural-network outputs are systematically
mis-calibrated (especially on the imbalanced sets used here), so a calibrator is fitted
on the *validation* fold and its reliability is measured and published on the model.

Two calibrators:

* **Platt scaling** -- a single-parameter logistic fit. Robust with few hundred
  validation points, which is the common case here.
* **Isotonic regression** -- non-parametric, more flexible, needs more data; chosen
  automatically when the validation fold is large enough and it actually improves the
  Brier score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from theragraph.core.logging import get_logger

log = get_logger(__name__)

Method = Literal["platt", "isotonic", "none"]

MIN_ISOTONIC_SAMPLES = 400
EPSILON = 1e-7


@dataclass
class Calibrator:
    """Fitted probability calibrator."""

    method: Method
    platt_a: float = 1.0
    platt_b: float = 0.0
    isotonic_x: list[float] | None = None
    isotonic_y: list[float] | None = None
    n_calibration_samples: int = 0
    improvement: dict[str, float] | None = None

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1 - EPSILON)
        if self.method == "platt":
            logits = np.log(clipped / (1 - clipped))
            return 1.0 / (1.0 + np.exp(-(self.platt_a * logits + self.platt_b)))
        if self.method == "isotonic" and self.isotonic_x and self.isotonic_y:
            return np.interp(clipped, self.isotonic_x, self.isotonic_y)
        return clipped

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "isotonic_x": self.isotonic_x,
            "isotonic_y": self.isotonic_y,
            "n_calibration_samples": self.n_calibration_samples,
            "improvement": self.improvement,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Calibrator:
        return cls(
            method=payload.get("method", "none"),
            platt_a=float(payload.get("platt_a", 1.0)),
            platt_b=float(payload.get("platt_b", 0.0)),
            isotonic_x=payload.get("isotonic_x"),
            isotonic_y=payload.get("isotonic_y"),
            n_calibration_samples=int(payload.get("n_calibration_samples", 0)),
            improvement=payload.get("improvement"),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_calibrator(
    probabilities: np.ndarray, labels: np.ndarray, *, method: Method | None = None
) -> Calibrator:
    """Fit a calibrator, auto-selecting the method when not specified.

    Selection rule: try Platt always; try isotonic only with enough samples; keep
    whichever lowers the Brier score on the calibration set, and fall back to no
    calibration if neither helps. Refusing to calibrate is a valid outcome -- applying a
    transform that makes things worse is the failure mode to avoid.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    binary_labels = (np.asarray(labels, dtype=np.float64).reshape(-1) >= 0.5).astype(np.float64)

    if len(probabilities) < 20 or len(np.unique(binary_labels)) < 2:
        log.info("calibration.skipped", reason="insufficient_data", n=len(probabilities))
        return Calibrator(method="none", n_calibration_samples=len(probabilities))

    baseline_brier = float(np.mean((probabilities - binary_labels) ** 2))
    candidates: list[tuple[float, Calibrator]] = []

    if method in (None, "platt"):
        platt = _fit_platt(probabilities, binary_labels)
        score = float(np.mean((platt.apply(probabilities) - binary_labels) ** 2))
        candidates.append((score, platt))

    if method == "isotonic" or (method is None and len(probabilities) >= MIN_ISOTONIC_SAMPLES):
        isotonic = _fit_isotonic(probabilities, binary_labels)
        score = float(np.mean((isotonic.apply(probabilities) - binary_labels) ** 2))
        candidates.append((score, isotonic))

    if not candidates:
        return Calibrator(method="none", n_calibration_samples=len(probabilities))

    best_score, best = min(candidates, key=lambda item: item[0])
    if best_score >= baseline_brier:
        log.info(
            "calibration.rejected",
            reason="no_improvement",
            baseline_brier=round(baseline_brier, 5),
            best_brier=round(best_score, 5),
        )
        return Calibrator(
            method="none",
            n_calibration_samples=len(probabilities),
            improvement={"baseline_brier": baseline_brier, "calibrated_brier": baseline_brier},
        )

    best.n_calibration_samples = len(probabilities)
    best.improvement = {
        "baseline_brier": round(baseline_brier, 6),
        "calibrated_brier": round(best_score, 6),
        "brier_reduction": round(baseline_brier - best_score, 6),
    }
    log.info("calibration.fitted", method=best.method, **best.improvement)
    return best


def _fit_platt(probabilities: np.ndarray, labels: np.ndarray) -> Calibrator:
    """Logistic regression on the logit of the raw probability."""
    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, labels)
    return Calibrator(
        method="platt",
        platt_a=float(model.coef_[0][0]),
        platt_b=float(model.intercept_[0]),
    )


def _fit_isotonic(probabilities: np.ndarray, labels: np.ndarray) -> Calibrator:
    """Isotonic regression, stored as an interpolation table for dependency-free serving."""
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(probabilities, labels)
    grid = np.linspace(0.0, 1.0, 101)
    return Calibrator(
        method="isotonic",
        isotonic_x=[float(value) for value in grid],
        isotonic_y=[float(value) for value in model.predict(grid)],
    )


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> dict[str, Any]:
    """Reliability diagram data plus ECE / MCE / Brier.

    Equal-width bins (not quantile bins) so the diagram is directly comparable across
    models, with per-bin counts included: a bin holding three molecules must not be read
    as strongly as one holding three hundred, and the UI renders bin width accordingly.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    binary_labels = (np.asarray(labels, dtype=np.float64).reshape(-1) >= 0.5).astype(np.float64)
    if len(probabilities) == 0:
        return {"bins": [], "ece": None, "mce": None, "brier": None, "n": 0}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    expected_error = 0.0
    max_error = 0.0

    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        mask = (
            (probabilities >= low) & (probabilities < high)
            if index < n_bins - 1
            else (probabilities >= low) & (probabilities <= high)
        )
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_lower": round(float(low), 3),
                    "bin_upper": round(float(high), 3),
                    "count": 0,
                    "mean_predicted": None,
                    "observed_frequency": None,
                    "gap": None,
                }
            )
            continue
        mean_predicted = float(probabilities[mask].mean())
        observed = float(binary_labels[mask].mean())
        gap = abs(mean_predicted - observed)
        expected_error += (count / len(probabilities)) * gap
        max_error = max(max_error, gap)
        bins.append(
            {
                "bin_lower": round(float(low), 3),
                "bin_upper": round(float(high), 3),
                "count": count,
                "mean_predicted": round(mean_predicted, 4),
                "observed_frequency": round(observed, 4),
                "gap": round(gap, 4),
            }
        )

    return {
        "bins": bins,
        "ece": round(expected_error, 5),
        "mce": round(max_error, 5),
        "brier": round(float(np.mean((probabilities - binary_labels) ** 2)), 5),
        "n": len(probabilities),
        "n_bins": n_bins,
    }


def regression_interval_coverage(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    actuals: np.ndarray,
    *,
    z: float = 1.96,
) -> dict[str, Any]:
    """Do the uncertainty estimates behave like error bars?

    Measures the fraction of true values falling inside ``prediction ± z*sigma``. A
    well-behaved estimate gives ~95% at z=1.96; MC-dropout and ensemble spread typically
    *under*-cover, and reporting the observed number is more useful than pretending
    otherwise.
    """
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    uncertainties = np.asarray(uncertainties, dtype=np.float64).reshape(-1)
    actuals = np.asarray(actuals, dtype=np.float64).reshape(-1)
    if len(predictions) == 0 or uncertainties.size != predictions.size:
        return {"coverage": None, "nominal": 0.95, "n": 0}

    lower = predictions - z * uncertainties
    upper = predictions + z * uncertainties
    inside = (actuals >= lower) & (actuals <= upper)
    residuals = np.abs(actuals - predictions)
    safe_uncertainty = np.where(uncertainties <= 1e-9, np.nan, uncertainties)

    with np.errstate(invalid="ignore"):
        normalized = residuals / safe_uncertainty
        correlation = (
            float(np.corrcoef(uncertainties, residuals)[0, 1])
            if uncertainties.std() > 1e-9 and residuals.std() > 1e-9
            else None
        )

    coverage = float(inside.mean())
    nominal = float(2 * _standard_normal_cdf(z) - 1)

    return {
        "coverage": round(coverage, 4),
        "nominal": round(nominal, 4),
        "mean_uncertainty": round(float(uncertainties.mean()), 5),
        "mean_absolute_error": round(float(residuals.mean()), 5),
        "uncertainty_error_correlation": round(correlation, 4) if correlation is not None else None,
        "median_standardized_residual": round(float(np.nanmedian(normalized)), 4)
        if np.isfinite(np.nanmedian(normalized))
        else None,
        "n": len(predictions),
        "interpretation": _coverage_verdict(coverage, nominal),
    }


def _coverage_verdict(coverage: float, nominal: float) -> str:
    """Say what *this* coverage means, not what low coverage would mean in general.

    A fixed legend ("coverage below nominal means over-confident") reads as a verdict on
    the model in front of you, and is simply wrong when coverage came out above nominal.
    """
    if coverage < nominal - 0.05:
        return (
            f"coverage {coverage:.1%} is below the nominal {nominal:.0%}: the uncertainty "
            "estimate is over-confident, so treat the error bars as a lower bound"
        )
    if coverage > nominal + 0.03:
        return (
            f"coverage {coverage:.1%} exceeds the nominal {nominal:.0%}: the intervals are "
            "conservative -- wider than they need to be, which is the safer direction"
        )
    return (
        f"coverage {coverage:.1%} matches the nominal {nominal:.0%}: the uncertainty "
        "estimate behaves like a calibrated error bar"
    )


def _standard_normal_cdf(z: float) -> float:
    import math

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
