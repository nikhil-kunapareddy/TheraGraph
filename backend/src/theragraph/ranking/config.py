"""Ranking configuration: objectives, desirability curves and hard filters.

The ranking is a weighted sum of **desirability functions**, the standard
multi-parameter optimisation approach in medicinal chemistry (Segall, 2012). Each raw
value -- a predicted logS, a measured pIC50, a TPSA -- is mapped onto [0, 1] by an
explicit, inspectable curve, then weighted.

Three properties this design deliberately has:

* **Every configuration is data, not code.** A ranking config is JSON, stored on the
  research run, so a ranking can be reproduced or re-run with different weights.
* **Nothing is hidden.** Each candidate's score decomposes into per-objective
  contributions with the raw value, the curve output, the weight and the *basis*
  (retrieved measurement vs model prediction). The UI shows this decomposition.
* **Missing data is not zero.** An objective with no data for a compound is excluded
  and the remaining weights renormalise, with the omission recorded. Scoring a missing
  measurement as 0 would silently punish under-characterised compounds -- exactly the
  novel chemistry a discovery programme cares about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from theragraph.core.errors import ValidationError
from theragraph.core.hashing import hash_payload

CurveKind = Literal["higher_better", "lower_better", "window", "step", "identity"]
Basis = Literal["measured", "predicted", "computed", "derived"]


@dataclass(frozen=True)
class Desirability:
    """Maps a raw value onto [0, 1].

    ``higher_better`` / ``lower_better`` ramp linearly between ``bad`` and ``good``.
    ``window`` is a plateau between ``low``/``high`` with linear shoulders of width
    ``tolerance`` -- the right shape for properties like LogP or molecular weight where
    both extremes are undesirable.
    """

    kind: CurveKind
    good: float | None = None
    bad: float | None = None
    low: float | None = None
    high: float | None = None
    tolerance: float = 1.0
    threshold: float | None = None

    def apply(self, value: float) -> float:
        if self.kind == "identity":
            return max(0.0, min(1.0, float(value)))

        if self.kind == "step":
            if self.threshold is None:
                raise ValidationError("step desirability requires a threshold")
            return 1.0 if value >= self.threshold else 0.0

        if self.kind == "window":
            if self.low is None or self.high is None:
                raise ValidationError("window desirability requires low and high")
            if self.low <= value <= self.high:
                return 1.0
            distance = self.low - value if value < self.low else value - self.high
            return max(0.0, 1.0 - distance / max(self.tolerance, 1e-9))

        if self.good is None or self.bad is None:
            raise ValidationError(f"{self.kind} desirability requires good and bad values")

        if self.kind == "higher_better":
            if value >= self.good:
                return 1.0
            if value <= self.bad:
                return 0.0
            return (value - self.bad) / (self.good - self.bad)

        # lower_better
        if value <= self.good:
            return 1.0
        if value >= self.bad:
            return 0.0
        return (self.bad - value) / (self.bad - self.good)

    def describe(self) -> str:
        if self.kind == "window":
            return f"ideal between {self.low} and {self.high} (±{self.tolerance} shoulder)"
        if self.kind == "step":
            return f"acceptable at or above {self.threshold}"
        if self.kind == "identity":
            return "already on a 0-1 scale"
        direction = "higher" if self.kind == "higher_better" else "lower"
        return f"{direction} is better: {self.bad} scores 0, {self.good} scores 1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "good": self.good,
            "bad": self.bad,
            "low": self.low,
            "high": self.high,
            "tolerance": self.tolerance,
            "threshold": self.threshold,
            "description": self.describe(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Desirability:
        return cls(
            kind=payload["kind"],
            good=payload.get("good"),
            bad=payload.get("bad"),
            low=payload.get("low"),
            high=payload.get("high"),
            tolerance=float(payload.get("tolerance", 1.0)),
            threshold=payload.get("threshold"),
        )


@dataclass(frozen=True)
class Objective:
    """One scored dimension."""

    key: str
    label: str
    weight: float
    source_field: str
    basis: Basis
    desirability: Desirability
    explanation: str
    required: bool = False
    """When True, a candidate lacking this value is filtered out rather than scored
    with renormalised weights."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "weight": self.weight,
            "source_field": self.source_field,
            "basis": self.basis,
            "desirability": self.desirability.to_dict(),
            "explanation": self.explanation,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Objective:
        return cls(
            key=payload["key"],
            label=payload["label"],
            weight=float(payload["weight"]),
            source_field=payload["source_field"],
            basis=payload["basis"],
            desirability=Desirability.from_dict(payload["desirability"]),
            explanation=payload.get("explanation", ""),
            required=bool(payload.get("required", False)),
        )


@dataclass
class HardFilter:
    """A pass/fail gate applied before scoring."""

    key: str
    label: str
    source_field: str
    operator: Literal["<=", ">=", "==", "!=", "between", "is_true", "is_false"]
    value: Any = None
    upper: Any = None
    reason: str = ""
    enabled: bool = True

    def passes(self, candidate_value: Any) -> bool | None:
        """``None`` means "cannot evaluate" -- the filter abstains rather than rejects."""
        if candidate_value is None:
            return None
        try:
            if self.operator == "<=":
                return float(candidate_value) <= float(self.value)
            if self.operator == ">=":
                return float(candidate_value) >= float(self.value)
            if self.operator == "==":
                return candidate_value == self.value
            if self.operator == "!=":
                return candidate_value != self.value
            if self.operator == "between":
                return float(self.value) <= float(candidate_value) <= float(self.upper)
            if self.operator == "is_true":
                return bool(candidate_value)
            if self.operator == "is_false":
                return not bool(candidate_value)
        except (TypeError, ValueError):
            return None
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "source_field": self.source_field,
            "operator": self.operator,
            "value": self.value,
            "upper": self.upper,
            "reason": self.reason,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HardFilter:
        return cls(
            key=payload["key"],
            label=payload["label"],
            source_field=payload["source_field"],
            operator=payload["operator"],
            value=payload.get("value"),
            upper=payload.get("upper"),
            reason=payload.get("reason", ""),
            enabled=bool(payload.get("enabled", True)),
        )


@dataclass
class RankingConfig:
    """A complete, serialisable prioritisation configuration."""

    name: str
    description: str
    objectives: list[Objective]
    filters: list[HardFilter] = field(default_factory=list)
    min_objectives_present: int = 3
    """A candidate scored on fewer than this many objectives is reported as
    under-characterised rather than ranked against fully characterised ones."""

    confidence_penalty: float = 0.25
    """Fraction by which a low-confidence prediction's contribution is discounted.
    Applied per-objective, so a candidate ranked on shaky predictions cannot outrank one
    ranked on measurements at equal raw desirability."""

    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValidationError("ranking config needs at least one objective")
        total = sum(objective.weight for objective in self.objectives)
        if total <= 0:
            raise ValidationError("objective weights must sum to a positive number")

    @property
    def total_weight(self) -> float:
        return sum(objective.weight for objective in self.objectives)

    def normalized_weights(self) -> dict[str, float]:
        total = self.total_weight
        return {objective.key: objective.weight / total for objective in self.objectives}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "objectives": [objective.to_dict() for objective in self.objectives],
            "filters": [item.to_dict() for item in self.filters],
            "min_objectives_present": self.min_objectives_present,
            "confidence_penalty": self.confidence_penalty,
            "notes": self.notes,
            "normalized_weights": {
                key: round(value, 4) for key, value in self.normalized_weights().items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RankingConfig:
        return cls(
            name=payload.get("name", "custom"),
            description=payload.get("description", ""),
            objectives=[Objective.from_dict(item) for item in payload["objectives"]],
            filters=[HardFilter.from_dict(item) for item in payload.get("filters") or []],
            min_objectives_present=int(payload.get("min_objectives_present", 3)),
            confidence_penalty=float(payload.get("confidence_penalty", 0.25)),
            notes=payload.get("notes") or [],
        )

    @property
    def config_hash(self) -> str:
        """Identity of this configuration, for reproducing or comparing rankings."""
        return hash_payload(self.to_dict(), length=16)

    def with_weights(self, weights: dict[str, float]) -> RankingConfig:
        """Return a copy with updated weights (the 'what if' control in the UI)."""
        objectives = [
            Objective(
                key=objective.key,
                label=objective.label,
                weight=float(weights.get(objective.key, objective.weight)),
                source_field=objective.source_field,
                basis=objective.basis,
                desirability=objective.desirability,
                explanation=objective.explanation,
                required=objective.required,
            )
            for objective in self.objectives
        ]
        return RankingConfig(
            name=f"{self.name}+custom_weights",
            description=self.description,
            objectives=[objective for objective in objectives if objective.weight > 0],
            filters=self.filters,
            min_objectives_present=self.min_objectives_present,
            confidence_penalty=self.confidence_penalty,
            notes=[*self.notes, "weights adjusted by user"],
        )


# --------------------------------------------------------------------- presets
def _general_objectives() -> list[Objective]:
    return [
        Objective(
            key="measured_potency",
            label="Measured potency (pChEMBL)",
            weight=3.0,
            source_field="best_pchembl",
            basis="measured",
            desirability=Desirability(kind="higher_better", good=8.0, bad=5.0),
            explanation=(
                "Highest experimentally measured pChEMBL value against the target, from "
                "ChEMBL. pChEMBL 8 corresponds to 10 nM potency; 5 corresponds to 10 uM."
            ),
        ),
        Objective(
            key="evidence_support",
            label="Target evidence support",
            weight=1.5,
            source_field="evidence_score",
            basis="measured",
            desirability=Desirability(kind="identity"),
            explanation=(
                "Open Targets association score for the target-disease pair, retrieved "
                "verbatim. Rewards compounds acting on better-evidenced targets."
            ),
        ),
        Objective(
            key="predicted_solubility",
            label="Predicted aqueous solubility",
            weight=1.5,
            source_field="prediction.solubility",
            basis="predicted",
            desirability=Desirability(kind="higher_better", good=-2.0, bad=-6.0),
            explanation=(
                "Model-predicted log S (mol/L). Above -2 is freely soluble; below -6 is "
                "a formulation problem. Prediction, not measurement."
            ),
        ),
        Objective(
            key="predicted_toxicity",
            label="Predicted clinical toxicity risk",
            weight=2.0,
            source_field="prediction.clinical_toxicity",
            basis="predicted",
            desirability=Desirability(kind="lower_better", good=0.1, bad=0.6),
            explanation=(
                "Model-predicted probability of clinical-trial failure for toxicity "
                "(ClinTox). Prediction on a small, biased public dataset -- a risk "
                "signal for triage, not a safety assessment."
            ),
        ),
        Objective(
            key="drug_likeness",
            label="Drug-likeness (QED)",
            weight=1.0,
            source_field="qed",
            basis="computed",
            desirability=Desirability(kind="higher_better", good=0.7, bad=0.2),
            explanation="RDKit QED: a 0-1 composite of drug-like physicochemistry.",
        ),
        Objective(
            key="molecular_weight",
            label="Molecular weight window",
            weight=0.75,
            source_field="molecular_weight",
            basis="computed",
            desirability=Desirability(kind="window", low=250.0, high=500.0, tolerance=150.0),
            explanation=(
                "Oral small-molecule sweet spot. Below 250 Da tends to lack potency; "
                "above 500 Da tends to lose permeability."
            ),
        ),
        Objective(
            key="lipophilicity",
            label="Lipophilicity window (LogP)",
            weight=0.75,
            source_field="logp",
            basis="computed",
            desirability=Desirability(kind="window", low=1.0, high=4.0, tolerance=2.0),
            explanation=(
                "Crippen LogP. Too low impairs membrane permeability; too high drives "
                "promiscuity, metabolic instability and poor solubility."
            ),
        ),
        Objective(
            key="structural_novelty",
            label="Structural novelty",
            weight=1.0,
            source_field="novelty_score",
            basis="derived",
            desirability=Desirability(kind="identity"),
            explanation=(
                "1 minus the maximum Tanimoto similarity to approved drugs and to the "
                "most common scaffolds in the local corpus. Rewards chemistry that is "
                "not already crowded, without penalising a compound for being known."
            ),
        ),
        Objective(
            key="model_confidence",
            label="Model confidence",
            weight=1.0,
            source_field="mean_confidence",
            basis="derived",
            desirability=Desirability(kind="identity"),
            explanation=(
                "Mean confidence across the predictions used, combining ensemble spread "
                "with applicability-domain distance. Down-weights candidates whose "
                "ranking rests on extrapolation."
            ),
        ),
    ]


def _cns_objectives() -> list[Objective]:
    objectives = _general_objectives()
    objectives.append(
        Objective(
            key="predicted_bbb",
            label="Predicted BBB permeability",
            weight=2.5,
            source_field="prediction.bbb_permeability",
            basis="predicted",
            desirability=Desirability(kind="higher_better", good=0.8, bad=0.3),
            explanation=(
                "Model-predicted probability of crossing the blood-brain barrier. "
                "Essential for a CNS target; the BBBP training set is small and "
                "class-imbalanced, so treat this as a coarse filter."
            ),
        )
    )
    # CNS drugs live in a tighter property window than oral drugs generally.
    return [
        Objective(
            key=objective.key,
            label=objective.label,
            weight=objective.weight,
            source_field=objective.source_field,
            basis=objective.basis,
            desirability=Desirability(kind="window", low=200.0, high=400.0, tolerance=120.0)
            if objective.key == "molecular_weight"
            else objective.desirability,
            explanation=objective.explanation
            + (" CNS programmes favour 200-400 Da." if objective.key == "molecular_weight" else ""),
            required=objective.required,
        )
        for objective in objectives
    ]


DEFAULT_FILTERS = [
    HardFilter(
        key="not_withdrawn",
        label="Not a withdrawn drug",
        source_field="withdrawn_flag",
        operator="is_false",
        reason="Withdrawn from market for safety reasons.",
    ),
    HardFilter(
        key="heavy_atom_floor",
        label="At least 10 heavy atoms",
        source_field="heavy_atom_count",
        operator=">=",
        value=10,
        reason="Fragments below 10 heavy atoms are not comparable to lead-like compounds.",
        enabled=True,
    ),
    HardFilter(
        key="molecular_weight_ceiling",
        label="Molecular weight below 800 Da",
        source_field="molecular_weight",
        operator="<=",
        value=800.0,
        reason="Beyond 800 Da the oral small-molecule property models do not apply.",
    ),
]


DEFAULT_CONFIG = RankingConfig(
    name="general_oral_small_molecule",
    description=(
        "Balanced prioritisation for an oral small-molecule programme: measured potency "
        "and target evidence dominate, with predicted properties, drug-likeness, "
        "novelty and model confidence as modifiers."
    ),
    objectives=_general_objectives(),
    filters=list(DEFAULT_FILTERS),
    notes=[
        "Scores express prioritisation for follow-up, not predicted clinical efficacy.",
        "Predicted objectives are labelled 'predicted' and discounted when confidence is low.",
    ],
)

CNS_CONFIG = RankingConfig(
    name="cns_small_molecule",
    description=(
        "Prioritisation for a CNS programme: adds predicted blood-brain barrier "
        "permeability and tightens the molecular-weight window."
    ),
    objectives=_cns_objectives(),
    filters=list(DEFAULT_FILTERS),
    notes=[
        "BBB predictions come from a small, imbalanced public dataset (BBBP); use as a "
        "coarse triage signal only.",
    ],
)

EVIDENCE_FIRST_CONFIG = RankingConfig(
    name="evidence_first",
    description=(
        "Conservative configuration that ranks almost entirely on retrieved "
        "experimental evidence, with predictions as tie-breakers only."
    ),
    objectives=[
        Objective(
            key=objective.key,
            label=objective.label,
            weight=objective.weight * (3.0 if objective.basis == "measured" else 0.25),
            source_field=objective.source_field,
            basis=objective.basis,
            desirability=objective.desirability,
            explanation=objective.explanation,
            required=objective.required,
        )
        for objective in _general_objectives()
    ],
    filters=list(DEFAULT_FILTERS),
    min_objectives_present=2,
    notes=["Use when model predictions are not trusted for the chemistry in question."],
)

PRESETS: dict[str, RankingConfig] = {
    DEFAULT_CONFIG.name: DEFAULT_CONFIG,
    CNS_CONFIG.name: CNS_CONFIG,
    EVIDENCE_FIRST_CONFIG.name: EVIDENCE_FIRST_CONFIG,
}


def preset(name: str) -> RankingConfig:
    config = PRESETS.get(name)
    if config is None:
        raise ValidationError(
            f"unknown ranking preset {name!r}", details={"available": sorted(PRESETS)}
        )
    return config


def list_presets() -> list[dict[str, Any]]:
    return [
        {
            "name": config.name,
            "description": config.description,
            "n_objectives": len(config.objectives),
            "config_hash": config.config_hash,
        }
        for config in PRESETS.values()
    ]
