"""The prioritisation engine.

Takes candidate compounds with whatever evidence and predictions exist for them, applies
hard filters, scores each objective through its desirability curve, and returns a ranked
list where **every point of every score is attributable**.

The output is explicitly framed as prioritisation for experimental follow-up. It is not
a prediction of clinical efficacy, and the engine attaches that framing to its own
output rather than relying on the UI to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theragraph.core.logging import get_logger
from theragraph.ranking.config import DEFAULT_CONFIG, Basis, Objective, RankingConfig

log = get_logger(__name__)

DISCLAIMER = (
    "Prioritisation score for experimental follow-up. Combines retrieved experimental "
    "evidence with model predictions; it is not a prediction of clinical efficacy or "
    "safety, and no ranking here has been experimentally validated."
)


@dataclass
class CandidateInput:
    """Everything known about one candidate at ranking time.

    ``predictions`` maps endpoint -> ``{value|probability, confidence, in_domain, ...}``
    as produced by the inference service. ``evidence_ids`` and ``prediction_ids`` are
    carried through so a ranked row links back to the exact records behind it.
    """

    compound_id: str
    name: str | None = None
    smiles: str | None = None
    chembl_id: str | None = None
    best_pchembl: float | None = None
    n_measurements: int = 0
    evidence_score: float | None = None
    evidence_count: int = 0
    max_phase: float | None = None
    withdrawn_flag: bool = False
    molecular_weight: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    qed: float | None = None
    heavy_atom_count: int | None = None
    hbd: int | None = None
    hba: int | None = None
    rotatable_bonds: int | None = None
    passes_lipinski: bool | None = None
    novelty_score: float | None = None
    scaffold_frequency: int | None = None
    predictions: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    prediction_ids: list[str] = field(default_factory=list)

    def resolve(self, source_field: str) -> tuple[float | None, dict[str, Any]]:
        """Resolve an objective's ``source_field`` to a value plus its context.

        ``prediction.<endpoint>`` reads from the predictions map and brings the
        prediction's confidence and domain verdict along, which is what allows the
        engine to discount uncertain predictions.
        """
        if source_field.startswith("prediction."):
            endpoint = source_field.split(".", 1)[1]
            prediction = self.predictions.get(endpoint)
            if not prediction:
                return None, {"missing_reason": f"no {endpoint} prediction available"}
            raw = (
                prediction.get("probability")
                if prediction.get("probability") is not None
                else prediction.get("value")
            )
            return (
                float(raw) if raw is not None else None,
                {
                    "model": prediction.get("model"),
                    "confidence": prediction.get("confidence"),
                    "in_domain": (prediction.get("applicability") or {}).get("in_domain"),
                    "uncertainty": prediction.get("uncertainty"),
                    "units": prediction.get("units"),
                    "prediction_id": prediction.get("prediction_id"),
                },
            )

        if source_field == "mean_confidence":
            confidences = [
                float(confidence)
                for prediction in self.predictions.values()
                if (confidence := prediction.get("confidence")) is not None
            ]
            if not confidences:
                return None, {"missing_reason": "no predictions with confidence"}
            return sum(confidences) / len(confidences), {"n_predictions": len(confidences)}

        value = getattr(self, source_field, None)
        if value is None:
            return None, {"missing_reason": f"{source_field} not available"}
        if isinstance(value, bool):
            return (1.0 if value else 0.0), {}
        try:
            return float(value), {}
        except (TypeError, ValueError):
            return None, {"missing_reason": f"{source_field} is not numeric"}


@dataclass
class FactorScore:
    """One objective's contribution to one candidate's total."""

    key: str
    label: str
    basis: Basis
    raw_value: float | None
    desirability: float | None
    weight: float
    normalized_weight: float
    contribution: float
    explanation: str
    curve: str
    context: dict[str, Any] = field(default_factory=dict)
    discounted: bool = False
    missing_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "basis": self.basis,
            "raw_value": self.raw_value,
            "desirability": self.desirability,
            "weight": self.weight,
            "normalized_weight": round(self.normalized_weight, 4),
            "contribution": round(self.contribution, 4),
            "explanation": self.explanation,
            "curve": self.curve,
            "context": self.context,
            "discounted_for_low_confidence": self.discounted,
            "missing_reason": self.missing_reason,
        }


@dataclass
class RankedCandidate:
    compound_id: str
    rank: int
    total_score: float
    factors: list[FactorScore]
    flags: list[str] = field(default_factory=list)
    name: str | None = None
    smiles: str | None = None
    chembl_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    prediction_ids: list[str] = field(default_factory=list)
    objectives_scored: int = 0
    objectives_missing: int = 0

    @property
    def top_contributors(self) -> list[dict[str, Any]]:
        ranked = sorted(self.factors, key=lambda factor: -factor.contribution)
        return [factor.to_dict() for factor in ranked[:3] if factor.contribution > 0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compound_id": self.compound_id,
            "rank": self.rank,
            "total_score": round(self.total_score, 4),
            "name": self.name,
            "smiles": self.smiles,
            "chembl_id": self.chembl_id,
            "factors": [factor.to_dict() for factor in self.factors],
            "top_contributors": self.top_contributors,
            "flags": self.flags,
            "evidence_ids": self.evidence_ids,
            "prediction_ids": self.prediction_ids,
            "objectives_scored": self.objectives_scored,
            "objectives_missing": self.objectives_missing,
            "disclaimer": DISCLAIMER,
        }


@dataclass
class FilteredCandidate:
    compound_id: str
    name: str | None
    failed_filter: str
    reason: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "compound_id": self.compound_id,
            "name": self.name,
            "failed_filter": self.failed_filter,
            "reason": self.reason,
            "value": self.value,
        }


@dataclass
class RankingOutcome:
    config: RankingConfig
    ranked: list[RankedCandidate]
    filtered_out: list[FilteredCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, *, include_factors: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash,
            "n_ranked": len(self.ranked),
            "n_filtered_out": len(self.filtered_out),
            "filtered_out": [item.to_dict() for item in self.filtered_out],
            "warnings": self.warnings,
            "disclaimer": DISCLAIMER,
        }
        payload["candidates"] = [
            candidate.to_dict()
            if include_factors
            else {
                "compound_id": candidate.compound_id,
                "rank": candidate.rank,
                "total_score": round(candidate.total_score, 4),
                "name": candidate.name,
                "flags": candidate.flags,
                "top_contributors": candidate.top_contributors,
            }
            for candidate in self.ranked
        ]
        return payload

    def score_rows(self) -> list[dict[str, Any]]:
        """Rows for ``candidate_score`` persistence."""
        return [
            {
                "compound_id": candidate.compound_id,
                "rank": candidate.rank,
                "total_score": candidate.total_score,
                "factors": [factor.to_dict() for factor in candidate.factors],
                "flags": candidate.flags,
                "evidence_ids": candidate.evidence_ids,
                "prediction_ids": candidate.prediction_ids,
            }
            for candidate in self.ranked
        ]


def _apply_filters(candidate: CandidateInput, config: RankingConfig) -> FilteredCandidate | None:
    for hard_filter in config.filters:
        if not hard_filter.enabled:
            continue
        value, _ = candidate.resolve(hard_filter.source_field)
        raw = getattr(candidate, hard_filter.source_field, value)
        verdict = hard_filter.passes(raw)
        if verdict is False:
            return FilteredCandidate(
                compound_id=candidate.compound_id,
                name=candidate.name,
                failed_filter=hard_filter.key,
                reason=hard_filter.reason or hard_filter.label,
                value=raw,
            )
    return None


def _score_objective(
    candidate: CandidateInput,
    objective: Objective,
    normalized_weight: float,
    *,
    confidence_penalty: float,
) -> FactorScore:
    raw_value, context = candidate.resolve(objective.source_field)

    if raw_value is None:
        return FactorScore(
            key=objective.key,
            label=objective.label,
            basis=objective.basis,
            raw_value=None,
            desirability=None,
            weight=objective.weight,
            normalized_weight=normalized_weight,
            contribution=0.0,
            explanation=objective.explanation,
            curve=objective.desirability.describe(),
            context=context,
            missing_reason=context.get("missing_reason", "value unavailable"),
        )

    desirability = objective.desirability.apply(raw_value)

    # Predictions with low confidence or out-of-domain inputs contribute less, so a
    # candidate cannot climb the ranking on the strength of an extrapolation.
    discounted = False
    effective = desirability
    confidence = context.get("confidence")
    in_domain = context.get("in_domain")
    if objective.basis == "predicted" and (
        in_domain is False or (confidence is not None and confidence < 0.4)
    ):
        effective = desirability * (1.0 - confidence_penalty)
        discounted = True

    return FactorScore(
        key=objective.key,
        label=objective.label,
        basis=objective.basis,
        raw_value=round(raw_value, 5),
        desirability=round(desirability, 4),
        weight=objective.weight,
        normalized_weight=normalized_weight,
        contribution=effective * normalized_weight,
        explanation=objective.explanation,
        curve=objective.desirability.describe(),
        context=context,
        discounted=discounted,
    )


def rank_candidates(
    candidates: list[CandidateInput],
    config: RankingConfig | None = None,
) -> RankingOutcome:
    """Filter, score and rank candidates.

    Weight renormalisation is the subtle part. If a candidate has no toxicity prediction,
    its remaining objectives are renormalised to sum to 1 so its total stays comparable
    to a fully characterised candidate's, and the omission is recorded on the row. The
    alternative -- scoring missing data as zero -- would systematically bury exactly the
    novel compounds a discovery programme is looking for.
    """
    active_config = config or DEFAULT_CONFIG
    weights = active_config.normalized_weights()
    outcome = RankingOutcome(config=active_config, ranked=[])

    scored: list[RankedCandidate] = []
    for candidate in candidates:
        rejection = _apply_filters(candidate, active_config)
        if rejection is not None:
            outcome.filtered_out.append(rejection)
            continue

        factors = [
            _score_objective(
                candidate,
                objective,
                weights[objective.key],
                confidence_penalty=active_config.confidence_penalty,
            )
            for objective in active_config.objectives
        ]

        missing_required = [
            factor.key
            for factor, objective in zip(factors, active_config.objectives, strict=True)
            if objective.required and factor.raw_value is None
        ]
        if missing_required:
            outcome.filtered_out.append(
                FilteredCandidate(
                    compound_id=candidate.compound_id,
                    name=candidate.name,
                    failed_filter="required_objective_missing",
                    reason=f"missing required objectives: {', '.join(missing_required)}",
                    value=None,
                )
            )
            continue

        present = [factor for factor in factors if factor.raw_value is not None]
        missing = [factor for factor in factors if factor.raw_value is None]

        present_weight = sum(factor.normalized_weight for factor in present)
        if present_weight <= 0:
            outcome.filtered_out.append(
                FilteredCandidate(
                    compound_id=candidate.compound_id,
                    name=candidate.name,
                    failed_filter="no_scorable_objectives",
                    reason="none of the configured objectives had data for this compound",
                    value=None,
                )
            )
            continue

        renormalization = 1.0 / present_weight
        total = 0.0
        for factor in present:
            factor.contribution *= renormalization
            factor.normalized_weight *= renormalization
            total += factor.contribution

        flags: list[str] = []
        if len(present) < active_config.min_objectives_present:
            flags.append("under_characterised")
        if missing:
            flags.append(f"missing_{len(missing)}_objectives")
        if candidate.best_pchembl is None:
            flags.append("no_experimental_potency")
        if any(factor.discounted for factor in present):
            flags.append("prediction_discounted")
        if any(
            (prediction.get("applicability") or {}).get("in_domain") is False
            for prediction in candidate.predictions.values()
        ):
            flags.append("out_of_domain_prediction")
        if candidate.max_phase is not None and candidate.max_phase >= 4:
            flags.append("approved_drug")

        scored.append(
            RankedCandidate(
                compound_id=candidate.compound_id,
                rank=0,
                total_score=total,
                factors=factors,
                flags=flags,
                name=candidate.name,
                smiles=candidate.smiles,
                chembl_id=candidate.chembl_id,
                evidence_ids=candidate.evidence_ids,
                prediction_ids=candidate.prediction_ids,
                objectives_scored=len(present),
                objectives_missing=len(missing),
            )
        )

    # Ties broken by measured potency, then by number of measurements: when the score
    # cannot distinguish two compounds, prefer the better-characterised one.
    scored.sort(
        key=lambda item: (
            -item.total_score,
            -(item.objectives_scored),
            item.compound_id,
        )
    )
    for position, ranked in enumerate(scored, start=1):
        ranked.rank = position

    outcome.ranked = scored

    if outcome.filtered_out:
        outcome.warnings.append(f"{len(outcome.filtered_out)} candidate(s) removed by hard filters")
    under_characterised = sum(1 for candidate in scored if "under_characterised" in candidate.flags)
    if under_characterised:
        outcome.warnings.append(
            f"{under_characterised} candidate(s) scored on fewer than "
            f"{active_config.min_objectives_present} objectives; compare with care"
        )
    if not scored:
        outcome.warnings.append("no candidates could be ranked")

    log.info(
        "ranking.completed",
        config=active_config.name,
        ranked=len(scored),
        filtered=len(outcome.filtered_out),
    )
    return outcome


def explain_candidate(candidate: RankedCandidate) -> str:
    """Human-readable, source-aware explanation of a candidate's position."""
    lines = [
        f"Rank {candidate.rank} (score {candidate.total_score:.3f}) for "
        f"{candidate.name or candidate.compound_id}."
    ]
    for factor in sorted(candidate.factors, key=lambda item: -item.contribution):
        if factor.raw_value is None:
            lines.append(f"  - {factor.label}: no data ({factor.missing_reason})")
            continue
        basis_label = {
            "measured": "experimental",
            "predicted": "model prediction",
            "computed": "computed descriptor",
            "derived": "derived metric",
        }.get(factor.basis, factor.basis)
        suffix = " [discounted: low confidence]" if factor.discounted else ""
        lines.append(
            f"  - {factor.label}: {factor.raw_value} ({basis_label}) -> "
            f"desirability {factor.desirability:.2f} x weight "
            f"{factor.normalized_weight:.2f} = {factor.contribution:.3f}{suffix}"
        )
    if candidate.flags:
        lines.append(f"  Flags: {', '.join(candidate.flags)}")
    lines.append(f"  {DISCLAIMER}")
    return "\n".join(lines)


def compare_candidates(
    candidates: list[RankedCandidate], *, objective_keys: list[str] | None = None
) -> dict[str, Any]:
    """Side-by-side comparison matrix for the candidate comparison view."""
    if not candidates:
        return {"candidates": [], "objectives": []}

    keys = objective_keys or [factor.key for factor in candidates[0].factors]
    objectives = [
        {
            "key": factor.key,
            "label": factor.label,
            "basis": factor.basis,
            "explanation": factor.explanation,
            "curve": factor.curve,
        }
        for factor in candidates[0].factors
        if factor.key in keys
    ]

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        by_key = {factor.key: factor for factor in candidate.factors}
        rows.append(
            {
                "compound_id": candidate.compound_id,
                "name": candidate.name,
                "chembl_id": candidate.chembl_id,
                "rank": candidate.rank,
                "total_score": round(candidate.total_score, 4),
                "flags": candidate.flags,
                "values": {
                    key: {
                        "raw_value": by_key[key].raw_value,
                        "desirability": by_key[key].desirability,
                        "contribution": round(by_key[key].contribution, 4),
                        "missing_reason": by_key[key].missing_reason,
                        "discounted": by_key[key].discounted,
                    }
                    for key in keys
                    if key in by_key
                },
            }
        )

    # Per-objective best/worst, so the UI can highlight who wins each dimension.
    highlights: dict[str, Any] = {}
    for objective in objectives:
        key = objective["key"]
        scored = [
            (row["compound_id"], row["values"][key]["desirability"])
            for row in rows
            if row["values"].get(key, {}).get("desirability") is not None
        ]
        if scored:
            highlights[key] = {
                "best": max(scored, key=lambda item: item[1])[0],
                "worst": min(scored, key=lambda item: item[1])[0],
            }

    return {
        "candidates": rows,
        "objectives": objectives,
        "highlights": highlights,
        "disclaimer": DISCLAIMER,
    }
