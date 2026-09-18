"""ML serving.

What this layer adds over calling ``model.predict`` directly:

* **Model resolution.** Callers ask for an *endpoint* ("solubility"); the service picks
  the active model version, or honours an explicit pin, and always reports which
  version answered.
* **Applicability domain.** Every prediction carries the nearest-neighbour Tanimoto to
  the model's training reference set and an in/out-of-domain verdict. Out-of-domain
  predictions are flagged, not hidden and not silently served as equivalent.
* **Confidence.** A single [0, 1] number combining ensemble/MC-dropout spread with the
  applicability score, defined in one place so the UI, the ranking system and the agent
  all mean the same thing by "confidence".
* **Idempotent persistence.** Predictions are written with a hash of the featurised
  input, so re-scoring the same molecule with the same model version updates one row
  instead of accumulating duplicates -- and a research run can point at the exact row.
* **Caching.** Redis short-circuits repeat scoring inside a session; the database is the
  durable record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.chem.molecule import safe_standardize
from theragraph.config import Settings
from theragraph.core.cache import Cache, cache_key
from theragraph.core.clock import Stopwatch
from theragraph.core.errors import ModelNotAvailableError, ValidationError
from theragraph.core.hashing import hash_payload
from theragraph.core.logging import get_logger
from theragraph.core.metrics import INFERENCE_LATENCY, INFERENCE_PREDICTIONS
from theragraph.core.storage import ObjectStore
from theragraph.core.telemetry import span
from theragraph.db.models import MLModel
from theragraph.db.repositories import ml as ml_repo
from theragraph.ml import registry
from theragraph.ml.evaluation import AD_SIMILARITY_THRESHOLD, nearest_training_similarity

log = get_logger(__name__)

#: Similarity at which a molecule counts as fully inside the applicability domain.
FULL_DOMAIN_SIMILARITY = 0.55

#: Endpoints whose higher values are *worse* for a candidate. Used to render direction
#: in the UI and to orient the ranking desirability curves.
ENDPOINT_DIRECTION: dict[str, str] = {
    "solubility": "higher_is_better",
    "lipophilicity": "target_window",
    "bbb_permeability": "context_dependent",
    "clinical_toxicity": "lower_is_better",
    "bioactivity": "higher_is_better",
}


@dataclass
class MoleculePrediction:
    """One model's answer for one molecule, with every caveat attached."""

    compound_id: str | None
    smiles: str
    endpoint: str
    model_ref: str
    model_id: str
    model_family: str
    task_type: str
    value: float | None = None
    probability: float | None = None
    predicted_class: str | None = None
    uncertainty: float | None = None
    confidence: float | None = None
    units: str | None = None
    nearest_training_similarity: float | None = None
    in_domain: bool | None = None
    flags: list[str] = field(default_factory=list)
    input_hash: str = ""
    latency_ms: float | None = None
    prediction_id: str | None = None
    cache_hit: bool = False
    positive_class_meaning: str | None = None
    direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "compound_id": self.compound_id,
            "smiles": self.smiles,
            "endpoint": self.endpoint,
            "model": self.model_ref,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "task_type": self.task_type,
            "value": self.value,
            "probability": self.probability,
            "predicted_class": self.predicted_class,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "units": self.units,
            "applicability": {
                "nearest_training_similarity": self.nearest_training_similarity,
                "in_domain": self.in_domain,
                "threshold": AD_SIMILARITY_THRESHOLD,
            },
            "flags": self.flags,
            "prediction_id": self.prediction_id,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "positive_class_meaning": self.positive_class_meaning,
            "direction": self.direction,
            "is_prediction": True,
            "disclaimer": ("Model prediction, not an experimental measurement."),
        }


@dataclass
class InferenceResult:
    predictions: list[MoleculePrediction] = field(default_factory=list)
    models_used: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def by_compound(self) -> dict[str, list[MoleculePrediction]]:
        grouped: dict[str, list[MoleculePrediction]] = {}
        for prediction in self.predictions:
            key = prediction.compound_id or prediction.smiles
            grouped.setdefault(key, []).append(prediction)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "models_used": self.models_used,
            "failures": self.failures,
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


def compute_confidence(
    *,
    task_type: str,
    uncertainty: float | None,
    nearest_similarity: float | None,
    label_scale: float = 1.0,
) -> float | None:
    """Combine ensemble spread and applicability into one [0, 1] confidence.

    Two components, equally weighted:

    * *Precision* -- how tightly the ensemble agrees, normalised by the label spread for
      regression (so 0.3 log-units means something different on a dataset with std 2.1
      than on one with std 0.4) or by the 0.5 maximum std of a Bernoulli for
      classification.
    * *Applicability* -- nearest-neighbour Tanimoto to the training set, saturating at
      :data:`FULL_DOMAIN_SIMILARITY`.

    Returning ``None`` when a model cannot report spread is deliberate: a fabricated
    confidence is worse than an absent one.
    """
    if uncertainty is None and nearest_similarity is None:
        return None

    components: list[float] = []
    if uncertainty is not None:
        if task_type == "classification":
            precision = 1.0 - min(1.0, float(uncertainty) / 0.5)
        else:
            scale = label_scale if label_scale and label_scale > 1e-9 else 1.0
            precision = 1.0 - min(1.0, float(uncertainty) / scale)
        components.append(max(0.0, precision))
    if nearest_similarity is not None:
        components.append(min(1.0, float(nearest_similarity) / FULL_DOMAIN_SIMILARITY))

    return round(float(sum(components) / len(components)), 4)


class InferenceService:
    """Resolves models, scores molecules, persists predictions."""

    def __init__(
        self,
        settings: Settings,
        *,
        cache: Cache | None = None,
        store: ObjectStore | None = None,
        model_cache: registry.ModelCache | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.store = store
        self.model_cache = model_cache or registry.ModelCache()

    # ----------------------------------------------------------- model resolution
    async def resolve_model(
        self,
        session: AsyncSession,
        *,
        endpoint: str,
        model_name: str | None = None,
        model_version: str | None = None,
        family: str | None = None,
    ) -> MLModel:
        if model_name:
            row = await ml_repo.get_model(session, model_name, model_version)
        else:
            row = await ml_repo.get_active_model_for_endpoint(session, endpoint, family=family)
        if row is None:
            raise ModelNotAvailableError(
                f"no active model registered for endpoint {endpoint!r}",
                details={
                    "endpoint": endpoint,
                    "hint": "train a model with `theragraph ml train --endpoint <name>`",
                },
            )
        return row

    async def load(self, row: MLModel) -> registry.LoadedModel:
        cached = self.model_cache.get(row)
        if cached is not None:
            return cached
        loaded = await registry.load_model(
            row, local_root=self.settings.model_registry_dir, store=self.store
        )
        self.model_cache.put(row, loaded)
        return loaded

    async def available_endpoints(self, session: AsyncSession) -> list[dict[str, Any]]:
        models = await ml_repo.list_models(session, status="active")
        by_endpoint: dict[str, list[MLModel]] = {}
        for model in models:
            by_endpoint.setdefault(model.endpoint, []).append(model)
        return [
            {
                "endpoint": endpoint,
                "task_type": rows[0].task_type,
                "units": rows[0].units,
                "direction": ENDPOINT_DIRECTION.get(endpoint),
                "positive_class_meaning": rows[0].positive_class_meaning,
                "models": [
                    {"ref": row.ref, "family": row.family, "version": row.version} for row in rows
                ],
            }
            for endpoint, rows in sorted(by_endpoint.items())
        ]

    # ---------------------------------------------------------------- prediction
    async def predict(
        self,
        session: AsyncSession,
        *,
        molecules: list[tuple[str | None, str]],
        endpoints: list[str],
        model_pins: dict[str, str] | None = None,
        family: str | None = None,
        persist: bool = True,
        research_run_id: str | None = None,
        use_cache: bool = True,
    ) -> InferenceResult:
        """Score ``(compound_id, smiles)`` pairs across one or more endpoints."""
        if not molecules:
            return InferenceResult()
        if not endpoints:
            raise ValidationError("at least one endpoint must be requested")

        result = InferenceResult()
        overall = Stopwatch()

        standardized: list[tuple[str | None, str]] = []
        for compound_id, smiles in molecules:
            info, error = safe_standardize(smiles)
            if info is None:
                result.failures.append(
                    {"compound_id": compound_id, "smiles": smiles[:120], "error": error}
                )
                continue
            standardized.append((compound_id, info.canonical_smiles))

        with overall:
            for endpoint in endpoints:
                pin = (model_pins or {}).get(endpoint)
                name, version = _parse_pin(pin) if pin else (None, None)
                try:
                    row = await self.resolve_model(
                        session,
                        endpoint=endpoint,
                        model_name=name,
                        model_version=version,
                        family=family,
                    )
                    loaded = await self.load(row)
                except ModelNotAvailableError as exc:
                    result.failures.append(
                        {"endpoint": endpoint, "error": exc.message, "code": exc.code}
                    )
                    continue

                endpoint_predictions = await self._predict_endpoint(
                    session,
                    loaded=loaded,
                    molecules=standardized,
                    persist=persist,
                    research_run_id=research_run_id,
                    use_cache=use_cache,
                )
                result.predictions.extend(endpoint_predictions)
                result.models_used.append(
                    {
                        "endpoint": endpoint,
                        "ref": row.ref,
                        "model_id": row.id,
                        "family": row.family,
                        "version": row.version,
                        "task_type": row.task_type,
                        "units": row.units,
                        "trained_on": row.training_dataset_version_id,
                        "scaffold_metrics": (row.metrics or {}).get("scaffold"),
                    }
                )

        result.total_latency_ms = overall.elapsed_ms
        return result

    async def _predict_endpoint(
        self,
        session: AsyncSession,
        *,
        loaded: registry.LoadedModel,
        molecules: list[tuple[str | None, str]],
        persist: bool,
        research_run_id: str | None,
        use_cache: bool,
    ) -> list[MoleculePrediction]:
        row = loaded.row
        feature_spec = row.feature_spec or {}

        pending: list[tuple[str | None, str, str]] = []
        predictions: list[MoleculePrediction] = []

        for compound_id, smiles in molecules:
            input_hash = hash_payload(
                {
                    "smiles": smiles,
                    "model": row.id,
                    "version": row.version,
                    "feature_spec": feature_spec,
                },
                length=32,
            )
            cached = (
                await self._cached_prediction(loaded, smiles, input_hash) if use_cache else None
            )
            if cached is not None:
                predictions.append(cached)
                continue
            pending.append((compound_id, smiles, input_hash))

        if not pending:
            return predictions

        stopwatch = Stopwatch()
        with span("ml.inference", model=row.ref, n=len(pending)), stopwatch:
            batch = loaded.predict([smiles for _, smiles, _ in pending])
        per_molecule_latency = stopwatch.elapsed_ms / max(1, len(pending))
        INFERENCE_LATENCY.labels(model=row.name, endpoint=row.endpoint).observe(
            stopwatch.elapsed_ms / 1000
        )
        INFERENCE_PREDICTIONS.labels(model=row.name, endpoint=row.endpoint).inc(len(pending))

        training_bits = loaded.training_fingerprints()
        rows_to_persist: list[dict[str, Any]] = []

        for index, (compound_id, smiles, input_hash) in enumerate(pending):
            uncertainty = (
                float(batch.uncertainty[index])
                if batch.uncertainty is not None and index < len(batch.uncertainty)
                else None
            )
            similarity = (
                nearest_training_similarity(smiles, training_bits) if training_bits else None
            )
            in_domain = similarity >= AD_SIMILARITY_THRESHOLD if similarity is not None else None
            confidence = compute_confidence(
                task_type=row.task_type,
                uncertainty=uncertainty,
                nearest_similarity=similarity,
                label_scale=loaded.label_scale,
            )

            flags: list[str] = []
            if in_domain is False:
                flags.append("out_of_domain")
            if confidence is not None and confidence < 0.4:
                flags.append("low_confidence")
            if uncertainty is None:
                flags.append("no_uncertainty_estimate")

            prediction = MoleculePrediction(
                compound_id=compound_id,
                smiles=smiles,
                endpoint=row.endpoint,
                model_ref=row.ref,
                model_id=row.id,
                model_family=row.family,
                task_type=row.task_type,
                uncertainty=round(uncertainty, 5) if uncertainty is not None else None,
                confidence=confidence,
                units=row.units,
                nearest_training_similarity=round(similarity, 4)
                if similarity is not None
                else None,
                in_domain=in_domain,
                flags=flags,
                input_hash=input_hash,
                latency_ms=round(per_molecule_latency, 3),
                positive_class_meaning=row.positive_class_meaning,
                direction=ENDPOINT_DIRECTION.get(row.endpoint),
            )

            if row.task_type == "classification":
                probability = (
                    float(batch.probabilities[index]) if batch.probabilities is not None else None
                )
                prediction.probability = round(probability, 5) if probability is not None else None
                prediction.predicted_class = (
                    ("positive" if probability >= 0.5 else "negative")
                    if probability is not None
                    else None
                )
            else:
                value = float(batch.values[index]) if batch.values is not None else None
                prediction.value = round(value, 5) if value is not None else None

            predictions.append(prediction)

            if persist and compound_id:
                rows_to_persist.append(
                    {
                        "compound_id": compound_id,
                        "model_id": row.id,
                        "endpoint": row.endpoint,
                        "task_type": row.task_type,
                        "value": prediction.value,
                        "probability": prediction.probability,
                        "predicted_class": prediction.predicted_class,
                        "uncertainty": prediction.uncertainty,
                        "confidence": prediction.confidence,
                        "applicability": {
                            "nearest_training_similarity": prediction.nearest_training_similarity,
                            "in_domain": in_domain,
                            "threshold": AD_SIMILARITY_THRESHOLD,
                            "training_reference_size": len(training_bits),
                        },
                        "input_hash": input_hash,
                        "latency_ms": prediction.latency_ms,
                        "research_run_id": research_run_id,
                    }
                )

        if rows_to_persist:
            persisted_ids = await ml_repo.upsert_predictions(session, rows_to_persist)
            by_hash = {
                persisted["input_hash"]: prediction_id
                for persisted, prediction_id in zip(rows_to_persist, persisted_ids, strict=False)
            }
            for prediction in predictions:
                if prediction.prediction_id is None:
                    prediction.prediction_id = by_hash.get(prediction.input_hash)

        if self.cache is not None:
            for prediction in predictions:
                if not prediction.cache_hit:
                    await self.cache.set_json(
                        cache_key("prediction", row.id, prediction.input_hash),
                        prediction.to_dict(),
                        ttl_s=self.settings.cache_default_ttl_s,
                    )

        return predictions

    async def _cached_prediction(
        self, loaded: registry.LoadedModel, smiles: str, input_hash: str
    ) -> MoleculePrediction | None:
        if self.cache is None:
            return None
        payload = await self.cache.get_json(cache_key("prediction", loaded.row.id, input_hash))
        if not payload:
            return None
        applicability = payload.get("applicability") or {}
        return MoleculePrediction(
            compound_id=payload.get("compound_id"),
            smiles=smiles,
            endpoint=payload["endpoint"],
            model_ref=payload["model"],
            model_id=payload["model_id"],
            model_family=payload.get("model_family", loaded.row.family),
            task_type=payload["task_type"],
            value=payload.get("value"),
            probability=payload.get("probability"),
            predicted_class=payload.get("predicted_class"),
            uncertainty=payload.get("uncertainty"),
            confidence=payload.get("confidence"),
            units=payload.get("units"),
            nearest_training_similarity=applicability.get("nearest_training_similarity"),
            in_domain=applicability.get("in_domain"),
            flags=payload.get("flags") or [],
            input_hash=input_hash,
            prediction_id=payload.get("prediction_id"),
            cache_hit=True,
            positive_class_meaning=payload.get("positive_class_meaning"),
            direction=payload.get("direction"),
        )

    async def warm(self, session: AsyncSession, endpoints: list[str]) -> dict[str, Any]:
        """Pre-load models at startup so the first user request is not the slow one."""
        loaded: list[str] = []
        errors: list[dict[str, str]] = []
        for endpoint in endpoints:
            try:
                row = await self.resolve_model(session, endpoint=endpoint)
                await self.load(row)
                loaded.append(row.ref)
            except Exception as exc:
                errors.append({"endpoint": endpoint, "error": str(exc)})
        return {"loaded": loaded, "errors": errors, **self.model_cache.stats()}


def _parse_pin(pin: str) -> tuple[str, str | None]:
    """``'solubility-gin@2026.09.18.abc'`` -> ``('solubility-gin', '2026.09.18.abc')``."""
    if "@" in pin:
        name, version = pin.split("@", 1)
        return name, version
    return pin, None


def summarize_predictions(predictions: list[MoleculePrediction]) -> dict[str, Any]:
    """Compact summary used in agent tool output and report tables."""
    by_endpoint: dict[str, list[MoleculePrediction]] = {}
    for prediction in predictions:
        by_endpoint.setdefault(prediction.endpoint, []).append(prediction)

    summary: dict[str, Any] = {}
    for endpoint, group in by_endpoint.items():
        values = [
            prediction.probability if prediction.probability is not None else prediction.value
            for prediction in group
        ]
        numeric = [value for value in values if value is not None]
        summary[endpoint] = {
            "n": len(group),
            "model": group[0].model_ref,
            "mean": round(float(np.mean(numeric)), 4) if numeric else None,
            "min": round(float(np.min(numeric)), 4) if numeric else None,
            "max": round(float(np.max(numeric)), 4) if numeric else None,
            "out_of_domain": sum(1 for p in group if p.in_domain is False),
            "low_confidence": sum(1 for p in group if "low_confidence" in p.flags),
        }
    return summary
