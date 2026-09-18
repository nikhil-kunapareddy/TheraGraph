"""Model registry, evaluation and experiment endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query

from theragraph.api.deps import ContextDep
from theragraph.core.errors import NotFoundError
from theragraph.db.repositories import ml as ml_repo

router = APIRouter(tags=["models"])


def _model_payload(model: Any, *, include_evaluations: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": model.id,
        "ref": model.ref,
        "name": model.name,
        "version": model.version,
        "family": model.family,
        "endpoint": model.endpoint,
        "task_type": model.task_type,
        "status": model.status,
        "framework": f"{model.framework} {model.framework_version or ''}".strip(),
        "description": model.description,
        "units": model.units,
        "positive_class_meaning": model.positive_class_meaning,
        "split_strategy": model.split_strategy,
        "metrics": model.metrics,
        "calibration": {
            key: value for key, value in (model.calibration or {}).items() if key != "reliability"
        },
        "hyperparameters": model.hyperparameters,
        "feature_spec": {
            key: value
            for key, value in (model.feature_spec or {}).items()
            # The scaler statistics are hundreds of floats; useful in the artifact,
            # noise in an API response.
            if key not in ("descriptor_mean", "descriptor_scale", "descriptor_names")
        },
        "n_descriptor_features": len((model.feature_spec or {}).get("descriptor_names") or []),
        "training_dataset_version_id": model.training_dataset_version_id,
        "experiment_id": model.experiment_id,
        "trained_at": model.trained_at.isoformat() if model.trained_at else None,
        "training_runtime_s": model.training_runtime_s,
        "artifact_hash": model.artifact_hash,
        "artifact_size_bytes": model.artifact_size_bytes,
        "git_sha": model.git_sha,
        "random_seed": model.random_seed,
    }
    if include_evaluations:
        payload["evaluations"] = [
            {
                "id": evaluation.id,
                "split_strategy": evaluation.split_strategy,
                "split_name": evaluation.split_name,
                "n_samples": evaluation.n_samples,
                "metrics": evaluation.metrics,
                "calibration": evaluation.calibration,
                "error_analysis": evaluation.error_analysis,
                "prediction_distribution": evaluation.prediction_distribution,
                "per_fold_metrics": evaluation.per_fold_metrics,
                "notes": evaluation.notes,
            }
            for evaluation in sorted(model.evaluations, key=lambda item: item.split_strategy)
        ]
    return payload


@router.get("/models", summary="Registered models")
async def list_models(
    context: ContextDep,
    endpoint: Annotated[str | None, Query()] = None,
    family: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    models = await ml_repo.list_models(
        context.session, endpoint=endpoint, family=family, status=status
    )
    return {
        "models": [_model_payload(model) for model in models],
        "count": len(models),
        "reporting_note": (
            "Headline metrics are scaffold-split. Random-split figures are included only "
            "to quantify how optimistic a random split is on molecular data."
        ),
    }


@router.get("/models/{model_id}", summary="Model detail with full evaluations")
async def get_model(context: ContextDep, model_id: Annotated[str, Path()]) -> dict[str, Any]:
    model = await ml_repo.get_model_by_id(context.session, model_id)
    if model is None:
        raise NotFoundError(f"model {model_id!r} not found")
    await context.session.refresh(model, ["evaluations"])
    payload = _model_payload(model, include_evaluations=True)
    payload["prediction_distribution_served"] = await ml_repo.prediction_distribution(
        context.session, model.id
    )
    payload["reliability"] = (model.calibration or {}).get("reliability")
    return payload


@router.get("/evaluations", summary="Model evaluations")
async def list_evaluations(
    context: ContextDep,
    model_id: Annotated[str | None, Query()] = None,
    endpoint: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    evaluations = await ml_repo.list_evaluations(
        context.session, model_id=model_id, endpoint=endpoint
    )
    return {
        "evaluations": [
            {
                "id": evaluation.id,
                "model": evaluation.model.ref if evaluation.model else None,
                "model_id": evaluation.model_id,
                "endpoint": evaluation.model.endpoint if evaluation.model else None,
                "family": evaluation.model.family if evaluation.model else None,
                "split_strategy": evaluation.split_strategy,
                "split_name": evaluation.split_name,
                "n_samples": evaluation.n_samples,
                "metrics": evaluation.metrics,
                "calibration": evaluation.calibration,
                "error_analysis": evaluation.error_analysis,
                "prediction_distribution": evaluation.prediction_distribution,
                "per_fold_metrics": evaluation.per_fold_metrics,
                "notes": evaluation.notes,
                "created_at": evaluation.created_at.isoformat(),
            }
            for evaluation in evaluations
        ],
        "count": len(evaluations),
    }


@router.get("/experiments", summary="Training experiments")
async def list_experiments(
    context: ContextDep,
    endpoint: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> dict[str, Any]:
    experiments = await ml_repo.list_experiments(context.session, endpoint=endpoint, limit=limit)
    return {
        "experiments": [
            {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "endpoint": experiment.endpoint,
                "status": experiment.status,
                "dataset_version_id": experiment.dataset_version_id,
                "parameters": experiment.parameters,
                "metrics": experiment.metrics,
                "comparison": experiment.comparison,
                "artifacts": experiment.artifacts,
                "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                "finished_at": experiment.finished_at.isoformat()
                if experiment.finished_at
                else None,
                "runtime_s": experiment.runtime_s,
                "git_sha": experiment.git_sha,
                "random_seed": experiment.random_seed,
                "environment": experiment.environment,
                "error": experiment.error,
                "models": [model.ref for model in experiment.models],
            }
            for experiment in experiments
        ],
        "count": len(experiments),
    }


@router.get("/experiments/{experiment_id}", summary="Experiment detail")
async def get_experiment(
    context: ContextDep, experiment_id: Annotated[str, Path(description="e.g. EXP-2026-00007")]
) -> dict[str, Any]:
    experiment = await ml_repo.get_experiment(context.session, experiment_id)
    if experiment is None:
        raise NotFoundError(f"experiment {experiment_id!r} not found")
    return {
        "id": experiment.id,
        "name": experiment.name,
        "description": experiment.description,
        "endpoint": experiment.endpoint,
        "status": experiment.status,
        "dataset_version_id": experiment.dataset_version_id,
        "parameters": experiment.parameters,
        "metrics": experiment.metrics,
        "comparison": experiment.comparison,
        "artifacts": experiment.artifacts,
        "environment": experiment.environment,
        "runtime_s": experiment.runtime_s,
        "git_sha": experiment.git_sha,
        "random_seed": experiment.random_seed,
        "error": experiment.error,
        "models": [_model_payload(model, include_evaluations=True) for model in experiment.models],
    }
