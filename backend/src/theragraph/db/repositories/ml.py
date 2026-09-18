"""ML repository: model registry, experiments, evaluations, predictions, rankings."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from theragraph.core.clock import utcnow
from theragraph.db.models import (
    CandidateRanking,
    CandidateScore,
    Experiment,
    MLModel,
    ModelEvaluation,
    Prediction,
)
from theragraph.db.repositories.base import upsert_many


# ------------------------------------------------------------------ model registry
async def upsert_model(session: AsyncSession, **fields: Any) -> MLModel:
    returned = await upsert_many(session, MLModel, [fields], conflict_columns=["name", "version"])
    model = await session.get(MLModel, returned[0][0], populate_existing=True)
    assert model is not None
    return model


async def get_model_by_id(session: AsyncSession, model_id: str) -> MLModel | None:
    return await session.get(MLModel, model_id)


async def get_model(session: AsyncSession, name: str, version: str | None = None) -> MLModel | None:
    """Resolve a model by name, defaulting to the newest active version.

    Pinning a version is what makes a run reproducible; omitting it is what makes the
    product usable. Both paths are supported, and whichever is resolved is recorded on
    the run.
    """
    stmt = sa.select(MLModel).where(MLModel.name == name)
    if version:
        stmt = stmt.where(MLModel.version == version)
    else:
        stmt = stmt.where(MLModel.status == "active")
    stmt = stmt.order_by(MLModel.version.desc(), MLModel.created_at.desc()).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_active_model_for_endpoint(
    session: AsyncSession, endpoint: str, *, family: str | None = None
) -> MLModel | None:
    stmt = sa.select(MLModel).where(MLModel.endpoint == endpoint, MLModel.status == "active")
    if family:
        stmt = stmt.where(MLModel.family == family)
    # Prefer the graph model when several are active: it is the better performer on
    # scaffold splits, and the baseline stays available for explicit comparison.
    priority = sa.case((MLModel.family == "gin", 0), else_=1)
    stmt = stmt.order_by(priority, MLModel.version.desc()).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_models(
    session: AsyncSession,
    *,
    endpoint: str | None = None,
    family: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[MLModel]:
    stmt = sa.select(MLModel).options(selectinload(MLModel.evaluations))
    if endpoint:
        stmt = stmt.where(MLModel.endpoint == endpoint)
    if family:
        stmt = stmt.where(MLModel.family == family)
    if status:
        stmt = stmt.where(MLModel.status == status)
    stmt = stmt.order_by(MLModel.endpoint, MLModel.name, MLModel.version.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().unique())


async def get_models_by_ids(session: AsyncSession, ids: list[str]) -> list[MLModel]:
    if not ids:
        return []
    result = await session.execute(sa.select(MLModel).where(MLModel.id.in_(ids)))
    return list(result.scalars())


# -------------------------------------------------------------------- evaluations
async def upsert_evaluation(session: AsyncSession, **fields: Any) -> ModelEvaluation:
    returned = await upsert_many(
        session,
        ModelEvaluation,
        [fields],
        conflict_columns=["model_id", "split_strategy", "split_name"],
    )
    evaluation = await session.get(ModelEvaluation, returned[0][0], populate_existing=True)
    assert evaluation is not None
    return evaluation


async def list_evaluations(
    session: AsyncSession, *, model_id: str | None = None, endpoint: str | None = None
) -> list[ModelEvaluation]:
    stmt = sa.select(ModelEvaluation).options(selectinload(ModelEvaluation.model))
    if model_id:
        stmt = stmt.where(ModelEvaluation.model_id == model_id)
    if endpoint:
        stmt = stmt.join(MLModel).where(MLModel.endpoint == endpoint)
    result = await session.execute(stmt.order_by(ModelEvaluation.created_at.desc()))
    return list(result.scalars().unique())


# -------------------------------------------------------------------- experiments
async def allocate_experiment_id(session: AsyncSession, year: int) -> tuple[str, int]:
    from theragraph.db.repositories.runs import allocate_sequence

    sequence = await allocate_sequence(session, scope=f"experiment:{year}")
    from theragraph.core.ids import format_experiment_id

    return format_experiment_id(year, sequence), sequence


async def create_experiment(session: AsyncSession, **fields: Any) -> Experiment:
    experiment = Experiment(**fields)
    session.add(experiment)
    await session.flush()
    return experiment


async def get_experiment(session: AsyncSession, experiment_id: str) -> Experiment | None:
    result = await session.execute(
        sa.select(Experiment)
        .options(selectinload(Experiment.models))
        .where(Experiment.id == experiment_id)
    )
    return result.scalar_one_or_none()


async def list_experiments(
    session: AsyncSession, *, endpoint: str | None = None, limit: int = 50
) -> list[Experiment]:
    stmt = sa.select(Experiment).options(selectinload(Experiment.models))
    if endpoint:
        stmt = stmt.where(Experiment.endpoint == endpoint)
    result = await session.execute(stmt.order_by(Experiment.created_at.desc()).limit(limit))
    return list(result.scalars().unique())


# -------------------------------------------------------------------- predictions
async def upsert_predictions(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    """Persist predictions idempotently on ``(compound, model, input_hash)``."""
    returned = await upsert_many(
        session,
        Prediction,
        rows,
        conflict_columns=["compound_id", "model_id", "input_hash"],
        update_columns=[
            "value",
            "probability",
            "predicted_class",
            "uncertainty",
            "confidence",
            "applicability",
            "latency_ms",
            "research_run_id",
        ],
    )
    return [row[0] for row in returned]


async def get_predictions_for_compounds(
    session: AsyncSession,
    compound_ids: list[str],
    *,
    model_ids: list[str] | None = None,
    endpoints: list[str] | None = None,
) -> list[Prediction]:
    if not compound_ids:
        return []
    stmt = (
        sa.select(Prediction)
        .options(selectinload(Prediction.model))
        .where(Prediction.compound_id.in_(compound_ids))
    )
    if model_ids:
        stmt = stmt.where(Prediction.model_id.in_(model_ids))
    if endpoints:
        stmt = stmt.where(Prediction.endpoint.in_(endpoints))
    result = await session.execute(stmt.order_by(Prediction.created_at.desc()))
    return list(result.scalars().unique())


async def get_cached_prediction(
    session: AsyncSession, *, compound_id: str, model_id: str, input_hash: str
) -> Prediction | None:
    result = await session.execute(
        sa.select(Prediction)
        .where(
            Prediction.compound_id == compound_id,
            Prediction.model_id == model_id,
            Prediction.input_hash == input_hash,
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


async def prediction_distribution(
    session: AsyncSession, model_id: str, *, bins: int = 10
) -> dict[str, Any]:
    """Summary statistics of served predictions -- the drift signal on the dashboard."""
    column = sa.case(
        (Prediction.probability.is_not(None), Prediction.probability), else_=Prediction.value
    )
    result = await session.execute(
        sa.select(
            sa.func.count(),
            sa.func.min(column),
            sa.func.max(column),
            sa.func.avg(column),
        ).where(Prediction.model_id == model_id)
    )
    count, minimum, maximum, mean = result.one()
    if not count:
        return {"count": 0, "bins": bins}
    return {
        "count": int(count),
        "min": float(minimum) if minimum is not None else None,
        "max": float(maximum) if maximum is not None else None,
        "mean": float(mean) if mean is not None else None,
        "bins": bins,
    }


# ----------------------------------------------------------------------- rankings
async def create_ranking(
    session: AsyncSession, *, scores: list[dict[str, Any]], **fields: Any
) -> CandidateRanking:
    ranking = CandidateRanking(**fields)
    session.add(ranking)
    await session.flush()
    session.add_all([CandidateScore(ranking_id=ranking.id, **score) for score in scores])
    await session.flush()
    return ranking


async def get_ranking(session: AsyncSession, ranking_id: str) -> CandidateRanking | None:
    result = await session.execute(
        sa.select(CandidateRanking)
        .options(selectinload(CandidateRanking.scores).selectinload(CandidateScore.compound))
        .where(CandidateRanking.id == ranking_id)
    )
    return result.scalar_one_or_none()


async def list_rankings(
    session: AsyncSession, *, research_run_id: str | None = None, limit: int = 25
) -> list[CandidateRanking]:
    stmt = sa.select(CandidateRanking)
    if research_run_id:
        stmt = stmt.where(CandidateRanking.research_run_id == research_run_id)
    result = await session.execute(stmt.order_by(CandidateRanking.created_at.desc()).limit(limit))
    return list(result.scalars())


async def count_models(session: AsyncSession) -> int:
    return int(await session.scalar(sa.select(sa.func.count()).select_from(MLModel)) or 0)


async def count_predictions(session: AsyncSession) -> int:
    return int(await session.scalar(sa.select(sa.func.count()).select_from(Prediction)) or 0)


async def touch_model(session: AsyncSession, model: MLModel, **fields: Any) -> MLModel:
    for key, value in fields.items():
        setattr(model, key, value)
    model.updated_at = utcnow()
    await session.flush()
    return model
