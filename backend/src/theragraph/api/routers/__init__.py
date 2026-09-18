"""API routers, one per resource area."""

from theragraph.api.routers import (
    agent,
    compounds,
    diseases,
    graph,
    health,
    ingestion,
    jobs,
    models,
    predictions,
    ranking,
    runs,
    search,
    targets,
)

ALL_ROUTERS = [
    health.router,
    search.router,
    diseases.router,
    targets.router,
    compounds.router,
    predictions.router,
    ranking.router,
    graph.router,
    agent.router,
    runs.router,
    models.router,
    ingestion.router,
    jobs.router,
]

__all__ = ["ALL_ROUTERS"]
