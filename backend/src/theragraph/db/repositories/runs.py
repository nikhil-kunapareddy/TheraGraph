"""Research-run repository: id allocation, trace persistence, comparison and replay."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from theragraph.core.clock import elapsed_ms_since, utcnow
from theragraph.core.ids import format_run_id
from theragraph.db.models import (
    AgentExecution,
    Report,
    ResearchRun,
    RunCounter,
    ToolCall,
)


async def allocate_sequence(session: AsyncSession, *, scope: str) -> int:
    """Atomically allocate the next number in a named sequence.

    Implemented as an ``UPDATE ... RETURNING`` on a single counter row, which is a
    single atomic statement -- no read-modify-write race, and no gap on rollback
    because the increment shares the caller's transaction.
    """
    stmt = (
        sa.update(RunCounter)
        .where(RunCounter.scope == scope)
        .values(last_value=RunCounter.last_value + 1)
        .returning(RunCounter.last_value)
        .execution_options(synchronize_session=False)
    )
    result = await session.execute(stmt)
    row = result.first()
    if row is not None:
        return int(row[0])

    # First allocation for this scope.
    session.add(RunCounter(scope=scope, last_value=1))
    await session.flush()
    return 1


async def allocate_run_id(session: AsyncSession, year: int) -> tuple[str, int]:
    """Return ``('RUN-2026-00183', 183)``."""
    sequence = await allocate_sequence(session, scope=f"run:{year}")
    return format_run_id(year, sequence), sequence


async def create_run(session: AsyncSession, **fields: Any) -> ResearchRun:
    run = ResearchRun(**fields)
    session.add(run)
    await session.flush()
    return run


async def get_run(
    session: AsyncSession, run_id: str, *, with_trace: bool = True
) -> ResearchRun | None:
    stmt = sa.select(ResearchRun).where(ResearchRun.id == run_id)
    if with_trace:
        stmt = stmt.options(
            selectinload(ResearchRun.executions).selectinload(AgentExecution.tool_calls),
            selectinload(ResearchRun.tool_calls),
            selectinload(ResearchRun.report),
        )
    result = await session.execute(stmt.limit(1))
    return result.scalar_one_or_none()


async def get_run_by_idempotency_key(session: AsyncSession, key: str) -> ResearchRun | None:
    result = await session.execute(
        sa.select(ResearchRun).where(ResearchRun.idempotency_key == key).limit(1)
    )
    return result.scalar_one_or_none()


async def list_runs(
    session: AsyncSession,
    *,
    status: str | None = None,
    workflow: str | None = None,
    principal: str | None = None,
    query_contains: str | None = None,
    limit: int = 25,
    offset: int = 0,
) -> list[ResearchRun]:
    stmt = sa.select(ResearchRun).options(selectinload(ResearchRun.report))
    if status:
        stmt = stmt.where(ResearchRun.status == status)
    if workflow:
        stmt = stmt.where(ResearchRun.workflow == workflow)
    if principal:
        stmt = stmt.where(ResearchRun.principal == principal)
    if query_contains:
        stmt = stmt.where(ResearchRun.query.ilike(f"%{query_contains}%"))
    result = await session.execute(
        stmt.order_by(ResearchRun.created_at.desc()).limit(limit).offset(offset)
    )
    return list(result.scalars().unique())


async def count_runs(session: AsyncSession, *, status: str | None = None) -> int:
    stmt = sa.select(sa.func.count()).select_from(ResearchRun)
    if status:
        stmt = stmt.where(ResearchRun.status == status)
    return int(await session.scalar(stmt) or 0)


async def update_run(session: AsyncSession, run: ResearchRun, **fields: Any) -> ResearchRun:
    for key, value in fields.items():
        setattr(run, key, value)
    await session.flush()
    return run


async def finish_run(
    session: AsyncSession,
    run: ResearchRun,
    *,
    status: str,
    errors: list[dict[str, Any]] | None = None,
) -> ResearchRun:
    finished = utcnow()
    run.status = status
    run.finished_at = finished
    run.runtime_ms = elapsed_ms_since(run.started_at, finished)
    if errors:
        run.errors = errors
    await session.flush()
    return run


async def add_execution(session: AsyncSession, **fields: Any) -> AgentExecution:
    execution = AgentExecution(**fields)
    session.add(execution)
    await session.flush()
    return execution


async def add_tool_call(session: AsyncSession, **fields: Any) -> ToolCall:
    call = ToolCall(**fields)
    session.add(call)
    await session.flush()
    return call


async def set_report(session: AsyncSession, **fields: Any) -> Report:
    existing = await session.execute(
        sa.select(Report).where(Report.research_run_id == fields["research_run_id"]).limit(1)
    )
    report = existing.scalar_one_or_none()
    if report is None:
        report = Report(**fields)
        session.add(report)
    else:
        for key, value in fields.items():
            setattr(report, key, value)
    await session.flush()
    return report


async def get_report(session: AsyncSession, run_id: str) -> Report | None:
    result = await session.execute(
        sa.select(Report).where(Report.research_run_id == run_id).limit(1)
    )
    return result.scalar_one_or_none()


async def tool_call_stats(session: AsyncSession, *, limit_runs: int = 200) -> list[dict[str, Any]]:
    """Per-tool call counts, error rate and p50/p95-ish latency for observability."""
    recent_runs = (
        sa.select(ResearchRun.id)
        .order_by(ResearchRun.created_at.desc())
        .limit(limit_runs)
        .subquery()
    )
    result = await session.execute(
        sa.select(
            ToolCall.tool_name,
            sa.func.count().label("calls"),
            sa.func.sum(sa.case((ToolCall.status != "succeeded", 1), else_=0)).label("errors"),
            sa.func.avg(ToolCall.latency_ms).label("avg_latency_ms"),
            sa.func.max(ToolCall.latency_ms).label("max_latency_ms"),
            sa.func.sum(sa.case((ToolCall.cache_hit.is_(True), 1), else_=0)).label("cache_hits"),
        )
        .where(ToolCall.research_run_id.in_(sa.select(recent_runs.c.id)))
        .group_by(ToolCall.tool_name)
        .order_by(sa.func.count().desc())
    )
    return [
        {
            "tool_name": row[0],
            "calls": int(row[1]),
            "errors": int(row[2] or 0),
            "error_rate": round(float(row[2] or 0) / int(row[1]), 4) if row[1] else 0.0,
            "avg_latency_ms": round(float(row[3]), 2) if row[3] is not None else None,
            "max_latency_ms": round(float(row[4]), 2) if row[4] is not None else None,
            "cache_hit_rate": round(float(row[5] or 0) / int(row[1]), 4) if row[1] else 0.0,
        }
        for row in result.all()
    ]


async def run_usage_summary(session: AsyncSession, *, limit_runs: int = 200) -> dict[str, Any]:
    """Aggregate latency, token and cost figures over recent runs."""
    subquery = (
        sa.select(
            ResearchRun.runtime_ms,
            ResearchRun.input_tokens,
            ResearchRun.output_tokens,
            ResearchRun.cost_usd,
            ResearchRun.status,
        )
        .order_by(ResearchRun.created_at.desc())
        .limit(limit_runs)
        .subquery()
    )
    result = await session.execute(
        sa.select(
            sa.func.count(),
            sa.func.avg(subquery.c.runtime_ms),
            sa.func.max(subquery.c.runtime_ms),
            sa.func.sum(subquery.c.input_tokens),
            sa.func.sum(subquery.c.output_tokens),
            sa.func.sum(subquery.c.cost_usd),
            sa.func.sum(sa.case((subquery.c.status == "failed", 1), else_=0)),
        ).select_from(subquery)
    )
    count, avg_ms, max_ms, in_tokens, out_tokens, cost, failures = result.one()
    total = int(count or 0)
    return {
        "runs": total,
        "avg_runtime_ms": round(float(avg_ms), 2) if avg_ms is not None else None,
        "max_runtime_ms": round(float(max_ms), 2) if max_ms is not None else None,
        "input_tokens": int(in_tokens or 0),
        "output_tokens": int(out_tokens or 0),
        "cost_usd": round(float(cost or 0.0), 6),
        "failed_runs": int(failures or 0),
        "failure_rate": round(int(failures or 0) / total, 4) if total else 0.0,
    }
