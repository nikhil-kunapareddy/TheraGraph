"""Reproducible research runs, agent executions, tool calls and reports.

A :class:`ResearchRun` is the unit of reproducibility. It pins everything that could
change an answer -- the query, the dataset versions, the model versions, the prompts,
the ranking configuration, the random seed, the code revision -- and stores the full
execution trace so a scientist can audit or replay the workflow.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from theragraph.db.base import Base, TimestampMixin, ULIDPrimaryKey
from theragraph.db.types import JSONColumn, UTCDateTime


class RunCounter(Base):
    """Year-scoped sequence backing human-readable run ids (``RUN-2026-00183``).

    A dedicated counter row (locked with ``SELECT ... FOR UPDATE``) rather than a
    database sequence, because the counter must reset per year and must not leave gaps
    when a transaction rolls back.
    """

    __tablename__ = "run_counter"

    scope: Mapped[str] = mapped_column(sa.String(32), primary_key=True)
    """``run:2026`` or ``experiment:2026``."""

    last_value: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)


class ResearchRun(Base, TimestampMixin):
    """A complete, reproducible research workflow execution."""

    __tablename__ = "research_run"
    __table_args__ = (
        sa.Index("ix_research_run_status_created", "status", "created_at"),
        sa.Index("ix_research_run_reproducibility", "reproducibility_hash"),
        sa.UniqueConstraint("idempotency_key"),
    )

    id: Mapped[str] = mapped_column(sa.String(32), primary_key=True)
    sequence: Mapped[int] = mapped_column(sa.Integer(), nullable=False)
    year: Mapped[int] = mapped_column(sa.Integer(), nullable=False)

    query: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    workflow: Mapped[str] = mapped_column(sa.String(64), nullable=False, default="research_agent")
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="queued")
    """``queued`` | ``planning`` | ``running`` | ``validating`` | ``reporting`` |
    ``succeeded`` | ``failed`` | ``cancelled``."""

    principal: Mapped[str | None] = mapped_column(sa.String(128))
    idempotency_key: Mapped[str | None] = mapped_column(sa.String(128))

    parameters: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Caller-supplied knobs: disease/target hints, max candidates, similarity
    threshold, which endpoints to predict."""

    plan: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """The planner's structured plan: ordered steps with tool names and arguments."""

    state: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Structured intermediate state accumulated across steps (never free text)."""

    dataset_versions: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    model_versions: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    ranking_config: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    prompts: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    """Every prompt sent to the LLM, with its template id and rendered content, so a
    replay uses byte-identical instructions."""

    retrieved_record_ids: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Primary keys of every record the run retrieved, by entity type. Replay compares
    these to detect upstream drift."""

    software_version: Mapped[str | None] = mapped_column(sa.String(32))
    git_sha: Mapped[str | None] = mapped_column(sa.String(40))
    random_seed: Mapped[int | None] = mapped_column(sa.Integer())
    agent_model: Mapped[str | None] = mapped_column(sa.String(64))
    llm_backend: Mapped[str | None] = mapped_column(sa.String(32))

    reproducibility_hash: Mapped[str | None] = mapped_column(sa.String(64))
    """Hash of (query, parameters, dataset versions, model versions, ranking config,
    prompt templates, seed). Two runs with the same hash should produce the same
    scientific content."""

    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    runtime_ms: Mapped[float | None] = mapped_column(sa.Float())

    n_tool_calls: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    n_steps: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(sa.Float(), nullable=False, default=0.0)

    errors: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    warnings: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)

    parent_run_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("research_run.id", ondelete="SET NULL"), index=True
    )
    replay_of_run_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("research_run.id", ondelete="SET NULL"), index=True
    )
    job_id: Mapped[str | None] = mapped_column(sa.String(64), index=True)

    executions: Mapped[list[AgentExecution]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="AgentExecution.step_index",
    )
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="ToolCall.step_index"
    )
    report: Mapped[Report | None] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False
    )

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled"}


class AgentExecution(Base, ULIDPrimaryKey, TimestampMixin):
    """One phase of the agent workflow (plan / act / validate / report).

    Stores the *summary* of the model's work and the structured state it produced.
    Hidden chain-of-thought is never persisted or exposed; what is stored is the
    machine-readable plan, the tool calls, and a short human-readable rationale the
    model is explicitly asked to produce for the user.
    """

    __tablename__ = "agent_execution"
    __table_args__ = (sa.UniqueConstraint("research_run_id", "step_index"),)

    research_run_id: Mapped[str] = mapped_column(
        sa.ForeignKey("research_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_index: Mapped[int] = mapped_column(sa.Integer(), nullable=False)
    phase: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="running")

    model: Mapped[str | None] = mapped_column(sa.String(64))
    prompt_template_id: Mapped[str | None] = mapped_column(sa.String(64))
    rationale: Mapped[str | None] = mapped_column(sa.Text())
    """Short user-facing justification requested from the model -- a summary, not raw
    reasoning traces."""

    output_state: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    latency_ms: Mapped[float | None] = mapped_column(sa.Float())
    input_tokens: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(sa.Float(), nullable=False, default=0.0)
    error: Mapped[str | None] = mapped_column(sa.Text())

    run: Mapped[ResearchRun] = relationship(back_populates="executions")
    tool_calls: Mapped[list[ToolCall]] = relationship(back_populates="execution")


class ToolCall(Base, ULIDPrimaryKey, TimestampMixin):
    """A single structured tool invocation made by the agent.

    Arguments and results are validated against Pydantic schemas before and after
    execution, and ``record_ids`` captures exactly which database rows the tool
    returned -- the basis for citation checking in the validation phase.
    """

    __tablename__ = "tool_call"
    __table_args__ = (
        sa.Index("ix_tool_call_run_step", "research_run_id", "step_index"),
        sa.Index("ix_tool_call_tool_status", "tool_name", "status"),
    )

    research_run_id: Mapped[str] = mapped_column(
        sa.ForeignKey("research_run.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_execution_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("agent_execution.id", ondelete="SET NULL"), index=True
    )
    step_index: Mapped[int] = mapped_column(sa.Integer(), nullable=False)
    tool_name: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    tool_version: Mapped[str | None] = mapped_column(sa.String(32))
    call_id: Mapped[str | None] = mapped_column(sa.String(64))

    arguments: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONColumn)
    """Structured tool output, truncated for storage with a pointer to full results."""

    result_summary: Mapped[str | None] = mapped_column(sa.Text())
    record_ids: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    citations: Mapped[list[str]] = mapped_column(JSONColumn, default=list)

    status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="succeeded")
    error_code: Mapped[str | None] = mapped_column(sa.String(64))
    error: Mapped[str | None] = mapped_column(sa.Text())
    latency_ms: Mapped[float | None] = mapped_column(sa.Float())
    cache_hit: Mapped[bool] = mapped_column(sa.Boolean(), nullable=False, default=False)
    validation: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Output-validation verdict: schema ok, empty-result handling, provenance present."""

    run: Mapped[ResearchRun] = relationship(back_populates="tool_calls")
    execution: Mapped[AgentExecution | None] = relationship(back_populates="tool_calls")


class Report(Base, ULIDPrimaryKey, TimestampMixin):
    """An evidence-backed research report generated at the end of a run."""

    __tablename__ = "report"
    __table_args__ = (sa.UniqueConstraint("research_run_id"),)

    research_run_id: Mapped[str] = mapped_column(
        sa.ForeignKey("research_run.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    question: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    summary: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    markdown: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    sections: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)

    claims: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    """Each claim carries ``kind`` (``retrieved_fact`` | ``model_prediction`` |
    ``interpretation``) plus the citation keys backing it. Claims that fail validation
    never reach the report."""

    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    limitations: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    validation: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    storage_uri: Mapped[str | None] = mapped_column(sa.String(1024))
    content_hash: Mapped[str | None] = mapped_column(sa.String(64))

    run: Mapped[ResearchRun] = relationship(back_populates="report")
