"""Provenance entities: where data came from, which snapshot, and how healthy it was."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from theragraph.db.base import Base, TimestampMixin, ULIDPrimaryKey
from theragraph.db.types import IdColumn, JSONColumn, UTCDateTime

if TYPE_CHECKING:
    pass


class DataSource(Base, TimestampMixin):
    """A public biomedical data source (Open Targets, ChEMBL, PubChem, ...).

    Licensing and citation text live here so the UI and generated reports can attribute
    every retrieved record correctly.
    """

    __tablename__ = "data_source"

    id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text())
    homepage_url: Mapped[str | None] = mapped_column(sa.String(1024))
    api_base_url: Mapped[str | None] = mapped_column(sa.String(1024))
    license: Mapped[str | None] = mapped_column(sa.String(255))
    citation: Mapped[str | None] = mapped_column(sa.Text())
    entity_types: Mapped[list[str]] = mapped_column(JSONColumn, default=list)

    dataset_versions: Mapped[list[DatasetVersion]] = relationship(
        back_populates="data_source", cascade="all, delete-orphan"
    )


class DatasetVersion(Base, ULIDPrimaryKey, TimestampMixin):
    """An immutable snapshot of one dataset pulled from one source.

    ``content_hash`` is an order-independent hash of the normalised records, so two
    ingestions of identical upstream data produce the same hash and a research run can
    prove it saw the same inputs.
    """

    __tablename__ = "dataset_version"
    __table_args__ = (
        sa.UniqueConstraint("source_id", "dataset_name", "version"),
        sa.Index("ix_dataset_version_lookup", "source_id", "dataset_name", "is_current"),
        sa.CheckConstraint("record_count >= 0", name="record_count_non_negative"),
    )

    source_id: Mapped[str] = mapped_column(
        sa.ForeignKey("data_source.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_name: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    version: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    """Upstream release label when the source publishes one (Open Targets ``24.06``,
    ChEMBL ``34``), otherwise a date-based label assigned at retrieval."""

    content_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    record_count: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    retrieved_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)
    source_url: Mapped[str | None] = mapped_column(sa.String(1024))
    retrieval_params: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    quality: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Data-quality report: completeness, duplicate/reject counts, field coverage."""

    schema_version: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="1")
    is_current: Mapped[bool] = mapped_column(sa.Boolean(), nullable=False, default=True)
    notes: Mapped[str | None] = mapped_column(sa.Text())

    data_source: Mapped[DataSource] = relationship(back_populates="dataset_versions")

    @property
    def label(self) -> str:
        """``open_targets/associations@24.06`` -- what gets quoted in reports."""
        return f"{self.source_id}/{self.dataset_name}@{self.version}"


class IngestionRun(Base, ULIDPrimaryKey, TimestampMixin):
    """One execution of an ingestion pipeline, successful or not."""

    __tablename__ = "ingestion_run"
    __table_args__ = (
        sa.Index("ix_ingestion_run_status", "source_id", "status", "started_at"),
        sa.UniqueConstraint("idempotency_key"),
    )

    source_id: Mapped[str] = mapped_column(
        sa.ForeignKey("data_source.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_name: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="running")
    mode: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="live")
    """``live`` (network), ``fixture`` (replayed snapshot) or ``seed``."""

    idempotency_key: Mapped[str | None] = mapped_column(sa.String(128))
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    runtime_ms: Mapped[float | None] = mapped_column(sa.Float())
    records_fetched: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    records_written: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    records_rejected: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    rejects: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    params: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    error: Mapped[str | None] = mapped_column(sa.Text())

    dataset_version: Mapped[DatasetVersion | None] = relationship()


class ProvenanceRecord(Base, ULIDPrimaryKey, TimestampMixin):
    """Raw payload archive for spot-checking a normalisation decision.

    Stores the upstream response (or its object-storage URI when large) keyed by entity,
    so a scientist can always answer "what exactly did the API return for CHEMBL25?".
    """

    __tablename__ = "provenance_record"
    __table_args__ = (sa.Index("ix_provenance_record_entity", "entity_type", "entity_id"),)

    dataset_version_id: Mapped[str] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity_type: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(IdColumn, nullable=False)
    source_identifier: Mapped[str | None] = mapped_column(sa.String(255), index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONColumn)
    payload_uri: Mapped[str | None] = mapped_column(sa.String(1024))
    payload_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
