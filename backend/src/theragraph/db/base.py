"""Declarative base, naming conventions and shared mixins."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from theragraph.core.ids import new_ulid
from theragraph.db.types import IdColumn, JSONColumn, UTCDateTime

#: Deterministic constraint names so Alembic migrations are stable and reviewable.
#:
#: PostgreSQL truncates identifiers at 63 bytes *silently*, which would leave the model
#: declaring one name and the database holding another -- and a later
#: ``op.drop_constraint`` would then fail. So the foreign-key template omits the
#: referred table (a column has at most one foreign key, so table+column is already
#: unique), and the two widest unique constraints carry explicit short names.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map = {
        dict[str, Any]: JSONColumn,
        list[str]: JSONColumn,
        list[dict[str, Any]]: JSONColumn,
        datetime: UTCDateTime,
    }

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        pk = getattr(self, "id", None)
        return f"<{type(self).__name__} id={pk!r}>"


class ULIDPrimaryKey:
    """ULID primary key, generated in Python so callers can pre-declare ids."""

    id: Mapped[str] = mapped_column(IdColumn, primary_key=True, default=new_ulid)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, server_default=sa.func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
        nullable=False,
    )


class ProvenanceMixin:
    """Every scientific record must say where it came from and when.

    ``dataset_version_id`` points at the immutable snapshot the row was ingested from,
    which is what makes a research run reproducible: re-running with the same dataset
    version must reproduce the same retrieved records.
    """

    source: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    source_identifier: Mapped[str | None] = mapped_column(sa.String(255), index=True)
    source_url: Mapped[str | None] = mapped_column(sa.String(1024))
    retrieved_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)

    @classmethod
    def provenance_columns(cls) -> tuple[str, ...]:
        return ("source", "source_identifier", "source_url", "retrieved_at")
