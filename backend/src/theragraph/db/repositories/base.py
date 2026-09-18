"""Shared repository helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar, cast

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


def _insert_for(session: AsyncSession) -> Any:
    """Return the dialect-specific ``INSERT`` supporting ``ON CONFLICT``.

    PostgreSQL and SQLite both implement upsert, with the same SQLAlchemy API but
    different constructs. Selecting it at runtime keeps one code path for production
    and tests.
    """
    dialect = session.bind.dialect.name if session.bind is not None else "postgresql"
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return sqlite_insert
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    return pg_insert


async def upsert_many(
    session: AsyncSession,
    model: type[ModelT],
    rows: Sequence[dict[str, Any]],
    *,
    conflict_columns: Sequence[str],
    update_columns: Sequence[str] | None = None,
    returning: Sequence[str] = ("id",),
    chunk_size: int = 500,
) -> list[Any]:
    """Insert-or-update ``rows``, returning the requested columns.

    Chunked so a large ingestion batch does not build a multi-megabyte statement, and
    Python-side column defaults (ULID ids, ``created_at``) are applied per row.
    """
    if not rows:
        return []

    insert = _insert_for(session)
    # ``__table__`` is typed FromClause; on a mapped class it is always a Table.
    table = cast(sa.Table, model.__table__)
    updatable = (
        list(update_columns)
        if update_columns is not None
        else [
            column.name
            for column in table.columns
            if column.name not in set(conflict_columns) | {"id", "created_at"}
        ]
    )

    results: list[Any] = []
    for start in range(0, len(rows), chunk_size):
        chunk = list(rows[start : start + chunk_size])
        chunk = _apply_python_defaults(table, chunk)
        stmt = insert(table).values(chunk)
        if updatable:
            excluded = {name: getattr(stmt.excluded, name) for name in updatable}
            if "updated_at" in table.columns:
                excluded["updated_at"] = sa.func.now()
            stmt = stmt.on_conflict_do_update(index_elements=list(conflict_columns), set_=excluded)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=list(conflict_columns))
        stmt = stmt.returning(*(table.c[name] for name in returning))
        result = await session.execute(stmt)
        results.extend(result.fetchall())
    return results


def _apply_python_defaults(table: sa.Table, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill in callable/scalar column defaults so every row has identical keys.

    ``INSERT ... VALUES`` with heterogeneous dicts raises in SQLAlchemy; and for
    multi-row inserts the ORM does not apply Python defaults automatically.
    """
    defaulted: list[dict[str, Any]] = []
    columns_with_defaults = [
        column
        for column in table.columns
        if column.default is not None and not column.default.is_sequence
    ]
    keys = {key for row in rows for key in row}
    for row in rows:
        new_row = dict(row)
        for column in columns_with_defaults:
            if column.name not in new_row:
                default = column.default
                arg = default.arg  # type: ignore[union-attr]
                new_row[column.name] = arg({}) if callable(arg) else arg
        for key in keys:
            new_row.setdefault(key, None)
        defaulted.append(new_row)
    return defaulted


def normalize_text(value: str) -> str:
    """Lowercase + collapse whitespace, for case-insensitive lookups."""
    return " ".join(value.lower().split())


def escape_like(query: str) -> str:
    """Escape LIKE metacharacters in user input.

    Without this a query of ``%`` matches every row, and ``_`` silently becomes a
    single-character wildcard. Both the ``%term%`` and the ``term%`` predicates need it.
    """
    return normalize_text(query).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def like_pattern(query: str) -> str:
    """``%escaped-term%`` for a substring match."""
    return f"%{escape_like(query)}%"


def like_prefix(query: str) -> str:
    """``escaped-term%`` for a prefix match."""
    return f"{escape_like(query)}%"


#: ``INSERT ... ON CONFLICT DO UPDATE`` and ``UPDATE`` are Core statements: they bypass
#: the ORM identity map, so an instance loaded earlier in the same session keeps its
#: pre-write values. Reads that can follow a write in one session pass this so the row
#: is re-populated from the database. (Expiring the whole session instead would make the
#: next attribute access perform IO, which raises ``MissingGreenlet`` in async code.)
REFRESH = {"populate_existing": True}


async def scalar_count(session: AsyncSession, stmt: sa.Select[Any]) -> int:
    """``COUNT(*)`` over an existing select, without loading rows."""
    subquery = stmt.order_by(None).limit(None).offset(None).subquery()
    total = await session.scalar(sa.select(sa.func.count()).select_from(subquery))
    return int(total or 0)
