"""Portable column types.

The production database is PostgreSQL; the fast test suite runs on SQLite. Rather than
maintaining two schemas, every non-trivial column is declared once with a
PostgreSQL-specific variant, so ``JSONB`` and native UUID/array behaviour are used in
production while SQLite still round-trips the same Python values.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

#: ``JSONB`` on PostgreSQL (indexable, binary), plain ``JSON`` elsewhere.
JSONColumn = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

#: Timezone-aware timestamps everywhere. Naive datetimes are a persistent source of
#: provenance bugs in scientific data, so the schema forbids them.
UTCDateTime = sa.DateTime(timezone=True)

#: Identifier columns: ULID strings (26 chars) or namespaced source ids.
IdColumn = sa.String(64)

#: Fixed-precision score in [0, 1] -- ``Numeric`` avoids float drift in ranking output.
ScoreColumn = sa.Numeric(precision=8, scale=6).with_variant(sa.Float(), "sqlite")


def json_default(value: Any) -> Any:
    return value
