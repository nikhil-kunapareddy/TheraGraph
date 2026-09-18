"""postgres trigram search indexes

Adds PostgreSQL-only trigram indexes so fuzzy name search stays fast as the corpus
grows. Name search uses ``LIKE '%term%'``, which cannot use a B-tree index; a GIN
trigram index makes it an index scan.

The whole migration is a no-op on any other dialect, which is what lets the SQLite test
suite run the same migration chain as production.

Revision ID: b2c4d6e8f012
Revises: 769f8f171cd5
Create Date: 2026-09-17 21:05:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2c4d6e8f012"
down_revision: str | None = "769f8f171cd5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: (index name, table, expression) for each searched text column.
TRIGRAM_INDEXES = [
    ("ix_disease_name_trgm", "disease", "name"),
    ("ix_target_symbol_trgm", "target", "approved_symbol"),
    ("ix_target_name_trgm", "target", "approved_name"),
    ("ix_compound_name_trgm", "compound", "name"),
    ("ix_publication_title_trgm", "publication", "title"),
]


def _is_postgres() -> bool:
    bind = op.get_bind()
    return bind.dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    for name, table, column in TRIGRAM_INDEXES:
        op.execute(
            sa.text(
                f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" '
                f'USING gin ("{column}" gin_trgm_ops)'
            )
        )

    # JSONB containment queries on therapeutic areas and synonyms benefit from GIN too.
    op.execute(
        'CREATE INDEX IF NOT EXISTS "ix_disease_synonyms_gin" ON "disease" '
        'USING gin ("synonyms" jsonb_path_ops)'
    )
    op.execute(
        'CREATE INDEX IF NOT EXISTS "ix_target_synonyms_gin" ON "target" '
        'USING gin ("synonyms" jsonb_path_ops)'
    )


def downgrade() -> None:
    if not _is_postgres():
        return

    for name, _table, _column in TRIGRAM_INDEXES:
        op.execute(f'DROP INDEX IF EXISTS "{name}"')
    op.execute('DROP INDEX IF EXISTS "ix_disease_synonyms_gin"')
    op.execute('DROP INDEX IF EXISTS "ix_target_synonyms_gin"')
    # The extension is left in place: other objects may depend on it.
