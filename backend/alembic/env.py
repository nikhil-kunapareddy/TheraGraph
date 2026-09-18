"""Alembic environment.

Two deliberate choices:

* The URL comes from application settings, never from ``alembic.ini``. One source of
  truth for where the database is, and no credentials in version control.
* ``compare_type`` and ``compare_server_default`` are on, so autogenerate catches a
  column whose type changed -- the kind of drift that otherwise only surfaces as a
  runtime error in production.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from theragraph.config import get_settings
from theragraph.db import models  # noqa: F401  (registers every mapper)
from theragraph.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)


#: Indexes autogenerate cannot compare, and would therefore propose forever.
#:
#: * The trigram and JSONB-GIN indexes exist only in the PostgreSQL-only migration, not
#:   in the models -- SQLite has no equivalent -- so comparing the live schema against
#:   the metadata wants to drop them on every run.
#: * ``lower(symbol)`` is a functional index. Autogenerate cannot introspect the
#:   expression, so it wants to re-add one that is already there.
#:
#: Listing them here is what makes ``alembic check`` usable as a CI gate: any *other*
#: difference between the models and the migration chain is real drift and fails.
UNCOMPARABLE_INDEXES = frozenset(
    {
        "ix_disease_name_trgm",
        "ix_target_symbol_trgm",
        "ix_target_name_trgm",
        "ix_compound_name_trgm",
        "ix_publication_title_trgm",
        "ix_disease_synonyms_gin",
        "ix_target_synonyms_gin",
        "ix_gene_symbol_lower",
    }
)


def include_object(obj: object, name: str, type_: str, reflected: bool, compare_to: object) -> bool:
    """Skip objects Alembic should not manage.

    PostgreSQL extensions create their own indexes (pg_trgm); reflecting and dropping
    them would break search on the next migration.
    """
    unmanaged = name.startswith("pg_") or name in UNCOMPARABLE_INDEXES
    return not (type_ == "index" and name and unmanaged)


def render_item(type_: str, obj: object, autogen_context: object) -> str | bool:
    """Render TheraGraph's portable column types by reference, not by dialect.

    Autogenerate resolves a ``with_variant`` type against whichever dialect it was run
    on, which would bake ``JSON`` into the migration when generated on SQLite and
    ``JSONB`` when generated on PostgreSQL. Emitting the shared helper instead keeps one
    migration correct on both: the variant resolves at DDL-compile time, so PostgreSQL
    gets JSONB and SQLite gets JSON from the same file.
    """
    if type_ == "type":
        from theragraph.db.types import JSONColumn, ScoreColumn

        if obj is JSONColumn or repr(obj) == repr(JSONColumn):
            autogen_context.imports.add("from theragraph.db import types as tg_types")  # type: ignore[attr-defined]
            return "tg_types.JSONColumn"
        if obj is ScoreColumn or repr(obj) == repr(ScoreColumn):
            autogen_context.imports.add("from theragraph.db import types as tg_types")  # type: ignore[attr-defined]
            return "tg_types.ScoreColumn"
    return False


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it (``alembic upgrade head --sql``)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        render_item=render_item,
        # Wrap each migration in its own transaction so a failure rolls back cleanly.
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
