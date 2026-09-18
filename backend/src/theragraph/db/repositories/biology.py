"""Biology repository: disease/target search, associations and evidence."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from theragraph.db.models import (
    Disease,
    Evidence,
    Gene,
    Publication,
    Target,
    TargetDiseaseAssociation,
)
from theragraph.db.repositories.base import (
    like_pattern,
    like_prefix,
    normalize_text,
    upsert_many,
)

#: Relevance tiers for text search. Deliberately explicit rather than a black-box
#: ranking function: a scientist searching "alzheimer" must get *Alzheimer disease*
#: first, and must be able to understand why.
_EXACT, _PREFIX, _CONTAINS, _SYNONYM = 100.0, 80.0, 60.0, 40.0


def _relevance(
    name_column: sa.ColumnExpressionArgument[str],
    synonyms_column: sa.ColumnExpressionArgument[Any],
    query: str,
) -> sa.Case[float]:
    normalized = normalize_text(query)
    pattern = like_pattern(query)
    lowered = sa.func.lower(name_column)
    synonyms_text = sa.func.lower(sa.cast(synonyms_column, sa.Text))
    return sa.case(
        (lowered == normalized, _EXACT),
        (lowered.like(like_prefix(query), escape="\\"), _PREFIX),
        (lowered.like(pattern, escape="\\"), _CONTAINS),
        (synonyms_text.like(pattern, escape="\\"), _SYNONYM),
        else_=0.0,
    )


# --------------------------------------------------------------------- diseases
async def search_diseases(
    session: AsyncSession, query: str, *, limit: int = 20
) -> list[tuple[Disease, float]]:
    """Rank diseases by name/synonym match, most specific first."""
    normalized = normalize_text(query)
    if not normalized:
        return []
    relevance = _relevance(Disease.name, Disease.synonyms, query).label("relevance")
    stmt = (
        sa.select(Disease, relevance)
        .where(
            sa.or_(
                relevance > 0,
                sa.func.lower(Disease.efo_id) == normalized,
            )
        )
        .order_by(relevance.desc(), sa.func.length(Disease.name), Disease.name)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [(row[0], float(row[1])) for row in result.all()]


async def get_disease(session: AsyncSession, identifier: str) -> Disease | None:
    """Look up a disease by internal id or EFO id (callers use either)."""
    stmt = sa.select(Disease).where(sa.or_(Disease.id == identifier, Disease.efo_id == identifier))
    result = await session.execute(stmt.limit(1).execution_options(populate_existing=True))
    return result.scalar_one_or_none()


async def upsert_diseases(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(session, Disease, rows, conflict_columns=["efo_id"])
    return [row[0] for row in returned]


# ---------------------------------------------------------------------- targets
async def search_targets(
    session: AsyncSession, query: str, *, limit: int = 20
) -> list[tuple[Target, float]]:
    normalized = normalize_text(query)
    if not normalized:
        return []
    pattern = like_pattern(query)
    symbol = sa.func.lower(Target.approved_symbol)
    name = sa.func.lower(Target.approved_name)
    synonyms = sa.func.lower(sa.cast(Target.synonyms, sa.Text))
    # One ordered CASE rather than GREATEST(): SQLite has no GREATEST and PostgreSQL's
    # max() is an aggregate, so a single expression keeps both dialects on one query.
    relevance = sa.case(
        (symbol == normalized, _EXACT + 20.0),
        (name == normalized, _EXACT),
        (symbol.like(like_prefix(query), escape="\\"), _PREFIX + 15.0),
        (name.like(like_prefix(query), escape="\\"), _PREFIX),
        (name.like(pattern, escape="\\"), _CONTAINS),
        (synonyms.like(pattern, escape="\\"), _SYNONYM),
        else_=0.0,
    ).label("relevance")
    stmt = (
        sa.select(Target, relevance)
        .where(
            sa.or_(
                relevance > 0,
                Target.target_key == query,
                sa.func.lower(sa.cast(Target.uniprot_ids, sa.Text)).like(pattern, escape="\\"),
            )
        )
        .order_by(relevance.desc(), Target.approved_symbol)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [(row[0], float(row[1])) for row in result.all()]


async def get_target(session: AsyncSession, identifier: str) -> Target | None:
    stmt = (
        sa.select(Target)
        .where(
            sa.or_(
                Target.id == identifier,
                Target.target_key == identifier,
                sa.func.lower(Target.approved_symbol) == normalize_text(identifier),
            )
        )
        .options(selectinload(Target.gene))
        .limit(1)
        .execution_options(populate_existing=True)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_targets_by_ids(session: AsyncSession, ids: list[str]) -> list[Target]:
    if not ids:
        return []
    result = await session.execute(sa.select(Target).where(Target.id.in_(ids)))
    return list(result.scalars())


async def upsert_targets(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(session, Target, rows, conflict_columns=["target_key"])
    return [row[0] for row in returned]


async def upsert_genes(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(session, Gene, rows, conflict_columns=["ensembl_id"])
    return [row[0] for row in returned]


# ----------------------------------------------------------------- associations
async def upsert_associations(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(
        session,
        TargetDiseaseAssociation,
        rows,
        conflict_columns=["target_id", "disease_id", "source"],
    )
    return [row[0] for row in returned]


async def targets_for_disease(
    session: AsyncSession,
    disease_id: str,
    *,
    min_score: float = 0.0,
    limit: int = 25,
    offset: int = 0,
) -> list[tuple[Target, TargetDiseaseAssociation]]:
    """Associated targets ordered by the source's own association score."""
    stmt = (
        sa.select(Target, TargetDiseaseAssociation)
        .join(TargetDiseaseAssociation, TargetDiseaseAssociation.target_id == Target.id)
        .where(
            TargetDiseaseAssociation.disease_id == disease_id,
            TargetDiseaseAssociation.overall_score >= min_score,
        )
        .order_by(TargetDiseaseAssociation.overall_score.desc(), Target.approved_symbol)
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return [(row[0], row[1]) for row in result.all()]


async def diseases_for_target(
    session: AsyncSession, target_id: str, *, min_score: float = 0.0, limit: int = 25
) -> list[tuple[Disease, TargetDiseaseAssociation]]:
    stmt = (
        sa.select(Disease, TargetDiseaseAssociation)
        .join(TargetDiseaseAssociation, TargetDiseaseAssociation.disease_id == Disease.id)
        .where(
            TargetDiseaseAssociation.target_id == target_id,
            TargetDiseaseAssociation.overall_score >= min_score,
        )
        .order_by(TargetDiseaseAssociation.overall_score.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [(row[0], row[1]) for row in result.all()]


async def get_association(
    session: AsyncSession, target_id: str, disease_id: str
) -> TargetDiseaseAssociation | None:
    result = await session.execute(
        sa.select(TargetDiseaseAssociation)
        .where(
            TargetDiseaseAssociation.target_id == target_id,
            TargetDiseaseAssociation.disease_id == disease_id,
        )
        .order_by(TargetDiseaseAssociation.overall_score.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# --------------------------------------------------------------------- evidence
async def upsert_evidence(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(
        session,
        Evidence,
        rows,
        conflict_columns=[
            "source",
            "source_identifier",
            "evidence_type",
            "target_id",
            "disease_id",
            "compound_id",
        ],
    )
    return [row[0] for row in returned]


async def list_evidence(
    session: AsyncSession,
    *,
    target_id: str | None = None,
    disease_id: str | None = None,
    compound_id: str | None = None,
    evidence_types: list[str] | None = None,
    sources: list[str] | None = None,
    min_score: float | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Evidence]:
    """Retrieve evidence rows with eager-loaded publication metadata.

    ``selectinload`` avoids the N+1 that would otherwise fire once per citation when
    the report generator renders references.
    """
    stmt = sa.select(Evidence).options(selectinload(Evidence.publication))
    if target_id:
        stmt = stmt.where(Evidence.target_id == target_id)
    if disease_id:
        stmt = stmt.where(Evidence.disease_id == disease_id)
    if compound_id:
        stmt = stmt.where(Evidence.compound_id == compound_id)
    if evidence_types:
        stmt = stmt.where(Evidence.evidence_type.in_(evidence_types))
    if sources:
        stmt = stmt.where(Evidence.source.in_(sources))
    if min_score is not None:
        stmt = stmt.where(Evidence.score >= min_score)
    stmt = (
        stmt.order_by(Evidence.score.desc().nullslast(), Evidence.retrieved_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return list(result.scalars())


async def get_evidence_by_ids(session: AsyncSession, ids: list[str]) -> list[Evidence]:
    if not ids:
        return []
    result = await session.execute(
        sa.select(Evidence).options(selectinload(Evidence.publication)).where(Evidence.id.in_(ids))
    )
    return list(result.scalars())


async def evidence_type_counts(
    session: AsyncSession, *, target_id: str | None = None, disease_id: str | None = None
) -> dict[str, int]:
    """Counts per evidence type -- drives the evidence breakdown chart."""
    stmt = sa.select(Evidence.evidence_type, sa.func.count()).group_by(Evidence.evidence_type)
    if target_id:
        stmt = stmt.where(Evidence.target_id == target_id)
    if disease_id:
        stmt = stmt.where(Evidence.disease_id == disease_id)
    result = await session.execute(stmt)
    return {row[0]: int(row[1]) for row in result.all()}


async def upsert_publications(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(
        session, Publication, rows, conflict_columns=["external_id", "source"]
    )
    return [row[0] for row in returned]


async def get_publication_by_external_id(
    session: AsyncSession, external_id: str, source: str
) -> Publication | None:
    result = await session.execute(
        sa.select(Publication)
        .where(Publication.external_id == external_id, Publication.source == source)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def count_diseases(session: AsyncSession) -> int:
    return int(await session.scalar(sa.select(sa.func.count()).select_from(Disease)) or 0)


async def count_targets(session: AsyncSession) -> int:
    return int(await session.scalar(sa.select(sa.func.count()).select_from(Target)) or 0)
