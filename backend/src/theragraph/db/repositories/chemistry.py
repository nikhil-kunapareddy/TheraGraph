"""Chemistry repository: compounds, descriptors, fingerprints, assays, bioactivity."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from theragraph.db.models import (
    Assay,
    Bioactivity,
    Compound,
    CompoundFingerprint,
    MolecularProperty,
    Target,
)
from theragraph.db.repositories.base import (
    like_pattern,
    like_prefix,
    normalize_text,
    upsert_many,
)


async def upsert_compounds(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    """Insert-or-update compounds, deduplicated on InChIKey."""
    returned = await upsert_many(session, Compound, rows, conflict_columns=["inchikey"])
    return [row[0] for row in returned]


async def get_compound(session: AsyncSession, identifier: str) -> Compound | None:
    """Resolve a compound by internal id, ChEMBL id, PubChem CID or InChIKey."""
    stmt = (
        sa.select(Compound)
        .where(
            sa.or_(
                Compound.id == identifier,
                Compound.chembl_id == identifier.upper(),
                Compound.inchikey == identifier.upper(),
                Compound.pubchem_cid == identifier,
            )
        )
        .limit(1)
        .execution_options(populate_existing=True)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_compound_with_properties(session: AsyncSession, identifier: str) -> Compound | None:
    compound = await get_compound(session, identifier)
    if compound is None:
        return None
    await session.refresh(compound, ["properties"])
    return compound


async def get_compounds_by_ids(session: AsyncSession, ids: list[str]) -> list[Compound]:
    if not ids:
        return []
    result = await session.execute(
        sa.select(Compound).options(selectinload(Compound.properties)).where(Compound.id.in_(ids))
    )
    # Preserve caller ordering (rankings care about order).
    by_id = {compound.id: compound for compound in result.scalars()}
    return [by_id[cid] for cid in ids if cid in by_id]


async def search_compounds(
    session: AsyncSession, query: str, *, limit: int = 20
) -> list[tuple[Compound, float]]:
    """Text search across names, synonyms and external identifiers."""
    normalized = normalize_text(query)
    if not normalized:
        return []
    pattern = like_pattern(query)
    name = sa.func.lower(Compound.name)
    synonyms = sa.func.lower(sa.cast(Compound.synonyms, sa.Text))
    relevance = sa.case(
        (Compound.chembl_id == query.upper(), 130.0),
        (Compound.inchikey == query.upper(), 130.0),
        (name == normalized, 120.0),
        (name.like(like_prefix(query), escape="\\"), 90.0),
        (name.like(pattern, escape="\\"), 70.0),
        (synonyms.like(pattern, escape="\\"), 50.0),
        else_=0.0,
    ).label("relevance")
    stmt = (
        sa.select(Compound, relevance)
        .where(relevance > 0)
        .order_by(relevance.desc(), Compound.max_phase.desc().nullslast(), Compound.name)
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [(row[0], float(row[1])) for row in result.all()]


async def filter_compounds(
    session: AsyncSession,
    *,
    max_phase_min: float | None = None,
    mw_range: tuple[float, float] | None = None,
    logp_range: tuple[float, float] | None = None,
    tpsa_max: float | None = None,
    passes_lipinski: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Compound]:
    """Property-window filtering, used by compound discovery and ranking pre-filters."""
    stmt = sa.select(Compound).options(selectinload(Compound.properties))
    needs_properties = any(
        value is not None for value in (mw_range, logp_range, tpsa_max, passes_lipinski)
    )
    if needs_properties:
        stmt = stmt.join(MolecularProperty, MolecularProperty.compound_id == Compound.id)
        if mw_range:
            stmt = stmt.where(MolecularProperty.molecular_weight.between(mw_range[0], mw_range[1]))
        if logp_range:
            stmt = stmt.where(MolecularProperty.logp.between(logp_range[0], logp_range[1]))
        if tpsa_max is not None:
            stmt = stmt.where(MolecularProperty.tpsa <= tpsa_max)
        if passes_lipinski is not None:
            stmt = stmt.where(MolecularProperty.passes_lipinski.is_(passes_lipinski))
    if max_phase_min is not None:
        stmt = stmt.where(Compound.max_phase >= max_phase_min)
    stmt = (
        stmt.order_by(Compound.max_phase.desc().nullslast(), Compound.id)
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return list(result.scalars().unique())


# ------------------------------------------------------------------- properties
async def upsert_properties(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(
        session,
        MolecularProperty,
        rows,
        conflict_columns=["compound_id", "provider", "descriptor_set_version"],
    )
    return [row[0] for row in returned]


async def get_properties(
    session: AsyncSession, compound_id: str, *, provider: str = "rdkit"
) -> MolecularProperty | None:
    result = await session.execute(
        sa.select(MolecularProperty)
        .where(
            MolecularProperty.compound_id == compound_id,
            MolecularProperty.provider == provider,
        )
        .order_by(MolecularProperty.descriptor_set_version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_properties_for_compounds(
    session: AsyncSession, compound_ids: list[str], *, provider: str = "rdkit"
) -> dict[str, MolecularProperty]:
    """Batch descriptor fetch -- one query for a whole candidate set."""
    if not compound_ids:
        return {}
    result = await session.execute(
        sa.select(MolecularProperty).where(
            MolecularProperty.compound_id.in_(compound_ids),
            MolecularProperty.provider == provider,
        )
    )
    return {row.compound_id: row for row in result.scalars()}


async def compounds_missing_properties(
    session: AsyncSession, *, provider: str = "rdkit", limit: int = 1000
) -> list[Compound]:
    """Compounds whose descriptors have not been computed yet (backfill job input)."""
    existing = sa.select(MolecularProperty.compound_id).where(
        MolecularProperty.provider == provider
    )
    result = await session.execute(
        sa.select(Compound).where(Compound.id.notin_(existing)).limit(limit)
    )
    return list(result.scalars())


# ----------------------------------------------------------------- fingerprints
async def upsert_fingerprints(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(
        session, CompoundFingerprint, rows, conflict_columns=["compound_id", "kind"]
    )
    return [row[0] for row in returned]


async def get_fingerprint(
    session: AsyncSession, compound_id: str, kind: str
) -> CompoundFingerprint | None:
    result = await session.execute(
        sa.select(CompoundFingerprint)
        .where(
            CompoundFingerprint.compound_id == compound_id,
            CompoundFingerprint.kind == kind,
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


async def iter_fingerprints_for_similarity(
    session: AsyncSession,
    *,
    kind: str,
    min_popcount: int,
    max_popcount: int,
    exclude_compound_id: str | None = None,
    candidate_ids: list[str] | None = None,
    limit: int = 200_000,
) -> list[tuple[str, bytes, int]]:
    """Candidate fingerprints for a Tanimoto search.

    The popcount window comes from the Tanimoto bound: for a query with ``n`` set bits
    and threshold ``t``, any molecule with popcount outside ``[t*n, n/t]`` cannot reach
    the threshold. Pushing that bound into SQL turns a full table scan into an index
    range scan on ``(kind, popcount)``.
    """
    stmt = sa.select(
        CompoundFingerprint.compound_id,
        CompoundFingerprint.bits,
        CompoundFingerprint.popcount,
    ).where(
        CompoundFingerprint.kind == kind,
        CompoundFingerprint.popcount >= min_popcount,
        CompoundFingerprint.popcount <= max_popcount,
    )
    if exclude_compound_id:
        stmt = stmt.where(CompoundFingerprint.compound_id != exclude_compound_id)
    if candidate_ids is not None:
        stmt = stmt.where(CompoundFingerprint.compound_id.in_(candidate_ids))
    result = await session.execute(stmt.limit(limit))
    return [(row[0], row[1], row[2]) for row in result.all()]


async def all_fingerprints(
    session: AsyncSession, *, kind: str, limit: int = 200_000
) -> list[tuple[str, bytes, int]]:
    """Every fingerprint of a kind -- used by substructure screening and AD checks."""
    result = await session.execute(
        sa.select(
            CompoundFingerprint.compound_id,
            CompoundFingerprint.bits,
            CompoundFingerprint.popcount,
        )
        .where(CompoundFingerprint.kind == kind)
        .limit(limit)
    )
    return [(row[0], row[1], row[2]) for row in result.all()]


async def compounds_missing_fingerprint(
    session: AsyncSession, *, kind: str, limit: int = 1000
) -> list[Compound]:
    existing = sa.select(CompoundFingerprint.compound_id).where(CompoundFingerprint.kind == kind)
    result = await session.execute(
        sa.select(Compound).where(Compound.id.notin_(existing)).limit(limit)
    )
    return list(result.scalars())


# ----------------------------------------------------------- assays/bioactivity
async def upsert_assays(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(session, Assay, rows, conflict_columns=["chembl_assay_id"])
    return [row[0] for row in returned]


async def upsert_bioactivities(session: AsyncSession, rows: list[dict[str, Any]]) -> list[str]:
    returned = await upsert_many(
        session, Bioactivity, rows, conflict_columns=["source", "source_identifier"]
    )
    return [row[0] for row in returned]


async def bioactivities_for_compound(
    session: AsyncSession, compound_id: str, *, limit: int = 100
) -> list[Bioactivity]:
    result = await session.execute(
        sa.select(Bioactivity)
        .options(selectinload(Bioactivity.assay), selectinload(Bioactivity.target))
        .where(Bioactivity.compound_id == compound_id)
        .order_by(Bioactivity.pchembl_value.desc().nullslast())
        .limit(limit)
    )
    return list(result.scalars())


async def bioactivities_for_target(
    session: AsyncSession,
    target_id: str,
    *,
    min_pchembl: float | None = None,
    activity_types: list[str] | None = None,
    exclude_flagged: bool = True,
    limit: int = 500,
) -> list[Bioactivity]:
    """Measured activities against a target.

    ``exclude_flagged`` drops rows carrying a ChEMBL data-validity comment or marked as
    potential duplicates -- the minimum hygiene needed before these numbers are shown
    next to predictions or used for training.
    """
    stmt = (
        sa.select(Bioactivity)
        .options(selectinload(Bioactivity.compound), selectinload(Bioactivity.assay))
        .where(Bioactivity.target_id == target_id)
    )
    if min_pchembl is not None:
        stmt = stmt.where(Bioactivity.pchembl_value >= min_pchembl)
    if activity_types:
        stmt = stmt.where(Bioactivity.activity_type.in_(activity_types))
    if exclude_flagged:
        stmt = stmt.where(
            Bioactivity.data_validity_comment.is_(None),
            sa.or_(
                Bioactivity.potential_duplicate.is_(None),
                Bioactivity.potential_duplicate.is_(False),
            ),
        )
    stmt = stmt.order_by(Bioactivity.pchembl_value.desc().nullslast()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars())


async def known_compounds_for_target(
    session: AsyncSession,
    target_id: str,
    *,
    min_pchembl: float = 6.0,
    limit: int = 50,
) -> list[tuple[Compound, float, int]]:
    """Compounds with measured activity against a target.

    Returns ``(compound, best_pchembl, n_measurements)`` aggregated in SQL so a target
    with 40k activities does not stream into Python.
    """
    aggregate = (
        sa.select(
            Bioactivity.compound_id.label("compound_id"),
            sa.func.max(Bioactivity.pchembl_value).label("best_pchembl"),
            sa.func.count().label("n_measurements"),
        )
        .where(
            Bioactivity.target_id == target_id,
            Bioactivity.pchembl_value.is_not(None),
            Bioactivity.data_validity_comment.is_(None),
        )
        .group_by(Bioactivity.compound_id)
        .having(sa.func.max(Bioactivity.pchembl_value) >= min_pchembl)
        .subquery()
    )
    stmt = (
        sa.select(Compound, aggregate.c.best_pchembl, aggregate.c.n_measurements)
        .join(aggregate, aggregate.c.compound_id == Compound.id)
        .options(selectinload(Compound.properties))
        .order_by(aggregate.c.best_pchembl.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [(row[0], float(row[1]), int(row[2])) for row in result.all()]


async def targets_for_compound(
    session: AsyncSession, compound_id: str, *, limit: int = 25
) -> list[tuple[Target, float, int]]:
    aggregate = (
        sa.select(
            Bioactivity.target_id.label("target_id"),
            sa.func.max(Bioactivity.pchembl_value).label("best_pchembl"),
            sa.func.count().label("n"),
        )
        .where(
            Bioactivity.compound_id == compound_id,
            Bioactivity.target_id.is_not(None),
            Bioactivity.pchembl_value.is_not(None),
        )
        .group_by(Bioactivity.target_id)
        .subquery()
    )
    result = await session.execute(
        sa.select(Target, aggregate.c.best_pchembl, aggregate.c.n)
        .join(aggregate, aggregate.c.target_id == Target.id)
        .order_by(aggregate.c.best_pchembl.desc())
        .limit(limit)
    )
    return [(row[0], float(row[1]), int(row[2])) for row in result.all()]


async def count_compounds(session: AsyncSession) -> int:
    return int(await session.scalar(sa.select(sa.func.count()).select_from(Compound)) or 0)


async def count_bioactivities(session: AsyncSession) -> int:
    return int(await session.scalar(sa.select(sa.func.count()).select_from(Bioactivity)) or 0)


async def scaffold_counts(session: AsyncSession, limit: int = 20) -> list[tuple[str, int]]:
    """Most common Bemis-Murcko scaffolds -- a novelty reference for ranking."""
    result = await session.execute(
        sa.select(Compound.murcko_scaffold_inchikey, sa.func.count())
        .where(Compound.murcko_scaffold_inchikey.is_not(None))
        .group_by(Compound.murcko_scaffold_inchikey)
        .order_by(sa.func.count().desc())
        .limit(limit)
    )
    return [(row[0], int(row[1])) for row in result.all()]


async def scaffold_frequency(session: AsyncSession, scaffold_inchikey: str) -> int:
    return int(
        await session.scalar(
            sa.select(sa.func.count())
            .select_from(Compound)
            .where(Compound.murcko_scaffold_inchikey == scaffold_inchikey)
        )
        or 0
    )
