"""Biological entities: genes, targets, diseases, their associations and evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from theragraph.db.base import Base, ProvenanceMixin, TimestampMixin, ULIDPrimaryKey
from theragraph.db.types import JSONColumn, UTCDateTime


class Gene(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """A gene, keyed by Ensembl gene id."""

    __tablename__ = "gene"
    __table_args__ = (
        sa.UniqueConstraint("ensembl_id"),
        sa.Index("ix_gene_symbol_lower", sa.text("lower(symbol)")),
    )

    ensembl_id: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(sa.String(512))
    biotype: Mapped[str | None] = mapped_column(sa.String(64))
    chromosome: Mapped[str | None] = mapped_column(sa.String(16))
    genomic_start: Mapped[int | None] = mapped_column(sa.BigInteger())
    genomic_end: Mapped[int | None] = mapped_column(sa.BigInteger())
    uniprot_ids: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )

    targets: Mapped[list[Target]] = relationship(back_populates="gene")


class Target(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """A therapeutic target (Open Targets target / protein product of a gene)."""

    __tablename__ = "target"
    __table_args__ = (
        sa.UniqueConstraint("target_key"),
        sa.Index("ix_target_symbol_name", "approved_symbol", "approved_name"),
    )

    target_key: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    """Canonical external id -- Ensembl gene id for Open Targets targets."""

    approved_symbol: Mapped[str] = mapped_column(sa.String(64), nullable=False, index=True)
    approved_name: Mapped[str | None] = mapped_column(sa.String(512))
    biotype: Mapped[str | None] = mapped_column(sa.String(64))
    target_class: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    synonyms: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    uniprot_ids: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    chembl_target_ids: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    subcellular_locations: Mapped[list[str]] = mapped_column(JSONColumn, default=list)

    tractability: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Open Targets tractability buckets per modality (small molecule, antibody, ...).
    Retrieved verbatim -- never inferred."""

    safety_liabilities: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    description: Mapped[str | None] = mapped_column(sa.Text())

    gene_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("gene.id", ondelete="SET NULL"), index=True
    )
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )

    gene: Mapped[Gene | None] = relationship(back_populates="targets")
    associations: Mapped[list[TargetDiseaseAssociation]] = relationship(
        back_populates="target", cascade="all, delete-orphan"
    )

    @property
    def tractability_summary(self) -> list[str]:
        buckets = self.tractability or {}
        return sorted(
            f"{modality}: {value}" for modality, value in buckets.items() if isinstance(value, str)
        )


class Disease(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """A disease / phenotype, keyed by EFO (or MONDO / Orphanet) id."""

    __tablename__ = "disease"
    __table_args__ = (
        sa.UniqueConstraint("efo_id"),
        sa.Index("ix_disease_name", "name"),
    )

    efo_id: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    name: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text())
    synonyms: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    therapeutic_areas: Mapped[list[dict[str, Any]]] = mapped_column(JSONColumn, default=list)
    parent_ids: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )

    associations: Mapped[list[TargetDiseaseAssociation]] = relationship(
        back_populates="disease", cascade="all, delete-orphan"
    )


class TargetDiseaseAssociation(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """A target-disease association score as published by the source.

    ``overall_score`` and ``datatype_scores`` are copied from Open Targets without
    modification; TheraGraph never computes or adjusts association scores.
    """

    __tablename__ = "target_disease_association"
    __table_args__ = (
        sa.UniqueConstraint("target_id", "disease_id", "source"),
        sa.Index("ix_tda_disease_score", "disease_id", "overall_score"),
        sa.Index("ix_tda_target_score", "target_id", "overall_score"),
        sa.CheckConstraint("overall_score >= 0 AND overall_score <= 1", name="overall_score_range"),
    )

    target_id: Mapped[str] = mapped_column(
        sa.ForeignKey("target.id", ondelete="CASCADE"), nullable=False, index=True
    )
    disease_id: Mapped[str] = mapped_column(
        sa.ForeignKey("disease.id", ondelete="CASCADE"), nullable=False, index=True
    )
    overall_score: Mapped[float] = mapped_column(sa.Float(), nullable=False)
    datatype_scores: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    evidence_count: Mapped[int] = mapped_column(sa.Integer(), nullable=False, default=0)
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )

    target: Mapped[Target] = relationship(back_populates="associations")
    disease: Mapped[Disease] = relationship(back_populates="associations")


class Publication(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """Literature *metadata* only (title/journal/year/abstract snippet)."""

    __tablename__ = "publication"
    __table_args__ = (sa.UniqueConstraint("external_id", "source"),)

    external_id: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    pmid: Mapped[str | None] = mapped_column(sa.String(32), index=True)
    doi: Mapped[str | None] = mapped_column(sa.String(255), index=True)
    title: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    journal: Mapped[str | None] = mapped_column(sa.String(512))
    year: Mapped[int | None] = mapped_column(sa.Integer(), index=True)
    authors: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    abstract_snippet: Mapped[str | None] = mapped_column(sa.Text())
    citation_count: Mapped[int | None] = mapped_column(sa.Integer())
    is_open_access: Mapped[bool | None] = mapped_column(sa.Boolean())
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )


class Evidence(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """A single piece of retrieved scientific evidence linking two entities.

    This table is the backbone of the evidence graph and of report citations. Rows are
    *only* ever created by ingestion from a named source: there is no code path that
    lets the language model write here.
    """

    __tablename__ = "evidence"
    __table_args__ = (
        # Named explicitly: six columns would generate ~79 bytes, over PostgreSQL's
        # 63-byte identifier limit, and be silently truncated.
        sa.UniqueConstraint(
            "source",
            "source_identifier",
            "evidence_type",
            "target_id",
            "disease_id",
            "compound_id",
            name="evidence_natural_key",
        ),
        sa.Index("ix_evidence_target_disease", "target_id", "disease_id", "score"),
        sa.Index("ix_evidence_type", "evidence_type", "source"),
        sa.Index("ix_evidence_compound", "compound_id", "evidence_type"),
    )

    evidence_type: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    """``genetic_association``, ``known_drug``, ``literature``, ``animal_model``,
    ``expression``, ``pathway``, ``bioactivity``, ``clinical_trial``."""

    relation: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    """``disease_target``, ``target_compound``, ``disease_compound``,
    ``compound_assay``."""

    target_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("target.id", ondelete="CASCADE"), index=True
    )
    disease_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("disease.id", ondelete="CASCADE"), index=True
    )
    compound_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("compound.id", ondelete="CASCADE"), index=True
    )
    publication_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("publication.id", ondelete="SET NULL"), index=True
    )

    score: Mapped[float | None] = mapped_column(sa.Float())
    """Confidence as provided by the source. ``NULL`` when the source gives none --
    the UI shows "not provided" rather than inventing a number."""

    score_type: Mapped[str | None] = mapped_column(sa.String(64))
    title: Mapped[str | None] = mapped_column(sa.Text())
    summary: Mapped[str | None] = mapped_column(sa.Text())
    """Verbatim text from the source (never model-generated)."""

    extra: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )

    target: Mapped[Target | None] = relationship()
    disease: Mapped[Disease | None] = relationship()
    publication: Mapped[Publication | None] = relationship()

    @property
    def citation_key(self) -> str:
        """Stable handle the agent must use when citing this row (``open_targets:EFO_0000305``)."""
        return f"{self.source}:{self.source_identifier or self.id}"


class ClinicalTrial(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """Trial metadata attached to known-drug evidence (phase, status, ids)."""

    __tablename__ = "clinical_trial"
    __table_args__ = (sa.UniqueConstraint("nct_id"),)

    nct_id: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    title: Mapped[str | None] = mapped_column(sa.Text())
    phase: Mapped[str | None] = mapped_column(sa.String(32), index=True)
    status: Mapped[str | None] = mapped_column(sa.String(64), index=True)
    start_date: Mapped[datetime | None] = mapped_column(UTCDateTime)
    disease_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("disease.id", ondelete="SET NULL"), index=True
    )
    compound_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("compound.id", ondelete="SET NULL"), index=True
    )
    target_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("target.id", ondelete="SET NULL"), index=True
    )
    urls: Mapped[list[str]] = mapped_column(JSONColumn, default=list)
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )
