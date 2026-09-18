"""Chemical entities: compounds, computed properties, fingerprints, assays, bioactivity."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from theragraph.db.base import Base, ProvenanceMixin, TimestampMixin, ULIDPrimaryKey
from theragraph.db.types import JSONColumn


class Compound(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """A small molecule.

    ``inchikey`` is the deduplication key: the same molecule arriving from ChEMBL and
    PubChem must land on one row. ``canonical_smiles`` is RDKit-canonicalised at
    ingestion so downstream featurisation is deterministic.
    """

    __tablename__ = "compound"
    __table_args__ = (
        sa.UniqueConstraint("inchikey"),
        sa.Index("ix_compound_chembl_id", "chembl_id"),
        sa.Index("ix_compound_pubchem_cid", "pubchem_cid"),
        sa.Index("ix_compound_phase", "max_phase"),
        sa.Index("ix_compound_scaffold", "murcko_scaffold_inchikey"),
    )

    inchikey: Mapped[str] = mapped_column(sa.String(27), nullable=False)
    canonical_smiles: Mapped[str] = mapped_column(sa.Text(), nullable=False)
    inchi: Mapped[str | None] = mapped_column(sa.Text())
    name: Mapped[str | None] = mapped_column(sa.String(512), index=True)
    synonyms: Mapped[list[str]] = mapped_column(JSONColumn, default=list)

    chembl_id: Mapped[str | None] = mapped_column(sa.String(32))
    pubchem_cid: Mapped[str | None] = mapped_column(sa.String(32))
    drugbank_id: Mapped[str | None] = mapped_column(sa.String(32))

    max_phase: Mapped[float | None] = mapped_column(sa.Float())
    """Highest clinical phase reached per ChEMBL (0-4). Retrieved, not predicted."""

    molecule_type: Mapped[str | None] = mapped_column(sa.String(64))
    first_approval: Mapped[int | None] = mapped_column(sa.Integer())
    is_approved_drug: Mapped[bool] = mapped_column(sa.Boolean(), nullable=False, default=False)
    withdrawn_flag: Mapped[bool] = mapped_column(sa.Boolean(), nullable=False, default=False)
    oral: Mapped[bool | None] = mapped_column(sa.Boolean())
    parenteral: Mapped[bool | None] = mapped_column(sa.Boolean())

    murcko_scaffold: Mapped[str | None] = mapped_column(sa.Text())
    murcko_scaffold_inchikey: Mapped[str | None] = mapped_column(sa.String(27))
    heavy_atom_count: Mapped[int | None] = mapped_column(sa.Integer())

    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )

    properties: Mapped[list[MolecularProperty]] = relationship(
        back_populates="compound", cascade="all, delete-orphan"
    )
    fingerprints: Mapped[list[CompoundFingerprint]] = relationship(
        back_populates="compound", cascade="all, delete-orphan"
    )
    bioactivities: Mapped[list[Bioactivity]] = relationship(
        back_populates="compound", cascade="all, delete-orphan"
    )

    @property
    def display_name(self) -> str:
        return self.name or self.chembl_id or self.inchikey


class MolecularProperty(Base, ULIDPrimaryKey, TimestampMixin):
    """Computed physicochemical descriptors for a compound.

    Deliberately separate from :class:`Compound`: descriptors are *derived* data with
    their own provenance (which RDKit version computed them, or which source reported
    them) and get recomputed when the descriptor pipeline changes.
    """

    __tablename__ = "molecular_property"
    __table_args__ = (
        # Named explicitly: the convention would generate 65 bytes, over PostgreSQL's
        # 63-byte identifier limit, and be silently truncated.
        sa.UniqueConstraint(
            "compound_id", "provider", "descriptor_set_version", name="compound_provider_version"
        ),
    )

    compound_id: Mapped[str] = mapped_column(
        sa.ForeignKey("compound.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(sa.String(64), nullable=False, default="rdkit")
    """``rdkit`` (computed here) or a source name when the value was reported upstream."""

    descriptor_set_version: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    rdkit_version: Mapped[str | None] = mapped_column(sa.String(32))

    # The descriptors scientists actually filter on get first-class columns so they are
    # indexable and queryable; the full descriptor vector lives in ``descriptors``.
    molecular_weight: Mapped[float | None] = mapped_column(sa.Float(), index=True)
    exact_mass: Mapped[float | None] = mapped_column(sa.Float())
    logp: Mapped[float | None] = mapped_column(sa.Float(), index=True)
    tpsa: Mapped[float | None] = mapped_column(sa.Float(), index=True)
    hbd: Mapped[int | None] = mapped_column(sa.Integer())
    hba: Mapped[int | None] = mapped_column(sa.Integer())
    rotatable_bonds: Mapped[int | None] = mapped_column(sa.Integer())
    aromatic_rings: Mapped[int | None] = mapped_column(sa.Integer())
    ring_count: Mapped[int | None] = mapped_column(sa.Integer())
    heavy_atom_count: Mapped[int | None] = mapped_column(sa.Integer())
    fraction_csp3: Mapped[float | None] = mapped_column(sa.Float())
    formal_charge: Mapped[int | None] = mapped_column(sa.Integer())
    stereocentres: Mapped[int | None] = mapped_column(sa.Integer())
    qed: Mapped[float | None] = mapped_column(sa.Float(), index=True)
    molar_refractivity: Mapped[float | None] = mapped_column(sa.Float())
    heteroatom_count: Mapped[int | None] = mapped_column(sa.Integer())

    lipinski_violations: Mapped[int | None] = mapped_column(sa.Integer())
    passes_lipinski: Mapped[bool | None] = mapped_column(sa.Boolean())
    passes_veber: Mapped[bool | None] = mapped_column(sa.Boolean())
    cns_mpo_score: Mapped[float | None] = mapped_column(sa.Float())

    descriptors: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict)
    """Full RDKit descriptor vector, keyed by descriptor name."""

    compound: Mapped[Compound] = relationship(back_populates="properties")


class CompoundFingerprint(Base, ULIDPrimaryKey, TimestampMixin):
    """Packed binary molecular fingerprint used for similarity/substructure search.

    ``popcount`` is stored alongside the bits because Tanimoto similarity has a hard
    bound based on set sizes: for threshold *t*, only molecules whose popcount lies in
    ``[t*|A|, |A|/t]`` can possibly qualify. Indexing it turns a full scan into a narrow
    range scan (see ``chem.similarity``).
    """

    __tablename__ = "compound_fingerprint"
    __table_args__ = (
        sa.UniqueConstraint("compound_id", "kind"),
        sa.Index("ix_fingerprint_kind_popcount", "kind", "popcount"),
    )

    compound_id: Mapped[str] = mapped_column(
        sa.ForeignKey("compound.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    """``morgan_r2_2048``, ``maccs_166``, ``rdkit_pattern_2048``."""

    bits: Mapped[bytes] = mapped_column(sa.LargeBinary(), nullable=False)
    n_bits: Mapped[int] = mapped_column(sa.Integer(), nullable=False)
    radius: Mapped[int | None] = mapped_column(sa.Integer())
    popcount: Mapped[int] = mapped_column(sa.Integer(), nullable=False)

    compound: Mapped[Compound] = relationship(back_populates="fingerprints")


class Assay(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """A biological assay (ChEMBL assay record)."""

    __tablename__ = "assay"
    __table_args__ = (
        sa.UniqueConstraint("chembl_assay_id"),
        sa.Index("ix_assay_target_type", "target_id", "assay_type"),
    )

    chembl_assay_id: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text())
    assay_type: Mapped[str | None] = mapped_column(sa.String(32))
    assay_category: Mapped[str | None] = mapped_column(sa.String(64))
    organism: Mapped[str | None] = mapped_column(sa.String(255))
    tissue: Mapped[str | None] = mapped_column(sa.String(255))
    cell_type: Mapped[str | None] = mapped_column(sa.String(255))
    confidence_score: Mapped[int | None] = mapped_column(sa.Integer())
    """ChEMBL assay confidence (0-9): how well the assay maps to a molecular target."""

    bao_format: Mapped[str | None] = mapped_column(sa.String(64))
    bao_label: Mapped[str | None] = mapped_column(sa.String(255))
    document_chembl_id: Mapped[str | None] = mapped_column(sa.String(32))
    target_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("target.id", ondelete="SET NULL"), index=True
    )
    chembl_target_id: Mapped[str | None] = mapped_column(sa.String(32), index=True)
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )

    bioactivities: Mapped[list[Bioactivity]] = relationship(back_populates="assay")


class Bioactivity(Base, ULIDPrimaryKey, TimestampMixin, ProvenanceMixin):
    """A measured compound-target activity (ChEMBL activity record).

    This is *experimental* data. It is never written by a model and is always displayed
    separately from predictions in the UI and in reports.
    """

    __tablename__ = "bioactivity"
    __table_args__ = (
        sa.UniqueConstraint("source", "source_identifier"),
        sa.Index("ix_bioactivity_compound_target", "compound_id", "target_id"),
        sa.Index("ix_bioactivity_target_pchembl", "target_id", "pchembl_value"),
        sa.Index("ix_bioactivity_type", "activity_type"),
    )

    compound_id: Mapped[str] = mapped_column(
        sa.ForeignKey("compound.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("target.id", ondelete="SET NULL"), index=True
    )
    assay_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("assay.id", ondelete="SET NULL"), index=True
    )

    activity_type: Mapped[str | None] = mapped_column(sa.String(64))
    """``IC50``, ``Ki``, ``Kd``, ``EC50``, ``Potency``, ..."""

    standard_value: Mapped[float | None] = mapped_column(sa.Float())
    standard_units: Mapped[str | None] = mapped_column(sa.String(32))
    standard_relation: Mapped[str | None] = mapped_column(sa.String(8))
    """``=``, ``>``, ``<`` -- censored measurements matter for modelling."""

    pchembl_value: Mapped[float | None] = mapped_column(sa.Float())
    """-log10(molar activity); the comparable potency scale."""

    is_active: Mapped[bool | None] = mapped_column(sa.Boolean())
    activity_comment: Mapped[str | None] = mapped_column(sa.String(255))
    data_validity_comment: Mapped[str | None] = mapped_column(sa.String(255))
    """ChEMBL curator warning (e.g. "Potential transcription error"). Rows with a
    validity comment are excluded from model training."""

    potential_duplicate: Mapped[bool | None] = mapped_column(sa.Boolean())
    document_chembl_id: Mapped[str | None] = mapped_column(sa.String(32))
    dataset_version_id: Mapped[str | None] = mapped_column(
        sa.ForeignKey("dataset_version.id", ondelete="SET NULL"), index=True
    )

    compound: Mapped[Compound] = relationship(back_populates="bioactivities")
    assay: Mapped[Assay | None] = relationship(back_populates="bioactivities")
    target: Mapped[Any] = relationship("Target")
