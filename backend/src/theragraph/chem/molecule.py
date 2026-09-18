"""SMILES parsing, standardisation and scaffold extraction.

Standardisation happens once, at ingestion, and is the single source of truth for every
downstream step (descriptors, fingerprints, model features, dedup keys). Two records
describing the same molecule must produce the same canonical SMILES and InChIKey
regardless of which source they came from, or the whole provenance story falls apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from theragraph.core.errors import ChemistryError

# RDKit logs parse failures to stderr; we surface them as structured errors instead.
RDLogger.DisableLog("rdApp.*")

MAX_SMILES_LENGTH = 1000
MAX_HEAVY_ATOMS = 300


@dataclass(frozen=True)
class MoleculeInfo:
    """A standardised molecule plus the identifiers used to deduplicate it."""

    input_smiles: str
    canonical_smiles: str
    inchi: str
    inchikey: str
    formula: str
    heavy_atom_count: int
    num_fragments: int
    murcko_scaffold: str | None = None
    murcko_scaffold_inchikey: str | None = None
    standardization_notes: list[str] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return bool(self.standardization_notes)


def parse_smiles(smiles: str, *, sanitize: bool = True) -> Chem.Mol:
    """Parse SMILES into an RDKit molecule, raising a typed error on failure."""
    if not smiles or not smiles.strip():
        raise ChemistryError("SMILES string is empty")
    if len(smiles) > MAX_SMILES_LENGTH:
        raise ChemistryError(
            f"SMILES exceeds {MAX_SMILES_LENGTH} characters",
            details={"length": len(smiles)},
        )
    mol = Chem.MolFromSmiles(smiles.strip(), sanitize=sanitize)
    if mol is None:
        raise ChemistryError("could not parse SMILES", details={"smiles": smiles[:200]})
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        raise ChemistryError(
            f"molecule has {mol.GetNumHeavyAtoms()} heavy atoms (limit {MAX_HEAVY_ATOMS})",
            details={"heavy_atoms": mol.GetNumHeavyAtoms()},
        )
    return mol


def parse_smarts(smarts: str) -> Chem.Mol:
    """Parse a SMARTS query for substructure search."""
    if not smarts or not smarts.strip():
        raise ChemistryError("SMARTS pattern is empty")
    if len(smarts) > MAX_SMILES_LENGTH:
        raise ChemistryError("SMARTS pattern is too long")
    query = Chem.MolFromSmarts(smarts.strip())
    if query is None:
        raise ChemistryError("could not parse SMARTS", details={"smarts": smarts[:200]})
    return query


@lru_cache(maxsize=1)
def _uncharger() -> rdMolStandardize.Uncharger:
    return rdMolStandardize.Uncharger()


@lru_cache(maxsize=1)
def _fragment_chooser() -> rdMolStandardize.LargestFragmentChooser:
    return rdMolStandardize.LargestFragmentChooser()


def standardize_mol(mol: Chem.Mol) -> tuple[Chem.Mol, list[str]]:
    """Normalise a molecule and report what changed.

    Steps, in order: cleanup (normalise functional groups, disconnect metals),
    keep the largest organic fragment (drops salts/solvates), then neutralise charges.
    The notes are stored with the compound so a scientist can see that, say, a
    hydrochloride salt was reduced to its parent.
    """
    notes: list[str] = []
    original_fragments = len(Chem.GetMolFrags(mol))

    try:
        cleaned = rdMolStandardize.Cleanup(mol)
    except Exception as exc:  # pragma: no cover - RDKit internal failure
        raise ChemistryError(f"standardisation failed: {exc}") from exc

    if Chem.MolToSmiles(cleaned) != Chem.MolToSmiles(mol):
        notes.append("normalized_functional_groups")

    if original_fragments > 1:
        cleaned = _fragment_chooser().choose(cleaned)
        notes.append(f"kept_largest_of_{original_fragments}_fragments")

    charged = Chem.GetFormalCharge(cleaned)
    if charged != 0:
        uncharged = _uncharger().uncharge(cleaned)
        if Chem.GetFormalCharge(uncharged) != charged:
            notes.append("neutralized_charges")
        cleaned = uncharged

    if cleaned.GetNumHeavyAtoms() == 0:
        raise ChemistryError("molecule has no heavy atoms after standardisation")
    return cleaned, notes


def murcko_scaffold(mol: Chem.Mol) -> tuple[str | None, str | None]:
    """Bemis-Murcko scaffold SMILES + InChIKey.

    Returns ``(None, None)`` for acyclic molecules, which genuinely have no scaffold --
    important for scaffold splitting, where acyclic molecules must be grouped
    explicitly rather than silently collapsed into an empty-string bucket.
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:  # pragma: no cover
        return None, None
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None, None
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if not scaffold_smiles:
        return None, None
    try:
        scaffold_key = inchi.MolToInchiKey(scaffold)
    except Exception:  # pragma: no cover
        scaffold_key = None
    return scaffold_smiles, scaffold_key or None


def standardize_smiles(smiles: str, *, compute_scaffold: bool = True) -> MoleculeInfo:
    """Full standardisation pipeline for one SMILES string."""
    mol = parse_smiles(smiles)
    num_fragments = len(Chem.GetMolFrags(mol))
    standardized, notes = standardize_mol(mol)

    canonical = Chem.MolToSmiles(standardized)
    try:
        inchi_string = inchi.MolToInchi(standardized)
        inchikey = inchi.MolToInchiKey(standardized)
    except Exception as exc:  # pragma: no cover
        raise ChemistryError(f"InChI generation failed: {exc}") from exc
    if not inchikey:
        raise ChemistryError("InChIKey generation returned empty result")

    scaffold_smiles, scaffold_key = (
        murcko_scaffold(standardized) if compute_scaffold else (None, None)
    )

    from rdkit.Chem import rdMolDescriptors

    return MoleculeInfo(
        input_smiles=smiles.strip(),
        canonical_smiles=canonical,
        inchi=inchi_string or "",
        inchikey=inchikey,
        formula=rdMolDescriptors.CalcMolFormula(standardized),
        heavy_atom_count=standardized.GetNumHeavyAtoms(),
        num_fragments=num_fragments,
        murcko_scaffold=scaffold_smiles,
        murcko_scaffold_inchikey=scaffold_key,
        standardization_notes=notes,
    )


def safe_standardize(smiles: str) -> tuple[MoleculeInfo | None, str | None]:
    """Non-raising variant for batch ingestion: returns ``(info, error)``."""
    try:
        return standardize_smiles(smiles), None
    except ChemistryError as exc:
        return None, exc.message
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"unexpected error: {exc}"


def has_substructure(mol: Chem.Mol, query: Chem.Mol) -> bool:
    return mol.HasSubstructMatch(query)


def substructure_match_count(mol: Chem.Mol, query: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(query, uniquify=True))


def to_molblock(smiles: str) -> str:
    """2D coordinates for structure rendering in the UI."""
    from rdkit.Chem import AllChem

    mol = parse_smiles(smiles)
    AllChem.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol)


def rdkit_version() -> str:
    import rdkit

    return str(rdkit.__version__)


def molecule_summary(smiles: str) -> dict[str, Any]:
    """Compact dict used by API responses and agent tool output."""
    info = standardize_smiles(smiles)
    return {
        "input_smiles": info.input_smiles,
        "canonical_smiles": info.canonical_smiles,
        "inchikey": info.inchikey,
        "formula": info.formula,
        "heavy_atom_count": info.heavy_atom_count,
        "murcko_scaffold": info.murcko_scaffold,
        "standardization_notes": info.standardization_notes,
    }
