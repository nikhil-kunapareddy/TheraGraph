"""Molecular descriptor computation.

Two tiers are computed for every molecule:

1. A **curated set** of the descriptors medicinal chemists actually reason about
   (MW, LogP, TPSA, HBD/HBA, rotatable bonds, QED, ...), promoted to typed columns and
   used by the ranking system and the UI.
2. The **full RDKit descriptor block** (~200 values), stored as JSON and used as the
   baseline model's feature vector.

``DESCRIPTOR_SET_VERSION`` is bumped whenever the curated list or its computation
changes, so cached values are never silently mixed across definitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

from theragraph.chem.molecule import parse_smiles, rdkit_version

DESCRIPTOR_SET_VERSION = "2"

#: Curated descriptors: ``key -> (human label, unit, short explanation)``. The UI reads
#: this so every number shown to a scientist is self-documenting.
CURATED_DESCRIPTORS: dict[str, tuple[str, str | None, str]] = {
    "molecular_weight": ("Molecular weight", "Da", "Average molecular mass."),
    "exact_mass": ("Exact mass", "Da", "Monoisotopic mass."),
    "logp": ("LogP", None, "Crippen octanol/water partition coefficient (lipophilicity)."),
    "tpsa": ("TPSA", "Å²", "Topological polar surface area; correlates with permeability."),
    "hbd": ("H-bond donors", None, "Lipinski hydrogen-bond donor count."),
    "hba": ("H-bond acceptors", None, "Lipinski hydrogen-bond acceptor count."),
    "rotatable_bonds": ("Rotatable bonds", None, "Conformational flexibility."),
    "aromatic_rings": ("Aromatic rings", None, "Number of aromatic rings."),
    "ring_count": ("Rings", None, "Total ring count (SSSR)."),
    "heavy_atom_count": ("Heavy atoms", None, "Non-hydrogen atom count."),
    "fraction_csp3": ("Fraction Csp3", None, "Saturated carbon fraction; 3D character."),
    "formal_charge": ("Formal charge", None, "Net formal charge."),
    "stereocentres": ("Stereocentres", None, "Atoms with defined or possible stereochemistry."),
    "qed": ("QED", None, "Quantitative estimate of drug-likeness (0-1, higher is better)."),
    "molar_refractivity": ("Molar refractivity", None, "Crippen molar refractivity."),
    "heteroatom_count": ("Heteroatoms", None, "Non-carbon, non-hydrogen atoms."),
}


@dataclass(frozen=True)
class DescriptorResult:
    """Curated descriptors, rule-of-thumb verdicts and the full RDKit block."""

    curated: dict[str, float | int | None]
    rule_filters: dict[str, Any]
    descriptors: dict[str, float]
    descriptor_set_version: str = DESCRIPTOR_SET_VERSION
    rdkit_version: str = field(default_factory=rdkit_version)

    def get(self, key: str) -> float | int | None:
        return self.curated.get(key)

    def to_row(self) -> dict[str, Any]:
        """Flatten into the ``molecular_property`` column shape."""
        row: dict[str, Any] = dict(self.curated)
        row.update(
            {
                "lipinski_violations": self.rule_filters.get("lipinski_violations"),
                "passes_lipinski": self.rule_filters.get("passes_lipinski"),
                "passes_veber": self.rule_filters.get("passes_veber"),
                "cns_mpo_score": self.rule_filters.get("cns_mpo_score"),
                "descriptors": self.descriptors,
                "descriptor_set_version": self.descriptor_set_version,
                "rdkit_version": self.rdkit_version,
                "provider": "rdkit",
            }
        )
        return row


@lru_cache(maxsize=1)
def _descriptor_functions() -> list[tuple[str, Any]]:
    """All RDKit descriptor callables, ordered so the feature vector is stable.

    Order matters: the baseline model's feature vector is positional, and the order is
    recorded in the model's ``feature_spec``.
    """
    return sorted(Descriptors.descList, key=lambda item: item[0])


def descriptor_names() -> list[str]:
    return [name for name, _ in _descriptor_functions()]


def compute_full_descriptor_block(mol: Chem.Mol) -> dict[str, float]:
    """Every RDKit descriptor, with non-finite values coerced to 0.0.

    Some descriptors return ``inf``/``nan`` for exotic structures (e.g. Ipc on large
    fused systems). Those poison downstream models and JSON serialisation, so they are
    zeroed here -- once, centrally -- rather than in every consumer.
    """
    values: dict[str, float] = {}
    for name, function in _descriptor_functions():
        try:
            raw = function(mol)
        except Exception:
            raw = 0.0
        if raw is None or isinstance(raw, bool):
            raw = float(raw or 0.0)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            number = 0.0
        if not math.isfinite(number):
            number = 0.0
        values[name] = number
    return values


def _cns_mpo(mw: float, logp: float, tpsa: float, hbd: int, logd: float, pka: float) -> float:
    """CNS multi-parameter optimisation score (0-6).

    Wager et al. desirability functions. TheraGraph cannot measure LogD or pKa, so
    those two axes use conservative proxies (LogD approximated by LogP - 1, basic pKa
    assumed neutral). The score is therefore reported as an *indicative* CNS
    drug-likeness heuristic, and the UI labels it as such.
    """

    def ramp(value: float, good: float, bad: float) -> float:
        if good == bad:
            return 1.0 if value == good else 0.0
        if good < bad:
            if value <= good:
                return 1.0
            if value >= bad:
                return 0.0
            return (bad - value) / (bad - good)
        if value >= good:
            return 1.0
        if value <= bad:
            return 0.0
        return (value - bad) / (good - bad)

    return round(
        ramp(mw, 360.0, 500.0)
        + ramp(logp, 3.0, 5.0)
        + ramp(logd, 2.0, 4.0)
        + ramp(tpsa, 60.0, 20.0) * 0.5
        + ramp(tpsa, 90.0, 120.0) * 0.5
        + ramp(float(hbd), 0.5, 3.5)
        + ramp(pka, 8.0, 10.0),
        3,
    )


def compute_descriptors(smiles_or_mol: str | Chem.Mol) -> DescriptorResult:
    """Compute curated + full descriptors for a molecule."""
    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol

    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotatable = int(Lipinski.NumRotatableBonds(mol))

    try:
        qed_value = float(QED.qed(mol))
    except Exception:  # pragma: no cover - QED fails on some exotic valences
        qed_value = None  # type: ignore[assignment]

    curated: dict[str, float | int | None] = {
        "molecular_weight": round(mw, 4),
        "exact_mass": round(float(Descriptors.ExactMolWt(mol)), 4),
        "logp": round(logp, 4),
        "tpsa": round(tpsa, 3),
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotatable,
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "fraction_csp3": round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 4),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "stereocentres": int(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
        if _can_count_stereo(mol)
        else None,
        "qed": round(qed_value, 4) if qed_value is not None else None,
        "molar_refractivity": round(float(Crippen.MolMR(mol)), 4),
        "heteroatom_count": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
    }

    lipinski_violations = sum([mw > 500.0, logp > 5.0, hbd > 5, hba > 10])
    rule_filters: dict[str, Any] = {
        "lipinski_violations": lipinski_violations,
        "passes_lipinski": lipinski_violations <= 1,
        "passes_veber": rotatable <= 10 and tpsa <= 140.0,
        "lipinski_detail": {
            "mw_le_500": mw <= 500.0,
            "logp_le_5": logp <= 5.0,
            "hbd_le_5": hbd <= 5,
            "hba_le_10": hba <= 10,
        },
        "cns_mpo_score": _cns_mpo(mw, logp, tpsa, hbd, logd=logp - 1.0, pka=8.0),
        "cns_mpo_caveat": "LogD and pKa are approximated from LogP; indicative only.",
    }

    return DescriptorResult(
        curated=curated,
        rule_filters=rule_filters,
        descriptors=compute_full_descriptor_block(mol),
    )


def _can_count_stereo(mol: Chem.Mol) -> bool:
    try:
        Chem.FindPotentialStereo(mol)
        return True
    except Exception:  # pragma: no cover
        return False


def describe_descriptor(key: str) -> dict[str, str | None]:
    """Label/unit/explanation for a curated descriptor key."""
    label, unit, explanation = CURATED_DESCRIPTORS.get(
        key, (key.replace("_", " ").title(), None, "")
    )
    return {"key": key, "label": label, "unit": unit, "explanation": explanation}


def descriptor_dictionary() -> list[dict[str, str | None]]:
    return [describe_descriptor(key) for key in CURATED_DESCRIPTORS]


def descriptor_vector(mol: Chem.Mol, names: list[str]) -> list[float]:
    """Positional feature vector for the exact descriptor names a model was trained on.

    Missing names resolve to 0.0 rather than raising, so a model trained with an older
    RDKit still serves; the name list itself is pinned in the model's feature spec.
    """
    block = compute_full_descriptor_block(mol)
    return [block.get(name, 0.0) for name in names]
