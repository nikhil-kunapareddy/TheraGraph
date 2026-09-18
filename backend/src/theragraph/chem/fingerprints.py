"""Molecular fingerprints.

Fingerprints are stored as packed bytes, not Python lists, because a 2048-bit Morgan
fingerprint is 256 bytes packed versus ~16 KB as a JSON array -- the difference between
a similarity search that scans memory and one that thrashes the database.

Tanimoto is computed with numpy bitwise ops over ``uint8`` views, which is ~50x faster
than per-bit Python loops and needs no native extension beyond numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

from theragraph.chem.molecule import parse_smiles
from theragraph.core.errors import ChemistryError

#: Supported fingerprint kinds. The key is what gets stored in
#: ``compound_fingerprint.kind`` and referenced by a model's ``feature_spec``.
FINGERPRINT_KINDS: dict[str, dict[str, Any]] = {
    "morgan_r2_2048": {
        "algorithm": "morgan",
        "radius": 2,
        "n_bits": 2048,
        "use_features": False,
        "description": "ECFP4-equivalent circular fingerprint; the similarity-search default.",
    },
    "morgan_r3_2048": {
        "algorithm": "morgan",
        "radius": 3,
        "n_bits": 2048,
        "use_features": False,
        "description": "ECFP6-equivalent; larger environments, sparser bits.",
    },
    "fcfp_r2_2048": {
        "algorithm": "morgan",
        "radius": 2,
        "n_bits": 2048,
        "use_features": True,
        "description": "Feature-based (pharmacophoric) circular fingerprint.",
    },
    "maccs_167": {
        "algorithm": "maccs",
        "radius": None,
        "n_bits": 167,
        "description": "166 curated structural keys; interpretable, low resolution.",
    },
    "rdkit_pattern_2048": {
        "algorithm": "pattern",
        "radius": None,
        "n_bits": 2048,
        "description": (
            "RDKit pattern fingerprint: the substructure *screening* fingerprint. It is "
            "defined for query molecules as well as for complete molecules, which is "
            "what makes the superset test valid -- a candidate can only contain a query "
            "if its bits are a superset of the query's."
        ),
    },
}

DEFAULT_KIND = "morgan_r2_2048"


@dataclass(frozen=True)
class Fingerprint:
    kind: str
    bits: bytes
    n_bits: int
    popcount: int
    radius: int | None = None

    def to_array(self) -> np.ndarray:
        return bits_to_array(self.bits, self.n_bits)

    def to_row(self, compound_id: str) -> dict[str, Any]:
        return {
            "compound_id": compound_id,
            "kind": self.kind,
            "bits": self.bits,
            "n_bits": self.n_bits,
            "popcount": self.popcount,
            "radius": self.radius,
        }


_generators: dict[str, Any] = {}


def _generator(kind: str) -> Any:
    """Cached RDKit fingerprint generator (construction is not free)."""
    if kind in _generators:
        return _generators[kind]
    spec = FINGERPRINT_KINDS[kind]
    algorithm = spec["algorithm"]
    if algorithm == "morgan":
        invariants = (
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            if spec.get("use_features")
            else None
        )
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=spec["radius"],
            fpSize=spec["n_bits"],
            atomInvariantsGenerator=invariants,
        )
    elif algorithm == "pattern":
        # PatternFingerprint has no generator API and must be called directly; using the
        # topological RDKit fingerprint here instead would silently break substructure
        # screening, because query and target bits would come from different algorithms.
        generator = None
    else:
        generator = None
    _generators[kind] = generator
    return generator


def compute_fingerprint(smiles_or_mol: str | Chem.Mol, kind: str = DEFAULT_KIND) -> Fingerprint:
    """Compute a packed fingerprint of the requested kind."""
    if kind not in FINGERPRINT_KINDS:
        raise ChemistryError(
            f"unknown fingerprint kind {kind!r}",
            details={"supported": sorted(FINGERPRINT_KINDS)},
        )
    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    spec = FINGERPRINT_KINDS[kind]

    if spec["algorithm"] == "maccs":
        bit_vector = MACCSkeys.GenMACCSKeys(mol)
    elif spec["algorithm"] == "pattern":
        from rdkit.Chem import rdmolops

        bit_vector = rdmolops.PatternFingerprint(mol, fpSize=spec["n_bits"])
    else:
        bit_vector = _generator(kind).GetFingerprint(mol)

    packed = DataStructs.BitVectToBinaryText(bit_vector)
    return Fingerprint(
        kind=kind,
        bits=packed,
        n_bits=int(bit_vector.GetNumBits()),
        popcount=int(bit_vector.GetNumOnBits()),
        radius=spec.get("radius"),
    )


def compute_count_fingerprint(
    smiles_or_mol: str | Chem.Mol, kind: str = DEFAULT_KIND
) -> np.ndarray:
    """Folded *count* fingerprint (bit collisions summed rather than saturated).

    Counts carry more information than bits for tree models, so the baseline models
    train on these while similarity search uses the binary form.
    """
    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    spec = FINGERPRINT_KINDS[kind]
    if spec["algorithm"] != "morgan":
        return bits_to_array(compute_fingerprint(mol, kind).bits, spec["n_bits"]).astype(np.float32)
    fp = _generator(kind).GetCountFingerprint(mol)
    vector = np.zeros(spec["n_bits"], dtype=np.float32)
    for index, count in fp.GetNonzeroElements().items():
        vector[index] = float(count)
    return vector


def bits_to_array(packed: bytes, n_bits: int) -> np.ndarray:
    """Unpack stored bytes into a ``uint8`` array of 0/1 values."""
    array = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    if array.size < n_bits:
        array = np.pad(array, (0, n_bits - array.size))
    return array[:n_bits]


def bytes_to_uint8(packed: bytes) -> np.ndarray:
    return np.frombuffer(packed, dtype=np.uint8)


_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def popcount(packed: bytes) -> int:
    """Number of set bits, via a 256-entry lookup table."""
    return int(_POPCOUNT_TABLE[bytes_to_uint8(packed)].sum())


def tanimoto(a: bytes, b: bytes) -> float:
    """Tanimoto (Jaccard) similarity between two packed fingerprints."""
    left = bytes_to_uint8(a)
    right = bytes_to_uint8(b)
    if left.size != right.size:
        raise ChemistryError(
            "fingerprints have different lengths",
            details={"left_bytes": int(left.size), "right_bytes": int(right.size)},
        )
    intersection = int(_POPCOUNT_TABLE[np.bitwise_and(left, right)].sum())
    union = int(_POPCOUNT_TABLE[np.bitwise_or(left, right)].sum())
    return intersection / union if union else 0.0


def tanimoto_bulk(query: bytes, candidates: list[bytes]) -> np.ndarray:
    """Vectorised Tanimoto of one query against many candidates.

    All candidates are stacked into a single 2D ``uint8`` matrix so the AND/OR and
    popcount run as three numpy kernels instead of ``len(candidates)`` Python calls.
    """
    if not candidates:
        return np.empty(0, dtype=np.float32)
    query_array = bytes_to_uint8(query)
    width = query_array.size
    matrix = np.zeros((len(candidates), width), dtype=np.uint8)
    for index, candidate in enumerate(candidates):
        row = bytes_to_uint8(candidate)
        if row.size == width:
            matrix[index] = row
        else:  # tolerate mixed widths rather than failing a whole batch
            matrix[index, : min(row.size, width)] = row[:width]

    intersection = _POPCOUNT_TABLE[np.bitwise_and(matrix, query_array)].sum(axis=1)
    union = _POPCOUNT_TABLE[np.bitwise_or(matrix, query_array)].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, intersection / union, 0.0)
    return similarity.astype(np.float32)


def dice_similarity(a: bytes, b: bytes) -> float:
    left, right = bytes_to_uint8(a), bytes_to_uint8(b)
    intersection = int(_POPCOUNT_TABLE[np.bitwise_and(left, right)].sum())
    total = int(_POPCOUNT_TABLE[left].sum()) + int(_POPCOUNT_TABLE[right].sum())
    return 2 * intersection / total if total else 0.0


def describe_kind(kind: str) -> dict[str, Any]:
    spec = FINGERPRINT_KINDS.get(kind)
    if spec is None:
        raise ChemistryError(f"unknown fingerprint kind {kind!r}")
    return {"kind": kind, **spec}
