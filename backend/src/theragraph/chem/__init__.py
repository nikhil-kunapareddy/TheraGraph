"""Molecular processing: standardisation, descriptors, fingerprints, similarity.

This package is the only place RDKit is imported. Everything above it works with plain
Python dataclasses, which keeps the API layer, the ML layer and the agent tools free of
chemistry-toolkit specifics and makes them trivially testable.
"""

from theragraph.chem.descriptors import (
    DESCRIPTOR_SET_VERSION,
    DescriptorResult,
    compute_descriptors,
    describe_descriptor,
)
from theragraph.chem.fingerprints import (
    FINGERPRINT_KINDS,
    Fingerprint,
    bits_to_array,
    compute_fingerprint,
    tanimoto,
)
from theragraph.chem.molecule import (
    MoleculeInfo,
    murcko_scaffold,
    parse_smiles,
    standardize_smiles,
)
from theragraph.chem.similarity import (
    SimilarityHit,
    popcount_window,
    rank_by_tanimoto,
)

__all__ = [
    "DESCRIPTOR_SET_VERSION",
    "FINGERPRINT_KINDS",
    "DescriptorResult",
    "Fingerprint",
    "MoleculeInfo",
    "SimilarityHit",
    "bits_to_array",
    "compute_descriptors",
    "compute_fingerprint",
    "describe_descriptor",
    "murcko_scaffold",
    "parse_smiles",
    "popcount_window",
    "rank_by_tanimoto",
    "standardize_smiles",
    "tanimoto",
]
