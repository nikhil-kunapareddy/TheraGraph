"""Similarity and substructure search.

The searchable space here is tens of thousands of compounds, which is small enough that
an exact scan beats an approximate index -- *provided* the scan is narrowed first. Two
cheap filters do that:

* **Popcount bounds.** For a query with ``n`` set bits and threshold ``t``, Tanimoto
  cannot reach ``t`` unless the candidate's popcount lies in ``[t*n, n/t]``. That bound
  is pushed into SQL as an index range scan.
* **Screening fingerprints** for substructure search: a candidate can only contain the
  query pattern if its pattern-fingerprint bits are a superset of the query's, so the
  expensive graph-isomorphism check runs on a small survivor set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from theragraph.chem.fingerprints import (
    DEFAULT_KIND,
    bytes_to_uint8,
    compute_fingerprint,
    tanimoto_bulk,
)
from theragraph.chem.molecule import parse_smarts, parse_smiles


@dataclass(frozen=True)
class SimilarityHit:
    compound_id: str
    similarity: float


def popcount_window(query_popcount: int, threshold: float) -> tuple[int, int]:
    """Popcount range that can possibly reach ``threshold`` Tanimoto.

    Derivation: with |A| and |B| set bits, Tanimoto is
    ``intersection / union <= min(|A|, |B|) / max(|A|, |B|)``. Requiring that upper
    bound to be at least ``t`` gives ``t*|A| <= |B| <= |A|/t``.
    """
    if threshold <= 0:
        return 0, 1 << 30
    low = math.ceil(threshold * query_popcount)
    high = math.floor(query_popcount / threshold)
    return max(0, low), max(low, high)


def rank_by_tanimoto(
    query_bits: bytes,
    candidates: list[tuple[str, bytes, int]],
    *,
    threshold: float = 0.7,
    limit: int = 25,
) -> list[SimilarityHit]:
    """Score candidates against the query and return the top hits above threshold."""
    if not candidates:
        return []
    ids = [candidate[0] for candidate in candidates]
    similarities = tanimoto_bulk(query_bits, [candidate[1] for candidate in candidates])

    keep = np.flatnonzero(similarities >= threshold)
    if keep.size == 0:
        return []
    # argpartition keeps top-k selection linear instead of sorting the whole array.
    if keep.size > limit:
        scores = similarities[keep]
        top = keep[np.argpartition(-scores, limit - 1)[:limit]]
    else:
        top = keep
    order = top[np.argsort(-similarities[top], kind="stable")]
    return [SimilarityHit(compound_id=ids[i], similarity=float(similarities[i])) for i in order]


def nearest_neighbour_similarity(
    query_bits: bytes, candidates: list[tuple[str, bytes, int]]
) -> tuple[str | None, float]:
    """Closest candidate and its similarity -- the applicability-domain signal."""
    if not candidates:
        return None, 0.0
    similarities = tanimoto_bulk(query_bits, [candidate[1] for candidate in candidates])
    if similarities.size == 0:
        return None, 0.0
    best = int(np.argmax(similarities))
    return candidates[best][0], float(similarities[best])


def screen_substructure_candidates(
    query_pattern_bits: bytes, candidates: list[tuple[str, bytes, int]]
) -> list[str]:
    """Candidates whose screening bits are a superset of the query's.

    ``query AND NOT candidate == 0`` means every query bit is present in the candidate.
    Molecules failing this cannot contain the substructure, so they never reach the
    graph matcher.
    """
    if not candidates:
        return []
    query = bytes_to_uint8(query_pattern_bits)
    survivors: list[str] = []
    for compound_id, bits, _ in candidates:
        candidate = bytes_to_uint8(bits)
        if candidate.size != query.size:
            survivors.append(compound_id)
            continue
        if not np.any(np.bitwise_and(query, np.bitwise_not(candidate))):
            survivors.append(compound_id)
    return survivors


def matches_substructure(smiles: str, smarts: str) -> bool:
    """Exact substructure test (graph isomorphism) for a single molecule."""
    return parse_smiles(smiles).HasSubstructMatch(parse_smarts(smarts))


def substructure_query_fingerprint(smarts: str, *, n_bits: int = 2048) -> bytes:
    """Screening fingerprint of a SMARTS query.

    Must use the *same* algorithm and bit count as the stored molecule fingerprints
    (``rdkit_pattern_2048``); mixing algorithms makes the superset test meaningless and
    silently drops real matches.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdmolops

    query = parse_smarts(smarts)
    fingerprint = rdmolops.PatternFingerprint(query, fpSize=n_bits)
    return DataStructs.BitVectToBinaryText(fingerprint)


def similarity_matrix(fingerprints: list[bytes]) -> np.ndarray:
    """Pairwise Tanimoto matrix -- used for diversity analysis of a candidate set."""
    size = len(fingerprints)
    matrix = np.eye(size, dtype=np.float32)
    for index in range(size):
        if index + 1 >= size:
            continue
        row = tanimoto_bulk(fingerprints[index], fingerprints[index + 1 :])
        matrix[index, index + 1 :] = row
        matrix[index + 1 :, index] = row
    return matrix


def diversity_score(fingerprints: list[bytes]) -> float:
    """Mean pairwise distance (1 - Tanimoto) of a set; higher means more diverse."""
    if len(fingerprints) < 2:
        return 0.0
    matrix = similarity_matrix(fingerprints)
    upper = matrix[np.triu_indices(len(fingerprints), k=1)]
    return float(1.0 - upper.mean())


def query_fingerprint(smiles: str, kind: str = DEFAULT_KIND) -> bytes:
    return compute_fingerprint(smiles, kind).bits
