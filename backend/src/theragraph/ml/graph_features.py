"""Molecular graph featurisation for the neural model.

A molecule becomes a graph: atoms are nodes with chemistry-aware features, bonds are
undirected edges (stored as two directed edges) with their own features. Feature
vocabularies are *fixed and versioned* -- adding an element to the vocabulary changes
the input dimension and therefore requires a new model version, which is exactly why
the vocabulary is pinned here rather than inferred from whatever data happened to be
in the training set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

from theragraph.chem.molecule import parse_smiles

GRAPH_FEATURE_VERSION = "1"

#: Elements covered explicitly; anything else lands in the "other" slot. Chosen to
#: cover >99.9% of atoms in drug-like public datasets.
ATOM_VOCAB: tuple[int, ...] = (
    1,
    5,
    6,
    7,
    8,
    9,
    14,
    15,
    16,
    17,
    33,
    34,
    35,
    53,
)
DEGREES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
FORMAL_CHARGES: tuple[int, ...] = (-2, -1, 0, 1, 2)
NUM_HS: tuple[int, ...] = (0, 1, 2, 3, 4)
HYBRIDIZATIONS: tuple[str, ...] = ("SP", "SP2", "SP3", "SP3D", "SP3D2", "S")
CHIRAL_TAGS: tuple[str, ...] = (
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
)
BOND_TYPES: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
BOND_STEREO: tuple[str, ...] = ("STEREONONE", "STEREOE", "STEREOZ", "STEREOANY")


def _one_hot(value: Any, vocabulary: tuple[Any, ...], *, allow_other: bool = True) -> list[float]:
    size = len(vocabulary) + (1 if allow_other else 0)
    encoded = [0.0] * size
    try:
        encoded[vocabulary.index(value)] = 1.0
    except ValueError:
        if allow_other:
            encoded[-1] = 1.0
    return encoded


ATOM_FEATURE_DIM = (
    len(ATOM_VOCAB)
    + 1
    + len(DEGREES)
    + 1
    + len(FORMAL_CHARGES)
    + 1
    + len(NUM_HS)
    + 1
    + len(HYBRIDIZATIONS)
    + 1
    + len(CHIRAL_TAGS)
    + 1
    + 4  # aromatic, in-ring, mass-scaled, is-in-conjugated-system
)
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + len(BOND_STEREO) + 1 + 2  # + conjugated, in-ring


def atom_features(atom: Chem.Atom) -> list[float]:
    return [
        *_one_hot(atom.GetAtomicNum(), ATOM_VOCAB),
        *_one_hot(atom.GetTotalDegree(), DEGREES),
        *_one_hot(atom.GetFormalCharge(), FORMAL_CHARGES),
        *_one_hot(atom.GetTotalNumHs(), NUM_HS),
        *_one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS),
        *_one_hot(str(atom.GetChiralTag()), CHIRAL_TAGS),
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        atom.GetMass() * 0.01,
        1.0 if any(bond.GetIsConjugated() for bond in atom.GetBonds()) else 0.0,
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    return [
        *_one_hot(str(bond.GetBondType()), BOND_TYPES),
        *_one_hot(str(bond.GetStereo()), BOND_STEREO),
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]


@dataclass
class MolGraph:
    """One molecule as arrays. ``edge_index`` is shape (2, 2*n_bonds)."""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    n_nodes: int
    smiles: str

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def mol_to_graph(smiles: str) -> MolGraph:
    """Build the graph representation of a molecule.

    Single-atom molecules (e.g. ``[Na+]``) have no bonds; rather than special-casing
    them downstream, a self-loop is added so message passing and pooling stay
    well-defined.
    """
    mol = parse_smiles(smiles)
    nodes = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)

    sources: list[int] = []
    targets: list[int] = []
    edges: list[list[float]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        sources.extend([begin, end])
        targets.extend([end, begin])
        edges.extend([features, features])

    if not sources:
        sources = list(range(mol.GetNumAtoms()))
        targets = list(sources)
        edges = [[0.0] * BOND_FEATURE_DIM for _ in sources]

    return MolGraph(
        node_features=nodes,
        edge_index=np.asarray([sources, targets], dtype=np.int64),
        edge_features=np.asarray(edges, dtype=np.float32),
        n_nodes=int(mol.GetNumAtoms()),
        smiles=smiles,
    )


@dataclass
class GraphBatch:
    """Several graphs concatenated into one disjoint graph.

    This is how batching works without PyTorch Geometric: node features stack, edge
    indices are offset by the running node count, and a ``batch`` vector records which
    graph each node belongs to so pooling can scatter back to per-molecule vectors.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    batch: np.ndarray
    n_graphs: int
    labels: np.ndarray | None = None

    def sizes(self) -> dict[str, int]:
        return {
            "graphs": self.n_graphs,
            "nodes": int(self.node_features.shape[0]),
            "edges": int(self.edge_index.shape[1]),
        }


def collate_graphs(graphs: list[MolGraph], labels: np.ndarray | None = None) -> GraphBatch:
    node_blocks: list[np.ndarray] = []
    edge_blocks: list[np.ndarray] = []
    edge_feature_blocks: list[np.ndarray] = []
    batch_ids: list[np.ndarray] = []
    offset = 0

    for index, graph in enumerate(graphs):
        node_blocks.append(graph.node_features)
        edge_blocks.append(graph.edge_index + offset)
        edge_feature_blocks.append(graph.edge_features)
        batch_ids.append(np.full(graph.n_nodes, index, dtype=np.int64))
        offset += graph.n_nodes

    return GraphBatch(
        node_features=np.vstack(node_blocks),
        edge_index=np.hstack(edge_blocks) if edge_blocks else np.zeros((2, 0), dtype=np.int64),
        edge_features=np.vstack(edge_feature_blocks)
        if edge_feature_blocks
        else np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32),
        batch=np.concatenate(batch_ids),
        n_graphs=len(graphs),
        labels=labels,
    )


def graph_feature_spec() -> dict[str, Any]:
    """Recorded on the model so a checkpoint is never fed mismatched features."""
    return {
        "representation": "graph",
        "graph_feature_version": GRAPH_FEATURE_VERSION,
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "atom_vocab": list(ATOM_VOCAB),
        "n_features": ATOM_FEATURE_DIM,
    }
