"""Dataset container and featurisation for model training.

Featurisation is expensive (RDKit descriptors for 4000 molecules take seconds), so a
dataset computes each representation at most once and caches it in memory. The feature
*specification* -- fingerprint kind, radius, bit count, descriptor names and their
order -- is emitted alongside, stored on the model, and replayed verbatim at inference
time. That is the mechanism that prevents train/serve skew.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from theragraph.chem.descriptors import (
    DESCRIPTOR_SET_VERSION,
    compute_full_descriptor_block,
    descriptor_names,
)
from theragraph.chem.fingerprints import (
    DEFAULT_KIND,
    FINGERPRINT_KINDS,
    compute_count_fingerprint,
    compute_fingerprint,
)
from theragraph.chem.molecule import parse_smiles, rdkit_version
from theragraph.core.errors import ValidationError
from theragraph.core.logging import get_logger

log = get_logger(__name__)

Representation = Literal["fingerprint", "descriptors", "fingerprint+descriptors", "graph"]

#: Descriptors excluded from model features. These are either redundant with the label
#: (a solubility model must not see an estimated-solubility descriptor) or numerically
#: unstable across RDKit versions.
EXCLUDED_DESCRIPTORS = frozenset(
    {
        "Ipc",
        "MaxPartialCharge",
        "MinPartialCharge",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
        "BCUT2D_MWHI",
        "BCUT2D_MWLOW",
        "BCUT2D_CHGHI",
        "BCUT2D_CHGLO",
        "BCUT2D_LOGPHI",
        "BCUT2D_LOGPLOW",
        "BCUT2D_MRHI",
        "BCUT2D_MRLOW",
    }
)


def model_descriptor_names() -> list[str]:
    """Stable, filtered, alphabetically ordered descriptor list used as features."""
    return [name for name in descriptor_names() if name not in EXCLUDED_DESCRIPTORS]


@dataclass
class FeatureSpec:
    """Serialisable description of how inputs become model features."""

    representation: Representation
    fingerprint_kind: str | None = None
    fingerprint_bits: int | None = None
    fingerprint_radius: int | None = None
    use_counts: bool = True
    descriptor_names: list[str] = field(default_factory=list)
    descriptor_set_version: str = DESCRIPTOR_SET_VERSION
    standardize_descriptors: bool = True
    descriptor_mean: list[float] | None = None
    descriptor_scale: list[float] | None = None
    rdkit_version: str = field(default_factory=rdkit_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "fingerprint_kind": self.fingerprint_kind,
            "fingerprint_bits": self.fingerprint_bits,
            "fingerprint_radius": self.fingerprint_radius,
            "use_counts": self.use_counts,
            "descriptor_names": self.descriptor_names,
            "descriptor_set_version": self.descriptor_set_version,
            "standardize_descriptors": self.standardize_descriptors,
            "descriptor_mean": self.descriptor_mean,
            "descriptor_scale": self.descriptor_scale,
            "rdkit_version": self.rdkit_version,
            "n_features": self.n_features,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSpec:
        return cls(
            representation=payload["representation"],
            fingerprint_kind=payload.get("fingerprint_kind"),
            fingerprint_bits=payload.get("fingerprint_bits"),
            fingerprint_radius=payload.get("fingerprint_radius"),
            use_counts=payload.get("use_counts", True),
            descriptor_names=payload.get("descriptor_names") or [],
            descriptor_set_version=payload.get("descriptor_set_version", DESCRIPTOR_SET_VERSION),
            standardize_descriptors=payload.get("standardize_descriptors", True),
            descriptor_mean=payload.get("descriptor_mean"),
            descriptor_scale=payload.get("descriptor_scale"),
            rdkit_version=payload.get("rdkit_version", ""),
        )

    @property
    def n_features(self) -> int:
        total = 0
        if self.fingerprint_bits and "fingerprint" in self.representation:
            total += self.fingerprint_bits
        if "descriptors" in self.representation:
            total += len(self.descriptor_names)
        return total

    def featurize_one(self, smiles: str) -> np.ndarray:
        """Featurise a single molecule exactly as training did."""
        mol = parse_smiles(smiles)
        parts: list[np.ndarray] = []
        if "fingerprint" in self.representation and self.fingerprint_kind:
            if self.use_counts:
                parts.append(compute_count_fingerprint(mol, self.fingerprint_kind))
            else:
                fingerprint = compute_fingerprint(mol, self.fingerprint_kind)
                parts.append(fingerprint.to_array().astype(np.float32))
        if "descriptors" in self.representation and self.descriptor_names:
            block = compute_full_descriptor_block(mol)
            values = np.array(
                [block.get(name, 0.0) for name in self.descriptor_names], dtype=np.float32
            )
            if self.standardize_descriptors and self.descriptor_mean is not None:
                mean = np.asarray(self.descriptor_mean, dtype=np.float32)
                scale = np.asarray(self.descriptor_scale, dtype=np.float32)
                values = (values - mean) / np.where(scale == 0, 1.0, scale)
            parts.append(values)
        if not parts:
            raise ValidationError(f"feature spec produced no features: {self.representation}")
        return np.concatenate(parts).astype(np.float32)

    def featurize_many(self, smiles_list: list[str]) -> np.ndarray:
        return np.vstack([self.featurize_one(smiles) for smiles in smiles_list])


@dataclass
class MolecularDataset:
    """SMILES + labels + scaffolds, with lazily computed feature matrices."""

    name: str
    endpoint: str
    task_type: Literal["classification", "regression"]
    smiles: list[str]
    labels: np.ndarray
    scaffolds: list[str | None]
    identifiers: list[str] = field(default_factory=list)
    units: str | None = None
    dataset_version_id: str | None = None
    dataset_version_label: str | None = None
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not (len(self.smiles) == len(self.labels) == len(self.scaffolds)):
            raise ValidationError(
                "dataset arrays must be the same length",
                details={
                    "smiles": len(self.smiles),
                    "labels": len(self.labels),
                    "scaffolds": len(self.scaffolds),
                },
            )
        if not self.identifiers:
            self.identifiers = [f"{self.name}-{index}" for index in range(len(self.smiles))]

    def __len__(self) -> int:
        return len(self.smiles)

    @property
    def n_positive(self) -> int:
        return int((self.labels >= 0.5).sum()) if self.task_type == "classification" else 0

    def subset(self, indices: np.ndarray | list[int]) -> MolecularDataset:
        index_list = list(indices)
        return MolecularDataset(
            name=self.name,
            endpoint=self.endpoint,
            task_type=self.task_type,
            smiles=[self.smiles[i] for i in index_list],
            labels=self.labels[index_list],
            scaffolds=[self.scaffolds[i] for i in index_list],
            identifiers=[self.identifiers[i] for i in index_list],
            units=self.units,
            dataset_version_id=self.dataset_version_id,
            dataset_version_label=self.dataset_version_label,
        )

    # ----------------------------------------------------------------- features
    def fingerprint_matrix(
        self, kind: str = DEFAULT_KIND, *, use_counts: bool = True
    ) -> np.ndarray:
        cache_key = f"fp:{kind}:{use_counts}"
        if cache_key not in self._cache:
            if use_counts:
                matrix = np.vstack(
                    [compute_count_fingerprint(smiles, kind) for smiles in self.smiles]
                )
            else:
                matrix = np.vstack(
                    [
                        compute_fingerprint(smiles, kind).to_array().astype(np.float32)
                        for smiles in self.smiles
                    ]
                )
            self._cache[cache_key] = matrix.astype(np.float32)
        return self._cache[cache_key]

    def descriptor_matrix(self, names: list[str] | None = None) -> np.ndarray:
        selected = names or model_descriptor_names()
        cache_key = f"desc:{len(selected)}:{hash(tuple(selected))}"
        if cache_key not in self._cache:
            rows = []
            for smiles in self.smiles:
                block = compute_full_descriptor_block(parse_smiles(smiles))
                rows.append([block.get(name, 0.0) for name in selected])
            self._cache[cache_key] = np.asarray(rows, dtype=np.float32)
        return self._cache[cache_key]

    def build_features(
        self,
        representation: Representation,
        *,
        fingerprint_kind: str = DEFAULT_KIND,
        use_counts: bool = True,
        descriptor_list: list[str] | None = None,
        fit_scaler: bool = True,
    ) -> tuple[np.ndarray, FeatureSpec]:
        """Build the feature matrix and the spec needed to reproduce it at inference.

        Descriptor standardisation statistics are fitted on *this* dataset (the training
        split) and baked into the spec, so serving never recomputes them from serving
        traffic -- a subtle but real source of drift.
        """
        parts: list[np.ndarray] = []
        spec = FeatureSpec(representation=representation)

        if "fingerprint" in representation:
            matrix = self.fingerprint_matrix(fingerprint_kind, use_counts=use_counts)
            parts.append(matrix)
            spec.fingerprint_kind = fingerprint_kind
            spec.fingerprint_bits = int(matrix.shape[1])
            spec.fingerprint_radius = FINGERPRINT_KINDS[fingerprint_kind].get("radius")
            spec.use_counts = use_counts

        if "descriptors" in representation:
            names = descriptor_list or model_descriptor_names()
            matrix = self.descriptor_matrix(names)
            spec.descriptor_names = names
            if fit_scaler:
                mean = matrix.mean(axis=0)
                scale = matrix.std(axis=0)
                scale = np.where(scale < 1e-8, 1.0, scale)
                spec.descriptor_mean = [float(value) for value in mean]
                spec.descriptor_scale = [float(value) for value in scale]
                matrix = (matrix - mean) / scale
            parts.append(matrix.astype(np.float32))

        if not parts:
            raise ValidationError(f"unsupported representation: {representation}")
        return np.hstack(parts).astype(np.float32), spec

    def label_summary(self) -> dict[str, Any]:
        labels = np.asarray(self.labels, dtype=np.float64)
        if self.task_type == "classification":
            positives = int((labels >= 0.5).sum())
            return {
                "n": len(labels),
                "positive": positives,
                "negative": len(labels) - positives,
                "positive_rate": round(positives / len(labels), 4) if len(labels) else 0.0,
            }
        return {
            "n": len(labels),
            "min": round(float(labels.min()), 4),
            "max": round(float(labels.max()), 4),
            "mean": round(float(labels.mean()), 4),
            "std": round(float(labels.std()), 4),
        }

    def scaffold_summary(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        acyclic = 0
        for scaffold in self.scaffolds:
            if scaffold is None:
                acyclic += 1
                continue
            groups[scaffold] = groups.get(scaffold, 0) + 1
        sizes = sorted(groups.values(), reverse=True)
        return {
            "unique_scaffolds": len(groups),
            "acyclic_molecules": acyclic,
            "largest_scaffold_size": sizes[0] if sizes else 0,
            "singleton_scaffolds": sum(1 for size in sizes if size == 1),
        }


def load_dataset_from_snapshot(
    path: Path,
    *,
    name: str,
    endpoint: str,
    task_type: Literal["classification", "regression"],
    units: str | None = None,
    dataset_version_id: str | None = None,
    dataset_version_label: str | None = None,
) -> MolecularDataset:
    """Load a standardised MoleculeNet snapshot written by the ingestion pipeline."""
    import csv

    smiles: list[str] = []
    labels: list[float] = []
    scaffolds: list[str | None] = []
    identifiers: list[str] = []

    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            smiles.append(row["smiles"])
            labels.append(float(row["label"]))
            scaffolds.append(row["scaffold"] or None)
            identifiers.append(row.get("inchikey") or row["smiles"])

    if not smiles:
        raise ValidationError(f"snapshot {path} contained no rows")

    log.info("dataset.loaded", name=name, rows=len(smiles), path=str(path))
    return MolecularDataset(
        name=name,
        endpoint=endpoint,
        task_type=task_type,
        smiles=smiles,
        labels=np.asarray(labels, dtype=np.float32),
        scaffolds=scaffolds,
        identifiers=identifiers,
        units=units,
        dataset_version_id=dataset_version_id,
        dataset_version_label=dataset_version_label,
    )
