"""Train/validation/test splitting protocols.

Why this module exists at all: a random split of a molecular dataset is a measurement
error, not a design choice. Public property datasets are built from congeneric series --
dozens of molecules sharing one scaffold with a methyl moved around. A random split puts
some of those in train and some in test, so the model gets credit for memorising a
scaffold rather than generalising to new chemistry. Reported scaffold-split numbers are
typically much worse *and* much closer to what a project would actually experience with
a new series.

Three protocols are provided:

* :func:`scaffold_split` -- deterministic Bemis-Murcko grouping, largest groups first
  into train (the standard MoleculeNet protocol).
* :func:`random_split` -- reported only as the optimism baseline.
* :func:`cross_validation_folds` -- scaffold-grouped K-fold, for confidence intervals.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from theragraph.core.errors import ValidationError
from theragraph.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class SplitResult:
    strategy: str
    train_indices: np.ndarray
    valid_indices: np.ndarray
    test_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_indices),
            "valid": len(self.valid_indices),
            "test": len(self.test_indices),
        }

    def assert_disjoint(self) -> None:
        """Guard against the silent worst bug in this codebase: a leaked split."""
        train = set(self.train_indices.tolist())
        valid = set(self.valid_indices.tolist())
        test = set(self.test_indices.tolist())
        overlaps = {
            "train_valid": train & valid,
            "train_test": train & test,
            "valid_test": valid & test,
        }
        leaked = {name: len(values) for name, values in overlaps.items() if values}
        if leaked:
            raise ValidationError("split produced overlapping index sets", details=leaked)

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "sizes": self.sizes(), **self.metadata}


def _scaffold_groups(scaffolds: list[str | None]) -> dict[str, list[int]]:
    """Group indices by scaffold.

    Acyclic molecules have no Bemis-Murcko scaffold. Bundling them all into one bucket
    would drop ~20% of a solubility dataset into a single fold, so each acyclic molecule
    becomes its own singleton group instead.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, scaffold in enumerate(scaffolds):
        key = scaffold if scaffold else f"__acyclic__{index}"
        groups[key].append(index)
    return dict(groups)


def scaffold_split(
    scaffolds: list[str | None],
    *,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
    balanced: bool = False,
    seed: int = 1337,
) -> SplitResult:
    """Deterministic scaffold split.

    ``balanced=False`` reproduces the MoleculeNet protocol: scaffold groups sorted by
    descending size, filled into train, then validation, then test. The test set ends up
    dominated by rare scaffolds, which is exactly the hard, realistic case.

    ``balanced=True`` shuffles groups with a fixed seed before filling, which trades
    some difficulty for more comparable label distributions across folds.
    """
    if not scaffolds:
        raise ValidationError("cannot split an empty dataset")
    if not 0 < train_fraction < 1 or not 0 <= valid_fraction < 1:
        raise ValidationError("split fractions must lie in (0, 1)")
    if train_fraction + valid_fraction >= 1.0:
        raise ValidationError("train + valid fractions must leave room for a test set")

    total = len(scaffolds)
    groups = _scaffold_groups(scaffolds)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))

    if balanced:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ordered))
        ordered = [ordered[i] for i in order]

    train_cutoff = train_fraction * total
    valid_cutoff = (train_fraction + valid_fraction) * total

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for _, indices in ordered:
        if len(train) + len(indices) <= train_cutoff:
            train.extend(indices)
        elif len(train) + len(valid) + len(indices) <= valid_cutoff:
            valid.extend(indices)
        else:
            test.extend(indices)

    # Degenerate datasets (one giant scaffold) can starve a fold; move the smallest
    # groups rather than returning an empty test set.
    if not test and valid:
        test, valid = valid, []
    if not test:
        train, test = train[: -max(1, total // 10)], train[-max(1, total // 10) :]

    result = SplitResult(
        strategy="scaffold_balanced" if balanced else "scaffold",
        train_indices=np.asarray(sorted(train), dtype=np.int64),
        valid_indices=np.asarray(sorted(valid), dtype=np.int64),
        test_indices=np.asarray(sorted(test), dtype=np.int64),
        metadata={
            "n_scaffold_groups": len(groups),
            "train_fraction": train_fraction,
            "valid_fraction": valid_fraction,
            "balanced": balanced,
            "seed": seed if balanced else None,
            "scaffold_overlap": 0,
        },
    )
    result.assert_disjoint()
    _assert_no_scaffold_overlap(result, scaffolds)
    log.info("split.scaffold", **result.sizes(), groups=len(groups))
    return result


def _assert_no_scaffold_overlap(result: SplitResult, scaffolds: list[str | None]) -> None:
    """A scaffold must not appear in two folds -- otherwise the split does nothing."""

    def scaffold_set(indices: np.ndarray) -> set[str]:
        return {scaffolds[i] for i in indices.tolist() if scaffolds[i]}

    train = scaffold_set(result.train_indices)
    valid = scaffold_set(result.valid_indices)
    test = scaffold_set(result.test_indices)
    shared = (train & test) | (train & valid) | (valid & test)
    if shared:
        raise ValidationError(
            "scaffold split leaked scaffolds across folds",
            details={"shared_scaffolds": sorted(shared)[:5], "count": len(shared)},
        )


def random_split(
    n_samples: int,
    *,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
    seed: int = 1337,
) -> SplitResult:
    """Random split -- reported only to quantify how optimistic it is."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_samples)
    train_end = int(train_fraction * n_samples)
    valid_end = int((train_fraction + valid_fraction) * n_samples)
    result = SplitResult(
        strategy="random",
        train_indices=np.sort(permutation[:train_end]),
        valid_indices=np.sort(permutation[train_end:valid_end]),
        test_indices=np.sort(permutation[valid_end:]),
        metadata={"seed": seed, "train_fraction": train_fraction},
    )
    result.assert_disjoint()
    return result


def cross_validation_folds(
    scaffolds: list[str | None], *, n_folds: int = 5, seed: int = 1337
) -> list[SplitResult]:
    """Scaffold-grouped K-fold.

    Groups are distributed greedily into the currently smallest fold, which keeps fold
    sizes close despite highly uneven scaffold group sizes. No scaffold ever spans two
    folds, so each fold's score is an honest held-out estimate and their spread is a
    usable confidence interval.
    """
    if n_folds < 2:
        raise ValidationError("cross-validation needs at least 2 folds")

    groups = _scaffold_groups(scaffolds)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    ordered = sorted(keys, key=lambda key: -len(groups[key]))

    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    for key in ordered:
        smallest = min(range(n_folds), key=lambda i: len(buckets[i]))
        buckets[smallest].extend(groups[key])

    folds: list[SplitResult] = []
    for fold_index in range(n_folds):
        test = buckets[fold_index]
        train = [
            index for other in range(n_folds) if other != fold_index for index in buckets[other]
        ]
        result = SplitResult(
            strategy="cv_scaffold",
            train_indices=np.asarray(sorted(train), dtype=np.int64),
            valid_indices=np.asarray([], dtype=np.int64),
            test_indices=np.asarray(sorted(test), dtype=np.int64),
            metadata={
                "fold": fold_index,
                "n_folds": n_folds,
                "seed": seed,
                "n_scaffold_groups": len(groups),
            },
        )
        result.assert_disjoint()
        _assert_no_scaffold_overlap(result, scaffolds)
        folds.append(result)

    log.info(
        "split.cross_validation",
        n_folds=n_folds,
        fold_sizes=[len(fold.test_indices) for fold in folds],
    )
    return folds


def describe_split_difficulty(scaffolds: list[str | None], result: SplitResult) -> dict[str, Any]:
    """How hard is this split? Reported next to the metrics it produced."""

    def scaffold_set(indices: np.ndarray) -> set[str]:
        return {scaffolds[i] for i in indices.tolist() if scaffolds[i]}

    train_scaffolds = scaffold_set(result.train_indices)
    test_scaffolds = scaffold_set(result.test_indices)
    n_acyclic_test = sum(1 for index in result.test_indices.tolist() if not scaffolds[index])
    return {
        "train_scaffolds": len(train_scaffolds),
        "test_scaffolds": len(test_scaffolds),
        "shared_scaffolds": len(train_scaffolds & test_scaffolds),
        "acyclic_test_molecules": n_acyclic_test,
        # ``None``, not 0.0, when the test fold holds no scaffolded molecules: "0% of
        # test scaffolds are novel" and "there are no test scaffolds" are different
        # facts, and reporting the first for the second misleads the reader.
        "test_scaffolds_unseen_fraction": round(
            len(test_scaffolds - train_scaffolds) / len(test_scaffolds), 4
        )
        if test_scaffolds
        else None,
    }
