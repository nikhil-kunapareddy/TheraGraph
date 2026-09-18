"""Machine learning: datasets, splits, models, evaluation and inference.

The ML layer has one non-negotiable rule: **no random-split-only reporting**. Molecular
datasets contain families of near-identical analogues, so a random split leaks close
relatives between train and test and inflates every metric. Scaffold splits are the
default reported protocol here, random splits are shown alongside purely to quantify
that optimism gap.
"""

from theragraph.ml.datasets import MolecularDataset, load_dataset_from_snapshot
from theragraph.ml.splits import (
    SplitResult,
    cross_validation_folds,
    random_split,
    scaffold_split,
)

__all__ = [
    "MolecularDataset",
    "SplitResult",
    "cross_validation_folds",
    "load_dataset_from_snapshot",
    "random_split",
    "scaffold_split",
]
