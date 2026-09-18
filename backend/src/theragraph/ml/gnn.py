"""Graph neural network for molecular property prediction.

Architecture: a GINE-style message-passing network (Graph Isomorphism Network with edge
features), followed by mean+max pooling and an MLP head.

**Why no PyTorch Geometric.** PyG pins a compiled extension matrix against torch/CUDA
versions and is a recurring source of "works on my machine" deployment failures. The
operations actually needed -- scatter-add for message aggregation, scatter-mean/max for
pooling, and index offsetting for batching -- are ~40 lines of plain PyTorch using
``index_add_`` and ``scatter_reduce_``. The result is a smaller image, a dependency-free
install, and no version roulette. The trade-off is that exotic layers would have to be
written by hand; for a GINE encoder that cost is near zero.

**Uncertainty.** MC dropout: dropout stays active at inference and the forward pass runs
``n`` times. The spread across passes is an epistemic estimate, reported (and labelled)
as such -- never as a calibrated confidence interval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from torch import Tensor, nn

from theragraph.core.clock import Stopwatch
from theragraph.core.errors import ModelNotAvailableError
from theragraph.core.logging import get_logger
from theragraph.ml.base import PredictionBatch, TaskType, TrainingReport
from theragraph.ml.datasets import MolecularDataset
from theragraph.ml.graph_features import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    GRAPH_FEATURE_VERSION,
    GraphBatch,
    collate_graphs,
    graph_feature_spec,
    mol_to_graph,
)

log = get_logger(__name__)

DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "hidden_dim": 128,
    "n_layers": 4,
    "dropout": 0.15,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 120,
    "patience": 20,
    "grad_clip": 5.0,
    "seed": 1337,
    "mc_dropout_passes": 20,
}


def scatter_sum(source: Tensor, index: Tensor, n_bins: int) -> Tensor:
    """Sum ``source`` rows into ``n_bins`` buckets given by ``index``."""
    shape = (n_bins, source.shape[-1])
    output = torch.zeros(shape, dtype=source.dtype, device=source.device)
    return output.index_add_(0, index, source)


def scatter_mean(source: Tensor, index: Tensor, n_bins: int) -> Tensor:
    totals = scatter_sum(source, index, n_bins)
    counts = torch.zeros(n_bins, dtype=source.dtype, device=source.device).index_add_(
        0, index, torch.ones_like(index, dtype=source.dtype)
    )
    return totals / counts.clamp(min=1.0).unsqueeze(-1)


def scatter_max(source: Tensor, index: Tensor, n_bins: int) -> Tensor:
    output = torch.full(
        (n_bins, source.shape[-1]),
        float("-inf"),
        dtype=source.dtype,
        device=source.device,
    )
    expanded = index.unsqueeze(-1).expand(-1, source.shape[-1])
    output = output.scatter_reduce(0, expanded, source, reduce="amax", include_self=True)
    # Empty buckets would stay at -inf and poison the head; zero them instead.
    return torch.nan_to_num(output, neginf=0.0)


class GINEConv(nn.Module):
    """GIN convolution with edge features.

    ``h_i' = MLP((1 + eps) * h_i + sum_{j in N(i)} ReLU(h_j + W_e e_ji))``

    ``eps`` is learnable, so the layer can weight self-information against neighbour
    information per layer rather than being fixed at the GIN default.
    """

    def __init__(self, dim: int, edge_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.edge_encoder = nn.Linear(edge_dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        source, target = edge_index[0], edge_index[1]
        messages = torch.relu(x[source] + self.edge_encoder(edge_attr))
        aggregated = scatter_sum(messages, target, x.shape[0])
        return self.mlp((1.0 + self.eps) * x + aggregated)


class MolecularGIN(nn.Module):
    """Atom encoder -> L GINE layers with residuals -> mean+max pool -> MLP head."""

    def __init__(
        self,
        *,
        node_dim: int = ATOM_FEATURE_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList(
            GINEConv(hidden_dim, edge_dim, dropout=dropout) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor, n_graphs: int
    ) -> Tensor:
        h = self.node_encoder(x)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            # Residual connections keep 4+ message-passing layers trainable; without
            # them deeper GNNs over-smooth and every molecule collapses to one vector.
            h = h + self.dropout(torch.relu(norm(layer(h, edge_index, edge_attr))))
        pooled = torch.cat(
            [scatter_mean(h, batch, n_graphs), scatter_max(h, batch, n_graphs)], dim=-1
        )
        return self.head(pooled).squeeze(-1)

    def embed(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor, n_graphs: int
    ) -> Tensor:
        """Graph-level embedding, used for nearest-neighbour applicability checks."""
        h = self.node_encoder(x)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            h = h + self.dropout(torch.relu(norm(layer(h, edge_index, edge_attr))))
        return torch.cat(
            [scatter_mean(h, batch, n_graphs), scatter_max(h, batch, n_graphs)], dim=-1
        )


class _GraphTensors(TypedDict):
    """Keyword arguments for ``MolecularGIN.forward``/``embed``."""

    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch: Tensor
    n_graphs: int


def _to_tensors(batch: GraphBatch, device: torch.device) -> _GraphTensors:
    return {
        "x": torch.from_numpy(batch.node_features).to(device),
        "edge_index": torch.from_numpy(batch.edge_index).to(device),
        "edge_attr": torch.from_numpy(batch.edge_features).to(device),
        "batch": torch.from_numpy(batch.batch).to(device),
        "n_graphs": batch.n_graphs,
    }


@dataclass
class GraphModel:
    """A trained GIN wrapped with its featurisation and serving logic."""

    task_type: TaskType
    endpoint: str
    hyperparameters: dict[str, Any]
    network: MolecularGIN
    units: str | None = None
    family: str = "gin"
    label_mean: float = 0.0
    label_std: float = 1.0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # ---------------------------------------------------------------------- fit
    @classmethod
    def train(
        cls,
        dataset: MolecularDataset,
        train_indices: np.ndarray,
        valid_indices: np.ndarray,
        *,
        hyperparameters: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> tuple[GraphModel, TrainingReport]:
        """Train with early stopping on the validation fold.

        Regression labels are standardised using *training-fold* statistics only, and
        the statistics travel with the checkpoint so predictions are de-standardised
        identically at serving time.
        """
        params = {**DEFAULT_HYPERPARAMETERS, **(hyperparameters or {})}
        torch.manual_seed(int(params["seed"]))
        np.random.seed(int(params["seed"]))

        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        train_subset = dataset.subset(train_indices)
        valid_subset = (
            dataset.subset(valid_indices)
            if len(valid_indices)
            else dataset.subset(train_indices[:1])
        )

        train_graphs = [mol_to_graph(smiles) for smiles in train_subset.smiles]
        valid_graphs = [mol_to_graph(smiles) for smiles in valid_subset.smiles]
        train_labels = np.asarray(train_subset.labels, dtype=np.float32)
        valid_labels = np.asarray(valid_subset.labels, dtype=np.float32)

        if dataset.task_type == "regression":
            label_mean = float(train_labels.mean())
            label_std = float(train_labels.std()) or 1.0
        else:
            label_mean, label_std = 0.0, 1.0

        network = MolecularGIN(
            hidden_dim=int(params["hidden_dim"]),
            n_layers=int(params["n_layers"]),
            dropout=float(params["dropout"]),
        ).to(resolved_device)

        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(3, int(params["patience"]) // 3)
        )

        if dataset.task_type == "classification":
            positives = float((train_labels >= 0.5).sum())
            negatives = float(len(train_labels) - positives)
            pos_weight = torch.tensor(
                [negatives / positives if positives else 1.0], device=resolved_device
            )
            loss_fn: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            # Huber over MSE: solubility sets contain genuine extreme values
            # (log S = -11.6) that would otherwise dominate the gradient.
            loss_fn = nn.HuberLoss(delta=1.0)

        batch_size = int(params["batch_size"])
        best_loss = float("inf")
        best_state: dict[str, Tensor] | None = None
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []

        stopwatch = Stopwatch()
        with stopwatch:
            for epoch in range(int(params["max_epochs"])):
                network.train()
                order = np.random.permutation(len(train_graphs))
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, len(order), batch_size):
                    indices = order[start : start + batch_size]
                    if len(indices) < 2:  # BatchNorm needs >1 sample
                        continue
                    graph_batch = collate_graphs([train_graphs[i] for i in indices])
                    tensors = _to_tensors(graph_batch, resolved_device)
                    targets = torch.from_numpy(train_labels[indices]).to(resolved_device)
                    if dataset.task_type == "regression":
                        targets = (targets - label_mean) / label_std

                    optimizer.zero_grad(set_to_none=True)
                    predictions = network(**tensors)
                    loss = loss_fn(predictions, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), float(params["grad_clip"]))
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1

                validation_loss = _evaluate_loss(
                    network,
                    valid_graphs,
                    valid_labels,
                    loss_fn,
                    resolved_device,
                    batch_size,
                    task_type=dataset.task_type,
                    label_mean=label_mean,
                    label_std=label_std,
                )
                scheduler.step(validation_loss)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": round(epoch_loss / max(1, n_batches), 5),
                        "valid_loss": round(validation_loss, 5),
                        "lr": round(optimizer.param_groups[0]["lr"], 7),
                    }
                )

                if validation_loss < best_loss - 1e-5:
                    best_loss = validation_loss
                    best_state = {
                        key: value.detach().clone() for key, value in network.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(params["patience"]):
                        log.info("gnn.early_stop", epoch=epoch, best_valid_loss=best_loss)
                        break

        if best_state is not None:
            network.load_state_dict(best_state)

        model = cls(
            task_type=dataset.task_type,
            endpoint=dataset.endpoint,
            hyperparameters=params,
            network=network,
            units=dataset.units,
            label_mean=label_mean,
            label_std=label_std,
            device=resolved_device,
        )
        report = TrainingReport(
            family="gin",
            task_type=dataset.task_type,
            n_train=len(train_indices),
            n_valid=len(valid_indices),
            hyperparameters=params,
            training_seconds=stopwatch.elapsed_ms / 1000,
            history=history,
            feature_count=ATOM_FEATURE_DIM,
            notes=[
                f"epochs_run={len(history)}",
                f"best_valid_loss={round(best_loss, 5)}",
                f"device={resolved_device.type}",
                f"parameters={sum(p.numel() for p in network.parameters())}",
            ],
        )
        log.info(
            "gnn.trained",
            endpoint=dataset.endpoint,
            epochs=len(history),
            best_valid_loss=round(best_loss, 5),
            seconds=round(stopwatch.elapsed_ms / 1000, 2),
        )
        return model, report

    # ------------------------------------------------------------------ predict
    def predict_smiles(
        self, smiles: list[str], *, mc_dropout_passes: int | None = None
    ) -> PredictionBatch:
        """Predict with MC-dropout uncertainty."""
        if not smiles:
            return PredictionBatch()
        passes = int(
            mc_dropout_passes
            if mc_dropout_passes is not None
            else self.hyperparameters.get("mc_dropout_passes", 20)
        )
        graphs = [mol_to_graph(item) for item in smiles]
        batch_size = int(self.hyperparameters.get("batch_size", 64))

        samples: list[np.ndarray] = []
        for pass_index in range(max(1, passes)):
            # Deterministic mean prediction on the first pass (eval mode), then
            # stochastic passes with dropout active for the spread.
            self.network.train(mode=pass_index > 0)
            outputs: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(graphs), batch_size):
                    graph_batch = collate_graphs(graphs[start : start + batch_size])
                    tensors = _to_tensors(graph_batch, self.device)
                    logits = self.network(**tensors)
                    outputs.append(logits.detach().cpu().numpy().reshape(-1))
            samples.append(np.concatenate(outputs))
        self.network.eval()

        stacked = np.vstack(samples)
        mean_output = stacked[0]  # deterministic pass
        spread = stacked[1:].std(axis=0) if stacked.shape[0] > 1 else np.zeros_like(mean_output)

        if self.task_type == "classification":
            probabilities = 1.0 / (1.0 + np.exp(-mean_output))
            probability_samples = (
                1.0 / (1.0 + np.exp(-stacked[1:])) if stacked.shape[0] > 1 else None
            )
            uncertainty = (
                probability_samples.std(axis=0)
                if probability_samples is not None
                else np.zeros_like(probabilities)
            )
            return PredictionBatch(
                probabilities=probabilities.astype(np.float32),
                uncertainty=uncertainty.astype(np.float32),
                n_estimators=passes,
            )

        values = mean_output * self.label_std + self.label_mean
        return PredictionBatch(
            values=values.astype(np.float32),
            uncertainty=(spread * self.label_std).astype(np.float32),
            n_estimators=passes,
        )

    def embed_smiles(self, smiles: list[str]) -> np.ndarray:
        """Graph embeddings for the applicability-domain check."""
        if not smiles:
            return np.zeros((0, 2 * int(self.hyperparameters["hidden_dim"])), dtype=np.float32)
        graphs = [mol_to_graph(item) for item in smiles]
        self.network.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(graphs), 64):
                graph_batch = collate_graphs(graphs[start : start + 64])
                tensors = _to_tensors(graph_batch, self.device)
                outputs.append(self.network.embed(**tensors).detach().cpu().numpy())
        return np.vstack(outputs)

    # --------------------------------------------------------------- persistence
    @property
    def feature_spec_dict(self) -> dict[str, Any]:
        return {
            **graph_feature_spec(),
            "label_mean": self.label_mean,
            "label_std": self.label_std,
        }

    def save(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "hyperparameters": self.hyperparameters,
                "label_mean": self.label_mean,
                "label_std": self.label_std,
                "graph_feature_version": GRAPH_FEATURE_VERSION,
            },
            path,
        )
        metadata = {
            "family": "gin",
            "task_type": self.task_type,
            "endpoint": self.endpoint,
            "units": self.units,
            "hyperparameters": self.hyperparameters,
            "feature_spec": self.feature_spec_dict,
            "n_parameters": sum(p.numel() for p in self.network.parameters()),
        }
        path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
        return metadata

    @classmethod
    def load(cls, path: Path, *, device: str | None = None) -> GraphModel:
        path = Path(path)
        metadata_path = path.with_suffix(".meta.json")
        if not path.exists() or not metadata_path.exists():
            raise ModelNotAvailableError(f"graph model artifact missing at {path}")
        resolved_device = torch.device(device or "cpu")
        checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
        metadata = json.loads(metadata_path.read_text())

        stored_version = checkpoint.get("graph_feature_version")
        if stored_version != GRAPH_FEATURE_VERSION:
            raise ModelNotAvailableError(
                "graph feature version mismatch; retrain the model",
                details={"checkpoint": stored_version, "code": GRAPH_FEATURE_VERSION},
            )

        params = checkpoint["hyperparameters"]
        network = MolecularGIN(
            hidden_dim=int(params["hidden_dim"]),
            n_layers=int(params["n_layers"]),
            dropout=float(params["dropout"]),
        ).to(resolved_device)
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()

        return cls(
            task_type=metadata["task_type"],
            endpoint=metadata["endpoint"],
            hyperparameters=params,
            network=network,
            units=metadata.get("units"),
            label_mean=float(checkpoint.get("label_mean", 0.0)),
            label_std=float(checkpoint.get("label_std", 1.0)),
            device=resolved_device,
        )


def _evaluate_loss(
    network: MolecularGIN,
    graphs: list[Any],
    labels: np.ndarray,
    loss_fn: nn.Module,
    device: torch.device,
    batch_size: int,
    *,
    task_type: TaskType,
    label_mean: float,
    label_std: float,
) -> float:
    network.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            chunk = graphs[start : start + batch_size]
            if not chunk:
                continue
            graph_batch = collate_graphs(chunk)
            tensors = _to_tensors(graph_batch, device)
            targets = torch.from_numpy(labels[start : start + len(chunk)]).to(device)
            if task_type == "regression":
                targets = (targets - label_mean) / label_std
            predictions = network(**tensors)
            total += float(loss_fn(predictions, targets).item()) * len(chunk)
            count += len(chunk)
    return total / max(1, count)
