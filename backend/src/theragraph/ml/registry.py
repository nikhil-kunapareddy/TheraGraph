"""Model registry: artifact storage, versioning and in-process loading.

Model versions are *derived*, not chosen by hand:

``version = <retrieval date>.<hash of (dataset content hash, hyperparameters, feature
spec, code version)>``

Two consequences. Training twice with identical inputs yields the same version and
overwrites the same artifact -- no accidental version inflation. Changing anything that
could change a prediction yields a new version automatically, so a research run that
pins ``solubility-gin@2026.09.18.a1b2c3d4`` is pinning the exact numbers it saw.

Loaded models are cached per process with an LRU keyed by model id, because
deserialising a Random Forest with 500 trees takes ~200 ms and a ranking request scores
dozens of compounds across several endpoints.
"""

from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.core.clock import utcnow
from theragraph.core.errors import ModelNotAvailableError
from theragraph.core.hashing import hash_file, hash_payload
from theragraph.core.logging import get_logger
from theragraph.core.storage import ObjectStore
from theragraph.db.models import MLModel
from theragraph.db.repositories import ml as ml_repo
from theragraph.ml.baseline import BaselineModel
from theragraph.ml.calibration import Calibrator
from theragraph.ml.gnn import GraphModel

log = get_logger(__name__)

MODEL_FILENAMES = {
    "random_forest": "model.joblib",
    "xgboost": "model.joblib",
    "gin": "model.pt",
}
CALIBRATOR_FILENAME = "calibrator.json"
TRAINING_REFERENCE_FILENAME = "training_reference.json"
MAX_TRAINING_REFERENCE = 2000


@dataclass
class LoadedModel:
    """A model ready to serve, with its database row, calibrator and AD reference."""

    row: MLModel
    model: BaselineModel | GraphModel
    calibrator: Calibrator
    training_reference: TrainingReference | None = None
    _training_bits: list[bytes] | None = field(default=None, repr=False)

    @property
    def ref(self) -> str:
        return self.row.ref

    def predict(self, smiles: list[str]) -> Any:
        """Predict and apply calibration in one place, so no caller can forget to."""
        batch = self.model.predict_smiles(smiles)
        if batch.probabilities is not None and self.calibrator.method != "none":
            batch.probabilities = self.calibrator.apply(batch.probabilities).astype("float32")
        return batch

    def training_fingerprints(self) -> list[bytes]:
        """Fingerprints of the training reference set, computed once per process.

        Needed for the applicability-domain check on every prediction, so it is worth
        holding in memory next to the model rather than recomputing per request.
        """
        if self._training_bits is None:
            from theragraph.chem.fingerprints import compute_fingerprint

            reference = self.training_reference
            bits: list[bytes] = []
            for smiles in reference.smiles if reference else []:
                try:
                    bits.append(compute_fingerprint(smiles).bits)
                except Exception:
                    continue
            self._training_bits = bits
        return self._training_bits

    @property
    def label_scale(self) -> float:
        """Label standard deviation, used to normalise regression uncertainty."""
        if self.training_reference and self.training_reference.label_std:
            return self.training_reference.label_std
        return 1.0


@dataclass
class TrainingReference:
    """A subsample of the training set kept alongside the model artifact.

    Applicability-domain checks need to ask "how close is this molecule to anything the
    model has seen?". Persisting a bounded sample (rather than the whole training set,
    or nothing at all) keeps that question answerable at serving time without a database
    round trip and without unbounded artifact growth.
    """

    smiles: list[str]
    label_std: float | None = None
    label_mean: float | None = None
    n_training_total: int = 0
    scaffold_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "label_std": self.label_std,
            "label_mean": self.label_mean,
            "n_training_total": self.n_training_total,
            "scaffold_counts": self.scaffold_counts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingReference:
        return cls(
            smiles=list(payload.get("smiles") or []),
            label_std=payload.get("label_std"),
            label_mean=payload.get("label_mean"),
            n_training_total=int(payload.get("n_training_total") or 0),
            scaffold_counts=payload.get("scaffold_counts") or {},
        )


def compute_version(
    *,
    dataset_content_hash: str,
    hyperparameters: dict[str, Any],
    feature_spec: dict[str, Any],
    code_version: str,
    split_strategy: str,
) -> str:
    """Deterministic model version derived from everything that affects predictions."""
    digest = hash_payload(
        {
            "dataset": dataset_content_hash,
            "hyperparameters": hyperparameters,
            "feature_spec": feature_spec,
            "code": code_version,
            "split": split_strategy,
        },
        length=8,
    )
    return f"{utcnow().strftime('%Y.%m.%d')}.{digest}"


def artifact_key(endpoint: str, family: str, version: str, filename: str) -> str:
    return f"models/{endpoint}/{family}/{version}/{filename}"


def local_model_dir(root: Path, endpoint: str, family: str, version: str) -> Path:
    return Path(root) / endpoint / family / version


async def save_artifacts(
    model: BaselineModel | GraphModel,
    calibrator: Calibrator,
    *,
    endpoint: str,
    family: str,
    version: str,
    local_root: Path,
    store: ObjectStore | None = None,
    training_reference: TrainingReference | None = None,
) -> dict[str, Any]:
    """Write the model locally and, when configured, mirror it to object storage.

    Local-first: inference must work from disk even if object storage is unreachable,
    and the object store is the durable copy for other replicas.
    """
    directory = local_model_dir(local_root, endpoint, family, version)
    directory.mkdir(parents=True, exist_ok=True)
    filename = MODEL_FILENAMES.get(family, "model.bin")
    path = directory / filename

    metadata = model.save(path)
    calibrator.save(directory / CALIBRATOR_FILENAME)
    if training_reference is not None:
        import json as _json

        trimmed = TrainingReference(
            smiles=training_reference.smiles[:MAX_TRAINING_REFERENCE],
            label_std=training_reference.label_std,
            label_mean=training_reference.label_mean,
            n_training_total=training_reference.n_training_total,
            scaffold_counts=training_reference.scaffold_counts,
        )
        (directory / TRAINING_REFERENCE_FILENAME).write_text(_json.dumps(trimmed.to_dict()))

    artifact_hash = hash_file(path, length=32)
    size_bytes = path.stat().st_size
    uri = f"file://{path}"

    if store is not None:
        try:
            stored = await store.put_file(artifact_key(endpoint, family, version, filename), path)
            await store.put_file(
                artifact_key(endpoint, family, version, CALIBRATOR_FILENAME),
                directory / CALIBRATOR_FILENAME,
            )
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                await store.put_file(
                    artifact_key(endpoint, family, version, meta_path.name), meta_path
                )
            reference_path = directory / TRAINING_REFERENCE_FILENAME
            if reference_path.exists():
                await store.put_file(
                    artifact_key(endpoint, family, version, TRAINING_REFERENCE_FILENAME),
                    reference_path,
                )
            uri = stored.uri
        except Exception as exc:  # pragma: no cover - storage outage
            log.warning("registry.upload_failed", error=str(exc), version=version)

    return {
        "artifact_uri": uri,
        "artifact_hash": artifact_hash,
        "artifact_size_bytes": size_bytes,
        "local_path": str(path),
        "metadata": metadata,
    }


async def register_model(
    session: AsyncSession,
    *,
    name: str,
    version: str,
    family: str,
    endpoint: str,
    task_type: str,
    framework: str,
    framework_version: str,
    hyperparameters: dict[str, Any],
    feature_spec: dict[str, Any],
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    training_summary: dict[str, Any],
    artifact: dict[str, Any],
    training_dataset_version_id: str | None,
    experiment_id: str | None,
    split_strategy: str,
    units: str | None = None,
    description: str | None = None,
    positive_class_meaning: str | None = None,
    git_sha: str | None = None,
    random_seed: int | None = None,
    training_runtime_s: float | None = None,
    status: str = "active",
) -> MLModel:
    """Create/update the ``ml_model`` row for a trained artifact."""
    return await ml_repo.upsert_model(
        session,
        name=name,
        version=version,
        family=family,
        endpoint=endpoint,
        task_type=task_type,
        framework=framework,
        framework_version=framework_version,
        status=status,
        description=description,
        units=units,
        positive_class_meaning=positive_class_meaning,
        hyperparameters=hyperparameters,
        feature_spec=feature_spec,
        metrics=metrics,
        calibration=calibration,
        training_summary=training_summary,
        artifact_uri=artifact["artifact_uri"],
        artifact_hash=artifact["artifact_hash"],
        artifact_size_bytes=artifact["artifact_size_bytes"],
        training_dataset_version_id=training_dataset_version_id,
        experiment_id=experiment_id,
        trained_at=utcnow(),
        training_runtime_s=training_runtime_s,
        split_strategy=split_strategy,
        git_sha=git_sha,
        random_seed=random_seed,
    )


def _resolve_local_path(row: MLModel, local_root: Path) -> Path:
    """Find the artifact on disk, whether recorded as a file:// URI or an s3:// key."""
    filename = MODEL_FILENAMES.get(row.family, "model.bin")
    if row.artifact_uri and row.artifact_uri.startswith("file://"):
        candidate = Path(row.artifact_uri.removeprefix("file://"))
        if candidate.exists():
            return candidate
    return local_model_dir(local_root, row.endpoint, row.family, row.version) / filename


async def load_model(
    row: MLModel, *, local_root: Path, store: ObjectStore | None = None
) -> LoadedModel:
    """Load a registered model, pulling from object storage if absent locally."""
    path = _resolve_local_path(row, local_root)

    if not path.exists() and store is not None:
        filename = MODEL_FILENAMES.get(row.family, "model.bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await store.get_file(
                artifact_key(row.endpoint, row.family, row.version, filename), path
            )
            await store.get_file(
                artifact_key(row.endpoint, row.family, row.version, f"{path.stem}.meta.json"),
                path.with_suffix(".meta.json"),
            )
            with contextlib.suppress(Exception):
                await store.get_file(
                    artifact_key(row.endpoint, row.family, row.version, CALIBRATOR_FILENAME),
                    path.parent / CALIBRATOR_FILENAME,
                )
        except Exception as exc:
            raise ModelNotAvailableError(
                f"could not fetch artifact for {row.ref}: {exc}",
                details={"model": row.ref, "uri": row.artifact_uri},
            ) from exc

    if not path.exists():
        raise ModelNotAvailableError(
            f"artifact for {row.ref} not found on disk",
            details={"expected_path": str(path), "uri": row.artifact_uri},
        )

    if row.artifact_hash:
        actual = hash_file(path, length=32)
        if actual != row.artifact_hash:
            # A silently swapped artifact would invalidate every recorded prediction.
            raise ModelNotAvailableError(
                f"artifact hash mismatch for {row.ref}",
                details={"expected": row.artifact_hash, "actual": actual},
            )

    model: BaselineModel | GraphModel = (
        GraphModel.load(path) if row.family == "gin" else BaselineModel.load(path)
    )

    calibrator_path = path.parent / CALIBRATOR_FILENAME
    calibrator = (
        Calibrator.load(calibrator_path)
        if calibrator_path.exists()
        else Calibrator.from_dict(row.calibration or {"method": "none"})
    )

    reference_path = path.parent / TRAINING_REFERENCE_FILENAME
    training_reference: TrainingReference | None = None
    if reference_path.exists():
        import json as _json

        training_reference = TrainingReference.from_dict(_json.loads(reference_path.read_text()))

    log.info("registry.model_loaded", model=row.ref, family=row.family, path=str(path))
    return LoadedModel(
        row=row,
        model=model,
        calibrator=calibrator,
        training_reference=training_reference,
    )


class ModelCache:
    """Process-local LRU of loaded models, keyed by ``(model_id, artifact_hash)``."""

    def __init__(self, maxsize: int = 8) -> None:
        self._maxsize = maxsize
        self._entries: dict[str, LoadedModel] = {}
        self._order: list[str] = []

    def _key(self, row: MLModel) -> str:
        return f"{row.id}:{row.artifact_hash}"

    def get(self, row: MLModel) -> LoadedModel | None:
        key = self._key(row)
        entry = self._entries.get(key)
        if entry is not None:
            self._order.remove(key)
            self._order.append(key)
        return entry

    def put(self, row: MLModel, loaded: LoadedModel) -> None:
        key = self._key(row)
        if key in self._entries:
            self._order.remove(key)
        self._entries[key] = loaded
        self._order.append(key)
        while len(self._order) > self._maxsize:
            evicted = self._order.pop(0)
            self._entries.pop(evicted, None)

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()

    def stats(self) -> dict[str, Any]:
        return {"cached": len(self._entries), "maxsize": self._maxsize}


@lru_cache(maxsize=1)
def code_version() -> str:
    """Version string used in model version derivation."""
    from theragraph import __version__

    return __version__


def git_sha() -> str | None:
    """Current commit, recorded on models and runs for reproducibility."""
    import subprocess

    if shutil.which("git") is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # pragma: no cover
        return None
    sha = result.stdout.strip()
    return sha or None
