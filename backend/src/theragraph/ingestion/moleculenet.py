"""MoleculeNet connector -- public molecular property benchmarks.

These datasets supply the *labels* that TheraGraph's property models are trained on.
They are handled differently from ChEMBL/Open Targets data on purpose:

MoleculeNet rows are **training data**, not curated knowledge about a specific
therapeutic programme. They are therefore snapshotted to versioned files with a content
hash and a ``dataset_version`` row, but they are **not** inserted into the ``compound``
table. Keeping them out means a compound shown in the Compound Explorer always traces
back to a real curated chemistry source, while the ML layer still gets reproducible,
version-pinned labels.

Every molecule is re-standardised with the same RDKit pipeline used for ingested
compounds, and its Bemis-Murcko scaffold is computed at load time so scaffold splitting
is deterministic and does not depend on the ordering of the source file.

Data licence: MoleculeNet datasets are distributed by DeepChem under the MIT licence;
the underlying measurements come from the cited primary publications.
"""

from __future__ import annotations

import csv
import gzip
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from theragraph.chem.molecule import safe_standardize
from theragraph.core.errors import LiveIngestionDisabledError, ValidationError
from theragraph.core.hashing import hash_records, sha256_hex
from theragraph.core.http_client import SourceHttpClient
from theragraph.core.logging import get_logger
from theragraph.ingestion.base import IngestionMode, ResponseRecorder

log = get_logger(__name__)

SOURCE_ID = "moleculenet"

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to fetch, parse and interpret one benchmark."""

    key: str
    filename: str
    endpoint: str
    task_type: TaskType
    smiles_column: str
    label_column: str
    units: str | None
    description: str
    citation: str
    positive_class_meaning: str | None = None
    higher_is_better: bool | None = None
    gzipped: bool = False
    drop_na: bool = True
    expected_rows: int | None = None

    @property
    def dataset_name(self) -> str:
        return self.key


#: The benchmarks TheraGraph trains on. Endpoints were chosen for drug-discovery
#: relevance: solubility and lipophilicity drive formulation, BBB permeability gates CNS
#: programmes, and clinical toxicity is a hard filter in candidate triage.
DATASETS: dict[str, DatasetSpec] = {
    "esol": DatasetSpec(
        key="esol",
        filename="delaney-processed.csv",
        endpoint="solubility",
        task_type="regression",
        smiles_column="smiles",
        label_column="measured log solubility in mols per litre",
        units="log mol/L",
        description=("ESOL (Delaney): measured aqueous solubility for 1128 small molecules."),
        citation=(
            "Delaney JS. ESOL: estimating aqueous solubility directly from molecular "
            "structure. J Chem Inf Comput Sci (2004)."
        ),
        higher_is_better=True,
        expected_rows=1128,
    ),
    "bbbp": DatasetSpec(
        key="bbbp",
        filename="BBBP.csv",
        endpoint="bbb_permeability",
        task_type="classification",
        smiles_column="smiles",
        label_column="p_np",
        units=None,
        description=("BBBP: binary blood-brain barrier penetration labels for 2050 compounds."),
        citation=(
            "Martins IF, et al. A Bayesian approach to in silico blood-brain barrier "
            "penetration modeling. J Chem Inf Model (2012)."
        ),
        positive_class_meaning="BBB permeable (penetrates the blood-brain barrier)",
        expected_rows=2050,
    ),
    "lipophilicity": DatasetSpec(
        key="lipophilicity",
        filename="Lipophilicity.csv",
        endpoint="lipophilicity",
        task_type="regression",
        smiles_column="smiles",
        label_column="exp",
        units="logD (pH 7.4)",
        description=(
            "Lipophilicity (ChEMBL-derived): experimental octanol/water logD at pH 7.4 "
            "for 4200 compounds."
        ),
        citation="Wang NN, et al. / ChEMBL curated logD set, via MoleculeNet (2018).",
        expected_rows=4200,
    ),
    "clintox": DatasetSpec(
        key="clintox",
        filename="clintox.csv.gz",
        endpoint="clinical_toxicity",
        task_type="classification",
        smiles_column="smiles",
        label_column="CT_TOX",
        units=None,
        description=(
            "ClinTox: compounds that failed clinical trials for toxicity reasons versus "
            "FDA-approved drugs."
        ),
        citation=(
            "Gayvert KM, et al. A data-driven approach to predicting successes and "
            "failures of clinical trials. Cell Chemical Biology (2016)."
        ),
        positive_class_meaning="failed clinical trials for toxicity reasons",
        gzipped=True,
        expected_rows=1484,
    ),
}


@dataclass
class MoleculeNetRecord:
    """One standardised, labelled molecule."""

    smiles: str
    label: float
    scaffold: str | None
    inchikey: str
    original_smiles: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "label": self.label,
            "scaffold": self.scaffold,
            "inchikey": self.inchikey,
            "original_smiles": self.original_smiles,
        }


@dataclass
class MoleculeNetDataset:
    spec: DatasetSpec
    records: list[MoleculeNetRecord] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)
    retrieved_at: datetime | None = None
    source_url: str | None = None
    raw_bytes_hash: str | None = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def content_hash(self) -> str:
        return hash_records([record.to_dict() for record in self.records])

    def labels(self) -> list[float]:
        return [record.label for record in self.records]

    def smiles(self) -> list[str]:
        return [record.smiles for record in self.records]

    def scaffolds(self) -> list[str | None]:
        return [record.scaffold for record in self.records]

    def quality(self) -> dict[str, Any]:
        total = len(self.records) + len(self.rejects)
        labels = self.labels()
        report: dict[str, Any] = {
            "total_rows": total,
            "accepted_rows": len(self.records),
            "rejected_rows": len(self.rejects),
            "score": round(len(self.records) / total, 4) if total else 1.0,
            "duplicate_inchikeys": total
            - len({record.inchikey for record in self.records})
            - len(self.rejects),
            "unique_scaffolds": len({record.scaffold for record in self.records}),
            "reject_reasons": _count_reasons(self.rejects),
        }
        if labels:
            if self.spec.task_type == "classification":
                positives = sum(1 for label in labels if label >= 0.5)
                report["class_balance"] = {
                    "positive": positives,
                    "negative": len(labels) - positives,
                    "positive_rate": round(positives / len(labels), 4),
                }
            else:
                ordered = sorted(labels)
                report["label_stats"] = {
                    "min": round(ordered[0], 4),
                    "max": round(ordered[-1], 4),
                    "mean": round(sum(labels) / len(labels), 4),
                    "median": round(ordered[len(ordered) // 2], 4),
                }
        return report


def _count_reasons(rejects: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reject in rejects:
        for reason in reject.get("_quality_errors", ["unknown"]):
            key = str(reason).split(":")[0]
            counts[key] = counts.get(key, 0) + 1
    return counts


class MoleculeNetConnector:
    source_id = SOURCE_ID

    def __init__(
        self,
        client: SourceHttpClient,
        *,
        base_url: str,
        cache_dir: Path,
        recorder: ResponseRecorder | None = None,
    ) -> None:
        self._client = client
        self._base = base_url.rstrip("/")
        self._cache_dir = Path(cache_dir)
        self._recorder = recorder

    async def download(self, spec: DatasetSpec, *, force: bool = False) -> bytes:
        """Fetch the raw CSV, caching it on disk.

        The on-disk cache is what makes repeated training runs and CI fast, and it is
        also the artifact whose hash is recorded on the dataset version.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self._cache_dir / spec.filename
        if cached.exists() and not force:
            log.info("moleculenet.cache_hit", dataset=spec.key, path=str(cached))
            return cached.read_bytes()

        # Replay mode promises no egress. The benchmark CSVs are committed, so a miss
        # here means the file was deleted -- say so instead of quietly downloading it,
        # which would also change the dataset hash if upstream has since been revised.
        if self._recorder is not None and self._recorder.mode is IngestionMode.REPLAY:
            raise LiveIngestionDisabledError(
                f"{spec.key}: {spec.filename} is not in the local cache and live "
                "ingestion is disabled",
                details={"expected_path": str(cached), "mode": "replay"},
            )

        url = f"{self._base}/{spec.filename}"
        payload = await self._client.get_bytes(url)
        cached.write_bytes(payload)
        log.info("moleculenet.downloaded", dataset=spec.key, bytes=len(payload), url=url)
        return payload


def parse_dataset(
    spec: DatasetSpec, raw: bytes, *, retrieved_at: datetime, source_url: str | None = None
) -> MoleculeNetDataset:
    """Parse, standardise and label-check a downloaded benchmark."""
    text = gzip.decompress(raw).decode("utf-8") if spec.gzipped else raw.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValidationError(f"{spec.key}: CSV has no header row")
    missing = {spec.smiles_column, spec.label_column} - set(reader.fieldnames)
    if missing:
        raise ValidationError(
            f"{spec.key}: expected columns missing from source file",
            details={"missing": sorted(missing), "found": reader.fieldnames},
        )

    dataset = MoleculeNetDataset(
        spec=spec,
        retrieved_at=retrieved_at,
        source_url=source_url,
        raw_bytes_hash=sha256_hex(raw)[:16],
    )
    seen: set[str] = set()

    for index, row in enumerate(reader):
        raw_smiles = (row.get(spec.smiles_column) or "").strip()
        raw_label = (row.get(spec.label_column) or "").strip()

        if not raw_smiles:
            dataset.rejects.append({"row": index, "_quality_errors": ["missing_smiles"]})
            continue
        if not raw_label:
            if spec.drop_na:
                dataset.rejects.append({"row": index, "_quality_errors": ["missing_label"]})
                continue
            raw_label = "0"

        try:
            label = float(raw_label)
        except ValueError:
            dataset.rejects.append(
                {"row": index, "label": raw_label[:40], "_quality_errors": ["unparseable_label"]}
            )
            continue

        info, error = safe_standardize(raw_smiles)
        if info is None:
            dataset.rejects.append(
                {
                    "row": index,
                    "smiles": raw_smiles[:120],
                    "_quality_errors": [f"unparseable_structure: {error}"],
                }
            )
            continue

        # Duplicate structures would leak between scaffold-split folds and inflate
        # measured performance, so the first occurrence wins and the rest are dropped.
        if info.inchikey in seen:
            dataset.rejects.append(
                {
                    "row": index,
                    "inchikey": info.inchikey,
                    "_quality_errors": ["duplicate_structure"],
                }
            )
            continue
        seen.add(info.inchikey)

        dataset.records.append(
            MoleculeNetRecord(
                smiles=info.canonical_smiles,
                label=label,
                scaffold=info.murcko_scaffold,
                inchikey=info.inchikey,
                original_smiles=raw_smiles,
            )
        )

    log.info(
        "moleculenet.parsed",
        dataset=spec.key,
        accepted=len(dataset.records),
        rejected=len(dataset.rejects),
    )
    return dataset


def snapshot_path(root: Path, spec: DatasetSpec, version: str) -> Path:
    return Path(root) / "moleculenet" / spec.key / f"{spec.key}-{version}.csv"


def write_snapshot(dataset: MoleculeNetDataset, path: Path) -> Path:
    """Persist the standardised dataset -- this file is what training actually reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "label", "scaffold", "inchikey", "original_smiles"])
        for record in dataset.records:
            writer.writerow(
                [
                    record.smiles,
                    record.label,
                    record.scaffold or "",
                    record.inchikey,
                    record.original_smiles,
                ]
            )
    return path


def read_snapshot(path: Path, spec: DatasetSpec) -> MoleculeNetDataset:
    """Load a previously written snapshot (no network, no RDKit re-standardisation)."""
    dataset = MoleculeNetDataset(spec=spec)
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            dataset.records.append(
                MoleculeNetRecord(
                    smiles=row["smiles"],
                    label=float(row["label"]),
                    scaffold=row["scaffold"] or None,
                    inchikey=row["inchikey"],
                    original_smiles=row.get("original_smiles", row["smiles"]),
                )
            )
    return dataset


def get_spec(key: str) -> DatasetSpec:
    spec = DATASETS.get(key)
    if spec is None:
        raise ValidationError(
            f"unknown MoleculeNet dataset {key!r}",
            details={"available": sorted(DATASETS)},
        )
    return spec


def build_connector(
    *,
    base_url: str,
    cache_dir: Path,
    timeout_s: float,
    max_retries: int,
    user_agent: str,
    recorder: ResponseRecorder | None = None,
) -> MoleculeNetConnector:
    client = SourceHttpClient(
        SOURCE_ID,
        timeout_s=max(timeout_s, 120.0),
        max_retries=max_retries,
        user_agent=user_agent,
    )
    return MoleculeNetConnector(client, base_url=base_url, cache_dir=cache_dir, recorder=recorder)


DATA_SOURCE_DEFINITION = {
    "id": SOURCE_ID,
    "name": "MoleculeNet",
    "description": (
        "Curated public benchmarks for molecular machine learning (solubility, "
        "lipophilicity, BBB permeability, clinical toxicity)."
    ),
    "homepage_url": "https://moleculenet.org/",
    "api_base_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets",
    "license": "MIT (DeepChem distribution); underlying measurements per original publication",
    "citation": (
        "Wu Z, et al. MoleculeNet: a benchmark for molecular machine learning. "
        "Chemical Science (2018)."
    ),
    "entity_types": ["training_dataset"],
}

__all__ = [
    "DATASETS",
    "DATA_SOURCE_DEFINITION",
    "SOURCE_ID",
    "DatasetSpec",
    "MoleculeNetConnector",
    "MoleculeNetDataset",
    "MoleculeNetRecord",
    "build_connector",
    "get_spec",
    "parse_dataset",
    "read_snapshot",
    "snapshot_path",
    "write_snapshot",
]
