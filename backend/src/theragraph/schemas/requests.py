"""Request bodies. Strict by construction: unknown fields are rejected."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128)]
Smiles = Annotated[str, Field(min_length=1, max_length=1000)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SimilarityRequest(StrictModel):
    smiles: Smiles = Field(description="Query structure as SMILES.")
    threshold: float = Field(default=0.7, ge=0.3, le=1.0)
    limit: int = Field(default=25, ge=1, le=100)
    fingerprint_kind: Literal["morgan_r2_2048", "morgan_r3_2048", "fcfp_r2_2048"] = "morgan_r2_2048"


class SubstructureRequest(StrictModel):
    smarts: Annotated[str, Field(min_length=1, max_length=1000)]
    limit: int = Field(default=25, ge=1, le=100)


class PropertiesRequest(StrictModel):
    smiles: Smiles


class PredictRequest(StrictModel):
    compound_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    smiles: list[Smiles] = Field(default_factory=list, max_length=100)
    endpoints: list[Annotated[str, Field(max_length=64)]] = Field(
        default_factory=list,
        description="Endpoints to predict. Empty means every active endpoint.",
    )
    model_pins: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Pin a specific model per endpoint, e.g. "
            "{'solubility': 'solubility-gin@2026.09.18.ab12cd34'}."
        ),
    )
    persist: bool = Field(
        default=True, description="Persist predictions for compounds in the corpus."
    )

    @field_validator("smiles", "compound_ids")
    @classmethod
    def _drop_empty(cls, value: list[str]) -> list[str]:
        return [item for item in value if item.strip()]

    def has_input(self) -> bool:
        return bool(self.compound_ids or self.smiles)


class PrioritizeRequest(StrictModel):
    target: Identifier
    disease: Identifier | None = None
    compound_ids: list[Identifier] | None = Field(default=None, max_length=200)
    config_name: (
        Literal["general_oral_small_molecule", "cns_small_molecule", "evidence_first"] | None
    ) = None
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Override objective weights by key; others keep their preset value.",
    )
    min_pchembl: float = Field(default=5.0, ge=0.0, le=15.0)
    limit: int = Field(default=25, ge=1, le=100)
    include_similar_to: Smiles | None = Field(
        default=None, description="Also consider structural neighbours of this molecule."
    )
    similarity_threshold: float = Field(default=0.7, ge=0.3, le=1.0)
    run_predictions: bool = True
    persist: bool = True

    @field_validator("weights")
    @classmethod
    def _non_negative(cls, value: dict[str, float]) -> dict[str, float]:
        negative = [key for key, weight in value.items() if weight < 0]
        if negative:
            raise ValueError(f"weights must be non-negative: {negative}")
        return value


class CompareRequest(StrictModel):
    compound_ids: list[Identifier] = Field(min_length=1, max_length=12)
    target: Identifier | None = None
    disease: Identifier | None = None
    config_name: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)


class AgentRunRequest(StrictModel):
    query: Annotated[str, Field(min_length=5, max_length=2000)]
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Workflow knobs: disease, target, n_targets, min_pchembl, max_candidates, "
            "ranking_preset, allow_ingestion."
        ),
    )
    seed: int = Field(default=1337, ge=0, le=2**31 - 1)
    idempotency_key: Annotated[str, Field(max_length=128)] | None = Field(
        default=None,
        description="Re-submitting the same key returns the original run instead of a new one.",
    )
    execute_inline: bool = Field(
        default=False,
        description=(
            "Run synchronously and return the result. Only for small workloads; the "
            "default enqueues a background job."
        ),
    )


class ReplayRequest(StrictModel):
    allow_version_drift: bool = Field(
        default=False,
        description=(
            "Permit replay when the pinned dataset or model versions have changed. "
            "The drift is recorded on the new run."
        ),
    )
    execute_inline: bool = False


class IngestDiseaseRequest(StrictModel):
    efo_id: Annotated[str, Field(min_length=3, max_length=64)]
    n_targets: int = Field(default=25, ge=1, le=100)
    mode: Literal["live", "record", "replay", "auto"] = "auto"


class IngestBioactivityRequest(StrictModel):
    target: Identifier
    max_activities: int = Field(default=300, ge=10, le=2000)
    activity_types: list[str] | None = Field(default=None, max_length=12)
    mode: Literal["live", "record", "replay", "auto"] = "auto"


class IngestEvidenceRequest(StrictModel):
    efo_id: Annotated[str, Field(min_length=3, max_length=64)]
    target_keys: list[Identifier] = Field(min_length=1, max_length=25)
    datasource_ids: list[str] | None = Field(default=None, max_length=25)
    per_datasource: int = Field(default=8, ge=1, le=50)
    mode: Literal["live", "record", "replay", "auto"] = "auto"


class IngestMoleculeNetRequest(StrictModel):
    dataset_key: Literal["esol", "bbbp", "lipophilicity", "clintox"]
    force_download: bool = False


class IngestSimilarRequest(StrictModel):
    smiles: Smiles
    threshold_percent: int = Field(default=70, ge=40, le=100)
    limit: int = Field(default=30, ge=1, le=100)


ModelFamily = Literal["random_forest", "xgboost", "gin"]
#: Baseline + graph by default: the spec's two-approach comparison.
DEFAULT_FAMILIES: tuple[ModelFamily, ...] = ("random_forest", "gin")


class TrainRequest(StrictModel):
    endpoint: Annotated[str, Field(min_length=2, max_length=64)]
    dataset_key: Literal["esol", "bbbp", "lipophilicity", "clintox"]
    families: list[ModelFamily] = Field(
        default_factory=lambda: list(DEFAULT_FAMILIES), max_length=3
    )
    cv_folds: int = Field(default=5, ge=2, le=10)
    run_cross_validation: bool = True
    run_random_split_reference: bool = True
    seed: int = Field(default=1337, ge=0, le=2**31 - 1)
    experiment_name: str | None = Field(default=None, max_length=255)
    baseline_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    graph_hyperparameters: dict[str, Any] = Field(default_factory=dict)


class SeedRequest(StrictModel):
    plan: Literal["demo", "smoke"] = "demo"
    mode: Literal["live", "record", "replay", "auto"] = "auto"


class CompareRunsRequest(StrictModel):
    run_ids: list[Annotated[str, Field(max_length=32)]] = Field(min_length=2, max_length=5)


class TokenRequest(StrictModel):
    api_key: Annotated[str, Field(min_length=8, max_length=256)]
