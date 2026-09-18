"""Pydantic request/response contracts for the HTTP API.

Request models are strict: unknown fields are rejected, ranges are bounded, and
identifiers are length-limited. That is the input-validation layer -- by the time a
router body reaches a service, it is known-good.

Response envelopes are intentionally thin. Service payloads are deeply nested
scientific documents whose shape is documented by the services themselves; wrapping
them in exhaustive response models would duplicate hundreds of fields and drift. The
envelopes pin the *stable* parts (ids, counts, provenance, pagination) and pass the
nested document through.
"""

from theragraph.schemas.common import (
    ErrorResponse,
    HealthResponse,
    JobAccepted,
    Page,
    ProvenanceInfo,
)
from theragraph.schemas.requests import (
    AgentRunRequest,
    CompareRequest,
    IngestBioactivityRequest,
    IngestDiseaseRequest,
    IngestMoleculeNetRequest,
    PredictRequest,
    PrioritizeRequest,
    PropertiesRequest,
    ReplayRequest,
    SimilarityRequest,
    SubstructureRequest,
    TrainRequest,
)

__all__ = [
    "AgentRunRequest",
    "CompareRequest",
    "ErrorResponse",
    "HealthResponse",
    "IngestBioactivityRequest",
    "IngestDiseaseRequest",
    "IngestMoleculeNetRequest",
    "JobAccepted",
    "Page",
    "PredictRequest",
    "PrioritizeRequest",
    "PropertiesRequest",
    "ProvenanceInfo",
    "ReplayRequest",
    "SimilarityRequest",
    "SubstructureRequest",
    "TrainRequest",
]
