"""Ingestion: connectors and normalisers for public biomedical data sources.

Each connector is responsible for exactly three things:

1. talking to one upstream API through the resilient HTTP client,
2. normalising the response into TheraGraph's internal schema, and
3. attaching provenance (source, dataset version, retrieval timestamp, identifiers).

Connectors never touch the database. Pipelines (``ingestion.pipelines``) compose
connectors with repositories, which keeps normalisation logic pure and unit-testable.
"""

from theragraph.ingestion.base import (
    IngestionMode,
    IngestionOutcome,
    NormalizedBatch,
    ResponseRecorder,
)
from theragraph.ingestion.quality import DataQualityReport, QualityRule, evaluate_quality

__all__ = [
    "DataQualityReport",
    "IngestionMode",
    "IngestionOutcome",
    "NormalizedBatch",
    "QualityRule",
    "ResponseRecorder",
    "evaluate_quality",
]
