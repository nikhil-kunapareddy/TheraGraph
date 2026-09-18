"""Prometheus metrics.

Deliberately a small, fixed set of instruments with bounded label cardinality: route
templates (never raw paths), source names, tool names, model names. Unbounded labels
are the classic way to melt a metrics backend.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)

HTTP_REQUESTS = Counter(
    "theragraph_http_requests_total",
    "HTTP requests by route template, method and status class.",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "theragraph_http_request_duration_seconds",
    "HTTP request latency by route template.",
    ["method", "route"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
HTTP_IN_FLIGHT = Gauge(
    "theragraph_http_requests_in_flight",
    "Requests currently being served.",
    registry=REGISTRY,
)

DB_QUERIES = Counter(
    "theragraph_db_queries_total",
    "Database statements executed, by operation.",
    ["operation"],
    registry=REGISTRY,
)
DB_QUERY_LATENCY = Histogram(
    "theragraph_db_query_duration_seconds",
    "Database statement latency.",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
    registry=REGISTRY,
)
DB_SLOW_QUERIES = Counter(
    "theragraph_db_slow_queries_total",
    "Statements slower than the configured threshold.",
    ["operation"],
    registry=REGISTRY,
)

CACHE_EVENTS = Counter(
    "theragraph_cache_events_total",
    "Cache hits and misses.",
    ["result"],
    registry=REGISTRY,
)

UPSTREAM_REQUESTS = Counter(
    "theragraph_upstream_requests_total",
    "Requests to public biomedical data sources.",
    ["source", "outcome"],
    registry=REGISTRY,
)
UPSTREAM_LATENCY = Histogram(
    "theragraph_upstream_request_duration_seconds",
    "Public data source latency.",
    ["source"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
UPSTREAM_RETRIES = Counter(
    "theragraph_upstream_retries_total",
    "Retried upstream requests.",
    ["source", "reason"],
    registry=REGISTRY,
)

INFERENCE_LATENCY = Histogram(
    "theragraph_ml_inference_duration_seconds",
    "ML inference latency by model and endpoint.",
    ["model", "endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0),
    registry=REGISTRY,
)
INFERENCE_PREDICTIONS = Counter(
    "theragraph_ml_predictions_total",
    "Molecules scored, by model and endpoint.",
    ["model", "endpoint"],
    registry=REGISTRY,
)
FEATURIZATION_FAILURES = Counter(
    "theragraph_featurization_failures_total",
    "Molecules that failed featurisation.",
    ["reason"],
    registry=REGISTRY,
)

AGENT_RUNS = Counter(
    "theragraph_agent_runs_total",
    "Agent research runs by terminal status.",
    ["status"],
    registry=REGISTRY,
)
AGENT_RUN_LATENCY = Histogram(
    "theragraph_agent_run_duration_seconds",
    "End-to-end agent run latency.",
    buckets=(0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300),
    registry=REGISTRY,
)
AGENT_TOOL_CALLS = Counter(
    "theragraph_agent_tool_calls_total",
    "Agent tool invocations by tool and outcome.",
    ["tool", "outcome"],
    registry=REGISTRY,
)
AGENT_TOOL_LATENCY = Histogram(
    "theragraph_agent_tool_duration_seconds",
    "Agent tool latency.",
    ["tool"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
AGENT_TOKENS = Counter(
    "theragraph_agent_tokens_total",
    "LLM tokens consumed by model and kind (input/output/cache_read).",
    ["model", "kind"],
    registry=REGISTRY,
)
AGENT_COST_USD = Counter(
    "theragraph_agent_cost_usd_total",
    "Estimated LLM spend in USD.",
    ["model"],
    registry=REGISTRY,
)
AGENT_VALIDATION_FAILURES = Counter(
    "theragraph_agent_validation_failures_total",
    "Guardrail rejections (unsupported citations, invalid tool output, ...).",
    ["kind"],
    registry=REGISTRY,
)

JOBS_ENQUEUED = Counter(
    "theragraph_jobs_enqueued_total", "Jobs enqueued.", ["task"], registry=REGISTRY
)
JOBS_COMPLETED = Counter(
    "theragraph_jobs_completed_total",
    "Jobs finished, by task and terminal status.",
    ["task", "status"],
    registry=REGISTRY,
)
JOB_LATENCY = Histogram(
    "theragraph_job_duration_seconds",
    "Job execution latency.",
    ["task"],
    buckets=(0.05, 0.25, 1, 5, 15, 60, 300, 900),
    registry=REGISTRY,
)
JOB_QUEUE_DEPTH = Gauge(
    "theragraph_job_queue_depth", "Jobs waiting in the queue.", ["queue"], registry=REGISTRY
)

INGESTION_RECORDS = Counter(
    "theragraph_ingestion_records_total",
    "Records ingested by source and entity type.",
    ["source", "entity"],
    registry=REGISTRY,
)
INGESTION_REJECTS = Counter(
    "theragraph_ingestion_rejected_total",
    "Records rejected by data-quality checks.",
    ["source", "reason"],
    registry=REGISTRY,
)


def render_metrics() -> tuple[bytes, str]:
    """Return the exposition payload and content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def status_class(status_code: int) -> str:
    """Bucket status codes (``2xx``, ``4xx``, ...) to keep label cardinality low."""
    return f"{status_code // 100}xx"
