"""Guardrails applied to tool output and to generated claims.

Validation runs *between* execution and synthesis, and again on the synthesised claims.
It answers three questions the model cannot be trusted to answer about itself:

1. Did each tool return something structurally usable, with provenance attached?
2. Did the run actually retrieve enough to answer the question, or is the model about to
   narrate an empty result set?
3. Does every factual claim cite a record that was genuinely retrieved in this run?

Failures are recorded, not silently corrected -- a run that produced nothing should say
so, and a claim that cannot be traced is removed rather than softened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theragraph.core.logging import get_logger
from theragraph.core.metrics import AGENT_VALIDATION_FAILURES

log = get_logger(__name__)

#: Fields that make a retrieved record citable. A tool result claiming to return
#: evidence without these is a bug in the tool, not a model error.
REQUIRED_PROVENANCE_FIELDS = ("source", "source_identifier", "retrieved_at")


@dataclass
class ToolOutputVerdict:
    tool: str
    ok: bool
    checks: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "checks": self.checks, "issues": self.issues}


def validate_tool_output(tool_name: str, result: Any) -> ToolOutputVerdict:
    """Structural and provenance checks on one tool result."""
    verdict = ToolOutputVerdict(tool=tool_name, ok=True)

    if result is None:
        verdict.ok = False
        verdict.issues.append("tool returned no result object")
        AGENT_VALIDATION_FAILURES.labels(kind="null_tool_result").inc()
        return verdict

    verdict.checks["has_summary"] = bool(getattr(result, "summary", None))
    verdict.checks["is_empty"] = bool(getattr(result, "is_empty", False))
    verdict.checks["n_record_types"] = len(getattr(result, "record_ids", {}) or {})
    verdict.checks["n_citations"] = len(getattr(result, "citations", []) or [])

    if not verdict.checks["has_summary"]:
        verdict.ok = False
        verdict.issues.append("tool result has no human-readable summary")

    data = getattr(result, "data", None)
    if not isinstance(data, dict):
        verdict.ok = False
        verdict.issues.append("tool result data is not a structured object")
        AGENT_VALIDATION_FAILURES.labels(kind="unstructured_tool_result").inc()
        return verdict

    # Evidence-bearing payloads must carry provenance on every record.
    missing_provenance = _find_records_without_provenance(data)
    verdict.checks["records_missing_provenance"] = len(missing_provenance)
    if missing_provenance:
        verdict.ok = False
        verdict.issues.append(
            f"{len(missing_provenance)} evidence record(s) lack source provenance"
        )
        AGENT_VALIDATION_FAILURES.labels(kind="missing_provenance").inc()

    # Any citation key offered to the model must be well-formed, or the model will
    # faithfully reproduce a malformed key and the claim will be rejected downstream.
    malformed = [
        key
        for key in (getattr(result, "citations", []) or [])
        if not isinstance(key, str) or ":" not in key
    ]
    verdict.checks["malformed_citation_keys"] = len(malformed)
    if malformed:
        verdict.ok = False
        verdict.issues.append(f"{len(malformed)} malformed citation key(s)")
        AGENT_VALIDATION_FAILURES.labels(kind="malformed_citation_key").inc()

    return verdict


def _find_records_without_provenance(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk a tool payload for evidence-shaped records missing provenance."""
    offenders: list[dict[str, Any]] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6 or len(offenders) > 20:
            return
        if isinstance(node, dict):
            looks_like_evidence = "evidence_type" in node and "id" in node
            if looks_like_evidence:
                provenance = node.get("provenance") or {}
                if not all(provenance.get(field) for field in REQUIRED_PROVENANCE_FIELDS):
                    offenders.append({"id": node.get("id"), "type": node.get("evidence_type")})
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node[:50]:
                walk(item, depth + 1)

    walk(data)
    return offenders


@dataclass
class RunReadiness:
    """Whether the run retrieved enough to justify writing conclusions."""

    ready: bool
    reasons: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    successful_tools: int = 0
    failed_tools: int = 0
    empty_tools: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reasons": self.reasons,
            "retrieved_counts": self.counts,
            "successful_tools": self.successful_tools,
            "failed_tools": self.failed_tools,
            "empty_tools": self.empty_tools,
        }


def assess_readiness(
    *,
    record_ids: dict[str, list[str]],
    successful_tools: int,
    failed_tools: int,
    empty_tools: int,
) -> RunReadiness:
    """Decide whether synthesis should proceed, and say why if not.

    A run with zero retrieved records still produces a report -- one that states the
    corpus had nothing, which is the correct scientific answer. What it must not produce
    is a confident narrative built on nothing, so readiness is recorded on the run and
    surfaced in the report.
    """
    counts = {entity: len(set(ids)) for entity, ids in (record_ids or {}).items() if ids}
    total = sum(counts.values())
    reasons: list[str] = []

    if successful_tools == 0:
        reasons.append("no tool call succeeded")
    if total == 0:
        reasons.append("no database records were retrieved")
    if failed_tools > successful_tools and failed_tools > 0:
        reasons.append(
            f"more tool calls failed ({failed_tools}) than succeeded ({successful_tools})"
        )

    ready = not reasons
    if not ready:
        log.warning("agent.run_not_ready", reasons=reasons, counts=counts)
        AGENT_VALIDATION_FAILURES.labels(kind="insufficient_retrieval").inc()

    return RunReadiness(
        ready=ready,
        reasons=reasons,
        counts=counts,
        successful_tools=successful_tools,
        failed_tools=failed_tools,
        empty_tools=empty_tools,
    )


def check_prediction_labelling(text: str, endpoints: list[str]) -> list[str]:
    """Warn when a summary quotes a predicted endpoint without saying it is predicted.

    A heuristic, and reported as a warning rather than used to block the report: the
    hard guarantee comes from the templated tables, which label predictions
    structurally. This catches sloppy prose in the narrative section.
    """
    warnings: list[str] = []
    lowered = text.lower()
    hedges = ("predict", "model", "estimated", "in silico", "computed")
    for endpoint in endpoints:
        label = endpoint.replace("_", " ")
        if label in lowered and not any(hedge in lowered for hedge in hedges):
            warnings.append(f"summary mentions {label} without indicating it is a model prediction")
    return warnings
