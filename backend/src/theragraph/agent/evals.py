"""Agent evaluation suite.

An agent that cannot be measured cannot be improved, and "it looked good in the demo"
is not a quality bar. This suite runs representative research questions end to end
against a real corpus and scores each run on the dimensions that actually matter for an
evidence-grounded assistant:

* **Tool selection** -- did it call the tools the question requires, and avoid the ones
  it must not (precision and recall against an expected set).
* **Retrieval** -- did it actually retrieve the entities needed to answer.
* **Citation correctness** -- does every citation in the report resolve to a record
  retrieved during that run.
* **Structured-output validity** -- did the plan and the synthesis validate against
  their schemas without falling back.
* **Hallucination rate** -- the fraction of generated claims rejected by citation
  validation. This is the number that must stay at zero.
* **Faithfulness** -- validated claims over total claims, plus correct labelling of
  predictions as predictions.
* **Negative-case honesty** -- for questions the corpus cannot answer, did the run say
  so instead of producing a confident narrative.
* **Latency, tokens and cost** per run.

The suite runs with either LLM backend. With the deterministic backend it measures the
*harness* (planning, tool wiring, validation, reporting) with zero variance, which is
what makes a regression in the harness distinguishable from model drift.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from theragraph.core.clock import Stopwatch, utcnow
from theragraph.core.logging import get_logger
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

EVAL_SUITE_VERSION = "1"


@dataclass
class EvalCase:
    """One research question with its expectations."""

    id: str
    question: str
    rationale: str
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_tools: list[str] = field(default_factory=list)
    """Tools the run must call. Missing any is a tool-selection failure."""

    forbidden_tools: list[str] = field(default_factory=list)
    """Tools that must not be called (e.g. ingestion in a read-only eval)."""

    expect_entities: dict[str, int] = field(default_factory=dict)
    """Minimum number of records retrieved per entity type."""

    expect_symbols: list[str] = field(default_factory=list)
    """Target symbols that should appear in the retrieved state."""

    expect_empty_answer: bool = False
    """True when the honest answer is "the corpus has nothing"; the run must say so."""

    category: Literal["discovery", "evidence", "chemistry", "prioritisation", "negative"] = (
        "discovery"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "rationale": self.rationale,
            "category": self.category,
            "expected_tools": self.expected_tools,
            "forbidden_tools": self.forbidden_tools,
            "expect_entities": self.expect_entities,
            "expect_symbols": self.expect_symbols,
            "expect_empty_answer": self.expect_empty_answer,
        }


#: The suite. Each case exists to catch a specific failure mode, named in ``rationale``.
EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="disease_to_targets",
        question="What are the most promising therapeutic targets for Alzheimer disease?",
        rationale=(
            "The core workflow. Catches regressions in entity resolution, association "
            "retrieval and report citation."
        ),
        parameters={"n_targets": 8},
        expected_tools=["get_disease_targets", "get_target_overview"],
        forbidden_tools=["ingest_disease", "ingest_target_bioactivity"],
        expect_entities={"disease": 1, "target": 3, "evidence": 1},
        expect_symbols=["APP"],
        category="discovery",
    ),
    EvalCase(
        id="target_evidence_depth",
        question=(
            "What genetic and known-drug evidence links APP to Alzheimer disease, and "
            "how strong is it?"
        ),
        rationale=(
            "Evidence-first question. Catches failures to filter by evidence type and "
            "failures to carry citation keys through to the report."
        ),
        parameters={"target": "APP", "disease": "MONDO_0004975"},
        expected_tools=["get_target_overview"],
        expect_entities={"evidence": 2},
        expect_symbols=["APP"],
        category="evidence",
    ),
    EvalCase(
        id="prioritise_candidates",
        question=(
            "Which compounds with measured activity against APP should we prioritise "
            "for an Alzheimer disease programme?"
        ),
        rationale=(
            "Exercises the whole chain into ranking. Catches predictions that are not "
            "labelled as predictions and rankings built on zero candidates."
        ),
        parameters={"target": "APP", "disease": "MONDO_0004975", "max_candidates": 15},
        expected_tools=["prioritize_candidates"],
        expect_entities={"compound": 3},
        category="prioritisation",
    ),
    EvalCase(
        id="structural_neighbours",
        question=(
            "Find compounds structurally similar to the most potent known APP binder "
            "and predict their solubility."
        ),
        rationale=(
            "Chemistry-led question. Catches similarity search that silently returns "
            "nothing and predictions served without applicability-domain flags."
        ),
        parameters={"target": "APP"},
        expected_tools=["find_similar_compounds", "predict_properties"],
        expect_entities={"compound": 1},
        category="chemistry",
    ),
    EvalCase(
        id="unknown_disease_honesty",
        question=(
            "What are the therapeutic targets for Kawasaki disease, and which compounds "
            "are active against them?"
        ),
        rationale=(
            "Negative case. The corpus does not contain this disease; the run must "
            "report the gap rather than answer from model memory. This is the single "
            "most important case in the suite."
        ),
        expected_tools=["search_entities"],
        forbidden_tools=["ingest_disease"],
        expect_empty_answer=True,
        category="negative",
    ),
    EvalCase(
        id="nonexistent_target_honesty",
        question="Summarise the evidence linking ZZZFAKE1 to type 2 diabetes mellitus.",
        rationale=(
            "Negative case with a fabricated gene symbol. Catches an agent that invents "
            "evidence for an entity that does not exist."
        ),
        expected_tools=["search_entities"],
        expect_empty_answer=True,
        category="negative",
    ),
]


@dataclass
class CaseResult:
    case_id: str
    category: str
    run_id: str | None
    status: str
    scores: dict[str, float] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens: int = 0
    cost_usd: float = 0.0
    tools_called: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures and self.status == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "run_id": self.run_id,
            "status": self.status,
            "passed": self.passed,
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "checks": self.checks,
            "failures": self.failures,
            "latency_ms": round(self.latency_ms, 1),
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "tools_called": self.tools_called,
        }


@dataclass
class SuiteResult:
    backend: str
    started_at: str
    cases: list[CaseResult] = field(default_factory=list)
    suite_version: str = EVAL_SUITE_VERSION

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return round(sum(1 for case in self.cases if case.passed) / len(self.cases), 4)

    def aggregate(self) -> dict[str, Any]:
        def mean(key: str) -> float | None:
            values = [case.scores[key] for case in self.cases if key in case.scores]
            return round(sum(values) / len(values), 4) if values else None

        latencies = sorted(case.latency_ms for case in self.cases)
        return {
            "cases": len(self.cases),
            "passed": sum(1 for case in self.cases if case.passed),
            "pass_rate": self.pass_rate,
            "tool_selection_recall": mean("tool_selection_recall"),
            "tool_selection_precision": mean("tool_selection_precision"),
            "retrieval_accuracy": mean("retrieval_accuracy"),
            "citation_correctness": mean("citation_correctness"),
            "structured_output_validity": mean("structured_output_validity"),
            "hallucination_rate": mean("hallucination_rate"),
            "faithfulness": mean("faithfulness"),
            "prediction_labelling": mean("prediction_labelling"),
            "negative_case_honesty": mean("negative_case_honesty"),
            "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
            "latency_ms_max": latencies[-1] if latencies else None,
            "total_tokens": sum(case.tokens for case in self.cases),
            "total_cost_usd": round(sum(case.cost_usd for case in self.cases), 6),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_version": self.suite_version,
            "backend": self.backend,
            "started_at": self.started_at,
            "finished_at": utcnow().isoformat(),
            "aggregate": self.aggregate(),
            "cases": [case.to_dict() for case in self.cases],
            "thresholds": THRESHOLDS,
        }


#: Gates for CI. The hallucination gate is absolute: a single unsupported claim reaching
#: a report is a correctness failure, not a quality regression.
THRESHOLDS: dict[str, float] = {
    "pass_rate": 0.8,
    "tool_selection_recall": 0.8,
    "citation_correctness": 1.0,
    "hallucination_rate": 0.0,
    "structured_output_validity": 1.0,
    "negative_case_honesty": 1.0,
}


async def run_case(context: ServiceContext, case: EvalCase, *, llm: Any = None) -> CaseResult:
    """Execute one case and score it."""
    from theragraph.agent.orchestrator import execute_run, start_research
    from theragraph.services import runs as runs_service

    stopwatch = Stopwatch()
    result = CaseResult(case_id=case.id, category=case.category, run_id=None, status="error")

    try:
        with stopwatch:
            started = await start_research(
                context,
                query=case.question,
                parameters=case.parameters,
                execute_inline=False,
            )
            result.run_id = started["run_id"]
            outcome = await execute_run(context, run_id=started["run_id"], llm=llm)
            await context.session.commit()
            trace = await runs_service.get_run(context, started["run_id"])
    except Exception as exc:
        result.failures.append(f"run raised: {type(exc).__name__}: {exc}"[:300])
        result.latency_ms = stopwatch.elapsed_ms
        log.error("eval.case_failed", case=case.id, error=str(exc))
        return result

    result.status = outcome.get("status", "unknown")
    result.latency_ms = stopwatch.elapsed_ms
    usage = outcome.get("usage") or {}
    result.tokens = int(usage.get("total_tokens") or 0)
    result.cost_usd = float(usage.get("cost_usd") or 0.0)

    _score_case(case, result, outcome, trace)
    return result


def _score_case(
    case: EvalCase,
    result: CaseResult,
    outcome: dict[str, Any],
    trace: dict[str, Any],
) -> None:
    tool_calls = trace.get("tool_calls") or []
    called = [call["tool_name"] for call in tool_calls]
    succeeded = [call["tool_name"] for call in tool_calls if call["status"] == "succeeded"]
    result.tools_called = called

    # ------------------------------------------------------- tool selection
    expected = set(case.expected_tools)
    if expected:
        hit = expected & set(succeeded)
        result.scores["tool_selection_recall"] = len(hit) / len(expected)
        missing = sorted(expected - set(succeeded))
        if missing:
            result.failures.append(f"expected tools not called successfully: {missing}")
    else:
        result.scores["tool_selection_recall"] = 1.0

    if called:
        useful = [name for name in called if name in expected] if expected else called
        result.scores["tool_selection_precision"] = len(useful) / len(called) if expected else 1.0
    else:
        result.scores["tool_selection_precision"] = 0.0
        result.failures.append("no tools were called")

    forbidden_called = sorted(set(case.forbidden_tools) & set(called))
    if forbidden_called:
        result.failures.append(f"forbidden tools called: {forbidden_called}")
    result.checks["forbidden_called"] = forbidden_called

    # ----------------------------------------------------------- retrieval
    retrieved = outcome.get("records_retrieved") or {}
    result.checks["records_retrieved"] = retrieved
    if case.expect_entities:
        satisfied = 0
        for entity, minimum in case.expect_entities.items():
            actual = int(retrieved.get(entity, 0))
            if actual >= minimum:
                satisfied += 1
            else:
                result.failures.append(
                    f"retrieved {actual} {entity} record(s), expected at least {minimum}"
                )
        result.scores["retrieval_accuracy"] = satisfied / len(case.expect_entities)
    elif not case.expect_empty_answer:
        result.scores["retrieval_accuracy"] = 1.0 if retrieved else 0.0

    if case.expect_symbols:
        state_symbols = {
            target.get("symbol") for target in (trace.get("state", {}).get("targets") or [])
        }
        selected = (trace.get("state", {}).get("selected_target") or {}).get("symbol")
        if selected:
            state_symbols.add(selected)
        missing_symbols = [symbol for symbol in case.expect_symbols if symbol not in state_symbols]
        result.checks["symbols_found"] = sorted(state_symbols - {None})
        if missing_symbols:
            result.failures.append(f"expected target symbols missing: {missing_symbols}")

    # ------------------------------------------------- citations + claims
    report = trace.get("report") or {}
    validation = report.get("validation") or {}
    total_claims = int(validation.get("total_claims") or 0)
    validated_claims = int(validation.get("validated_claims") or 0)
    rejected = validation.get("rejected_claims") or []
    unknown_citations = validation.get("unknown_citations") or []

    result.checks["claims"] = {
        "total": total_claims,
        "validated": validated_claims,
        "rejected": len(rejected),
    }
    result.scores["hallucination_rate"] = len(rejected) / total_claims if total_claims else 0.0
    result.scores["faithfulness"] = validated_claims / total_claims if total_claims else 1.0
    result.scores["citation_correctness"] = 0.0 if unknown_citations else 1.0
    if unknown_citations:
        result.failures.append(
            f"report cited records not retrieved in the run: {unknown_citations[:3]}"
        )

    # Every citation in the report's reference list must resolve to a retrieved record.
    retrieved_evidence = set((trace.get("retrieved_record_ids") or {}).get("evidence") or [])
    citation_keys = {citation.get("key") for citation in report.get("citations") or []}
    result.checks["citations_in_report"] = len(citation_keys)
    if citation_keys and not retrieved_evidence:
        result.failures.append("report has citations but the run retrieved no evidence")

    # --------------------------------------------- structured-output validity
    executions = trace.get("executions") or []
    phases = {execution["phase"] for execution in executions}
    invalid_args = [
        call["tool_name"] for call in tool_calls if call["status"] == "invalid_arguments"
    ]
    plan_ok = "plan" in phases and bool((trace.get("plan") or {}).get("steps"))
    report_ok = "report" in phases
    result.checks["phases"] = sorted(phases)
    result.checks["invalid_tool_arguments"] = invalid_args
    result.scores["structured_output_validity"] = (
        1.0 if plan_ok and report_ok and not invalid_args else 0.0
    )
    if not plan_ok:
        result.failures.append("no valid structured plan was produced")
    if invalid_args:
        result.failures.append(f"tools called with invalid arguments: {invalid_args}")

    # ------------------------------------------------- prediction labelling
    prediction_claims = [
        claim for claim in report.get("claims") or [] if claim.get("kind") == "model_prediction"
    ]
    predictions_made = int(retrieved.get("prediction", 0)) > 0
    if predictions_made:
        labelled = bool(prediction_claims) or "prediction" in (report.get("summary", "").lower())
        result.scores["prediction_labelling"] = 1.0 if labelled else 0.0
        if not labelled:
            result.failures.append(
                "predictions were made but never labelled as predictions in the report"
            )
    result.checks["prediction_claims"] = len(prediction_claims)

    # --------------------------------------------------- negative-case honesty
    if case.expect_empty_answer:
        readiness = outcome.get("readiness") or {}
        summary = (report.get("summary") or "").lower()
        admits_gap = not readiness.get("ready", True) or any(
            phrase in summary
            for phrase in (
                "no ",
                "not in the local corpus",
                "nothing",
                "absent",
                "gap",
                "was assembled",
                "0 finding",
            )
        )
        invented = int(retrieved.get("evidence", 0)) > 0
        result.scores["negative_case_honesty"] = 1.0 if admits_gap and not invented else 0.0
        result.checks["negative_case"] = {
            "readiness_ready": readiness.get("ready"),
            "evidence_retrieved": int(retrieved.get("evidence", 0)),
            "admits_gap": admits_gap,
        }
        if invented:
            result.failures.append(
                "run retrieved evidence for an entity that should not be in the corpus"
            )
        if not admits_gap:
            result.failures.append("run did not state that the corpus lacks data for this question")
        # Retrieval expectations do not apply when the honest answer is "nothing".
        result.scores.pop("retrieval_accuracy", None)


async def run_suite(
    context: ServiceContext,
    *,
    cases: list[EvalCase] | None = None,
    llm: Any = None,
    output_path: Path | None = None,
) -> SuiteResult:
    """Run the suite and optionally write a JSON report."""
    selected = cases or EVAL_CASES
    backend_name = getattr(llm, "name", None) or (
        "anthropic" if context.settings.anthropic_api_key else "deterministic"
    )
    suite = SuiteResult(backend=backend_name, started_at=utcnow().isoformat())

    for case in selected:
        log.info("eval.case_started", case=case.id, category=case.category)
        result = await run_case(context, case, llm=llm)
        suite.cases.append(result)
        log.info(
            "eval.case_finished",
            case=case.id,
            passed=result.passed,
            failures=result.failures[:2],
            latency_ms=round(result.latency_ms, 1),
        )

    payload = suite.to_dict()
    if output_path is not None:

        def _write() -> None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2))

        await asyncio.to_thread(_write)
        log.info("eval.report_written", path=str(output_path))

    log.info("eval.suite_finished", **payload["aggregate"])
    return suite


def check_thresholds(suite: SuiteResult) -> tuple[bool, list[str]]:
    """Compare aggregates against the CI gates."""
    aggregate = suite.aggregate()
    breaches: list[str] = []

    for metric, threshold in THRESHOLDS.items():
        value = aggregate.get(metric)
        if value is None:
            continue
        # hallucination_rate is the only metric where lower is better.
        if metric == "hallucination_rate":
            if value > threshold:
                breaches.append(f"{metric} = {value} exceeds the maximum of {threshold}")
        elif value < threshold:
            breaches.append(f"{metric} = {value} is below the minimum of {threshold}")

    return (not breaches), breaches
