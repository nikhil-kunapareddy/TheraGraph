"""The agent workflow end to end, plus the eval harness itself.

A scripted LLM backend stands in for the model, so these tests assert the *harness*:
that a refusal is survivable, that a hallucinated citation is stripped, that a plan
naming a non-existent tool degrades gracefully, and that the trace is complete.
"""

from __future__ import annotations

from typing import Any

import pytest

from theragraph.agent.evals import (
    EVAL_CASES,
    THRESHOLDS,
    CaseResult,
    SuiteResult,
    check_thresholds,
    run_case,
    run_suite,
)
from theragraph.agent.llm import LLMResponse, ModelUsage
from theragraph.agent.orchestrator import execute_run, start_research
from theragraph.agent.planner import PlanStep, ResearchPlan, Synthesis
from theragraph.agent.tools import execute_tool
from theragraph.services import runs as runs_service
from theragraph.services.context import ServiceContext

pytestmark = pytest.mark.asyncio


class ScriptedBackend:
    """An LLM stand-in that returns exactly what a test needs."""

    name = "anthropic"  # exercise the LLM code path, not the deterministic one

    def __init__(
        self,
        *,
        plan: ResearchPlan | None = None,
        synthesis: Synthesis | None = None,
        refuse_plan: bool = False,
        refuse_synthesis: bool = False,
    ) -> None:
        self.plan = plan
        self.synthesis = synthesis
        self.refuse_plan = refuse_plan
        self.refuse_synthesis = refuse_synthesis
        self.calls: list[str] = []

    async def structured(
        self, *, system: str, user: str, schema_model: Any, max_tokens: Any = None
    ) -> LLMResponse:
        usage = ModelUsage(model="claude-opus-5", input_tokens=120, output_tokens=60, calls=1)
        if schema_model is ResearchPlan:
            self.calls.append("plan")
            if self.refuse_plan:
                return LLMResponse(
                    stop_reason="refusal", refusal_category="test", usage=usage, backend="anthropic"
                )
            return LLMResponse(
                parsed=self.plan, stop_reason="end_turn", usage=usage, backend="anthropic"
            )

        self.calls.append("synthesis")
        if self.refuse_synthesis:
            return LLMResponse(
                stop_reason="refusal", refusal_category="test", usage=usage, backend="anthropic"
            )
        return LLMResponse(
            parsed=self.synthesis, stop_reason="end_turn", usage=usage, backend="anthropic"
        )

    async def converse(
        self, *, system: str, messages: Any, tools: Any = None, max_tokens: Any = None
    ) -> LLMResponse:
        # No adaptive tool calls: return a closing statement.
        self.calls.append("converse")
        return LLMResponse(
            text="I have what I need.",
            stop_reason="end_turn",
            usage=ModelUsage(model="claude-opus-5", input_tokens=50, output_tokens=20, calls=1),
            backend="anthropic",
        )

    async def aclose(self) -> None:
        return None


def simple_plan() -> ResearchPlan:
    return ResearchPlan(
        objective="Find targets for Alzheimer disease",
        strategy="Resolve the disease, then read the leading target's evidence.",
        steps=[
            PlanStep(
                step=1,
                tool="get_disease_targets",
                purpose="Retrieve associated targets",
                arguments={"disease": "MONDO_0004975", "n_targets": 5},
            ),
            PlanStep(
                step=2,
                tool="get_target_overview",
                purpose="Read the leading target's evidence",
                arguments={"target": "{{prev.top_target}}", "min_pchembl": 0.0},
                depends_on_previous=True,
            ),
        ],
    )


class TestToolExecution:
    async def test_tool_returns_structured_output_with_provenance(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        execution = await execute_tool(
            context, "get_target_overview", {"target": "APP", "min_pchembl": 0.0}
        )
        assert execution.succeeded
        assert execution.result is not None
        assert execution.result.citations
        assert execution.result.record_ids["evidence"]

    async def test_invalid_arguments_are_reported_not_raised(self, context: ServiceContext) -> None:
        """The model must be able to correct itself, so a bad call is data, not a crash."""
        execution = await execute_tool(context, "find_similar_compounds", {"smiles": ""})
        assert execution.status == "invalid_arguments"
        assert execution.error

    async def test_unknown_tool_is_reported(self, context: ServiceContext) -> None:
        execution = await execute_tool(context, "no_such_tool", {})
        assert execution.status == "invalid_arguments"
        assert "available_tools" in str(execution.error) or execution.error

    async def test_write_tools_are_refused_unless_permitted(self, context: ServiceContext) -> None:
        execution = await execute_tool(
            context, "ingest_disease", {"efo_id": "MONDO_0004975"}, allow_mutating=False
        )
        assert execution.status == "failed"
        assert execution.error_code == "tool_not_permitted"

    async def test_write_tools_stay_offline_when_live_ingestion_is_disabled(
        self, context: ServiceContext
    ) -> None:
        """ALLOW_LIVE_INGESTION=false binds the agent exactly as it binds the REST API.

        The test settings disable live ingestion and the seed directory has no fixtures,
        so the only way this call can succeed is by reaching the network.
        """
        from theragraph.ingestion.pipelines import ensure_data_sources

        assert context.settings.allow_live_ingestion is False
        await ensure_data_sources(context.session)
        execution = await execute_tool(
            context, "ingest_disease", {"efo_id": "MONDO_0004975"}, allow_mutating=True
        )
        assert execution.status == "failed"
        assert execution.error_code == "live_ingestion_disabled"

    async def test_domain_errors_become_structured_failures(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        execution = await execute_tool(context, "get_target_overview", {"target": "NOPE"})
        assert execution.status == "failed"
        assert execution.error_code == "not_found"

    async def test_empty_result_is_flagged_with_guidance(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        execution = await execute_tool(
            context, "find_similar_compounds", {"smiles": "CCCCCCCCCCCCCCCC", "threshold": 0.95}
        )
        assert execution.succeeded
        assert execution.result is not None
        assert execution.result.is_empty
        assert execution.result.warnings


class TestWorkflow:
    async def test_full_workflow_produces_a_complete_trace(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        backend = ScriptedBackend(
            plan=simple_plan(),
            synthesis=Synthesis(
                summary="Alzheimer disease has two associated targets in this corpus, "
                "APP and BACE1, with genetic and known-drug evidence respectively.",
                claims=[],
                open_questions=["Is the corpus current?"],
                rationale="Weighted genetics above literature.",
            ),
        )
        started = await start_research(
            context, query="What targets are associated with Alzheimer disease?"
        )
        outcome = await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()

        assert outcome["status"] == "succeeded"
        assert outcome["records_retrieved"]["target"] >= 1
        assert outcome["usage"]["cost_usd"] > 0  # the scripted model reports token usage

        trace = await runs_service.get_run(context, started["run_id"])
        phases = [execution["phase"] for execution in trace["executions"]]
        assert phases == ["plan", "act", "validate", "report"] or set(phases) >= {
            "plan",
            "validate",
            "report",
        }
        assert [call["tool_name"] for call in trace["tool_calls"]] == [
            "get_disease_targets",
            "get_target_overview",
        ]
        assert trace["prompts"]
        assert trace["reproducibility_hash"]
        assert trace["report"]["markdown"]

    async def test_placeholder_resolution_across_steps(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        """Step 2's target came from step 1's result, not from the question."""
        backend = ScriptedBackend(
            plan=simple_plan(),
            synthesis=Synthesis(summary="A" * 50, claims=[], rationale="r"),
        )
        started = await start_research(context, query="Targets for Alzheimer disease?")
        await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()

        trace = await runs_service.get_run(context, started["run_id"])
        second = trace["tool_calls"][1]
        assert second["arguments"]["target"] in {"APP", "BACE1"}
        assert second["status"] == "succeeded"

    async def test_hallucinated_citation_is_stripped_from_the_report(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        """The central guardrail: an invented reference must not reach the reader."""
        # Pin the target so the run retrieves APP's genetic evidence: which target the
        # binding prefers is covered by its own test, and this one must isolate the
        # citation guardrail.
        plan = ResearchPlan(
            objective="Evidence for APP",
            strategy="Read APP's evidence directly.",
            steps=[
                PlanStep(
                    step=1,
                    tool="get_target_overview",
                    purpose="Retrieve APP evidence",
                    arguments={"target": "APP", "disease": "MONDO_0004975", "min_pchembl": 0.0},
                )
            ],
        )
        backend = ScriptedBackend(
            plan=plan,
            synthesis=Synthesis(
                summary="A" * 60,
                claims=[
                    {
                        "text": "APP has genetic association evidence in Alzheimer disease.",
                        "kind": "retrieved_fact",
                        "citations": ["open_targets:evidence-genetic-1"],
                    },
                    {
                        "text": "A landmark 2019 trial showed 80% efficacy.",
                        "kind": "retrieved_fact",
                        "citations": ["pmid:99999999"],
                    },
                ],
                rationale="r",
            ),
        )
        started = await start_research(context, query="Evidence for APP in Alzheimer disease?")
        await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()

        report = await runs_service.get_run(context, started["run_id"])
        validation = report["report"]["validation"]
        assert validation["total_claims"] == 2
        assert validation["validated_claims"] == 1
        assert "pmid:99999999" in validation["unknown_citations"]
        assert "80% efficacy" not in report["report"]["markdown"]

    async def test_plan_refusal_falls_back_to_the_deterministic_planner(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        """A safety refusal must not end the run with nothing."""
        backend = ScriptedBackend(
            refuse_plan=True,
            synthesis=Synthesis(summary="A" * 60, claims=[], rationale="r"),
        )
        started = await start_research(
            context, query="What targets are associated with Alzheimer disease?"
        )
        outcome = await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()

        assert outcome["status"] == "succeeded"
        trace = await runs_service.get_run(context, started["run_id"])
        plan_phase = next(e for e in trace["executions"] if e["phase"] == "plan")
        assert "deterministic planner" in plan_phase["rationale"]

    async def test_synthesis_refusal_still_produces_a_cited_report(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        backend = ScriptedBackend(plan=simple_plan(), refuse_synthesis=True)
        started = await start_research(context, query="Targets for Alzheimer disease?")
        outcome = await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()
        assert outcome["status"] == "succeeded"
        trace = await runs_service.get_run(context, started["run_id"])
        assert trace["report"]["markdown"]

    async def test_plan_naming_an_unknown_tool_is_sanitised(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        """A hallucinated tool name must not abort the run."""
        plan = ResearchPlan(
            objective="x",
            strategy="y",
            steps=[
                PlanStep(step=1, tool="magical_oracle", purpose="invent an answer", arguments={}),
                PlanStep(
                    step=2,
                    tool="get_disease_targets",
                    purpose="real work",
                    arguments={"disease": "MONDO_0004975"},
                ),
            ],
        )
        backend = ScriptedBackend(
            plan=plan, synthesis=Synthesis(summary="A" * 60, claims=[], rationale="r")
        )
        started = await start_research(context, query="Targets for Alzheimer disease?")
        outcome = await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()

        assert outcome["status"] == "succeeded"
        trace = await runs_service.get_run(context, started["run_id"])
        assert "magical_oracle" not in [call["tool_name"] for call in trace["tool_calls"]]
        assert "get_disease_targets" in [call["tool_name"] for call in trace["tool_calls"]]

    async def test_plan_requesting_a_write_tool_is_dropped_when_not_permitted(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        plan = ResearchPlan(
            objective="x",
            strategy="y",
            steps=[
                PlanStep(
                    step=1,
                    tool="ingest_disease",
                    purpose="fetch new data",
                    arguments={"efo_id": "MONDO_0004975"},
                ),
                PlanStep(
                    step=2,
                    tool="get_disease_targets",
                    purpose="read targets",
                    arguments={"disease": "MONDO_0004975"},
                ),
            ],
        )
        backend = ScriptedBackend(
            plan=plan, synthesis=Synthesis(summary="A" * 60, claims=[], rationale="r")
        )
        started = await start_research(context, query="Targets for Alzheimer disease?")
        await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()
        trace = await runs_service.get_run(context, started["run_id"])
        assert "ingest_disease" not in [call["tool_name"] for call in trace["tool_calls"]]

    async def test_unresolvable_required_argument_skips_the_step_with_a_reason(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        plan = ResearchPlan(
            objective="x",
            strategy="y",
            steps=[
                PlanStep(
                    step=1,
                    tool="find_similar_compounds",
                    purpose="expand chemistry",
                    arguments={"smiles": "{{state.best_compound_smiles}}"},
                )
            ],
        )
        backend = ScriptedBackend(
            plan=plan, synthesis=Synthesis(summary="A" * 60, claims=[], rationale="r")
        )
        started = await start_research(context, query="Find neighbours")
        outcome = await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()

        trace = await runs_service.get_run(context, started["run_id"])
        skipped = trace["tool_calls"][0]
        assert skipped["status"] == "skipped"
        assert skipped["error_code"] == "unresolved_arguments"
        assert outcome["readiness"]["ready"] is False

    async def test_failed_run_is_recorded_not_lost(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        class ExplodingBackend(ScriptedBackend):
            async def structured(self, **kwargs: Any) -> LLMResponse:
                raise RuntimeError("model transport exploded")

        started = await start_research(context, query="Targets for Alzheimer disease?")
        with pytest.raises(RuntimeError):
            await execute_run(context, run_id=started["run_id"], llm=ExplodingBackend())
        await context.session.commit()

        trace = await runs_service.get_run(context, started["run_id"])
        assert trace["status"] == "failed"
        assert trace["errors"]

    async def test_a_finished_run_cannot_be_executed_twice(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        from theragraph.core.errors import AgentError

        backend = ScriptedBackend(
            plan=simple_plan(), synthesis=Synthesis(summary="A" * 60, claims=[], rationale="r")
        )
        started = await start_research(context, query="Targets for Alzheimer disease?")
        await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()
        with pytest.raises(AgentError, match="already finished"):
            await execute_run(context, run_id=started["run_id"], llm=backend)


class TestReproducibility:
    async def test_identical_inputs_produce_the_same_hash(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        first = await start_research(context, query="Targets for Alzheimer disease?", seed=7)
        second = await start_research(context, query="Targets for Alzheimer disease?", seed=7)
        from theragraph.db.repositories import runs as runs_repo

        a = await runs_repo.get_run(context.session, first["run_id"], with_trace=False)
        b = await runs_repo.get_run(context.session, second["run_id"], with_trace=False)
        assert a is not None and b is not None
        assert a.reproducibility_hash == b.reproducibility_hash

    async def test_a_different_seed_changes_the_hash(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        from theragraph.db.repositories import runs as runs_repo

        first = await start_research(context, query="Targets?", seed=1)
        second = await start_research(context, query="Targets?", seed=2)
        a = await runs_repo.get_run(context.session, first["run_id"], with_trace=False)
        b = await runs_repo.get_run(context.session, second["run_id"], with_trace=False)
        assert a is not None and b is not None
        assert a.reproducibility_hash != b.reproducibility_hash

    async def test_replay_plan_is_refused_when_versions_drift(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        """A "replay" that silently used different data would break the guarantee."""
        from theragraph.core.clock import utcnow
        from theragraph.core.errors import ConflictError
        from theragraph.db.repositories import provenance as prov_repo

        backend = ScriptedBackend(
            plan=simple_plan(), synthesis=Synthesis(summary="A" * 60, claims=[], rationale="r")
        )
        started = await start_research(context, query="Targets for Alzheimer disease?")
        await execute_run(context, run_id=started["run_id"], llm=backend)
        await context.session.commit()

        await prov_repo.create_dataset_version(
            context.session,
            source_id="open_targets",
            dataset_name="disease_associations",
            version="test.2",
            content_hash="a-different-hash",
            record_count=99,
            retrieved_at=utcnow(),
        )
        await context.session.commit()

        with pytest.raises(ConflictError, match="version"):
            await runs_service.prepare_replay(context, started["run_id"])

        allowed = await runs_service.prepare_replay(
            context, started["run_id"], allow_version_drift=True
        )
        assert allowed["drift"]["reproducible"] is False

    async def test_run_comparison_explains_the_difference(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        backend = ScriptedBackend(
            plan=simple_plan(), synthesis=Synthesis(summary="A" * 60, claims=[], rationale="r")
        )
        first = await start_research(context, query="Targets for Alzheimer disease?")
        await execute_run(context, run_id=first["run_id"], llm=backend)
        second = await start_research(context, query="Compounds for Alzheimer disease?")
        await execute_run(context, run_id=second["run_id"], llm=backend)
        await context.session.commit()

        comparison = await runs_service.compare_runs(context, [first["run_id"], second["run_id"]])
        difference = comparison["differences"][0]
        assert difference["query_changed"] is True
        assert difference["dataset_versions"]["identical"] is True
        assert "query differs" in difference["verdict"]


class TestEvalHarness:
    async def test_eval_case_scores_a_good_run(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        case = next(item for item in EVAL_CASES if item.id == "disease_to_targets")
        result = await run_case(context, case)
        assert result.status == "succeeded"
        assert result.scores["hallucination_rate"] == 0.0
        assert result.scores["citation_correctness"] == 1.0

    async def test_negative_case_requires_honesty(
        self, context: ServiceContext, seeded: dict[str, Any]
    ) -> None:
        """The corpus has no Kawasaki disease; answering anyway must score zero."""
        case = next(item for item in EVAL_CASES if item.id == "unknown_disease_honesty")
        result = await run_case(context, case)
        assert result.scores["negative_case_honesty"] == 1.0
        assert result.checks["negative_case"]["evidence_retrieved"] == 0

    async def test_suite_aggregates_and_gates(
        self, context: ServiceContext, seeded: dict[str, Any], tmp_path: Any
    ) -> None:
        cases = [
            item
            for item in EVAL_CASES
            if item.id in ("disease_to_targets", "unknown_disease_honesty")
        ]
        suite = await run_suite(context, cases=cases, output_path=tmp_path / "eval.json")
        aggregate = suite.aggregate()
        assert aggregate["cases"] == 2
        assert aggregate["hallucination_rate"] == 0.0
        assert (tmp_path / "eval.json").exists()

    def test_threshold_gate_flags_a_hallucination(self) -> None:
        """The hallucination gate is absolute: any unsupported claim fails CI."""
        failing = SuiteResult(backend="test", started_at="2026-01-01T00:00:00Z")
        failing.cases = [
            CaseResult(
                case_id="x",
                category="discovery",
                run_id="RUN-1",
                status="succeeded",
                scores={
                    "hallucination_rate": 0.2,
                    "citation_correctness": 1.0,
                    "tool_selection_recall": 1.0,
                    "structured_output_validity": 1.0,
                },
            )
        ]
        ok, breaches = check_thresholds(failing)
        assert not ok
        assert any("hallucination_rate" in breach for breach in breaches)

    def test_threshold_gate_passes_a_clean_suite(self) -> None:
        clean = SuiteResult(backend="test", started_at="2026-01-01T00:00:00Z")
        clean.cases = [
            CaseResult(
                case_id="x",
                category="discovery",
                run_id="RUN-1",
                status="succeeded",
                scores={
                    "hallucination_rate": 0.0,
                    "citation_correctness": 1.0,
                    "tool_selection_recall": 1.0,
                    "structured_output_validity": 1.0,
                    "negative_case_honesty": 1.0,
                },
            )
        ]
        ok, breaches = check_thresholds(clean)
        assert ok, breaches

    def test_every_case_declares_what_it_guards(self) -> None:
        for case in EVAL_CASES:
            assert len(case.rationale) > 40, case.id
            assert case.expected_tools or case.expect_empty_answer

    def test_thresholds_include_the_non_negotiables(self) -> None:
        assert THRESHOLDS["hallucination_rate"] == 0.0
        assert THRESHOLDS["citation_correctness"] == 1.0
