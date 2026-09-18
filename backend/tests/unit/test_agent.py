"""Agent internals: planning, placeholder binding, output validation, cost accounting."""

from __future__ import annotations

import json

import pytest
from jinja2 import UndefinedError
from pydantic import ValidationError

from theragraph.agent.bindings import BindingStore, resolve_arguments
from theragraph.agent.llm import (
    MODEL_PRICING,
    DeterministicBackend,
    ModelUsage,
    build_llm,
    estimate_cost_usd,
)
from theragraph.agent.planner import (
    MAX_PLAN_STEPS,
    PlanStep,
    ResearchPlan,
    Synthesis,
    detect_intents,
    deterministic_plan,
    extract_identifiers,
    synthesise_from_findings,
)
from theragraph.agent.prompts import PROMPT_TEMPLATES, all_template_ids, render, template_hash
from theragraph.agent.tools import TOOL_REGISTRY, ToolResult, tool_schemas
from theragraph.agent.validator import (
    assess_readiness,
    check_prediction_labelling,
    validate_tool_output,
)
from theragraph.services.reports import Citation, Claim, validate_claims


class TestPlanner:
    def test_standard_question_produces_the_discovery_workflow(self) -> None:
        plan = deterministic_plan("What are the most promising targets for Alzheimer disease?")
        sequence = plan.tool_sequence()
        assert "get_disease_targets" in sequence
        assert "get_target_overview" in sequence
        assert sequence.index("get_disease_targets") < sequence.index("get_target_overview")

    def test_plan_respects_the_step_cap(self) -> None:
        plan = deterministic_plan("Find compounds for Alzheimer disease and rank them")
        assert len(plan.steps) <= MAX_PLAN_STEPS

    def test_cns_question_selects_the_cns_preset(self) -> None:
        assert (
            deterministic_plan("brain penetrant inhibitors for Parkinson disease").ranking_preset
            == "cns_small_molecule"
        )

    def test_non_cns_question_uses_the_general_preset(self) -> None:
        assert (
            deterministic_plan("kinase inhibitors for lung carcinoma").ranking_preset
            == "general_oral_small_molecule"
        )

    def test_caller_parameters_are_authoritative(self) -> None:
        """A UI-resolved target needs no verification search."""
        plan = deterministic_plan(
            "Summarise the evidence", {"target": "BACE1", "disease": "MONDO_0004975"}
        )
        assert "search_entities" not in plan.tool_sequence()
        assert plan.steps[0].arguments["disease"] == "MONDO_0004975"

    def test_symbol_scraped_from_text_is_verified_first(self) -> None:
        """A gene symbol in a sentence may be a typo or not exist; verify it."""
        plan = deterministic_plan("Summarise the evidence linking ZZZFAKE1 to diabetes.")
        assert plan.tool_sequence()[0] == "search_entities"

    def test_compound_question_starts_from_the_compound(self) -> None:
        plan = deterministic_plan("Find compounds similar to CHEMBL25 and predict solubility")
        assert plan.tool_sequence()[0] == "get_compound"
        assert "find_similar_compounds" in plan.tool_sequence()

    def test_identifiers_are_extracted(self) -> None:
        found = extract_identifiers("Does BACE1 (ENSG00000186318) bind CHEMBL25 in MONDO_0004975?")
        assert found["efo_ids"] == ["MONDO_0004975"]
        assert found["chembl_ids"] == ["CHEMBL25"]
        assert found["ensembl_ids"] == ["ENSG00000186318"]
        assert "BACE1" in found["gene_symbols"]

    def test_common_words_are_not_mistaken_for_gene_symbols(self) -> None:
        found = extract_identifiers("WHAT ARE THE BEST TARGETS FOR IC50 AND LOGP?")
        assert found["gene_symbols"] == []

    def test_intents_are_detected(self) -> None:
        assert "evidence_focus" in detect_intents("what genetic evidence supports this target?")
        assert "ranking_focus" in detect_intents("prioritise these candidates")
        assert "property_focus" in detect_intents("predict the solubility and toxicity")

    def test_plan_validates_against_its_schema(self) -> None:
        plan = deterministic_plan("targets for Alzheimer disease")
        assert ResearchPlan.model_validate(plan.model_dump(mode="json"))

    def test_plan_step_rejects_an_out_of_range_position(self) -> None:
        with pytest.raises(ValidationError):
            PlanStep(step=0, tool="search_entities", purpose="x")

    def test_every_planned_tool_exists(self) -> None:
        """A plan naming a tool that does not exist would fail at execution."""
        for question in (
            "targets for Alzheimer disease",
            "compounds similar to CHEMBL25",
            "genetic evidence for BACE1",
            "prioritise candidates for type 2 diabetes mellitus",
        ):
            for step in deterministic_plan(question).steps:
                assert step.tool in TOOL_REGISTRY, step.tool


class TestBindings:
    def test_placeholders_resolve_from_the_previous_step(self) -> None:
        store = BindingStore()
        store.record_step(
            1,
            "get_disease_targets",
            {
                "disease": {"efo_id": "MONDO_0004975", "name": "AD", "id": "d1"},
                "associated_targets": [
                    {
                        "target": {
                            "symbol": "APP",
                            "target_key": "E1",
                            "id": "t1",
                            "tractability": {},
                        }
                    }
                ],
            },
        )
        resolved = resolve_arguments(
            {"target": "{{prev.top_target}}", "disease": "{{state.disease_efo_id}}"}, store
        )
        assert resolved.ok
        assert resolved.arguments == {"target": "APP", "disease": "MONDO_0004975"}

    def test_tractable_target_is_preferred_over_the_top_scorer(self) -> None:
        """A target with no small-molecule chemistry is a dead end for compound work."""
        store = BindingStore()
        store.record_step(
            1,
            "get_disease_targets",
            {
                "disease": {"efo_id": "X", "name": "X", "id": "d1"},
                "associated_targets": [
                    {
                        "target": {
                            "symbol": "APP",
                            "target_key": "E1",
                            "id": "t1",
                            "tractability": {},
                        }
                    },
                    {
                        "target": {
                            "symbol": "BACE1",
                            "target_key": "E2",
                            "id": "t2",
                            "tractability": {"Small molecule": "Clinical Precedence"},
                        }
                    },
                ],
            },
        )
        assert resolve_arguments({"t": "{{prev.top_target}}"}, store).arguments["t"] == "BACE1"

    def test_unresolvable_placeholder_is_reported_not_passed_through(self) -> None:
        """Calling a tool with the literal string "{{state.x}}" would be a silent bug."""
        resolved = resolve_arguments({"compounds": "{{state.missing}}"}, BindingStore())
        assert not resolved.ok
        assert "compounds=state.missing" in resolved.unresolved
        assert "compounds" not in resolved.arguments

    def test_list_placeholder_resolves_to_a_list(self) -> None:
        store = BindingStore()
        store.record_step(
            2,
            "get_target_overview",
            {
                "target": {"symbol": "BACE1", "target_key": "E2", "id": "t2"},
                "known_compounds": [
                    {"compound_id": "c1", "smiles": "CCO"},
                    {"compound_id": "c2", "smiles": "CCC"},
                ],
            },
        )
        resolved = resolve_arguments({"compounds": "{{state.candidate_compound_ids}}"}, store)
        assert resolved.arguments["compounds"] == ["c1", "c2"]

    def test_bindings_are_keyed_by_plan_step(self) -> None:
        store = BindingStore()
        store.record_step(3, "get_compound", {"compound": {"id": "c9", "smiles": "CCO"}})
        assert resolve_arguments({"s": "{{step_3.smiles}}"}, store).arguments["s"] == "CCO"

    def test_literal_arguments_pass_through_untouched(self) -> None:
        resolved = resolve_arguments({"limit": 10, "threshold": 0.7}, BindingStore())
        assert resolved.arguments == {"limit": 10, "threshold": 0.7}

    def test_interpolated_string_is_substituted(self) -> None:
        store = BindingStore()
        store.record_step(1, "get_compound", {"compound": {"id": "c1", "smiles": "CCO"}})
        resolved = resolve_arguments({"note": "smiles is {{step_1.smiles}}"}, store)
        assert resolved.arguments["note"] == "smiles is CCO"


class TestToolContract:
    def test_every_tool_schema_is_self_contained_and_strict(self) -> None:
        """A `$ref` in a tool schema would not resolve on the model's side."""
        for schema in tool_schemas(include_mutating=True):
            serialised = json.dumps(schema)
            assert "$ref" not in serialised
            assert "$defs" not in serialised
            assert schema["strict"] is True
            assert schema["input_schema"]["additionalProperties"] is False

    def test_write_tools_are_hidden_by_default(self) -> None:
        exposed = {schema["name"] for schema in tool_schemas()}
        assert "ingest_disease" not in exposed
        assert "ingest_disease" in {
            schema["name"] for schema in tool_schemas(include_mutating=True)
        }

    def test_every_tool_documents_itself(self) -> None:
        """The description is the model's only guide to when a tool applies."""
        for spec in TOOL_REGISTRY.values():
            assert len(spec.description) > 60, spec.name
            assert spec.category

    def test_model_payload_is_bounded(self) -> None:
        """A 200-row result must not consume the context window."""
        result = ToolResult(
            data={"hits": [{"id": index} for index in range(500)]},
            summary="many hits",
        )
        payload = result.to_model_payload(max_items=10)
        assert len(payload["data"]["hits"]) == 11  # 10 items plus the omission notice
        assert "omitted" in str(payload["data"]["hits"][-1])


class TestValidator:
    def test_missing_summary_fails_validation(self) -> None:
        verdict = validate_tool_output("x", ToolResult(data={}, summary=""))
        assert not verdict.ok
        assert any("summary" in issue for issue in verdict.issues)

    def test_evidence_without_provenance_fails(self) -> None:
        """Evidence the UI cannot attribute is a bug in the tool, not a model error."""
        verdict = validate_tool_output(
            "get_target_evidence",
            ToolResult(
                data={"items": [{"id": "e1", "evidence_type": "literature", "provenance": {}}]},
                summary="one record",
            ),
        )
        assert not verdict.ok
        assert verdict.checks["records_missing_provenance"] == 1

    def test_well_formed_evidence_passes(self) -> None:
        verdict = validate_tool_output(
            "get_target_evidence",
            ToolResult(
                data={
                    "items": [
                        {
                            "id": "e1",
                            "evidence_type": "literature",
                            "provenance": {
                                "source": "open_targets",
                                "source_identifier": "x",
                                "retrieved_at": "2026-06-01T00:00:00Z",
                            },
                        }
                    ]
                },
                summary="one record",
                citations=["open_targets:x"],
            ),
        )
        assert verdict.ok

    def test_malformed_citation_key_is_caught(self) -> None:
        """A malformed key offered to the model guarantees a rejected claim later."""
        verdict = validate_tool_output(
            "x", ToolResult(data={}, summary="s", citations=["no-colon-here"])
        )
        assert not verdict.ok
        assert verdict.checks["malformed_citation_keys"] == 1

    def test_null_result_fails_loudly(self) -> None:
        assert not validate_tool_output("x", None).ok

    def test_readiness_requires_retrieval(self) -> None:
        readiness = assess_readiness(
            record_ids={}, successful_tools=0, failed_tools=2, empty_tools=0
        )
        assert not readiness.ready
        assert "no tool call succeeded" in readiness.reasons

    def test_readiness_passes_with_records(self) -> None:
        readiness = assess_readiness(
            record_ids={"evidence": ["e1", "e2"]}, successful_tools=3, failed_tools=0, empty_tools=1
        )
        assert readiness.ready
        assert readiness.counts["evidence"] == 2

    def test_prediction_labelling_heuristic(self) -> None:
        assert check_prediction_labelling("solubility is high", ["solubility"])
        assert not check_prediction_labelling("predicted solubility is high", ["solubility"])


class TestClaimValidation:
    def test_unresolvable_citation_removes_the_claim(self) -> None:
        """The single most important guardrail: a fabricated reference cannot ship."""
        index = {
            "open_targets:real": Citation(
                key="open_targets:real",
                source="open_targets",
                source_identifier="real",
                title="t",
                url=None,
                kind="genetic_association",
            )
        }
        claims = [
            Claim(text="a supported fact", kind="retrieved_fact", citations=["open_targets:real"]),
            Claim(text="an invented fact", kind="retrieved_fact", citations=["pmid:99999999"]),
        ]
        accepted, report = validate_claims(claims, index)
        assert [claim.text for claim in accepted] == ["a supported fact"]
        assert report.rejection_rate == 0.5
        assert "pmid:99999999" in report.unknown_citations

    def test_uncited_factual_claim_is_removed(self) -> None:
        accepted, report = validate_claims(
            [Claim(text="an uncited assertion", kind="retrieved_fact")], {}
        )
        assert accepted == []
        assert report.rejected_claims[0]["reason"] == "no citations provided"

    def test_prediction_claims_are_accepted_but_labelled(self) -> None:
        """Predictions are backed by persisted prediction rows, not evidence citations."""
        accepted, _ = validate_claims(
            [Claim(text="predicted logS is -3.2", kind="model_prediction")], {}
        )
        assert len(accepted) == 1
        assert "not an experimental measurement" in accepted[0].validation_note

    def test_evidence_id_is_accepted_as_a_citation_key(self) -> None:
        citation = Citation(
            key="open_targets:x",
            source="open_targets",
            source_identifier="x",
            title="t",
            url=None,
            kind="literature",
        )
        accepted, _ = validate_claims(
            [
                Claim(
                    text="a fact backed by an internal id",
                    kind="interpretation",
                    citations=["evidence-row-1"],
                )
            ],
            {"open_targets:x": citation, "evidence-row-1": citation},
        )
        assert len(accepted) == 1


class TestDeterministicSynthesis:
    def test_claims_cite_the_tool_that_produced_them(self) -> None:
        payload = synthesise_from_findings(
            "What targets matter?",
            [
                {
                    "tool": "get_disease_targets",
                    "summary": "Alzheimer disease: 10 associated targets. Top: APP 0.872",
                    "citations": ["open_targets:a1", "open_targets:a2"],
                    "status": "succeeded",
                },
                {
                    "tool": "predict_properties",
                    "summary": "8 predictions across 2 endpoints",
                    "citations": [],
                    "status": "succeeded",
                },
            ],
        )
        synthesis = Synthesis.model_validate(payload)
        by_kind = {claim.kind for claim in synthesis.claims}
        assert "retrieved_fact" in by_kind
        assert "model_prediction" in by_kind
        fact = next(claim for claim in synthesis.claims if claim.kind == "retrieved_fact")
        assert fact.citations == ["open_targets:a1", "open_targets:a2"]

    def test_failed_tools_are_counted_not_claimed(self) -> None:
        payload = synthesise_from_findings(
            "Anything?",
            [
                {
                    "tool": "x",
                    "summary": "FAILED: upstream down",
                    "citations": [],
                    "status": "failed",
                }
            ],
        )
        assert payload["claims"] == []
        assert "1 of which failed" in payload["summary"]

    def test_synthesis_states_it_did_not_interpret(self) -> None:
        payload = synthesise_from_findings("q", [])
        assert "deterministic backend" in payload["summary"].lower()
        assert Synthesis.model_validate(payload)


class TestPrompts:
    def test_templates_render_with_required_inputs(self) -> None:
        rendered = render(
            "agent.system.v3", max_steps=8, corpus_summary="1 disease", allow_ingestion=False
        )
        assert "Absolute rules" in rendered
        assert "DISABLED" in rendered

    def test_missing_input_raises_rather_than_rendering_blank(self) -> None:
        """A silently blank prompt section is worse than a loud failure."""
        with pytest.raises(UndefinedError):
            render("agent.system.v3", max_steps=8)

    def test_template_hash_is_stable(self) -> None:
        assert template_hash("agent.system.v3") == template_hash("agent.system.v3")

    def test_every_registered_template_is_listed(self) -> None:
        assert set(all_template_ids()) == set(PROMPT_TEMPLATES)

    def test_system_prompt_states_the_non_negotiables(self) -> None:
        rendered = render("agent.system.v3", max_steps=8, corpus_summary="x", allow_ingestion=False)
        for requirement in (
            "Never invent experimental results",
            "Cite retrieved facts",
            "Distinguish measurements from predictions",
            "Never claim a target is validated",
        ):
            assert requirement in rendered


class TestCostAccounting:
    def test_pricing_is_applied_per_million_tokens(self) -> None:
        cost = estimate_cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(5.0)

    def test_cached_input_is_cheaper(self) -> None:
        uncached = estimate_cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=0)
        cached = estimate_cost_usd(
            "claude-opus-5", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000
        )
        assert cached < uncached

    def test_unknown_model_returns_zero_rather_than_a_guess(self) -> None:
        """A silently wrong cost figure is worse than an obviously absent one."""
        assert estimate_cost_usd("some-other-model", input_tokens=1000, output_tokens=1000) == 0.0

    def test_usage_accumulates(self) -> None:
        total = ModelUsage(model="claude-opus-5")
        total.add(ModelUsage(model="claude-opus-5", input_tokens=100, output_tokens=50, calls=1))
        total.add(ModelUsage(model="claude-opus-5", input_tokens=200, output_tokens=25, calls=1))
        assert total.input_tokens == 300
        assert total.calls == 2
        assert total.cost_usd > 0

    def test_known_models_are_priced(self) -> None:
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"):
            assert model in MODEL_PRICING


class TestBackendSelection:
    def test_deterministic_backend_is_chosen_without_a_key(self, settings) -> None:
        """The product must run for anyone who clones the repository."""
        backend = build_llm(settings)
        assert isinstance(backend, DeterministicBackend)

    async def test_deterministic_backend_returns_a_valid_plan(self, settings) -> None:
        backend = DeterministicBackend(settings)
        response = await backend.structured(
            system="s",
            user="## Question\nWhat targets matter for Alzheimer disease?\n",
            schema_model=ResearchPlan,
        )
        assert isinstance(response.parsed, ResearchPlan)
        assert response.usage.cost_usd == 0.0
