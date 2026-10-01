"""The research workflow.

    question
      -> plan            (structured, stored, replayable)
      -> execute         (plan-driven tool calls, then optional adaptive tool calls)
      -> validate        (tool output checks + retrieval readiness)
      -> synthesise      (model writes claims; nothing factual originates here)
      -> report          (templated facts + validated claims + citations)

Each phase writes an ``agent_execution`` row and each tool call writes a ``tool_call``
row, inside the same transaction as the domain data, so the trace can never disagree
with what happened. Nothing in this module can put a fact into the report that did not
come from a tool result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from theragraph.agent import prompts
from theragraph.agent.bindings import BindingStore, resolve_arguments
from theragraph.agent.llm import LLMBackend, ModelUsage, build_llm
from theragraph.agent.planner import (
    MAX_PLAN_STEPS,
    PlanStep,
    ResearchPlan,
    Synthesis,
    corpus_brief,
    deterministic_plan,
)
from theragraph.agent.tools import TOOL_REGISTRY, execute_tool, tool_schemas
from theragraph.agent.validator import (
    assess_readiness,
    check_prediction_labelling,
    validate_tool_output,
)
from theragraph.core.clock import Stopwatch, utcnow
from theragraph.core.errors import AgentError, NotFoundError
from theragraph.core.logging import get_logger, log_context
from theragraph.core.metrics import AGENT_RUN_LATENCY, AGENT_RUNS
from theragraph.core.telemetry import span
from theragraph.db.models import ResearchRun
from theragraph.db.repositories import runs as runs_repo
from theragraph.services import reports as reports_service
from theragraph.services import runs as runs_service
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

#: Extra tool calls the model may make after the plan, when adapting to what it found.
MAX_ADAPTIVE_STEPS = 6


@dataclass
class ExecutionAccumulator:
    """Everything gathered while executing a plan."""

    bindings: BindingStore = field(default_factory=BindingStore)
    state: dict[str, Any] = field(default_factory=dict)
    record_ids: dict[str, list[str]] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    """Structured per-tool findings: ``{tool, summary, citations, status, purpose}``.
    Structured rather than pre-rendered text so the deterministic synthesiser can cite
    the records each individual tool returned."""
    warnings: list[str] = field(default_factory=list)
    succeeded: int = 0
    failed: int = 0
    empty: int = 0
    skipped: list[dict[str, Any]] = field(default_factory=list)
    usage: ModelUsage = field(default_factory=ModelUsage)
    validations: list[dict[str, Any]] = field(default_factory=list)

    def absorb_records(self, record_ids: dict[str, list[str]]) -> None:
        for entity, ids in record_ids.items():
            existing = self.record_ids.setdefault(entity, [])
            for identifier in ids:
                if identifier and identifier not in existing:
                    existing.append(identifier)


async def start_research(
    context: ServiceContext,
    *,
    query: str,
    parameters: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    seed: int = 1337,
    execute_inline: bool = False,
    queue: Any = None,
) -> dict[str, Any]:
    """Create a run and either execute it now or enqueue it.

    Research runs take tens of seconds (and spend LLM tokens), so the default is to
    enqueue and return the run id immediately; the UI polls the run. Inline execution is
    for the CLI, tests and the eval suite.
    """
    from theragraph.ranking.config import preset

    parameters = dict(parameters or {})
    ranking_preset = parameters.get("ranking_preset")
    ranking_config = preset(ranking_preset).to_dict() if ranking_preset else {}

    run = await runs_service.create_run(
        context,
        query=query,
        workflow="research_agent",
        parameters=parameters,
        ranking_config=ranking_config,
        prompt_template_ids=prompts.all_template_ids(),
        seed=seed,
        idempotency_key=idempotency_key,
    )
    await context.session.commit()

    if run.is_terminal:
        # An idempotency key matched a run that already finished. Re-executing it would
        # spend model tokens again and overwrite its trace, so return the original.
        log.info("agent.idempotent_replay_suppressed", run_id=run.id, status=run.status)
        return {
            "run_id": run.id,
            "status": run.status,
            "inline": execute_inline,
            "idempotent_hit": True,
            "runtime_ms": run.runtime_ms,
            "usage": {
                "input_tokens": run.input_tokens,
                "output_tokens": run.output_tokens,
                "cost_usd": round(run.cost_usd, 6),
            },
        }

    if execute_inline:
        result = await execute_run(context, run_id=run.id)
        return {"run_id": run.id, "status": result.get("status"), "inline": True, **result}

    if queue is not None:
        job = await queue.enqueue(
            context.session,
            task="run_research_workflow",
            payload={"run_id": run.id},
            idempotency_key=f"research:{run.id}",
            priority=50,
        )
        # Record the job link, and *only* the job link. The run's status belongs to
        # whoever executes it: the inline queue runs the workflow inside ``enqueue``
        # above, so writing "queued" here would overwrite the terminal status it just
        # set -- leaving a finished run that reports itself as still waiting.
        await runs_repo.update_run(context.session, run, job_id=job.id)
        await context.session.commit()
        return {
            "run_id": run.id,
            "status": run.status,
            "job_id": job.id,
            "inline": False,
        }

    return {"run_id": run.id, "status": run.status, "inline": False}


async def execute_run(
    context: ServiceContext,
    *,
    run_id: str,
    heartbeat: Any = None,
    llm: LLMBackend | None = None,
) -> dict[str, Any]:
    """Execute the full workflow for an existing run."""
    run = await runs_repo.get_run(context.session, run_id, with_trace=False)
    if run is None:
        raise NotFoundError(f"run {run_id!r} not found")
    if run.is_terminal:
        raise AgentError(
            f"run {run_id} already finished with status {run.status!r}",
            details={"status": run.status},
        )

    backend = llm or build_llm(context.settings)
    stopwatch = Stopwatch()
    accumulator = ExecutionAccumulator()
    step_index = 0

    with log_context(run_id=run.id), stopwatch:
        try:
            await runs_service.mark_running(context, run, "planning")
            await context.session.commit()

            brief = await corpus_brief(context)
            # The ingestion tools themselves fall back to fixture replay when the
            # deployment disables live ingestion, so this flag cannot reach the network.
            allow_ingestion = bool((run.parameters or {}).get("allow_ingestion", False))

            # ------------------------------------------------------------- plan
            plan, plan_usage = await _plan_phase(
                context, run, backend, brief=brief, step_index=step_index
            )
            accumulator.usage.add(plan_usage)
            step_index += 1
            await context.session.commit()
            if heartbeat:
                await heartbeat()

            # ---------------------------------------------------------- execute
            await runs_service.mark_running(context, run, "running")
            step_index = await _execute_plan(
                context,
                run,
                plan,
                accumulator,
                start_step=step_index,
                allow_ingestion=allow_ingestion,
                heartbeat=heartbeat,
            )
            await context.session.commit()

            # Adaptive phase: let the model react to what the plan actually found.
            if backend.name == "anthropic":
                step_index = await _adaptive_phase(
                    context,
                    run,
                    backend,
                    accumulator,
                    brief=brief,
                    start_step=step_index,
                    allow_ingestion=allow_ingestion,
                    heartbeat=heartbeat,
                )
                await context.session.commit()

            # --------------------------------------------------------- validate
            await runs_service.mark_running(context, run, "validating")
            readiness = assess_readiness(
                record_ids=accumulator.record_ids,
                successful_tools=accumulator.succeeded,
                failed_tools=accumulator.failed,
                empty_tools=accumulator.empty,
            )
            await runs_service.record_execution(
                context,
                run=run,
                step_index=step_index,
                phase="validate",
                status="succeeded" if readiness.ready else "warning",
                rationale=(
                    "All tool outputs carried provenance and enough records were "
                    "retrieved to answer the question."
                    if readiness.ready
                    else "Retrieval was insufficient: " + "; ".join(readiness.reasons)
                ),
                output_state={
                    "readiness": readiness.to_dict(),
                    "tool_validations": accumulator.validations,
                },
            )
            step_index += 1
            if not readiness.ready:
                accumulator.warnings.extend(readiness.reasons)
            await context.session.commit()

            # -------------------------------------------------------- synthesise
            await runs_service.mark_running(context, run, "reporting")
            synthesis, synthesis_usage = await _synthesis_phase(
                context, run, backend, accumulator, step_index=step_index
            )
            accumulator.usage.add(synthesis_usage)
            step_index += 1

            await runs_service.update_state(
                context,
                run=run,
                patch={
                    **accumulator.state,
                    "plan": plan.model_dump(mode="json"),
                    "readiness": readiness.to_dict(),
                    "open_questions": synthesis.open_questions,
                    "limitations": accumulator.warnings[:10],
                    "skipped_steps": accumulator.skipped,
                    "llm_usage": accumulator.usage.to_dict(),
                },
            )
            await context.session.commit()

            # ------------------------------------------------------------ report
            claims = [
                reports_service.Claim(text=claim.text, kind=claim.kind, citations=claim.citations)
                for claim in synthesis.claims
            ]
            report = await reports_service.generate_report(
                context,
                run_id=run.id,
                summary=synthesis.summary,
                claims=claims,
            )
            await context.session.commit()

            labelling_warnings = check_prediction_labelling(
                synthesis.summary,
                sorted(
                    {
                        prediction.get("endpoint", "")
                        for prediction in (accumulator.state.get("predictions") or [])
                    }
                    - {""}
                ),
            )

            run.input_tokens = accumulator.usage.input_tokens
            run.output_tokens = accumulator.usage.output_tokens
            run.cost_usd = accumulator.usage.cost_usd
            await runs_service.finish_run(
                context,
                run=run,
                status="succeeded",
                warnings=[
                    {"kind": "execution", "message": message}
                    for message in [*accumulator.warnings[:10], *labelling_warnings]
                ],
            )
            await context.session.commit()

            AGENT_RUNS.labels(status="succeeded").inc()
            AGENT_RUN_LATENCY.observe(stopwatch.elapsed_ms / 1000)

            return {
                "run_id": run.id,
                "status": "succeeded",
                "plan": plan.model_dump(mode="json"),
                "tool_calls": accumulator.succeeded + accumulator.failed,
                "records_retrieved": {
                    entity: len(ids) for entity, ids in accumulator.record_ids.items()
                },
                "readiness": readiness.to_dict(),
                "report": {
                    "title": report["title"],
                    "summary": report["summary"],
                    "validation": report["validation"],
                    "content_hash": report["content_hash"],
                    "n_citations": len(report["citations"]),
                },
                "usage": accumulator.usage.to_dict(),
                "runtime_ms": round(stopwatch.elapsed_ms, 2),
                "backend": backend.name,
            }

        except Exception as exc:
            await context.session.rollback()
            failed = await runs_repo.get_run(context.session, run_id, with_trace=False)
            if failed is not None:
                await runs_service.finish_run(
                    context,
                    run=failed,
                    status="failed",
                    errors=[
                        {
                            "phase": failed.status,
                            "error": f"{type(exc).__name__}: {exc}"[:2000],
                            "at": utcnow().isoformat(),
                        }
                    ],
                )
                await context.session.commit()
            AGENT_RUNS.labels(status="failed").inc()
            log.error("agent.run_failed", run_id=run_id, error=str(exc), exc_info=True)
            raise
        finally:
            if llm is None:
                await backend.aclose()


# ------------------------------------------------------------------ plan phase
async def _plan_phase(
    context: ServiceContext,
    run: ResearchRun,
    backend: LLMBackend,
    *,
    brief: str,
    step_index: int,
) -> tuple[ResearchPlan, ModelUsage]:
    tool_summaries = [
        {
            "name": spec.name,
            "category": spec.category,
            "description": spec.description,
        }
        for spec in sorted(TOOL_REGISTRY.values(), key=lambda item: item.name)
        if not spec.mutates or (run.parameters or {}).get("allow_ingestion")
    ]
    system = prompts.render(
        prompts.SYSTEM_TEMPLATE_ID,
        max_steps=MAX_PLAN_STEPS,
        corpus_summary=brief,
        allow_ingestion=bool((run.parameters or {}).get("allow_ingestion")),
    )
    user = prompts.render(
        prompts.PLANNER_TEMPLATE_ID,
        query=run.query,
        parameters=json.dumps(run.parameters or {}, indent=2) if run.parameters else "",
        corpus_summary=brief,
        tools=tool_summaries,
        max_steps=MAX_PLAN_STEPS,
    )
    await runs_service.record_prompt(
        context, run=run, template_id=prompts.SYSTEM_TEMPLATE_ID, rendered=system
    )
    await runs_service.record_prompt(
        context, run=run, template_id=prompts.PLANNER_TEMPLATE_ID, rendered=user, role="user"
    )

    if backend.name == "deterministic":
        # Call the deterministic planner directly rather than round-tripping through the
        # rendered prompt: the prompt text loses the run's structured parameters (an
        # explicit target or disease), and those are authoritative.
        from theragraph.agent.llm import LLMResponse, ModelUsage

        response = LLMResponse(
            parsed=deterministic_plan(run.query, run.parameters or {}),
            stop_reason="end_turn",
            usage=ModelUsage(model="deterministic-planner-v1", calls=1),
            backend="deterministic",
        )
    else:
        with span("agent.plan", run_id=run.id):
            response = await backend.structured(system=system, user=user, schema_model=ResearchPlan)

    if response.refused or response.parsed is None:
        # A refusal or an unusable plan must not end the run: fall back to the
        # deterministic planner, which produces a valid plan from the question alone.
        log.warning(
            "agent.plan_fallback",
            run_id=run.id,
            refused=response.refused,
            category=response.refusal_category,
        )
        plan = deterministic_plan(run.query, run.parameters or {})
        rationale = (
            "The language model did not return a usable plan"
            + (f" (refusal: {response.refusal_category})" if response.refusal_category else "")
            + "; the deterministic planner produced the standard discovery workflow."
        )
    else:
        plan = response.parsed
        rationale = plan.strategy

    plan = _sanitize_plan(plan, allow_ingestion=bool((run.parameters or {}).get("allow_ingestion")))

    await runs_repo.update_run(context.session, run, plan=plan.model_dump(mode="json"))
    await runs_service.record_execution(
        context,
        run=run,
        step_index=step_index,
        phase="plan",
        status="succeeded",
        model=response.usage.model or backend.name,
        prompt_template_id=prompts.PLANNER_TEMPLATE_ID,
        rationale=rationale,
        output_state={
            "objective": plan.objective,
            "tool_sequence": plan.tool_sequence(),
            "ranking_preset": plan.ranking_preset,
            "open_questions": plan.open_questions,
        },
        latency_ms=response.latency_ms,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=response.usage.cost_usd,
    )
    log.info(
        "agent.planned",
        run_id=run.id,
        steps=len(plan.steps),
        tools=plan.tool_sequence(),
        backend=backend.name,
    )
    return plan, response.usage


def _sanitize_plan(plan: ResearchPlan, *, allow_ingestion: bool) -> ResearchPlan:
    """Drop steps naming unknown or disallowed tools, and renumber.

    A plan referencing a hallucinated tool is a real failure mode. Rather than erroring
    the run, the invalid steps are removed and the remainder renumbered, so execution
    proceeds with whatever was valid.
    """
    valid: list[PlanStep] = []
    for step in sorted(plan.steps, key=lambda item: item.step):
        spec = TOOL_REGISTRY.get(step.tool)
        if spec is None:
            log.warning("agent.plan_unknown_tool", tool=step.tool)
            continue
        if spec.mutates and not allow_ingestion:
            log.info("agent.plan_dropped_mutating_step", tool=step.tool)
            continue
        valid.append(step)

    if not valid:
        return deterministic_plan(plan.objective)

    renumbered = [
        PlanStep(
            step=index,
            tool=step.tool,
            purpose=step.purpose,
            arguments=step.arguments,
            depends_on_previous=step.depends_on_previous,
            optional=step.optional,
        )
        for index, step in enumerate(valid[:MAX_PLAN_STEPS], start=1)
    ]
    return ResearchPlan(
        objective=plan.objective,
        strategy=plan.strategy,
        steps=renumbered,
        open_questions=plan.open_questions,
        ranking_preset=plan.ranking_preset,
    )


# --------------------------------------------------------------- execute phase
async def _execute_plan(
    context: ServiceContext,
    run: ResearchRun,
    plan: ResearchPlan,
    accumulator: ExecutionAccumulator,
    *,
    start_step: int,
    allow_ingestion: bool,
    heartbeat: Any = None,
) -> int:
    step_index = start_step

    for plan_step in sorted(plan.steps, key=lambda item: item.step):
        resolution = resolve_arguments(plan_step.arguments, accumulator.bindings)

        # An unresolvable placeholder only blocks the step when it fills a *required*
        # argument. Optional context (a disease for a target lookup, say) is simply
        # dropped: the tool's own schema decides what it can do without.
        if not resolution.ok:
            spec = TOOL_REGISTRY.get(plan_step.tool)
            required_fields = (
                {
                    name
                    for name, field in spec.input_model.model_fields.items()
                    if field.is_required()
                }
                if spec is not None
                else set()
            )
            unresolved_names = {item.split("=", 1)[0] for item in resolution.unresolved}
            if not (unresolved_names & required_fields):
                accumulator.warnings.append(
                    f"step {plan_step.step} ({plan_step.tool}) ran without optional "
                    f"argument(s) {sorted(unresolved_names)}"
                )
                resolution.unresolved = []

        if not resolution.ok:
            reason = (
                f"step {plan_step.step} ({plan_step.tool}) skipped: could not resolve "
                f"{', '.join(resolution.unresolved)} from earlier results"
            )
            accumulator.skipped.append(
                {
                    "step": plan_step.step,
                    "tool": plan_step.tool,
                    "unresolved": resolution.unresolved,
                    "optional": plan_step.optional,
                }
            )
            accumulator.warnings.append(reason)
            log.info("agent.step_skipped", run_id=run.id, tool=plan_step.tool)
            await runs_service.record_tool_call(
                context,
                run=run,
                step_index=step_index,
                tool_name=plan_step.tool,
                arguments=plan_step.arguments,
                result=None,
                result_summary=reason,
                record_ids={},
                citations=[],
                status="skipped",
                latency_ms=0.0,
                error=reason,
                error_code="unresolved_arguments",
                tool_version="1",
            )
            step_index += 1
            continue

        execution = await execute_tool(
            context,
            plan_step.tool,
            resolution.arguments,
            allow_mutating=allow_ingestion,
        )
        step_index = await _record_execution_result(
            context,
            run,
            accumulator,
            execution,
            step_index,
            purpose=plan_step.purpose,
            # Bindings are keyed by the *plan* step number, not the trace step index:
            # the plan's `{{step_2.x}}` placeholders refer to plan positions, and the
            # trace index also counts the planning and validation phases.
            binding_key=plan_step.step,
        )
        if heartbeat:
            await heartbeat()

    return step_index


async def _record_execution_result(
    context: ServiceContext,
    run: ResearchRun,
    accumulator: ExecutionAccumulator,
    execution: Any,
    step_index: int,
    *,
    purpose: str | None = None,
    binding_key: int | None = None,
) -> int:
    """Persist one tool call and fold its output into the accumulator."""
    verdict = validate_tool_output(execution.name, execution.result)
    accumulator.validations.append(verdict.to_dict())

    if execution.succeeded and execution.result is not None:
        result = execution.result
        accumulator.succeeded += 1
        if result.is_empty:
            accumulator.empty += 1
        accumulator.absorb_records(result.record_ids)
        accumulator.citations.extend(result.citations)
        accumulator.warnings.extend(result.warnings)
        accumulator.bindings.record_step(
            binding_key if binding_key is not None else step_index,
            execution.name,
            result.data,
        )
        if result.state_patch:
            accumulator.state.update(result.state_patch)
            accumulator.bindings.merge_state(result.state_patch)
        accumulator.findings.append(
            {
                "tool": execution.name,
                "summary": result.summary,
                "citations": result.citations[:5],
                "status": "succeeded",
                "purpose": purpose,
                "is_empty": result.is_empty,
            }
        )
        await runs_service.record_tool_call(
            context,
            run=run,
            step_index=step_index,
            tool_name=execution.name,
            arguments=execution.arguments,
            result=result.to_model_payload(max_items=12),
            result_summary=result.summary,
            record_ids=result.record_ids,
            citations=result.citations[:80],
            status="succeeded",
            latency_ms=execution.latency_ms,
            validation=verdict.to_dict(),
            tool_version="1",
        )
    else:
        accumulator.failed += 1
        accumulator.warnings.append(
            f"{execution.name} failed: {execution.error or 'unknown error'}"
        )
        accumulator.findings.append(
            {
                "tool": execution.name,
                "summary": f"FAILED: {execution.error or 'unknown error'}",
                "citations": [],
                "status": execution.status,
                "purpose": purpose,
            }
        )
        await runs_service.record_tool_call(
            context,
            run=run,
            step_index=step_index,
            tool_name=execution.name,
            arguments=execution.arguments,
            result=None,
            result_summary=execution.error,
            record_ids={},
            citations=[],
            status=execution.status,
            latency_ms=execution.latency_ms,
            error=execution.error,
            error_code=execution.error_code,
            validation=verdict.to_dict(),
            tool_version="1",
        )

    return step_index + 1


# -------------------------------------------------------------- adaptive phase
async def _adaptive_phase(
    context: ServiceContext,
    run: ResearchRun,
    backend: LLMBackend,
    accumulator: ExecutionAccumulator,
    *,
    brief: str,
    start_step: int,
    allow_ingestion: bool,
    heartbeat: Any = None,
) -> int:
    """Let the model make additional tool calls in response to the plan's results.

    This is where the agent earns its name: the plan is a hypothesis about what to do,
    and the results frequently contradict it (a target with no chemistry, a similarity
    search that returns nothing). The loop is bounded, every call is validated and
    persisted exactly like a planned one, and repeated identical calls are refused.
    """
    step_index = start_step
    system = prompts.render(
        prompts.SYSTEM_TEMPLATE_ID,
        max_steps=MAX_ADAPTIVE_STEPS,
        corpus_summary=brief,
        allow_ingestion=allow_ingestion,
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Research question: {run.query}\n\n"
                "The planned tool calls have run. Results:\n\n"
                + _render_findings(accumulator.findings[-20:])
                + "\n\nIf anything essential is missing or a result was empty, make at "
                "most a few more tool calls to fill the gap. If you have what you need, "
                "reply with a one-paragraph statement of what you found and make no "
                "further tool calls."
            ),
        }
    ]
    schemas = tool_schemas(include_mutating=allow_ingestion)
    attempted: set[str] = set()

    for iteration in range(MAX_ADAPTIVE_STEPS):
        response = await backend.converse(system=system, messages=messages, tools=schemas)
        accumulator.usage.add(response.usage)

        if response.refused:
            accumulator.warnings.append(
                f"adaptive phase stopped: model refusal ({response.refusal_category})"
            )
            break

        if not response.wants_tools:
            if response.text:
                await runs_service.record_execution(
                    context,
                    run=run,
                    step_index=step_index,
                    phase="act",
                    status="succeeded",
                    model=response.usage.model,
                    rationale=response.text[:2000],
                    output_state={"adaptive_iterations": iteration},
                    latency_ms=response.latency_ms,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cost_usd=response.usage.cost_usd,
                )
                step_index += 1
            break

        messages.append({"role": "assistant", "content": response.raw_blocks})
        tool_results: list[dict[str, Any]] = []

        for invocation in response.tool_calls:
            signature = f"{invocation.name}:{json.dumps(invocation.arguments, sort_keys=True)}"
            if signature in attempted:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": invocation.id,
                        "content": (
                            "This exact call was already made in this run. Vary the "
                            "arguments or stop."
                        ),
                        "is_error": True,
                    }
                )
                continue
            attempted.add(signature)

            execution = await execute_tool(
                context,
                invocation.name,
                invocation.arguments,
                allow_mutating=allow_ingestion,
            )
            step_index = await _record_execution_result(
                context, run, accumulator, execution, step_index
            )
            if execution.succeeded and execution.result is not None:
                payload = json.dumps(execution.result.to_model_payload(max_items=12), default=str)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": invocation.id,
                        "content": payload[:20000],
                    }
                )
            else:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": invocation.id,
                        "content": f"Error: {execution.error}",
                        "is_error": True,
                    }
                )

        # All results go back in a single user message: splitting them teaches the model
        # to stop making parallel calls.
        messages.append({"role": "user", "content": tool_results})
        if heartbeat:
            await heartbeat()

    return step_index


def _render_findings(findings: list[dict[str, Any]]) -> str:
    """Render structured findings as the bullet list the prompts expect."""
    lines: list[str] = []
    for finding in findings:
        citations = finding.get("citations") or []
        suffix = f" [cites: {', '.join(citations[:3])}]" if citations else ""
        lines.append(f"- [{finding.get('tool')}] {finding.get('summary')}{suffix}")
    return "\n".join(lines)


# ------------------------------------------------------------- synthesis phase
async def _synthesis_phase(
    context: ServiceContext,
    run: ResearchRun,
    backend: LLMBackend,
    accumulator: ExecutionAccumulator,
    *,
    step_index: int,
) -> tuple[Synthesis, ModelUsage]:
    findings = _render_findings(accumulator.findings) or "No tool produced a usable result."
    citation_hint = (
        "\n\nCitation keys available for use:\n"
        + "\n".join(f"- {key}" for key in dict.fromkeys(accumulator.citations[:40]))
        if accumulator.citations
        else "\n\nNo citation keys were returned by the tools in this run."
    )

    system = prompts.render(
        prompts.SYSTEM_TEMPLATE_ID,
        max_steps=MAX_PLAN_STEPS,
        corpus_summary=await corpus_brief(context),
        allow_ingestion=False,
    )
    user = prompts.render(
        prompts.SYNTHESIS_TEMPLATE_ID,
        query=run.query,
        findings=findings + citation_hint,
        warnings=accumulator.warnings[:10],
    )
    await runs_service.record_prompt(
        context,
        run=run,
        template_id=prompts.SYNTHESIS_TEMPLATE_ID,
        rendered=user,
        role="user",
    )

    if backend.name == "deterministic":
        # No round-trip through a rendered prompt: the structured findings are richer
        # than their text rendering, so the deterministic synthesiser reads them
        # directly and cites per tool.
        from theragraph.agent.llm import LLMResponse, ModelUsage
        from theragraph.agent.planner import synthesise_from_findings

        response = LLMResponse(
            parsed=Synthesis.model_validate(
                synthesise_from_findings(
                    run.query, accumulator.findings, warnings=accumulator.warnings
                )
            ),
            stop_reason="end_turn",
            usage=ModelUsage(model="deterministic-planner-v1", calls=1),
            backend="deterministic",
        )
    else:
        with span("agent.synthesise", run_id=run.id):
            response = await backend.structured(system=system, user=user, schema_model=Synthesis)

    if response.refused or response.parsed is None:
        from theragraph.agent.planner import synthesise_from_findings

        log.warning("agent.synthesis_fallback", run_id=run.id, refused=response.refused)
        synthesis = Synthesis.model_validate(
            synthesise_from_findings(run.query, accumulator.findings, warnings=accumulator.warnings)
        )
        rationale = (
            "The language model did not return a usable synthesis; the report was "
            "assembled from tool output alone."
        )
    else:
        synthesis = response.parsed
        rationale = synthesis.rationale or "Synthesis produced from validated tool output."

    await runs_service.record_execution(
        context,
        run=run,
        step_index=step_index,
        phase="report",
        status="succeeded",
        model=response.usage.model or backend.name,
        prompt_template_id=prompts.SYNTHESIS_TEMPLATE_ID,
        rationale=rationale,
        output_state={
            "n_claims": len(synthesis.claims),
            "claim_kinds": {
                kind: sum(1 for claim in synthesis.claims if claim.kind == kind)
                for kind in ("retrieved_fact", "model_prediction", "interpretation")
            },
            "open_questions": synthesis.open_questions,
        },
        latency_ms=response.latency_ms,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=response.usage.cost_usd,
    )
    return synthesis, response.usage
