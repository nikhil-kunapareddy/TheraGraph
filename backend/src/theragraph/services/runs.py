"""Research run service: creation, tracing, comparison and replay.

A run is the unit of reproducibility. What makes replay meaningful is that the run
records everything that could change the answer *at the moment it ran*:

* dataset versions with content hashes (so upstream drift is detectable),
* resolved model versions with artifact hashes (so weights are pinned),
* the ranking configuration,
* the prompt templates and their rendered content,
* the random seed and software/git revision,
* the primary key of every record retrieved.

``reproducibility_hash`` is the digest of that set. Replaying a run and getting the same
hash means the inputs were identical; getting a different hash tells you precisely
*which* input moved, which is more useful than a vague "results differ".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from theragraph.core.clock import utcnow
from theragraph.core.errors import ConflictError, NotFoundError, ValidationError
from theragraph.core.hashing import hash_payload
from theragraph.core.logging import get_logger
from theragraph.db.models import ResearchRun
from theragraph.db.repositories import ml as ml_repo
from theragraph.db.repositories import provenance as prov_repo
from theragraph.db.repositories import runs as runs_repo
from theragraph.ml.registry import code_version, git_sha
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass
class RunPins:
    """The versioned inputs a run is pinned to."""

    dataset_versions: list[dict[str, Any]]
    model_versions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_versions": self.dataset_versions,
            "model_versions": self.model_versions,
        }


async def resolve_pins(context: ServiceContext) -> RunPins:
    """Snapshot the currently active dataset and model versions."""
    versions = await prov_repo.list_dataset_versions(context.session, current_only=True, limit=50)
    models = await ml_repo.list_models(context.session, status="active")
    return RunPins(
        dataset_versions=[
            {
                "id": version.id,
                "label": version.label,
                "source": version.source_id,
                "dataset": version.dataset_name,
                "version": version.version,
                "content_hash": version.content_hash,
                "record_count": version.record_count,
                "retrieved_at": version.retrieved_at.isoformat(),
            }
            for version in versions
        ],
        model_versions=[
            {
                "id": model.id,
                "ref": model.ref,
                "endpoint": model.endpoint,
                "family": model.family,
                "version": model.version,
                "artifact_hash": model.artifact_hash,
                "task_type": model.task_type,
                "scaffold_metrics": (model.metrics or {}).get("scaffold"),
            }
            for model in models
        ],
    )


#: Parameter keys that describe *bookkeeping* rather than scientific inputs. They are
#: excluded from the reproducibility hash so a replay of an unchanged run matches it.
NON_SCIENTIFIC_PARAMETERS = frozenset({"replay_of", "requested_by", "note", "label"})


def compute_reproducibility_hash(
    *,
    query: str,
    parameters: dict[str, Any],
    pins: RunPins,
    ranking_config: dict[str, Any] | None,
    prompt_template_ids: list[str],
    seed: int | None,
    software_version: str,
) -> str:
    """Digest of every input that could change a run's scientific content."""
    scientific_parameters = {
        key: value
        for key, value in (parameters or {}).items()
        if key not in NON_SCIENTIFIC_PARAMETERS
    }
    return hash_payload(
        {
            "query": query.strip().lower(),
            "parameters": scientific_parameters,
            "dataset_versions": sorted(
                f"{item['label']}#{item['content_hash']}" for item in pins.dataset_versions
            ),
            "model_versions": sorted(
                f"{item['ref']}#{item.get('artifact_hash')}" for item in pins.model_versions
            ),
            "ranking_config_hash": (ranking_config or {}).get("config_hash")
            or hash_payload(ranking_config or {}, length=16),
            "prompt_templates": sorted(prompt_template_ids),
            "seed": seed,
            "software_version": software_version,
        },
        length=32,
    )


async def create_run(
    context: ServiceContext,
    *,
    query: str,
    workflow: str = "research_agent",
    parameters: dict[str, Any] | None = None,
    ranking_config: dict[str, Any] | None = None,
    prompt_template_ids: list[str] | None = None,
    seed: int = 1337,
    idempotency_key: str | None = None,
    parent_run_id: str | None = None,
    replay_of_run_id: str | None = None,
    agent_model: str | None = None,
    llm_backend: str | None = None,
) -> ResearchRun:
    """Allocate a run id and pin its inputs.

    An ``idempotency_key`` makes run submission safe to retry: a duplicate submission
    returns the original run instead of starting a second, near-identical workflow that
    would spend LLM tokens twice.
    """
    if not query or not query.strip():
        raise ValidationError("a research query is required")

    if idempotency_key:
        existing = await runs_repo.get_run_by_idempotency_key(context.session, idempotency_key)
        if existing is not None:
            log.info("run.idempotent_hit", run_id=existing.id)
            return existing

    now = utcnow()
    run_id, sequence = await runs_repo.allocate_run_id(context.session, now.year)
    pins = await resolve_pins(context)
    settings = context.settings

    run = await runs_repo.create_run(
        context.session,
        id=run_id,
        sequence=sequence,
        year=now.year,
        query=query.strip(),
        workflow=workflow,
        status="queued",
        principal=context.principal.subject,
        idempotency_key=idempotency_key,
        parameters=parameters or {},
        plan={},
        state={},
        dataset_versions=pins.dataset_versions,
        model_versions=pins.model_versions,
        ranking_config=ranking_config or {},
        prompts=[],
        retrieved_record_ids={},
        software_version=code_version(),
        git_sha=git_sha(),
        random_seed=seed,
        agent_model=agent_model or settings.agent_model,
        llm_backend=llm_backend or settings.agent_llm_backend,
        reproducibility_hash=compute_reproducibility_hash(
            query=query,
            parameters=parameters or {},
            pins=pins,
            ranking_config=ranking_config,
            prompt_template_ids=prompt_template_ids or [],
            seed=seed,
            software_version=code_version(),
        ),
        parent_run_id=parent_run_id,
        replay_of_run_id=replay_of_run_id,
        started_at=now,
        errors=[],
        warnings=[],
    )
    log.info("run.created", run_id=run.id, workflow=workflow, query=query[:120])
    return run


async def mark_running(context: ServiceContext, run: ResearchRun, status: str) -> ResearchRun:
    return await runs_repo.update_run(context.session, run, status=status)


async def record_execution(
    context: ServiceContext,
    *,
    run: ResearchRun,
    step_index: int,
    phase: str,
    status: str = "succeeded",
    model: str | None = None,
    prompt_template_id: str | None = None,
    rationale: str | None = None,
    output_state: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    error: str | None = None,
) -> Any:
    """Persist one phase of the agent workflow."""
    execution = await runs_repo.add_execution(
        context.session,
        research_run_id=run.id,
        step_index=step_index,
        phase=phase,
        status=status,
        model=model,
        prompt_template_id=prompt_template_id,
        rationale=rationale,
        output_state=output_state or {},
        started_at=utcnow(),
        finished_at=utcnow(),
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        error=error,
    )
    run.n_steps = max(run.n_steps, step_index + 1)
    run.input_tokens += input_tokens
    run.output_tokens += output_tokens
    run.cost_usd += cost_usd
    await context.session.flush()
    return execution


async def record_tool_call(
    context: ServiceContext,
    *,
    run: ResearchRun,
    step_index: int,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any] | None,
    result_summary: str | None,
    record_ids: dict[str, Any] | None,
    citations: list[str] | None,
    status: str,
    latency_ms: float | None,
    validation: dict[str, Any] | None = None,
    agent_execution_id: str | None = None,
    error: str | None = None,
    error_code: str | None = None,
    cache_hit: bool = False,
    tool_version: str | None = None,
    call_id: str | None = None,
) -> Any:
    """Persist one tool invocation and accumulate its retrieved ids onto the run."""
    call = await runs_repo.add_tool_call(
        context.session,
        research_run_id=run.id,
        agent_execution_id=agent_execution_id,
        step_index=step_index,
        tool_name=tool_name,
        tool_version=tool_version,
        call_id=call_id,
        arguments=arguments,
        result=result,
        result_summary=result_summary,
        record_ids=record_ids or {},
        citations=citations or [],
        status=status,
        error=error,
        error_code=error_code,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        validation=validation or {},
    )
    run.n_tool_calls += 1

    if record_ids:
        merged = dict(run.retrieved_record_ids or {})
        for entity_type, ids in record_ids.items():
            if not isinstance(ids, list):
                continue
            existing = set(merged.get(entity_type) or [])
            existing.update(str(item) for item in ids)
            merged[entity_type] = sorted(existing)
        run.retrieved_record_ids = merged

    await context.session.flush()
    return call


async def record_prompt(
    context: ServiceContext,
    *,
    run: ResearchRun,
    template_id: str,
    rendered: str,
    role: str = "system",
) -> None:
    """Store a prompt so a replay uses byte-identical instructions."""
    from theragraph.core.security import redact

    prompts = list(run.prompts or [])
    prompts.append(
        {
            "template_id": template_id,
            "role": role,
            "content": redact(rendered, max_len=20000),
            "content_hash": hash_payload(rendered, length=16),
            "recorded_at": utcnow().isoformat(),
        }
    )
    run.prompts = prompts
    await context.session.flush()


async def update_state(
    context: ServiceContext, *, run: ResearchRun, patch: dict[str, Any]
) -> ResearchRun:
    """Merge structured intermediate state onto the run."""
    state = dict(run.state or {})
    state.update(patch)
    run.state = state
    await context.session.flush()
    return run


async def finish_run(
    context: ServiceContext,
    *,
    run: ResearchRun,
    status: str,
    errors: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> ResearchRun:
    if status not in TERMINAL_STATUSES:
        raise ValidationError(f"{status!r} is not a terminal run status")
    if warnings:
        run.warnings = [*(run.warnings or []), *warnings]
    finished = await runs_repo.finish_run(context.session, run, status=status, errors=errors)
    log.info(
        "run.finished",
        run_id=run.id,
        status=status,
        runtime_ms=run.runtime_ms,
        tool_calls=run.n_tool_calls,
        cost_usd=round(run.cost_usd, 6),
    )
    return finished


# ------------------------------------------------------------------- retrieval
def _run_summary(run: ResearchRun) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "query": run.query,
        "workflow": run.workflow,
        "status": run.status,
        "principal": run.principal,
        "created_at": run.created_at.isoformat(),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "runtime_ms": run.runtime_ms,
        "n_steps": run.n_steps,
        "n_tool_calls": run.n_tool_calls,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cost_usd": round(run.cost_usd, 6),
        "reproducibility_hash": run.reproducibility_hash,
        "software_version": run.software_version,
        "git_sha": run.git_sha,
        "agent_model": run.agent_model,
        "llm_backend": run.llm_backend,
        "random_seed": run.random_seed,
        "parent_run_id": run.parent_run_id,
        "replay_of_run_id": run.replay_of_run_id,
        "has_report": run.report is not None,
        "n_errors": len(run.errors or []),
    }


async def get_run(context: ServiceContext, run_id: str) -> dict[str, Any]:
    """The complete, auditable execution trace."""
    run = await runs_repo.get_run(context.session, run_id)
    if run is None:
        raise NotFoundError(f"run {run_id!r} not found")

    return {
        **_run_summary(run),
        "parameters": run.parameters,
        "plan": run.plan,
        "state": run.state,
        "ranking_config": run.ranking_config,
        "pins": {
            "dataset_versions": run.dataset_versions,
            "model_versions": run.model_versions,
        },
        "prompts": run.prompts,
        "retrieved_record_ids": run.retrieved_record_ids,
        "errors": run.errors,
        "warnings": run.warnings,
        "executions": [
            {
                "id": execution.id,
                "step_index": execution.step_index,
                "phase": execution.phase,
                "status": execution.status,
                "model": execution.model,
                "prompt_template_id": execution.prompt_template_id,
                "rationale": execution.rationale,
                "output_state": execution.output_state,
                "latency_ms": execution.latency_ms,
                "input_tokens": execution.input_tokens,
                "output_tokens": execution.output_tokens,
                "cost_usd": round(execution.cost_usd, 6),
                "error": execution.error,
                "tool_call_ids": [call.id for call in execution.tool_calls],
            }
            for execution in run.executions
        ],
        "tool_calls": [
            {
                "id": call.id,
                "step_index": call.step_index,
                "tool_name": call.tool_name,
                "tool_version": call.tool_version,
                "arguments": call.arguments,
                "result": call.result,
                "result_summary": call.result_summary,
                "record_ids": call.record_ids,
                "citations": call.citations,
                "status": call.status,
                "error": call.error,
                "error_code": call.error_code,
                "latency_ms": call.latency_ms,
                "cache_hit": call.cache_hit,
                "validation": call.validation,
                "created_at": call.created_at.isoformat(),
            }
            for call in run.tool_calls
        ],
        "report": (
            {
                "id": run.report.id,
                "title": run.report.title,
                "summary": run.report.summary,
                "markdown": run.report.markdown,
                "sections": run.report.sections,
                "claims": run.report.claims,
                "citations": run.report.citations,
                "limitations": run.report.limitations,
                "validation": run.report.validation,
                "content_hash": run.report.content_hash,
            }
            if run.report is not None
            else None
        ),
    }


async def list_runs(
    context: ServiceContext,
    *,
    status: str | None = None,
    workflow: str | None = None,
    query_contains: str | None = None,
    limit: int = 25,
    offset: int = 0,
) -> dict[str, Any]:
    runs = await runs_repo.list_runs(
        context.session,
        status=status,
        workflow=workflow,
        query_contains=query_contains,
        limit=limit,
        offset=offset,
    )
    return {
        "runs": [_run_summary(run) for run in runs],
        "total": await runs_repo.count_runs(context.session, status=status),
        "limit": limit,
        "offset": offset,
    }


async def compare_runs(context: ServiceContext, run_ids: list[str]) -> dict[str, Any]:
    """Diff two or more runs across inputs, execution and outputs.

    The valuable output is not "these runs differ" but *where*: a changed dataset
    version, a new model version, different parameters, or genuinely different model
    behaviour on identical inputs.
    """
    if len(run_ids) < 2:
        raise ValidationError("comparing runs requires at least two run ids")
    if len(run_ids) > 5:
        raise ValidationError("compare at most five runs at a time")

    runs = []
    for run_id in run_ids:
        run = await runs_repo.get_run(context.session, run_id)
        if run is None:
            raise NotFoundError(f"run {run_id!r} not found")
        runs.append(run)

    def dataset_labels(run: ResearchRun) -> dict[str, str]:
        return {item["label"]: item["content_hash"] for item in (run.dataset_versions or [])}

    def model_refs(run: ResearchRun) -> dict[str, str]:
        return {item["ref"]: item.get("artifact_hash") or "" for item in (run.model_versions or [])}

    baseline = runs[0]
    differences: list[dict[str, Any]] = []
    for run in runs[1:]:
        dataset_diff = _diff_mappings(dataset_labels(baseline), dataset_labels(run))
        model_diff = _diff_mappings(model_refs(baseline), model_refs(run))
        parameter_diff = _diff_mappings(
            _flatten(baseline.parameters or {}), _flatten(run.parameters or {})
        )
        differences.append(
            {
                "baseline_run_id": baseline.id,
                "run_id": run.id,
                "same_reproducibility_hash": baseline.reproducibility_hash
                == run.reproducibility_hash,
                "query_changed": baseline.query.strip().lower() != run.query.strip().lower(),
                "dataset_versions": dataset_diff,
                "model_versions": model_diff,
                "parameters": parameter_diff,
                "ranking_config_changed": (baseline.ranking_config or {}).get("config_hash")
                != (run.ranking_config or {}).get("config_hash"),
                "tool_sequence_changed": _tool_sequence(baseline) != _tool_sequence(run),
                "verdict": _comparison_verdict(
                    baseline, run, dataset_diff, model_diff, parameter_diff
                ),
            }
        )

    return {
        "runs": [
            {
                **_run_summary(run),
                "tool_sequence": _tool_sequence(run),
                "report_hash": run.report.content_hash if run.report else None,
            }
            for run in runs
        ],
        "differences": differences,
        "metrics": {
            "runtime_ms": {run.id: run.runtime_ms for run in runs},
            "tool_calls": {run.id: run.n_tool_calls for run in runs},
            "cost_usd": {run.id: round(run.cost_usd, 6) for run in runs},
            "tokens": {run.id: run.input_tokens + run.output_tokens for run in runs},
        },
    }


def _tool_sequence(run: ResearchRun) -> list[str]:
    return [call.tool_name for call in sorted(run.tool_calls, key=lambda c: c.step_index)]


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = str(value)
    return flat


def _diff_mappings(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    added = sorted(set(right) - set(left))
    removed = sorted(set(left) - set(right))
    changed = {
        key: {"baseline": left[key], "other": right[key]}
        for key in sorted(set(left) & set(right))
        if left[key] != right[key]
    }
    return {
        "identical": not (added or removed or changed),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def _comparison_verdict(
    baseline: ResearchRun,
    other: ResearchRun,
    dataset_diff: dict[str, Any],
    model_diff: dict[str, Any],
    parameter_diff: dict[str, Any],
) -> str:
    if baseline.reproducibility_hash == other.reproducibility_hash:
        return (
            "Identical pinned inputs. Any difference in output came from model "
            "non-determinism, not from the data or configuration."
        )
    reasons = []
    if not dataset_diff["identical"]:
        reasons.append("dataset versions differ")
    if not model_diff["identical"]:
        reasons.append("model versions differ")
    if not parameter_diff["identical"]:
        reasons.append("parameters differ")
    if baseline.query.strip().lower() != other.query.strip().lower():
        reasons.append("the query differs")
    return "Inputs differ: " + ", ".join(reasons) if reasons else "Inputs differ."


async def prepare_replay(
    context: ServiceContext, run_id: str, *, allow_version_drift: bool = False
) -> dict[str, Any]:
    """Build the parameters needed to re-execute a run against its original pins.

    If the pinned dataset or model versions are no longer available, replay is refused
    unless ``allow_version_drift`` is set -- and even then the drift is recorded on the
    new run so the comparison is honest.
    """
    original = await runs_repo.get_run(context.session, run_id)
    if original is None:
        raise NotFoundError(f"run {run_id!r} not found")

    current = await resolve_pins(context)
    current_datasets = {item["label"]: item["content_hash"] for item in current.dataset_versions}
    current_models = {item["ref"]: item.get("artifact_hash") for item in current.model_versions}

    missing_datasets = [
        item
        for item in (original.dataset_versions or [])
        if current_datasets.get(item["label"]) != item["content_hash"]
    ]
    missing_models = [
        item
        for item in (original.model_versions or [])
        if current_models.get(item["ref"]) != item.get("artifact_hash")
    ]

    drift = {
        "dataset_versions_changed": [item["label"] for item in missing_datasets],
        "model_versions_changed": [item["ref"] for item in missing_models],
        "reproducible": not (missing_datasets or missing_models),
    }

    if not drift["reproducible"] and not allow_version_drift:
        raise ConflictError(
            "pinned dataset or model versions have changed since this run; "
            "pass allow_version_drift=true to replay against current versions",
            details=drift,
        )

    return {
        "original_run_id": original.id,
        "query": original.query,
        "workflow": original.workflow,
        "parameters": original.parameters or {},
        "ranking_config": original.ranking_config or {},
        "seed": original.random_seed or 1337,
        "model_pins": {item["endpoint"]: item["ref"] for item in (original.model_versions or [])},
        "prompts": original.prompts or [],
        "drift": drift,
    }


async def usage_overview(context: ServiceContext) -> dict[str, Any]:
    """Aggregate observability numbers for the runs dashboard."""
    return {
        "runs": await runs_repo.run_usage_summary(context.session),
        "tools": await runs_repo.tool_call_stats(context.session),
    }
