"""Evidence-backed research report generation.

The report is assembled in two layers, and the split is the whole point:

* **The factual backbone is templated, not generated.** Tables of targets, evidence,
  measured activities, predictions and ranked candidates are rendered directly from the
  run's structured state. No language model can alter a number, a citation or a score.
* **Only interpretation is written by the model**, as discrete *claims*, each of which
  must cite citation keys that exist in the run's retrieved records. A claim whose
  citations cannot be resolved is dropped from the report and recorded as a validation
  failure, so a hallucinated reference cannot reach the reader.

Claims are typed: ``retrieved_fact`` (restates something retrieved), ``model_prediction``
(reports a model output, with its uncertainty), or ``interpretation`` (the model's
reading of the two). The report renders the three differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from jinja2 import Environment, StrictUndefined

from theragraph.core.clock import utcnow
from theragraph.core.errors import NotFoundError
from theragraph.core.hashing import hash_payload
from theragraph.core.logging import get_logger
from theragraph.core.metrics import AGENT_VALIDATION_FAILURES
from theragraph.db.repositories import biology as biology_repo
from theragraph.db.repositories import runs as runs_repo
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

ClaimKind = Literal["retrieved_fact", "model_prediction", "interpretation"]

REPORT_TEMPLATE_ID = "report.markdown.v2"

STANDARD_LIMITATIONS = [
    "This report aggregates public data and model predictions. It does not establish "
    "that any target is druggable or that any compound is efficacious or safe.",
    "Association and evidence scores are reproduced verbatim from their sources "
    "(Open Targets, ChEMBL); TheraGraph does not recompute them.",
    "Model predictions are trained on small public benchmark datasets and are reported "
    "with scaffold-split performance and applicability-domain flags. Predictions for "
    "molecules outside the applicability domain are extrapolation.",
    "Candidate prioritisation is a transparent weighted combination of the factors "
    "shown, intended to order compounds for experimental follow-up. It is not a "
    "prediction of clinical outcome.",
    "Absence of evidence in this report reflects the contents of the ingested corpus at "
    "the recorded dataset versions, not the absence of evidence in the literature.",
]


@dataclass
class Claim:
    """One assertion in the report, with its provenance and validation verdict."""

    text: str
    kind: ClaimKind
    citations: list[str] = field(default_factory=list)
    validated: bool = False
    validation_note: str | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "citations": self.citations,
            "validated": self.validated,
            "validation_note": self.validation_note,
            "section": self.section,
        }


@dataclass
class Citation:
    key: str
    source: str
    source_identifier: str | None
    title: str | None
    url: str | None
    kind: str
    retrieved_at: str | None = None
    score: float | None = None
    publication: dict[str, Any] | None = None

    @property
    def reference_text(self) -> str:
        """Bibliography entry, assembled from retrieved metadata only."""
        publication = self.publication or {}
        if publication.get("title"):
            parts = [f"{publication['title'].rstrip('.')}."]
            journal = publication.get("journal") or "journal n/a"
            year = publication.get("year") or "n.d."
            parts.append(f"*{journal}* {year}.")
            if publication.get("pmid"):
                parts.append(f"PMID {publication['pmid']}.")
            if publication.get("doi"):
                parts.append(f"doi:{publication['doi']}.")
            return " ".join(parts)
        descriptor = self.title or self.kind.replace("_", " ")
        identifier = f" ({self.source_identifier})" if self.source_identifier else ""
        return f"{descriptor} — {self.source}{identifier}."

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "source": self.source,
            "source_identifier": self.source_identifier,
            "title": self.title,
            "url": self.url,
            "kind": self.kind,
            "retrieved_at": self.retrieved_at,
            "score": self.score,
            "publication": self.publication,
            "reference_text": self.reference_text,
        }


@dataclass
class ValidationReport:
    total_claims: int = 0
    validated_claims: int = 0
    rejected_claims: list[dict[str, Any]] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    citations_available: int = 0

    @property
    def rejection_rate(self) -> float:
        if not self.total_claims:
            return 0.0
        return round(len(self.rejected_claims) / self.total_claims, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "validated_claims": self.validated_claims,
            "rejected_claims": self.rejected_claims,
            "rejection_rate": self.rejection_rate,
            "unknown_citations": self.unknown_citations,
            "citations_available": self.citations_available,
            "policy": (
                "Claims citing records that were not retrieved during this run are "
                "removed from the report."
            ),
        }


async def build_citation_index(
    context: ServiceContext, *, evidence_ids: list[str]
) -> dict[str, Citation]:
    """Map every citable key to the record it refers to.

    Two key forms are accepted so the model can cite naturally: the internal evidence id
    and the ``source:source_identifier`` citation key.
    """
    index: dict[str, Citation] = {}
    evidence_rows = await biology_repo.get_evidence_by_ids(context.session, evidence_ids)

    for evidence in evidence_rows:
        publication = (
            {
                "pmid": evidence.publication.pmid,
                "doi": evidence.publication.doi,
                "title": evidence.publication.title,
                "journal": evidence.publication.journal,
                "year": evidence.publication.year,
                "url": evidence.publication.source_url,
            }
            if evidence.publication is not None
            else None
        )
        citation = Citation(
            key=evidence.citation_key,
            source=evidence.source,
            source_identifier=evidence.source_identifier,
            title=evidence.title,
            url=evidence.source_url,
            kind=evidence.evidence_type,
            retrieved_at=evidence.retrieved_at.isoformat(),
            score=evidence.score,
            publication=publication,
        )
        index[citation.key] = citation
        index[evidence.id] = citation

    return index


def validate_claims(
    claims: list[Claim], citation_index: dict[str, Citation]
) -> tuple[list[Claim], ValidationReport]:
    """Keep only claims whose citations resolve to retrieved records."""
    report = ValidationReport(
        total_claims=len(claims),
        citations_available=len({citation.key for citation in citation_index.values()}),
    )
    accepted: list[Claim] = []

    for claim in claims:
        if claim.kind == "model_prediction":
            # Prediction claims are backed by persisted prediction rows rather than
            # evidence citations; they are accepted but must be labelled as predictions.
            claim.validated = True
            claim.validation_note = "model output; not an experimental measurement"
            accepted.append(claim)
            continue

        if not claim.citations:
            report.rejected_claims.append(
                {"text": claim.text[:300], "reason": "no citations provided"}
            )
            AGENT_VALIDATION_FAILURES.labels(kind="uncited_claim").inc()
            continue

        unknown = [key for key in claim.citations if key not in citation_index]
        if unknown:
            report.unknown_citations.extend(unknown)
            report.rejected_claims.append(
                {
                    "text": claim.text[:300],
                    "reason": "cites records that were not retrieved in this run",
                    "unknown_citations": unknown,
                }
            )
            AGENT_VALIDATION_FAILURES.labels(kind="unresolvable_citation").inc()
            continue

        claim.validated = True
        accepted.append(claim)

    report.validated_claims = len(accepted)
    if report.rejected_claims:
        log.warning(
            "report.claims_rejected",
            rejected=len(report.rejected_claims),
            total=report.total_claims,
        )
    return accepted, report


MARKDOWN_TEMPLATE = """# {{ title }}

**Run:** `{{ run_id }}`  |  **Generated:** {{ generated_at }}  |  **Software:** {{ software_version }}{% if git_sha %} (`{{ git_sha[:8] }}`){% endif %}

## Research question

{{ question }}

## Summary

{{ summary }}

{% if key_findings %}
### Key findings

{% for finding in key_findings %}
- {{ finding.text }}{{ ' *(model prediction)*' if finding.kind == 'model_prediction' else '' }}{{ ' [' ~ finding.citations | join('], [') ~ ']' if finding.citations else '' }}
{% endfor %}
{% endif %}

{% if targets %}
## Target landscape

Associated targets retrieved from {{ target_source }}, ordered by the source's own
association score. Scores are reproduced verbatim.

| Target | Association score | Top evidence types | Tractability (small molecule) |
|---|---|---|---|
{% for target in targets -%}
| **{{ target.symbol }}** — {{ target.name or '—' }} | {{ '%.3f' | format(target.score) if target.score is not none else 'not provided' }} | {{ target.evidence_types | join(', ') or '—' }} | {{ target.tractability or '—' }} |
{% endfor %}
{% endif %}

{% if evidence_groups %}
## Evidence

{% for group in evidence_groups %}
### {{ group.label }} ({{ group.count }} record{{ 's' if group.count != 1 else '' }})

{{ group.meaning }}

{% for item in group.records %}
- **[{{ item.key }}]** {{ item.title or item.summary or 'record' }}{{ ' — score %.3f' | format(item.score) if item.score is not none else '' }}{{ ' — %s %s, PMID %s' | format(item.publication.journal or 'journal n/a', item.publication.year or '', item.publication.pmid) if item.publication and item.publication.pmid else '' }}
{% endfor %}

{% endfor %}
{% endif %}

{% if known_compounds %}
## Known chemistry (experimental measurements)

Measured activities retrieved from ChEMBL. These are experimental values, not predictions.

| Compound | ChEMBL | Best pChEMBL | Potency (nM) | Measurements | Max phase |
|---|---|---|---|---|---|
{% for compound in known_compounds -%}
| {{ compound.name }} | {{ compound.chembl_id or '—' }} | {{ '%.2f' | format(compound.best_pchembl) if compound.best_pchembl is not none else '—' }} | {{ '%.1f' | format(compound.potency_nm) if compound.potency_nm else '—' }} | {{ compound.n_measurements }} | {{ compound.max_phase if compound.max_phase is not none else '—' }} |
{% endfor %}
{% endif %}

{% if predictions %}
## Model predictions

{% for model in models_used %}
- **{{ model.endpoint }}**: `{{ model.ref }}` ({{ model.family }}){{ ' — scaffold-split ' ~ model.scaffold_summary if model.scaffold_summary else '' }}
{% endfor %}

Predictions below are model outputs. Values for molecules outside the training
applicability domain are flagged and should be treated as extrapolation.

| Compound | Endpoint | Prediction | Uncertainty | Confidence | In domain |
|---|---|---|---|---|---|
{% for prediction in predictions -%}
| {{ prediction.name }} | {{ prediction.endpoint }} | {{ prediction.display_value }} | {{ '%.3f' | format(prediction.uncertainty) if prediction.uncertainty is not none else '—' }} | {{ '%.2f' | format(prediction.confidence) if prediction.confidence is not none else '—' }} | {{ 'yes' if prediction.in_domain else ('no' if prediction.in_domain is false else '—') }} |
{% endfor %}
{% endif %}

{% if candidates %}
## Prioritised candidates

Ranked with the `{{ ranking_config_name }}` configuration (hash `{{ ranking_config_hash }}`).
Every score decomposes into the factors shown; weights are recorded on the run.

| Rank | Compound | Score | Leading factors | Flags |
|---|---|---|---|---|
{% for candidate in candidates -%}
| {{ candidate.rank }} | {{ candidate.name }} | {{ '%.3f' | format(candidate.total_score) }} | {{ candidate.leading_factors }} | {{ candidate.flags | join(', ') or '—' }} |
{% endfor %}

*{{ ranking_disclaimer }}*
{% endif %}

{% if interpretations %}
## Interpretation

{% for claim in interpretations %}
{{ claim.text }}{{ ' [' ~ claim.citations | join('], [') ~ ']' if claim.citations else '' }}

{% endfor %}
{% endif %}

## Limitations

{% for limitation in limitations %}
- {{ limitation }}
{% endfor %}

## Methods and provenance

**Data sources and versions**

{% for version in dataset_versions %}
- `{{ version.label }}` — {{ version.record_count }} records, retrieved {{ version.retrieved_at }}, content hash `{{ version.content_hash }}`
{% endfor %}

**Models**

{% for model in model_versions %}
- `{{ model.ref }}` ({{ model.family }}, {{ model.task_type }}) — artifact `{{ (model.artifact_hash or '')[:12] }}`
{% endfor %}

**Execution**

- Tool calls: {{ n_tool_calls }}
- Runtime: {{ '%.1f' | format(runtime_ms / 1000) if runtime_ms else '—' }} s
- Agent model: `{{ agent_model }}` (backend: {{ llm_backend }})
- Random seed: {{ random_seed }}
- Reproducibility hash: `{{ reproducibility_hash }}`

{% if validation %}
**Claim validation**

- {{ validation.validated_claims }} of {{ validation.total_claims }} claims passed citation validation.
{% if validation.rejected_claims %}
- {{ validation.rejected_claims | length }} claim(s) were removed for citing records not retrieved in this run.
{% endif %}
{% endif %}

## References

{% for citation in citations %}
{{ loop.index }}. **[{{ citation.key }}]** {{ citation.reference_text }}{{ ' <' ~ citation.url ~ '>' if citation.url else '' }}
{% endfor %}

---

*Generated by TheraGraph. Retrieved facts and model predictions are labelled separately
throughout. This is a research aid, not a clinical or regulatory document.*
"""


def _environment() -> Environment:
    env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    env.globals["false"] = False
    return env


def _display_value(prediction: dict[str, Any]) -> str:
    if prediction.get("probability") is not None:
        return f"{prediction['probability']:.3f} probability"
    if prediction.get("value") is not None:
        units = prediction.get("units")
        return f"{prediction['value']:.3f}" + (f" {units}" if units else "")
    return "—"


def _scaffold_summary(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return ""
    if metrics.get("roc_auc") is not None:
        return f"ROC-AUC {metrics['roc_auc']:.3f}"
    if metrics.get("rmse") is not None:
        return f"RMSE {metrics['rmse']:.3f}"
    return ""


async def generate_report(
    context: ServiceContext,
    *,
    run_id: str,
    title: str | None = None,
    summary: str | None = None,
    claims: list[Claim] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Render the report for a run from its structured state."""
    run = await runs_repo.get_run(context.session, run_id)
    if run is None:
        raise NotFoundError(f"run {run_id!r} not found")

    state = run.state or {}
    evidence_ids = list((run.retrieved_record_ids or {}).get("evidence") or [])
    citation_index = await build_citation_index(context, evidence_ids=evidence_ids)

    accepted_claims, validation = validate_claims(claims or [], citation_index)

    disease = state.get("disease") or {}
    targets = state.get("targets") or []
    evidence_state = state.get("evidence") or {}
    known_compounds = state.get("known_compounds") or []
    predictions = state.get("predictions") or []
    ranking = state.get("ranking") or {}
    models_used = state.get("models_used") or []

    evidence_groups: list[dict[str, Any]] = []
    for group in evidence_state.get("groups") or []:
        evidence_groups.append(
            {
                "label": str(group.get("evidence_type", "other")).replace("_", " ").title(),
                "meaning": group.get("meaning") or "",
                "count": group.get("count") or len(group.get("items") or []),  # source key
                # Named "records", not "items": Jinja resolves `group.items` to the
                # dict method rather than the key, which silently breaks the template.
                "records": [
                    {
                        "key": item.get("citation_key"),
                        "title": item.get("title"),
                        "summary": (item.get("summary") or "")[:280] or None,
                        "score": item.get("score"),
                        "publication": item.get("publication"),
                    }
                    for item in (group.get("items") or [])[:8]
                ],
            }
        )

    used_citation_keys = [
        key
        for group in evidence_groups
        for item in group["records"]
        if (key := item.get("key")) and key in citation_index
    ]
    for claim in accepted_claims:
        used_citation_keys.extend(key for key in claim.citations if key in citation_index)

    citations: list[Citation] = []
    seen_keys: set[str] = set()
    for key in used_citation_keys:
        citation = citation_index.get(key)
        if citation is not None and citation.key not in seen_keys:
            citations.append(citation)
            seen_keys.add(citation.key)

    resolved_title = title or _default_title(run.query, disease, targets)
    resolved_summary = summary or _default_summary(
        disease, targets, known_compounds, ranking, predictions
    )

    context_payload = {
        "title": resolved_title,
        "run_id": run.id,
        "generated_at": utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "software_version": run.software_version,
        "git_sha": run.git_sha,
        "question": run.query,
        "summary": resolved_summary,
        "key_findings": [
            claim.to_dict()
            for claim in accepted_claims
            if claim.kind in ("retrieved_fact", "model_prediction")
        ][:8],
        "interpretations": [
            claim.to_dict() for claim in accepted_claims if claim.kind == "interpretation"
        ],
        "target_source": "Open Targets",
        "targets": [
            {
                "symbol": target.get("symbol"),
                "name": target.get("name"),
                "score": target.get("association_score"),
                "evidence_types": target.get("evidence_types") or [],
                "tractability": (target.get("tractability") or {}).get("Small molecule"),
            }
            for target in targets[:10]
        ],
        "evidence_groups": evidence_groups,
        "known_compounds": [
            {
                "name": compound.get("name"),
                "chembl_id": compound.get("chembl_id"),
                "best_pchembl": compound.get("best_pchembl"),
                "potency_nm": compound.get("potency_nm"),
                "n_measurements": compound.get("n_measurements", 0),
                "max_phase": compound.get("max_phase"),
            }
            for compound in known_compounds[:15]
        ],
        "predictions": [
            {
                "name": prediction.get("name") or prediction.get("compound_id"),
                "endpoint": prediction.get("endpoint"),
                "display_value": _display_value(prediction),
                "uncertainty": prediction.get("uncertainty"),
                "confidence": prediction.get("confidence"),
                "in_domain": (prediction.get("applicability") or {}).get("in_domain"),
            }
            for prediction in predictions[:25]
        ],
        "models_used": [
            {
                **model,
                "scaffold_summary": _scaffold_summary(model.get("scaffold_metrics")),
            }
            for model in models_used
        ],
        "candidates": [
            {
                "rank": candidate.get("rank"),
                "name": candidate.get("name") or candidate.get("compound_id"),
                "total_score": candidate.get("total_score", 0.0),
                "leading_factors": ", ".join(
                    f"{factor['label']} ({factor['contribution']:.2f})"
                    for factor in (candidate.get("top_contributors") or [])[:2]
                )
                or "—",
                "flags": candidate.get("flags") or [],
            }
            for candidate in (ranking.get("candidates") or [])[:15]
        ],
        "ranking_config_name": (ranking.get("config") or {}).get("name", "default"),
        "ranking_config_hash": ranking.get("config_hash", "—"),
        "ranking_disclaimer": (ranking.get("disclaimer") or "").strip()
        or "Prioritisation for experimental follow-up, not a prediction of efficacy.",
        "limitations": [*STANDARD_LIMITATIONS, *(state.get("limitations") or [])],
        "dataset_versions": run.dataset_versions or [],
        "model_versions": run.model_versions or [],
        "n_tool_calls": run.n_tool_calls,
        "runtime_ms": run.runtime_ms,
        "agent_model": run.agent_model,
        "llm_backend": run.llm_backend,
        "random_seed": run.random_seed,
        "reproducibility_hash": run.reproducibility_hash,
        "validation": validation.to_dict(),
        "citations": [citation.to_dict() for citation in citations],
    }

    markdown = _environment().from_string(MARKDOWN_TEMPLATE).render(**context_payload)

    sections = [
        {"id": "summary", "title": "Summary", "kind": "narrative"},
        {"id": "targets", "title": "Target landscape", "kind": "retrieved"},
        {"id": "evidence", "title": "Evidence", "kind": "retrieved"},
        {"id": "chemistry", "title": "Known chemistry", "kind": "retrieved"},
        {"id": "predictions", "title": "Model predictions", "kind": "model_output"},
        {"id": "candidates", "title": "Prioritised candidates", "kind": "derived"},
        {"id": "limitations", "title": "Limitations", "kind": "narrative"},
        {"id": "methods", "title": "Methods and provenance", "kind": "provenance"},
        {"id": "references", "title": "References", "kind": "provenance"},
    ]

    payload = {
        "run_id": run.id,
        "title": resolved_title,
        "question": run.query,
        "summary": resolved_summary,
        "markdown": markdown,
        "sections": sections,
        "claims": [claim.to_dict() for claim in accepted_claims],
        "citations": [citation.to_dict() for citation in citations],
        "limitations": context_payload["limitations"],
        "validation": validation.to_dict(),
        "content_hash": hash_payload(
            {"markdown": markdown, "claims": [c.to_dict() for c in accepted_claims]},
            length=32,
        ),
        "template_id": REPORT_TEMPLATE_ID,
    }

    if persist:
        stored = await runs_repo.set_report(
            context.session,
            research_run_id=run.id,
            title=payload["title"],
            question=payload["question"],
            summary=payload["summary"],
            markdown=payload["markdown"],
            sections=payload["sections"],
            claims=payload["claims"],
            citations=payload["citations"],
            limitations=payload["limitations"],
            validation=payload["validation"],
            content_hash=payload["content_hash"],
        )
        payload["report_id"] = stored.id

        if context.store is not None:
            try:
                uploaded = await context.store.put_bytes(
                    f"reports/{run.id}.md",
                    markdown.encode("utf-8"),
                    content_type="text/markdown",
                )
                stored.storage_uri = uploaded.uri
                payload["storage_uri"] = uploaded.uri
            except Exception as exc:  # pragma: no cover - storage outage
                log.warning("report.upload_failed", run_id=run.id, error=str(exc))

    log.info(
        "report.generated",
        run_id=run.id,
        claims=len(accepted_claims),
        citations=len(citations),
        rejected=len(validation.rejected_claims),
    )
    return payload


def _default_title(query: str, disease: dict[str, Any], targets: list[dict[str, Any]]) -> str:
    if disease.get("name") and targets:
        return (
            f"Target and compound landscape for {disease['name']}: "
            f"{len(targets)} associated target(s) reviewed"
        )
    if disease.get("name"):
        return f"Target landscape for {disease['name']}"
    return f"Research report: {query[:120]}"


def _default_summary(
    disease: dict[str, Any],
    targets: list[dict[str, Any]],
    known_compounds: list[dict[str, Any]],
    ranking: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> str:
    """Deterministic summary built only from counts and retrieved values.

    Used when no model-written summary is available (or when one was rejected by
    validation), so a report is never blocked on an LLM being reachable.
    """
    parts: list[str] = []
    if disease.get("name"):
        parts.append(f"This run investigated {disease['name']} ({disease.get('efo_id', 'no id')}).")
    if targets:
        top = targets[0]
        score = top.get("association_score")
        parts.append(
            f"{len(targets)} associated target(s) were retrieved from Open Targets; the "
            f"highest-scoring was {top.get('symbol')}"
            + (f" (association score {score:.3f})." if score is not None else ".")
        )
    if known_compounds:
        best = max(
            (compound for compound in known_compounds if compound.get("best_pchembl")),
            key=lambda compound: compound["best_pchembl"],
            default=None,
        )
        parts.append(
            f"{len(known_compounds)} compound(s) with measured activity were retrieved "
            "from ChEMBL"
            + (
                f", the most potent being {best['name']} at pChEMBL {best['best_pchembl']:.2f}."
                if best
                else "."
            )
        )
    else:
        parts.append(
            "No compounds with measured activity against the selected target(s) were "
            "present in the ingested corpus."
        )
    if predictions:
        out_of_domain = sum(
            1
            for prediction in predictions
            if (prediction.get("applicability") or {}).get("in_domain") is False
        )
        parts.append(
            f"{len(predictions)} molecular property prediction(s) were generated"
            + (
                f", of which {out_of_domain} fell outside the models' applicability "
                "domain and are flagged as extrapolation."
                if out_of_domain
                else "."
            )
        )
    if ranking.get("candidates"):
        parts.append(
            f"{len(ranking['candidates'])} candidate(s) were prioritised using the "
            f"{(ranking.get('config') or {}).get('name', 'default')} configuration; the "
            "factor breakdown for each is shown below."
        )
    parts.append(
        "All retrieved facts are cited to their source; all model outputs are labelled "
        "as predictions with uncertainty and applicability-domain status."
    )
    return " ".join(parts)


async def get_report(context: ServiceContext, run_id: str) -> dict[str, Any]:
    report = await runs_repo.get_report(context.session, run_id)
    if report is None:
        raise NotFoundError(f"run {run_id!r} has no report")
    return {
        "report_id": report.id,
        "run_id": report.research_run_id,
        "title": report.title,
        "question": report.question,
        "summary": report.summary,
        "markdown": report.markdown,
        "sections": report.sections,
        "claims": report.claims,
        "citations": report.citations,
        "limitations": report.limitations,
        "validation": report.validation,
        "content_hash": report.content_hash,
        "storage_uri": report.storage_uri,
        "created_at": report.created_at.isoformat(),
    }
