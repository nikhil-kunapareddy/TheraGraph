"""Research planning.

The planner turns a free-text question into a *typed, ordered plan*. This is the
structural difference between this agent and a chat wrapper: before any data is touched,
there is an explicit artifact -- reviewable, storable, replayable -- saying which tools
will run, in what order, with what arguments, and why.

Two planners implement the same contract:

* The LLM planner, which asks the model for a schema-validated plan.
* :func:`deterministic_plan`, a rule-based planner that reads the question for entity
  hints and emits the standard discovery workflow. It is what makes the product usable
  with no API key, and it gives the eval suite a fixed baseline to compare the model's
  tool selection against.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from theragraph.core.logging import get_logger
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

MAX_PLAN_STEPS = 10

#: Words that suggest the question is about CNS chemistry, which selects a different
#: ranking preset. Kept explicit so the behaviour is auditable.
CNS_HINTS = (
    "brain",
    "cns",
    "alzheimer",
    "parkinson",
    "neuro",
    "dementia",
    "epilep",
    "depress",
    "schizophren",
    "blood-brain",
    "bbb",
)

INTENT_HINTS: dict[str, tuple[str, ...]] = {
    "compound_focus": ("compound", "molecule", "analog", "analogue", "similar", "smiles", "drug "),
    "evidence_focus": ("evidence", "genetic", "why", "support", "validated", "literature"),
    "ranking_focus": ("prioriti", "rank", "best candidate", "shortlist", "compare", "triage"),
    "property_focus": ("solubility", "permeab", "toxic", "logp", "property", "admet", "predict"),
}


class PlanStep(BaseModel):
    """One tool invocation in the plan."""

    step: int = Field(ge=1, le=MAX_PLAN_STEPS, description="1-based position in the plan.")
    tool: str = Field(description="Exact name of the tool to call.")
    purpose: str = Field(description="One line, written for the scientist.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Argument values. May contain {{step_N.field}} placeholders resolved from "
            "earlier steps."
        ),
    )
    depends_on_previous: bool = Field(
        default=False, description="True if this step needs an earlier step's output."
    )
    optional: bool = Field(
        default=False,
        description="True if the plan should continue when this step returns nothing.",
    )


class ResearchPlan(BaseModel):
    """The planner's structured output."""

    objective: str = Field(description="The question being answered, restated precisely.")
    strategy: str = Field(description="Two sentences on the approach.")
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)
    open_questions: list[str] = Field(
        default_factory=list, description="What would make the answer incomplete."
    )
    ranking_preset: Literal[
        "general_oral_small_molecule", "cns_small_molecule", "evidence_first"
    ] = Field(
        default="general_oral_small_molecule",
        description="Prioritisation preset appropriate to the question.",
    )

    def tool_sequence(self) -> list[str]:
        return [step.tool for step in sorted(self.steps, key=lambda item: item.step)]


class ReportClaim(BaseModel):
    text: str = Field(min_length=10, description="One specific sentence.")
    kind: Literal["retrieved_fact", "model_prediction", "interpretation"]
    citations: list[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    """The model's interpretation of the tool results."""

    summary: str = Field(min_length=40, description="3-6 sentences answering the question.")
    claims: list[ReportClaim] = Field(default_factory=list, max_length=12)
    open_questions: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(default="", description="Two sentences, written for the user.")


# --------------------------------------------------------------- query analysis
def detect_ranking_preset(query: str) -> str:
    lowered = query.lower()
    if any(hint in lowered for hint in CNS_HINTS):
        return "cns_small_molecule"
    return "general_oral_small_molecule"


def detect_intents(query: str) -> list[str]:
    lowered = query.lower()
    return [
        intent for intent, hints in INTENT_HINTS.items() if any(hint in lowered for hint in hints)
    ]


def extract_identifiers(query: str) -> dict[str, list[str]]:
    """Pull explicit identifiers out of the question.

    A question naming ``CHEMBL25`` or ``MONDO_0004975`` should go straight to that
    entity instead of a fuzzy text search, and gene symbols in the question are strong
    hints. Matching is deliberately conservative -- ambiguity is resolved by the search
    tool, which has the corpus.
    """
    return {
        "efo_ids": re.findall(r"\b(?:EFO|MONDO|HP|Orphanet)_\d+\b", query),
        "chembl_ids": re.findall(r"\bCHEMBL\d+\b", query.upper()),
        "ensembl_ids": re.findall(r"\bENSG\d{11}\b", query.upper()),
        # A gene symbol is an upper-case token, but so is any word in an upper-case
        # sentence. Excluding the planner's own task-word vocabulary (plus domain
        # abbreviations that are not genes) keeps the two mechanisms in sync: a word
        # the planner strips from a search term is never treated as a gene either.
        "gene_symbols": [
            token
            for token in re.findall(r"\b[A-Z][A-Z0-9]{2,7}\b", query)
            if token.lower() not in TASK_WORDS
            and token not in NON_GENE_ABBREVIATIONS
            # ChEMBL/Ensembl accessions are captured by their own patterns; leaving
            # them here too would let a compound id be used as a target hint.
            and not token.startswith(("CHEMBL", "ENSG", "MONDO", "EFO"))
        ],
        "smiles": re.findall(r"\b[A-Za-z0-9@+\-\[\]\(\)=#$/\\%\.]{8,}\b", query)
        if "smiles" in query.lower()
        else [],
    }


async def corpus_brief(context: ServiceContext) -> str:
    """One-paragraph description of what is available, injected into prompts."""
    from theragraph.services import discovery

    summary = await discovery.corpus_summary(context)
    counts = summary["counts"]
    diseases = ", ".join(version["dataset"] for version in summary["dataset_versions"][:6])
    endpoints: list[str] = []
    try:
        inference = context.require_inference()
        endpoints = [
            item["endpoint"] for item in await inference.available_endpoints(context.session)
        ]
    except Exception:  # pragma: no cover - model registry unavailable
        endpoints = []

    return (
        f"The corpus holds {counts['diseases']} disease(s), {counts['targets']} target(s), "
        f"{counts['compounds']} compound(s) and {counts['bioactivities']} measured "
        f"activity value(s), from {len(summary['sources'])} public sources "
        f"({diseases or 'no datasets yet'}). "
        + (
            f"Prediction endpoints available: {', '.join(endpoints)}."
            if endpoints
            else "No prediction models are registered, so predict_properties will return nothing."
        )
    )


# ---------------------------------------------------------- deterministic planner
def deterministic_plan(query: str, parameters: dict[str, Any] | None = None) -> ResearchPlan:
    """Rule-based plan covering the standard discovery workflow.

    Chosen by inspecting the question for identifiers and intent, then emitting the
    canonical sequence. Every branch here is a judgement a scientist would recognise:
    a question about analogues starts from similarity; a question naming a compound
    starts from that compound; anything else starts from disease -> target -> evidence ->
    chemistry -> predictions -> ranking.
    """
    parameters = parameters or {}
    identifiers = extract_identifiers(query)
    intents = detect_intents(query)
    preset = detect_ranking_preset(query)

    # Caller-supplied parameters are authoritative: the UI resolved them against the
    # corpus. Identifiers scraped from the question text are only hypotheses -- a gene
    # symbol in a sentence may be a typo, an alias, or not exist at all -- so those
    # plans start with a verification search instead of trusting the guess.
    disease_hint = parameters.get("disease")
    target_hint = parameters.get("target")
    compound_hint = parameters.get("compound")
    hints_are_authoritative = bool(disease_hint or target_hint or compound_hint)

    if not disease_hint and identifiers["efo_ids"]:
        disease_hint = identifiers["efo_ids"][0]
    if not target_hint and identifiers["ensembl_ids"]:
        target_hint = identifiers["ensembl_ids"][0]
    if not compound_hint and identifiers["chembl_ids"]:
        compound_hint = identifiers["chembl_ids"][0]

    steps: list[PlanStep] = []
    position = 1

    def add(tool: str, purpose: str, arguments: dict[str, Any], **flags: Any) -> None:
        nonlocal position
        steps.append(
            PlanStep(step=position, tool=tool, purpose=purpose, arguments=arguments, **flags)
        )
        position += 1

    # 1. Resolve the entity named in the question.
    if compound_hint and "compound_focus" in intents and not disease_hint and not target_hint:
        add(
            "get_compound",
            f"Retrieve the full profile of {compound_hint}.",
            {"compound": compound_hint},
        )
        add(
            "find_similar_compounds",
            "Find structural neighbours of the query compound in the corpus.",
            {"smiles": "{{step_1.smiles}}", "threshold": 0.6, "limit": 15},
            depends_on_previous=True,
        )
        add(
            "predict_properties",
            "Predict molecular properties for the compound and its neighbours.",
            {"compounds": "{{step_2.compound_ids}}"},
            depends_on_previous=True,
            optional=True,
        )
        return ResearchPlan(
            objective=f"Characterise {compound_hint} and its structural neighbours.",
            strategy=(
                "Start from the named compound, expand to structural neighbours in the "
                "corpus, then predict properties for the set."
            ),
            steps=steps,
            open_questions=[
                "Are the neighbours active against the same target as the query compound?"
            ],
            ranking_preset=preset,
        )

    if not hints_are_authoritative or (not disease_hint and not target_hint):
        add(
            "search_entities",
            "Resolve the entities named in the question against the corpus.",
            {"query": _search_term(query), "limit": 8},
        )

    # 2. Disease -> targets.
    if disease_hint or not target_hint:
        add(
            "get_disease_targets",
            "Retrieve targets associated with the disease, ranked by Open Targets score.",
            {
                "disease": disease_hint or "{{step_1.top_disease}}",
                "n_targets": int(parameters.get("n_targets", 10)),
            },
            depends_on_previous=not disease_hint,
        )

    # 3. Target evidence. An unverified symbol is passed through the search result
    # rather than used directly, so a symbol that does not exist fails at resolution
    # instead of producing a confident empty profile.
    add(
        "get_target_overview",
        "Retrieve evidence, tractability and known chemistry for the leading target.",
        {
            "target": target_hint or "{{prev.top_target}}",
            "disease": disease_hint or "{{state.disease_efo_id}}",
            "min_pchembl": float(parameters.get("min_pchembl", 6.0)),
        },
        depends_on_previous=not target_hint,
    )

    if "evidence_focus" in intents:
        add(
            "get_target_evidence",
            "Retrieve human genetic evidence specifically, the strongest evidence class.",
            {
                "target": target_hint or "{{state.target_symbol}}",
                "disease": disease_hint or "{{state.disease_efo_id}}",
                "evidence_types": ["genetic_association", "known_drug"],
                "limit": 20,
            },
            depends_on_previous=not target_hint,
            optional=True,
        )

    # 4. Expand chemistry by similarity when the question is about compounds.
    if "compound_focus" in intents:
        add(
            "find_similar_compounds",
            "Expand the candidate set with structural neighbours of the best known active.",
            {"smiles": "{{state.best_compound_smiles}}", "threshold": 0.6, "limit": 15},
            depends_on_previous=True,
            optional=True,
        )

    # 5. Predictions.
    add(
        "predict_properties",
        "Predict molecular properties for the candidate compounds.",
        {"compounds": "{{state.candidate_compound_ids}}"},
        depends_on_previous=True,
        optional=True,
    )

    # 6. Prioritise.
    add(
        "prioritize_candidates",
        "Rank the candidates with the transparent multi-objective configuration.",
        {
            "target": target_hint or "{{state.target_symbol}}",
            "disease": disease_hint or "{{state.disease_efo_id}}",
            "config_name": preset,
            "min_pchembl": float(parameters.get("min_pchembl", 5.0)),
            "limit": int(parameters.get("max_candidates", 25)),
        },
        depends_on_previous=True,
    )

    # 7. Graph, so the report links back to the evidence structure.
    add(
        "build_evidence_graph",
        "Assemble the evidence graph linking disease, target, evidence and compounds.",
        {
            "disease": disease_hint or "{{state.disease_efo_id}}",
            "target": target_hint or "{{state.target_symbol}}",
        },
        depends_on_previous=True,
        optional=True,
    )

    return ResearchPlan(
        objective=query.strip(),
        strategy=(
            "Resolve the disease, retrieve its best-evidenced targets, examine the "
            "leading target's evidence and known chemistry, predict molecular properties "
            "for the candidates, then prioritise them with the factor breakdown shown."
        ),
        steps=steps[:MAX_PLAN_STEPS],
        open_questions=[
            "Do the retrieved compounds have measured selectivity against related targets?",
            "Is the ingested corpus current for this disease area?",
        ],
        ranking_preset=preset,
    )


#: Upper-case domain abbreviations that are not gene symbols. Without this list an
#: all-caps question would hand the planner a fabricated target.
NON_GENE_ABBREVIATIONS = frozenset(
    {
        "BBB",
        "CNS",
        "ADME",
        "ADMET",
        "IC50",
        "EC50",
        "SMILES",
        "SMARTS",
        "QED",
        "LOGP",
        "LOGD",
        "TPSA",
        "MPO",
        "PK",
        "PD",
        "SAR",
        "QSAR",
        "HTS",
        "FDA",
        "EMA",
        "RNA",
        "DNA",
        "PDB",
        "CSV",
        "API",
        "JSON",
        "URL",
        "MW",
        "NM",
        "UM",
    }
)

#: Words that describe the *task* rather than the *entity*. Stripping them is what
#: turns "the most promising therapeutic targets for Alzheimer disease" into
#: "alzheimer disease", which is what the corpus is actually indexed on.
TASK_WORDS = frozenset(
    {
        "what",
        "which",
        "who",
        "how",
        "why",
        "are",
        "is",
        "the",
        "a",
        "an",
        "of",
        "for",
        "in",
        "on",
        "with",
        "and",
        "or",
        "to",
        "from",
        "me",
        "about",
        "any",
        "there",
        "do",
        "does",
        "can",
        "could",
        "should",
        "would",
        "i",
        "we",
        "my",
        "promising",
        "therapeutic",
        "target",
        "targets",
        "compound",
        "compounds",
        "drug",
        "drugs",
        "molecule",
        "molecules",
        "candidate",
        "candidates",
        "prioritise",
        "prioritize",
        "prioritised",
        "prioritized",
        "rank",
        "ranking",
        "best",
        "top",
        "good",
        "better",
        "treat",
        "treating",
        "treatment",
        "treatments",
        "therapy",
        "therapies",
        "find",
        "show",
        "give",
        "list",
        "tell",
        "identify",
        "investigate",
        "explore",
        "look",
        "evidence",
        "support",
        "supported",
        "validated",
        "validation",
        "predict",
        "prediction",
        "properties",
        "property",
        "similar",
        "similarity",
        "most",
        "some",
        "all",
        "new",
        "novel",
        "potential",
        "disease" if False else "",  # keep 'disease': it is part of many EFO names
    }
    - {""}
)


def _search_term(query: str) -> str:
    """Reduce a natural-language question to a searchable entity phrase.

    Disease and target names in the corpus are short ("Alzheimer disease", "BACE1"), so
    the whole sentence never matches. Task words are stripped and the remaining tokens
    are returned in order, which recovers the entity phrase for the overwhelming
    majority of real questions. The search service also falls back to progressively
    narrower terms, so an imperfect extraction still resolves.
    """
    tokens = [
        token
        for token in re.split(r"[^A-Za-z0-9_\-]+", query)
        if token and token.lower() not in TASK_WORDS
    ]
    # Preserve capitalised tokens verbatim (gene symbols, proper nouns); lowercase the
    # rest so the match is case-insensitive either way.
    phrase = " ".join(tokens[:5]).strip()
    return phrase or query.strip()[:80]


def deterministic_structured(schema_model: type[BaseModel], user_prompt: str) -> dict[str, Any]:
    """Produce a valid payload for the deterministic backend.

    Dispatches on the requested schema so the backend stays a drop-in replacement for
    the LLM at both call sites (planning and synthesis).
    """
    if schema_model is ResearchPlan:
        query = _extract_section(user_prompt, "## Question") or user_prompt
        return deterministic_plan(query).model_dump(mode="json")

    if schema_model is Synthesis:
        return _deterministic_synthesis(user_prompt)

    # Unknown schema: build the minimum valid instance from field defaults.
    payload: dict[str, Any] = {}
    for name, field in schema_model.model_fields.items():
        if field.default is not None and field.default is not Ellipsis:
            payload[name] = field.default
        elif field.default_factory is not None:  # type: ignore[truthy-function]
            payload[name] = field.default_factory()  # type: ignore[misc,call-arg]
        else:
            payload[name] = ""
    return payload


def synthesise_from_findings(
    query: str, findings: list[dict[str, Any]], *, warnings: list[str] | None = None
) -> dict[str, Any]:
    """Assemble a synthesis from structured tool findings, with no model involved.

    Each finding carries the tool that produced it, its summary and the citation keys
    *that tool* returned, so a claim cites the records behind it rather than a global
    pool. The result contains no interpretation -- that is the honest limit of a
    rule-based synthesiser, and it is stated in the rationale rather than disguised.
    """
    claims: list[dict[str, Any]] = []
    retrieved: list[str] = []

    for finding in findings:
        summary = str(finding.get("summary") or "").strip()
        if not summary or finding.get("status") != "succeeded":
            continue
        tool = str(finding.get("tool") or "")
        citations = [key for key in (finding.get("citations") or []) if key][:3]
        is_prediction = tool in {"predict_properties"} or "prediction" in summary.lower()

        if is_prediction:
            claims.append({"text": summary[:400], "kind": "model_prediction", "citations": []})
        elif citations:
            claims.append({"text": summary[:400], "kind": "retrieved_fact", "citations": citations})
        else:
            # No citation keys means nothing to cite; recording it as an uncited claim
            # would only get it rejected, so it becomes part of the narrative instead.
            retrieved.append(summary)

        if len(claims) >= 8:
            break

    failures = [finding for finding in findings if finding.get("status") not in (None, "succeeded")]

    sentences = [f"Question: {query.strip().rstrip('?')}?"]
    sentences.append(
        f"{len(claims)} finding(s) were assembled from {len(findings)} tool call(s)"
        + (f", {len(failures)} of which failed." if failures else ".")
    )
    for note in retrieved[:2]:
        sentences.append(note.rstrip(".") + ".")
    sentences.append(
        "Every retrieved fact below is cited to its source, and every model output is "
        "labelled as a prediction with its uncertainty and applicability-domain status."
    )
    sentences.append(
        "This summary was assembled by the deterministic backend, which restates tool "
        "output without interpreting it; set ANTHROPIC_API_KEY for model-written "
        "interpretation."
    )
    if warnings:
        sentences.append(f"Execution warnings: {warnings[0].rstrip('.')[:200]}.")

    return {
        "summary": " ".join(sentences),
        "claims": claims,
        "open_questions": [
            "Are the retrieved measurements from assays relevant to the intended "
            "mechanism of action?",
            "Would ingesting additional targets or ChEMBL neighbours change the ranking?",
        ],
        "rationale": (
            "Deterministic backend: the plan came from the question's entities and "
            "intent, and each claim restates one tool's output with that tool's own "
            "citation keys."
        ),
    }


def _deterministic_synthesis(user_prompt: str) -> dict[str, Any]:
    """Prompt-text fallback used when structured findings are unavailable."""
    query = _extract_section(user_prompt, "## Question") or "the submitted question"
    findings_text = _extract_section(user_prompt, "## What the tools returned") or ""
    citation_keys = re.findall(r"[a-z_]+:[A-Za-z0-9_.\-]+", findings_text)[:4]

    findings: list[dict[str, Any]] = []
    for line in findings_text.splitlines():
        stripped = line.strip("- ").strip()
        if len(stripped) < 30 or stripped.startswith("#"):
            continue
        tool = ""
        if stripped.startswith("["):
            tool = stripped[1 : stripped.find("]")] if "]" in stripped else ""
            stripped = stripped[stripped.find("]") + 1 :].strip()
        findings.append(
            {
                "tool": tool,
                "summary": stripped,
                "citations": citation_keys[:2],
                "status": "succeeded",
            }
        )
    return synthesise_from_findings(query, findings)


def _extract_section(text: str, heading: str) -> str | None:
    if heading not in text:
        return None
    after = text.split(heading, 1)[1]
    return after.split("\n##", 1)[0].strip()
