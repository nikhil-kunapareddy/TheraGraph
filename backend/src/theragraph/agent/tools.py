"""The agent's tool surface.

Every tool has a Pydantic input model and returns a :class:`ToolResult`. Three fields on
that result do the load-bearing work:

* ``data`` -- the structured payload the model sees.
* ``record_ids`` -- primary keys of the database rows the tool actually returned. These
  accumulate on the run and are what citation validation checks against, so a citation
  is only valid if the corresponding record was genuinely retrieved.
* ``citations`` -- citation keys the model is permitted to use in the report.

Tools are read-only by default. The two write tools (ingestion) are marked ``mutates``
and are only exposed when the caller opts in, because they spend upstream API quota and
create new dataset versions.

The same registry backs the REST API's agent endpoints *and* the MCP server, so an
external MCP client sees exactly the tools the agent sees.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from theragraph.core.clock import Stopwatch
from theragraph.core.errors import NotFoundError, TheraGraphError, ValidationError
from theragraph.core.logging import get_logger
from theragraph.core.metrics import AGENT_TOOL_CALLS, AGENT_TOOL_LATENCY
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

TOOL_VERSION = "1"


@dataclass
class ToolResult:
    """Structured tool output plus everything needed to audit it."""

    data: dict[str, Any]
    summary: str
    record_ids: dict[str, list[str]] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    is_empty: bool = False
    warnings: list[str] = field(default_factory=list)
    state_patch: dict[str, Any] = field(default_factory=dict)
    """Structured intermediate state to merge onto the run (never free text)."""

    def to_model_payload(self, *, max_items: int = 25) -> dict[str, Any]:
        """Compact projection handed to the model.

        Full results live in the run trace; the model gets a bounded view so a 200-row
        result set cannot consume the context window.
        """
        payload = {
            "summary": self.summary,
            "data": _truncate(self.data, max_items=max_items),
            "citations_available": self.citations[:60],
            "is_empty": self.is_empty,
        }
        if self.warnings:
            payload["warnings"] = self.warnings
        return payload


def _truncate(value: Any, *, max_items: int, depth: int = 0) -> Any:
    if depth > 6:
        return "..."
    if isinstance(value, dict):
        return {
            key: _truncate(item, max_items=max_items, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list):
        trimmed = [
            _truncate(item, max_items=max_items, depth=depth + 1) for item in value[:max_items]
        ]
        if len(value) > max_items:
            trimmed.append(f"... {len(value) - max_items} more items omitted")
        return trimmed
    if isinstance(value, str) and len(value) > 1500:
        return value[:1500] + "..."
    return value


ToolHandler = Callable[[ServiceContext, BaseModel], Awaitable[ToolResult]]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    category: Literal["search", "evidence", "chemistry", "ml", "ranking", "report", "ingestion"]
    mutates: bool = False
    expensive: bool = False

    def schema(self) -> dict[str, Any]:
        """Anthropic tool definition.

        ``strict`` plus ``additionalProperties: false`` means arguments are guaranteed to
        validate against the schema, so the handler never has to defend against a
        malformed shape -- only against semantically wrong values.
        """
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        schema["additionalProperties"] = False
        schema.setdefault("required", [])
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": _inline_refs(schema),
        }


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs`` so the tool schema is self-contained."""
    definitions = schema.pop("$defs", {})
    if not definitions:
        return schema

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = str(node["$ref"]).rsplit("/", 1)[-1]
                return resolve(dict(definitions.get(ref, {})))
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)  # type: ignore[no-any-return]


# ------------------------------------------------------------------ input models
class SearchInput(BaseModel):
    query: str = Field(description="Free-text disease, target, gene symbol or compound name.")
    entity_types: list[Literal["disease", "target", "compound"]] | None = Field(
        default=None, description="Restrict the search to these entity types."
    )
    limit: int = Field(default=10, ge=1, le=50)
    include_upstream: bool = Field(
        default=False,
        description=(
            "If nothing is found locally, also ask Open Targets what exists upstream. "
            "Returns discoverable entities that are not yet ingested."
        ),
    )


class DiseaseInput(BaseModel):
    disease: str = Field(description="Disease name, internal id, or EFO/MONDO id.")
    n_targets: int = Field(default=15, ge=1, le=50)
    min_association_score: float = Field(default=0.0, ge=0.0, le=1.0)


class TargetInput(BaseModel):
    target: str = Field(description="Target symbol (e.g. BACE1), Ensembl id, or internal id.")
    disease: str | None = Field(
        default=None, description="Optional disease context for association and evidence."
    )
    evidence_limit: int = Field(default=30, ge=1, le=100)
    compound_limit: int = Field(default=20, ge=1, le=50)
    min_pchembl: float = Field(
        default=6.0, ge=0.0, le=15.0, description="Minimum measured pChEMBL for known compounds."
    )


class EvidenceInput(BaseModel):
    target: str
    disease: str | None = None
    evidence_types: list[str] | None = Field(
        default=None,
        description=(
            "Filter by evidence type: genetic_association, known_drug, literature, "
            "animal_model, expression, pathway, functional_genomics, somatic_mutation."
        ),
    )
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    limit: int = Field(default=30, ge=1, le=100)


class CompoundInput(BaseModel):
    compound: str = Field(description="Compound name, ChEMBL id, PubChem CID or InChIKey.")
    include_bioactivity: bool = True
    include_similar: bool = True
    similarity_threshold: float = Field(default=0.7, ge=0.3, le=1.0)


class SimilarityInput(BaseModel):
    smiles: str = Field(description="Query structure as SMILES.")
    threshold: float = Field(
        default=0.7, ge=0.3, le=1.0, description="Minimum Tanimoto similarity."
    )
    limit: int = Field(default=20, ge=1, le=50)

    @field_validator("smiles")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("smiles must not be empty")
        return value.strip()


class SubstructureInput(BaseModel):
    smarts: str = Field(description="SMARTS substructure query.")
    limit: int = Field(default=20, ge=1, le=50)


class PropertiesInput(BaseModel):
    smiles: str | None = Field(default=None, description="Compute properties for this structure.")
    compound: str | None = Field(
        default=None, description="Or look up an ingested compound by id/ChEMBL id."
    )


class PredictInput(BaseModel):
    compounds: list[str] = Field(
        default_factory=list,
        description="Compound ids or ChEMBL ids already in the corpus.",
        max_length=50,
    )
    smiles: list[str] = Field(
        default_factory=list, description="Or raw SMILES to score ad hoc.", max_length=50
    )
    endpoints: list[str] = Field(
        default_factory=list,
        description=(
            "Endpoints to predict (solubility, bbb_permeability, clinical_toxicity, "
            "lipophilicity). Empty means every active endpoint."
        ),
    )


class PrioritizeInput(BaseModel):
    target: str
    disease: str | None = None
    compounds: list[str] | None = Field(default=None, max_length=100)
    config_name: str | None = Field(
        default=None,
        description=(
            "Ranking preset: general_oral_small_molecule, cns_small_molecule, evidence_first."
        ),
    )
    min_pchembl: float = Field(default=5.0, ge=0.0, le=15.0)
    limit: int = Field(default=25, ge=1, le=100)


class CompareInput(BaseModel):
    compounds: list[str] = Field(min_length=1, max_length=12)
    target: str | None = None
    disease: str | None = None
    config_name: str | None = None


class GraphInput(BaseModel):
    disease: str | None = None
    target: str | None = None
    compound: str | None = None
    max_targets: int = Field(default=6, ge=1, le=15)


class CorpusInput(BaseModel):
    pass


class IngestDiseaseInput(BaseModel):
    efo_id: str = Field(description="EFO or MONDO identifier, e.g. MONDO_0004975.")
    n_targets: int = Field(default=15, ge=1, le=50)


class IngestBioactivityInput(BaseModel):
    target: str = Field(description="Target symbol or Ensembl id already in the corpus.")
    max_activities: int = Field(default=200, ge=10, le=1000)


# ---------------------------------------------------------------------- handlers
async def _search(context: ServiceContext, args: SearchInput) -> ToolResult:
    from theragraph.services import discovery

    payload = await discovery.search(
        context,
        args.query,
        entity_types=args.entity_types,  # type: ignore[arg-type]
        limit=args.limit,
        include_upstream=args.include_upstream,
    )
    hits = payload["hits"]
    record_ids: dict[str, list[str]] = {}
    for hit in hits:
        record_ids.setdefault(hit["entity_type"], []).append(hit["id"])
    return ToolResult(
        data=payload,
        summary=(
            f"{len(hits)} local match(es) for {args.query!r}: "
            + ", ".join(f"{hit['name']} ({hit['entity_type']})" for hit in hits[:5])
            if hits
            else f"No local matches for {args.query!r}."
        ),
        record_ids=record_ids,
        is_empty=not hits,
        warnings=(
            []
            if hits
            else [
                "Nothing in the local corpus matched. Use include_upstream=true to see "
                "what exists in Open Targets, then ingest it."
            ]
        ),
    )


async def _disease_targets(context: ServiceContext, args: DiseaseInput) -> ToolResult:
    from theragraph.services import discovery

    payload = await discovery.get_disease_overview(
        context,
        args.disease,
        n_targets=args.n_targets,
        min_score=args.min_association_score,
    )
    disease = payload["disease"]
    targets = payload["associated_targets"]
    return ToolResult(
        data=payload,
        summary=(
            f"{disease['name']} ({disease['efo_id']}): {len(targets)} associated "
            "target(s) from Open Targets. Top: "
            + ", ".join(
                f"{item['target']['symbol']} {item['association']['overall_score']:.3f}"
                for item in targets[:5]
            )
        ),
        record_ids={
            "disease": [disease["id"]],
            "target": [item["target"]["id"] for item in targets],
        },
        is_empty=not targets,
        state_patch={
            "disease": {
                "id": disease["id"],
                "efo_id": disease["efo_id"],
                "name": disease["name"],
            },
            "targets": [
                {
                    "id": item["target"]["id"],
                    "target_key": item["target"]["target_key"],
                    "symbol": item["target"]["symbol"],
                    "name": item["target"]["name"],
                    "association_score": item["association"]["overall_score"],
                    "tractability": item["target"]["tractability"],
                    "evidence_types": list(
                        (item["association"].get("datatype_scores") or {}).keys()
                    ),
                }
                for item in targets
            ],
        },
    )


async def _target_overview(context: ServiceContext, args: TargetInput) -> ToolResult:
    from theragraph.services import targets as targets_service

    payload = await targets_service.get_target_overview(
        context,
        args.target,
        disease_identifier=args.disease,
        evidence_limit=args.evidence_limit,
        compound_limit=args.compound_limit,
        min_pchembl=args.min_pchembl,
    )
    target = payload["target"]
    evidence_ids = [
        item["id"] for group in payload["evidence"]["groups"] for item in group["items"]
    ]
    citations = [
        item["citation_key"] for group in payload["evidence"]["groups"] for item in group["items"]
    ]
    compounds = payload["known_compounds"]
    return ToolResult(
        data=payload,
        summary=(
            f"{target['symbol']} ({target['name'] or 'no name'}): "
            f"{payload['evidence']['total']} evidence record(s) across "
            f"{len(payload['evidence']['groups'])} type(s); "
            f"{len(compounds)} compound(s) with measured activity above pChEMBL "
            f"{args.min_pchembl}."
        ),
        record_ids={
            "target": [target["id"]],
            "evidence": evidence_ids,
            "compound": [item["compound_id"] for item in compounds],
        },
        citations=citations,
        is_empty=not evidence_ids and not compounds,
        warnings=(
            [payload["chemistry_summary"]["note"]]
            if payload["chemistry_summary"].get("note")
            else []
        ),
        state_patch={
            "evidence": payload["evidence"],
            "known_compounds": compounds,
            "tractability": payload["tractability"],
            "selected_target": {
                "id": target["id"],
                "symbol": target["symbol"],
                "target_key": target["target_key"],
            },
        },
    )


async def _target_evidence(context: ServiceContext, args: EvidenceInput) -> ToolResult:
    from theragraph.services import targets as targets_service

    payload = await targets_service.get_target_evidence(
        context,
        args.target,
        disease_identifier=args.disease,
        evidence_types=args.evidence_types,
        min_score=args.min_score,
        limit=args.limit,
    )
    items = payload["items"]
    by_type: dict[str, int] = {}
    for item in items:
        by_type[item["evidence_type"]] = by_type.get(item["evidence_type"], 0) + 1
    return ToolResult(
        data=payload,
        summary=(
            f"{len(items)} evidence record(s) for {payload['target']['symbol']}"
            + (f" in {payload['disease']['name']}" if payload.get("disease") else "")
            + (f": {by_type}" if by_type else ".")
        ),
        record_ids={"evidence": [item["id"] for item in items]},
        citations=payload["citation_keys"],
        is_empty=not items,
    )


async def _compound_overview(context: ServiceContext, args: CompoundInput) -> ToolResult:
    from theragraph.services import compounds as compounds_service

    payload = await compounds_service.get_compound_overview(
        context,
        args.compound,
        include_bioactivity=args.include_bioactivity,
        include_similar=args.include_similar,
        similarity_threshold=args.similarity_threshold,
    )
    compound = payload["compound"]
    values = (payload.get("properties") or {}).get("values") or {}
    return ToolResult(
        data=payload,
        summary=(
            f"{compound['name']} ({compound['chembl_id'] or compound['inchikey']}): "
            f"MW {values.get('molecular_weight', 'n/a')}, LogP {values.get('logp', 'n/a')}, "
            f"TPSA {values.get('tpsa', 'n/a')}, max phase {compound['max_phase']}."
        ),
        record_ids={"compound": [compound["id"]]},
        is_empty=False,
    )


async def _similar(context: ServiceContext, args: SimilarityInput) -> ToolResult:
    from theragraph.services import compounds as compounds_service

    payload = await compounds_service.find_similar(
        context, smiles=args.smiles, threshold=args.threshold, limit=args.limit
    )
    hits = payload["hits"]
    return ToolResult(
        data=payload,
        summary=(
            f"{len(hits)} compound(s) at Tanimoto >= {args.threshold} "
            f"(searched {payload['search_stats']['candidates_examined']} of "
            f"{payload['search_stats']['corpus_compounds']} after popcount screening)."
        ),
        record_ids={"compound": [hit["compound_id"] for hit in hits]},
        is_empty=not hits,
        warnings=(
            []
            if hits
            else [
                f"No compound in the corpus is within Tanimoto {args.threshold} of this "
                "structure. Lower the threshold or ingest ChEMBL neighbours."
            ]
        ),
        state_patch={"similar_compounds": hits},
    )


async def _substructure(context: ServiceContext, args: SubstructureInput) -> ToolResult:
    from theragraph.services import compounds as compounds_service

    payload = await compounds_service.substructure_search(
        context, smarts=args.smarts, limit=args.limit
    )
    return ToolResult(
        data=payload,
        summary=(
            f"{payload['count']} compound(s) contain the substructure "
            f"{args.smarts!r} (screened {payload['search_stats']['corpus_compounds']} "
            "molecules)."
        ),
        record_ids={"compound": [item["compound_id"] for item in payload["matches"]]},
        is_empty=payload["count"] == 0,
    )


async def _properties(context: ServiceContext, args: PropertiesInput) -> ToolResult:
    from theragraph.services import compounds as compounds_service

    if not args.smiles and not args.compound:
        raise ValidationError("provide either 'smiles' or 'compound'")

    if args.smiles:
        payload = await compounds_service.compute_properties_for_smiles(args.smiles)
        values = payload["properties"]["values"]
        return ToolResult(
            data=payload,
            summary=(
                f"Computed descriptors for {payload['canonical_smiles']}: MW "
                f"{values.get('molecular_weight')}, LogP {values.get('logp')}, TPSA "
                f"{values.get('tpsa')}, QED {values.get('qed')}."
            ),
        )

    assert args.compound is not None
    payload = await compounds_service.get_compound_overview(
        context, args.compound, include_bioactivity=False, include_similar=False
    )
    values = (payload.get("properties") or {}).get("values") or {}
    return ToolResult(
        data={"compound": payload["compound"], "properties": payload.get("properties")},
        summary=(
            f"{payload['compound']['name']}: MW {values.get('molecular_weight')}, "
            f"LogP {values.get('logp')}, TPSA {values.get('tpsa')}, "
            f"QED {values.get('qed')}."
        ),
        record_ids={"compound": [payload["compound"]["id"]]},
    )


async def _predict(context: ServiceContext, args: PredictInput) -> ToolResult:
    from theragraph.db.repositories import chemistry as chem_repo
    from theragraph.ml.inference import summarize_predictions

    if not args.compounds and not args.smiles:
        raise ValidationError("provide at least one compound id or SMILES string")

    inference = context.require_inference()
    molecules: list[tuple[str | None, str]] = []
    names: dict[str, str] = {}

    for identifier in args.compounds:
        compound = await chem_repo.get_compound(context.session, identifier)
        if compound is None:
            raise NotFoundError(f"compound {identifier!r} is not in the local corpus")
        molecules.append((compound.id, compound.canonical_smiles))
        names[compound.id] = compound.display_name
    for smiles in args.smiles:
        molecules.append((None, smiles))

    endpoints = args.endpoints or [
        item["endpoint"] for item in await inference.available_endpoints(context.session)
    ]
    if not endpoints:
        return ToolResult(
            data={"predictions": [], "available_endpoints": []},
            summary="No models are registered, so no predictions can be made.",
            is_empty=True,
            warnings=["Train models first: `theragraph ml train --endpoint solubility`."],
        )

    result = await inference.predict(context.session, molecules=molecules, endpoints=endpoints)
    payload = result.to_dict()
    payload["summary_by_endpoint"] = summarize_predictions(result.predictions)

    enriched = []
    for prediction in payload["predictions"]:
        enriched.append(
            {
                **prediction,
                "name": names.get(prediction.get("compound_id") or "", prediction.get("smiles")),
            }
        )
    payload["predictions"] = enriched

    out_of_domain = sum(1 for prediction in result.predictions if prediction.in_domain is False)
    return ToolResult(
        data=payload,
        summary=(
            f"{len(result.predictions)} prediction(s) across {len(endpoints)} endpoint(s) "
            f"using {', '.join(model['ref'] for model in result.models_used)}. "
            f"{out_of_domain} were outside the applicability domain."
        ),
        record_ids={
            "prediction": [
                prediction.prediction_id
                for prediction in result.predictions
                if prediction.prediction_id
            ]
        },
        is_empty=not result.predictions,
        warnings=(
            [
                f"{out_of_domain} prediction(s) are extrapolation beyond the training "
                "chemistry; treat them as unreliable."
            ]
            if out_of_domain
            else []
        )
        + [failure.get("error", "prediction failed") for failure in result.failures],
        state_patch={"predictions": enriched, "models_used": result.models_used},
    )


async def _prioritize(context: ServiceContext, args: PrioritizeInput) -> ToolResult:
    from theragraph.services import prioritization

    payload = await prioritization.prioritize(
        context,
        target_identifier=args.target,
        disease_identifier=args.disease,
        compound_ids=args.compounds,
        config_name=args.config_name,
        min_pchembl=args.min_pchembl,
        limit=args.limit,
        research_run_id=None,
    )
    candidates = payload.get("candidates") or []
    top = candidates[0] if candidates else None
    return ToolResult(
        data=payload,
        summary=(
            f"Ranked {len(candidates)} candidate(s) with the "
            f"{(payload.get('config') or {}).get('name')} configuration"
            + (f". Top: {top.get('name')} (score {top.get('total_score'):.3f})." if top else ".")
        ),
        record_ids={
            "compound": [candidate["compound_id"] for candidate in candidates],
            "ranking": [payload["ranking_id"]] if payload.get("ranking_id") else [],
        },
        is_empty=not candidates,
        warnings=payload.get("warnings") or [],
        state_patch={"ranking": payload},
    )


async def _compare(context: ServiceContext, args: CompareInput) -> ToolResult:
    from theragraph.services import prioritization

    payload = await prioritization.compare(
        context,
        compound_ids=args.compounds,
        target_identifier=args.target,
        disease_identifier=args.disease,
        config_name=args.config_name,
    )
    rows = payload.get("candidates") or []
    return ToolResult(
        data=payload,
        summary=(
            f"Compared {len(rows)} compound(s) across "
            f"{len(payload.get('objectives') or [])} objectives."
            + (f" {payload['note']}" if payload.get("note") else "")
        ),
        record_ids={"compound": [row["compound_id"] for row in rows]},
        is_empty=not rows,
        state_patch={"comparison": payload},
    )


async def _graph(context: ServiceContext, args: GraphInput) -> ToolResult:
    from theragraph.services import graph as graph_service

    payload = await graph_service.build_evidence_graph(
        context,
        disease_identifier=args.disease,
        target_identifier=args.target,
        compound_identifier=args.compound,
        max_targets=args.max_targets,
    )
    stats = payload["stats"]
    return ToolResult(
        data=payload,
        summary=(
            f"Evidence graph: {stats['n_nodes']} nodes, {stats['n_edges']} edges, "
            f"sources {stats['sources']}."
        ),
        record_ids={"graph_node": [node["id"] for node in payload["nodes"]]},
        is_empty=stats["n_nodes"] == 0,
        state_patch={"graph_stats": stats},
    )


async def _corpus(context: ServiceContext, args: CorpusInput) -> ToolResult:
    from theragraph.services import discovery

    payload = await discovery.corpus_summary(context)
    counts = payload["counts"]
    return ToolResult(
        data=payload,
        summary=(
            f"Corpus: {counts['diseases']} diseases, {counts['targets']} targets, "
            f"{counts['compounds']} compounds, {counts['bioactivities']} measured "
            f"activities across {len(payload['sources'])} sources."
        ),
        is_empty=counts["compounds"] == 0 and counts["targets"] == 0,
    )


async def _ingest_disease(context: ServiceContext, args: IngestDiseaseInput) -> ToolResult:
    from theragraph.ingestion.base import IngestionMode
    from theragraph.ingestion.pipelines import IngestionService, build_connectors

    connectors = build_connectors(context.settings, mode=IngestionMode.AUTO)
    service = IngestionService(context.settings, connectors, mode=IngestionMode.AUTO)
    try:
        outcome = await service.ingest_disease(
            context.session, args.efo_id, n_targets=args.n_targets
        )
    finally:
        await connectors.aclose()
    return ToolResult(
        data=outcome.summary(),
        summary=(
            f"Ingested {args.efo_id} from Open Targets {outcome.version}: {outcome.entity_counts}."
        ),
        record_ids={"dataset_version": [outcome.dataset_version_id or ""]},
    )


async def _ingest_bioactivity(context: ServiceContext, args: IngestBioactivityInput) -> ToolResult:
    from theragraph.ingestion.base import IngestionMode
    from theragraph.ingestion.pipelines import IngestionService, build_connectors

    connectors = build_connectors(context.settings, mode=IngestionMode.AUTO)
    service = IngestionService(context.settings, connectors, mode=IngestionMode.AUTO)
    try:
        outcome = await service.ingest_target_bioactivity(
            context.session, target_key=args.target, max_activities=args.max_activities
        )
    finally:
        await connectors.aclose()
    return ToolResult(
        data={**outcome.summary(), "warnings": outcome.warnings},
        summary=(f"Ingested ChEMBL bioactivity for {args.target}: {outcome.entity_counts}."),
        record_ids={"dataset_version": [outcome.dataset_version_id or ""]},
        warnings=outcome.warnings,
    )


# ---------------------------------------------------------------------- registry
TOOL_REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            name="search_entities",
            description=(
                "Search the ingested corpus for diseases, targets and compounds by name "
                "or identifier. Start here when the question names a disease, gene or "
                "drug. Returns only entities present locally unless include_upstream is "
                "set."
            ),
            input_model=SearchInput,
            handler=_search,  # type: ignore[arg-type]
            category="search",
        ),
        ToolSpec(
            name="get_disease_targets",
            description=(
                "Retrieve a disease and its associated therapeutic targets, ordered by "
                "the Open Targets association score. Use after identifying a disease to "
                "find candidate targets. Scores are retrieved verbatim from the source."
            ),
            input_model=DiseaseInput,
            handler=_disease_targets,  # type: ignore[arg-type]
            category="search",
        ),
        ToolSpec(
            name="get_target_overview",
            description=(
                "Full profile of one target: description, tractability buckets, safety "
                "liabilities, grouped evidence, associated diseases, and compounds with "
                "measured activity. The main tool for investigating a target."
            ),
            input_model=TargetInput,
            handler=_target_overview,  # type: ignore[arg-type]
            category="evidence",
        ),
        ToolSpec(
            name="get_target_evidence",
            description=(
                "Retrieve evidence records for a target (optionally in a disease "
                "context), filterable by evidence type, source and score. Use this to "
                "substantiate a specific claim; each record returns a citation key that "
                "must be used when citing it."
            ),
            input_model=EvidenceInput,
            handler=_target_evidence,  # type: ignore[arg-type]
            category="evidence",
        ),
        ToolSpec(
            name="get_compound",
            description=(
                "Full profile of a compound: structure, computed descriptors, measured "
                "bioactivities with assay metadata, and structurally similar compounds."
            ),
            input_model=CompoundInput,
            handler=_compound_overview,  # type: ignore[arg-type]
            category="chemistry",
        ),
        ToolSpec(
            name="find_similar_compounds",
            description=(
                "Tanimoto similarity search over Morgan fingerprints of the ingested "
                "corpus. Use to expand from a known active to structural neighbours."
            ),
            input_model=SimilarityInput,
            handler=_similar,  # type: ignore[arg-type]
            category="chemistry",
        ),
        ToolSpec(
            name="substructure_search",
            description=(
                "Find compounds containing a SMARTS substructure. Use for "
                "pharmacophore or chemotype questions."
            ),
            input_model=SubstructureInput,
            handler=_substructure,  # type: ignore[arg-type]
            category="chemistry",
        ),
        ToolSpec(
            name="compute_properties",
            description=(
                "Compute RDKit physicochemical descriptors and rule-of-thumb filters "
                "(Lipinski, Veber, CNS MPO) for a SMILES string or an ingested compound."
            ),
            input_model=PropertiesInput,
            handler=_properties,  # type: ignore[arg-type]
            category="chemistry",
        ),
        ToolSpec(
            name="predict_properties",
            description=(
                "Run TheraGraph's ML models on compounds or SMILES. Returns predicted "
                "values with uncertainty, calibrated confidence and applicability-domain "
                "status. These are PREDICTIONS, not measurements, and must be labelled "
                "as such in any output."
            ),
            input_model=PredictInput,
            handler=_predict,  # type: ignore[arg-type]
            category="ml",
            expensive=True,
        ),
        ToolSpec(
            name="prioritize_candidates",
            description=(
                "Run the transparent multi-objective prioritisation for a target's "
                "candidate compounds. Combines measured potency, target evidence, "
                "predicted properties, novelty and model confidence, and returns the "
                "full factor breakdown for every candidate."
            ),
            input_model=PrioritizeInput,
            handler=_prioritize,  # type: ignore[arg-type]
            category="ranking",
            expensive=True,
        ),
        ToolSpec(
            name="compare_candidates",
            description=(
                "Compare specific compounds side by side across every scored objective, "
                "with per-objective best/worst highlights."
            ),
            input_model=CompareInput,
            handler=_compare,  # type: ignore[arg-type]
            category="ranking",
        ),
        ToolSpec(
            name="build_evidence_graph",
            description=(
                "Build the disease -> target -> evidence -> compound -> assay -> "
                "prediction graph for a chosen root entity."
            ),
            input_model=GraphInput,
            handler=_graph,  # type: ignore[arg-type]
            category="evidence",
        ),
        ToolSpec(
            name="describe_corpus",
            description=(
                "Report what data is currently ingested: entity counts, sources, "
                "licences and dataset versions. Use when a question may be outside the "
                "corpus, or to state provenance."
            ),
            input_model=CorpusInput,
            handler=_corpus,  # type: ignore[arg-type]
            category="search",
        ),
        ToolSpec(
            name="ingest_disease",
            description=(
                "Fetch a disease and its associated targets from Open Targets into the "
                "corpus. Only use when the disease is absent locally and the user asked "
                "for it. Creates a new dataset version."
            ),
            input_model=IngestDiseaseInput,
            handler=_ingest_disease,  # type: ignore[arg-type]
            category="ingestion",
            mutates=True,
            expensive=True,
        ),
        ToolSpec(
            name="ingest_target_bioactivity",
            description=(
                "Fetch ChEMBL compounds, assays and measured activities for a target "
                "into the corpus. Use when a target has no local chemistry. Creates a "
                "new dataset version."
            ),
            input_model=IngestBioactivityInput,
            handler=_ingest_bioactivity,  # type: ignore[arg-type]
            category="ingestion",
            mutates=True,
            expensive=True,
        ),
    ]
}

READ_ONLY_TOOLS = [name for name, spec in TOOL_REGISTRY.items() if not spec.mutates]


def tool_schemas(*, include_mutating: bool = False) -> list[dict[str, Any]]:
    """Tool definitions for the model, in a stable order so the prompt cache holds."""
    return [
        spec.schema()
        for name, spec in sorted(TOOL_REGISTRY.items())
        if include_mutating or not spec.mutates
    ]


def get_tool(name: str) -> ToolSpec:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        raise ValidationError(
            f"unknown tool {name!r}",
            details={"available_tools": sorted(TOOL_REGISTRY)},
        )
    return spec


@dataclass
class ToolExecution:
    """Outcome of one tool invocation, including timing and failure detail."""

    name: str
    arguments: dict[str, Any]
    result: ToolResult | None
    status: Literal["succeeded", "invalid_arguments", "failed"]
    latency_ms: float
    error: str | None = None
    error_code: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


async def execute_tool(
    context: ServiceContext,
    name: str,
    arguments: dict[str, Any],
    *,
    allow_mutating: bool = False,
) -> ToolExecution:
    """Validate arguments, run the tool, and never let it raise into the agent loop.

    A failed tool must produce a *usable* error for the model (so it can adapt) rather
    than aborting the run. Errors are returned as structured results with a stable code.
    """
    stopwatch = Stopwatch()
    try:
        spec = get_tool(name)
    except ValidationError as exc:
        AGENT_TOOL_CALLS.labels(tool=name, outcome="unknown_tool").inc()
        return ToolExecution(
            name=name,
            arguments=arguments,
            result=None,
            status="invalid_arguments",
            latency_ms=0.0,
            error=exc.message,
            error_code=exc.code,
        )

    if spec.mutates and not allow_mutating:
        AGENT_TOOL_CALLS.labels(tool=name, outcome="forbidden").inc()
        return ToolExecution(
            name=name,
            arguments=arguments,
            result=None,
            status="failed",
            latency_ms=0.0,
            error=(
                f"{name} modifies the corpus and is disabled for this run; "
                "re-run with allow_ingestion=true to permit it"
            ),
            error_code="tool_not_permitted",
        )

    try:
        parsed = spec.input_model.model_validate(arguments)
    except Exception as exc:
        AGENT_TOOL_CALLS.labels(tool=name, outcome="invalid_arguments").inc()
        return ToolExecution(
            name=name,
            arguments=arguments,
            result=None,
            status="invalid_arguments",
            latency_ms=0.0,
            error=f"invalid arguments for {name}: {exc}",
            error_code="validation_error",
        )

    try:
        with stopwatch:
            result = await spec.handler(context, parsed)
    except TheraGraphError as exc:
        AGENT_TOOL_CALLS.labels(tool=name, outcome="error").inc()
        AGENT_TOOL_LATENCY.labels(tool=name).observe(stopwatch.elapsed_ms / 1000)
        log.warning("agent.tool_failed", tool=name, error=exc.message, code=exc.code)
        return ToolExecution(
            name=name,
            arguments=arguments,
            result=None,
            status="failed",
            latency_ms=stopwatch.elapsed_ms,
            error=exc.message,
            error_code=exc.code,
        )
    except Exception as exc:  # pragma: no cover - defensive
        AGENT_TOOL_CALLS.labels(tool=name, outcome="error").inc()
        log.error("agent.tool_crashed", tool=name, error=str(exc), exc_info=True)
        return ToolExecution(
            name=name,
            arguments=arguments,
            result=None,
            status="failed",
            latency_ms=stopwatch.elapsed_ms,
            error=f"{type(exc).__name__}: {exc}",
            error_code="internal_error",
        )

    AGENT_TOOL_CALLS.labels(tool=name, outcome="empty" if result.is_empty else "succeeded").inc()
    AGENT_TOOL_LATENCY.labels(tool=name).observe(stopwatch.elapsed_ms / 1000)
    return ToolExecution(
        name=name,
        arguments=parsed.model_dump(mode="json"),
        result=result,
        status="succeeded",
        latency_ms=stopwatch.elapsed_ms,
    )
