# The research agent

The assistant is not "an LLM with a prompt." It is an explicit **plan → execute →
validate → synthesise** workflow in which the model chooses *what to do* and typed tools
do *all* of the retrieval and computation. The model never produces a scientific value,
and a claim that cites a record the run did not actually retrieve is deleted before the
user sees it.

## The workflow

```
  question + parameters
          │
          ▼
  ┌───────────────┐   ResearchPlan: objective, strategy, typed steps with
  │ 1. PLAN       │   arguments, dependencies, optional flags, open questions,
  │               │   ranking preset.  LLM (structured output) or the
  └───────┬───────┘   deterministic planner.
          ▼
  ┌───────────────┐   For each step:
  │ 2. EXECUTE    │     bind placeholders  {{prev.top_target}} → ENSG00000186868
  │               │     validate input against the tool's pydantic model
  │               │     run the tool (the same code the REST API runs)
  │               │     write a tool_call row: input, output, latency, status
  │               │     update structured state (never free text)
  └───────┬───────┘
          ▼
  ┌───────────────┐   A step that fails or returns nothing can insert a recovery
  │ 3. ADAPT      │   step — broaden a similarity threshold, resolve an entity,
  │               │   fall back to a different evidence class. Bounded, logged.
  └───────┬───────┘
          ▼
  ┌───────────────┐   Every claim's citations are checked against the records the
  │ 4. VALIDATE   │   run retrieved. Unsupported claims are REMOVED and counted.
  │               │   This is the anti-fabrication gate, and it is not optional.
  └───────┬───────┘
          ▼
  ┌───────────────┐   Findings, each tagged retrieved_fact | model_prediction |
  │ 5. SYNTHESISE │   interpretation, with citation keys and stated uncertainty.
  └───────┬───────┘
          ▼
  ┌───────────────┐   Markdown + JSON. Facts and predictions in separate sections.
  │ 6. REPORT     │   Every citation resolves to a source, an identifier and a
  └───────────────┘   retrieval date.
```

Each stage writes rows: `research_run`, `agent_execution` (one per stage),
`tool_call` (one per tool invocation), `report`. The Research Run view renders those
rows. What the user sees *is* the execution record.

## Tools

Fifteen tools, thirteen read-only and two that mutate (they ingest from public APIs and
are disabled when `ALLOW_LIVE_INGESTION=false`):

| Tool | Category | Mutates | What it does |
|---|---|---|---|
| `search_entities` | search | no | Resolve names to diseases, targets, compounds in the corpus |
| `describe_corpus` | search | no | Report what is ingested: counts, sources, licences |
| `get_disease_targets` | search | no | A disease and its targets, ordered by Open Targets score |
| `get_target_overview` | evidence | no | One target: description, tractability, safety, known chemistry |
| `get_target_evidence` | evidence | no | Evidence records, filterable by evidence class |
| `build_evidence_graph` | evidence | no | Disease → target → evidence → compound → assay → prediction |
| `get_compound` | chemistry | no | Structure, descriptors, measured activities, predictions |
| `compute_properties` | chemistry | no | RDKit descriptors and rule-of-thumb filters |
| `find_similar_compounds` | chemistry | no | Tanimoto search over Morgan fingerprints |
| `substructure_search` | chemistry | no | SMARTS substructure match |
| `predict_properties` | ml | no | Run the registered models; returns values *and* uncertainty |
| `prioritize_candidates` | ranking | no | Multi-objective ranking with per-factor contributions |
| `compare_candidates` | ranking | no | Side-by-side across every scored objective |
| `ingest_disease` | ingestion | **yes** | Fetch a disease + targets from Open Targets |
| `ingest_target_bioactivity` | ingestion | **yes** | Fetch ChEMBL compounds, assays, activities |

Schemas are generated from the pydantic models with `strict: true` and `$defs` inlined,
so the model cannot pass an argument the tool would reject. Inputs are validated again
on the way in, because a schema is a contract, not a guarantee.

Tool outputs are **validated before use**: an empty result is an empty result, not an
invitation to fill the gap. `get_target_overview` on a gene symbol that does not exist
returns "not in the corpus", and the plan's next step fails at resolution rather than
producing a confident empty profile.

## Placeholder binding

Steps reference earlier results by name rather than by the model re-typing an identifier:

| Placeholder | Resolves to |
|---|---|
| `{{prev.top_target}}` | the highest-scoring target from the previous step |
| `{{step_2.compound_ids}}` | a named field from a specific step's output |
| `{{state.disease_efo_id}}` | accumulated run state |

This matters for correctness, not just convenience. A model asked to repeat
`ENSG00000186868` across four steps will eventually mistype it. Binding means the
identifier is carried by the runtime, and an unresolved *required* argument fails the
step loudly — while an unresolved *optional* one is simply dropped.

Identifiers scraped from the question text are treated as hypotheses. A bare gene symbol
in a sentence may be a typo, an alias or nothing at all, so those plans begin with a
verification search; caller-supplied parameters, which the UI already resolved against
the corpus, are authoritative.

## Two backends, one workflow

```
AGENT_LLM_BACKEND=anthropic     # claude-opus-5, adaptive thinking, structured output
AGENT_LLM_BACKEND=deterministic # rule-based planner, no network, no key
```

The deterministic backend is not a stub. It classifies intent from the question,
constructs a real plan over the real tools, and runs the same execution, validation,
synthesis and report code. That is what lets the whole product — including the eval
suite and CI — run with no API key and no network access. With a key, the LLM plans;
everything downstream is identical.

Token usage and cost are recorded per run from the API's own usage numbers and priced
per model, so `GET /api/v1/runs/usage` reports what the workflow actually cost.

## Reproducible runs

Each run gets an id like `RUN-2026-00183`, allocated by an atomic
`UPDATE ... RETURNING` on a counter row (no gaps, no duplicates under concurrency), and
stores:

| Recorded | Why |
|---|---|
| query, parameters | the question as asked |
| dataset versions read | the corpus state, by content hash |
| model versions used | which predictor produced each number |
| prompt template hashes | prompts change; a run says which one it used |
| every tool call | input, output, latency, status, error |
| plan and adaptations | what was intended and what changed |
| predictions | with uncertainty and applicability flags |
| ranking configuration | weights and curves, not just the result |
| final report | markdown and JSON |
| timings, tokens, cost, errors | operational truth |
| `reproducibility_hash` | over the scientific inputs only |

The hash deliberately excludes non-scientific parameters (an idempotency key, a UI
preference) — otherwise two identical experiments would hash differently because someone
clicked a different button. Lineage lives in `parent_run_id` / `replay_of_run_id`.

```bash
GET  /api/v1/runs/{id}                 # the full trace
GET  /api/v1/runs/{id}/replay-plan     # what would differ if re-run now, and why
POST /api/v1/runs/{id}/replay          # re-execute; a new run linked to the original
POST /api/v1/runs/compare              # two runs, field by field
```

`replay-plan` is the interesting one: it reports **drift** — the corpus has since grown,
a model was retrained, a prompt changed — before you re-run, so a different answer is
explained rather than mysterious.

## Evaluation

`theragraph agent eval` runs six cases and gates on aggregate thresholds. CI runs it on
every push.

| Case | Category | What it catches |
|---|---|---|
| `disease_to_targets` | discovery | Target retrieval that returns nothing, or unsourced target claims |
| `target_evidence_depth` | evidence | Evidence summarised without citations, or evidence classes conflated |
| `prioritise_candidates` | prioritisation | Predictions not labelled as predictions; a ranking built on zero candidates |
| `structural_neighbours` | chemistry | Similarity search silently returning nothing; predictions served without applicability flags |
| `unknown_disease_honesty` | negative | **Inventing** targets for a disease that is not in the corpus |
| `nonexistent_target_honesty` | negative | Producing a confident profile for a gene symbol that does not exist |

Measured per case and aggregated:

| Metric | Threshold | Latest run |
|---|---|---|
| `pass_rate` | ≥ 0.80 | **1.00** |
| `tool_selection_recall` | ≥ 0.80 | **1.00** |
| `citation_correctness` | = 1.00 | **1.00** |
| `structured_output_validity` | = 1.00 | **1.00** |
| `negative_case_honesty` | = 1.00 | **1.00** |
| `hallucination_rate` | **= 0.00** | **0.00** |
| `retrieval_accuracy` | reported | 1.00 |
| `faithfulness` | reported | 1.00 |
| `prediction_labelling` | reported | 1.00 |
| `tool_selection_precision` | reported | 0.22 |
| latency p50 / max | reported | 489 ms / 3160 ms |
| tokens / cost | reported | 0 / $0.00 (deterministic backend) |

Two notes on reading that table honestly:

* **`hallucination_rate` has a zero threshold.** Any claim that survives to the report
  while citing a record the run never retrieved fails the build. There is no budget for
  "mostly grounded."
* **`tool_selection_precision` is 0.22 and is not gated.** The deterministic planner
  calls more tools than a case strictly needs — it builds the evidence graph and runs
  prioritisation even for a pure evidence question. That costs latency, not correctness,
  so it is reported rather than enforced. Gating it would pressure the planner to skip
  work that makes the run's output more complete.

## MCP

The same fifteen tools are exposed over the Model Context Protocol, so an external MCP
client (Claude Desktop, an IDE) can query the corpus directly:

```bash
theragraph mcp                 # stdio
make mcp
```

Schemas are synthesised from the same pydantic models via `inspect.Signature`, flattened
to the shapes MCP expects. There is one definition of each tool, not two.

## What the agent will not do

* It will not produce a scientific value. Numbers come from tools, which come from the
  database or a registered model.
* It will not cite a record the run did not retrieve. The validator removes the claim and
  records the removal in the run.
* It will not report a prediction as a measurement. Findings are typed, and the report
  keeps them in separate sections.
* It will not answer a question the corpus cannot support. "That disease is not in the
  corpus; here is how to ingest it" is the correct answer, and two eval cases exist
  solely to keep it that way.
* Hidden reasoning is not surfaced. The UI shows the plan, the tool calls, the retrieved
  evidence and a written reasoning *summary* — not raw chain-of-thought.
