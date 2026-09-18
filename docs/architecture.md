# Architecture

TheraGraph is a **modular monolith**: one FastAPI application, one PostgreSQL database,
one worker process running the same code. Modules have explicit boundaries and talk to
each other through typed function calls, not HTTP. That is a deliberate choice — see
[Why a monolith](#why-a-monolith).

## System view

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Browser                                                                     │
│  React 18 + TypeScript + Vite  ·  TanStack Query  ·  Recharts               │
│  Dashboard │ Targets │ Compounds │ Compare │ Assistant │ Run │ Evaluation    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │  JSON over HTTP (nginx serves the SPA,
                                │  proxies /api to the API container)
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  FastAPI application  (theragraph.main)                                      │
│                                                                              │
│  middleware:  request id → access log → body size → rate limit → CORS → gzip │
│  routers:     search  targets  diseases  compounds  predictions  ranking     │
│               graph  agent  runs  models  ingestion  jobs  system            │
├──────────────────────────────────────────────────────────────────────────────┤
│  services/        the use cases. One module per view of the product.          │
│    discovery  targets  compounds  graph  prioritization  runs  reports       │
├───────────────┬───────────────┬───────────────┬──────────────┬───────────────┤
│  agent/       │  ml/          │  ranking/     │  chem/       │  ingestion/   │
│  plan→execute │  baseline +   │  desirability │  RDKit:      │  Open Targets │
│  →validate→   │  GIN, splits, │  curves,      │  standardise │  ChEMBL       │
│  synthesise   │  calibration, │  weights,     │  descriptors │  PubChem      │
│  15 tools     │  registry     │  filters      │  fingerprints│  Europe PMC   │
│  eval suite   │  inference    │  explanations │  similarity  │  MoleculeNet  │
├───────────────┴───────────────┴───────────────┴──────────────┴───────────────┤
│  db/            28 tables · repositories · Alembic migrations                 │
│  core/          config  logging  metrics  tracing  cache  storage  security   │
│  jobs/          durable queue: Redis transport, PostgreSQL truth              │
└──────┬──────────────────┬──────────────────┬───────────────────┬──────────────┘
       │                  │                  │                   │
┌──────▼──────┐   ┌───────▼──────┐   ┌───────▼───────┐   ┌───────▼──────────┐
│ PostgreSQL  │   │    Redis     │   │ S3 / MinIO    │   │  Anthropic API   │
│ 16          │   │ cache +      │   │ model         │   │  (optional — the │
│ scientific  │   │ queue        │   │ artifacts,    │   │  agent falls back│
│ record      │   │ transport    │   │ reports       │   │  to a determin-  │
│             │   │              │   │               │   │  istic planner)  │
└─────────────┘   └──────────────┘   └───────────────┘   └──────────────────┘

        ┌────────────────────────────┐        ┌──────────────────────────┐
        │  worker process            │        │  MCP server (stdio/HTTP) │
        │  same image, same code,    │        │  exposes the same 15     │
        │  consumes the job table    │        │  tools to MCP clients    │
        └────────────────────────────┘        └──────────────────────────┘
```

## The data path

Every scientific value in the UI can be traced back along this path, and every arrow
records where the data came from and when.

```
  Public source                 Normalised                  Stored with provenance
  ─────────────                 ──────────                  ──────────────────────
  Open Targets GraphQL   ──┐
  (release 26.06)          │    NormalizedBatch             source
                           │    ├── rows                    source_identifier
  ChEMBL REST              │    ├── rejects                 source_url
  (ChEMBL_37)            ──┼──► └── quality report   ──►    retrieved_at
                           │                                dataset_version_id ──┐
  PubChem PUG REST         │    identity keys:                                   │
  Europe PMC REST        ──┤    InChIKey (compounds)     ┌──────────────────────┐ │
  MoleculeNet CSV        ──┘    Ensembl id (targets)     │ dataset_version      │◄┘
                                EFO/MONDO id (diseases)  │  content_hash        │
                                                         │  record_count        │
                                                         │  retrieved_at        │
                                                         └──────────────────────┘
```

A `dataset_version` row is an immutable snapshot: its `content_hash` is computed from
the normalised records in an order-independent way, so re-ingesting the same upstream
payload produces the same hash. A research run stores the dataset versions it read,
which is what makes "reproduce this run" a meaningful operation rather than a hope.

## Request lifecycles

### A synchronous read (`GET /api/v1/targets/BACE1`)

```
request id assigned → access log started → rate limit checked
  → router validates the path/query against a pydantic model
    → services.targets.overview()
      → repositories read target, evidence, tractability, known chemistry
      → chem/ computes descriptors for the returned compounds
      → response model serialises, marking each field as measured or computed
  → access log closed with status + duration → Prometheus histogram observed
```

### A research run (`POST /api/v1/agent/research`)

```
 1. runs.create_run()          allocate RUN-2026-00042 (atomic sequence)
 2. agent.plan()               LLM or deterministic planner → ResearchPlan
                               (typed steps, arguments, dependencies)
 3. for each step:
      bindings.resolve()       {{prev.top_target}} → a concrete id
      tools.execute()          typed input → typed output, logged as a tool_call
      state.update()           structured intermediate state, never free text
      adapt()                  a failed or empty step can insert a recovery step
 4. validator.validate()       claims citing records the run never retrieved are
                               removed. This is the anti-fabrication gate.
 5. agent.synthesise()         findings + citations, facts separated from predictions
 6. reports.generate()         markdown + JSON, every claim carrying its citation keys
 7. runs.finish()              status, runtime, token usage, cost, reproducibility hash
```

Each numbered step writes a row. The Research Run view is a rendering of those rows, so
what the user sees is the execution record, not a summary of it.

### A background job (`POST /api/v1/jobs/train`)

```
submit → idempotency key derived from (task, payload)
       → existing job returned if the key is already known
       → row INSERTed in PostgreSQL (the truth), id pushed to Redis (the transport)
worker → BLPOP, or a periodic sweep of the table if Redis dropped the id
       → claim by conditional UPDATE ... WHERE status = 'pending'  (no double-claim)
       → lease held with locked_until; a reaper requeues expired leases
       → on failure: attempts += 1, jittered exponential backoff, error appended
       → after max_attempts: status = 'dead', visible in the jobs API
```

PostgreSQL is the source of truth and Redis is only a wake-up channel, so losing Redis
delays jobs rather than losing them.

## Module boundaries

| Module | Owns | Never does |
|---|---|---|
| `core/` | config, logging, metrics, tracing, cache, storage, hashing, ids, errors | import from `services/`, `api/`, `db/models` |
| `db/` | the schema, repositories, migrations | business rules, HTTP concerns |
| `chem/` | RDKit: standardisation, descriptors, fingerprints, similarity | database access |
| `ingestion/` | one connector per source, normalisation, quality rules | ranking, prediction |
| `ml/` | featurisation, splits, training, evaluation, registry, inference | HTTP, ingestion |
| `ranking/` | desirability curves, weights, filters, explanations | prediction, retrieval |
| `services/` | use cases; composes the modules above | raw SQL, RDKit calls |
| `agent/` | planning, tool execution, validation, synthesis, evals | direct SQL |
| `api/` | HTTP: validation, status codes, serialisation | business rules |
| `jobs/` | durable queue, worker loop, task registry | the work itself |

`ServiceContext` (settings + session + cache + object store) is the single thing passed
into services and tools, which is why the agent's tools and the REST endpoints run the
same code with the same guarantees.

## Why a monolith

The spec asked for this explicitly, and it is the right call here:

* **The expensive part is the data, not the traffic.** Nothing in this system needs to
  scale independently. Splitting `ml/` into a service would add a network hop, a
  serialisation format and a deployment target to solve a problem that does not exist.
* **The evidence graph is a join.** Disease → target → evidence → compound → assay →
  prediction is a query across six tables. Across services it becomes six API calls and
  a consistency problem.
* **Transactions are the point.** A research run writes to `research_run`,
  `agent_execution`, `tool_call`, `prediction` and `report`. One transaction makes a
  half-recorded run impossible.
* Where there *is* a clear reason, the process is already separate: the **worker** (a
  slow training run must not hold an HTTP connection) and the **MCP server** (a
  different protocol for a different kind of client). Both run the same image.

## Technology choices and the reasons

| Choice | Why, and what was rejected |
|---|---|
| FastAPI + pydantic | The schemas are the validation, the docs and the TypeScript contract. The agent's tool schemas come from the same models. |
| SQLAlchemy 2.0 async | Typed ORM for reads; Core for bulk upserts, where the ORM's per-row overhead would dominate ingestion. |
| PostgreSQL | JSONB for the parts of an upstream payload worth keeping verbatim, trigram indexes for name search, real transactions. Column types are `with_variant`, so the test suite runs the same models on SQLite. |
| Alembic, dialect-agnostic | One migration chain that applies to both PostgreSQL and SQLite; `alembic check` in CI fails when the models drift from it. |
| RDKit | The reference implementation for the chemistry. Descriptors are named, versioned and labelled with units, because an unlabelled LogP is not data. |
| Pure-PyTorch GNN | A GINE-style network using `index_add_`/`scatter_reduce_` rather than PyTorch Geometric: one fewer compiled dependency pinned to a CUDA version, for a model this size. |
| Redis + PostgreSQL queue | Celery/RQ would work; neither gives idempotency keys, leases and a durable audit row without extra work. ~200 lines buys exactly-once-enough delivery that the API can query. |
| Anthropic tool calling | Typed tools with `strict: true`. A deterministic planner implements the same interface, so the whole workflow — tools, validation, reports, evals — runs with no API key and no network. |

## Observability

Structured logs (structlog, JSON in production) carry `request_id` and `run_id` on every
line, so one run is one `grep`. Prometheus metrics cover HTTP latency and error rates by
route, DB query duration, cache hit rate, job queue depth and outcomes, ML inference
latency by endpoint, agent tool latency and failures, and token usage and cost per run.
OpenTelemetry spans are emitted when an OTLP endpoint is configured and are off
otherwise. `/metrics` is served on the API port but blocked at the edge by nginx — it is
for the scrape target, not the internet.

## Related documents

* [README](../README.md) — what this is, and how to run it
* [Data & ingestion](data.md) — the sources, the normalisation, the provenance record
* [Machine learning](ml.md) — features, splits, evaluation, calibration, uncertainty
* [The research agent](agent.md) — planning, tools, validation, evaluation
* [Operations](operations.md) — deployment, configuration, monitoring, runbook
* [Design decisions and limitations](decisions.md) — what was traded away, and what this
  system cannot tell you
