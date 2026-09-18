# TheraGraph

**An evidence-grounded computational drug-discovery workbench.**

TheraGraph takes a disease, retrieves the public evidence linking it to therapeutic
targets, finds the compounds with measured activity against those targets, predicts their
molecular properties, and ranks them for follow-up — while keeping a permanent, auditable
line between **what was retrieved from a public source** and **what a model predicted**.

> TheraGraph does not discover drugs and cannot validate one. It aggregates public
> evidence, computes molecular properties, predicts a small set of physicochemical and
> toxicity endpoints, and produces transparent prioritisation scores for human
> follow-up. Nothing it outputs is evidence that a compound treats a disease.

```
   Disease  →  Target  →  Evidence  →  Compound  →  Properties  →  Ranking  →  Report
  MONDO_…      BACE1      PMID +       CHEMBL…      predicted      weighted    cited,
              ENSG…       OT score     pChEMBL      ± uncertainty  objectives  reproducible
  └──────── retrieved, with provenance ─────────┘  └── predicted, with uncertainty ──┘
```

---

## Contents

| Document | What is in it |
|---|---|
| **[Architecture](docs/architecture.md)** | System diagram, request lifecycles, module boundaries, technology choices |
| **[Data & ingestion](docs/data.md)** | The five public sources, normalisation, quality rules, provenance, how to run and re-record ingestion |
| **[Machine learning](docs/ml.md)** | Features, scaffold splits, metrics, calibration, uncertainty, error analysis, measured results |
| **[The research agent](docs/agent.md)** | Plan → execute → validate → synthesise, the 15 tools, reproducible runs, the evaluation suite |
| **[Example workflows](docs/workflows.md)** | Three worked examples with every API call |
| **[Operations](docs/operations.md)** | Configuration, security, deployment, monitoring, runbook, CI |
| **[Design decisions & limitations](docs/decisions.md)** | What was traded away, and what this system cannot tell you |
| **[API reference](docs/openapi.json)** | OpenAPI 3.1; also served live at `/docs` |

---

## Quick start

Needs Python 3.11, [uv](https://docs.astral.sh/uv/) and Node 20+. No API keys, no network
access, and no PostgreSQL required to get the whole thing running.

```bash
make setup                                        # venv + backend + frontend deps
DATABASE_URL=sqlite+aiosqlite:///./tg.sqlite3 make migrate
make seed-smoke                                   # build the corpus from committed fixtures
make train                                        # train and register the models
make api                                          # API on :8000
make frontend-dev                                 # UI on :5173
```

Or with Docker (PostgreSQL, Redis, MinIO, API, two workers, the UI):

```bash
cp .env.example .env && make docker-up
```

Then open <http://localhost:5173>, or walk the demo from the terminal with
[docs/workflows.md](docs/workflows.md).

**Everything above is offline.** Every upstream API response used to build the shipped
corpus is committed under `data/seed/fixtures/`, the four MoleculeNet benchmark CSVs are
committed under `data/raw/moleculenet/`, and with no `ANTHROPIC_API_KEY` the research
agent runs a deterministic planner over the same tools with the same validation and the
same reports. A demo that needs the internet is a demo that breaks during the demo.

---

## What it does

### 1. Disease and target discovery

Progressive-tier search (exact → prefix → contains → synonym) over diseases, targets and
compounds, so an exact match never loses to a substring. Targets come back ordered by
Open Targets' own association score with its per-datatype breakdown, tractability buckets
and known safety liabilities.

### 2. Evidence aggregation with provenance on every row

Evidence from Open Targets' 25 datasources, mapped onto a fixed vocabulary
(`genetic_association`, `known_drug`, `literature`, `animal_model`, `expression`,
`pathway`, `functional_genomics`, `somatic_mutation`, `biomarker`), each record carrying
its source, upstream identifier, resolvable URL, the source's own confidence score, its
retrieval date and the content-hashed dataset version it belongs to. Literature evidence
is linked to Europe PMC publication metadata.

### 3. Compound discovery and molecular intelligence

RDKit throughout: standardisation (largest fragment, neutralised), canonical SMILES,
InChI, InChIKey as the cross-source identity key, Bemis–Murcko scaffolds. Morgan / MACCS /
pattern fingerprints stored as packed bytes. Tanimoto similarity search with the popcount
bound pushed into SQL as an index range scan, and SMARTS substructure search with
fingerprint pre-screening. 16 curated descriptors — MW, LogP, TPSA, HBD, HBA, rotatable
bonds, QED and more — each with a label, a unit and a one-line explanation.

### 4. Two model families, evaluated properly

A fingerprint+descriptor **Random Forest / XGBoost** baseline and a **GINE-style graph
neural network** in pure PyTorch, trained on the identical split and compared with
bootstrap confidence intervals. Scaffold splits are the headline; random splits are
reported *only* as the labelled optimism reference, with the gap computed. Calibration
(Platt vs isotonic, auto-selected, refused when it does not help), MC-dropout and
ensemble uncertainty, applicability-domain flags, and error analysis by property bucket,
uncertainty decile and scaffold familiarity.

Measured, offline, on the committed MoleculeNet snapshots:

<!-- METRICS_TABLE -->

### 5. Transparent multi-objective prioritisation

Nine objectives across measured potency, retrieved evidence support, predicted solubility
and toxicity, drug-likeness, molecular weight, lipophilicity, structural novelty and
model confidence. Every candidate's score decomposes into each factor's raw value,
desirability, weight and contribution — and whether it was discounted for low confidence
or an out-of-domain prediction. Three presets ship, weights are user-adjustable, weights
renormalise when data is missing, and hard filters exclude with a stated reason.

### 6. A research agent that cannot fabricate evidence

An explicit **plan → execute → adapt → validate → synthesise → report** workflow over 15
typed tools. The model decides *what to do*; tools do every retrieval and computation, and
they are the same functions the REST API calls. Then the validator checks each claim's
citations against the records the run actually retrieved and **deletes** the unsupported
ones, recording the deletion.

Latest evaluation run — 6 cases, gated in CI:

| Metric | Threshold | Result |
|---|---|---|
| Pass rate | ≥ 0.80 | **1.00** |
| Tool-selection recall | ≥ 0.80 | **1.00** |
| Citation correctness | = 1.00 | **1.00** |
| Structured-output validity | = 1.00 | **1.00** |
| Honesty on unanswerable questions | = 1.00 | **1.00** |
| **Hallucination rate** | **= 0.00** | **0.00** |

Two of the six cases exist only to check that a question the corpus cannot answer gets
"that is not in the corpus" rather than an invention.

### 7. Reproducible research runs

Every run gets an id like `RUN-2026-00183` and stores the query, the dataset versions
read, the model versions used, the prompt template hashes, every tool call with its input
and output, the plan and any adaptations, the predictions, the ranking configuration, the
report, the timings, the token usage and the cost — plus a reproducibility hash over the
scientific inputs only. `GET /runs/{id}/replay-plan` reports what has drifted since
(corpus growth, retrained models, changed prompts) *before* you re-run, so a different
answer is explained rather than mysterious.

### 8. The interface

React 18 + TypeScript across seven views: Research Dashboard, Target Explorer, Compound
Explorer, Candidate Comparison, AI Research Assistant, Research Run, Model Evaluation.
The evidence graph is interactive and every edge names its source, evidence type, upstream
identifier, score and retrieval date. Measured and predicted values are visually distinct
everywhere, including in a colourblind-safe chart palette validated against adjacent-pair
CVD separation. The assistant shows its plan, its tool calls, its retrieved evidence and a
written reasoning summary — never raw chain-of-thought.

---

## Repository layout

```
backend/
  src/theragraph/
    core/          config, logging, metrics, tracing, cache, storage, security, errors
    db/            28 models, repositories, Alembic migrations
    chem/          RDKit: standardisation, descriptors, fingerprints, similarity
    ingestion/     one connector per source + normalisation + quality rules
    ml/            features, splits, baseline, GNN, evaluation, calibration, registry
    ranking/       desirability curves, weights, filters, explanations
    services/      the use cases: discovery, targets, compounds, graph, ranking, runs
    agent/         planner, tools, bindings, validator, orchestrator, evals
    api/           13 routers, middleware, error mapping
    jobs/          durable queue + worker
    mcp_server/    the same tools over MCP
    tests_support/ the in-process demo corpus (shipped: CI and the CLI use it)
  tests/           unit · integration · api · ml · agent · e2e   (428 tests)
  alembic/         migrations, dialect-agnostic
frontend/
  src/{pages,components,lib,types}                                (43 tests)
data/
  seed/fixtures/   recorded upstream responses — the offline corpus
  raw/moleculenet/ the four benchmark CSVs
docs/              architecture · data · ml · agent · workflows · operations · decisions
deploy/            nginx configuration for the SPA + API proxy
.github/workflows/ CI: 6 jobs, all offline
```

## Common commands

```bash
make help              # every target, described
make check             # lint + typecheck + tests, everything CI runs
make test              # 428 backend + 43 frontend tests
make coverage          # backend coverage report
make evals             # the agent evaluation suite, with its CI thresholds
make demo              # bootstrap → seed → train → evals, from scratch, offline
make openapi           # regenerate docs/openapi.json
make docker-up         # the full stack
```

## Engineering

| | |
|---|---|
| **Backend** | Python 3.11, FastAPI, SQLAlchemy 2.0 async, PostgreSQL 16, Redis, Alembic, PyTorch, scikit-learn, RDKit, S3/MinIO |
| **Frontend** | React 18, TypeScript, Vite, Tailwind, TanStack Query, Recharts, smiles-drawer |
| **Tests** | 428 backend (unit, integration, API, database, ML pipeline, agent, end-to-end) at 77% coverage, 30 modules at 100%; 43 frontend |
| **Quality gates** | ruff lint + format, mypy (strict, zero errors across 116 modules), eslint (zero warnings), `tsc --noEmit`, `alembic check` on both dialects, and the agent eval thresholds |
| **CI** | 6 offline jobs: backend, backend-on-PostgreSQL, migrations round-trip, frontend, agent evals, container images |
| **Deploy** | Multi-stage Docker images running as non-root, Compose for local and a fail-closed production overlay, health checks, structured logs, Prometheus metrics, optional OpenTelemetry |

Architecture: a **well-structured modular monolith**. One FastAPI application, one
database, typed calls across explicit module boundaries. Two processes are split out
because there is a real reason — the **worker** (a slow training run must not hold an HTTP
connection) and the **MCP server** (a different protocol for a different client) — and
both run the same image. The reasoning is in
[docs/architecture.md](docs/architecture.md#why-a-monolith).

## Data sources and licences

| Source | Version | Licence |
|---|---|---|
| [Open Targets Platform](https://platform.opentargets.org/) | release 26.06 | CC0 1.0 (evidence inherits its datasource licence) |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | ChEMBL_37 | CC BY-SA 3.0 |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | rolling | Public domain |
| [Europe PMC](https://europepmc.org/) | rolling | Metadata free to use |
| [MoleculeNet](https://moleculenet.org/) | content-hashed snapshot | MIT (distribution) |

Full citations live in the `data_source` table and are returned by
`GET /api/v1/ingestion/sources`, so the UI can show a licence next to the data it draws.

## Limitations

Read [docs/decisions.md](docs/decisions.md#limitations) before drawing any scientific
conclusion. The short version: the property models are trained on public benchmarks of a
few thousand molecules, their scaffold-split confidence intervals are wide and stated,
out-of-domain predictions are extrapolations and are flagged as such, evidence scores are
Open Targets' judgement rather than TheraGraph's, the corpus covers three disease areas,
and **nothing here is a claim about clinical efficacy**.

## Licence

Apache-2.0. The public datasets carry their own licences, listed above.
