# Design decisions and limitations

What this system deliberately does and does not do, and why. The limitations section is
the more important half.

## Decisions

### 1. Retrieved evidence and model predictions are different kinds of thing, everywhere

Not a UI convention — a property of the data model. Every prediction row points at the
`ml_model` that produced it and carries its uncertainty and applicability flag; every
evidence row points at a `dataset_version`, a source identifier and a retrieval date.
The API response types keep them in separate fields, the report keeps them in separate
sections, and the agent's findings are typed `retrieved_fact | model_prediction |
interpretation`.

The alternative — one "value" field with a `is_predicted` boolean — collapses under its
own weight the first time someone averages across both.

### 2. The LLM never produces a scientific value

The model chooses *what to do*. Tools do the retrieval and computation, and they are the
same functions the REST API calls. Then the validator checks every claim's citations
against the records the run actually retrieved and **deletes** the ones that do not
resolve, recording the deletion.

This is stronger than prompting for citations. A prompt asks the model to behave; the
validator makes misbehaviour not reach the user. Two of the six eval cases exist only to
check that an unanswerable question gets "not in the corpus" rather than an invention,
and `hallucination_rate` has a threshold of exactly zero.

### 3. Scaffold split is the headline; random split is shown as the optimism reference

A random split on molecular data puts analogues on both sides of the line and measures
interpolation within chemical series. Reporting it as "the" accuracy is the single most
common way ML-for-chemistry results are overstated.

TheraGraph trains both — on the same data, separately — and reports the gap. On ESOL:

```
random_forest: scaffold RMSE 0.889, random RMSE 0.593 → 33.3% optimism
gin:           scaffold RMSE 1.059, random RMSE 0.668 → 36.9% optimism
```

The UI shows the scaffold number. The random number is present, labelled, and adjacent,
because a reader who has only ever seen random splits needs to see the size of the
difference.

This is also what makes the two-family comparison worth having rather than decorative.
Measured across the four endpoints, the graph model wins on the largest dataset
(Lipophilicity, 4,200 molecules, non-overlapping intervals), ties on ESOL (1,117) and
loses on BBBP (1,961, imbalanced) — and its MC-dropout uncertainty is over-confident on
both regression endpoints (56% and 47% coverage against a 95% nominal) where the forest's
tree variance is conservative (99% and 98%). Neither family is simply better, which is
precisely the thing an evaluation is for.

### 4. The corpus is finite and says so

`GET /api/v1/search/corpus` returns live entity counts, and every list response reports
what it searched. "No similar compounds above 0.6 Tanimoto **among the 30 compounds
ingested for this target**" is a true statement; "no similar compounds" is not.

The failure mode this avoids is the demo that looks like it knows all of chemistry.

### 5. Ranking is transparent and is not an efficacy claim

Every candidate's score decomposes into its contributing factors, each with its raw
value, its desirability, its weight, its contribution, and whether it was discounted:

```
measured_potency      pChEMBL 8.1  → desirability 0.82 × weight 3.00 = 2.46
predicted_solubility  −2.4 log S   → desirability 0.61 × weight 1.50 = 0.92  (predicted)
structural_novelty    0.71         → desirability 0.71 × weight 1.00 = 0.71  (derived)
model_confidence      0.34         → discounted: out of applicability domain
```

Weights are renormalised when a candidate is missing data, so a compound with five
scorable objectives is comparable to one with nine rather than being penalised twice.
Hard filters (withdrawn drugs, fragments under 10 heavy atoms, MW over 800 Da) exclude
with a stated reason rather than silently scoring low.

Three presets ship — `general_oral_small_molecule`, `cns_small_molecule` (adds BBB,
tightens the MW window), `evidence_first` (triples measured weights, quarters predicted
ones). The response carries the disclaimer that these are prioritisation scores for
follow-up, not predicted clinical efficacy.

### 6. A modular monolith, with two processes split out for real reasons

One FastAPI app, one database, typed function calls across module boundaries. Nothing
here needs independent scaling, the evidence graph is a six-table join, and a research
run's five tables want one transaction.

Two processes *are* separate, because there is a reason: the **worker** (a five-minute
training run must not hold an HTTP connection) and the **MCP server** (a different
protocol for a different client). Both run the same image and the same code.

### 7. Connectors were validated against live APIs, and their responses are committed

Hand-written mocks confirm what you already believe. Running against the real endpoints
found four Open Targets schema mismatches and a renamed PubChem property — bugs that
would have shipped.

The responses are then committed as fixtures, so ingestion tests exercise real payload
shapes, the demo corpus is reproducible from a clean clone, and CI needs no network. When
upstream changes, re-recording produces a reviewable diff.

### 8. One migration chain for PostgreSQL and SQLite

Column types are `sa.JSON().with_variant(postgresql.JSONB, "postgresql")` and the
autogenerate hook emits the shared helper rather than whichever dialect it ran on. So the
test suite runs the production models on in-memory SQLite (fast, isolated per test) while
`alembic check` on both dialects in CI proves the chain applies to each.

PostgreSQL-only objects — trigram and JSONB GIN indexes — live in a migration that is a
no-op on other dialects, and are named in an exclusion list so `alembic check` does not
propose dropping them forever.

### 9. Fail closed on secrets, fail loud on missing models

The production compose overlay uses `${VAR:?}` for every secret, so a misconfigured
deploy does not start with a development default. The application refuses to boot in
production with the development JWT secret.

Symmetrically: a prediction request with no registered model returns **503
`model_not_available`** with a hint to train one — not a 422 "validation error", and
certainly not an empty prediction. Both failures are loud.

### 10. Uncertainty is a first-class output

Every prediction carries an uncertainty estimate and an applicability-domain flag, and
the ranking engine discounts predicted objectives when confidence is under 0.4 or the
molecule is out of domain. A single `compute_confidence` function defines the number, so
the API, the ranking engine and the agent cannot disagree about what "confidence" means.

Where uncertainty cannot be estimated honestly — a bootstrap interval on a 4-molecule
fold — it is omitted rather than shown wide. A confidence interval computed from nothing
is worse than no interval.

## Limitations

### Scientific

* **This system does not discover drugs, and cannot validate one.** It retrieves public
  evidence, computes molecular properties, predicts a small number of physicochemical and
  toxicity endpoints, and ranks candidates for human follow-up. Nothing in it constitutes
  evidence that a compound treats a disease.
* **The property models are small.** Trained on public benchmarks of 1,000–4,200
  molecules. ESOL's scaffold test fold is 112 molecules, and its RMSE confidence interval
  is 0.75–1.03 — a width that is the honest summary of what the number is worth.
* **ClinTox is a weak signal about a hard question.** It labels clinical trial outcomes,
  which conflate toxicity with commercial and trial-design decisions. Treat it as triage,
  never as a safety assessment.
* **BBBP is small and imbalanced.** The CNS preset's BBB objective is a coarse signal,
  and the preset's notes say so in the API response.
* **CNS MPO is computed with descriptor proxies.** The published definition uses measured
  LogD and pKa; TheraGraph substitutes LogP and a heuristic pKa, which is documented at
  the point of computation. The value is indicative, not the published metric.
* **Out-of-domain predictions are extrapolations.** Below 0.35 maximum Tanimoto
  similarity to any training molecule, the number is a prior with error bars, and the
  applicability flag is the important part of the response.
* **Evidence scores are Open Targets' scores, not TheraGraph's judgement.** They are
  passed through with their datasource attribution. A high genetic association score
  means the genetics are strong, not that a drug against that target will work — and the
  Alzheimer's field is the standing reminder.
* **The corpus is a demo corpus.** Three disease areas, a few thousand compounds. Real
  use would mean ingesting far more, and the ingestion pipelines are built for that, but
  what ships is a demonstration.

### Engineering

* **No GPU path for training.** The GNN trains on CPU in a few minutes at this data
  scale. A larger dataset would need a device-aware training loop and a batch scheduler.
* **Similarity search is exact, not approximate.** A popcount range scan plus an
  in-process Tanimoto pass, which is fast to roughly 10⁵–10⁶ fingerprints. Beyond that it
  wants an ANN index (FAISS, or `pgvector` with binary quantisation).
* **The job queue is not a workflow engine.** It has idempotency, leases, backoff and a
  dead state; it has no DAGs, fan-out/fan-in, or cron. A multi-stage pipeline with
  dependencies would want Temporal or Prefect.
* **No multi-tenancy.** Authentication and rate limiting are per-caller, but there is no
  per-user data isolation. Every run is visible to every authenticated caller.
* **Frontend state is server state.** TanStack Query handles caching and invalidation;
  there is no client-side store, no optimistic updates, and no offline mode.
* **Search is trigram and prefix matching, not semantic.** "Alzheimer" finds Alzheimer
  disease; "memory loss disorder" does not. Embedding search over disease descriptions is
  the obvious next step and is not built.
* **The deterministic planner is rule-based.** It covers the demo question shapes well
  and is what makes the offline path real, but it is not a substitute for the LLM on a
  question it has no rule for.
* **No public demo is deployed.** The images build, compose validates, and the
  production overlay is written, but deploying it needs hosting credentials that belong
  to whoever operates it. `docs/operations.md` has the deployment steps.

### Testing

* **467 backend tests, 79% line coverage.** The gaps are concentrated in error paths of
  the external connectors — the code that handles an upstream API returning malformed
  data at a particular stage. Those paths are exercised by fixtures where the fixture
  exists.
* **The eval suite is six cases.** They cover the demo's question shapes and, critically,
  two negative cases. A production agent would want dozens, with per-category thresholds
  and a held-out set that is not used for development.
* **No load testing.** Latency is measured per request and per tool, and the numbers in
  these docs are single-run measurements on a laptop, not a benchmark.
* **No frontend end-to-end browser test.** Components, formatting, the API client and the
  chart helpers are unit-tested (43 tests); the ten-step demo scenario is tested
  end-to-end through the HTTP API rather than through a browser driver.

## If this were going further

In the order that would matter most:

1. **Bioactivity prediction per target.** The interesting model is "will this compound
   bind this target", which needs per-target ChEMBL datasets, a proper
   temporal/scaffold-split protocol, and much more care about assay heterogeneity than
   the property models need.
2. **More corpus, with incremental ingestion.** Full Open Targets and ChEMBL, ingested
   incrementally with change detection, rather than three curated disease areas.
3. **Semantic search over diseases and targets.** Embeddings plus `pgvector`, so a
   researcher can describe a phenotype instead of knowing the ontology id.
4. **An expanded eval suite with a held-out set.** Including adversarial cases designed
   to elicit fabrication, and per-category thresholds rather than one aggregate.
5. **Human-in-the-loop feedback on rankings.** Record which candidates a researcher
   actually pursued, and use it to evaluate the ranking — which is the only real way to
   know whether the weights are any good.
