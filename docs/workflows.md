# Example research workflows

Three worked examples, with the commands and the API calls behind each UI action. All of
them run offline against the committed fixtures.

Prerequisites once:

```bash
make migrate
make seed-smoke      # Alzheimer disease corpus from fixtures, no network
make train           # register models for the four endpoints
make api             # :8000
make frontend-dev    # :5173
```

---

## Workflow 1 — Disease to prioritised candidates

The ten-step scenario the product is built around. `backend/tests/e2e/test_demo_scenario.py`
asserts every step of it over HTTP, so the walkthrough below is executable, not
aspirational.

### 1. Enter a disease

**UI:** Research Dashboard → search "Alzheimer disease".

```bash
curl "localhost:8000/api/v1/search?q=Alzheimer%20disease&limit=5"
```

Returns ranked hits with their EFO/MONDO ids and a relevance tier (exact → prefix →
contains → synonym), so an exact match never loses to a substring. Alongside it,
`GET /api/v1/search/corpus` reports what the corpus actually holds — the dashboard shows
those counts rather than implying completeness.

### 2. Retrieve associated targets

**UI:** click the disease → Target Explorer.

```bash
curl "localhost:8000/api/v1/diseases/MONDO_0004975"
```

Targets ordered by Open Targets' own association score, each with its per-datatype
breakdown (genetic association, known drug, literature, animal model…) and its evidence
count. The score is attributed to Open Targets, not presented as TheraGraph's judgement.

### 3. Inspect the evidence

**UI:** click a target → the evidence table, grouped by evidence class.

```bash
curl "localhost:8000/api/v1/targets/BACE1/evidence?evidence_types=genetic_association,known_drug&limit=20"
```

Every record carries: source, evidence type, upstream identifier, the source's own
confidence score where one exists, retrieval date, and a resolvable URL. Literature
evidence carries PMIDs — e.g. 1520398, 33008897, 37569624 — and the linked publication
metadata from Europe PMC. Nothing in this table was generated.

### 4. Retrieve known compounds

**UI:** the target page's chemistry section.

```bash
curl "localhost:8000/api/v1/targets/BACE1" | jq '.known_compounds'
```

Compounds with measured activity against the target, each with its best pChEMBL value,
its number of measurements, and the assay those came from. `is_active` is set only for
pChEMBL ≥ 6 with an exact `=` relation, so a censored `>` measurement is never read as
potency.

### 5. Search structurally similar compounds

**UI:** Compound Explorer → paste a SMILES → similarity search.

```bash
curl -X POST localhost:8000/api/v1/compounds/similarity \
  -H 'Content-Type: application/json' \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "threshold": 0.6, "limit": 15}'
```

Tanimoto over Morgan fingerprints. The query is standardised the same way the corpus was
(largest fragment, neutralised), so a salt form and its parent match. The response states
how many compounds were searched, because "no hits" and "no hits in 30 compounds" are
different answers.

Substructure search is the SMARTS counterpart:

```bash
curl -X POST localhost:8000/api/v1/compounds/substructure \
  -H 'Content-Type: application/json' -d '{"smarts": "c1ccccc1C(=O)O", "limit": 20}'
```

### 6. Run predictions

**UI:** select compounds → Predict.

```bash
curl -X POST localhost:8000/api/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{"smiles": ["CC(=O)Oc1ccccc1C(=O)O"], "endpoints": ["solubility", "bbb_permeability"]}'
```

Each prediction returns the value, its uncertainty, the model version that produced it,
the nearest-training-set similarity, and an in-domain flag. With no model registered the
endpoint returns **503 `model_not_available`** with a hint to train one — it does not
return an empty prediction, and the e2e test asserts that behaviour.

### 7. Compare candidates

**UI:** Candidate Comparison.

```bash
curl -X POST localhost:8000/api/v1/ranking \
  -H 'Content-Type: application/json' \
  -d '{"target": "BACE1", "disease": "MONDO_0004975", "config_name": "cns_small_molecule"}'
```

Each candidate's total score decomposes into every objective's raw value, desirability,
weight, contribution, and whether it was discounted for low confidence — plus flags
(`under_characterised`, `missing_N_objectives`) and the reason for any hard-filter
exclusion. `GET /api/v1/ranking/presets` lists the three configurations and their curves;
`POST /api/v1/ranking/compare` gives the side-by-side matrix with per-objective best and
worst.

The response carries the disclaimer: these are prioritisation scores for follow-up, not
predicted clinical efficacy.

### 8. Ask the research assistant

**UI:** AI Research Assistant.

```bash
curl -X POST localhost:8000/api/v1/agent/research \
  -H 'Content-Type: application/json' \
  -d '{"query": "Which compounds with measured activity against BACE1 should we prioritise for an Alzheimer disease programme?",
       "parameters": {"target": "BACE1", "disease": "MONDO_0004975", "max_candidates": 15}}'
```

The response streams into the UI as it executes: the plan, then each tool call with its
input and output, then the retrieved evidence, then the predictions (labelled as
predictions), then a reasoning summary, then the report. What you see is the execution
record — not a narrative about it. No API key is needed; without one the deterministic
planner runs the same workflow.

### 9. Generate the report

```bash
curl "localhost:8000/api/v1/runs/RUN-2026-00001/report"
```

Markdown and JSON. Facts and predictions are in separate sections, every claim carries
citation keys, and every citation resolves to a source, an identifier and a retrieval
date. Claims whose citations did not resolve were removed before this point, and the run
records how many.

### 10. Open the run page and reproduce it

**UI:** Research Run.

```bash
curl "localhost:8000/api/v1/runs/RUN-2026-00001"              # full trace
curl "localhost:8000/api/v1/runs/RUN-2026-00001/replay-plan"  # what would differ now
curl -X POST "localhost:8000/api/v1/runs/RUN-2026-00001/replay"
```

`replay-plan` reports drift before you re-run: the corpus has grown, a model was
retrained, a prompt template changed. The replay is a new run linked to the original, and
`POST /api/v1/runs/compare` diffs them field by field.

---

## Workflow 2 — A chemistry-first question

Start from a molecule rather than a disease.

```bash
# 1. What is this compound, and what do we know about it?
curl "localhost:8000/api/v1/compounds/CHEMBL25"

# 2. What is in its structural neighbourhood?
curl -X POST localhost:8000/api/v1/compounds/similarity \
  -H 'Content-Type: application/json' \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "threshold": 0.5, "limit": 25}'

# 3. Compute descriptors for the neighbourhood, with units and explanations
curl -X POST localhost:8000/api/v1/compounds/properties \
  -H 'Content-Type: application/json' \
  -d '{"smiles": ["CC(=O)Oc1ccccc1C(=O)O", "Oc1ccccc1C(=O)O"]}'

# 4. Which targets does it have measured activity against?
curl "localhost:8000/api/v1/compounds/CHEMBL25" | jq '.targets'

# 5. The graph, rooted at the compound
curl "localhost:8000/api/v1/graph?compound=CHEMBL25"
```

`GET /api/v1/compounds/descriptors` documents all 16 curated descriptors with their
labels, units and one-line explanations, which is what the Compound Explorer renders as
tooltips. An unlabelled LogP is not data.

The same question through the assistant:

```bash
curl -X POST localhost:8000/api/v1/agent/research \
  -H 'Content-Type: application/json' \
  -d '{"query": "Find compounds structurally similar to aspirin and predict their solubility.",
       "parameters": {"compound": "CHEMBL25"}}'
```

The deterministic planner recognises the compound-first shape and plans
`get_compound → find_similar_compounds → predict_properties`, binding the SMILES from
step 1 into step 2 rather than re-deriving it.

---

## Workflow 3 — Train, evaluate and compare models

```bash
# 1. Snapshot a benchmark (content-hashed, registered as a dataset version)
theragraph ingest moleculenet bbbp --mode replay

# 2. Train both families on the identical split
theragraph ml train --endpoint bbb_permeability --dataset bbbp --families random_forest,gin

# 3. What is registered, with scaffold-split metrics
theragraph ml models

# 4. Score a molecule
theragraph ml predict "CC(=O)Oc1ccccc1C(=O)O" --endpoint bbb_permeability
```

**UI:** Model Evaluation shows, per endpoint: the scaffold-split headline with its
bootstrap confidence interval, the random-split reference labelled as the optimism
measure with the gap computed, the cross-validation fold spread, the calibration
reliability curve with ECE/MCE/Brier, the uncertainty-decile error table, and the error
breakdown by property bucket and scaffold familiarity.

Training as a background job, which is how the UI does it:

```bash
curl -X POST localhost:8000/api/v1/jobs/train \
  -H 'Content-Type: application/json' \
  -d '{"endpoint": "bbb_permeability", "dataset_key": "bbbp", "families": ["random_forest", "gin"]}'
# → {"job_id": "...", "status": "pending"}
curl "localhost:8000/api/v1/jobs/{job_id}"
```

Submitting the same payload twice returns the first job: the idempotency key is derived
from the task and payload, so a double-clicked button does not train twice.

---

## Running the whole thing from scratch

```bash
make demo    # bootstrap → seed-smoke → train → evals
```

That is also close to what CI runs, which is the point: if the demo works on a laptop
with no network, it works in CI, and a broken demo fails the build.
