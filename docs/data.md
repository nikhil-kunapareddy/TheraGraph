# Data and ingestion

Everything in TheraGraph's database came from a public source and says so. This document
covers what those sources are, how records are normalised, what provenance is recorded,
and how to run and reproduce ingestion.

## Sources

| Source | What TheraGraph takes | Version pinned | Licence |
|---|---|---|---|
| **Open Targets Platform** (GraphQL) | diseases, targets, target–disease associations with per-datatype scores, evidence records, tractability buckets, safety liabilities, known drugs | release **26.06** | CC0 1.0 for the platform; each evidence record inherits its own datasource's licence |
| **ChEMBL** (REST) | compounds (structures, max phase, synonyms), assays (type, organism, confidence score), measured bioactivities with pChEMBL values | **ChEMBL_37** | CC BY-SA 3.0 |
| **PubChem** (PUG REST) | compound cross-references and computed properties, used to reconcile identifiers | rolling | Public domain (depositor records may vary) |
| **Europe PMC** (REST) | publication metadata for cited evidence: title, journal, year, authors, DOI, PMID | rolling | Metadata free to use; article licences vary |
| **MoleculeNet** (CSV via DeepChem S3) | ESOL solubility, BBBP permeability, Lipophilicity, ClinTox — the ML training data | content-hashed snapshot | MIT for the distribution; measurements per original publication |

Citations for all five are stored in the `data_source` table and surfaced by
`GET /api/v1/ingestion/sources`, so the UI can show a licence next to the data it draws.

### What is *not* ingested

No proprietary data, no scraped content, and no clinical or patient-level data. Clinical
trial references come from Open Targets' known-drug evidence, not from a trial registry
feed.

## The connectors were validated against the live APIs

Hand-written mocks drift from reality. Every connector was run against the real endpoint,
which found four schema mismatches that a mock would have happily confirmed:

| What was assumed | What the API actually returns |
|---|---|
| `Tractability.id` | The type has `label`, `modality` and `value` — no `id`. |
| `Drug.maximumClinicalTrialPhase` | Renamed to `maximumClinicalStage`. |
| `Drug.isApproved` / `hasBeenWithdrawn` | Neither field exists; approval has to be inferred from the clinical stage. |
| `disease.evidences(ensemblIds:)` optional | It is `[String!]!` — required. The connector therefore takes `ensembl_ids` as a required argument, and the docstring says why. |
| PubChem `CanonicalSMILES` | Renamed; the client accepts `ConnectivitySMILES`, `CanonicalSMILES`, `SMILES` or `IsomericSMILES`. |

## Normalisation

Each connector returns a `NormalizedBatch`:

```python
NormalizedBatch(
    entity_type="compound",
    rows=[...],      # accepted, ready to upsert
    rejects=[...],   # rejected, each carrying _quality_errors
)
```

Normalisation is where upstream shapes become the internal schema:

* **Identity keys.** Compounds are keyed on **InChIKey**, computed locally by RDKit after
  standardisation — not on the upstream id. That is what lets the same molecule arriving
  from ChEMBL and from PubChem resolve to one row. Targets key on Ensembl gene id,
  diseases on EFO/MONDO id.
* **Standardisation.** `Cleanup → LargestFragmentChooser → Uncharger`, then canonical
  SMILES, InChI, InChIKey and the Bemis–Murcko scaffold. Salt forms and charge states
  collapse onto one record; the scaffold is stored because the ML splits depend on it.
* **Units.** Activities are normalised to pChEMBL (−log10 molar). A value without units
  is a quality warning, not a number.
* **Evidence typing.** Open Targets' 25 datasource ids map onto a fixed vocabulary —
  `genetic_association`, `known_drug`, `literature`, `animal_model`, `expression`,
  `pathway`, `functional_genomics`, `somatic_mutation`, `biomarker`, `other` — so the UI
  can group and weight evidence without hard-coding datasource names.
* **`is_active`** is set only when pChEMBL ≥ 6 **and** the relation is exactly `=`. A
  `>` relation means the assay hit its ceiling; treating that as a potency measurement is
  how a censored value silently becomes a training label.

## Quality rules

Rules run per batch and produce a report stored on the ingestion run. Severity matters:
an **error** rejects the row, a **warning** keeps it and records the concern.

| Entity | Rule | Severity |
|---|---|---|
| compound | `inchikey`, `canonical_smiles`, `chembl_id` present | error |
| compound | `max_phase` within 0–4 | error |
| bioactivity | `compound_id`, `source_identifier` present | error |
| bioactivity | `pchembl_value` within 0–15 | error |
| bioactivity | a numeric value carries units | warning |
| bioactivity | ChEMBL curators did not flag the measurement | warning |

Curator-flagged rows stay visible in the corpus and are excluded from *training*, rather
than being hidden. The report also records field completeness (how often optional fields
like `name`, `pubchem_cid`, `murcko_scaffold` were present) and a duplicate count from
the dedup key, so "we ingested 30 compounds" comes with "and here is what was thin about
them."

## Provenance

Every scientific row carries:

| Column | Meaning |
|---|---|
| `source` | `open_targets`, `chembl`, `pubchem`, `europepmc`, `moleculenet` |
| `source_identifier` | the upstream accession — `CHEMBL25`, `ENSG00000186868`, `MONDO_0004975` |
| `source_url` | a link a human can open to check the value |
| `retrieved_at` | when this value was fetched, not when the row was written |
| `dataset_version_id` | the immutable snapshot it belongs to |

Plus, per ingestion:

* **`ingestion_run`** — source, dataset, mode, status, counts, quality report, runtime,
  errors, idempotency key.
* **`dataset_version`** — `content_hash` over the normalised records (order-independent),
  `record_count`, `retrieved_at`. Re-ingesting unchanged upstream data produces the same
  hash and therefore the same version.
* **`provenance_record`** — the audit row linking an entity to the run that wrote it.

This is why the evidence graph can label every edge with its source, its type, the
upstream identifier, the source's own confidence score where one exists, and the
retrieval date — and why an LLM is never the origin of any of it.

## Running ingestion

### Offline, from committed fixtures (no network, no keys)

Every upstream response used to build the shipped corpus is committed under
`data/seed/fixtures/`, and the four MoleculeNet CSVs under `data/raw/moleculenet/`.
Replay mode reads only those:

```bash
make migrate                                     # schema
theragraph db sources                            # register the five sources
theragraph ingest seed --plan smoke --mode replay
```

The smoke plan builds one disease area (Alzheimer disease, MONDO_0004975) with its
targets, evidence, publications, 30 ChEMBL compounds with 40 measured activities, and the
ESOL benchmark. It reports `fixture_hits` and `upstream_calls`, and in replay mode the
second number is always zero. A missing fixture raises `LiveIngestionDisabledError`
naming the file it wanted — replay never silently falls back to the network.

There is also a tiny in-process corpus for smoke-testing without any files:

```bash
theragraph db demo-corpus    # 1 disease, 2 targets, 4 approved drugs, 3 evidence rows
```

### Live, from the public APIs

```bash
theragraph ingest seed --plan demo          # three disease areas + all four benchmarks
theragraph ingest disease MONDO_0004975     # one disease and its targets
theragraph ingest bioactivity BACE1 --max-activities 300
theragraph ingest moleculenet esol
```

Live ingestion is rate-limited and retried with backoff, and respects each API's usage
guidance through a shared `User-Agent`. Setting `ALLOW_LIVE_INGESTION=false` (the default
in production compose) makes the ingestion endpoints refuse anything but replay mode —
a deployed demo should not be generating traffic against EBI.

### Re-recording fixtures

```bash
theragraph ingest seed --plan smoke --mode record
```

Record mode calls the live API and writes each response to
`data/seed/fixtures/<source>/<operation>.<hash>.json`, pretty-printed and reviewable.
Committing the diff shows exactly what upstream changed.

## The demo corpus

`--plan demo` covers three disease areas chosen to exercise different evidence profiles:

| Disease | Identifier | Why it is in the plan |
|---|---|---|
| Alzheimer disease | `MONDO_0004975` | Strong human genetics (APP, BACE1), a long history of clinical failure, and CNS constraints that make the ranking's BBB and TPSA objectives matter |
| Type 2 diabetes | `MONDO_0005148` | Many approved drugs, so known-drug evidence dominates and the novelty score has something to push against |
| Non-small cell lung carcinoma | `MONDO_0005233` | Somatic-mutation and biomarker evidence classes that the other two barely have |

## Schema reference

28 tables. The groups and the key relationships:

```
data_source ──< dataset_version ──< ingestion_run
                      │
                      └──< provenance_record

gene ──< target ──< target_disease_association >── disease
                │                                    │
                └──< evidence >───────────────────────┘
                       │  └──> publication
                       └──> compound (known drugs)

compound ──< molecular_property        (computed, versioned descriptor set)
         ├──< compound_fingerprint     (Morgan / MACCS / pattern, packed bytes)
         ├──< bioactivity >── assay >── target
         └──< prediction >── ml_model ──< model_evaluation
                                    │
                              experiment

candidate_ranking ──< candidate_score >── compound

research_run ──< agent_execution ──< tool_call
            ├──< report
            └──> job
run_counter        (atomic RUN-YYYY-NNNNN allocation)
```

`GET /api/v1/search/corpus` reports live counts per entity type, so the UI can tell a
user "this corpus holds 30 compounds for this target" instead of implying completeness.
