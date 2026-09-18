# Machine learning

Two model families, evaluated the way computational chemists evaluate models rather than
the way a Kaggle notebook does. Every number the UI shows is from a **scaffold split**;
the random split appears only as a labelled reference, to show how much it flatters.

## Endpoints

| Endpoint | Dataset | Task | Direction | What it means for a candidate |
|---|---|---|---|---|
| `solubility` | MoleculeNet ESOL (Delaney) | regression, log mol/L | higher is better | Aqueous solubility; poor solubility limits oral exposure |
| `bbb_permeability` | MoleculeNet BBBP | binary classification | context-dependent | Blood–brain barrier penetration — required for a CNS target, a liability elsewhere |
| `lipophilicity` | MoleculeNet Lipophilicity | regression, logD 7.4 | target window | Drives permeability and off-target promiscuity; the useful range is a window, not a maximum |
| `clinical_toxicity` | MoleculeNet ClinTox | binary classification | lower is better | Trained on clinical trial toxicity outcomes; a weak signal about a hard question |

These are **property predictors**, not efficacy predictors. TheraGraph does not train a
model that claims a compound will treat a disease.

## Representations

**Baseline features** (`fingerprint+descriptors`, descriptor set version `2`):

* Morgan fingerprint, radius 2, 2048 bits (`morgan_r2_2048`)
* 16 curated RDKit descriptors, each with a label, a unit and a one-line explanation:
  molecular weight, exact mass, LogP, TPSA, H-bond donors and acceptors, rotatable
  bonds, aromatic rings, ring count, heavy atoms, fraction Csp3, formal charge,
  stereocentres, QED, molar refractivity, heteroatom count

**Graph features** (graph feature version `1`): atoms become nodes with a 49-dimensional
feature vector (element one-hot over 14 elements plus "other", degree, formal charge,
hydrogen count, hybridisation, chirality tag, aromaticity, ring membership, mass);
bonds become two directed edges with a 12-dimensional vector (bond type, stereo,
conjugation, ring membership).

The vocabularies are **pinned in code, not inferred from the training set**. Inferring
them would mean a model silently changing its input dimension when the next dataset
happens to contain boron. A checkpoint records the feature version it was trained with
and refuses to load against a different one.

## Model families

### Baseline — Random Forest / XGBoost

scikit-learn `RandomForestRegressor`/`Classifier` (500 trees, `max_features="sqrt"`) or
XGBoost (600 rounds, depth 7, lr 0.05, subsample 0.8) on the 2064-dimensional feature
vector. Uncertainty comes from the spread across trees, which is a genuine ensemble
disagreement rather than a softmax confidence.

### Graph — GINE-style GNN in pure PyTorch

4 message-passing layers, hidden dimension 128, dropout 0.15, batch norm, residual
connections; graph-level readout concatenating mean and max pooling; a 3-layer MLP head.
Trained with Adam (lr 1e-3, weight decay 1e-5), batch size 64, up to 120 epochs with
early stopping on the validation split (patience 20) and gradient clipping at 5.0.

There is no PyTorch Geometric dependency: scatter operations are implemented with
`index_add_` and `scatter_reduce_`. For a model this size that is a few dozen lines in
exchange for dropping a compiled dependency pinned to a CUDA version.

Uncertainty comes from **MC dropout** — 20 stochastic forward passes at inference, with
dropout left active. The standard deviation across passes is the uncertainty estimate.

## Splits: why not random

A random split puts near-identical analogues on both sides of the line. The model
memorises a scaffold and the test score measures interpolation within chemical series,
which is not the question anyone is asking. TheraGraph therefore reports:

1. **Scaffold split** (MoleculeNet protocol) — Bemis–Murcko scaffolds sorted by group
   size, largest groups to train. Test scaffolds are unseen by construction. **This is
   the headline number.**
2. **Scaffold-grouped K-fold cross-validation** (5 folds by default) — no scaffold spans
   the train/test boundary of any fold. Gives a variance estimate, not just a point.
3. **Random split** — trained separately and reported *only* as the optimism reference,
   with the gap computed explicitly.

Each split reports its own difficulty so a reader can tell an easy split from a hard one:

```json
"split_difficulty": {
  "train_scaffolds": 202,
  "test_scaffolds": 66,
  "shared_scaffolds": 0,
  "acyclic_test_molecules": 46,
  "test_scaffolds_unseen_fraction": 1.0
}
```

`shared_scaffolds: 0` is the assertion that the split did what it claims.
`acyclic_test_molecules` is there because molecules with no ring have no Bemis–Murcko
scaffold — with 46 of 112 test molecules acyclic, an unseen-scaffold fraction of 1.0
describes 66 molecules, not all of them. When a test fold contains no scaffolded
molecules at all, the fraction is reported as `null` rather than `0.0`, because "no
information" and "zero overlap" are different claims.

## Metrics

**Regression:** RMSE (primary), MAE, R², Spearman ρ, plus prediction-interval coverage —
the fraction of test molecules whose true value falls inside the model's own 95%
interval. A model claiming ±0.5 that is right 60% of the time is miscalibrated, and the
coverage number is what says so.

**Classification:** ROC-AUC (primary), PR-AUC with the positive-class base rate next to
it (PR-AUC of 0.7 means something different at 10% prevalence than at 50%), balanced
accuracy, MCC, precision/recall at the chosen threshold, and the confusion matrix.

**Bootstrap confidence intervals** (1000 resamples) on the primary metric, reported
whenever the fold has at least 10 samples. Below that the interval would be theatre, so
it is omitted rather than shown wide.

Comparison between families uses those intervals:

> `random_forest leads on rmse, but its confidence interval overlaps gin — treat the
> difference as unproven`

The comparison names ties explicitly. A 0.02 RMSE difference on 112 molecules is not a
result, and the report says so instead of declaring a winner.

## Calibration

Classifier probabilities are calibrated on the validation split. Platt scaling and
isotonic regression are both fitted, and the one with the better validation Brier score
wins — **unless neither beats the uncalibrated model**, in which case calibration is
refused and the model is served raw. Applying a calibrator that makes things worse is a
common and invisible failure.

Reported either way: expected calibration error (ECE, 10 bins), maximum calibration
error (MCE), Brier score, and the reliability curve the Model Evaluation view plots.

## Uncertainty and the applicability domain

Two independent things determine whether a prediction should be believed:

**Ensemble spread** — tree variance or MC-dropout standard deviation. On ESOL the
uncertainty estimate is usable: the lowest-uncertainty decile has mean absolute error
0.293 while the highest has 1.079, a rank correlation of 0.53 between predicted
uncertainty and actual error.

**Applicability domain** — the maximum Tanimoto similarity between the query molecule
and any training molecule. Below **0.35** the molecule is flagged out-of-domain; at
**0.55** or above it is fully in-domain, with a linear ramp in between. A model asked
about a chemotype it has never seen returns a number *and* the fact that it is
extrapolating.

`compute_confidence` combines the two into a single [0, 1] score, defined in one place so
that the API, the ranking engine and the agent all mean the same thing by "confidence".
The ranking engine then discounts predicted objectives when confidence is below 0.4 or
the molecule is out of domain — a candidate cannot climb the ranking on an extrapolation.

## Error analysis

Every evaluation slices the errors, because a single RMSE hides where a model fails:

* **By property bucket** — MW, LogP and TPSA terciles, so "this model is worse on large
  lipophilic molecules" becomes visible.
* **By uncertainty decile** — the check that the uncertainty estimate ranks errors.
* **By scaffold familiarity** — seen scaffold vs unseen vs acyclic.
* **Worst predictions** — the 10 largest residuals with their SMILES, so a chemist can
  look at them and recognise a class the model does not handle.

## Model versioning and the registry

A model version is derived from a hash of everything that could change its behaviour:

```
version = hash(dataset content hash, hyperparameters, feature spec,
               code version, split strategy)
```

so `solubility-random_forest@2026.09.18.302b57d6` is reproducible by construction. The
registry stores the artifact plus a `TrainingReference` — dataset version, split indices,
seed, hyperparameters, metrics — and verifies the artifact hash on load. A tampered or
truncated artifact raises rather than serving quiet nonsense.

Predictions are persisted idempotently on `(compound_id, model_id, input_hash)`, so
asking twice does not create two rows or two inference costs.

## Measured results

From `theragraph ml train`, offline, on the committed MoleculeNet snapshots. **Scaffold
split is the headline; the random column is the optimism reference.**

<!-- METRICS_TABLE -->

## Running training

```bash
# both families, scaffold split + 5-fold CV + the random reference
theragraph ml train --endpoint solubility --dataset esol --families random_forest,gin

# fast: one family, no CV, no random reference (what CI runs)
theragraph ml train --endpoint solubility --dataset esol \
  --families random_forest --no-cv --no-random-split

# all four endpoints
make train

# inspect what is registered, with scaffold-split metrics
theragraph ml models
theragraph ml predict "CC(=O)Oc1ccccc1C(=O)O" --endpoint solubility
```

Training is also available as a background job — `POST /api/v1/jobs/train` — so a slow
run does not hold an HTTP connection. The job is idempotent on its payload: submitting
the same training request twice returns the first job.

## What these models cannot do

* They predict **properties**, not efficacy. No model here says a compound will work.
* They were trained on public benchmarks of a few thousand molecules. ESOL's test fold
  is 112 molecules; a 0.89 RMSE has a confidence interval of 0.75–1.03, and that width
  is the honest summary.
* MoleculeNet labels carry the biases of their source assays. ClinTox in particular
  labels clinical trial *outcomes*, which conflate toxicity with commercial and trial
  design decisions.
* Out-of-domain means out-of-domain. The applicability flag is not a formality — for a
  novel chemotype, treat the number as a prior, not a measurement.
