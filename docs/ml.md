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

**Regression:** RMSE (primary), MAE, median and max absolute error, R², Pearson and
Spearman ρ, and bias. Separately, under calibration: **prediction-interval coverage** —
the fraction of test molecules whose true value falls inside `prediction ± 1.96σ`. A
model claiming ±0.5 that is right 60% of the time is miscalibrated, and the coverage
number is what says so.

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

**Classifiers.** Probabilities are calibrated on the validation split. Platt scaling and
isotonic regression are both fitted, and the one with the better validation Brier score
wins — **unless neither beats the uncalibrated model**, in which case calibration is
refused and the model is served raw. Applying a calibrator that makes things worse is a
common and invisible failure, and on ClinTox it is exactly what happened: neither method
improved the Brier score, so that model is served uncalibrated.

Reported either way: expected calibration error (ECE, 10 bins), maximum calibration
error (MCE), Brier score, and the reliability curve the Model Evaluation view plots.
Measured on the scaffold splits: BBBP ECE 0.071 / Brier 0.095, ClinTox ECE 0.038 /
Brier 0.052.

**Regressors.** There is no probability to calibrate, so what is measured instead is
whether the uncertainty behaves like an error bar — and the verdict is computed from the
observed coverage rather than being a fixed sentence:

| Model | Coverage at 95% nominal | Verdict |
|---|---|---|
| `solubility` random_forest | 99.1% | conservative — wider than needed, the safer direction |
| `solubility` gin | 56.2% | **over-confident** — treat the error bars as a lower bound |
| `lipophilicity` random_forest | 98.3% | conservative |
| `lipophilicity` gin | 47.4% | **over-confident** |

That is a real and useful finding, and it is the reason the two families are not
interchangeable in the ranking engine: at this data scale the Random Forest's tree
variance is a trustworthy (if cautious) error bar, while 20-pass MC dropout on a
4-layer GIN substantially understates its own error.

## Uncertainty and the applicability domain

Two independent things determine whether a prediction should be believed:

**Ensemble spread** — tree variance or MC-dropout standard deviation. On ESOL the Random
Forest's estimate is usable: mean absolute error rises from **0.266** in the
lowest-uncertainty decile to **1.007** in the highest, with a correlation of **0.527**
between predicted uncertainty and actual error. The GIN's, on the same split, correlates
at only 0.104 — it produces a number, but that number does not rank its own errors.

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

| Endpoint | Dataset | Model | Primary metric | **Scaffold split** | 95% CI | Secondary | Random split | Optimism |
|---|---|---|---|---|---|---|---|---|
| `solubility` | ESOL | random_forest | RMSE (log mol/L) | **0.889** | 0.751–1.026 | R2 0.887 | 0.593 | +33.27% |
| `solubility` | ESOL | gin | RMSE (log mol/L) | **1.059** | 0.876–1.245 | R2 0.840 | 0.668 | +36.88% |
| `bbb_permeability` | BBBP | random_forest | ROC-AUC | **0.802** | 0.686–0.898 | PR-AUC 0.956 | 0.923 | +15.1% |
| `bbb_permeability` | BBBP | gin | ROC-AUC | **0.640** | 0.528–0.745 | PR-AUC 0.929 | 0.900 | +40.53% |
| `lipophilicity` | Lipophilicity | random_forest | RMSE (logD) | **0.893** | 0.832–0.959 | R2 0.464 | 0.743 | +16.81% |
| `lipophilicity` | Lipophilicity | gin | RMSE (logD) | **0.731** | 0.673–0.786 | R2 0.641 | 0.597 | +18.25% |
| `clinical_toxicity` | ClinTox | random_forest | ROC-AUC | **0.842** | 0.698–0.949 | PR-AUC 0.226 | 0.531 | none |
| `clinical_toxicity` | ClinTox | gin | ROC-AUC | **0.749** | 0.574–0.896 | PR-AUC 0.243 | 0.496 | none |

* **`solubility`** — random_forest leads on rmse, but its confidence interval overlaps gin — treat the difference as unproven. Test fold 112 molecules, 66 scaffolds, 0 shared with train.
* **`bbb_permeability`** — random_forest leads on roc_auc, but its confidence interval overlaps gin — treat the difference as unproven. Test fold 197 molecules, 197 scaffolds, 0 shared with train.
* **`lipophilicity`** — gin leads on rmse with non-overlapping confidence intervals. Test fold 420 molecules, 420 scaffolds, 0 shared with train.
* **`clinical_toxicity`** — random_forest leads on roc_auc, but its confidence interval overlaps gin — treat the difference as unproven. Test fold 143 molecules, 143 scaffolds, 0 shared with train.

### Reading that table

Four things in it are worth saying out loud, because the table alone lets a reader draw
the wrong conclusion.

**The graph model wins where there is data to win with.** GIN beats the baseline on
Lipophilicity (4,200 molecules) with non-overlapping confidence intervals, ties it on
ESOL (1,117) and loses on BBBP (1,961, and imbalanced). That is the expected
relationship between dataset size and a graph network's advantage, and it is why
shipping only a GNN — or only a baseline — would have been the wrong call. Both are
trained, both are registered, and the comparison is reported rather than assumed.

**"Unproven" appears three times out of four, and that is the honest reading.** On a
112-molecule test fold, a 0.17 RMSE difference has overlapping bootstrap intervals. The
comparison says so instead of declaring a winner, which is what the intervals are for.

**ClinTox's ROC-AUC flatters it; everything else says so.** The positive class is 7.5% of
the dataset (112 of 1,484 molecules), and 5.6% of this scaffold-split test fold. ROC-AUC
0.842 sounds strong. But PR-AUC is 0.226 against that base rate — a ~4x lift, real signal
— and at the default threshold the confusion matrix is 135 true negatives, 8 false
negatives, **zero** true positives: MCC 0.0, recall 0.0, balanced accuracy 0.5. The model
*ranks* toxicity usefully and *decides* nothing at 0.5. Reporting only the AUC would hide
that completely, which is why the threshold metrics and the confusion matrix are reported
next to it.

**ClinTox is also the case where the random split is *worse*.** With so few positives, a
random test fold can end up close to chance (0.531), which is not "the scaffold split was
harder" — it is an unstable estimate. The optimism column reads `none` rather than
printing a percentage that would imply the opposite of the truth.

Two more results that do not fit in the table. On ClinTox, calibration was **refused** —
neither Platt scaling nor isotonic regression beat the uncalibrated model on the
validation Brier score, so it is served raw. And on both regression endpoints the GIN's
MC-dropout uncertainty is badly over-confident (56% and 47% coverage against a 95%
nominal) while the Random Forest's is conservative (99% and 98%). Both are the designed
behaviour firing on real data rather than a hypothetical.

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
