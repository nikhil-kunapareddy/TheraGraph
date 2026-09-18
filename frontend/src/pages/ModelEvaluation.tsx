/**
 * Model evaluation.
 *
 * Scaffold-split numbers are the headline everywhere; the random split appears only as
 * the optimism reference, with the gap stated explicitly. Confidence intervals are shown
 * next to every metric, and the experiment's own verdict ("the difference is unproven")
 * is surfaced rather than a leaderboard.
 */

import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import clsx from 'clsx';
import { api } from '@/lib/api';
import type { MlModel } from '@/types/api';
import {
  Badge,
  Card,
  EmptyState,
  ErrorState,
  Field,
  Spinner,
  StatTile,
  Table,
} from '@/components/Primitives';
import {
  CalibrationChart,
  DistributionChart,
  MetricComparisonChart,
  UncertaintyChart,
} from '@/components/Charts';
import { num, titleCase, truncate } from '@/lib/format';

const CLASSIFICATION_METRICS = [
  'roc_auc',
  'pr_auc',
  'mcc',
  'balanced_accuracy',
  'f1',
  'precision',
  'recall',
];
const REGRESSION_METRICS = ['rmse', 'mae', 'r2', 'spearman', 'pearson', 'bias'];

export default function ModelEvaluation() {
  const params = useParams<{ modelId?: string }>();
  const [selectedEndpoint, setSelectedEndpoint] = useState<string | null>(null);

  const models = useQuery({ queryKey: ['models'], queryFn: () => api.models() });
  const experiments = useQuery({ queryKey: ['experiments'], queryFn: () => api.experiments() });
  const detail = useQuery({
    queryKey: ['model', params.modelId],
    queryFn: () => api.model(params.modelId as string),
    enabled: Boolean(params.modelId),
  });

  const endpoints = Array.from(new Set((models.data?.models ?? []).map((model) => model.endpoint)));
  const activeEndpoint = selectedEndpoint ?? endpoints[0] ?? null;
  const endpointModels = (models.data?.models ?? []).filter(
    (model) => model.endpoint === activeEndpoint,
  );
  const focusModel = detail.data ?? endpointModels[0];

  if (models.isLoading) return <Spinner label="Loading the model registry…" />;
  if (models.error) return <ErrorState error={models.error} retry={() => void models.refetch()} />;

  if ((models.data?.models ?? []).length === 0) {
    return (
      <EmptyState
        title="No models are registered."
        detail="Train one with `theragraph ml train --endpoint solubility --dataset esol`. Training reports scaffold-split performance, calibration, uncertainty and error analysis."
      />
    );
  }

  return (
    <div className="space-y-4">
      <Card
        title="Model evaluation"
        subtitle={models.data?.reporting_note}
        actions={
          <div className="flex flex-wrap gap-1.5">
            {endpoints.map((endpoint) => (
              <button
                key={endpoint}
                type="button"
                onClick={() => setSelectedEndpoint(endpoint)}
                className={clsx(
                  'rounded border px-2 py-1 text-2xs transition',
                  endpoint === activeEndpoint
                    ? 'border-[color:var(--series-1)] bg-[color:var(--series-1)]/10 text-[color:var(--series-1)]'
                    : 'border-surface-border text-ink-muted hover:text-ink-secondary',
                )}
              >
                {titleCase(endpoint)}
              </button>
            ))}
          </div>
        }
      >
        <Table
          headers={[
            'Model',
            'Family',
            'Task',
            'Scaffold split (headline)',
            'Random split',
            'Trained',
            'Status',
          ]}
        >
          {endpointModels.map((model) => {
            const scaffold = model.metrics.scaffold ?? {};
            const random = model.metrics.random ?? {};
            const primary = model.task_type === 'classification' ? 'roc_auc' : 'rmse';
            return (
              <tr key={model.id} className="hover:bg-surface-raised/60">
                <td className="px-2 py-1.5 font-mono text-2xs text-ink">{model.version}</td>
                <td className="px-2 py-1.5 text-ink-secondary">{titleCase(model.family)}</td>
                <td className="px-2 py-1.5 text-2xs text-ink-muted">{model.task_type}</td>
                <td className="px-2 py-1.5 font-mono tabular-nums text-ink">
                  {titleCase(primary)} {num(scaffold[primary] as number, 3)}
                  {Array.isArray(scaffold[`${primary}_ci`]) && (
                    <span className="ml-1 text-2xs text-ink-muted">
                      [{num((scaffold[`${primary}_ci`] as number[])[0], 2)}–
                      {num((scaffold[`${primary}_ci`] as number[])[1], 2)}]
                    </span>
                  )}
                </td>
                <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                  {num(random[primary] as number, 3)}
                </td>
                <td className="px-2 py-1.5 text-2xs text-ink-muted">
                  {model.training_runtime_s ? `${num(model.training_runtime_s, 1)}s` : '—'}
                </td>
                <td className="px-2 py-1.5">
                  <Badge tone={model.status === 'active' ? 'good' : 'neutral'}>
                    {model.status}
                  </Badge>
                </td>
              </tr>
            );
          })}
        </Table>
      </Card>

      {experiments.data && experiments.data.experiments.length > 0 && (
        <Card
          title="Baseline versus graph model"
          subtitle="The comparison the training pipeline produced, including whether the difference is measurable"
        >
          {experiments.data.experiments
            .filter((experiment) => !activeEndpoint || experiment.endpoint === activeEndpoint)
            .slice(0, 2)
            .map((experiment) => {
              const rows = experiment.comparison.rows ?? [];
              return (
                <div key={experiment.id} className="space-y-3">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <code className="font-mono text-2xs text-ink">{experiment.id}</code>
                    <span className="text-xs text-ink-secondary">{experiment.name}</span>
                    <Badge tone={experiment.status === 'succeeded' ? 'good' : 'warning'}>
                      {experiment.status}
                    </Badge>
                    {experiment.runtime_s && (
                      <span className="text-2xs text-ink-muted">
                        {num(experiment.runtime_s, 1)}s
                      </span>
                    )}
                  </div>

                  {experiment.comparison.verdict && (
                    <p className="rounded border border-surface-border bg-surface-raised/50 p-2.5 text-xs leading-relaxed text-ink-secondary">
                      {experiment.comparison.verdict}
                    </p>
                  )}

                  {rows.length > 0 && (
                    <MetricComparisonChart
                      metric={experiment.comparison.primary_metric ?? 'roc_auc'}
                      higherIsBetter={experiment.comparison.higher_is_better ?? true}
                      rows={rows.map((row) => ({
                        label: titleCase(row.model),
                        split: row.split_strategy,
                        value: row.value ?? null,
                        ciLow: row.ci_low,
                        ciHigh: row.ci_high,
                      }))}
                    />
                  )}

                  {experiment.comparison.optimism_gap && (
                    <div>
                      <h3 className="text-2xs uppercase tracking-wide text-ink-muted">
                        How much better a random split makes each model look
                      </h3>
                      <Table headers={['Model', 'Metric', 'Scaffold', 'Random', 'Gap', 'Reading']}>
                        {Object.entries(experiment.comparison.optimism_gap).map(([family, gap]) => (
                          <tr key={family}>
                            <td className="px-2 py-1 text-ink-secondary">{titleCase(family)}</td>
                            <td className="px-2 py-1 text-2xs text-ink-muted">
                              {titleCase(gap.metric)}
                            </td>
                            <td className="px-2 py-1 font-mono tabular-nums text-ink">
                              {num(gap.scaffold, 3)}
                            </td>
                            <td className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                              {num(gap.random, 3)}
                            </td>
                            <td className="px-2 py-1 font-mono tabular-nums text-[color:var(--status-warning)]">
                              {num(gap.absolute_gap, 3)}
                              {gap.relative_gap_pct !== null &&
                                gap.relative_gap_pct !== undefined && (
                                  <span className="ml-1 text-2xs">
                                    ({num(gap.relative_gap_pct, 0)}%)
                                  </span>
                                )}
                            </td>
                            <td className="px-2 py-1 text-2xs text-ink-muted">
                              {gap.interpretation}
                            </td>
                          </tr>
                        ))}
                      </Table>
                      <p className="mt-1.5 text-2xs leading-relaxed text-ink-muted">
                        Molecular datasets contain families of near-identical analogues. A random
                        split puts some in train and some in test, so the model gets credit for
                        recognising a scaffold it has already seen. The scaffold-split figure is the
                        one that reflects performance on new chemistry.
                      </p>
                    </div>
                  )}

                  {experiment.comparison.split_difficulty && (
                    <p className="text-2xs text-ink-muted">
                      Split difficulty: {experiment.comparison.split_difficulty.test_scaffolds} test
                      scaffolds, {experiment.comparison.split_difficulty.shared_scaffolds} shared
                      with train (
                      {num(
                        (experiment.comparison.split_difficulty.test_scaffolds_unseen_fraction ??
                          0) * 100,
                        0,
                      )}
                      % of test scaffolds unseen).
                    </p>
                  )}

                  {Object.keys(experiment.environment).length > 0 && (
                    <details>
                      <summary className="cursor-pointer text-2xs uppercase tracking-wide text-ink-muted hover:text-ink-secondary">
                        Environment recorded at training time
                      </summary>
                      <dl className="mt-1 grid gap-x-6 sm:grid-cols-3">
                        {Object.entries(experiment.environment).map(([key, value]) => (
                          <div key={key} className="flex justify-between gap-2 text-2xs">
                            <dt className="text-ink-muted">{key}</dt>
                            <dd className="font-mono text-ink-secondary">
                              {truncate(String(value), 24)}
                            </dd>
                          </div>
                        ))}
                      </dl>
                    </details>
                  )}
                </div>
              );
            })}
        </Card>
      )}

      {focusModel && <ModelDetail model={focusModel} />}
    </div>
  );
}

function ModelDetail({ model }: { model: MlModel }) {
  const evaluations = useQuery({
    queryKey: ['evaluations', model.id],
    queryFn: () => api.evaluations({ modelId: model.id }),
  });

  const scaffold = evaluations.data?.evaluations.find(
    (evaluation) => evaluation.split_strategy === 'scaffold',
  );
  const cv = evaluations.data?.evaluations.find(
    (evaluation) => evaluation.split_strategy === 'cv_scaffold',
  );
  const metricKeys =
    model.task_type === 'classification' ? CLASSIFICATION_METRICS : REGRESSION_METRICS;

  return (
    <div className="space-y-4">
      <Card
        title={`${model.name} @ ${model.version}`}
        subtitle={model.description ?? undefined}
        actions={
          <Badge tone={model.status === 'active' ? 'good' : 'neutral'}>{model.status}</Badge>
        }
      >
        <dl className="grid gap-x-8 md:grid-cols-2">
          <Field label="Family">{titleCase(model.family)}</Field>
          <Field label="Framework">{model.framework}</Field>
          <Field label="Task">{model.task_type}</Field>
          <Field label="Units">{model.units ?? '—'}</Field>
          <Field label="Split protocol">{model.split_strategy ?? '—'}</Field>
          <Field label="Features">
            {String(model.feature_spec.representation ?? '—')}
            {model.feature_spec.fingerprint_bits
              ? ` · ${model.feature_spec.fingerprint_bits} bits`
              : ''}
            {model.n_descriptor_features ? ` + ${model.n_descriptor_features} descriptors` : ''}
          </Field>
          <Field label="Artifact">
            <code className="font-mono text-2xs">{model.artifact_hash?.slice(0, 16) ?? '—'}</code>
            {model.artifact_size_bytes && (
              <span className="ml-2 text-2xs text-ink-muted">
                {(model.artifact_size_bytes / 1024 / 1024).toFixed(1)} MB
              </span>
            )}
          </Field>
          <Field label="Seed / commit">
            {model.random_seed ?? '—'}
            {model.git_sha && (
              <code className="ml-2 font-mono text-2xs">{model.git_sha.slice(0, 8)}</code>
            )}
          </Field>
          {model.positive_class_meaning && (
            <Field label="Positive class">{model.positive_class_meaning}</Field>
          )}
        </dl>
      </Card>

      {scaffold && (
        <>
          <Card
            title="Scaffold-split metrics"
            subtitle={`n = ${scaffold.n_samples ?? '?'} · ${scaffold.notes ?? ''}`}
          >
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
              {metricKeys
                .filter((key) => (scaffold.metrics as Record<string, unknown>)[key])
                .map((key) => {
                  const metric = (
                    scaffold.metrics as Record<
                      string,
                      { value?: number; ci_low?: number | null; ci_high?: number | null }
                    >
                  )[key];
                  return (
                    <StatTile
                      key={key}
                      label={titleCase(key)}
                      value={num(metric?.value, 3)}
                      hint={
                        metric?.ci_low !== null && metric?.ci_low !== undefined
                          ? `95% CI ${num(metric.ci_low, 2)}–${num(metric.ci_high, 2)}`
                          : undefined
                      }
                    />
                  );
                })}
            </div>

            {model.task_type === 'classification' && (
              <>
                {(scaffold.metrics as Record<string, unknown>).baseline_pr_auc !== undefined && (
                  <p className="mt-3 text-2xs text-ink-muted">
                    PR-AUC must beat the positive-class rate of{' '}
                    {num((scaffold.metrics as Record<string, number>).baseline_pr_auc, 3)} to be
                    informative; with imbalanced data ROC-AUC alone can look respectable while the
                    model is useless at the operating point.
                  </p>
                )}
                {(scaffold.metrics as { confusion_matrix?: Record<string, number> })
                  .confusion_matrix && (
                  <div className="mt-3">
                    <h3 className="text-2xs uppercase tracking-wide text-ink-muted">
                      Confusion matrix
                    </h3>
                    <div className="mt-1 grid max-w-xs grid-cols-2 gap-1 text-2xs">
                      {Object.entries(
                        (scaffold.metrics as { confusion_matrix: Record<string, number> })
                          .confusion_matrix,
                      ).map(([key, value]) => (
                        <div
                          key={key}
                          className="flex items-center justify-between rounded border border-surface-border bg-surface-raised px-2 py-1"
                        >
                          <span className="text-ink-muted">{titleCase(key)}</span>
                          <span className="font-mono tabular-nums text-ink">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            {model.task_type === 'classification' ? (
              <Card title="Calibration" subtitle="Does a reported probability mean what it says?">
                <CalibrationChart calibration={scaffold.calibration} />
                {Boolean(model.calibration.method) && (
                  <p className="mt-2 text-2xs text-ink-muted">
                    Calibrator: {String(model.calibration.method)}
                    {model.calibration.improvement
                      ? ` · Brier ${num((model.calibration.improvement as Record<string, number>).baseline_brier, 4)} → ${num((model.calibration.improvement as Record<string, number>).calibrated_brier, 4)}`
                      : ''}
                    . Fitted on the validation fold only.
                  </p>
                )}
              </Card>
            ) : (
              <Card
                title="Uncertainty coverage"
                subtitle="Do the error bars behave like error bars?"
              >
                {scaffold.calibration.coverage !== null &&
                scaffold.calibration.coverage !== undefined ? (
                  <>
                    <div className="grid grid-cols-2 gap-3">
                      <StatTile
                        label="Observed coverage"
                        value={`${num(scaffold.calibration.coverage * 100, 1)}%`}
                        hint={`nominal ${num((scaffold.calibration.nominal ?? 0.95) * 100, 0)}%`}
                        tone={
                          scaffold.calibration.coverage >=
                          (scaffold.calibration.nominal ?? 0.95) - 0.1
                            ? 'good'
                            : 'warning'
                        }
                      />
                      <StatTile
                        label="Uncertainty↔error correlation"
                        value={num(scaffold.calibration.uncertainty_error_correlation, 3)}
                        hint="Above ~0.3 means the estimate carries signal"
                      />
                    </div>
                    <p className="mt-2 text-2xs leading-relaxed text-ink-muted">
                      {scaffold.calibration.interpretation}
                    </p>
                  </>
                ) : (
                  <p className="text-xs text-ink-muted">No uncertainty coverage recorded.</p>
                )}
              </Card>
            )}

            <Card
              title="Error by reported uncertainty"
              subtitle="Is the uncertainty usable for triage?"
            >
              <UncertaintyChart
                deciles={scaffold.error_analysis.error_by_uncertainty_decile ?? []}
              />
            </Card>

            <Card title="Prediction distribution" subtitle="On the held-out scaffold split">
              <DistributionChart
                distribution={scaffold.prediction_distribution}
                unit={model.units}
              />
            </Card>

            {cv && (
              <Card
                title="Cross-validation"
                subtitle={`${(cv.metrics as { folds?: number }).folds ?? '?'}-fold, scaffold-grouped`}
              >
                <div className="grid grid-cols-2 gap-3">
                  <StatTile
                    label={`Mean ${titleCase(String((cv.metrics as { primary_metric?: string }).primary_metric ?? ''))}`}
                    value={num((cv.metrics as { mean?: number }).mean, 4)}
                  />
                  <StatTile
                    label="Across-fold sd"
                    value={num((cv.metrics as { std?: number }).std, 4)}
                    hint="Spread across folds is split variance, not sampling noise"
                  />
                </div>
                {Array.isArray((cv.metrics as { per_fold?: number[] }).per_fold) && (
                  <p className="mt-2 font-mono text-2xs text-ink-muted">
                    per fold:{' '}
                    {(cv.metrics as { per_fold: number[] }).per_fold
                      .map((value) => num(value, 3))
                      .join(' · ')}
                  </p>
                )}
              </Card>
            )}
          </div>

          <Card
            title="Error analysis"
            subtitle="Where the model is wrong, and on what kind of molecule"
          >
            <div className="grid gap-4 lg:grid-cols-2">
              {scaffold.error_analysis.error_by_molecular_weight &&
                scaffold.error_analysis.error_by_molecular_weight.length > 0 && (
                  <div>
                    <h3 className="text-2xs uppercase tracking-wide text-ink-muted">
                      By molecular weight
                    </h3>
                    <Table headers={['Range (Da)', 'Molecules', 'Mean abs. error']}>
                      {scaffold.error_analysis.error_by_molecular_weight.map((bucket) => (
                        <tr key={bucket.range.join('-')}>
                          <td className="px-2 py-1 font-mono tabular-nums text-ink-secondary">
                            {num(bucket.range[0], 0)}–{num(bucket.range[1], 0)}
                          </td>
                          <td className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                            {bucket.count}
                          </td>
                          <td className="px-2 py-1 font-mono tabular-nums text-ink">
                            {num(bucket.mean_absolute_error, 3)}
                          </td>
                        </tr>
                      ))}
                    </Table>
                  </div>
                )}

              {scaffold.error_analysis.error_by_logp &&
                scaffold.error_analysis.error_by_logp.length > 0 && (
                  <div>
                    <h3 className="text-2xs uppercase tracking-wide text-ink-muted">By LogP</h3>
                    <Table headers={['Range', 'Molecules', 'Mean abs. error']}>
                      {scaffold.error_analysis.error_by_logp.map((bucket) => (
                        <tr key={bucket.range.join('-')}>
                          <td className="px-2 py-1 font-mono tabular-nums text-ink-secondary">
                            {num(bucket.range[0], 1)}–{num(bucket.range[1], 1)}
                          </td>
                          <td className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                            {bucket.count}
                          </td>
                          <td className="px-2 py-1 font-mono tabular-nums text-ink">
                            {num(bucket.mean_absolute_error, 3)}
                          </td>
                        </tr>
                      ))}
                    </Table>
                  </div>
                )}
            </div>

            {scaffold.error_analysis.worst_predictions &&
              scaffold.error_analysis.worst_predictions.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-2xs uppercase tracking-wide text-ink-muted">
                    Worst predictions on the held-out set
                  </h3>
                  <Table headers={['Molecule', 'Predicted', 'Actual', 'Abs. error', 'Uncertainty']}>
                    {scaffold.error_analysis.worst_predictions.slice(0, 8).map((worst) => (
                      <tr key={worst.identifier}>
                        <td
                          className="px-2 py-1 font-mono text-2xs text-ink-muted"
                          title={worst.smiles ?? ''}
                        >
                          {truncate(worst.identifier, 24)}
                        </td>
                        <td className="px-2 py-1 font-mono tabular-nums text-ink-secondary">
                          {num(worst.predicted, 3)}
                        </td>
                        <td className="px-2 py-1 font-mono tabular-nums text-ink">
                          {num(worst.actual, 3)}
                        </td>
                        <td className="px-2 py-1 font-mono tabular-nums text-[color:var(--status-warning)]">
                          {num(worst.absolute_error, 3)}
                        </td>
                        <td className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                          {num(worst.uncertainty, 3)}
                        </td>
                      </tr>
                    ))}
                  </Table>
                  <p className="mt-1.5 text-2xs text-ink-muted">
                    Reading the worst cases is how a chemist decides whether a model is safe for
                    their series. A large error paired with a small reported uncertainty is the
                    dangerous combination.
                  </p>
                </div>
              )}
          </Card>
        </>
      )}

      {!scaffold && evaluations.data && (
        <EmptyState
          title="No scaffold-split evaluation recorded for this model."
          detail="Re-run training: the pipeline always evaluates on a scaffold split."
        />
      )}
    </div>
  );
}
