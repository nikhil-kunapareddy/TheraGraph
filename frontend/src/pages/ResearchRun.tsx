/**
 * Research run page: the complete reproducible execution trace.
 *
 * This view exists to answer "how exactly did you get this answer, and can I get it
 * again?". It shows the pinned dataset and model versions with content hashes, the
 * prompts as sent, every tool call, the retrieved record ids, and the replay controls
 * with any version drift stated up front.
 */

import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import clsx from 'clsx';
import { api } from '@/lib/api';
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  Field,
  Spinner,
  StatTile,
  Table,
} from '@/components/Primitives';
import { cost, duration, isoDate, num, relativeTime, titleCase, truncate } from '@/lib/format';

export default function ResearchRun() {
  const params = useParams<{ runId?: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [compareWith, setCompareWith] = useState<string[]>([]);
  const [showPrompts, setShowPrompts] = useState(false);

  const runs = useQuery({ queryKey: ['runs', 'all'], queryFn: () => api.runs({ limit: 40 }) });

  const run = useQuery({
    queryKey: ['run', params.runId],
    queryFn: () => api.run(params.runId as string),
    enabled: Boolean(params.runId),
  });

  const replayPlan = useQuery({
    queryKey: ['replay-plan', params.runId],
    queryFn: () => api.replayPlan(params.runId as string, true),
    enabled: Boolean(params.runId),
  });

  const replay = useMutation({
    mutationFn: (allowDrift: boolean) =>
      api.replay(params.runId as string, { allow_version_drift: allowDrift, execute_inline: true }),
    onSuccess: (result) => {
      void queryClient.invalidateQueries({ queryKey: ['runs'] });
      navigate(`/runs/${result.run_id}`);
    },
  });

  const comparison = useQuery({
    queryKey: ['compare-runs', params.runId, compareWith],
    queryFn: () => api.compareRuns([params.runId as string, ...compareWith]),
    enabled: Boolean(params.runId) && compareWith.length > 0,
  });

  if (!params.runId) {
    return (
      <Card title="Research runs" subtitle="Every workflow execution, reproducible and comparable">
        {runs.isLoading && <Spinner />}
        {runs.data && runs.data.runs.length === 0 && (
          <EmptyState
            title="No runs yet."
            detail="Ask the assistant a question to create one."
            action={
              <Link to="/assistant">
                <Button variant="primary">Open the assistant</Button>
              </Link>
            }
          />
        )}
        {runs.data && runs.data.runs.length > 0 && (
          <Table headers={['Run', 'Question', 'Status', 'Tools', 'Runtime', 'Cost', 'When', '']}>
            {runs.data.runs.map((item) => (
              <tr key={item.run_id} className="hover:bg-surface-raised/60">
                <td className="px-2 py-1.5 font-mono text-2xs text-ink">{item.run_id}</td>
                <td className="max-w-md truncate px-2 py-1.5 text-ink-secondary" title={item.query}>
                  {item.query}
                </td>
                <td className="px-2 py-1.5">
                  <Badge
                    tone={
                      item.status === 'succeeded'
                        ? 'good'
                        : item.status === 'failed'
                          ? 'critical'
                          : 'neutral'
                    }
                  >
                    {item.status}
                  </Badge>
                </td>
                <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                  {item.n_tool_calls}
                </td>
                <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                  {duration(item.runtime_ms)}
                </td>
                <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                  {cost(item.cost_usd)}
                </td>
                <td className="px-2 py-1.5 text-2xs text-ink-muted">
                  {relativeTime(item.created_at)}
                </td>
                <td className="px-2 py-1.5 text-right">
                  <Link
                    to={`/runs/${item.run_id}`}
                    className="text-2xs text-[color:var(--series-1)] hover:underline"
                  >
                    Open →
                  </Link>
                </td>
              </tr>
            ))}
          </Table>
        )}
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {run.isLoading && <Spinner label="Loading the execution trace…" />}
      {run.error && <ErrorState error={run.error} retry={() => void run.refetch()} />}

      {run.data && (
        <>
          <Card
            title={`Run ${run.data.run_id}`}
            subtitle={run.data.query}
            actions={
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  tone={
                    run.data.status === 'succeeded'
                      ? 'good'
                      : run.data.status === 'failed'
                        ? 'critical'
                        : 'neutral'
                  }
                >
                  {run.data.status}
                </Badge>
                <Button
                  variant="primary"
                  disabled={replay.isPending}
                  onClick={() => replay.mutate(!replayPlan.data?.drift.reproducible)}
                  title={
                    replayPlan.data?.drift.reproducible
                      ? 'Re-execute against the original pinned versions'
                      : 'Pinned versions have changed; the replay will record the drift'
                  }
                >
                  {replay.isPending ? 'Replaying…' : 'Re-run this workflow'}
                </Button>
              </div>
            }
          >
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <StatTile label="Steps" value={run.data.n_steps} />
              <StatTile label="Tool calls" value={run.data.n_tool_calls} />
              <StatTile label="Runtime" value={duration(run.data.runtime_ms)} />
              <StatTile
                label="Tokens"
                value={(run.data.input_tokens + run.data.output_tokens).toLocaleString()}
              />
              <StatTile label="Cost" value={cost(run.data.cost_usd)} />
              <StatTile
                label="Errors"
                value={run.data.n_errors}
                tone={run.data.n_errors > 0 ? 'critical' : 'neutral'}
              />
            </div>

            <dl className="mt-4 grid gap-x-8 md:grid-cols-2">
              <Field label="Reproducibility hash">
                <code className="break-all font-mono text-2xs">
                  {run.data.reproducibility_hash ?? '—'}
                </code>
                <span className="mt-0.5 block text-2xs text-ink-muted">
                  Digest of the query, parameters, dataset versions, model versions, ranking config,
                  prompt templates and seed. Two runs with the same hash saw identical inputs.
                </span>
              </Field>
              <Field label="Software">
                {run.data.software_version ?? '—'}
                {run.data.git_sha && (
                  <code className="ml-2 font-mono text-2xs">{run.data.git_sha.slice(0, 8)}</code>
                )}
              </Field>
              <Field label="Agent model">
                <code className="font-mono text-2xs">{run.data.agent_model ?? '—'}</code>
                <Badge tone={run.data.llm_backend === 'deterministic' ? 'neutral' : 'info'}>
                  {run.data.llm_backend ?? 'unknown'} backend
                </Badge>
              </Field>
              <Field label="Random seed">{run.data.random_seed ?? '—'}</Field>
              <Field label="Started">
                {isoDate(run.data.started_at)} ({relativeTime(run.data.started_at)})
              </Field>
              <Field label="Parameters">
                <code className="break-all font-mono text-2xs">
                  {JSON.stringify(run.data.parameters)}
                </code>
              </Field>
              {run.data.replay_of_run_id && (
                <Field label="Replay of">
                  <Link
                    to={`/runs/${run.data.replay_of_run_id}`}
                    className="text-[color:var(--series-1)] hover:underline"
                  >
                    {run.data.replay_of_run_id}
                  </Link>
                </Field>
              )}
            </dl>

            {replay.error && (
              <div className="mt-3">
                <ErrorState error={replay.error} />
              </div>
            )}
          </Card>

          {replayPlan.data && (
            <Card
              title="Reproducibility"
              subtitle="What a replay would run against"
              actions={
                <Badge tone={replayPlan.data.drift.reproducible ? 'good' : 'warning'}>
                  {replayPlan.data.drift.reproducible
                    ? 'exactly reproducible'
                    : 'version drift detected'}
                </Badge>
              }
            >
              {replayPlan.data.drift.reproducible ? (
                <p className="text-xs text-ink-secondary">
                  Every dataset and model version this run used is still current. A replay will see
                  identical inputs, so any difference in output comes from model non-determinism
                  alone.
                </p>
              ) : (
                <div className="space-y-2 text-xs">
                  <p className="text-ink-secondary">
                    Some pinned versions have changed since this run. A replay will use current
                    versions and record the drift on the new run, so the comparison stays honest.
                  </p>
                  {replayPlan.data.drift.dataset_versions_changed.length > 0 && (
                    <p className="text-2xs text-ink-muted">
                      Datasets changed: {replayPlan.data.drift.dataset_versions_changed.join(', ')}
                    </p>
                  )}
                  {replayPlan.data.drift.model_versions_changed.length > 0 && (
                    <p className="text-2xs text-ink-muted">
                      Models changed: {replayPlan.data.drift.model_versions_changed.join(', ')}
                    </p>
                  )}
                </div>
              )}
            </Card>
          )}

          <div className="grid gap-4 lg:grid-cols-2">
            <Card
              title="Pinned dataset versions"
              subtitle="With content hashes, so inputs are verifiable"
            >
              {run.data.pins.dataset_versions.length === 0 ? (
                <p className="text-xs text-ink-muted">No dataset versions were pinned.</p>
              ) : (
                <Table headers={['Dataset', 'Version', 'Records', 'Retrieved', 'Hash']}>
                  {run.data.pins.dataset_versions.map((version) => (
                    <tr key={version.id}>
                      <td className="px-2 py-1 text-ink-secondary">
                        {version.source}/{version.dataset}
                      </td>
                      <td className="px-2 py-1 font-mono text-2xs text-ink">{version.version}</td>
                      <td className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                        {version.record_count.toLocaleString()}
                      </td>
                      <td className="px-2 py-1 text-2xs text-ink-muted">
                        {isoDate(version.retrieved_at)}
                      </td>
                      <td className="px-2 py-1 font-mono text-2xs text-ink-muted">
                        {version.content_hash.slice(0, 10)}
                      </td>
                    </tr>
                  ))}
                </Table>
              )}
            </Card>

            <Card
              title="Pinned model versions"
              subtitle="Artifact hashes, so the weights are identifiable"
            >
              {run.data.pins.model_versions.length === 0 ? (
                <p className="text-xs text-ink-muted">
                  No models were pinned (none registered at run time).
                </p>
              ) : (
                <Table headers={['Model', 'Endpoint', 'Family', 'Scaffold metric', 'Artifact']}>
                  {run.data.pins.model_versions.map((model) => (
                    <tr key={model.id}>
                      <td className="px-2 py-1 font-mono text-2xs text-ink">{model.version}</td>
                      <td className="px-2 py-1 text-ink-secondary">{titleCase(model.endpoint)}</td>
                      <td className="px-2 py-1 text-2xs text-ink-muted">{model.family}</td>
                      <td className="px-2 py-1 font-mono tabular-nums text-2xs text-ink-muted">
                        {model.scaffold_metrics?.roc_auc !== undefined
                          ? `ROC-AUC ${num(model.scaffold_metrics.roc_auc, 3)}`
                          : model.scaffold_metrics?.rmse !== undefined
                            ? `RMSE ${num(model.scaffold_metrics.rmse, 3)}`
                            : '—'}
                      </td>
                      <td className="px-2 py-1 font-mono text-2xs text-ink-muted">
                        {model.artifact_hash?.slice(0, 10) ?? '—'}
                      </td>
                    </tr>
                  ))}
                </Table>
              )}
            </Card>
          </div>

          <Card title="Execution phases">
            <ol className="space-y-2">
              {run.data.executions.map((execution) => (
                <li
                  key={execution.id}
                  className="rounded border border-surface-border bg-surface-raised/40 p-2.5"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-2xs text-ink-muted">
                      #{execution.step_index}
                    </span>
                    <Badge tone={execution.status === 'succeeded' ? 'good' : 'warning'}>
                      {execution.phase}
                    </Badge>
                    {execution.prompt_template_id && (
                      <code className="font-mono text-2xs text-ink-muted">
                        {execution.prompt_template_id}
                      </code>
                    )}
                    {execution.latency_ms !== null && execution.latency_ms !== undefined && (
                      <span className="text-2xs text-ink-muted">
                        {duration(execution.latency_ms)}
                      </span>
                    )}
                    {execution.cost_usd > 0 && (
                      <span className="text-2xs text-ink-muted">{cost(execution.cost_usd)}</span>
                    )}
                  </div>
                  {execution.rationale && (
                    <p className="mt-1.5 text-xs leading-relaxed text-ink-secondary">
                      {execution.rationale}
                    </p>
                  )}
                  {Object.keys(execution.output_state).length > 0 && (
                    <details className="mt-1.5">
                      <summary className="cursor-pointer text-2xs text-ink-muted hover:text-ink-secondary">
                        structured state
                      </summary>
                      <pre className="mt-1 max-h-48 overflow-auto rounded bg-surface-sunken p-2 font-mono text-2xs text-ink-muted">
                        {JSON.stringify(execution.output_state, null, 2)}
                      </pre>
                    </details>
                  )}
                  {execution.error && (
                    <p className="mt-1 text-2xs text-[color:var(--status-critical)]">
                      {execution.error}
                    </p>
                  )}
                </li>
              ))}
            </ol>
          </Card>

          <Card
            title="Tool calls"
            subtitle="Arguments, results, retrieved record ids and validation verdicts"
          >
            <Table
              headers={['#', 'Tool', 'Status', 'Latency', 'Retrieved', 'Summary', 'Validation']}
            >
              {run.data.tool_calls.map((call) => (
                <tr key={call.id} className="align-top">
                  <td className="px-2 py-1.5 font-mono text-2xs text-ink-muted">
                    {call.step_index}
                  </td>
                  <td className="px-2 py-1.5 font-mono text-2xs text-[color:var(--series-1)]">
                    {call.tool_name}
                  </td>
                  <td className="px-2 py-1.5">
                    <Badge
                      tone={
                        call.status === 'succeeded'
                          ? 'good'
                          : call.status === 'skipped'
                            ? 'neutral'
                            : 'warning'
                      }
                    >
                      {call.status}
                    </Badge>
                  </td>
                  <td className="px-2 py-1.5 font-mono tabular-nums text-2xs text-ink-muted">
                    {num(call.latency_ms, 0)}
                  </td>
                  <td className="px-2 py-1.5 text-2xs text-ink-muted">
                    {Object.entries(call.record_ids)
                      .filter(([, ids]) => ids.length > 0)
                      .map(([entity, ids]) => `${ids.length} ${entity}`)
                      .join(', ') || '—'}
                  </td>
                  <td className="max-w-md px-2 py-1.5 text-2xs text-ink-secondary">
                    {truncate(call.result_summary ?? call.error ?? '—', 140)}
                  </td>
                  <td className="px-2 py-1.5">
                    {(call.validation as { ok?: boolean }).ok === false ? (
                      <Badge tone="warning" title={JSON.stringify(call.validation)}>
                        issues
                      </Badge>
                    ) : (
                      <span className="text-2xs text-ink-muted">ok</span>
                    )}
                  </td>
                </tr>
              ))}
            </Table>
          </Card>

          <Card
            title="Prompts as sent"
            subtitle="Stored verbatim so a replay uses byte-identical instructions"
            actions={
              <Button onClick={() => setShowPrompts((value) => !value)}>
                {showPrompts ? 'Hide' : `Show ${run.data.prompts.length}`}
              </Button>
            }
          >
            {!showPrompts ? (
              <p className="text-xs text-ink-muted">
                {run.data.prompts.length} prompt(s) recorded, with content hashes.
              </p>
            ) : (
              <ol className="space-y-2">
                {run.data.prompts.map((prompt, index) => (
                  <li
                    key={index}
                    className="rounded border border-surface-border bg-surface-raised/40 p-2.5"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <code className="font-mono text-2xs text-ink">{prompt.template_id}</code>
                      <Badge tone="neutral">{prompt.role}</Badge>
                      <code className="font-mono text-2xs text-ink-muted">
                        {prompt.content_hash}
                      </code>
                    </div>
                    <pre className="mt-1.5 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-surface-sunken p-2 font-mono text-2xs leading-relaxed text-ink-muted">
                      {prompt.content}
                    </pre>
                  </li>
                ))}
              </ol>
            )}
          </Card>

          <Card
            title="Retrieved records"
            subtitle="Primary keys of every row this run read — the basis for citation validation"
          >
            <dl className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {Object.entries(run.data.retrieved_record_ids).map(([entity, ids]) => (
                <div
                  key={entity}
                  className="rounded border border-surface-border bg-surface-raised/40 p-2"
                >
                  <dt className="text-2xs uppercase tracking-wide text-ink-muted">
                    {titleCase(entity)}
                  </dt>
                  <dd className="mt-0.5 font-mono text-sm tabular-nums text-ink">{ids.length}</dd>
                </div>
              ))}
            </dl>
          </Card>

          {run.data.report && (
            <Card
              title="Report"
              subtitle={`${run.data.report.validation.validated_claims}/${run.data.report.validation.total_claims} claims validated · content hash ${run.data.report.content_hash?.slice(0, 10) ?? '—'}`}
            >
              <pre className="max-h-[640px] overflow-auto whitespace-pre-wrap rounded bg-surface-sunken p-3 font-mono text-2xs leading-relaxed text-ink-secondary">
                {run.data.report.markdown}
              </pre>
            </Card>
          )}

          <Card
            title="Compare with another run"
            subtitle="Diffs pinned inputs, the tool sequence and the outputs"
          >
            <div className="flex flex-wrap gap-1.5">
              {(runs.data?.runs ?? [])
                .filter((item) => item.run_id !== run.data?.run_id)
                .slice(0, 12)
                .map((item) => {
                  const selected = compareWith.includes(item.run_id);
                  return (
                    <button
                      key={item.run_id}
                      type="button"
                      onClick={() =>
                        setCompareWith((current) =>
                          selected
                            ? current.filter((id) => id !== item.run_id)
                            : [...current, item.run_id].slice(0, 4),
                        )
                      }
                      className={clsx(
                        'rounded border px-2 py-1 font-mono text-2xs transition',
                        selected
                          ? 'border-[color:var(--series-1)] bg-[color:var(--series-1)]/10 text-[color:var(--series-1)]'
                          : 'border-surface-border text-ink-muted hover:text-ink-secondary',
                      )}
                      title={item.query}
                    >
                      {item.run_id}
                    </button>
                  );
                })}
            </div>

            {comparison.data && (
              <div className="mt-3 space-y-3">
                {comparison.data.differences.map((difference, index) => {
                  const diff = difference as {
                    run_id: string;
                    same_reproducibility_hash: boolean;
                    verdict: string;
                    tool_sequence_changed: boolean;
                    dataset_versions: { identical: boolean; changed: Record<string, unknown> };
                    model_versions: { identical: boolean };
                    parameters: { identical: boolean; changed: Record<string, unknown> };
                  };
                  return (
                    <div
                      key={index}
                      className="rounded border border-surface-border bg-surface-raised/40 p-3"
                    >
                      <div className="flex flex-wrap items-center gap-2">
                        <code className="font-mono text-2xs text-ink">{diff.run_id}</code>
                        <Badge tone={diff.same_reproducibility_hash ? 'good' : 'warning'}>
                          {diff.same_reproducibility_hash ? 'identical inputs' : 'inputs differ'}
                        </Badge>
                        {diff.tool_sequence_changed && (
                          <Badge tone="info">different tool sequence</Badge>
                        )}
                      </div>
                      <p className="mt-1.5 text-xs text-ink-secondary">{diff.verdict}</p>
                      <ul className="mt-1 space-y-0.5 text-2xs text-ink-muted">
                        <li>
                          Dataset versions:{' '}
                          {diff.dataset_versions.identical ? 'identical' : 'changed'}
                        </li>
                        <li>
                          Model versions: {diff.model_versions.identical ? 'identical' : 'changed'}
                        </li>
                        <li>
                          Parameters:{' '}
                          {diff.parameters.identical
                            ? 'identical'
                            : `changed (${Object.keys(diff.parameters.changed).join(', ')})`}
                        </li>
                      </ul>
                    </div>
                  );
                })}

                <Table
                  headers={['Metric', ...Object.keys(comparison.data.metrics.runtime_ms ?? {})]}
                >
                  {Object.entries(comparison.data.metrics).map(([metric, values]) => (
                    <tr key={metric}>
                      <td className="px-2 py-1 text-ink-secondary">{titleCase(metric)}</td>
                      {Object.values(values).map((value, index) => (
                        <td key={index} className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                          {value === null ? '—' : num(value as number, 2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </Table>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
