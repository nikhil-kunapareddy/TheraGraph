/**
 * AI research assistant.
 *
 * The visible trace is the product, not a debug view: tool calls, retrieved records,
 * citations, predictions and the validation verdict are all shown. What is deliberately
 * *not* shown is hidden chain-of-thought — the model is asked for a short user-facing
 * rationale instead, and that is what appears.
 */

import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import clsx from 'clsx';
import { api } from '@/lib/api';
import type { RunDetail } from '@/types/api';
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  Spinner,
  StatTile,
} from '@/components/Primitives';
import { CitationKey } from '@/components/Provenance';
import { cost, duration, titleCase, truncate } from '@/lib/format';

const EXAMPLES = [
  'What are the most promising therapeutic targets for Alzheimer disease, and which compounds should we prioritise?',
  'What genetic evidence links APP to Alzheimer disease?',
  'Which compounds with measured activity against APP have the best predicted solubility?',
  'Find compounds structurally similar to the most potent known APP binder.',
];

const PHASE_LABELS: Record<string, string> = {
  plan: 'Planning',
  act: 'Adapting',
  validate: 'Validating retrieval',
  report: 'Writing the report',
};

export default function Assistant() {
  const [question, setQuestion] = useState('');
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [history, setHistory] = useState<Array<{ question: string; runId: string }>>([]);
  const queryClient = useQueryClient();
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const tools = useQuery({ queryKey: ['agent-tools'], queryFn: () => api.agentTools() });

  const start = useMutation({
    mutationFn: (value: string) =>
      api.research({ query: value, parameters: { n_targets: 10 }, execute_inline: true }),
    onSuccess: (result, value) => {
      setActiveRunId(result.run_id);
      setHistory((current) => [...current, { question: value, runId: result.run_id }]);
      setQuestion('');
      void queryClient.invalidateQueries({ queryKey: ['runs'] });
    },
  });

  const run = useQuery({
    queryKey: ['run', activeRunId],
    queryFn: () => api.run(activeRunId as string),
    enabled: Boolean(activeRunId),
    // While a run is still executing (queued via a worker), poll until it settles.
    refetchInterval: (query) => {
      const data = query.state.data as RunDetail | undefined;
      if (!data) return 2000;
      return ['succeeded', 'failed', 'cancelled'].includes(data.status) ? false : 2000;
    },
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [run.data?.tool_calls.length, run.data?.status]);

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
      <div className="space-y-4">
        <Card
          title="Research assistant"
          subtitle="Plans a workflow, calls typed tools, validates every citation, then writes an evidence-backed report"
        >
          <form
            className="space-y-2"
            onSubmit={(event) => {
              event.preventDefault();
              if (question.trim().length > 4) start.mutate(question.trim());
            }}
          >
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={3}
              placeholder="Ask a research question…"
              className="w-full resize-y rounded border border-surface-border bg-surface-raised px-3 py-2 text-sm text-ink placeholder:text-ink-muted focus:border-[color:var(--series-1)] focus:outline-none"
              onKeyDown={(event) => {
                if (
                  (event.metaKey || event.ctrlKey) &&
                  event.key === 'Enter' &&
                  question.trim().length > 4
                ) {
                  event.preventDefault();
                  start.mutate(question.trim());
                }
              }}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="submit"
                variant="primary"
                disabled={start.isPending || question.trim().length < 5}
              >
                {start.isPending ? 'Running the workflow…' : 'Run research'}
              </Button>
              <span className="text-2xs text-ink-muted">⌘/Ctrl + Enter</span>
            </div>
          </form>

          <div className="mt-3 space-y-1">
            <p className="text-2xs uppercase tracking-wide text-ink-muted">Example questions</p>
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setQuestion(example)}
                className="block w-full truncate rounded px-1 py-0.5 text-left text-2xs text-ink-muted hover:bg-surface-raised hover:text-ink-secondary"
                title={example}
              >
                {example}
              </button>
            ))}
          </div>

          {start.error && (
            <div className="mt-3">
              <ErrorState error={start.error} />
            </div>
          )}
        </Card>

        {history.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {history.map((entry) => (
              <button
                key={entry.runId}
                type="button"
                onClick={() => setActiveRunId(entry.runId)}
                className={clsx(
                  'rounded border px-2 py-1 text-2xs transition',
                  activeRunId === entry.runId
                    ? 'border-[color:var(--series-1)] bg-[color:var(--series-1)]/10 text-[color:var(--series-1)]'
                    : 'border-surface-border text-ink-muted hover:text-ink-secondary',
                )}
                title={entry.question}
              >
                {entry.runId} · {truncate(entry.question, 28)}
              </button>
            ))}
          </div>
        )}

        {start.isPending && <Spinner label="Planning, calling tools and validating citations…" />}

        {run.data && (
          <>
            <Card
              title={`Run ${run.data.run_id}`}
              subtitle={run.data.query}
              actions={
                <div className="flex items-center gap-2">
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
                  <Link to={`/runs/${run.data.run_id}`}>
                    <Button>Full trace →</Button>
                  </Link>
                </div>
              }
            >
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatTile label="Tool calls" value={run.data.n_tool_calls} />
                <StatTile label="Runtime" value={duration(run.data.runtime_ms)} />
                <StatTile
                  label="Tokens"
                  value={(run.data.input_tokens + run.data.output_tokens).toLocaleString()}
                  hint={
                    run.data.llm_backend === 'deterministic'
                      ? 'deterministic backend: no LLM calls'
                      : undefined
                  }
                />
                <StatTile label="Cost" value={cost(run.data.cost_usd)} />
              </div>

              {run.data.plan?.objective && (
                <div className="mt-4 rounded border border-surface-border bg-surface-raised/50 p-3">
                  <p className="text-2xs uppercase tracking-wide text-ink-muted">Plan</p>
                  <p className="mt-1 text-xs text-ink-secondary">{run.data.plan.strategy}</p>
                  <ol className="mt-2 space-y-1">
                    {(run.data.plan.steps ?? []).map((step) => (
                      <li key={step.step} className="flex gap-2 text-2xs">
                        <span className="font-mono text-ink-muted">{step.step}.</span>
                        <span className="font-mono text-[color:var(--series-1)]">{step.tool}</span>
                        <span className="text-ink-muted">{step.purpose}</span>
                      </li>
                    ))}
                  </ol>
                  {run.data.plan.ranking_preset && (
                    <p className="mt-2 text-2xs text-ink-muted">
                      Ranking configuration chosen: {titleCase(run.data.plan.ranking_preset)}
                    </p>
                  )}
                </div>
              )}
            </Card>

            <Card title="Execution trace" subtitle="Every tool call, with what it retrieved">
              <ol className="space-y-2">
                {run.data.executions.map((execution) => (
                  <li
                    key={execution.id}
                    className="rounded border border-surface-border bg-surface-raised/40 p-2.5"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge tone={execution.status === 'succeeded' ? 'good' : 'warning'}>
                        {PHASE_LABELS[execution.phase] ?? titleCase(execution.phase)}
                      </Badge>
                      {execution.model && (
                        <span className="font-mono text-2xs text-ink-muted">{execution.model}</span>
                      )}
                      {execution.latency_ms !== null && execution.latency_ms !== undefined && (
                        <span className="text-2xs text-ink-muted">
                          {duration(execution.latency_ms)}
                        </span>
                      )}
                      {(execution.input_tokens > 0 || execution.output_tokens > 0) && (
                        <span className="text-2xs text-ink-muted">
                          {execution.input_tokens}→{execution.output_tokens} tokens
                        </span>
                      )}
                    </div>
                    {execution.rationale && (
                      <p className="mt-1.5 text-xs leading-relaxed text-ink-secondary">
                        {execution.rationale}
                      </p>
                    )}
                  </li>
                ))}

                {run.data.tool_calls.map((call) => (
                  <li
                    key={call.id}
                    className={clsx(
                      'rounded border p-2.5',
                      call.status === 'succeeded'
                        ? 'border-surface-border bg-surface'
                        : 'border-[color:var(--status-warning)]/40 bg-[color:var(--status-warning)]/5',
                    )}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-2xs text-[color:var(--series-1)]">
                        {call.tool_name}
                      </span>
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
                      {call.latency_ms !== null && call.latency_ms !== undefined && (
                        <span className="text-2xs text-ink-muted">{duration(call.latency_ms)}</span>
                      )}
                      {call.cache_hit && <Badge tone="info">cache hit</Badge>}
                      {Object.entries(call.record_ids)
                        .filter(([, ids]) => ids.length > 0)
                        .map(([entity, ids]) => (
                          <span key={entity} className="text-2xs text-ink-muted">
                            {ids.length} {entity}
                          </span>
                        ))}
                    </div>

                    <details className="mt-1.5">
                      <summary className="cursor-pointer text-2xs text-ink-muted hover:text-ink-secondary">
                        arguments
                      </summary>
                      <pre className="mt-1 overflow-x-auto rounded bg-surface-sunken p-2 font-mono text-2xs text-ink-muted">
                        {JSON.stringify(call.arguments, null, 2)}
                      </pre>
                    </details>

                    {call.result_summary && (
                      <p className="mt-1.5 text-xs leading-relaxed text-ink-secondary">
                        {call.result_summary}
                      </p>
                    )}
                    {call.error && (
                      <p className="mt-1 text-2xs text-[color:var(--status-warning)]">
                        {call.error}
                      </p>
                    )}
                    {call.citations.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {call.citations.slice(0, 5).map((citation) => (
                          <CitationKey key={citation} value={citation} />
                        ))}
                        {call.citations.length > 5 && (
                          <span className="text-2xs text-ink-muted">
                            +{call.citations.length - 5} more
                          </span>
                        )}
                      </div>
                    )}
                  </li>
                ))}
              </ol>
              <div ref={bottomRef} />
            </Card>

            {run.data.report && (
              <Card
                title={run.data.report.title}
                subtitle="Evidence-backed report — facts are cited, predictions are labelled"
                actions={
                  <Badge
                    tone={
                      run.data.report.validation.rejected_claims.length === 0 ? 'good' : 'warning'
                    }
                    title="Claims citing records not retrieved in this run are removed before the report is shown"
                  >
                    {run.data.report.validation.validated_claims}/
                    {run.data.report.validation.total_claims} claims validated
                  </Badge>
                }
              >
                <p className="text-xs leading-relaxed text-ink-secondary">
                  {run.data.report.summary}
                </p>

                {run.data.report.claims.length > 0 && (
                  <ul className="mt-3 space-y-1.5">
                    {run.data.report.claims.map((claim, index) => (
                      <li key={index} className="flex items-start gap-2 text-xs">
                        <Badge
                          tone={
                            claim.kind === 'model_prediction'
                              ? 'predicted'
                              : claim.kind === 'retrieved_fact'
                                ? 'measured'
                                : 'info'
                          }
                        >
                          {claim.kind === 'model_prediction'
                            ? 'prediction'
                            : claim.kind === 'retrieved_fact'
                              ? 'fact'
                              : 'interpretation'}
                        </Badge>
                        <span className="min-w-0 flex-1 text-ink-secondary">
                          {claim.text}
                          {claim.citations.length > 0 && (
                            <span className="ml-1 inline-flex flex-wrap gap-1">
                              {claim.citations.map((citation) => (
                                <CitationKey key={citation} value={citation} />
                              ))}
                            </span>
                          )}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}

                {run.data.report.validation.rejected_claims.length > 0 && (
                  <div className="mt-3 rounded border border-[color:var(--status-warning)]/40 bg-[color:var(--status-warning)]/5 p-2.5">
                    <p className="text-2xs font-medium text-[color:var(--status-warning)]">
                      {run.data.report.validation.rejected_claims.length} claim(s) were removed
                    </p>
                    <ul className="mt-1 space-y-0.5 text-2xs text-ink-muted">
                      {run.data.report.validation.rejected_claims.map((rejected, index) => (
                        <li key={index}>
                          “{truncate(rejected.text, 90)}” — {rejected.reason}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {run.data.report.citations.length > 0 && (
                  <details className="mt-3">
                    <summary className="cursor-pointer text-2xs uppercase tracking-wide text-ink-muted hover:text-ink-secondary">
                      References ({run.data.report.citations.length})
                    </summary>
                    <ol className="mt-1.5 space-y-1 text-2xs text-ink-muted">
                      {run.data.report.citations.map((citation, index) => (
                        <li key={citation.key}>
                          {index + 1}. <CitationKey value={citation.key} />{' '}
                          {citation.reference_text ?? citation.title}
                          {citation.url && (
                            <a
                              href={citation.url}
                              target="_blank"
                              rel="noreferrer noopener"
                              className="ml-1 text-[color:var(--series-1)] hover:underline"
                            >
                              ↗
                            </a>
                          )}
                        </li>
                      ))}
                    </ol>
                  </details>
                )}

                <details className="mt-3">
                  <summary className="cursor-pointer text-2xs uppercase tracking-wide text-ink-muted hover:text-ink-secondary">
                    Full report (markdown)
                  </summary>
                  <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap rounded bg-surface-sunken p-3 font-mono text-2xs leading-relaxed text-ink-secondary">
                    {run.data.report.markdown}
                  </pre>
                </details>
              </Card>
            )}

            {run.data.warnings.length > 0 && (
              <Card title="Warnings raised during execution" dense>
                <ul className="space-y-0.5 text-2xs text-ink-muted">
                  {run.data.warnings.map((warning, index) => (
                    <li key={index}>
                      {String((warning as { message?: string }).message ?? warning)}
                    </li>
                  ))}
                </ul>
              </Card>
            )}
          </>
        )}

        {!run.data && !start.isPending && (
          <EmptyState
            title="Ask a question to start a research run."
            detail="Each run gets a permanent identifier, records its dataset and model versions, logs every tool call, and can be replayed."
          />
        )}
      </div>

      <aside className="space-y-4">
        <Card title="Guarantees" dense>
          <ul className="space-y-1.5 text-2xs leading-relaxed text-ink-secondary">
            {(tools.data?.guarantees ?? []).map((guarantee) => (
              <li key={guarantee} className="flex gap-1.5">
                <span aria-hidden className="text-[color:var(--status-good)]">
                  ✓
                </span>
                {guarantee}
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Tools available to the assistant" dense>
          {tools.isLoading && <Spinner />}
          {tools.data && (
            <ul className="space-y-1.5">
              {tools.data.tools.map((tool) => (
                <li key={tool.name}>
                  <details>
                    <summary className="cursor-pointer text-2xs">
                      <span className="font-mono text-[color:var(--series-1)]">{tool.name}</span>
                      {tool.mutates && (
                        <Badge tone="warning" title="Writes to the corpus; disabled by default">
                          writes
                        </Badge>
                      )}
                    </summary>
                    <p className="mt-1 text-2xs leading-relaxed text-ink-muted">
                      {tool.description}
                    </p>
                  </details>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </aside>
    </div>
  );
}
