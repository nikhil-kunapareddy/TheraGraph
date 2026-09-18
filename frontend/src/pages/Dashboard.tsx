/**
 * Research dashboard: the entry point.
 *
 * Search resolves diseases, targets and compounds in one box. When nothing matches
 * locally, the result explicitly offers upstream discovery rather than silently
 * returning an empty list -- "not in the corpus" and "does not exist" are different
 * answers and the UI must not conflate them.
 */

import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import clsx from 'clsx';
import { api } from '@/lib/api';
import type { EntityType, SearchHit } from '@/types/api';
import { useDebounced } from '@/hooks/useDebounced';
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
import { isoDate, num, titleCase } from '@/lib/format';

const ENTITY_FILTERS: Array<{ value: EntityType; label: string }> = [
  { value: 'disease', label: 'Diseases' },
  { value: 'target', label: 'Targets' },
  { value: 'compound', label: 'Compounds' },
];

const EXAMPLES = [
  'Alzheimer disease',
  'type 2 diabetes mellitus',
  'BACE1',
  'non-small cell lung carcinoma',
  'CHEMBL25',
];

export default function Dashboard() {
  const [query, setQuery] = useState('');
  const [types, setTypes] = useState<EntityType[]>([]);
  const [includeUpstream, setIncludeUpstream] = useState(false);
  const debounced = useDebounced(query, 300);
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const search = useQuery({
    queryKey: ['search', debounced, types, includeUpstream],
    queryFn: ({ signal }) =>
      api.search(debounced, {
        entityTypes: types.length ? types : undefined,
        limit: 10,
        includeUpstream,
        signal,
      }),
    enabled: debounced.trim().length > 1,
  });

  const corpus = useQuery({ queryKey: ['corpus'], queryFn: () => api.corpus() });
  const usage = useQuery({ queryKey: ['run-usage'], queryFn: () => api.runUsage() });
  const runs = useQuery({ queryKey: ['runs', 'recent'], queryFn: () => api.runs({ limit: 5 }) });

  const ingest = useMutation({
    mutationFn: (efoId: string) => api.ingestDisease({ efo_id: efoId, n_targets: 15 }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['corpus'] });
      void queryClient.invalidateQueries({ queryKey: ['search'] });
    },
  });

  function openHit(hit: SearchHit) {
    if (hit.entity_type === 'target') navigate(`/targets/${encodeURIComponent(hit.external_id)}`);
    else if (hit.entity_type === 'compound') navigate(`/compounds/${encodeURIComponent(hit.id)}`);
    else navigate(`/targets?disease=${encodeURIComponent(hit.external_id)}`);
  }

  return (
    <div className="space-y-5">
      <section className="rounded-lg border border-surface-border bg-surface p-5">
        <h1 className="text-lg font-semibold tracking-tight text-ink">
          Disease → target → evidence → compound → prioritised candidates
        </h1>
        <p className="mt-1 max-w-3xl text-xs leading-relaxed text-ink-secondary">
          Search the ingested corpus for a disease, a therapeutic target or a compound. Every value
          in this application carries its source and retrieval date; model predictions are labelled
          as predictions with their uncertainty and applicability domain.
        </p>

        <form
          className="mt-4 flex flex-wrap items-center gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            const first = search.data?.hits[0];
            if (first) openHit(first);
          }}
        >
          <label className="flex-1 min-w-[280px]">
            <span className="sr-only">Search diseases, targets and compounds</span>
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search a disease, gene symbol, or compound…"
              className="w-full rounded border border-surface-border bg-surface-raised px-3 py-2 text-sm text-ink placeholder:text-ink-muted focus:border-[color:var(--series-1)] focus:outline-none"
              autoFocus
            />
          </label>
          <div className="flex items-center gap-1" role="group" aria-label="Entity type filter">
            {ENTITY_FILTERS.map((filter) => {
              const active = types.includes(filter.value);
              return (
                <button
                  key={filter.value}
                  type="button"
                  aria-pressed={active}
                  onClick={() =>
                    setTypes((current) =>
                      current.includes(filter.value)
                        ? current.filter((value) => value !== filter.value)
                        : [...current, filter.value],
                    )
                  }
                  className={clsx(
                    'rounded border px-2 py-1.5 text-xs transition',
                    active
                      ? 'border-[color:var(--series-1)] bg-[color:var(--series-1)]/10 text-[color:var(--series-1)]'
                      : 'border-surface-border text-ink-muted hover:text-ink-secondary',
                  )}
                >
                  {filter.label}
                </button>
              );
            })}
          </div>
          <label className="flex items-center gap-1.5 text-2xs text-ink-muted">
            <input
              type="checkbox"
              checked={includeUpstream}
              onChange={(event) => setIncludeUpstream(event.target.checked)}
              className="accent-[color:var(--series-1)]"
            />
            also search Open Targets
          </label>
        </form>

        <div className="mt-2 flex flex-wrap items-center gap-2 text-2xs text-ink-muted">
          <span>Try:</span>
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => setQuery(example)}
              className="rounded border border-surface-border px-1.5 py-0.5 hover:text-ink-secondary"
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      {debounced.trim().length > 1 && (
        <Card
          title="Search results"
          subtitle={
            search.data
              ? `matched "${search.data.matched_term ?? debounced}" · tried ${search.data.terms_tried.length} term(s) · ${search.data.searched}`
              : undefined
          }
        >
          {search.isLoading && <Spinner label="Searching the corpus…" />}
          {search.error && <ErrorState error={search.error} retry={() => void search.refetch()} />}

          {search.data && search.data.hits.length > 0 && (
            <Table headers={['Entity', 'Name', 'Identifier', 'Source', 'Detail', '']}>
              {search.data.hits.map((hit) => (
                <tr key={`${hit.entity_type}:${hit.id}`} className="hover:bg-surface-raised/60">
                  <td className="px-2 py-2">
                    <Badge tone={hit.entity_type === 'compound' ? 'measured' : 'neutral'}>
                      {titleCase(hit.entity_type)}
                    </Badge>
                  </td>
                  <td className="px-2 py-2 font-medium text-ink">{hit.name}</td>
                  <td className="px-2 py-2 font-mono text-2xs text-ink-muted">{hit.external_id}</td>
                  <td className="px-2 py-2 text-2xs text-ink-muted">{hit.source}</td>
                  <td className="max-w-md px-2 py-2 text-2xs text-ink-muted">
                    {hit.entity_type === 'target' && Boolean(hit.metadata.tractability) && (
                      <span>
                        {Object.entries(hit.metadata.tractability as Record<string, string>)
                          .slice(0, 2)
                          .map(([modality, bucket]) => `${modality}: ${bucket}`)
                          .join(' · ') || '—'}
                      </span>
                    )}
                    {hit.entity_type === 'compound' && (
                      <span className="font-mono">
                        {String(hit.metadata.smiles ?? '').slice(0, 42)}
                      </span>
                    )}
                    {hit.entity_type === 'disease' && (
                      <span>
                        {(hit.metadata.therapeutic_areas as string[] | undefined)
                          ?.slice(0, 2)
                          .join(' · ') ?? '—'}
                      </span>
                    )}
                  </td>
                  <td className="px-2 py-2 text-right">
                    <Button onClick={() => openHit(hit)}>Open</Button>
                  </td>
                </tr>
              ))}
            </Table>
          )}

          {search.data && search.data.hits.length === 0 && (
            <EmptyState
              title={`Nothing in the ingested corpus matches "${debounced}".`}
              detail={
                search.data.upstream?.available
                  ? 'Open Targets does know about the entities below. Ingesting one fetches it, creates a new dataset version, and makes it searchable here.'
                  : 'Enable “also search Open Targets” to check whether the entity exists upstream and can be ingested. An empty result here means the corpus lacks the data, not that the literature does.'
              }
            />
          )}

          {search.data?.upstream?.hits && search.data.upstream.hits.length > 0 && (
            <div className="mt-4 rounded border border-dashed border-surface-border p-3">
              <p className="mb-2 text-xs text-ink-secondary">
                Discoverable in Open Targets {search.data.upstream.data_version} (not yet ingested)
              </p>
              <Table headers={['Entity', 'Name', 'Identifier', '']}>
                {search.data.upstream.hits.slice(0, 6).map((hit) => (
                  <tr key={hit.external_id}>
                    <td className="px-2 py-1.5">
                      <Badge tone="info">{titleCase(hit.entity_type)}</Badge>
                    </td>
                    <td className="px-2 py-1.5 text-ink-secondary">{hit.name}</td>
                    <td className="px-2 py-1.5 font-mono text-2xs text-ink-muted">
                      {hit.external_id}
                    </td>
                    <td className="px-2 py-1.5 text-right">
                      {hit.entity_type === 'disease' && (
                        <Button
                          variant="primary"
                          disabled={ingest.isPending}
                          onClick={() => ingest.mutate(hit.external_id)}
                        >
                          {ingest.isPending ? 'Queued…' : 'Ingest'}
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </Table>
              {ingest.data && (
                <p className="mt-2 text-2xs text-ink-muted">
                  Ingestion job <code className="font-mono">{ingest.data.job_id}</code> queued. A
                  worker must be running to process it.
                </p>
              )}
              {ingest.error && (
                <div className="mt-2">
                  <ErrorState error={ingest.error} />
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <Card title="Corpus" subtitle="What is currently ingested, and from where">
            {corpus.isLoading && <Spinner />}
            {corpus.error && <ErrorState error={corpus.error} />}
            {corpus.data && (
              <>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <StatTile label="Diseases" value={corpus.data.counts.diseases} />
                  <StatTile label="Targets" value={corpus.data.counts.targets} />
                  <StatTile
                    label="Compounds"
                    value={corpus.data.counts.compounds.toLocaleString()}
                  />
                  <StatTile
                    label="Measured activities"
                    value={corpus.data.counts.bioactivities.toLocaleString()}
                    hint="Experimental values from ChEMBL"
                  />
                </div>

                <h3 className="mt-4 text-2xs uppercase tracking-wide text-ink-muted">
                  Dataset versions in use
                </h3>
                <Table
                  headers={[
                    'Dataset',
                    'Version',
                    'Records',
                    'Retrieved',
                    'Quality',
                    'Content hash',
                  ]}
                >
                  {corpus.data.dataset_versions.slice(0, 8).map((version) => (
                    <tr key={version.label}>
                      <td className="px-2 py-1.5 text-ink-secondary">
                        {version.source}/{version.dataset}
                      </td>
                      <td className="px-2 py-1.5 font-mono text-2xs text-ink">{version.version}</td>
                      <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                        {version.record_count.toLocaleString()}
                      </td>
                      <td className="px-2 py-1.5 text-ink-muted">
                        {isoDate(version.retrieved_at)}
                      </td>
                      <td className="px-2 py-1.5">
                        {version.quality_score === null || version.quality_score === undefined ? (
                          <span className="text-2xs text-ink-muted">—</span>
                        ) : (
                          <Badge tone={version.quality_score >= 0.95 ? 'good' : 'warning'}>
                            {num(version.quality_score * 100, 1)}%
                          </Badge>
                        )}
                      </td>
                      <td className="px-2 py-1.5 font-mono text-2xs text-ink-muted">
                        {version.content_hash.slice(0, 12)}
                      </td>
                    </tr>
                  ))}
                </Table>
                <p className="mt-2 text-2xs text-ink-muted">
                  Quality is the fraction of upstream records that passed ingestion validation.
                  Content hashes make a dataset version verifiable: the same upstream data always
                  hashes the same.
                </p>
              </>
            )}
          </Card>

          <Card
            title="Recent research runs"
            subtitle="Every workflow is reproducible and fully traced"
          >
            {runs.isLoading && <Spinner />}
            {runs.data && runs.data.runs.length === 0 && (
              <EmptyState
                title="No research runs yet."
                detail="Ask the assistant a question to create one."
                action={
                  <Link to="/assistant">
                    <Button variant="primary">Open the assistant</Button>
                  </Link>
                }
              />
            )}
            {runs.data && runs.data.runs.length > 0 && (
              <Table headers={['Run', 'Question', 'Status', 'Tools', 'Runtime', '']}>
                {runs.data.runs.map((run) => (
                  <tr key={run.run_id} className="hover:bg-surface-raised/60">
                    <td className="px-2 py-1.5 font-mono text-2xs text-ink">{run.run_id}</td>
                    <td
                      className="max-w-sm truncate px-2 py-1.5 text-ink-secondary"
                      title={run.query}
                    >
                      {run.query}
                    </td>
                    <td className="px-2 py-1.5">
                      <Badge
                        tone={
                          run.status === 'succeeded'
                            ? 'good'
                            : run.status === 'failed'
                              ? 'critical'
                              : 'neutral'
                        }
                      >
                        {run.status}
                      </Badge>
                    </td>
                    <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                      {run.n_tool_calls}
                    </td>
                    <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                      {run.runtime_ms ? `${(run.runtime_ms / 1000).toFixed(1)}s` : '—'}
                    </td>
                    <td className="px-2 py-1.5 text-right">
                      <Link
                        to={`/runs/${run.run_id}`}
                        className="text-2xs text-[color:var(--series-1)] hover:underline"
                      >
                        Trace →
                      </Link>
                    </td>
                  </tr>
                ))}
              </Table>
            )}
          </Card>
        </div>

        <div className="space-y-4">
          <Card title="Agent observability" subtitle="Latency, spend and per-tool reliability">
            {usage.isLoading && <Spinner />}
            {usage.data && (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <StatTile label="Runs" value={usage.data.runs.runs} />
                  <StatTile
                    label="Failure rate"
                    value={`${num(usage.data.runs.failure_rate * 100, 1)}%`}
                    tone={usage.data.runs.failure_rate > 0.1 ? 'warning' : 'good'}
                  />
                  <StatTile
                    label="Avg runtime"
                    value={
                      usage.data.runs.avg_runtime_ms
                        ? `${(usage.data.runs.avg_runtime_ms / 1000).toFixed(1)}`
                        : '—'
                    }
                    unit="s"
                  />
                  <StatTile
                    label="Total spend"
                    value={`$${usage.data.runs.cost_usd.toFixed(4)}`}
                    hint={`${(usage.data.runs.input_tokens + usage.data.runs.output_tokens).toLocaleString()} tokens`}
                  />
                </div>
                {usage.data.tools.length > 0 && (
                  <>
                    <h3 className="mt-4 text-2xs uppercase tracking-wide text-ink-muted">
                      Tool latency
                    </h3>
                    <Table headers={['Tool', 'Calls', 'Errors', 'Avg ms']}>
                      {usage.data.tools.slice(0, 8).map((tool) => (
                        <tr key={tool.tool_name}>
                          <td className="px-2 py-1 font-mono text-2xs text-ink-secondary">
                            {tool.tool_name}
                          </td>
                          <td className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                            {tool.calls}
                          </td>
                          <td className="px-2 py-1 font-mono tabular-nums">
                            {tool.errors > 0 ? (
                              <span className="text-[color:var(--status-warning)]">
                                {tool.errors}
                              </span>
                            ) : (
                              <span className="text-ink-muted">0</span>
                            )}
                          </td>
                          <td className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                            {num(tool.avg_latency_ms, 0)}
                          </td>
                        </tr>
                      ))}
                    </Table>
                  </>
                )}
              </>
            )}
          </Card>

          <Card title="Data sources" subtitle="Licences and citations">
            {corpus.data && (
              <dl className="space-y-2">
                {corpus.data.sources.map((source) => (
                  <div
                    key={source.id}
                    className="border-b border-surface-border/60 pb-2 last:border-0"
                  >
                    <Field label={source.name}>
                      <span className="block text-2xs text-ink-muted">
                        {source.license ?? 'licence not recorded'}
                      </span>
                      {source.citation && (
                        <span className="mt-0.5 block text-2xs leading-snug text-ink-muted">
                          {source.citation}
                        </span>
                      )}
                    </Field>
                  </div>
                ))}
              </dl>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
