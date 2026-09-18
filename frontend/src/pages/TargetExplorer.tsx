/**
 * Target explorer.
 *
 * The evidence panel is the heart of this view. Evidence is grouped by type in a fixed
 * order (genetics first, literature last) with each type's meaning stated, because
 * "3 pieces of evidence" is meaningless without knowing whether they are human genetics
 * or text-mined co-mentions.
 */

import { useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import clsx from 'clsx';
import { api } from '@/lib/api';
import type { EvidenceItem } from '@/types/api';
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
import { BasisBadge, CitationKey, ProvenanceChip, SourceScore } from '@/components/Provenance';
import { EvidenceGraph } from '@/components/EvidenceGraph';
import { EvidenceTypeChart } from '@/components/Charts';
import { num, potency, titleCase, truncate } from '@/lib/format';

export default function TargetExplorer() {
  const params = useParams<{ identifier?: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const diseaseParam = searchParams.get('disease') ?? undefined;
  const [minPchembl, setMinPchembl] = useState(6);
  const [showGraph, setShowGraph] = useState(true);

  const disease = useQuery({
    queryKey: ['disease', diseaseParam],
    queryFn: () => api.disease(diseaseParam as string, { nTargets: 20 }),
    enabled: Boolean(diseaseParam),
  });

  const identifier = params.identifier ?? disease.data?.associated_targets[0]?.target.symbol;

  const target = useQuery({
    queryKey: ['target', identifier, diseaseParam, minPchembl],
    queryFn: () =>
      api.target(identifier as string, {
        disease: diseaseParam,
        evidenceLimit: 60,
        compoundLimit: 25,
        minPchembl,
      }),
    enabled: Boolean(identifier),
  });

  const graph = useQuery({
    queryKey: ['graph', diseaseParam, identifier],
    queryFn: () => api.graph({ disease: diseaseParam, target: identifier, maxTargets: 6 }),
    enabled: Boolean(identifier) && showGraph,
  });

  if (!identifier && !diseaseParam) {
    return (
      <EmptyState
        title="Pick a target or a disease."
        detail="Search on the dashboard, or open a disease to see its associated targets ranked by the Open Targets association score."
        action={
          <Link to="/">
            <Button variant="primary">Go to search</Button>
          </Link>
        }
      />
    );
  }

  return (
    <div className="space-y-4">
      {disease.data && (
        <Card
          title={`${disease.data.disease.name}`}
          subtitle={`${disease.data.disease.efo_id} · ${disease.data.counts.associated_targets} associated target(s)`}
          actions={<ProvenanceChip provenance={disease.data.disease.provenance} />}
        >
          <p className="max-w-4xl text-xs leading-relaxed text-ink-secondary">
            {truncate(
              disease.data.disease.description ?? 'No description published by the source.',
              420,
            )}
          </p>

          <h3 className="mt-4 text-2xs uppercase tracking-wide text-ink-muted">
            Associated targets — scores retrieved verbatim from Open Targets
          </h3>
          <Table
            headers={['Target', 'Association', 'Evidence types', 'Small-molecule tractability', '']}
            caption="An association score aggregates genetics, drugs, literature, animal models, expression and pathways. It is not a probability of success."
          >
            {disease.data.associated_targets.map(({ target: associated, association }) => (
              <tr
                key={associated.id}
                className={clsx(
                  'hover:bg-surface-raised/60',
                  associated.symbol === identifier && 'bg-surface-raised/80',
                )}
              >
                <td className="px-2 py-2">
                  <span className="font-medium text-ink">{associated.symbol}</span>
                  <span className="ml-2 text-2xs text-ink-muted">
                    {truncate(associated.name ?? '', 40)}
                  </span>
                </td>
                <td className="px-2 py-2">
                  <div className="flex items-center gap-2">
                    <span className="h-1.5 w-16 overflow-hidden rounded-sm bg-surface-raised">
                      <span
                        className="block h-full rounded-sm"
                        style={{
                          width: `${association.overall_score * 100}%`,
                          background: 'var(--series-1)',
                        }}
                      />
                    </span>
                    <span className="font-mono text-2xs tabular-nums text-ink">
                      {association.overall_score.toFixed(3)}
                    </span>
                  </div>
                </td>
                <td className="px-2 py-2 text-2xs text-ink-muted">
                  {Object.keys(association.datatype_scores).length > 0
                    ? Object.entries(association.datatype_scores)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 3)
                        .map(([type, score]) => `${titleCase(type)} ${score.toFixed(2)}`)
                        .join(' · ')
                    : '—'}
                </td>
                <td className="px-2 py-2 text-2xs text-ink-muted">
                  {Object.entries(associated.tractability)
                    .filter(([modality]) => modality.toLowerCase().includes('small molecule'))
                    .map(([, bucket]) => bucket)
                    .join(', ') || 'none reported'}
                </td>
                <td className="px-2 py-2 text-right">
                  <Button
                    onClick={() =>
                      navigate(
                        `/targets/${encodeURIComponent(associated.target_key)}?disease=${encodeURIComponent(diseaseParam ?? '')}`,
                      )
                    }
                  >
                    Inspect
                  </Button>
                </td>
              </tr>
            ))}
          </Table>
        </Card>
      )}

      {target.isLoading && <Spinner label="Loading target evidence…" />}
      {target.error && <ErrorState error={target.error} retry={() => void target.refetch()} />}

      {target.data && (
        <>
          <div className="grid gap-4 lg:grid-cols-3">
            <Card
              className="lg:col-span-2"
              title={`${target.data.target.symbol} — ${target.data.target.name ?? 'no approved name'}`}
              subtitle={
                target.data.target.target_class.join(' · ') ||
                target.data.target.biotype ||
                undefined
              }
              actions={<ProvenanceChip provenance={target.data.target.provenance} />}
            >
              <p className="max-w-4xl text-xs leading-relaxed text-ink-secondary">
                {truncate(
                  target.data.target.description ??
                    'No function description published by the source.',
                  520,
                )}
              </p>

              <dl className="mt-3 grid gap-x-6 sm:grid-cols-2">
                <Field label="Ensembl">
                  <code className="font-mono">{target.data.target.target_key}</code>
                </Field>
                <Field label="UniProt">
                  {target.data.target.uniprot_ids.slice(0, 3).join(', ') || '—'}
                </Field>
                <Field label="ChEMBL target">
                  {target.data.target.chembl_target_ids.join(', ') || 'no mapping'}
                </Field>
                <Field label="Biotype">{target.data.target.biotype ?? '—'}</Field>
                <Field label="Localisation">
                  {target.data.target.subcellular_locations.slice(0, 3).join(', ') || '—'}
                </Field>
                <Field label="Gene locus">
                  {target.data.target.gene
                    ? `chr${target.data.target.gene.chromosome ?? '?'}:${
                        target.data.target.gene.genomic_start ?? '?'
                      }`
                    : '—'}
                </Field>
              </dl>

              {target.data.disease_context?.association && (
                <div className="mt-3 rounded border border-surface-border bg-surface-raised p-3">
                  <p className="text-2xs uppercase tracking-wide text-ink-muted">
                    Association with {target.data.disease_context.disease.name}
                  </p>
                  <div className="mt-1 flex items-baseline gap-3">
                    <span className="font-mono text-lg tabular-nums text-ink">
                      {target.data.disease_context.association.overall_score.toFixed(3)}
                    </span>
                    <BasisBadge basis="measured" />
                    <ProvenanceChip
                      provenance={{
                        source: target.data.disease_context.association.source,
                        source_url: target.data.disease_context.association.source_url,
                        retrieved_at: target.data.disease_context.association.retrieved_at,
                      }}
                    />
                  </div>
                  <p className="mt-1 text-2xs text-ink-muted">
                    {target.data.disease_context.association.score_meaning}
                  </p>
                </div>
              )}
            </Card>

            <div className="space-y-4">
              <Card title="Tractability" subtitle="Retrieved verbatim from Open Targets" dense>
                {Object.keys(target.data.tractability.buckets).length === 0 ? (
                  <p className="text-xs text-ink-muted">No tractability assessment published.</p>
                ) : (
                  <dl className="space-y-1.5">
                    {Object.entries(target.data.tractability.buckets).map(([modality, bucket]) => (
                      <div
                        key={modality}
                        className="flex items-start justify-between gap-2 text-xs"
                      >
                        <dt className="text-ink-muted">{modality}</dt>
                        <dd className="text-right text-ink-secondary">{bucket}</dd>
                      </div>
                    ))}
                  </dl>
                )}
                <p className="mt-2 text-2xs leading-snug text-ink-muted">
                  {target.data.tractability.meaning}
                </p>
              </Card>

              <Card title="Evidence breakdown" dense>
                <EvidenceTypeChart counts={target.data.evidence.counts_by_type} />
                <p className="mt-2 text-2xs text-ink-muted">
                  {target.data.evidence.total} record(s) retrieved
                  {target.data.disease_context
                    ? ' for this target–disease pair'
                    : ' for this target'}
                  .
                </p>
              </Card>

              {target.data.safety_liabilities.length > 0 && (
                <Card title="Safety liabilities" dense>
                  <ul className="space-y-1 text-2xs text-ink-secondary">
                    {target.data.safety_liabilities.slice(0, 6).map((liability, index) => (
                      <li key={index} className="flex items-start gap-1.5">
                        <span aria-hidden className="text-[color:var(--status-warning)]">
                          ⚠
                        </span>
                        <span>
                          {String(liability.event ?? 'unspecified event')}
                          {liability.datasource ? ` — ${String(liability.datasource)}` : ''}
                        </span>
                      </li>
                    ))}
                  </ul>
                </Card>
              )}
            </div>
          </div>

          <Card
            title="Evidence"
            subtitle="Grouped by type, strongest class first. Every record links to its source."
          >
            {target.data.evidence.groups.length === 0 ? (
              <EmptyState
                title="No evidence records retrieved for this target."
                detail="This reflects the ingested corpus at the recorded dataset versions, not the absence of evidence in the literature."
              />
            ) : (
              <div className="space-y-4">
                {target.data.evidence.groups.map((group) => {
                  const records = group.records ?? group.items ?? [];
                  return (
                    <div key={group.evidence_type}>
                      <div className="flex flex-wrap items-baseline gap-2">
                        <h3 className="text-xs font-semibold text-ink">
                          {titleCase(group.evidence_type)}
                        </h3>
                        <span className="font-mono text-2xs tabular-nums text-ink-muted">
                          {group.count} record{group.count === 1 ? '' : 's'}
                        </span>
                        <BasisBadge basis="measured" compact />
                      </div>
                      {group.meaning && (
                        <p className="mt-0.5 max-w-3xl text-2xs leading-relaxed text-ink-muted">
                          {group.meaning}
                        </p>
                      )}
                      <ul className="mt-2 space-y-2">
                        {records.slice(0, 6).map((item) => (
                          <EvidenceRow key={item.id} item={item} />
                        ))}
                      </ul>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>

          <Card
            title="Known chemistry"
            subtitle="Experimental activity measurements from ChEMBL — not predictions"
            actions={
              <label className="flex items-center gap-2 text-2xs text-ink-muted">
                min pChEMBL
                <input
                  type="range"
                  min={4}
                  max={10}
                  step={0.5}
                  value={minPchembl}
                  onChange={(event) => setMinPchembl(Number(event.target.value))}
                  className="accent-[color:var(--series-1)]"
                />
                <span className="w-6 font-mono tabular-nums text-ink">{minPchembl}</span>
              </label>
            }
          >
            {target.data.known_compounds.length === 0 ? (
              <EmptyState
                title="No compounds above this potency threshold in the corpus."
                detail={
                  target.data.chemistry_summary.note ??
                  'Lower the threshold, or ingest this target’s ChEMBL bioactivity.'
                }
              />
            ) : (
              <>
                <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <StatTile label="Compounds" value={target.data.known_compounds.length} />
                  <StatTile
                    label="Most potent"
                    value={potency(target.data.known_compounds[0]?.best_pchembl)}
                    hint={`pChEMBL ${num(target.data.known_compounds[0]?.best_pchembl, 2)}`}
                  />
                  <StatTile
                    label="Approved drugs"
                    value={
                      target.data.known_compounds.filter((compound) => compound.is_approved_drug)
                        .length
                    }
                  />
                  <StatTile
                    label="Measurements"
                    value={target.data.known_compounds.reduce(
                      (sum, compound) => sum + compound.n_measurements,
                      0,
                    )}
                  />
                </div>
                <Table
                  headers={[
                    'Compound',
                    'ChEMBL',
                    'Best pChEMBL',
                    'Potency',
                    'Measurements',
                    'Phase',
                    '',
                  ]}
                >
                  {target.data.known_compounds.map((compound) => (
                    <tr key={compound.compound_id} className="hover:bg-surface-raised/60">
                      <td className="px-2 py-1.5 font-medium text-ink">
                        {truncate(compound.name, 34)}
                      </td>
                      <td className="px-2 py-1.5 font-mono text-2xs text-ink-muted">
                        {compound.chembl_id ?? '—'}
                      </td>
                      <td className="px-2 py-1.5 font-mono tabular-nums text-ink">
                        {compound.best_pchembl.toFixed(2)}
                      </td>
                      <td className="px-2 py-1.5 font-mono tabular-nums text-ink-secondary">
                        {potency(compound.best_pchembl)}
                      </td>
                      <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                        {compound.n_measurements}
                      </td>
                      <td className="px-2 py-1.5">
                        {compound.is_approved_drug ? (
                          <Badge tone="good">Approved</Badge>
                        ) : (
                          <span className="text-2xs text-ink-muted">
                            {num(compound.max_phase, 0)}
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-1.5 text-right">
                        <Link
                          to={`/compounds/${encodeURIComponent(compound.compound_id)}`}
                          className="text-2xs text-[color:var(--series-1)] hover:underline"
                        >
                          Open →
                        </Link>
                      </td>
                    </tr>
                  ))}
                </Table>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Link
                    to={`/compare?target=${encodeURIComponent(target.data.target.target_key)}${
                      diseaseParam ? `&disease=${encodeURIComponent(diseaseParam)}` : ''
                    }&compounds=${target.data.known_compounds
                      .slice(0, 4)
                      .map((compound) => compound.compound_id)
                      .join(',')}`}
                  >
                    <Button variant="primary">Compare the top candidates →</Button>
                  </Link>
                </div>
              </>
            )}
          </Card>

          <Card
            title="Evidence graph"
            subtitle="Disease → target → evidence → compound → assay → prediction"
            actions={
              <Button onClick={() => setShowGraph((value) => !value)}>
                {showGraph ? 'Hide' : 'Show'}
              </Button>
            }
          >
            {!showGraph && <p className="text-xs text-ink-muted">Graph hidden.</p>}
            {showGraph && graph.isLoading && <Spinner label="Building the graph…" />}
            {showGraph && graph.error && <ErrorState error={graph.error} />}
            {showGraph && graph.data && (
              <EvidenceGraph
                graph={graph.data}
                onSelectNode={(node) => {
                  if (node.kind === 'compound' && node.entity_id)
                    navigate(`/compounds/${node.entity_id}`);
                  if (node.kind === 'target' && node.entity_id)
                    navigate(`/targets/${node.entity_id}`);
                }}
              />
            )}
          </Card>
        </>
      )}
    </div>
  );
}

function EvidenceRow({ item }: { item: EvidenceItem }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <li className="rounded border border-surface-border bg-surface-raised/50 p-2.5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-xs text-ink-secondary">
            {item.title ?? titleCase(item.evidence_type)}
          </p>
          {item.summary && (
            <p className="mt-1 text-2xs leading-relaxed text-ink-muted">
              {expanded ? item.summary : truncate(item.summary, 220)}
              {item.summary.length > 220 && (
                <button
                  type="button"
                  onClick={() => setExpanded((value) => !value)}
                  className="ml-1 underline hover:text-ink-secondary"
                >
                  {expanded ? 'less' : 'more'}
                </button>
              )}
            </p>
          )}
          {item.publication && (
            <p className="mt-1 text-2xs text-ink-muted">
              {item.publication.journal ?? 'journal n/a'} {item.publication.year ?? ''}
              {item.publication.pmid && (
                <>
                  {' · '}
                  <a
                    href={
                      item.publication.url ??
                      `https://pubmed.ncbi.nlm.nih.gov/${item.publication.pmid}/`
                    }
                    target="_blank"
                    rel="noreferrer noopener"
                    className="text-[color:var(--series-1)] hover:underline"
                  >
                    PMID {item.publication.pmid}
                  </a>
                </>
              )}
              {item.publication.citation_count !== null &&
                item.publication.citation_count !== undefined && (
                  <> · {item.publication.citation_count} citations</>
                )}
            </p>
          )}
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <SourceScore
            score={item.score}
            scoreType={item.score_type}
            provided={item.score_provided_by_source}
          />
          <ProvenanceChip provenance={item.provenance} />
          <CitationKey value={item.citation_key} />
        </div>
      </div>
    </li>
  );
}
