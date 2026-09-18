/**
 * Candidate comparison and prioritisation.
 *
 * The weight sliders are the point: a researcher who disagrees with the default weighting
 * can change it and watch the ranking move, which is only honest because the ranking is a
 * transparent weighted sum rather than a black box. Weights are sent to the backend, so
 * the displayed score is always the backend's, never recomputed in the browser.
 */

import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useMutation, useQuery } from '@tanstack/react-query';
import clsx from 'clsx';
import { api } from '@/lib/api';
import type { RankedCandidate, RankingConfig } from '@/types/api';
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  Spinner,
  StatTile,
  Table,
} from '@/components/Primitives';
import { BasisBadge, PredictionDisclaimer } from '@/components/Provenance';
import { FactorBreakdownChart } from '@/components/Charts';
import { Molecule } from '@/components/Molecule';
import { num, titleCase, truncate } from '@/lib/format';

export default function CandidateComparison() {
  const [searchParams] = useSearchParams();
  const targetParam = searchParams.get('target') ?? '';
  const diseaseParam = searchParams.get('disease') ?? undefined;
  const compoundsParam = (searchParams.get('compounds') ?? '').split(',').filter(Boolean);

  const [target, setTarget] = useState(targetParam);
  const [presetName, setPresetName] = useState<string>('general_oral_small_molecule');
  const [weights, setWeights] = useState<Record<string, number>>({});
  const [expanded, setExpanded] = useState<string | null>(null);

  const presets = useQuery({ queryKey: ['ranking-presets'], queryFn: () => api.rankingPresets() });
  const activeConfig: RankingConfig | undefined = presets.data?.configs[presetName];

  // Sliders start from the preset's own weights, so "reset" is meaningful.
  useEffect(() => {
    if (activeConfig) {
      setWeights(
        Object.fromEntries(
          activeConfig.objectives.map((objective) => [objective.key, objective.weight]),
        ),
      );
    }
  }, [activeConfig]);

  const ranking = useMutation({
    mutationFn: () =>
      api.prioritize({
        target,
        disease: diseaseParam,
        compound_ids: compoundsParam.length ? compoundsParam : undefined,
        config_name: presetName,
        weights: Object.keys(weights).length ? weights : undefined,
        limit: 20,
      }),
  });

  const comparison = useQuery({
    queryKey: ['compare', compoundsParam, target, diseaseParam, presetName],
    queryFn: () =>
      api.compare({
        compound_ids: compoundsParam,
        target: target || undefined,
        disease: diseaseParam,
        config_name: presetName,
      }),
    enabled: compoundsParam.length > 1,
  });

  const candidates: RankedCandidate[] = useMemo(
    () => ranking.data?.candidates ?? [],
    [ranking.data],
  );

  const chartData = useMemo(
    () =>
      candidates.slice(0, 10).map((candidate) => ({
        name: truncate(candidate.name ?? candidate.compound_id, 18),
        total: candidate.total_score,
        factors: candidate.factors
          .filter((factor) => factor.contribution > 0)
          .map((factor) => ({
            key: factor.key,
            label: factor.label,
            basis: factor.basis,
            contribution: factor.contribution,
          })),
      })),
    [candidates],
  );

  return (
    <div className="space-y-4">
      <Card
        title="Candidate prioritisation"
        subtitle="A transparent weighted sum of measured evidence, computed properties and model predictions"
      >
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            if (target) ranking.mutate();
          }}
        >
          <label className="flex flex-col gap-1">
            <span className="text-2xs uppercase tracking-wide text-ink-muted">Target</span>
            <input
              value={target}
              onChange={(event) => setTarget(event.target.value)}
              placeholder="BACE1 or ENSG00000186318"
              className="w-56 rounded border border-surface-border bg-surface-raised px-2.5 py-1.5 text-xs text-ink placeholder:text-ink-muted focus:border-[color:var(--series-1)] focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-2xs uppercase tracking-wide text-ink-muted">Configuration</span>
            <select
              value={presetName}
              onChange={(event) => setPresetName(event.target.value)}
              className="rounded border border-surface-border bg-surface-raised px-2.5 py-1.5 text-xs text-ink focus:border-[color:var(--series-1)] focus:outline-none"
            >
              {presets.data?.presets.map((preset) => (
                <option key={preset.name} value={preset.name}>
                  {titleCase(preset.name)}
                </option>
              ))}
            </select>
          </label>
          <Button type="submit" variant="primary" disabled={!target || ranking.isPending}>
            {ranking.isPending ? 'Ranking…' : 'Rank candidates'}
          </Button>
          {diseaseParam && <Badge tone="neutral">disease context: {diseaseParam}</Badge>}
        </form>

        {activeConfig && (
          <p className="mt-2 max-w-3xl text-2xs leading-relaxed text-ink-muted">
            {activeConfig.description}
          </p>
        )}
        {ranking.error && (
          <div className="mt-3">
            <ErrorState error={ranking.error} />
          </div>
        )}
      </Card>

      {activeConfig && (
        <Card
          title="Objective weights"
          subtitle="Change a weight and re-rank. The backend recomputes the score; nothing is estimated in the browser."
          actions={
            <Button
              onClick={() =>
                setWeights(
                  Object.fromEntries(
                    activeConfig.objectives.map((objective) => [objective.key, objective.weight]),
                  ),
                )
              }
            >
              Reset to preset
            </Button>
          }
        >
          <div className="grid gap-x-6 gap-y-3 md:grid-cols-2">
            {activeConfig.objectives.map((objective) => (
              <div key={objective.key}>
                <div className="flex items-center justify-between gap-2">
                  <label
                    className="flex items-center gap-2 text-xs text-ink-secondary"
                    htmlFor={`w-${objective.key}`}
                  >
                    <BasisBadge basis={objective.basis} compact />
                    {objective.label}
                  </label>
                  <span className="font-mono text-2xs tabular-nums text-ink">
                    {num(weights[objective.key] ?? objective.weight, 2)}
                  </span>
                </div>
                <input
                  id={`w-${objective.key}`}
                  type="range"
                  min={0}
                  max={5}
                  step={0.25}
                  value={weights[objective.key] ?? objective.weight}
                  onChange={(event) =>
                    setWeights((current) => ({
                      ...current,
                      [objective.key]: Number(event.target.value),
                    }))
                  }
                  className="mt-1 w-full accent-[color:var(--series-1)]"
                />
                <p className="mt-0.5 text-2xs leading-snug text-ink-muted">
                  {objective.explanation}
                </p>
                <p className="text-2xs italic text-ink-muted">
                  Curve: {objective.desirability.description}
                </p>
              </div>
            ))}
          </div>

          {activeConfig.filters.length > 0 && (
            <div className="mt-4 border-t border-surface-border pt-3">
              <p className="text-2xs uppercase tracking-wide text-ink-muted">
                Hard filters (applied before scoring)
              </p>
              <ul className="mt-1 space-y-0.5 text-2xs text-ink-muted">
                {activeConfig.filters.map((filter) => (
                  <li key={filter.key}>
                    <span className="text-ink-secondary">{filter.label}</span> — {filter.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      )}

      {ranking.isPending && <Spinner label="Scoring candidates…" />}

      {ranking.data && candidates.length === 0 && (
        <EmptyState
          title="No candidates could be ranked."
          detail={ranking.data.warnings.join(' ') || ranking.data.metadata.warning || undefined}
        />
      )}

      {candidates.length > 0 && ranking.data && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile label="Ranked" value={ranking.data.n_ranked} />
            <StatTile
              label="Filtered out"
              value={ranking.data.n_filtered_out}
              hint={ranking.data.filtered_out[0]?.reason}
              tone={ranking.data.n_filtered_out > 0 ? 'warning' : 'neutral'}
            />
            <StatTile
              label="With measured potency"
              value={ranking.data.metadata.n_with_measured_potency ?? 0}
              hint="The rest are scored on properties and predictions only"
            />
            <StatTile
              label="Config hash"
              value={<span className="text-sm">{ranking.data.config_hash.slice(0, 10)}</span>}
              hint="Pinned on the research run for reproducibility"
            />
          </div>

          <Card
            title="Score composition"
            subtitle="Every point of every score is attributable to one objective; colour shows whether it came from a measurement or a model"
          >
            <FactorBreakdownChart candidates={chartData} />
            <PredictionDisclaimer>{ranking.data.disclaimer}</PredictionDisclaimer>
          </Card>

          <Card title="Ranked candidates">
            <Table headers={['Rank', 'Compound', 'Score', 'Leading factors', 'Flags', '']}>
              {candidates.map((candidate) => (
                <>
                  <tr key={candidate.compound_id} className="hover:bg-surface-raised/60">
                    <td className="px-2 py-2 font-mono tabular-nums text-ink">{candidate.rank}</td>
                    <td className="px-2 py-2">
                      <Link
                        to={`/compounds/${candidate.compound_id}`}
                        className="font-medium text-ink hover:underline"
                      >
                        {truncate(candidate.name ?? candidate.compound_id, 30)}
                      </Link>
                      {candidate.chembl_id && (
                        <span className="ml-2 font-mono text-2xs text-ink-muted">
                          {candidate.chembl_id}
                        </span>
                      )}
                    </td>
                    <td className="px-2 py-2">
                      <div className="flex items-center gap-2">
                        <span className="h-1.5 w-16 overflow-hidden rounded-sm bg-surface-raised">
                          <span
                            className="block h-full rounded-sm"
                            style={{
                              width: `${candidate.total_score * 100}%`,
                              background: 'var(--series-1)',
                            }}
                          />
                        </span>
                        <span className="font-mono text-2xs tabular-nums text-ink">
                          {candidate.total_score.toFixed(3)}
                        </span>
                      </div>
                    </td>
                    <td className="px-2 py-2 text-2xs text-ink-muted">
                      {candidate.top_contributors
                        .slice(0, 2)
                        .map((factor) => `${factor.label} ${factor.contribution.toFixed(2)}`)
                        .join(' · ') || '—'}
                    </td>
                    <td className="px-2 py-2">
                      <div className="flex flex-wrap gap-1">
                        {candidate.flags.slice(0, 3).map((flag) => (
                          <Badge
                            key={flag}
                            tone={
                              flag.includes('out_of_domain') || flag.includes('low_confidence')
                                ? 'warning'
                                : flag === 'approved_drug'
                                  ? 'good'
                                  : 'neutral'
                            }
                          >
                            {flag.replace(/_/g, ' ')}
                          </Badge>
                        ))}
                      </div>
                    </td>
                    <td className="px-2 py-2 text-right">
                      <Button
                        onClick={() =>
                          setExpanded(
                            expanded === candidate.compound_id ? null : candidate.compound_id,
                          )
                        }
                      >
                        {expanded === candidate.compound_id ? 'Hide' : 'Why?'}
                      </Button>
                    </td>
                  </tr>
                  {expanded === candidate.compound_id && (
                    <tr key={`${candidate.compound_id}-detail`}>
                      <td colSpan={6} className="bg-surface-raised/40 px-2 py-3">
                        <div className="flex flex-col gap-3 lg:flex-row">
                          {candidate.smiles && (
                            <Molecule smiles={candidate.smiles} width={200} height={140} />
                          )}
                          <div className="min-w-0 flex-1">
                            <Table
                              headers={[
                                'Objective',
                                'Basis',
                                'Value',
                                'Desirability',
                                'Weight',
                                'Contribution',
                              ]}
                            >
                              {candidate.factors.map((factor) => (
                                <tr key={factor.key}>
                                  <td
                                    className="px-2 py-1 text-ink-secondary"
                                    title={factor.explanation}
                                  >
                                    {factor.label}
                                  </td>
                                  <td className="px-2 py-1">
                                    <BasisBadge basis={factor.basis} compact />
                                  </td>
                                  <td className="px-2 py-1 font-mono tabular-nums text-ink">
                                    {factor.raw_value === null || factor.raw_value === undefined ? (
                                      <span className="text-2xs italic text-ink-muted">
                                        {factor.missing_reason ?? 'no data'}
                                      </span>
                                    ) : (
                                      num(factor.raw_value, 3)
                                    )}
                                  </td>
                                  <td className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                                    {num(factor.desirability, 2)}
                                  </td>
                                  <td className="px-2 py-1 font-mono tabular-nums text-ink-muted">
                                    {num(factor.normalized_weight, 2)}
                                  </td>
                                  <td className="px-2 py-1 font-mono tabular-nums text-ink">
                                    {num(factor.contribution, 3)}
                                    {factor.discounted_for_low_confidence && (
                                      <Badge
                                        tone="warning"
                                        title="Discounted: low confidence or out of domain"
                                      >
                                        discounted
                                      </Badge>
                                    )}
                                  </td>
                                </tr>
                              ))}
                            </Table>
                            <p className="mt-2 text-2xs text-ink-muted">
                              Scored on {candidate.objectives_scored} objective(s);{' '}
                              {candidate.objectives_missing} had no data and were excluded with the
                              remaining weights renormalised, so this score stays comparable with a
                              fully characterised compound.
                            </p>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </Table>

            {ranking.data.filtered_out.length > 0 && (
              <div className="mt-4 border-t border-surface-border pt-3">
                <p className="text-2xs uppercase tracking-wide text-ink-muted">
                  Removed by hard filters — shown so a filter is never invisible
                </p>
                <ul className="mt-1 space-y-0.5 text-2xs text-ink-muted">
                  {ranking.data.filtered_out.slice(0, 6).map((item) => (
                    <li key={item.compound_id}>
                      <span className="text-ink-secondary">{item.name ?? item.compound_id}</span> —{' '}
                      {item.reason} (value: {String(item.value)})
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </Card>
        </>
      )}

      {comparison.data && comparison.data.candidates.length > 1 && (
        <Card
          title="Side-by-side comparison"
          subtitle={comparison.data.note ?? 'Best and worst per objective are highlighted'}
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-surface-border text-left text-2xs uppercase tracking-wide text-ink-muted">
                  <th className="px-2 py-2 font-medium">Objective</th>
                  {comparison.data.candidates.map((candidate) => (
                    <th key={candidate.compound_id} className="px-2 py-2 font-medium">
                      <Link to={`/compounds/${candidate.compound_id}`} className="hover:underline">
                        {truncate(candidate.name ?? candidate.compound_id, 18)}
                      </Link>
                      <span className="ml-1 font-mono normal-case text-ink">
                        {candidate.total_score.toFixed(3)}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border/60">
                {comparison.data.objectives.map((objective) => {
                  const highlight = comparison.data.highlights[objective.key];
                  return (
                    <tr key={objective.key}>
                      <td className="px-2 py-1.5" title={objective.explanation}>
                        <span className="text-ink-secondary">{objective.label}</span>
                        <BasisBadge basis={objective.basis} compact />
                      </td>
                      {comparison.data.candidates.map((candidate) => {
                        const cell = candidate.values[objective.key];
                        const isBest = highlight?.best === candidate.compound_id;
                        const isWorst = highlight?.worst === candidate.compound_id;
                        return (
                          <td
                            key={candidate.compound_id}
                            className={clsx(
                              'px-2 py-1.5 font-mono tabular-nums',
                              isBest && 'bg-[color:var(--series-1)]/10 text-ink',
                              isWorst && !isBest && 'text-ink-muted',
                            )}
                          >
                            {cell?.raw_value === null || cell?.raw_value === undefined ? (
                              <span className="text-2xs italic text-ink-muted">
                                {cell?.missing_reason ? 'no data' : '—'}
                              </span>
                            ) : (
                              <>
                                {num(cell.raw_value, 3)}
                                {isBest && (
                                  <span className="ml-1 text-2xs text-[color:var(--series-1)]">
                                    best
                                  </span>
                                )}
                              </>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <PredictionDisclaimer>{comparison.data.disclaimer}</PredictionDisclaimer>
        </Card>
      )}

      {compoundsParam.length === 0 && !ranking.data && (
        <EmptyState
          title="Nothing selected to compare."
          detail="Enter a target above to rank its candidates, or pick compounds in the compound explorer and choose “compare”."
        />
      )}
    </div>
  );
}
