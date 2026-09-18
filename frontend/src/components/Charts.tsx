/**
 * Charts.
 *
 * Rules applied throughout, from the visualisation method:
 *
 * - One axis per chart. Never two y-scales -- two measures of different scale become
 *   two charts.
 * - Categorical hues assigned in fixed order, never cycled.
 * - Every chart has a hover layer; a legend whenever there are two or more series.
 * - A table view accompanies charts whose values a reader may need exactly.
 * - Sequential encoding is one hue light-to-dark; diverging is blue↔red with a neutral
 *   gray midpoint.
 */

import { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { ModelEvaluation } from '@/types/api';
import { BASIS_COLOR, BASIS_LABEL, SPLIT_COLOR, seriesColor } from '@/lib/palette';
import { num, titleCase } from '@/lib/format';
import { Table } from './Primitives';

const AXIS = {
  stroke: 'var(--surface-border)',
  tick: { fill: 'var(--text-secondary)', fontSize: 11 },
};

function TooltipBox({ rows, title }: { title?: string; rows: Array<[string, string]> }) {
  return (
    <div className="rounded border border-surface-border bg-surface-raised px-2.5 py-2 text-2xs shadow-lg">
      {title && <p className="mb-1 font-medium text-ink">{title}</p>}
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-ink-muted">{label}</dt>
            <dd className="font-mono tabular-nums text-ink-secondary">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/**
 * Reliability diagram.
 *
 * Bin width encodes how many molecules the bin holds, because a bin with three points
 * must not be read as strongly as one with three hundred. The diagonal is the
 * perfect-calibration reference.
 */
export function CalibrationChart({ calibration }: { calibration: ModelEvaluation['calibration'] }) {
  const bins = (calibration.bins ?? []).filter(
    (bin) => bin.count > 0 && bin.mean_predicted !== null,
  );
  if (bins.length === 0) {
    return (
      <p className="text-xs text-ink-muted">No reliability data: this model is not a classifier.</p>
    );
  }

  const maxCount = Math.max(...bins.map((bin) => bin.count));
  const data = bins.map((bin) => ({
    predicted: bin.mean_predicted ?? 0,
    observed: bin.observed_frequency ?? 0,
    count: bin.count,
    weight: bin.count / maxCount,
    gap: bin.gap ?? 0,
  }));

  return (
    <div className="space-y-2">
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
          <CartesianGrid strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="predicted"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            label={{
              value: 'Mean predicted probability',
              position: 'insideBottom',
              offset: -16,
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
            {...AXIS}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            label={{
              value: 'Observed frequency',
              angle: -90,
              position: 'insideLeft',
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
            {...AXIS}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="var(--text-muted)"
            strokeDasharray="4 4"
            strokeWidth={1}
            ifOverflow="extendDomain"
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const point = payload[0]?.payload as (typeof data)[number] | undefined;
              if (!point) return null;
              return (
                <TooltipBox
                  title="Calibration bin"
                  rows={[
                    ['Predicted', point.predicted.toFixed(3)],
                    ['Observed', point.observed.toFixed(3)],
                    ['Gap', point.gap.toFixed(3)],
                    ['Molecules', String(point.count)],
                  ]}
                />
              );
            }}
          />
          <Line
            type="monotone"
            dataKey="observed"
            name="Observed"
            stroke="var(--series-1)"
            strokeWidth={2}
            dot={(props) => {
              const { cx, cy, payload, index } = props as {
                cx: number;
                cy: number;
                index: number;
                payload: (typeof data)[number];
              };
              const radius = 4 + payload.weight * 6;
              return (
                <circle
                  key={index}
                  cx={cx}
                  cy={cy}
                  r={radius}
                  fill="var(--series-1)"
                  stroke="var(--surface-1)"
                  strokeWidth={2}
                />
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-2xs text-ink-muted">
        Marker size is the number of molecules in the bin. Points above the dashed line are
        under-confident; below it, over-confident.
        {calibration.ece !== null && calibration.ece !== undefined && (
          <> Expected calibration error {calibration.ece.toFixed(4)}.</>
        )}
      </p>
    </div>
  );
}

/**
 * Scaffold-versus-random comparison with confidence intervals.
 *
 * The interval is the point of this chart: with a 112-molecule test set, two models
 * whose intervals overlap are not measurably different, and the chart should make that
 * obvious rather than inviting a ranking.
 */
export function MetricComparisonChart({
  rows,
  metric,
  higherIsBetter,
}: {
  rows: Array<{
    label: string;
    split: string;
    value: number | null;
    ciLow?: number | null;
    ciHigh?: number | null;
  }>;
  metric: string;
  higherIsBetter: boolean;
}) {
  const [showTable, setShowTable] = useState(false);
  const data = rows
    .filter((row) => row.value !== null)
    .map((row) => ({
      name: `${row.label} · ${row.split}`,
      label: row.label,
      split: row.split,
      value: row.value as number,
      error:
        row.ciLow !== null &&
        row.ciLow !== undefined &&
        row.ciHigh !== null &&
        row.ciHigh !== undefined
          ? [
              Math.max(0, (row.value as number) - row.ciLow),
              Math.max(0, row.ciHigh - (row.value as number)),
            ]
          : undefined,
      ciLow: row.ciLow,
      ciHigh: row.ciHigh,
    }));

  if (data.length === 0) return <p className="text-xs text-ink-muted">No metrics recorded.</p>;

  return (
    <div className="space-y-2">
      <ResponsiveContainer width="100%" height={40 + data.length * 44}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="2 4" horizontal={false} />
          <XAxis
            type="number"
            domain={metric === 'roc_auc' ? [0.4, 1] : [0, 'auto']}
            label={{
              value: `${titleCase(metric)}${higherIsBetter ? ' (higher is better)' : ' (lower is better)'}`,
              position: 'insideBottom',
              offset: -12,
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
            {...AXIS}
          />
          <YAxis type="category" dataKey="name" width={200} {...AXIS} />
          {metric === 'roc_auc' && (
            <ReferenceLine
              x={0.5}
              stroke="var(--text-muted)"
              strokeDasharray="4 4"
              label={{ value: 'chance', fill: 'var(--text-muted)', fontSize: 10, position: 'top' }}
            />
          )}
          <Tooltip
            cursor={{ fill: 'var(--surface-raised)' }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const point = payload[0]?.payload as (typeof data)[number] | undefined;
              if (!point) return null;
              return (
                <TooltipBox
                  title={point.name}
                  rows={[
                    [titleCase(metric), num(point.value, 4)],
                    [
                      '95% CI',
                      point.ciLow !== null && point.ciLow !== undefined
                        ? `${num(point.ciLow, 3)} – ${num(point.ciHigh, 3)}`
                        : 'not computed',
                    ],
                  ]}
                />
              );
            }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18} isAnimationActive={false}>
            {data.map((row) => (
              <Cell key={row.name} fill={SPLIT_COLOR[row.split] ?? seriesColor(0)} />
            ))}
            {data.some((row) => row.error) && (
              <ErrorBar
                dataKey="error"
                width={4}
                strokeWidth={1.5}
                stroke="var(--text-secondary)"
              />
            )}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap items-center gap-3 text-2xs text-ink-muted">
        {Object.entries(SPLIT_COLOR).map(([split, color]) => (
          <span key={split} className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2 w-2 rounded-sm" style={{ background: color }} />
            {titleCase(split)}
          </span>
        ))}
        <button
          type="button"
          onClick={() => setShowTable((value) => !value)}
          className="ml-auto text-ink-muted underline hover:text-ink-secondary"
        >
          {showTable ? 'Hide table' : 'Show table'}
        </button>
      </div>

      {showTable && (
        <Table headers={['Model', 'Split', titleCase(metric), '95% CI']}>
          {data.map((row) => (
            <tr key={row.name}>
              <td className="px-2 py-1.5 text-ink-secondary">{row.label}</td>
              <td className="px-2 py-1.5 text-ink-muted">{titleCase(row.split)}</td>
              <td className="px-2 py-1.5 font-mono tabular-nums text-ink">{num(row.value, 4)}</td>
              <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                {row.ciLow !== null && row.ciLow !== undefined
                  ? `${num(row.ciLow, 3)} – ${num(row.ciHigh, 3)}`
                  : '—'}
              </td>
            </tr>
          ))}
        </Table>
      )}
    </div>
  );
}

/** Prediction distribution histogram. Single series, so no legend: the title names it. */
export function DistributionChart({
  distribution,
  unit,
}: {
  distribution: ModelEvaluation['prediction_distribution'];
  unit?: string | null;
}) {
  const histogram = distribution.histogram;
  if (!histogram || histogram.counts.length === 0) {
    return <p className="text-xs text-ink-muted">No prediction distribution recorded.</p>;
  }
  const data = histogram.counts.map((count, index) => ({
    bin: `${num(histogram.bin_edges[index], 2)}`,
    midpoint: ((histogram.bin_edges[index] ?? 0) + (histogram.bin_edges[index + 1] ?? 0)) / 2,
    count,
  }));

  return (
    <div className="space-y-2">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="bin"
            interval={Math.max(0, Math.floor(data.length / 8))}
            label={{
              value: unit ? `Predicted value (${unit})` : 'Predicted value',
              position: 'insideBottom',
              offset: -14,
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
            {...AXIS}
          />
          <YAxis allowDecimals={false} {...AXIS} />
          <Tooltip
            cursor={{ fill: 'var(--surface-raised)' }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const point = payload[0]?.payload as (typeof data)[number] | undefined;
              if (!point) return null;
              return (
                <TooltipBox
                  rows={[
                    ['Bin centre', num(point.midpoint, 3)],
                    ['Molecules', String(point.count)],
                  ]}
                />
              );
            }}
          />
          <Bar
            dataKey="count"
            fill="var(--series-1)"
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-2xs text-ink-muted">
        n = {distribution.count}, mean {num(distribution.mean, 3)}, sd {num(distribution.std, 3)}
        {distribution.quantiles && (
          <>
            {' '}
            · p05 {num(distribution.quantiles.p05, 2)} · median {num(distribution.quantiles.p50, 2)}{' '}
            · p95 {num(distribution.quantiles.p95, 2)}
          </>
        )}
      </p>
    </div>
  );
}

/**
 * Error by uncertainty decile.
 *
 * This chart answers the question that decides whether the uncertainty estimate is
 * usable: does reported uncertainty actually track error? A flat line means it does not,
 * and the caption says so.
 */
export function UncertaintyChart({
  deciles,
}: {
  deciles: NonNullable<ModelEvaluation['error_analysis']['error_by_uncertainty_decile']>;
}) {
  if (deciles.length === 0) {
    return (
      <p className="text-xs text-ink-muted">No uncertainty analysis recorded for this model.</p>
    );
  }
  const first = deciles[0]?.mean_absolute_error ?? 0;
  const last = deciles[deciles.length - 1]?.mean_absolute_error ?? 0;
  const ratio = first > 0 ? last / first : 0;

  return (
    <div className="space-y-2">
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={deciles} margin={{ top: 8, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="decile"
            label={{
              value: 'Reported-uncertainty decile (1 = most confident)',
              position: 'insideBottom',
              offset: -14,
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
            {...AXIS}
          />
          <YAxis {...AXIS} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const point = payload[0]?.payload as (typeof deciles)[number] | undefined;
              if (!point) return null;
              return (
                <TooltipBox
                  title={`Decile ${point.decile}`}
                  rows={[
                    ['Mean abs. error', num(point.mean_absolute_error, 4)],
                    ['Molecules', String(point.count)],
                    [
                      'Uncertainty range',
                      `${num(point.uncertainty_range[0], 3)} – ${num(point.uncertainty_range[1], 3)}`,
                    ],
                  ]}
                />
              );
            }}
          />
          <Line
            type="monotone"
            dataKey="mean_absolute_error"
            stroke="var(--series-2)"
            strokeWidth={2}
            dot={{ r: 4, fill: 'var(--series-2)', stroke: 'var(--surface-1)', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-2xs text-ink-muted">
        {ratio >= 1.5
          ? `Error in the least-confident decile is ${ratio.toFixed(1)}× the most-confident decile: the uncertainty estimate is informative and can be used for triage.`
          : 'Error barely changes across uncertainty deciles: this uncertainty estimate should not be used to triage predictions.'}
      </p>
    </div>
  );
}

/**
 * Ranking factor breakdown.
 *
 * A stacked horizontal bar per candidate, segmented by objective and coloured by
 * *basis* -- so a candidate carried by predictions rather than measurements is visible
 * at a glance. A 2px surface gap separates segments.
 */
export function FactorBreakdownChart({
  candidates,
}: {
  candidates: Array<{
    name: string;
    total: number;
    factors: Array<{ key: string; label: string; basis: string; contribution: number }>;
  }>;
}) {
  const [showTable, setShowTable] = useState(false);
  if (candidates.length === 0) return <p className="text-xs text-ink-muted">Nothing ranked.</p>;

  const objectiveKeys = Array.from(
    new Set(candidates.flatMap((candidate) => candidate.factors.map((factor) => factor.key))),
  );
  const basisByKey = new Map<string, string>();
  const labelByKey = new Map<string, string>();
  for (const candidate of candidates) {
    for (const factor of candidate.factors) {
      basisByKey.set(factor.key, factor.basis);
      labelByKey.set(factor.key, factor.label);
    }
  }

  // Order segments so no two adjacent segments share a basis colour where avoidable,
  // and measurements always read first.
  const basisOrder = ['measured', 'computed', 'predicted', 'derived'];
  const ordered = objectiveKeys.sort(
    (a, b) =>
      basisOrder.indexOf(basisByKey.get(a) ?? 'derived') -
      basisOrder.indexOf(basisByKey.get(b) ?? 'derived'),
  );

  const data = candidates.map((candidate) => {
    const row: Record<string, string | number> = { name: candidate.name, total: candidate.total };
    for (const factor of candidate.factors) {
      row[factor.key] = Math.max(0, factor.contribution);
    }
    return row;
  });

  return (
    <div className="space-y-2">
      <ResponsiveContainer width="100%" height={60 + candidates.length * 34}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="2 4" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 1]}
            label={{
              value: 'Prioritisation score contribution',
              position: 'insideBottom',
              offset: -14,
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
            {...AXIS}
          />
          <YAxis type="category" dataKey="name" width={150} {...AXIS} />
          <Tooltip
            cursor={{ fill: 'var(--surface-raised)' }}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const rows = payload
                .filter((entry) => Number(entry.value) > 0.0005)
                .sort((a, b) => Number(b.value) - Number(a.value))
                .map(
                  (entry) =>
                    [
                      `${labelByKey.get(String(entry.dataKey)) ?? entry.dataKey} (${
                        BASIS_LABEL[basisByKey.get(String(entry.dataKey)) ?? ''] ?? ''
                      })`,
                      num(Number(entry.value), 3),
                    ] as [string, string],
                );
              return <TooltipBox title={String(label)} rows={rows} />;
            }}
          />
          <Legend verticalAlign="bottom" height={1} content={() => null} />
          {ordered.map((key) => (
            <Bar
              key={key}
              dataKey={key}
              stackId="score"
              fill={BASIS_COLOR[basisByKey.get(key) ?? 'derived'] ?? seriesColor(0)}
              isAnimationActive={false}
              radius={key === ordered[ordered.length - 1] ? [0, 4, 4, 0] : undefined}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap items-center gap-3 text-2xs text-ink-muted">
        {basisOrder
          .filter((basis) => Array.from(basisByKey.values()).includes(basis))
          .map((basis) => (
            <span key={basis} className="inline-flex items-center gap-1.5">
              <span
                aria-hidden
                className="h-2 w-2 rounded-sm"
                style={{ background: BASIS_COLOR[basis] }}
              />
              {BASIS_LABEL[basis]}
            </span>
          ))}
        <button
          type="button"
          onClick={() => setShowTable((value) => !value)}
          className="ml-auto underline hover:text-ink-secondary"
        >
          {showTable ? 'Hide table' : 'Show table'}
        </button>
      </div>

      {showTable && (
        <Table
          headers={['Candidate', ...ordered.map((key) => labelByKey.get(key) ?? key), 'Total']}
        >
          {candidates.map((candidate) => (
            <tr key={candidate.name}>
              <td className="px-2 py-1.5 text-ink-secondary">{candidate.name}</td>
              {ordered.map((key) => {
                const factor = candidate.factors.find((item) => item.key === key);
                return (
                  <td key={key} className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
                    {factor ? num(factor.contribution, 3) : '—'}
                  </td>
                );
              })}
              <td className="px-2 py-1.5 font-mono tabular-nums text-ink">
                {num(candidate.total, 3)}
              </td>
            </tr>
          ))}
        </Table>
      )}
    </div>
  );
}

/** Evidence counts by type. Sequential magnitude, one hue, darkest = most evidence. */
export function EvidenceTypeChart({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort(([, a], [, b]) => b - a);
  if (entries.length === 0) return <p className="text-xs text-ink-muted">No evidence retrieved.</p>;
  const max = Math.max(...entries.map(([, count]) => count));
  const ramp = ['var(--seq-250)', 'var(--seq-400)', 'var(--seq-550)', 'var(--seq-700)'];

  return (
    <ul className="space-y-1.5">
      {entries.map(([type, count]) => {
        const fraction = count / max;
        const step = ramp[Math.min(ramp.length - 1, Math.floor(fraction * ramp.length))] ?? ramp[1];
        return (
          <li key={type} className="grid grid-cols-[9rem_1fr_2.5rem] items-center gap-2 text-2xs">
            <span className="truncate text-ink-secondary" title={titleCase(type)}>
              {titleCase(type)}
            </span>
            <span className="h-2.5 overflow-hidden rounded-sm bg-surface-raised">
              <span
                className="block h-full rounded-sm"
                style={{ width: `${Math.max(4, fraction * 100)}%`, background: step }}
              />
            </span>
            <span className="text-right font-mono tabular-nums text-ink">{count}</span>
          </li>
        );
      })}
    </ul>
  );
}

/** Scatter of predicted versus measured potency. Two axes of the same quantity. */
export function PredictedVsMeasuredChart({
  points,
  unit,
}: {
  points: Array<{ name: string; predicted: number; measured: number; inDomain?: boolean | null }>;
  unit?: string | null;
}) {
  if (points.length === 0)
    return <p className="text-xs text-ink-muted">No paired values available.</p>;
  const values = points.flatMap((point) => [point.predicted, point.measured]);
  const min = Math.min(...values);
  const max = Math.max(...values);

  return (
    <div className="space-y-2">
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
          <CartesianGrid strokeDasharray="2 4" />
          <XAxis
            type="number"
            dataKey="measured"
            domain={[min, max]}
            label={{
              value: `Measured${unit ? ` (${unit})` : ''}`,
              position: 'insideBottom',
              offset: -16,
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
            {...AXIS}
          />
          <YAxis
            type="number"
            dataKey="predicted"
            domain={[min, max]}
            label={{
              value: `Predicted${unit ? ` (${unit})` : ''}`,
              angle: -90,
              position: 'insideLeft',
              fill: 'var(--text-muted)',
              fontSize: 11,
            }}
            {...AXIS}
          />
          <ReferenceLine
            segment={[
              { x: min, y: min },
              { x: max, y: max },
            ]}
            stroke="var(--text-muted)"
            strokeDasharray="4 4"
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const point = payload[0]?.payload as (typeof points)[number] | undefined;
              if (!point) return null;
              return (
                <TooltipBox
                  title={point.name}
                  rows={[
                    ['Measured', num(point.measured, 3)],
                    ['Predicted', num(point.predicted, 3)],
                    ['Applicability', point.inDomain === false ? 'out of domain' : 'in domain'],
                  ]}
                />
              );
            }}
          />
          <Scatter data={points} isAnimationActive={false}>
            {points.map((point) => (
              <Cell
                key={point.name}
                fill={point.inDomain === false ? 'var(--status-warning)' : 'var(--series-1)'}
              />
            ))}
          </Scatter>
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-2xs text-ink-muted">
        The dashed diagonal is perfect agreement. Amber points are out of the model&apos;s
        applicability domain and should be read as extrapolation.
      </p>
    </div>
  );
}
