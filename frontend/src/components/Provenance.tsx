/**
 * Provenance display.
 *
 * Every scientific value in the UI is accompanied by where it came from. These
 * components are the single implementation of that rule, so a new view cannot
 * accidentally render a number without its source.
 */

import type { Provenance as ProvenanceData } from '@/types/api';
import { isoDate, sourceLabel } from '@/lib/format';
import { Badge } from './Primitives';

export function ProvenanceChip({ provenance }: { provenance: ProvenanceData }) {
  const label = sourceLabel(provenance.source);
  const detail = [
    provenance.source_identifier ? `id ${provenance.source_identifier}` : null,
    `retrieved ${isoDate(provenance.retrieved_at)}`,
    provenance.dataset_version ? `dataset ${provenance.dataset_version}` : null,
  ]
    .filter(Boolean)
    .join(' · ');

  const content = (
    <span className="inline-flex items-center gap-1 rounded border border-surface-border bg-surface-raised px-1.5 py-0.5 text-2xs text-ink-muted">
      <span aria-hidden>◈</span>
      {label}
      <span className="text-ink-muted/70">· {isoDate(provenance.retrieved_at)}</span>
    </span>
  );

  if (!provenance.source_url) return <span title={detail}>{content}</span>;
  return (
    <a
      href={provenance.source_url}
      target="_blank"
      rel="noreferrer noopener"
      title={`${detail} — open the source record`}
      className="hover:underline"
    >
      {content}
    </a>
  );
}

/**
 * The measured-versus-predicted marker.
 *
 * This is the product's central claim: retrieved facts and model outputs are never
 * presented as the same kind of thing. One component, used everywhere.
 */
export function BasisBadge({
  basis,
  compact,
}: {
  basis: 'measured' | 'predicted' | 'computed' | 'derived' | string;
  compact?: boolean;
}) {
  if (basis === 'predicted') {
    return (
      <Badge
        tone="predicted"
        icon={<span aria-hidden>◐</span>}
        title="Model prediction, not a measurement"
      >
        {compact ? 'pred' : 'Prediction'}
      </Badge>
    );
  }
  if (basis === 'measured') {
    return (
      <Badge
        tone="measured"
        icon={<span aria-hidden>●</span>}
        title="Experimental measurement from a public source"
      >
        {compact ? 'exp' : 'Measured'}
      </Badge>
    );
  }
  if (basis === 'computed') {
    return (
      <Badge
        tone="neutral"
        icon={<span aria-hidden>ƒ</span>}
        title="Computed from the structure with RDKit"
      >
        {compact ? 'calc' : 'Computed'}
      </Badge>
    );
  }
  return (
    <Badge tone="info" icon={<span aria-hidden>Σ</span>} title="Derived from other values">
      {compact ? 'deriv' : 'Derived'}
    </Badge>
  );
}

/** A score shown only when the source provided one. */
export function SourceScore({
  score,
  scoreType,
  provided,
}: {
  score?: number | null;
  scoreType?: string | null;
  provided?: boolean;
}) {
  if (score === null || score === undefined || provided === false) {
    return (
      <span
        className="text-2xs italic text-ink-muted"
        title="The source did not publish a score for this record"
      >
        not provided by source
      </span>
    );
  }
  return (
    <span className="font-mono text-xs tabular-nums text-ink" title={scoreType ?? undefined}>
      {score.toFixed(3)}
    </span>
  );
}

export function CitationKey({ value }: { value: string }) {
  return (
    <code
      className="rounded bg-surface-raised px-1 py-0.5 font-mono text-2xs text-ink-muted"
      title="Citation key: the handle the research agent must use to cite this record"
    >
      {value}
    </code>
  );
}

/** Standing disclaimer for anything model-generated. */
export function PredictionDisclaimer({ children }: { children?: string }) {
  return (
    <p className="mt-2 flex items-start gap-1.5 text-2xs leading-relaxed text-ink-muted">
      <span aria-hidden className="text-[color:var(--series-2)]">
        ◐
      </span>
      <span>
        {children ??
          'Model prediction, not an experimental measurement. Values for molecules outside the training applicability domain are extrapolation.'}
      </span>
    </p>
  );
}
