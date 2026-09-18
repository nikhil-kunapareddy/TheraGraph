/** Formatting helpers. Scientific values need consistent, honest rendering. */

/** A number, or an em dash when the value is genuinely absent. */
export function num(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toFixed(digits);
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

/** pChEMBL to a human potency string; pChEMBL 8 is 10 nM. */
export function potency(pchembl: number | null | undefined): string {
  if (pchembl === null || pchembl === undefined) return '—';
  const nanomolar = 10 ** (9 - pchembl);
  if (nanomolar < 1) return `${(nanomolar * 1000).toFixed(0)} pM`;
  if (nanomolar < 1000) return `${nanomolar.toFixed(1)} nM`;
  if (nanomolar < 1_000_000) return `${(nanomolar / 1000).toFixed(1)} µM`;
  return `${(nanomolar / 1_000_000).toFixed(1)} mM`;
}

export function clinicalPhase(maxPhase: number | null | undefined): string {
  if (maxPhase === null || maxPhase === undefined) return 'Unknown';
  const labels: Record<string, string> = {
    '0': 'Preclinical',
    '0.5': 'Early phase 1',
    '1': 'Phase 1',
    '2': 'Phase 2',
    '3': 'Phase 3',
    '4': 'Approved',
  };
  return labels[String(maxPhase)] ?? `Phase ${maxPhase}`;
}

export function duration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '—';
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86_400) return `${Math.floor(seconds / 3600)} h ago`;
  const days = Math.floor(seconds / 86_400);
  if (days < 31) return `${days} d ago`;
  return new Date(iso).toLocaleDateString();
}

export function isoDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? '—' : date.toISOString().slice(0, 10);
}

export function titleCase(value: string): string {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase());
}

export function truncate(value: string, length = 80): string {
  return value.length <= length ? value : `${value.slice(0, length - 1)}…`;
}

export function cost(usd: number | null | undefined): string {
  if (usd === null || usd === undefined) return '—';
  if (usd === 0) return '$0.00';
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

/** Colour role for a confidence value, used with an icon + label, never alone. */
export function confidenceStatus(confidence: number | null | undefined): {
  role: 'good' | 'warning' | 'critical' | 'unknown';
  label: string;
} {
  if (confidence === null || confidence === undefined) {
    return { role: 'unknown', label: 'not reported' };
  }
  if (confidence >= 0.7) return { role: 'good', label: 'high' };
  if (confidence >= 0.4) return { role: 'warning', label: 'moderate' };
  return { role: 'critical', label: 'low' };
}

const SOURCE_LABELS: Record<string, string> = {
  open_targets: 'Open Targets',
  chembl: 'ChEMBL',
  pubchem: 'PubChem',
  europepmc: 'Europe PMC',
  moleculenet: 'MoleculeNet',
  theragraph_model: 'TheraGraph model',
};

/** Human-readable name for a data source id. */
export function sourceLabel(source: string): string {
  return SOURCE_LABELS[source] ?? source;
}
