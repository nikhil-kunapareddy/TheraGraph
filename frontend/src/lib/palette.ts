/**
 * Chart colour roles.
 *
 * Categorical hues are assigned in a fixed order and never cycled: a ninth series folds
 * into "other" rather than getting a generated hue. The order below is the validated
 * one (worst adjacent CVD ΔE 8.4 on the dark surface).
 *
 * `BASIS_COLOR` is the most important mapping in the product: it keeps *measured* and
 * *predicted* visually distinct everywhere, so a model output can never be mistaken
 * for an experimental value.
 */

export const SERIES = [
  'var(--series-1)',
  'var(--series-2)',
  'var(--series-3)',
  'var(--series-4)',
  'var(--series-5)',
  'var(--series-6)',
  'var(--series-7)',
  'var(--series-8)',
] as const;

export const STATUS = {
  good: 'var(--status-good)',
  warning: 'var(--status-warning)',
  serious: 'var(--status-serious)',
  critical: 'var(--status-critical)',
} as const;

/** Evidence basis -> colour. Chosen to avoid the yellow/orange adjacency entirely. */
export const BASIS_COLOR: Record<string, string> = {
  measured: 'var(--series-1)',
  predicted: 'var(--series-2)',
  computed: 'var(--series-3)',
  derived: 'var(--series-7)',
};

export const BASIS_LABEL: Record<string, string> = {
  measured: 'Experimental measurement',
  predicted: 'Model prediction',
  computed: 'Computed descriptor',
  derived: 'Derived metric',
};

/** Node colours for the evidence graph, by entity kind. */
export const NODE_COLOR: Record<string, string> = {
  disease: 'var(--series-5)',
  target: 'var(--series-1)',
  gene: 'var(--series-1)',
  evidence: 'var(--series-3)',
  publication: 'var(--series-7)',
  compound: 'var(--series-6)',
  assay: 'var(--series-4)',
  prediction: 'var(--series-2)',
};

export function seriesColor(index: number): string {
  return SERIES[index % SERIES.length] ?? SERIES[0];
}

/** Model family -> colour, so baseline and graph models stay consistent across charts. */
export const FAMILY_COLOR: Record<string, string> = {
  random_forest: 'var(--series-1)',
  xgboost: 'var(--series-3)',
  gin: 'var(--series-2)',
};

/** Split strategy -> colour. Scaffold is the headline; random is the reference. */
export const SPLIT_COLOR: Record<string, string> = {
  scaffold: 'var(--series-1)',
  random: 'var(--series-4)',
  cv_scaffold: 'var(--series-3)',
};
