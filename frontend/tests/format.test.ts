import { describe, expect, it } from 'vitest';
import {
  clinicalPhase,
  confidenceStatus,
  cost,
  duration,
  isoDate,
  num,
  pct,
  potency,
  relativeTime,
  sourceLabel,
  titleCase,
  truncate,
} from '@/lib/format';

describe('num', () => {
  it('renders an em dash for genuinely absent values', () => {
    // A missing measurement must never render as 0.
    expect(num(null)).toBe('—');
    expect(num(undefined)).toBe('—');
    expect(num(Number.NaN)).toBe('—');
    expect(num(0)).toBe('0.00');
  });

  it('formats with the requested precision', () => {
    expect(num(3.14159, 3)).toBe('3.142');
    expect(num(1234.5)).toBe('1,235');
  });
});

describe('potency', () => {
  it('converts pChEMBL to a human potency', () => {
    // pChEMBL 8 is 10 nM by definition.
    expect(potency(8)).toBe('10.0 nM');
    expect(potency(9)).toBe('1.0 nM');
    expect(potency(6)).toBe('1.0 µM');
    expect(potency(3)).toBe('1.0 mM');
  });

  it('handles sub-nanomolar potency', () => {
    expect(potency(10)).toBe('100 pM');
  });

  it('returns an em dash when there is no measurement', () => {
    expect(potency(null)).toBe('—');
  });
});

describe('clinicalPhase', () => {
  it('maps ChEMBL phases to their meaning', () => {
    expect(clinicalPhase(4)).toBe('Approved');
    expect(clinicalPhase(1)).toBe('Phase 1');
    expect(clinicalPhase(0)).toBe('Preclinical');
  });

  it('says unknown rather than guessing', () => {
    expect(clinicalPhase(null)).toBe('Unknown');
  });
});

describe('confidenceStatus', () => {
  it('bands confidence and reports when it is absent', () => {
    expect(confidenceStatus(0.9).role).toBe('good');
    expect(confidenceStatus(0.5).role).toBe('warning');
    expect(confidenceStatus(0.2).role).toBe('critical');
    // An absent confidence must not render as low confidence.
    expect(confidenceStatus(null)).toEqual({ role: 'unknown', label: 'not reported' });
  });
});

describe('misc formatters', () => {
  it('formats durations across magnitudes', () => {
    expect(duration(450)).toBe('450 ms');
    expect(duration(2500)).toBe('2.5 s');
    expect(duration(125_000)).toBe('2m 5s');
    expect(duration(null)).toBe('—');
  });

  it('formats cost with enough precision to be useful', () => {
    expect(cost(0)).toBe('$0.00');
    expect(cost(0.0034)).toBe('$0.0034');
    expect(cost(1.5)).toBe('$1.50');
  });

  it('formats percentages and dates', () => {
    expect(pct(0.8342)).toBe('83.4%');
    expect(isoDate('2026-06-01T12:00:00Z')).toBe('2026-06-01');
    expect(isoDate(null)).toBe('—');
    expect(isoDate('not a date')).toBe('—');
  });

  it('renders relative time', () => {
    expect(relativeTime(new Date().toISOString())).toBe('just now');
    expect(relativeTime(null)).toBe('—');
  });

  it('titlecases snake_case and truncates', () => {
    expect(titleCase('genetic_association')).toBe('Genetic Association');
    expect(truncate('abcdefghij', 5)).toBe('abcd…');
    expect(truncate('abc', 5)).toBe('abc');
  });

  it('labels data sources with their published names', () => {
    expect(sourceLabel('open_targets')).toBe('Open Targets');
    expect(sourceLabel('chembl')).toBe('ChEMBL');
    // An unknown source shows its id rather than being hidden.
    expect(sourceLabel('some_new_source')).toBe('some_new_source');
  });
});
