import { describe, expect, it } from 'vitest';
import { BASIS_COLOR, BASIS_LABEL, FAMILY_COLOR, NODE_COLOR, SERIES, seriesColor } from '@/lib/palette';

describe('palette', () => {
  it('assigns categorical hues in fixed order and never generates one', () => {
    // A ninth series must fold back into the validated set, not invent a colour.
    expect(seriesColor(0)).toBe(SERIES[0]);
    expect(seriesColor(8)).toBe(SERIES[0]);
    expect(new Set(SERIES).size).toBe(SERIES.length);
  });

  it('keeps measured and predicted visually distinct', () => {
    // The product's central claim depends on these never colliding.
    expect(BASIS_COLOR.measured).not.toBe(BASIS_COLOR.predicted);
    expect(BASIS_LABEL.predicted).toContain('prediction');
    expect(BASIS_LABEL.measured).toContain('Experimental');
  });

  it('avoids the yellow/orange pair in the basis palette', () => {
    // That pair fails the colour-vision separation gate when both are on screen.
    const used = Object.values(BASIS_COLOR);
    expect(used).not.toContain('var(--series-4)');
  });

  it('gives every node kind and model family a colour', () => {
    for (const kind of ['disease', 'target', 'evidence', 'compound', 'assay', 'prediction']) {
      expect(NODE_COLOR[kind]).toBeTruthy();
    }
    for (const family of ['random_forest', 'gin']) {
      expect(FAMILY_COLOR[family]).toBeTruthy();
    }
  });

  it('colours predictions differently from every retrieved node kind', () => {
    const retrieved = Object.entries(NODE_COLOR)
      .filter(([kind]) => kind !== 'prediction')
      .map(([, color]) => color);
    expect(retrieved).not.toContain(NODE_COLOR.prediction);
  });
});
