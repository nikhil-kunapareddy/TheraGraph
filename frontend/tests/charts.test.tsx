import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CalibrationChart, FactorBreakdownChart, UncertaintyChart } from '@/components/Charts';

describe('CalibrationChart', () => {
  it('explains how to read the diagram and reports ECE', () => {
    render(
      <CalibrationChart
        calibration={{
          ece: 0.0712,
          brier: 0.095,
          bins: [
            { bin_lower: 0, bin_upper: 0.1, count: 12, mean_predicted: 0.05, observed_frequency: 0.08, gap: 0.03 },
            { bin_lower: 0.9, bin_upper: 1, count: 40, mean_predicted: 0.95, observed_frequency: 0.9, gap: 0.05 },
          ],
        }}
      />,
    );
    expect(screen.getByText(/Marker size is the number of molecules/)).toBeInTheDocument();
    expect(screen.getByText(/0.0712/)).toBeInTheDocument();
  });

  it('says a regression model has no reliability curve rather than rendering an empty axis', () => {
    render(<CalibrationChart calibration={{ bins: [] }} />);
    expect(screen.getByText(/not a classifier/)).toBeInTheDocument();
  });
});

describe('UncertaintyChart', () => {
  it('states that an informative uncertainty can be used for triage', () => {
    render(
      <UncertaintyChart
        deciles={[
          { decile: 1, uncertainty_range: [0, 0.1], count: 10, mean_absolute_error: 0.2 },
          { decile: 10, uncertainty_range: [0.9, 1.2], count: 10, mean_absolute_error: 1.1 },
        ]}
      />,
    );
    expect(screen.getByText(/can be used for triage/)).toBeInTheDocument();
  });

  it('warns when uncertainty does not track error', () => {
    render(
      <UncertaintyChart
        deciles={[
          { decile: 1, uncertainty_range: [0, 0.1], count: 10, mean_absolute_error: 0.5 },
          { decile: 10, uncertainty_range: [0.9, 1.2], count: 10, mean_absolute_error: 0.52 },
        ]}
      />,
    );
    expect(screen.getByText(/should not be used to triage/)).toBeInTheDocument();
  });

  it('reports absence rather than an empty chart', () => {
    render(<UncertaintyChart deciles={[]} />);
    expect(screen.getByText(/No uncertainty analysis/)).toBeInTheDocument();
  });
});

describe('FactorBreakdownChart', () => {
  it('legends every basis present so colour is never the only cue', () => {
    render(
      <FactorBreakdownChart
        candidates={[
          {
            name: 'Compound A',
            total: 0.81,
            factors: [
              { key: 'measured_potency', label: 'Measured potency', basis: 'measured', contribution: 0.24 },
              { key: 'predicted_solubility', label: 'Predicted solubility', basis: 'predicted', contribution: 0.12 },
            ],
          },
        ]}
      />,
    );
    expect(screen.getByText('Experimental measurement')).toBeInTheDocument();
    expect(screen.getByText('Model prediction')).toBeInTheDocument();
  });

  it('offers a table view of the values', () => {
    render(
      <FactorBreakdownChart
        candidates={[
          {
            name: 'Compound A',
            total: 0.5,
            factors: [{ key: 'qed', label: 'Drug-likeness', basis: 'computed', contribution: 0.5 }],
          },
        ]}
      />,
    );
    expect(screen.getByRole('button', { name: /Show table/ })).toBeInTheDocument();
  });

  it('reports an empty ranking plainly', () => {
    render(<FactorBreakdownChart candidates={[]} />);
    expect(screen.getByText('Nothing ranked.')).toBeInTheDocument();
  });
});
