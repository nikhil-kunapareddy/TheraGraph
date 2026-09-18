import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BasisBadge, ProvenanceChip, SourceScore } from '@/components/Provenance';
import { EmptyState, ErrorState, StatTile } from '@/components/Primitives';
import { ApiClientError } from '@/lib/api';

describe('BasisBadge', () => {
  it('labels a model output as a prediction', () => {
    // The single most important visual distinction in the product.
    render(<BasisBadge basis="predicted" />);
    const badge = screen.getByTitle(/Model prediction, not a measurement/);
    expect(badge).toHaveTextContent('Prediction');
  });

  it('labels an experimental value as measured', () => {
    render(<BasisBadge basis="measured" />);
    expect(screen.getByTitle(/Experimental measurement/)).toHaveTextContent('Measured');
  });

  it('distinguishes computed from derived', () => {
    const { container } = render(
      <>
        <BasisBadge basis="computed" />
        <BasisBadge basis="derived" />
      </>,
    );
    expect(container.textContent).toContain('Computed');
    expect(container.textContent).toContain('Derived');
  });
});

describe('SourceScore', () => {
  it('says the source provided no score rather than showing zero', () => {
    render(<SourceScore score={null} provided={false} />);
    expect(screen.getByText('not provided by source')).toBeInTheDocument();
  });

  it('renders a provided score at full precision', () => {
    render(<SourceScore score={0.8722} scoreType="open_targets_association_score" provided />);
    expect(screen.getByText('0.872')).toBeInTheDocument();
  });
});

describe('ProvenanceChip', () => {
  it('names the source and its retrieval date', () => {
    render(
      <ProvenanceChip
        provenance={{
          source: 'open_targets',
          source_identifier: 'MONDO_0004975',
          source_url: 'https://platform.opentargets.org/disease/MONDO_0004975',
          retrieved_at: '2026-06-01T00:00:00Z',
        }}
      />,
    );
    expect(screen.getByText('Open Targets')).toBeInTheDocument();
    expect(screen.getByText(/2026-06-01/)).toBeInTheDocument();
    expect(screen.getByRole('link')).toHaveAttribute(
      'href',
      'https://platform.opentargets.org/disease/MONDO_0004975',
    );
  });

  it('renders without a link when the source published no URL', () => {
    render(
      <ProvenanceChip
        provenance={{ source: 'chembl', retrieved_at: '2026-06-01T00:00:00Z' }}
      />,
    );
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText('ChEMBL')).toBeInTheDocument();
  });
});

describe('ErrorState', () => {
  it('surfaces the hint and request id so a bug report is traceable', () => {
    const error = new ApiClientError({
      message: 'no model registered',
      code: 'model_unavailable',
      status: 503,
      requestId: 'req_xyz',
    });
    render(<ErrorState error={error} />);
    expect(screen.getByRole('alert')).toHaveTextContent('no model registered');
    expect(screen.getByText(/theragraph ml train/)).toBeInTheDocument();
    expect(screen.getByText(/req_xyz/)).toBeInTheDocument();
  });
});

describe('StatTile and EmptyState', () => {
  it('shows a headline number with its unit and caveat', () => {
    render(<StatTile label="Best potency" value="10.0 nM" hint="pChEMBL 8.00" />);
    expect(screen.getByText('Best potency')).toBeInTheDocument();
    expect(screen.getByText('10.0 nM')).toBeInTheDocument();
    expect(screen.getByText('pChEMBL 8.00')).toBeInTheDocument();
  });

  it('explains what an empty result means', () => {
    render(
      <EmptyState
        title="No compounds above this potency threshold in the corpus."
        detail="This reflects the ingested corpus, not the absence of evidence in the literature."
      />,
    );
    expect(screen.getByText(/not the absence of evidence/)).toBeInTheDocument();
  });
});
