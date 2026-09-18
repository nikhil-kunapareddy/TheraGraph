import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import clsx from 'clsx';
import { api } from './lib/api';
import Dashboard from './pages/Dashboard';
import TargetExplorer from './pages/TargetExplorer';
import CompoundExplorer from './pages/CompoundExplorer';
import CandidateComparison from './pages/CandidateComparison';
import Assistant from './pages/Assistant';
import ResearchRun from './pages/ResearchRun';
import ModelEvaluation from './pages/ModelEvaluation';

const NAV = [
  { to: '/', label: 'Dashboard', hint: 'Search and corpus overview' },
  { to: '/targets', label: 'Targets', hint: 'Evidence, tractability, chemistry' },
  { to: '/compounds', label: 'Compounds', hint: 'Structures, properties, similarity' },
  { to: '/compare', label: 'Compare', hint: 'Side-by-side candidate comparison' },
  { to: '/assistant', label: 'Assistant', hint: 'Tool-calling research agent' },
  { to: '/runs', label: 'Runs', hint: 'Reproducible execution traces' },
  { to: '/models', label: 'Models', hint: 'Scaffold-split metrics and calibration' },
];

export default function App() {
  const location = useLocation();
  const { data: corpus } = useQuery({ queryKey: ['corpus'], queryFn: () => api.corpus() });

  return (
    <div className="min-h-screen bg-surface-sunken">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-surface-raised focus:px-3 focus:py-2 focus:text-xs focus:text-ink"
      >
        Skip to content
      </a>

      <header className="sticky top-0 z-30 border-b border-surface-border bg-surface-sunken/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5">
          <NavLink to="/" className="flex items-center gap-2">
            <span
              aria-hidden
              className="grid h-7 w-7 place-items-center rounded bg-[color:var(--series-1)] font-mono text-sm font-bold text-white"
            >
              T
            </span>
            <span className="text-sm font-semibold tracking-tight text-ink">TheraGraph</span>
          </NavLink>

          <nav aria-label="Main" className="flex flex-1 flex-wrap items-center gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                title={item.hint}
                end={item.to === '/'}
                className={({ isActive }) =>
                  clsx(
                    'rounded px-2.5 py-1.5 text-xs font-medium transition',
                    isActive || (item.to !== '/' && location.pathname.startsWith(item.to))
                      ? 'bg-surface-raised text-ink'
                      : 'text-ink-muted hover:text-ink-secondary',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          {corpus && (
            <p
              className="text-2xs text-ink-muted"
              title="Entities currently in the ingested corpus"
            >
              {corpus.counts.diseases} diseases · {corpus.counts.targets} targets ·{' '}
              {corpus.counts.compounds.toLocaleString()} compounds ·{' '}
              {corpus.counts.bioactivities.toLocaleString()} measured activities
            </p>
          )}
        </div>
      </header>

      <main id="main" className="mx-auto max-w-[1600px] px-4 py-5">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/targets" element={<TargetExplorer />} />
          <Route path="/targets/:identifier" element={<TargetExplorer />} />
          <Route path="/compounds" element={<CompoundExplorer />} />
          <Route path="/compounds/:identifier" element={<CompoundExplorer />} />
          <Route path="/compare" element={<CandidateComparison />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="/runs" element={<ResearchRun />} />
          <Route path="/runs/:runId" element={<ResearchRun />} />
          <Route path="/models" element={<ModelEvaluation />} />
          <Route path="/models/:modelId" element={<ModelEvaluation />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className="border-t border-surface-border px-4 py-4">
        <div className="mx-auto flex max-w-[1600px] flex-col gap-1 text-2xs text-ink-muted">
          <p>
            Retrieved facts and model predictions are reported separately throughout. TheraGraph
            does not discover or validate drugs; it aggregates public evidence and model output for
            human judgement.
          </p>
          {corpus && corpus.sources.length > 0 && (
            <p>
              Sources:{' '}
              {corpus.sources.map((source, index) => (
                <span key={source.id}>
                  {index > 0 && ' · '}
                  <a
                    href={source.homepage_url ?? '#'}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="hover:text-ink-secondary hover:underline"
                    title={source.license ? `Licence: ${source.license}` : undefined}
                  >
                    {source.name}
                  </a>
                </span>
              ))}
            </p>
          )}
        </div>
      </footer>
    </div>
  );
}
