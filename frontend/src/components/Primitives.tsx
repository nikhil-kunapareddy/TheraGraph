/** Small presentational building blocks shared across views. */

import clsx from 'clsx';
import type { ReactNode } from 'react';

export function Card({
  title,
  subtitle,
  actions,
  children,
  className,
  dense,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  dense?: boolean;
}) {
  return (
    <section
      className={clsx(
        'rounded-lg border border-surface-border bg-surface',
        dense ? 'p-3' : 'p-4',
        className,
      )}
    >
      {(title || actions) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

/**
 * A headline number. Used instead of a chart when there is exactly one value to
 * communicate -- a one-bar bar chart is never the right answer.
 */
export function StatTile({
  label,
  value,
  unit,
  hint,
  tone = 'neutral',
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: string;
  tone?: 'neutral' | 'good' | 'warning' | 'critical';
}) {
  const toneClass = {
    neutral: 'text-ink',
    good: 'text-[color:var(--status-good)]',
    warning: 'text-[color:var(--status-warning)]',
    critical: 'text-[color:var(--status-critical)]',
  }[tone];
  return (
    <div className="rounded-lg border border-surface-border bg-surface px-4 py-3">
      <div className="text-2xs uppercase tracking-wide text-ink-muted">{label}</div>
      <div className={clsx('mt-1 font-mono text-xl tabular-nums', toneClass)}>
        {value}
        {unit && <span className="ml-1 text-xs text-ink-muted">{unit}</span>}
      </div>
      {hint && <div className="mt-1 text-2xs leading-snug text-ink-muted">{hint}</div>}
    </div>
  );
}

export function Badge({
  children,
  tone = 'neutral',
  title,
  icon,
}: {
  children: ReactNode;
  tone?: 'neutral' | 'measured' | 'predicted' | 'good' | 'warning' | 'critical' | 'info';
  title?: string;
  icon?: ReactNode;
}) {
  const tones: Record<string, string> = {
    neutral: 'border-surface-border bg-surface-raised text-ink-secondary',
    measured:
      'border-[color:var(--series-1)]/40 bg-[color:var(--series-1)]/10 text-[color:var(--series-1)]',
    predicted:
      'border-[color:var(--series-2)]/40 bg-[color:var(--series-2)]/10 text-[color:var(--series-2)]',
    good: 'border-[color:var(--status-good)]/40 bg-[color:var(--status-good)]/10 text-[color:var(--status-good)]',
    warning:
      'border-[color:var(--status-warning)]/40 bg-[color:var(--status-warning)]/10 text-[color:var(--status-warning)]',
    critical:
      'border-[color:var(--status-critical)]/40 bg-[color:var(--status-critical)]/10 text-[color:var(--status-critical)]',
    info: 'border-[color:var(--series-7)]/40 bg-[color:var(--series-7)]/10 text-[color:var(--series-7)]',
  };
  return (
    <span
      title={title}
      className={clsx(
        'inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-2xs font-medium',
        tones[tone],
      )}
    >
      {icon}
      {children}
    </span>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div
      className="flex items-center gap-2 text-xs text-ink-muted"
      role="status"
      aria-live="polite"
    >
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-surface-border border-t-[color:var(--series-1)]" />
      {label ?? 'Loading…'}
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail?: string;
  action?: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-dashed border-surface-border bg-surface/50 px-4 py-6 text-center">
      <p className="text-sm text-ink-secondary">{title}</p>
      {detail && (
        <p className="mx-auto mt-1 max-w-lg text-xs leading-relaxed text-ink-muted">{detail}</p>
      )}
      {action && <div className="mt-3 flex justify-center">{action}</div>}
    </div>
  );
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }) {
  const message = error instanceof Error ? error.message : 'Something went wrong.';
  const hint =
    error && typeof error === 'object' && 'hint' in error
      ? (error as { hint?: string }).hint
      : undefined;
  const requestId =
    error && typeof error === 'object' && 'requestId' in error
      ? (error as { requestId?: string }).requestId
      : undefined;
  return (
    <div
      role="alert"
      className="rounded-lg border border-[color:var(--status-critical)]/40 bg-[color:var(--status-critical)]/5 px-4 py-3"
    >
      <p className="flex items-center gap-2 text-sm text-[color:var(--status-critical)]">
        <span aria-hidden>⚠</span>
        {message}
      </p>
      {hint && <p className="mt-1 text-xs text-ink-secondary">{hint}</p>}
      {requestId && <p className="mt-1 font-mono text-2xs text-ink-muted">request {requestId}</p>}
      {retry && (
        <button
          type="button"
          onClick={retry}
          className="mt-2 rounded border border-surface-border px-2 py-1 text-xs text-ink-secondary hover:bg-surface-raised"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = 'secondary',
  disabled,
  type = 'button',
  className,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  disabled?: boolean;
  type?: 'button' | 'submit';
  className?: string;
  title?: string;
}) {
  const variants = {
    primary:
      'bg-[color:var(--series-1)] text-white hover:brightness-110 disabled:bg-surface-raised disabled:text-ink-muted',
    secondary:
      'border border-surface-border bg-surface-raised text-ink-secondary hover:text-ink hover:border-ink-muted',
    ghost: 'text-ink-muted hover:text-ink',
  };
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'rounded px-2.5 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-60',
        variants[variant],
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Table({
  headers,
  children,
  caption,
}: {
  headers: ReactNode[];
  children: ReactNode;
  caption?: string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        {caption && <caption className="pb-2 text-left text-2xs text-ink-muted">{caption}</caption>}
        <thead>
          <tr className="border-b border-surface-border text-left text-2xs uppercase tracking-wide text-ink-muted">
            {headers.map((header, index) => (
              <th key={index} className="whitespace-nowrap px-2 py-2 font-medium">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-surface-border/60">{children}</tbody>
      </table>
    </div>
  );
}

/** A labelled definition row, for entity metadata panels. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-2 py-1 text-xs">
      <dt className="w-32 shrink-0 text-ink-muted">{label}</dt>
      <dd className="min-w-0 flex-1 break-words text-ink-secondary">{children}</dd>
    </div>
  );
}
