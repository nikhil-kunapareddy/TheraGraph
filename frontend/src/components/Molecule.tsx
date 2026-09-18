/**
 * Structure rendering.
 *
 * A drug-discovery UI that shows SMILES strings instead of structures is unusable for
 * its audience: a chemist reads a picture in a second and a SMILES string in twenty.
 * Rendering is client-side via smiles-drawer, so no server round-trip per structure
 * and no image cache to invalidate.
 */

import { useEffect, useId, useRef, useState } from 'react';
import clsx from 'clsx';

interface MoleculeProps {
  smiles: string;
  width?: number;
  height?: number;
  className?: string;
  title?: string;
}

const THEME = {
  // Tuned for the dark surface: bonds in light ink, heteroatoms in the categorical hues
  // so element identity does not rely on a legend.
  C: '#c3c2b7',
  O: '#e66767',
  N: '#3987e5',
  F: '#199e70',
  CL: '#008300',
  BR: '#d95926',
  I: '#9085e9',
  P: '#c98500',
  S: '#c98500',
  B: '#d55181',
  SI: '#8a897e',
  H: '#8a897e',
  BACKGROUND: '#1a1a19',
};

export function Molecule({ smiles, width = 260, height = 180, className, title }: MoleculeProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rawId = useId();
  const canvasId = `mol-${rawId.replace(/[:]/g, '')}`;
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setFailed(false);

    async function draw() {
      if (!canvasRef.current || !smiles) return;
      try {
        // Imported lazily so the chemistry renderer is not in the initial bundle.
        const module = await import('smiles-drawer');
        if (cancelled) return;
        const SmilesDrawer = (module as { default?: unknown }).default ?? module;
        const Drawer = SmilesDrawer as {
          SvgDrawer?: unknown;
          Drawer?: unknown;
          parse?: unknown;
        } as {
          Drawer: new (options: Record<string, unknown>) => {
            draw: (tree: unknown, id: string, theme: string) => void;
          };
          parse: (
            smiles: string,
            success: (tree: unknown) => void,
            failure: (error: unknown) => void,
          ) => void;
        };

        const drawer = new Drawer.Drawer({
          width,
          height,
          bondThickness: 1.1,
          bondLength: 16,
          shortBondLength: 0.85,
          atomVisualization: 'default',
          terminalCarbons: false,
          explicitHydrogens: false,
          compactDrawing: false,
          themes: { theragraph: THEME },
        });

        Drawer.parse(
          smiles,
          (tree: unknown) => {
            if (cancelled || !canvasRef.current) return;
            try {
              drawer.draw(tree, canvasId, 'theragraph');
            } catch {
              setFailed(true);
            }
          },
          () => setFailed(true),
        );
      } catch {
        setFailed(true);
      }
    }

    void draw();
    return () => {
      cancelled = true;
    };
  }, [smiles, width, height, canvasId]);

  if (failed) {
    return (
      <div
        className={clsx(
          'flex flex-col items-center justify-center gap-1 rounded border border-dashed border-surface-border bg-surface-raised p-3 text-center',
          className,
        )}
        style={{ width, height }}
      >
        <span className="text-2xs text-ink-muted">structure could not be rendered</span>
        <code className="max-w-full break-all font-mono text-2xs text-ink-muted">{smiles}</code>
      </div>
    );
  }

  return (
    <canvas
      id={canvasId}
      ref={canvasRef}
      width={width}
      height={height}
      className={clsx('rounded bg-surface', className)}
      role="img"
      aria-label={title ?? `Chemical structure: ${smiles}`}
      title={title ?? smiles}
    />
  );
}

/** Structure plus the SMILES string, with copy-to-clipboard. */
export function MoleculeCard({
  smiles,
  name,
  subtitle,
  width = 240,
  height = 170,
}: {
  smiles: string;
  name?: string | null;
  subtitle?: string | null;
  width?: number;
  height?: number;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex flex-col gap-2">
      <Molecule smiles={smiles} width={width} height={height} title={name ?? smiles} />
      <div className="min-w-0">
        {name && <p className="truncate text-xs font-medium text-ink">{name}</p>}
        {subtitle && <p className="truncate text-2xs text-ink-muted">{subtitle}</p>}
        <button
          type="button"
          onClick={() => {
            void navigator.clipboard?.writeText(smiles);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
          }}
          className="mt-1 block max-w-full truncate text-left font-mono text-2xs text-ink-muted hover:text-ink-secondary"
          title="Copy SMILES"
        >
          {copied ? 'copied ✓' : smiles}
        </button>
      </div>
    </div>
  );
}
