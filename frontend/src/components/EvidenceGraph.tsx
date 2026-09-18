/**
 * The evidence graph.
 *
 * A hand-rolled SVG force layout rather than a graph library: the graphs here are small
 * (tens of nodes), the layout requirements are specific (entity kinds should sit in
 * bands, predictions must be visually separated from evidence), and a general-purpose
 * library would add ~150 KB for behaviour we would then fight.
 *
 * The layout is deterministic -- seeded initial positions, fixed iteration count -- so
 * the same graph renders identically every time. A graph that reshuffles on every render
 * is unusable for comparing two runs.
 */

import { useMemo, useState } from 'react';
import clsx from 'clsx';
import type { EvidenceGraphResponse, GraphEdge, GraphNode } from '@/types/api';
import { NODE_COLOR } from '@/lib/palette';
import { isoDate, sourceLabel, titleCase } from '@/lib/format';

interface Positioned extends GraphNode {
  x: number;
  y: number;
}

/** Column order: the scientific narrative reads left to right. */
const BANDS: Record<string, number> = {
  disease: 0,
  target: 1,
  gene: 1,
  evidence: 2,
  publication: 3,
  compound: 3,
  assay: 4,
  prediction: 4,
};

const WIDTH = 900;
const HEIGHT = 520;
const ITERATIONS = 90;

function layout(nodes: GraphNode[], edges: GraphEdge[]): Positioned[] {
  if (nodes.length === 0) return [];

  const bandCount = Math.max(...nodes.map((node) => BANDS[node.kind] ?? 2)) + 1;
  const bandWidth = WIDTH / bandCount;
  const byBand = new Map<number, GraphNode[]>();
  for (const node of nodes) {
    const band = BANDS[node.kind] ?? 2;
    byBand.set(band, [...(byBand.get(band) ?? []), node]);
  }

  // Seeded initial placement: banded horizontally, evenly spread vertically.
  const positions = new Map<string, { x: number; y: number }>();
  for (const [band, bandNodes] of byBand) {
    bandNodes.forEach((node, index) => {
      const step = HEIGHT / (bandNodes.length + 1);
      positions.set(node.id, {
        x: bandWidth * band + bandWidth / 2,
        y: step * (index + 1),
      });
    });
  }

  // A few relaxation passes: edges pull connected nodes together vertically, same-band
  // nodes push apart. Horizontal position stays pinned to the band, which keeps the
  // left-to-right reading order intact.
  const adjacency = new Map<string, string[]>();
  for (const edge of edges) {
    adjacency.set(edge.source, [...(adjacency.get(edge.source) ?? []), edge.target]);
    adjacency.set(edge.target, [...(adjacency.get(edge.target) ?? []), edge.source]);
  }

  for (let iteration = 0; iteration < ITERATIONS; iteration += 1) {
    const cooling = 1 - iteration / ITERATIONS;
    for (const node of nodes) {
      const current = positions.get(node.id);
      if (!current) continue;
      let force = 0;

      for (const neighbourId of adjacency.get(node.id) ?? []) {
        const neighbour = positions.get(neighbourId);
        if (neighbour) force += (neighbour.y - current.y) * 0.08;
      }

      const band = BANDS[node.kind] ?? 2;
      for (const other of byBand.get(band) ?? []) {
        if (other.id === node.id) continue;
        const otherPosition = positions.get(other.id);
        if (!otherPosition) continue;
        const delta = current.y - otherPosition.y;
        const distance = Math.abs(delta) || 1;
        if (distance < 52) force += (delta / distance) * (52 - distance) * 0.35;
      }

      const next = Math.min(HEIGHT - 30, Math.max(30, current.y + force * cooling));
      positions.set(node.id, { x: current.x, y: next });
    }
  }

  return nodes.map((node) => ({
    ...node,
    x: positions.get(node.id)?.x ?? WIDTH / 2,
    y: positions.get(node.id)?.y ?? HEIGHT / 2,
  }));
}

export function EvidenceGraph({
  graph,
  onSelectNode,
}: {
  graph: EvidenceGraphResponse;
  onSelectNode?: (node: GraphNode) => void;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
  const [visibleKinds, setVisibleKinds] = useState<Set<string>>(
    () => new Set(Object.keys(graph.stats.node_counts)),
  );

  const filtered = useMemo(() => {
    const nodes = graph.nodes.filter((node) => visibleKinds.has(node.kind));
    const ids = new Set(nodes.map((node) => node.id));
    const edges = graph.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target));
    return { nodes, edges };
  }, [graph, visibleKinds]);

  const positioned = useMemo(() => layout(filtered.nodes, filtered.edges), [filtered]);
  const byId = useMemo(() => new Map(positioned.map((node) => [node.id, node])), [positioned]);

  const connected = useMemo(() => {
    if (!hovered) return null;
    const ids = new Set<string>([hovered]);
    for (const edge of filtered.edges) {
      if (edge.source === hovered) ids.add(edge.target);
      if (edge.target === hovered) ids.add(edge.source);
    }
    return ids;
  }, [hovered, filtered.edges]);

  if (graph.nodes.length === 0) {
    return (
      <p className="rounded border border-dashed border-surface-border px-4 py-6 text-center text-xs text-ink-muted">
        No graph could be built: nothing matching this root is in the corpus.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {/* Filters sit in one row above the plot. */}
      <div className="flex flex-wrap items-center gap-2">
        {Object.entries(graph.stats.node_counts)
          .sort(([a], [b]) => (BANDS[a] ?? 9) - (BANDS[b] ?? 9))
          .map(([kind, count]) => {
            const active = visibleKinds.has(kind);
            return (
              <button
                key={kind}
                type="button"
                onClick={() => {
                  const next = new Set(visibleKinds);
                  if (active) next.delete(kind);
                  else next.add(kind);
                  setVisibleKinds(next);
                }}
                aria-pressed={active}
                className={clsx(
                  'inline-flex items-center gap-1.5 rounded border px-2 py-1 text-2xs transition',
                  active
                    ? 'border-surface-border bg-surface-raised text-ink'
                    : 'border-transparent text-ink-muted hover:text-ink-secondary',
                )}
              >
                <span
                  aria-hidden
                  className="h-2 w-2 rounded-full"
                  style={{ background: NODE_COLOR[kind] ?? 'var(--text-muted)' }}
                />
                {titleCase(kind)}
                <span className="tabular-nums text-ink-muted">{count}</span>
              </button>
            );
          })}
        <span className="ml-auto text-2xs text-ink-muted">
          {filtered.nodes.length} nodes · {filtered.edges.length} edges
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-surface-border bg-surface">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="h-[520px] w-full min-w-[900px]"
          role="img"
          aria-label={`Evidence graph with ${filtered.nodes.length} nodes and ${filtered.edges.length} edges`}
        >
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--surface-border)" />
            </marker>
          </defs>

          {filtered.edges.map((edge) => {
            const from = byId.get(edge.source);
            const to = byId.get(edge.target);
            if (!from || !to) return null;
            const dim =
              connected !== null && !(connected.has(edge.source) && connected.has(edge.target));
            const isPrediction = edge.relation === 'predicted_property';
            return (
              <g key={edge.id} opacity={dim ? 0.12 : 1}>
                <path
                  d={`M ${from.x} ${from.y} Q ${(from.x + to.x) / 2} ${(from.y + to.y) / 2 - 18} ${to.x} ${to.y}`}
                  fill="none"
                  stroke={isPrediction ? 'var(--series-2)' : 'var(--surface-border)'}
                  strokeWidth={2}
                  strokeDasharray={isPrediction ? '4 3' : undefined}
                  markerEnd="url(#arrow)"
                  className="cursor-pointer"
                  onClick={() => setSelectedEdge(edge)}
                />
              </g>
            );
          })}

          {positioned.map((node) => {
            const dim = connected !== null && !connected.has(node.id);
            const color = NODE_COLOR[node.kind] ?? 'var(--text-muted)';
            const isPrediction = node.nature === 'model_output';
            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                opacity={dim ? 0.2 : 1}
                className="cursor-pointer"
                onMouseEnter={() => setHovered(node.id)}
                onMouseLeave={() => setHovered(null)}
                onClick={() => onSelectNode?.(node)}
              >
                {/* Predictions are squares with a dashed edge; retrieved records are
                    circles. Shape carries the distinction so it survives colour-blindness
                    and greyscale printing. */}
                {isPrediction ? (
                  <rect
                    x={-9}
                    y={-9}
                    width={18}
                    height={18}
                    rx={3}
                    fill={color}
                    stroke="var(--surface-1)"
                    strokeWidth={2}
                    strokeDasharray="3 2"
                  />
                ) : (
                  <circle r={9} fill={color} stroke="var(--surface-1)" strokeWidth={2} />
                )}
                <text
                  x={0}
                  y={-15}
                  textAnchor="middle"
                  className="pointer-events-none"
                  style={{ fill: 'var(--text-primary)', fontSize: 10, fontWeight: 500 }}
                >
                  {node.label.length > 26 ? `${node.label.slice(0, 25)}…` : node.label}
                </text>
                {node.sublabel && (
                  <text
                    x={0}
                    y={22}
                    textAnchor="middle"
                    className="pointer-events-none"
                    style={{ fill: 'var(--text-muted)', fontSize: 8.5 }}
                  >
                    {node.sublabel.length > 30 ? `${node.sublabel.slice(0, 29)}…` : node.sublabel}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="flex flex-wrap items-center gap-4 text-2xs text-ink-muted">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-ink-muted" /> retrieved record
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden
            className="h-2.5 w-2.5 rounded-sm border border-dashed"
            style={{ borderColor: 'var(--series-2)', background: 'var(--series-2)' }}
          />
          model output (prediction)
        </span>
        <span>
          Click an edge for its source and retrieval date; hover a node to isolate its
          neighbourhood.
        </span>
      </div>

      {selectedEdge && (
        <div className="rounded-lg border border-surface-border bg-surface-raised p-3 text-xs">
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-1">
              <p className="font-medium text-ink">{titleCase(selectedEdge.relation)}</p>
              <p className="text-ink-secondary">
                {byId.get(selectedEdge.source)?.label} → {byId.get(selectedEdge.target)?.label}
              </p>
              <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-2xs text-ink-muted">
                <dt>Source</dt>
                <dd className="text-ink-secondary">
                  {selectedEdge.evidence_source ? sourceLabel(selectedEdge.evidence_source) : '—'}
                </dd>
                <dt>Evidence type</dt>
                <dd className="text-ink-secondary">
                  {selectedEdge.evidence_type ? titleCase(selectedEdge.evidence_type) : '—'}
                </dd>
                <dt>Source id</dt>
                <dd className="break-all font-mono text-ink-secondary">
                  {selectedEdge.source_identifier ?? '—'}
                </dd>
                <dt>Score</dt>
                <dd className="text-ink-secondary">
                  {selectedEdge.score_provided_by_source && selectedEdge.score !== null
                    ? `${selectedEdge.score?.toFixed(3)} (${selectedEdge.score_type ?? 'score'})`
                    : 'not provided by source'}
                </dd>
                <dt>Retrieved</dt>
                <dd className="text-ink-secondary">{isoDate(selectedEdge.retrieved_at)}</dd>
              </dl>
              {selectedEdge.source_url && (
                <a
                  href={selectedEdge.source_url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="mt-1 inline-block text-2xs text-[color:var(--series-1)] hover:underline"
                >
                  Open the upstream record ↗
                </a>
              )}
            </div>
            <button
              type="button"
              onClick={() => setSelectedEdge(null)}
              className="text-ink-muted hover:text-ink"
              aria-label="Close edge details"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
