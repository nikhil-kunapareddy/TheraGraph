/**
 * API client.
 *
 * One place that knows how to talk to the backend, so components never build URLs or
 * parse error envelopes. Three behaviours worth noting:
 *
 * - **Typed errors.** The backend's error envelope is unwrapped into an `ApiClientError`
 *   carrying the stable `code`, so UI can branch on `not_found` versus `upstream_error`
 *   rather than matching message text.
 * - **Request ids surface to the user.** Every response's `X-Request-ID` is attached to
 *   the error, so a bug report can be traced to a log line.
 * - **Abort support.** Long queries (similarity over the whole corpus, agent runs) are
 *   cancelled when the view unmounts.
 */

import type {
  AgentTool,
  ComparisonResponse,
  CompoundOverview,
  CorpusSummary,
  DescriptorValues,
  DiseaseOverview,
  EvidenceGraphResponse,
  Experiment,
  MlModel,
  ModelEvaluation,
  PredictionResponse,
  RankingConfig,
  RankingResponse,
  Report,
  RunAccepted,
  RunDetail,
  RunListResponse,
  RunUsage,
  SearchResponse,
  SimilarityResponse,
  TargetOverview,
} from '@/types/api';

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api/v1';

export class ApiClientError extends Error {
  readonly code: string;
  readonly status: number;
  readonly retryable: boolean;
  readonly requestId?: string;
  readonly details?: Record<string, unknown>;

  constructor(init: {
    message: string;
    code: string;
    status: number;
    retryable?: boolean;
    requestId?: string;
    details?: Record<string, unknown>;
  }) {
    super(init.message);
    this.name = 'ApiClientError';
    this.code = init.code;
    this.status = init.status;
    this.retryable = init.retryable ?? false;
    this.requestId = init.requestId;
    this.details = init.details;
  }

  get isNotFound(): boolean {
    return this.code === 'not_found';
  }

  /** Guidance shown next to the error, chosen by code rather than by status alone. */
  get hint(): string | undefined {
    switch (this.code) {
      case 'not_found':
        return 'This entity is not in the ingested corpus. Search with upstream discovery enabled, then ingest it.';
      case 'model_unavailable':
        return 'No model is registered for this endpoint. Train one with `theragraph ml train`.';
      case 'invalid_chemistry':
        return 'The structure could not be parsed. Check the SMILES string.';
      case 'rate_limited':
        return 'Too many requests. Agent endpoints have a tighter limit because they spend model tokens.';
      case 'live_ingestion_disabled':
        return 'This deployment cannot call upstream APIs; it serves the pre-ingested corpus only.';
      case 'upstream_error':
      case 'upstream_unavailable':
        return 'A public data source failed. The request can be retried.';
      default:
        return undefined;
    }
  }
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'DELETE';
  body?: unknown;
  signal?: AbortSignal;
  query?: Record<string, string | number | boolean | string[] | undefined | null>;
}

function buildUrl(path: string, query?: RequestOptions['query']): string {
  const url = `${API_BASE}${path}`;
  if (!query) return url;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === '') continue;
    if (Array.isArray(value)) {
      value.forEach((item) => params.append(key, String(item)));
    } else {
      params.set(key, String(value));
    }
  }
  const qs = params.toString();
  return qs ? `${url}?${qs}` : url;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, query } = options;
  const response = await fetch(buildUrl(path, query), {
    method,
    signal,
    headers: {
      Accept: 'application/json',
      ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });

  const requestId = response.headers.get('X-Request-ID') ?? undefined;

  if (!response.ok) {
    let code = `http_${response.status}`;
    let message = response.statusText || 'request failed';
    let retryable = response.status >= 500 || response.status === 429;
    let details: Record<string, unknown> | undefined;
    try {
      const payload = (await response.json()) as { error?: Record<string, unknown> };
      if (payload.error) {
        code = String(payload.error.code ?? code);
        message = String(payload.error.message ?? message);
        retryable = Boolean(payload.error.retryable ?? retryable);
        details = payload.error.details as Record<string, unknown> | undefined;
      }
    } catch {
      // A non-JSON error body (a proxy or gateway page); keep the status-derived values.
    }
    throw new ApiClientError({
      message,
      code,
      status: response.status,
      retryable,
      requestId,
      details,
    });
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  // ------------------------------------------------------------------ search
  search: (
    q: string,
    opts: {
      entityTypes?: string[];
      limit?: number;
      includeUpstream?: boolean;
      signal?: AbortSignal;
    } = {},
  ) =>
    request<SearchResponse>('/search', {
      query: {
        q,
        entity_types: opts.entityTypes,
        limit: opts.limit ?? 10,
        include_upstream: opts.includeUpstream ?? false,
      },
      signal: opts.signal,
    }),

  corpus: (signal?: AbortSignal) => request<CorpusSummary>('/search/corpus', { signal }),

  // ---------------------------------------------------------------- entities
  disease: (identifier: string, opts: { nTargets?: number; minScore?: number } = {}) =>
    request<DiseaseOverview>(`/diseases/${encodeURIComponent(identifier)}`, {
      query: { n_targets: opts.nTargets ?? 25, min_score: opts.minScore ?? 0 },
    }),

  target: (
    identifier: string,
    opts: {
      disease?: string;
      evidenceLimit?: number;
      compoundLimit?: number;
      minPchembl?: number;
    } = {},
  ) =>
    request<TargetOverview>(`/targets/${encodeURIComponent(identifier)}`, {
      query: {
        disease: opts.disease,
        evidence_limit: opts.evidenceLimit ?? 50,
        compound_limit: opts.compoundLimit ?? 25,
        min_pchembl: opts.minPchembl ?? 6,
      },
    }),

  targetEvidence: (
    identifier: string,
    opts: { disease?: string; evidenceTypes?: string[]; minScore?: number; limit?: number } = {},
  ) =>
    request<{ items: TargetOverview['evidence']['groups'][number]['records']; count: number }>(
      `/targets/${encodeURIComponent(identifier)}/evidence`,
      {
        query: {
          disease: opts.disease,
          evidence_types: opts.evidenceTypes,
          min_score: opts.minScore,
          limit: opts.limit ?? 50,
        },
      },
    ),

  compound: (
    identifier: string,
    opts: {
      includeSimilar?: boolean;
      includeBioactivity?: boolean;
      similarityThreshold?: number;
    } = {},
  ) =>
    request<CompoundOverview>(`/compounds/${encodeURIComponent(identifier)}`, {
      query: {
        include_similar: opts.includeSimilar ?? true,
        include_bioactivity: opts.includeBioactivity ?? true,
        similarity_threshold: opts.similarityThreshold ?? 0.7,
      },
    }),

  descriptorDictionary: () =>
    request<{
      descriptors: DescriptorValues['dictionary'];
      fingerprints: Array<Record<string, unknown>>;
    }>('/compounds/descriptors'),

  computeProperties: (smiles: string) =>
    request<{
      canonical_smiles: string;
      inchikey: string;
      formula: string;
      murcko_scaffold?: string | null;
      standardization_notes: string[];
      properties: DescriptorValues;
    }>('/compounds/properties', { method: 'POST', body: { smiles } }),

  similarity: (body: { smiles: string; threshold?: number; limit?: number }) =>
    request<SimilarityResponse>('/compounds/similarity', { method: 'POST', body }),

  substructure: (body: { smarts: string; limit?: number }) =>
    request<{
      smarts: string;
      matches: Array<Record<string, unknown>>;
      count: number;
      search_stats: Record<string, number>;
    }>('/compounds/substructure', { method: 'POST', body }),

  // --------------------------------------------------------------- inference
  predictionEndpoints: () =>
    request<{
      endpoints: Array<{
        endpoint: string;
        task_type: string;
        units?: string | null;
        direction?: string | null;
        models: Array<{ ref: string; family: string; version: string }>;
      }>;
    }>('/predictions/endpoints'),

  predict: (body: {
    compound_ids?: string[];
    smiles?: string[];
    endpoints?: string[];
    persist?: boolean;
  }) => request<PredictionResponse>('/predictions', { method: 'POST', body }),

  // ----------------------------------------------------------------- ranking
  rankingPresets: () =>
    request<{
      presets: Array<{ name: string; description: string; n_objectives: number }>;
      configs: Record<string, RankingConfig>;
    }>('/ranking/presets'),

  prioritize: (body: {
    target: string;
    disease?: string;
    compound_ids?: string[];
    config_name?: string;
    weights?: Record<string, number>;
    min_pchembl?: number;
    limit?: number;
    include_similar_to?: string;
  }) => request<RankingResponse>('/ranking', { method: 'POST', body }),

  compare: (body: {
    compound_ids: string[];
    target?: string;
    disease?: string;
    config_name?: string;
  }) => request<ComparisonResponse>('/ranking/compare', { method: 'POST', body }),

  ranking: (rankingId: string) => request<Record<string, unknown>>(`/ranking/${rankingId}`),

  // ------------------------------------------------------------------- graph
  graph: (params: { disease?: string; target?: string; compound?: string; maxTargets?: number }) =>
    request<EvidenceGraphResponse>('/graph', {
      query: {
        disease: params.disease,
        target: params.target,
        compound: params.compound,
        max_targets: params.maxTargets ?? 8,
      },
    }),

  // ------------------------------------------------------------------- agent
  agentTools: () => request<{ tools: AgentTool[]; guarantees: string[] }>('/agent/tools'),

  research: (body: {
    query: string;
    parameters?: Record<string, unknown>;
    execute_inline?: boolean;
    idempotency_key?: string;
  }) => request<RunAccepted>('/agent/research', { method: 'POST', body }),

  // -------------------------------------------------------------------- runs
  runs: (opts: { status?: string; q?: string; limit?: number; offset?: number } = {}) =>
    request<RunListResponse>('/runs', {
      query: { status: opts.status, q: opts.q, limit: opts.limit ?? 25, offset: opts.offset ?? 0 },
    }),

  run: (runId: string) => request<RunDetail>(`/runs/${encodeURIComponent(runId)}`),

  report: (runId: string) => request<Report>(`/runs/${encodeURIComponent(runId)}/report`),

  runUsage: () => request<RunUsage>('/runs/usage'),

  replayPlan: (runId: string, allowDrift = false) =>
    request<{
      original_run_id: string;
      query: string;
      parameters: Record<string, unknown>;
      seed: number;
      model_pins: Record<string, string>;
      drift: {
        dataset_versions_changed: string[];
        model_versions_changed: string[];
        reproducible: boolean;
      };
    }>(`/runs/${encodeURIComponent(runId)}/replay-plan`, {
      query: { allow_version_drift: allowDrift },
    }),

  replay: (runId: string, body: { allow_version_drift?: boolean; execute_inline?: boolean } = {}) =>
    request<RunAccepted & { replay_of: string; drift: Record<string, unknown> }>(
      `/runs/${encodeURIComponent(runId)}/replay`,
      { method: 'POST', body },
    ),

  compareRuns: (runIds: string[]) =>
    request<{
      runs: Array<Record<string, unknown>>;
      differences: Array<Record<string, unknown>>;
      metrics: Record<string, Record<string, number | null>>;
    }>('/runs/compare', { method: 'POST', body: { run_ids: runIds } }),

  // ------------------------------------------------------------------ models
  models: (opts: { endpoint?: string; status?: string } = {}) =>
    request<{ models: MlModel[]; count: number; reporting_note: string }>('/models', {
      query: { endpoint: opts.endpoint, status: opts.status },
    }),

  model: (modelId: string) => request<MlModel>(`/models/${modelId}`),

  evaluations: (opts: { modelId?: string; endpoint?: string } = {}) =>
    request<{ evaluations: ModelEvaluation[]; count: number }>('/evaluations', {
      query: { model_id: opts.modelId, endpoint: opts.endpoint },
    }),

  experiments: (opts: { endpoint?: string; limit?: number } = {}) =>
    request<{ experiments: Experiment[]; count: number }>('/experiments', {
      query: { endpoint: opts.endpoint, limit: opts.limit ?? 25 },
    }),

  // --------------------------------------------------------------- ingestion
  sources: (currentOnly = false) =>
    request<{
      sources: Array<{
        id: string;
        name: string;
        license?: string | null;
        citation?: string | null;
        homepage_url?: string | null;
      }>;
      dataset_versions: Array<Record<string, unknown>>;
    }>('/ingestion/sources', { query: { current_only: currentOnly } }),

  ingestDisease: (body: { efo_id: string; n_targets?: number }) =>
    request<{ job_id: string; task: string; status: string; poll: string }>('/ingestion/disease', {
      method: 'POST',
      body,
    }),

  ingestBioactivity: (body: { target: string; max_activities?: number }) =>
    request<{ job_id: string; task: string; status: string; poll: string }>(
      '/ingestion/bioactivity',
      {
        method: 'POST',
        body,
      },
    ),

  job: (jobId: string) => request<Record<string, unknown>>(`/jobs/${jobId}`),

  health: () =>
    fetch('/healthz').then(
      (response) => response.json() as Promise<{ status: string; version: string }>,
    ),
};

export { API_BASE };
