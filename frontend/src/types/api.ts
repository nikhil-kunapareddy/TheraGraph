/**
 * Types for the TheraGraph API.
 *
 * Hand-written rather than generated from OpenAPI: the service payloads are deep
 * scientific documents, and a generated 4000-line type surface would be unreadable and
 * would churn on every backend tweak. These types pin the fields the UI actually
 * consumes and keep the rest as `unknown`, so a missing field is a compile error where
 * it matters and an ignored field elsewhere.
 */

export type EntityType = 'disease' | 'target' | 'compound';

export interface Provenance {
  source: string;
  source_identifier?: string | null;
  source_url?: string | null;
  retrieved_at: string;
  dataset_version?: string | null;
  datasource_id?: string | null;
}

export interface ApiError {
  error: {
    code: string;
    message: string;
    retryable: boolean;
    details?: Record<string, unknown>;
    request_id?: string;
  };
}

export interface SearchHit {
  entity_type: EntityType;
  id: string;
  external_id: string;
  name: string;
  relevance: number;
  description?: string | null;
  source: string;
  availability: 'ingested' | 'discoverable';
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  query: string;
  matched_term?: string | null;
  terms_tried: string[];
  hits: SearchHit[];
  counts: Record<string, number>;
  searched: string;
  upstream?: {
    available: boolean;
    reason?: string;
    source?: string;
    data_version?: string;
    hits?: Array<{
      entity_type: string;
      external_id: string;
      name: string;
      description?: string | null;
      availability: string;
    }>;
  };
}

export interface CorpusSummary {
  counts: { diseases: number; targets: number; compounds: number; bioactivities: number };
  sources: Array<{
    id: string;
    name: string;
    license?: string | null;
    citation?: string | null;
    homepage_url?: string | null;
    entity_types: string[];
  }>;
  dataset_versions: Array<{
    label: string;
    source: string;
    dataset: string;
    version: string;
    record_count: number;
    content_hash: string;
    retrieved_at: string;
    quality_score?: number | null;
  }>;
}

export interface Tractability {
  buckets: Record<string, string>;
  all_buckets: Record<string, string[]>;
  source: string;
  meaning: string;
}

export interface AssociationScore {
  overall_score: number;
  datatype_scores: Record<string, number>;
  source: string;
  source_url?: string | null;
  retrieved_at: string;
  score_meaning?: string;
}

export interface DiseaseOverview {
  disease: {
    id: string;
    efo_id: string;
    name: string;
    description?: string | null;
    synonyms: string[];
    therapeutic_areas: Array<{ id: string; name: string }>;
    provenance: Provenance;
  };
  associated_targets: Array<{
    target: {
      id: string;
      target_key: string;
      symbol: string;
      name?: string | null;
      biotype?: string | null;
      target_class: string[];
      tractability: Record<string, string>;
    };
    association: AssociationScore;
  }>;
  evidence_summary: { by_type: Record<string, number>; total: number };
  counts: { associated_targets: number };
}

export interface EvidenceItem {
  id: string;
  evidence_type: string;
  type_meaning?: string | null;
  relation: string;
  title?: string | null;
  summary?: string | null;
  score?: number | null;
  score_type?: string | null;
  score_provided_by_source: boolean;
  citation_key: string;
  provenance: Provenance;
  publication?: {
    pmid?: string | null;
    doi?: string | null;
    title: string;
    journal?: string | null;
    year?: number | null;
    authors: string[];
    citation_count?: number | null;
    is_open_access?: boolean | null;
    url?: string | null;
  } | null;
  metadata: Record<string, unknown>;
  urls: string[];
  is_retrieved_fact: boolean;
}

export interface EvidenceGroup {
  evidence_type: string;
  meaning?: string | null;
  count: number;
  records?: EvidenceItem[];
  items?: EvidenceItem[];
}

export interface KnownCompound {
  compound_id: string;
  chembl_id?: string | null;
  name: string;
  smiles: string;
  max_phase?: number | null;
  is_approved_drug: boolean;
  best_pchembl: number;
  n_measurements: number;
  potency_nm?: number | null;
  basis: string;
}

export interface TargetOverview {
  target: {
    id: string;
    target_key: string;
    symbol: string;
    name?: string | null;
    biotype?: string | null;
    description?: string | null;
    target_class: string[];
    synonyms: string[];
    uniprot_ids: string[];
    chembl_target_ids: string[];
    subcellular_locations: string[];
    gene?: {
      ensembl_id: string;
      symbol: string;
      chromosome?: string | null;
      genomic_start?: number | null;
      genomic_end?: number | null;
    } | null;
    provenance: Provenance;
  };
  tractability: Tractability;
  safety_liabilities: Array<Record<string, unknown>>;
  disease_context?: {
    disease: { id: string; efo_id: string; name: string };
    association?: AssociationScore | null;
  } | null;
  associated_diseases: Array<{
    id: string;
    efo_id: string;
    name: string;
    overall_score: number;
    datatype_scores: Record<string, number>;
  }>;
  evidence: {
    groups: EvidenceGroup[];
    counts_by_type: Record<string, number>;
    total: number;
    returned: number;
  };
  known_compounds: KnownCompound[];
  chemistry_summary: {
    n_known_compounds: number;
    min_pchembl_filter: number;
    has_small_molecule_chemistry: boolean;
    note?: string | null;
  };
}

export interface DescriptorValues {
  provider: string;
  descriptor_set_version: string;
  rdkit_version?: string | null;
  values: Record<string, number>;
  dictionary: Array<{ key: string; label: string; unit?: string | null; explanation?: string }>;
  rule_filters: {
    lipinski_violations?: number | null;
    passes_lipinski?: boolean | null;
    passes_veber?: boolean | null;
    cns_mpo_score?: number | null;
    cns_mpo_caveat?: string | null;
    lipinski_detail?: Record<string, boolean>;
  };
  basis: string;
}

export interface Measurement {
  id: string;
  target?: { id: string; symbol: string; name?: string | null } | null;
  activity_type?: string | null;
  standard_value?: number | null;
  standard_units?: string | null;
  standard_relation?: string | null;
  pchembl_value?: number | null;
  is_active?: boolean | null;
  assay?: {
    chembl_assay_id: string;
    description?: string | null;
    assay_type?: string | null;
    organism?: string | null;
    confidence_score?: number | null;
    confidence_meaning?: string;
  } | null;
  data_validity_comment?: string | null;
  quality_warning: boolean;
  provenance: Provenance;
  is_experimental_measurement: boolean;
}

export interface SimilarityHit {
  compound_id: string;
  similarity: number;
  name?: string | null;
  chembl_id?: string | null;
  smiles?: string | null;
  max_phase?: number | null;
  shares_scaffold?: boolean | null;
  properties: Record<string, number>;
}

export interface SimilarityResponse {
  query: {
    input_smiles: string;
    canonical_smiles: string;
    inchikey: string;
    fingerprint_kind: string;
    popcount: number;
    standardization_notes: string[];
  };
  threshold: number;
  hits: SimilarityHit[];
  count: number;
  search_stats: {
    corpus_compounds: number;
    popcount_window: [number, number];
    candidates_examined: number;
    screened_out_fraction: number;
    method: string;
  };
}

export interface CompoundOverview {
  compound: {
    id: string;
    inchikey: string;
    smiles: string;
    name: string;
    synonyms: string[];
    chembl_id?: string | null;
    pubchem_cid?: string | null;
    max_phase?: number | null;
    max_phase_meaning?: string | null;
    molecule_type?: string | null;
    first_approval?: number | null;
    is_approved_drug: boolean;
    withdrawn_flag: boolean;
    murcko_scaffold?: string | null;
    provenance: Provenance;
  };
  properties?: DescriptorValues | null;
  external_properties?: DescriptorValues | null;
  bioactivity?: {
    measurements: Measurement[];
    targets: Array<{
      target_id: string;
      symbol: string;
      name?: string | null;
      best_pchembl: number;
      n_measurements: number;
    }>;
    count: number;
  };
  similar_compounds?: SimilarityResponse;
}

export interface Prediction {
  compound_id?: string | null;
  smiles: string;
  endpoint: string;
  model: string;
  model_id: string;
  model_family: string;
  task_type: 'classification' | 'regression';
  value?: number | null;
  probability?: number | null;
  predicted_class?: string | null;
  uncertainty?: number | null;
  confidence?: number | null;
  units?: string | null;
  applicability: {
    nearest_training_similarity?: number | null;
    in_domain?: boolean | null;
    threshold: number;
  };
  flags: string[];
  prediction_id?: string | null;
  latency_ms?: number | null;
  cache_hit: boolean;
  positive_class_meaning?: string | null;
  direction?: string | null;
  is_prediction: true;
  disclaimer: string;
  name?: string | null;
}

export interface PredictionResponse {
  predictions: Prediction[];
  models_used: Array<{
    endpoint: string;
    ref: string;
    model_id: string;
    family: string;
    version: string;
    task_type: string;
    units?: string | null;
    scaffold_metrics?: Record<string, number> | null;
  }>;
  failures: Array<Record<string, unknown>>;
  total_latency_ms: number;
  summary_by_endpoint?: Record<string, Record<string, unknown>>;
  unresolved_compounds?: string[];
}

export interface FactorScore {
  key: string;
  label: string;
  basis: 'measured' | 'predicted' | 'computed' | 'derived';
  raw_value?: number | null;
  desirability?: number | null;
  weight: number;
  normalized_weight: number;
  contribution: number;
  explanation: string;
  curve: string;
  context: Record<string, unknown>;
  discounted_for_low_confidence: boolean;
  missing_reason?: string | null;
}

export interface RankedCandidate {
  compound_id: string;
  rank: number;
  total_score: number;
  name?: string | null;
  smiles?: string | null;
  chembl_id?: string | null;
  factors: FactorScore[];
  top_contributors: FactorScore[];
  flags: string[];
  evidence_ids: string[];
  prediction_ids: string[];
  objectives_scored: number;
  objectives_missing: number;
  disclaimer: string;
}

export interface RankingObjective {
  key: string;
  label: string;
  weight: number;
  source_field: string;
  basis: string;
  desirability: { kind: string; description: string };
  explanation: string;
  required: boolean;
}

export interface RankingConfig {
  name: string;
  description: string;
  objectives: RankingObjective[];
  filters: Array<{ key: string; label: string; reason: string; enabled: boolean }>;
  min_objectives_present: number;
  confidence_penalty: number;
  notes: string[];
  normalized_weights: Record<string, number>;
}

export interface RankingResponse {
  ranking_id?: string | null;
  config: RankingConfig;
  config_hash: string;
  n_ranked: number;
  n_filtered_out: number;
  candidates: RankedCandidate[];
  filtered_out: Array<{
    compound_id: string;
    name?: string | null;
    failed_filter: string;
    reason: string;
    value: unknown;
  }>;
  warnings: string[];
  disclaimer: string;
  metadata: {
    target?: { id: string; symbol: string; target_key: string } | null;
    disease?: { id: string; name: string; efo_id: string } | null;
    association_score?: number | null;
    n_candidates: number;
    n_with_measured_potency?: number;
    models_used: Array<{ endpoint: string; ref: string; family: string }>;
    prediction_summary?: Record<string, Record<string, unknown>>;
    warning?: string;
  };
}

export interface ComparisonResponse {
  candidates: Array<{
    compound_id: string;
    name?: string | null;
    chembl_id?: string | null;
    rank: number;
    total_score: number;
    flags: string[];
    values: Record<
      string,
      {
        raw_value?: number | null;
        desirability?: number | null;
        contribution: number;
        missing_reason?: string | null;
        discounted: boolean;
      }
    >;
  }>;
  objectives: Array<{
    key: string;
    label: string;
    basis: string;
    explanation: string;
    curve: string;
  }>;
  highlights: Record<string, { best: string; worst: string }>;
  disclaimer: string;
  note?: string;
  config?: RankingConfig;
}

export type GraphNodeKind =
  'disease' | 'target' | 'gene' | 'evidence' | 'compound' | 'assay' | 'prediction' | 'publication';

export interface GraphNode {
  id: string;
  kind: GraphNodeKind;
  nature: 'retrieved' | 'model_output';
  label: string;
  sublabel?: string | null;
  entity_id?: string | null;
  score?: number | null;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  evidence_source?: string | null;
  evidence_type?: string | null;
  source_identifier?: string | null;
  source_url?: string | null;
  score?: number | null;
  score_type?: string | null;
  score_provided_by_source: boolean;
  retrieved_at?: string | null;
  metadata: Record<string, unknown>;
}

export interface EvidenceGraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    n_nodes: number;
    n_edges: number;
    node_counts: Record<string, number>;
    sources: string[];
  };
  legend: { natures: Record<string, string>; node_kinds: string[] };
  root: { disease?: string | null; target?: string | null; compound?: string | null };
}

export interface ToolCall {
  id: string;
  step_index: number;
  tool_name: string;
  tool_version?: string | null;
  arguments: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  result_summary?: string | null;
  record_ids: Record<string, string[]>;
  citations: string[];
  status: string;
  error?: string | null;
  error_code?: string | null;
  latency_ms?: number | null;
  cache_hit: boolean;
  validation: Record<string, unknown>;
  created_at: string;
}

export interface AgentExecution {
  id: string;
  step_index: number;
  phase: string;
  status: string;
  model?: string | null;
  prompt_template_id?: string | null;
  rationale?: string | null;
  output_state: Record<string, unknown>;
  latency_ms?: number | null;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  error?: string | null;
  tool_call_ids: string[];
}

export interface ReportClaim {
  text: string;
  kind: 'retrieved_fact' | 'model_prediction' | 'interpretation';
  citations: string[];
  validated: boolean;
  validation_note?: string | null;
}

export interface ReportCitation {
  key: string;
  source: string;
  source_identifier?: string | null;
  title?: string | null;
  url?: string | null;
  kind: string;
  retrieved_at?: string | null;
  score?: number | null;
  reference_text?: string;
  publication?: Record<string, unknown> | null;
}

export interface Report {
  report_id?: string;
  run_id?: string;
  id?: string;
  title: string;
  question: string;
  summary: string;
  markdown: string;
  sections: Array<{ id: string; title: string; kind: string }>;
  claims: ReportClaim[];
  citations: ReportCitation[];
  limitations: string[];
  validation: {
    total_claims: number;
    validated_claims: number;
    rejected_claims: Array<{ text: string; reason: string; unknown_citations?: string[] }>;
    rejection_rate: number;
    unknown_citations: string[];
    citations_available: number;
    policy: string;
  };
  content_hash?: string | null;
}

export interface RunSummary {
  run_id: string;
  query: string;
  workflow: string;
  status: string;
  principal?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  runtime_ms?: number | null;
  n_steps: number;
  n_tool_calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  reproducibility_hash?: string | null;
  software_version?: string | null;
  git_sha?: string | null;
  agent_model?: string | null;
  llm_backend?: string | null;
  random_seed?: number | null;
  parent_run_id?: string | null;
  replay_of_run_id?: string | null;
  has_report: boolean;
  n_errors: number;
}

export interface RunDetail extends RunSummary {
  parameters: Record<string, unknown>;
  plan: {
    objective?: string;
    strategy?: string;
    ranking_preset?: string;
    open_questions?: string[];
    steps?: Array<{
      step: number;
      tool: string;
      purpose: string;
      arguments: Record<string, unknown>;
      depends_on_previous: boolean;
      optional: boolean;
    }>;
  };
  state: Record<string, unknown>;
  ranking_config: Record<string, unknown>;
  pins: {
    dataset_versions: Array<{
      id: string;
      label: string;
      source: string;
      dataset: string;
      version: string;
      content_hash: string;
      record_count: number;
      retrieved_at: string;
    }>;
    model_versions: Array<{
      id: string;
      ref: string;
      endpoint: string;
      family: string;
      version: string;
      artifact_hash?: string | null;
      task_type: string;
      scaffold_metrics?: Record<string, number> | null;
    }>;
  };
  prompts: Array<{
    template_id: string;
    role: string;
    content: string;
    content_hash: string;
    recorded_at: string;
  }>;
  retrieved_record_ids: Record<string, string[]>;
  errors: Array<Record<string, unknown>>;
  warnings: Array<Record<string, unknown>>;
  executions: AgentExecution[];
  tool_calls: ToolCall[];
  report?: Report | null;
}

export interface RunListResponse {
  runs: RunSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface RunUsage {
  runs: {
    runs: number;
    avg_runtime_ms?: number | null;
    max_runtime_ms?: number | null;
    input_tokens: number;
    output_tokens: number;
    cost_usd: number;
    failed_runs: number;
    failure_rate: number;
  };
  tools: Array<{
    tool_name: string;
    calls: number;
    errors: number;
    error_rate: number;
    avg_latency_ms?: number | null;
    max_latency_ms?: number | null;
    cache_hit_rate: number;
  }>;
}

export interface MetricWithCi {
  value?: number;
  ci_low?: number | null;
  ci_high?: number | null;
}

export interface ModelEvaluation {
  id: string;
  model?: string | null;
  model_id: string;
  endpoint?: string | null;
  family?: string | null;
  split_strategy: string;
  split_name: string;
  n_samples?: number | null;
  metrics: Record<string, unknown>;
  calibration: {
    bins?: Array<{
      bin_lower: number;
      bin_upper: number;
      count: number;
      mean_predicted?: number | null;
      observed_frequency?: number | null;
      gap?: number | null;
    }>;
    ece?: number | null;
    mce?: number | null;
    brier?: number | null;
    coverage?: number | null;
    nominal?: number | null;
    uncertainty_error_correlation?: number | null;
    mean_uncertainty?: number | null;
    interpretation?: string;
    n?: number;
  };
  error_analysis: {
    worst_predictions?: Array<{
      identifier: string;
      smiles?: string | null;
      predicted: number;
      actual: number;
      absolute_error: number;
      uncertainty?: number | null;
      scaffold?: string | null;
    }>;
    error_by_molecular_weight?: Array<{
      property: string;
      range: [number, number];
      count: number;
      mean_absolute_error: number;
    }>;
    error_by_logp?: Array<{
      property: string;
      range: [number, number];
      count: number;
      mean_absolute_error: number;
    }>;
    error_by_uncertainty_decile?: Array<{
      decile: number;
      uncertainty_range: [number, number];
      count: number;
      mean_absolute_error: number;
    }>;
    error_by_scaffold_familiarity?: Array<{
      familiarity: string;
      count: number;
      mean_absolute_error: number;
    }>;
    error_summary?: {
      mean_absolute_error: number;
      p90_absolute_error: number;
      worst_absolute_error: number;
    };
  };
  prediction_distribution: {
    count: number;
    min?: number;
    max?: number;
    mean?: number;
    std?: number;
    quantiles?: Record<string, number>;
    histogram?: { bin_edges: number[]; counts: number[] };
  };
  per_fold_metrics: Array<{
    fold: number;
    n_test: number;
    primary_metric: string;
    value?: number | null;
    metrics: Record<string, unknown>;
  }>;
  notes?: string | null;
  created_at: string;
}

export interface MlModel {
  id: string;
  ref: string;
  name: string;
  version: string;
  family: string;
  endpoint: string;
  task_type: 'classification' | 'regression';
  status: string;
  framework: string;
  description?: string | null;
  units?: string | null;
  positive_class_meaning?: string | null;
  split_strategy?: string | null;
  metrics: {
    scaffold?: Record<string, number | number[]> | null;
    random?: Record<string, number | number[]> | null;
    cross_validation?: Record<string, unknown> | null;
  };
  calibration: Record<string, unknown>;
  hyperparameters: Record<string, unknown>;
  feature_spec: Record<string, unknown>;
  n_descriptor_features: number;
  training_dataset_version_id?: string | null;
  experiment_id?: string | null;
  trained_at?: string | null;
  training_runtime_s?: number | null;
  artifact_hash?: string | null;
  artifact_size_bytes?: number | null;
  git_sha?: string | null;
  random_seed?: number | null;
  evaluations?: ModelEvaluation[];
  reliability?: ModelEvaluation['calibration'] | null;
}

export interface Experiment {
  id: string;
  name: string;
  description?: string | null;
  endpoint: string;
  status: string;
  dataset_version_id?: string | null;
  parameters: Record<string, unknown>;
  metrics: Record<string, Record<string, unknown>>;
  comparison: {
    primary_metric?: string;
    higher_is_better?: boolean;
    rows?: Array<{
      model: string;
      split_strategy: string;
      n_samples: number;
      primary_metric: string;
      value?: number | null;
      ci_low?: number | null;
      ci_high?: number | null;
      headline: string;
    }>;
    winner?: string | null;
    statistically_indistinguishable_from_winner?: string[];
    verdict?: string;
    split_difficulty?: Record<string, number>;
    optimism_gap?: Record<
      string,
      {
        metric: string;
        scaffold: number;
        random: number;
        absolute_gap: number;
        relative_gap_pct?: number | null;
        interpretation: string;
      }
    >;
  };
  artifacts: Record<string, { version: string; model_id?: string | null }>;
  started_at?: string | null;
  finished_at?: string | null;
  runtime_s?: number | null;
  git_sha?: string | null;
  random_seed?: number | null;
  environment: Record<string, string>;
  error?: string | null;
  models: string[] | MlModel[];
}

export interface AgentTool {
  name: string;
  category: string;
  description: string;
  mutates: boolean;
  expensive: boolean;
  input_schema: Record<string, unknown>;
}

export interface RunAccepted {
  run_id: string;
  status: string;
  job_id?: string | null;
  poll: string;
  inline: boolean;
  tool_calls?: number;
  records_retrieved?: Record<string, number>;
  readiness?: { ready: boolean; reasons: string[] };
  report?: Record<string, unknown>;
  usage?: Record<string, number | string>;
}
