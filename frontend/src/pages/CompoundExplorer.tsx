/**
 * Compound explorer.
 *
 * Shows structure, descriptors, measured activity, structural neighbours and model
 * predictions. Measured activity and predictions live in visually distinct panels with
 * different badges -- never interleaved in one table, because that is exactly how a
 * reader comes away thinking a predicted value was measured.
 */

import { useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useMutation, useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { DescriptorValues, Prediction } from '@/types/api';
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  Field,
  Spinner,
  StatTile,
  Table,
} from '@/components/Primitives';
import { BasisBadge, PredictionDisclaimer, ProvenanceChip } from '@/components/Provenance';
import { MoleculeCard, Molecule } from '@/components/Molecule';
import { confidenceStatus, clinicalPhase, num, potency, titleCase, truncate } from '@/lib/format';

export default function CompoundExplorer() {
  const params = useParams<{ identifier?: string }>();
  const navigate = useNavigate();
  const [smilesInput, setSmilesInput] = useState('');
  const [threshold, setThreshold] = useState(0.7);
  const [selected, setSelected] = useState<string[]>([]);

  const compound = useQuery({
    queryKey: ['compound', params.identifier, threshold],
    queryFn: () =>
      api.compound(params.identifier as string, {
        similarityThreshold: threshold,
        includeSimilar: true,
      }),
    enabled: Boolean(params.identifier),
  });

  const predictions = useQuery({
    queryKey: ['predict', params.identifier],
    queryFn: () => api.predict({ compound_ids: [params.identifier as string] }),
    enabled: Boolean(params.identifier),
  });

  const adhoc = useMutation({
    mutationFn: async (smiles: string) => {
      const [properties, predicted, similar] = await Promise.all([
        api.computeProperties(smiles),
        api.predict({ smiles: [smiles], persist: false }).catch(() => null),
        api.similarity({ smiles, threshold: 0.6, limit: 12 }).catch(() => null),
      ]);
      return { properties, predicted, similar };
    },
  });

  if (!params.identifier) {
    return (
      <div className="space-y-4">
        <Card
          title="Analyse a structure"
          subtitle="Paste a SMILES string to compute descriptors, run the models and search the corpus for neighbours"
        >
          <form
            className="flex flex-wrap items-center gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              if (smilesInput.trim()) adhoc.mutate(smilesInput.trim());
            }}
          >
            <input
              value={smilesInput}
              onChange={(event) => setSmilesInput(event.target.value)}
              placeholder="CC(=O)Oc1ccccc1C(=O)O"
              className="min-w-[320px] flex-1 rounded border border-surface-border bg-surface-raised px-3 py-2 font-mono text-xs text-ink placeholder:text-ink-muted focus:border-[color:var(--series-1)] focus:outline-none"
            />
            <Button type="submit" variant="primary" disabled={adhoc.isPending}>
              {adhoc.isPending ? 'Computing…' : 'Analyse'}
            </Button>
            {[
              'CC(=O)Oc1ccccc1C(=O)O',
              'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',
              'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
            ].map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setSmilesInput(example)}
                className="rounded border border-surface-border px-1.5 py-1 font-mono text-2xs text-ink-muted hover:text-ink-secondary"
              >
                {truncate(example, 22)}
              </button>
            ))}
          </form>
          {adhoc.error && (
            <div className="mt-3">
              <ErrorState error={adhoc.error} />
            </div>
          )}
        </Card>

        {adhoc.data && (
          <div className="grid gap-4 lg:grid-cols-3">
            <Card title="Structure" dense>
              <MoleculeCard
                smiles={adhoc.data.properties.canonical_smiles}
                name={adhoc.data.properties.formula}
                subtitle={adhoc.data.properties.inchikey}
                width={260}
                height={190}
              />
              {adhoc.data.properties.standardization_notes.length > 0 && (
                <p className="mt-2 text-2xs text-ink-muted">
                  Standardisation: {adhoc.data.properties.standardization_notes.join(', ')}
                </p>
              )}
            </Card>

            <Card
              className="lg:col-span-2"
              title="Computed descriptors"
              actions={<BasisBadge basis="computed" />}
            >
              <DescriptorGrid descriptors={adhoc.data.properties.properties} />
            </Card>

            {adhoc.data.predicted && adhoc.data.predicted.predictions.length > 0 && (
              <Card
                className="lg:col-span-2"
                title="Model predictions"
                actions={<BasisBadge basis="predicted" />}
              >
                <PredictionTable predictions={adhoc.data.predicted.predictions} />
                <PredictionDisclaimer />
              </Card>
            )}

            {adhoc.data.similar && (
              <Card title="Nearest neighbours in the corpus" dense>
                {adhoc.data.similar.hits.length === 0 ? (
                  <p className="text-xs text-ink-muted">
                    Nothing within Tanimoto 0.6 of this structure is in the corpus.
                  </p>
                ) : (
                  <ul className="space-y-1.5">
                    {adhoc.data.similar.hits.slice(0, 8).map((hit) => (
                      <li
                        key={hit.compound_id}
                        className="flex items-center justify-between gap-2 text-2xs"
                      >
                        <Link
                          to={`/compounds/${hit.compound_id}`}
                          className="truncate text-ink-secondary hover:text-ink hover:underline"
                        >
                          {hit.name ?? hit.chembl_id ?? hit.compound_id}
                        </Link>
                        <span className="font-mono tabular-nums text-ink">
                          {hit.similarity.toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {compound.isLoading && <Spinner label="Loading compound…" />}
      {compound.error && (
        <ErrorState error={compound.error} retry={() => void compound.refetch()} />
      )}

      {compound.data && (
        <>
          <div className="grid gap-4 lg:grid-cols-4">
            <Card title="Structure" dense>
              <MoleculeCard
                smiles={compound.data.compound.smiles}
                name={compound.data.compound.name}
                subtitle={compound.data.compound.chembl_id ?? compound.data.compound.inchikey}
                width={250}
                height={185}
              />
              <div className="mt-2 flex flex-wrap gap-1">
                {compound.data.compound.is_approved_drug && (
                  <Badge tone="good">Approved drug</Badge>
                )}
                {compound.data.compound.withdrawn_flag && <Badge tone="critical">Withdrawn</Badge>}
                {compound.data.compound.molecule_type && (
                  <Badge tone="neutral">{compound.data.compound.molecule_type}</Badge>
                )}
              </div>
            </Card>

            <Card
              className="lg:col-span-3"
              title={compound.data.compound.name}
              actions={<ProvenanceChip provenance={compound.data.compound.provenance} />}
            >
              <dl className="grid gap-x-6 sm:grid-cols-2">
                <Field label="ChEMBL">{compound.data.compound.chembl_id ?? '—'}</Field>
                <Field label="PubChem CID">{compound.data.compound.pubchem_cid ?? '—'}</Field>
                <Field label="InChIKey">
                  <code className="break-all font-mono text-2xs">
                    {compound.data.compound.inchikey}
                  </code>
                </Field>
                <Field label="Clinical phase">
                  {clinicalPhase(compound.data.compound.max_phase)}
                  {compound.data.compound.first_approval &&
                    ` (${compound.data.compound.first_approval})`}
                </Field>
                <Field label="Scaffold">
                  <code className="break-all font-mono text-2xs">
                    {compound.data.compound.murcko_scaffold ?? 'acyclic — no Bemis-Murcko scaffold'}
                  </code>
                </Field>
                <Field label="Synonyms">
                  {compound.data.compound.synonyms.slice(0, 4).join(', ') || '—'}
                </Field>
              </dl>
              {compound.data.compound.max_phase_meaning && (
                <p className="mt-2 text-2xs text-ink-muted">
                  Phase meaning: {compound.data.compound.max_phase_meaning}. Retrieved from ChEMBL,
                  not inferred.
                </p>
              )}
            </Card>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card
              title="Computed molecular properties"
              subtitle={
                compound.data.properties
                  ? `RDKit ${compound.data.properties.rdkit_version ?? ''} · descriptor set v${compound.data.properties.descriptor_set_version}`
                  : undefined
              }
              actions={<BasisBadge basis="computed" />}
            >
              {compound.data.properties ? (
                <DescriptorGrid descriptors={compound.data.properties} />
              ) : (
                <p className="text-xs text-ink-muted">
                  Descriptors have not been computed for this compound yet.
                </p>
              )}
              {compound.data.external_properties && (
                <div className="mt-3 border-t border-surface-border pt-2">
                  <p className="text-2xs uppercase tracking-wide text-ink-muted">
                    PubChem-reported values, for cross-checking
                  </p>
                  <dl className="mt-1 grid grid-cols-2 gap-x-4 sm:grid-cols-3">
                    {Object.entries(compound.data.external_properties.values)
                      .slice(0, 6)
                      .map(([key, value]) => (
                        <div key={key} className="flex justify-between gap-2 text-2xs">
                          <dt className="text-ink-muted">{titleCase(key)}</dt>
                          <dd className="font-mono tabular-nums text-ink-secondary">
                            {num(value, 2)}
                          </dd>
                        </div>
                      ))}
                  </dl>
                </div>
              )}
            </Card>

            <Card
              title="Model predictions"
              subtitle="TheraGraph models — predictions, not measurements"
              actions={<BasisBadge basis="predicted" />}
            >
              {predictions.isLoading && <Spinner />}
              {predictions.error && <ErrorState error={predictions.error} />}
              {predictions.data && predictions.data.predictions.length === 0 && (
                <EmptyState
                  title="No models are registered."
                  detail="Train one with `theragraph ml train --endpoint solubility --dataset esol`."
                />
              )}
              {predictions.data && predictions.data.predictions.length > 0 && (
                <>
                  <PredictionTable predictions={predictions.data.predictions} />
                  <PredictionDisclaimer />
                  <div className="mt-2 space-y-0.5 text-2xs text-ink-muted">
                    {predictions.data.models_used.map((model) => (
                      <p key={model.ref}>
                        <code className="font-mono">{model.ref}</code> ({model.family})
                        {model.scaffold_metrics && (
                          <>
                            {' '}
                            · scaffold-split{' '}
                            {model.scaffold_metrics.roc_auc !== undefined
                              ? `ROC-AUC ${num(model.scaffold_metrics.roc_auc, 3)}`
                              : `RMSE ${num(model.scaffold_metrics.rmse, 3)}`}
                          </>
                        )}
                      </p>
                    ))}
                  </div>
                </>
              )}
            </Card>
          </div>

          {compound.data.bioactivity && compound.data.bioactivity.measurements.length > 0 && (
            <Card
              title="Measured bioactivity"
              subtitle="Experimental values retrieved from ChEMBL"
              actions={<BasisBadge basis="measured" />}
            >
              <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatTile label="Measurements" value={compound.data.bioactivity.count} />
                <StatTile label="Targets hit" value={compound.data.bioactivity.targets.length} />
                <StatTile
                  label="Best potency"
                  value={potency(compound.data.bioactivity.targets[0]?.best_pchembl)}
                  hint={compound.data.bioactivity.targets[0]?.symbol}
                />
                <StatTile
                  label="Flagged rows"
                  value={
                    compound.data.bioactivity.measurements.filter((m) => m.quality_warning).length
                  }
                  hint="Curator warning or potential duplicate"
                  tone={
                    compound.data.bioactivity.measurements.some((m) => m.quality_warning)
                      ? 'warning'
                      : 'neutral'
                  }
                />
              </div>
              <Table
                headers={['Target', 'Type', 'Value', 'pChEMBL', 'Assay', 'Confidence', 'Quality']}
              >
                {compound.data.bioactivity.measurements.slice(0, 20).map((measurement) => (
                  <tr key={measurement.id} className="hover:bg-surface-raised/60">
                    <td className="px-2 py-1.5">
                      {measurement.target ? (
                        <Link
                          to={`/targets/${measurement.target.id}`}
                          className="text-ink-secondary hover:text-ink hover:underline"
                        >
                          {measurement.target.symbol}
                        </Link>
                      ) : (
                        <span className="text-ink-muted">—</span>
                      )}
                    </td>
                    <td className="px-2 py-1.5 text-ink-muted">
                      {measurement.activity_type ?? '—'}
                    </td>
                    <td className="px-2 py-1.5 font-mono tabular-nums text-ink-secondary">
                      {measurement.standard_relation && measurement.standard_relation !== '='
                        ? measurement.standard_relation
                        : ''}
                      {num(measurement.standard_value, 1)} {measurement.standard_units ?? ''}
                    </td>
                    <td className="px-2 py-1.5 font-mono tabular-nums text-ink">
                      {num(measurement.pchembl_value, 2)}
                    </td>
                    <td
                      className="max-w-xs truncate px-2 py-1.5 text-2xs text-ink-muted"
                      title={measurement.assay?.description ?? ''}
                    >
                      {measurement.assay?.chembl_assay_id ?? '—'}
                    </td>
                    <td
                      className="px-2 py-1.5 text-2xs text-ink-muted"
                      title={measurement.assay?.confidence_meaning}
                    >
                      {measurement.assay?.confidence_score ?? '—'}
                    </td>
                    <td className="px-2 py-1.5">
                      {measurement.quality_warning ? (
                        <Badge
                          tone="warning"
                          title={measurement.data_validity_comment ?? 'flagged'}
                        >
                          flagged
                        </Badge>
                      ) : (
                        <span className="text-2xs text-ink-muted">ok</span>
                      )}
                    </td>
                  </tr>
                ))}
              </Table>
              <p className="mt-2 text-2xs text-ink-muted">
                A non-equal relation (&gt; or &lt;) means the measurement is censored: “inactive up
                to this concentration” is not the same as an activity value.
              </p>
            </Card>
          )}

          {compound.data.similar_compounds && (
            <Card
              title="Structurally similar compounds"
              subtitle={`Tanimoto over ${compound.data.similar_compounds.query.fingerprint_kind} fingerprints`}
              actions={
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-2 text-2xs text-ink-muted">
                    threshold
                    <input
                      type="range"
                      min={0.3}
                      max={0.95}
                      step={0.05}
                      value={threshold}
                      onChange={(event) => setThreshold(Number(event.target.value))}
                      className="accent-[color:var(--series-1)]"
                    />
                    <span className="w-8 font-mono tabular-nums text-ink">
                      {threshold.toFixed(2)}
                    </span>
                  </label>
                  {selected.length > 0 && (
                    <Button
                      variant="primary"
                      onClick={() =>
                        navigate(`/compare?compounds=${[params.identifier, ...selected].join(',')}`)
                      }
                    >
                      Compare {selected.length + 1} →
                    </Button>
                  )}
                </div>
              }
            >
              {compound.data.similar_compounds.hits.length === 0 ? (
                <EmptyState
                  title={`Nothing in the corpus is within Tanimoto ${threshold.toFixed(2)}.`}
                  detail="Lower the threshold, or ingest ChEMBL neighbours of this structure to widen the local chemistry."
                />
              ) : (
                <>
                  <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5">
                    {compound.data.similar_compounds.hits.slice(0, 10).map((hit) => (
                      <div
                        key={hit.compound_id}
                        className="rounded border border-surface-border bg-surface-raised/40 p-2"
                      >
                        {hit.smiles && <Molecule smiles={hit.smiles} width={180} height={120} />}
                        <div className="mt-1 flex items-start justify-between gap-1">
                          <Link
                            to={`/compounds/${hit.compound_id}`}
                            className="min-w-0 flex-1 truncate text-2xs text-ink-secondary hover:text-ink hover:underline"
                            title={hit.name ?? undefined}
                          >
                            {hit.name ?? hit.chembl_id ?? 'unnamed'}
                          </Link>
                          <span className="font-mono text-2xs tabular-nums text-ink">
                            {hit.similarity.toFixed(2)}
                          </span>
                        </div>
                        <div className="mt-1 flex items-center justify-between">
                          {hit.shares_scaffold && <Badge tone="info">same scaffold</Badge>}
                          <label className="ml-auto flex items-center gap-1 text-2xs text-ink-muted">
                            <input
                              type="checkbox"
                              checked={selected.includes(hit.compound_id)}
                              onChange={(event) =>
                                setSelected((current) =>
                                  event.target.checked
                                    ? [...current, hit.compound_id].slice(0, 7)
                                    : current.filter((id) => id !== hit.compound_id),
                                )
                              }
                              className="accent-[color:var(--series-1)]"
                            />
                            compare
                          </label>
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="mt-3 text-2xs text-ink-muted">
                    Searched {compound.data.similar_compounds.search_stats.candidates_examined} of{' '}
                    {compound.data.similar_compounds.search_stats.corpus_compounds} compounds after
                    popcount screening (
                    {(
                      compound.data.similar_compounds.search_stats.screened_out_fraction * 100
                    ).toFixed(0)}
                    % eliminated without computing a similarity). Method:{' '}
                    {compound.data.similar_compounds.search_stats.method}.
                  </p>
                </>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function DescriptorGrid({ descriptors }: { descriptors: DescriptorValues }) {
  const dictionary = useMemo(
    () => new Map(descriptors.dictionary.map((entry) => [entry.key, entry])),
    [descriptors.dictionary],
  );
  const filters = descriptors.rule_filters;

  return (
    <>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3">
        {Object.entries(descriptors.values).map(([key, value]) => {
          const entry = dictionary.get(key);
          return (
            <div
              key={key}
              className="flex items-baseline justify-between gap-2 text-xs"
              title={entry?.explanation}
            >
              <dt className="truncate text-ink-muted">{entry?.label ?? titleCase(key)}</dt>
              <dd className="shrink-0 font-mono tabular-nums text-ink">
                {num(value, Number.isInteger(value) ? 0 : 2)}
                {entry?.unit && (
                  <span className="ml-0.5 text-2xs text-ink-muted">{entry.unit}</span>
                )}
              </dd>
            </div>
          );
        })}
      </dl>
      <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-surface-border pt-2">
        <Badge tone={filters.passes_lipinski ? 'good' : 'warning'}>
          Lipinski:{' '}
          {filters.passes_lipinski ? 'pass' : `${filters.lipinski_violations ?? '?'} violation(s)`}
        </Badge>
        <Badge tone={filters.passes_veber ? 'good' : 'warning'}>
          Veber: {filters.passes_veber ? 'pass' : 'fail'}
        </Badge>
        {filters.cns_mpo_score !== null && filters.cns_mpo_score !== undefined && (
          <Badge tone="neutral" title={filters.cns_mpo_caveat ?? undefined}>
            CNS MPO {num(filters.cns_mpo_score, 2)}/6
          </Badge>
        )}
      </div>
      {filters.cns_mpo_caveat && (
        <p className="mt-1 text-2xs text-ink-muted">{filters.cns_mpo_caveat}</p>
      )}
    </>
  );
}

function PredictionTable({ predictions }: { predictions: Prediction[] }) {
  return (
    <Table
      headers={['Endpoint', 'Prediction', 'Uncertainty', 'Confidence', 'Applicability', 'Model']}
    >
      {predictions.map((prediction) => {
        const status = confidenceStatus(prediction.confidence);
        return (
          <tr key={`${prediction.endpoint}-${prediction.model_id}-${prediction.smiles}`}>
            <td className="px-2 py-1.5">
              <span className="text-ink-secondary">{titleCase(prediction.endpoint)}</span>
              {prediction.positive_class_meaning && (
                <span
                  className="block text-2xs text-ink-muted"
                  title={prediction.positive_class_meaning}
                >
                  positive = {truncate(prediction.positive_class_meaning, 40)}
                </span>
              )}
            </td>
            <td className="px-2 py-1.5 font-mono tabular-nums text-ink">
              {prediction.probability !== null && prediction.probability !== undefined
                ? `${(prediction.probability * 100).toFixed(1)}%`
                : `${num(prediction.value, 3)}${prediction.units ? ` ${prediction.units}` : ''}`}
            </td>
            <td className="px-2 py-1.5 font-mono tabular-nums text-ink-muted">
              {prediction.uncertainty !== null && prediction.uncertainty !== undefined
                ? `± ${num(prediction.uncertainty, 3)}`
                : 'not reported'}
            </td>
            <td className="px-2 py-1.5">
              <Badge
                tone={status.role === 'unknown' ? 'neutral' : status.role}
                icon={
                  <span aria-hidden>
                    {status.role === 'good' ? '✓' : status.role === 'critical' ? '⚠' : '•'}
                  </span>
                }
              >
                {status.label}
                {prediction.confidence !== null &&
                  prediction.confidence !== undefined &&
                  ` ${num(prediction.confidence, 2)}`}
              </Badge>
            </td>
            <td className="px-2 py-1.5">
              {prediction.applicability.in_domain === false ? (
                <Badge
                  tone="warning"
                  icon={<span aria-hidden>⚠</span>}
                  title="Extrapolation beyond the training chemistry"
                >
                  out of domain
                </Badge>
              ) : (
                <span className="text-2xs text-ink-muted">
                  in domain
                  {prediction.applicability.nearest_training_similarity !== null &&
                    prediction.applicability.nearest_training_similarity !== undefined && (
                      <> (NN {num(prediction.applicability.nearest_training_similarity, 2)})</>
                    )}
                </span>
              )}
            </td>
            <td className="px-2 py-1.5 font-mono text-2xs text-ink-muted">
              {prediction.model_family}
            </td>
          </tr>
        );
      })}
    </Table>
  );
}
