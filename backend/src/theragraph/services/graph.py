"""Evidence graph construction.

Builds the Disease -> Target -> Evidence -> Compound -> Assay -> Prediction graph that
the UI renders. Three rules govern every edge:

1. **Every edge names its source.** An edge carries the source id, the source
   identifier, the retrieval timestamp and a URL back to the upstream record.
2. **Scores are only shown when the source provided one.** Absent scores render as
   "not provided by source", never as zero and never as an inferred value.
3. **Prediction nodes are visually and structurally distinct from evidence nodes.**
   The node ``kind`` drives the rendering, so a model output can never be mistaken for
   a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from theragraph.core.errors import NotFoundError
from theragraph.core.logging import get_logger
from theragraph.db.repositories import biology as biology_repo
from theragraph.db.repositories import chemistry as chem_repo
from theragraph.db.repositories import ml as ml_repo
from theragraph.services.context import ServiceContext

log = get_logger(__name__)

NodeKind = Literal[
    "disease", "target", "gene", "evidence", "compound", "assay", "prediction", "publication"
]

#: Whether a node represents retrieved data or model output. The UI uses this to keep
#: the two visually separate.
NODE_NATURE: dict[str, str] = {
    "disease": "retrieved",
    "target": "retrieved",
    "gene": "retrieved",
    "evidence": "retrieved",
    "compound": "retrieved",
    "assay": "retrieved",
    "publication": "retrieved",
    "prediction": "model_output",
}


@dataclass
class GraphNode:
    id: str
    kind: NodeKind
    label: str
    sublabel: str | None = None
    entity_id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "nature": NODE_NATURE.get(self.kind, "retrieved"),
            "label": self.label,
            "sublabel": self.sublabel,
            "entity_id": self.entity_id,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    evidence_source: str | None = None
    evidence_type: str | None = None
    source_identifier: str | None = None
    source_url: str | None = None
    score: float | None = None
    score_type: str | None = None
    retrieved_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": f"{self.source}->{self.target}:{self.relation}",
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "evidence_source": self.evidence_source,
            "evidence_type": self.evidence_type,
            "source_identifier": self.source_identifier,
            "source_url": self.source_url,
            "score": self.score,
            "score_type": self.score_type,
            "score_provided_by_source": self.score is not None,
            "retrieved_at": self.retrieved_at,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    _node_ids: set[str] = field(default_factory=set, repr=False)

    def add_node(self, node: GraphNode) -> GraphNode:
        if node.id not in self._node_ids:
            self.nodes.append(node)
            self._node_ids.add(node.id)
        return node

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.source in self._node_ids and edge.target in self._node_ids:
            self.edges.append(edge)

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.kind] = counts.get(node.kind, 0) + 1
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "stats": {
                "n_nodes": len(self.nodes),
                "n_edges": len(self.edges),
                "node_counts": counts,
                "sources": sorted(
                    {edge.evidence_source for edge in self.edges if edge.evidence_source}
                ),
            },
            "legend": {
                "natures": {
                    "retrieved": "Data retrieved from a named public source.",
                    "model_output": "Produced by a TheraGraph model; not a measurement.",
                },
                "node_kinds": sorted(NODE_NATURE),
            },
        }


async def build_evidence_graph(
    context: ServiceContext,
    *,
    disease_identifier: str | None = None,
    target_identifier: str | None = None,
    compound_identifier: str | None = None,
    max_targets: int = 8,
    max_evidence_per_target: int = 6,
    max_compounds_per_target: int = 6,
    include_predictions: bool = True,
) -> dict[str, Any]:
    """Assemble a graph rooted at a disease, a target or a compound."""
    if not any((disease_identifier, target_identifier, compound_identifier)):
        raise NotFoundError("a disease, target or compound is required to build a graph")

    graph = EvidenceGraph()

    disease = (
        await biology_repo.get_disease(context.session, disease_identifier)
        if disease_identifier
        else None
    )
    if disease_identifier and disease is None:
        raise NotFoundError(f"disease {disease_identifier!r} is not in the local corpus")

    disease_node_id = None
    if disease is not None:
        disease_node_id = f"disease:{disease.id}"
        graph.add_node(
            GraphNode(
                id=disease_node_id,
                kind="disease",
                label=disease.name,
                sublabel=disease.efo_id,
                entity_id=disease.id,
                metadata={
                    "therapeutic_areas": [
                        area.get("name") for area in disease.therapeutic_areas or []
                    ],
                    "source": disease.source,
                    "source_url": disease.source_url,
                },
            )
        )

    # ------------------------------------------------------------------ targets
    target_rows: list[tuple[Any, Any]] = []
    if target_identifier:
        target = await biology_repo.get_target(context.session, target_identifier)
        if target is None:
            raise NotFoundError(f"target {target_identifier!r} is not in the local corpus")
        association = (
            await biology_repo.get_association(context.session, target.id, disease.id)
            if disease is not None
            else None
        )
        target_rows = [(target, association)]
    elif disease is not None:
        target_rows = await biology_repo.targets_for_disease(
            context.session, disease.id, limit=max_targets
        )

    for target, association in target_rows:
        target_node_id = f"target:{target.id}"
        graph.add_node(
            GraphNode(
                id=target_node_id,
                kind="target",
                label=target.approved_symbol,
                sublabel=target.approved_name,
                entity_id=target.id,
                score=round(association.overall_score, 4) if association else None,
                metadata={
                    "biotype": target.biotype,
                    "target_class": target.target_class or [],
                    "tractability": {
                        key: value
                        for key, value in (target.tractability or {}).items()
                        if not key.startswith("_")
                    },
                    "source": target.source,
                    "source_url": target.source_url,
                },
            )
        )

        if disease_node_id and association is not None:
            graph.add_edge(
                GraphEdge(
                    source=disease_node_id,
                    target=target_node_id,
                    relation="associated_with",
                    evidence_source=association.source,
                    evidence_type="association",
                    source_identifier=association.source_identifier,
                    source_url=association.source_url,
                    score=round(association.overall_score, 5),
                    score_type="open_targets_association_score",
                    retrieved_at=association.retrieved_at.isoformat(),
                    metadata={
                        "datatype_scores": association.datatype_scores or {},
                        "evidence_count": association.evidence_count,
                    },
                )
            )

        # ----------------------------------------------------------- evidence
        evidence_rows = await biology_repo.list_evidence(
            context.session,
            target_id=target.id,
            disease_id=disease.id if disease else None,
            limit=max_evidence_per_target,
        )
        for evidence in evidence_rows:
            evidence_node_id = f"evidence:{evidence.id}"
            graph.add_node(
                GraphNode(
                    id=evidence_node_id,
                    kind="evidence",
                    label=evidence.evidence_type.replace("_", " ").title(),
                    sublabel=evidence.citation_key,
                    entity_id=evidence.id,
                    score=evidence.score,
                    metadata={
                        "title": evidence.title,
                        "summary": (evidence.summary or "")[:300] or None,
                        "source": evidence.source,
                        "source_url": evidence.source_url,
                        "datasource_id": (evidence.extra or {}).get("datasource_id"),
                        "pmid": (evidence.extra or {}).get("pmid"),
                    },
                )
            )
            graph.add_edge(
                GraphEdge(
                    source=target_node_id,
                    target=evidence_node_id,
                    relation="supported_by",
                    evidence_source=evidence.source,
                    evidence_type=evidence.evidence_type,
                    source_identifier=evidence.source_identifier,
                    source_url=evidence.source_url,
                    score=evidence.score,
                    score_type=evidence.score_type,
                    retrieved_at=evidence.retrieved_at.isoformat(),
                )
            )
            if disease_node_id:
                graph.add_edge(
                    GraphEdge(
                        source=evidence_node_id,
                        target=disease_node_id,
                        relation="evidence_for",
                        evidence_source=evidence.source,
                        evidence_type=evidence.evidence_type,
                        source_identifier=evidence.source_identifier,
                        source_url=evidence.source_url,
                        retrieved_at=evidence.retrieved_at.isoformat(),
                    )
                )

            if evidence.publication is not None:
                publication_node_id = f"publication:{evidence.publication.id}"
                graph.add_node(
                    GraphNode(
                        id=publication_node_id,
                        kind="publication",
                        label=evidence.publication.title[:90],
                        sublabel=(
                            f"{evidence.publication.journal or 'unknown journal'} "
                            f"({evidence.publication.year or 'n.d.'})"
                        ),
                        entity_id=evidence.publication.id,
                        metadata={
                            "pmid": evidence.publication.pmid,
                            "doi": evidence.publication.doi,
                            "url": evidence.publication.source_url,
                            "citation_count": evidence.publication.citation_count,
                        },
                    )
                )
                graph.add_edge(
                    GraphEdge(
                        source=evidence_node_id,
                        target=publication_node_id,
                        relation="cites",
                        evidence_source=evidence.publication.source,
                        source_identifier=evidence.publication.pmid,
                        source_url=evidence.publication.source_url,
                        retrieved_at=evidence.publication.retrieved_at.isoformat(),
                    )
                )

        # ---------------------------------------------------------- compounds
        known = await chem_repo.known_compounds_for_target(
            context.session, target.id, min_pchembl=5.0, limit=max_compounds_per_target
        )
        for compound, best_pchembl, n_measurements in known:
            compound_node_id = f"compound:{compound.id}"
            graph.add_node(
                GraphNode(
                    id=compound_node_id,
                    kind="compound",
                    label=compound.display_name,
                    sublabel=compound.chembl_id,
                    entity_id=compound.id,
                    metadata={
                        "smiles": compound.canonical_smiles,
                        "max_phase": compound.max_phase,
                        "is_approved_drug": compound.is_approved_drug,
                        "source": compound.source,
                        "source_url": compound.source_url,
                    },
                )
            )
            activities = await chem_repo.bioactivities_for_target(
                context.session, target.id, limit=3
            )
            best_activity = next(
                (activity for activity in activities if activity.compound_id == compound.id),
                None,
            )
            graph.add_edge(
                GraphEdge(
                    source=target_node_id,
                    target=compound_node_id,
                    relation="has_measured_activity",
                    evidence_source="chembl",
                    evidence_type="bioactivity",
                    source_identifier=best_activity.source_identifier
                    if best_activity
                    else compound.chembl_id,
                    source_url=best_activity.source_url if best_activity else compound.source_url,
                    score=round(best_pchembl, 3),
                    score_type="pchembl_value",
                    retrieved_at=(
                        best_activity.retrieved_at.isoformat()
                        if best_activity
                        else compound.retrieved_at.isoformat()
                    ),
                    metadata={
                        "n_measurements": n_measurements,
                        "activity_type": best_activity.activity_type if best_activity else None,
                        "potency_nm": round(10 ** (9 - best_pchembl), 2),
                    },
                )
            )

            if best_activity is not None and best_activity.assay is not None:
                assay_node_id = f"assay:{best_activity.assay.id}"
                graph.add_node(
                    GraphNode(
                        id=assay_node_id,
                        kind="assay",
                        label=best_activity.assay.chembl_assay_id,
                        sublabel=(best_activity.assay.description or "")[:80] or None,
                        entity_id=best_activity.assay.id,
                        metadata={
                            "assay_type": best_activity.assay.assay_type,
                            "organism": best_activity.assay.organism,
                            "confidence_score": best_activity.assay.confidence_score,
                            "source_url": best_activity.assay.source_url,
                        },
                    )
                )
                graph.add_edge(
                    GraphEdge(
                        source=compound_node_id,
                        target=assay_node_id,
                        relation="measured_in",
                        evidence_source="chembl",
                        evidence_type="assay",
                        source_identifier=best_activity.assay.chembl_assay_id,
                        source_url=best_activity.assay.source_url,
                        retrieved_at=best_activity.assay.retrieved_at.isoformat(),
                    )
                )

            if include_predictions:
                predictions = await ml_repo.get_predictions_for_compounds(
                    context.session, [compound.id]
                )
                for prediction in predictions[:4]:
                    prediction_node_id = f"prediction:{prediction.id}"
                    graph.add_node(
                        GraphNode(
                            id=prediction_node_id,
                            kind="prediction",
                            label=prediction.endpoint.replace("_", " ").title(),
                            sublabel=(
                                f"{prediction.probability:.2f}"
                                if prediction.probability is not None
                                else f"{prediction.value:.2f}"
                                if prediction.value is not None
                                else "n/a"
                            ),
                            entity_id=prediction.id,
                            score=prediction.confidence,
                            metadata={
                                "model": prediction.model.ref
                                if prediction.model is not None
                                else None,
                                "value": prediction.value,
                                "probability": prediction.probability,
                                "uncertainty": prediction.uncertainty,
                                "confidence": prediction.confidence,
                                "in_domain": (prediction.applicability or {}).get("in_domain"),
                                "is_prediction": True,
                            },
                        )
                    )
                    graph.add_edge(
                        GraphEdge(
                            source=compound_node_id,
                            target=prediction_node_id,
                            relation="predicted_property",
                            evidence_source="theragraph_model",
                            evidence_type="model_prediction",
                            source_identifier=prediction.model.ref
                            if prediction.model is not None
                            else prediction.model_id,
                            score=prediction.confidence,
                            score_type="model_confidence",
                            retrieved_at=prediction.created_at.isoformat(),
                            metadata={"note": "Model output, not an experimental measurement."},
                        )
                    )

    # -------------------------------------------------- compound-rooted graphs
    if compound_identifier and not target_rows:
        root = await chem_repo.get_compound(context.session, compound_identifier)
        if root is None:
            raise NotFoundError(f"compound {compound_identifier!r} is not in the local corpus")
        compound_node_id = f"compound:{root.id}"
        graph.add_node(
            GraphNode(
                id=compound_node_id,
                kind="compound",
                label=root.display_name,
                sublabel=root.chembl_id,
                entity_id=root.id,
                metadata={"smiles": root.canonical_smiles, "source": root.source},
            )
        )
        for target, best, count in await chem_repo.targets_for_compound(
            context.session, root.id, limit=max_targets
        ):
            target_node_id = f"target:{target.id}"
            graph.add_node(
                GraphNode(
                    id=target_node_id,
                    kind="target",
                    label=target.approved_symbol,
                    sublabel=target.approved_name,
                    entity_id=target.id,
                    metadata={"source": target.source, "source_url": target.source_url},
                )
            )
            graph.add_edge(
                GraphEdge(
                    source=compound_node_id,
                    target=target_node_id,
                    relation="has_measured_activity",
                    evidence_source="chembl",
                    evidence_type="bioactivity",
                    score=round(best, 3),
                    score_type="pchembl_value",
                    retrieved_at=root.retrieved_at.isoformat(),
                    metadata={"n_measurements": count},
                )
            )

    payload = graph.to_dict()
    payload["root"] = {
        "disease": disease_identifier,
        "target": target_identifier,
        "compound": compound_identifier,
    }
    log.info("graph.built", **payload["stats"])
    return payload
