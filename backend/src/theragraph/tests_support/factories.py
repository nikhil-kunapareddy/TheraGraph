"""Deterministic corpus factories.

Real structures and plausible values, constructed in-process. Used by the test suite and
by ``theragraph db bootstrap --with-sample-data`` so a developer can click through the UI
without waiting on ingestion.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from theragraph.chem.featurize import featurize_many
from theragraph.chem.molecule import standardize_smiles
from theragraph.db.repositories import biology as biology_repo
from theragraph.db.repositories import chemistry as chem_repo
from theragraph.db.repositories import provenance as prov_repo


#: Four real molecules with genuinely different chemistry, so similarity and
#: property tests have meaningful spread.
class SampleCompound(TypedDict):
    """Shape of the in-process compound fixtures (real molecules, real ChEMBL ids)."""

    smiles: str
    name: str
    chembl_id: str
    max_phase: float
    pchembl: float


SAMPLE_COMPOUNDS: list[SampleCompound] = [
    {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "name": "Aspirin",
        "chembl_id": "CHEMBL25",
        "max_phase": 4.0,
        "pchembl": 5.2,
    },
    {
        "smiles": "Oc1ccccc1C(=O)O",
        "name": "Salicylic acid",
        "chembl_id": "CHEMBL424",
        "max_phase": 4.0,
        "pchembl": 6.8,
    },
    {
        "smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "name": "Caffeine",
        "chembl_id": "CHEMBL113",
        "max_phase": 4.0,
        "pchembl": 7.4,
    },
    {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "name": "Ibuprofen",
        "chembl_id": "CHEMBL521",
        "max_phase": 4.0,
        "pchembl": 8.1,
    },
]


async def build_corpus(session: AsyncSession, *, now: datetime | None = None) -> dict[str, Any]:
    """Create a minimal but complete corpus and return the ids."""
    retrieved_at = now or datetime.now(UTC)

    from theragraph.ingestion.pipelines import ensure_data_sources

    await ensure_data_sources(session)

    dataset_version = await prov_repo.create_dataset_version(
        session,
        source_id="open_targets",
        dataset_name="disease_associations",
        version="test.1",
        content_hash="testhash0000",
        record_count=7,
        retrieved_at=retrieved_at,
        quality={"score": 1.0},
    )
    chembl_version = await prov_repo.create_dataset_version(
        session,
        source_id="chembl",
        dataset_name="bioactivity",
        version="ChEMBL_test",
        content_hash="testhash0001",
        record_count=8,
        retrieved_at=retrieved_at,
        quality={"score": 1.0},
    )

    provenance = {
        "source": "open_targets",
        "retrieved_at": retrieved_at,
        "dataset_version_id": dataset_version.id,
    }

    gene_ids = await biology_repo.upsert_genes(
        session,
        [
            {
                "ensembl_id": "ENSG00000186318",
                "symbol": "BACE1",
                "name": "beta-secretase 1",
                "biotype": "protein_coding",
                "chromosome": "11",
                "uniprot_ids": ["P56817"],
                "source_identifier": "ENSG00000186318",
                **provenance,
            },
            {
                "ensembl_id": "ENSG00000142192",
                "symbol": "APP",
                "name": "amyloid beta precursor protein",
                "biotype": "protein_coding",
                "chromosome": "21",
                "uniprot_ids": ["P05067"],
                "source_identifier": "ENSG00000142192",
                **provenance,
            },
        ],
    )

    target_ids = await biology_repo.upsert_targets(
        session,
        [
            {
                "target_key": "ENSG00000186318",
                "approved_symbol": "BACE1",
                "approved_name": "beta-secretase 1",
                "biotype": "protein_coding",
                "target_class": ["Enzyme", "Protease"],
                "synonyms": ["ASP2", "memapsin-2"],
                "uniprot_ids": ["P56817"],
                "chembl_target_ids": ["CHEMBL4822"],
                "tractability": {
                    "Small molecule": "Clinical Precedence",
                    "_all_buckets": {"Small molecule": ["Clinical Precedence"]},
                },
                "description": "Aspartic protease that cleaves amyloid precursor protein.",
                "gene_id": gene_ids[0],
                "source_identifier": "ENSG00000186318",
                "source_url": "https://platform.opentargets.org/target/ENSG00000186318",
                **provenance,
            },
            {
                "target_key": "ENSG00000142192",
                "approved_symbol": "APP",
                "approved_name": "amyloid beta precursor protein",
                "biotype": "protein_coding",
                "target_class": ["Other"],
                "synonyms": ["A4", "AD1"],
                "uniprot_ids": ["P05067"],
                "chembl_target_ids": ["CHEMBL2487"],
                "tractability": {"Antibody": "Clinical Precedence", "_all_buckets": {}},
                "description": "Cell-surface receptor cleaved to release amyloid beta.",
                "gene_id": gene_ids[1],
                "source_identifier": "ENSG00000142192",
                **provenance,
            },
        ],
    )

    disease_ids = await biology_repo.upsert_diseases(
        session,
        [
            {
                "efo_id": "MONDO_0004975",
                "name": "Alzheimer disease",
                "description": "A progressive neurodegenerative disease.",
                "synonyms": ["AD", "Alzheimer's disease"],
                "therapeutic_areas": [{"id": "MONDO_0005071", "name": "nervous system disorder"}],
                "source_identifier": "MONDO_0004975",
                **provenance,
            }
        ],
    )

    await biology_repo.upsert_associations(
        session,
        [
            {
                "target_id": target_ids[0],
                "disease_id": disease_ids[0],
                "overall_score": 0.74,
                "datatype_scores": {"genetic_association": 0.6, "known_drug": 0.9},
                "evidence_count": 2,
                "source_identifier": "ENSG00000186318-MONDO_0004975",
                **provenance,
            },
            {
                "target_id": target_ids[1],
                "disease_id": disease_ids[0],
                "overall_score": 0.87,
                "datatype_scores": {"genetic_association": 0.92, "literature": 0.99},
                "evidence_count": 3,
                "source_identifier": "ENSG00000142192-MONDO_0004975",
                **provenance,
            },
        ],
    )

    publication_ids = await biology_repo.upsert_publications(
        session,
        [
            {
                "external_id": "1520398",
                "pmid": "1520398",
                "doi": "10.1212/wnl.42.8.1445",
                "title": (
                    "Molecular characterisation of a pedigree with familial Alzheimer disease."
                ),
                "journal": "Neurology",
                "year": 1992,
                "authors": ["Karlinsky H", "Vaula G"],
                "citation_count": 42,
                "is_open_access": False,
                "source": "europepmc",
                "source_identifier": "1520398",
                "source_url": "https://europepmc.org/article/MED/1520398",
                "retrieved_at": retrieved_at,
                "dataset_version_id": dataset_version.id,
            }
        ],
    )

    evidence_ids = await biology_repo.upsert_evidence(
        session,
        [
            {
                "evidence_type": "genetic_association",
                "relation": "disease_target",
                "target_id": target_ids[1],
                "disease_id": disease_ids[0],
                "compound_id": None,
                "publication_id": publication_ids[0],
                "score": 0.95,
                "score_type": "open_targets_evidence_score",
                "title": "Genetic association evidence linking APP to Alzheimer disease (eva)",
                "summary": (
                    "Reported disease: Alzheimer disease type 1 | Clinical significance: pathogenic"
                ),
                "extra": {"datasource_id": "eva", "pmid": "1520398"},
                "source_identifier": "evidence-genetic-1",
                "source_url": "https://platform.opentargets.org/evidence/x",
                **provenance,
            },
            {
                "evidence_type": "known_drug",
                "relation": "disease_target",
                "target_id": target_ids[0],
                "disease_id": disease_ids[0],
                "compound_id": None,
                "publication_id": None,
                "score": 0.8,
                "score_type": "open_targets_evidence_score",
                "title": "Known drug evidence linking BACE1 to Alzheimer disease (chembl)",
                "summary": "Clinical stage: PHASE_3 | Drug: VERUBECESTAT",
                "extra": {"datasource_id": "chembl", "drug_name": "VERUBECESTAT"},
                "source_identifier": "evidence-drug-1",
                **provenance,
            },
            {
                "evidence_type": "literature",
                "relation": "disease_target",
                "target_id": target_ids[1],
                "disease_id": disease_ids[0],
                "compound_id": None,
                "publication_id": publication_ids[0],
                "score": None,
                "score_type": None,
                "title": "Literature evidence linking APP to Alzheimer disease (europepmc)",
                "summary": "[results] Amyloid precursor protein processing is altered.",
                "extra": {"datasource_id": "europepmc", "pmid": "1520398"},
                "source_identifier": "evidence-lit-1",
                **provenance,
            },
        ],
    )

    assay_ids = await chem_repo.upsert_assays(
        session,
        [
            {
                "chembl_assay_id": "CHEMBL653511",
                "description": "Inhibitory activity against Beta-secretase 1",
                "assay_type": "B",
                "organism": "Homo sapiens",
                "confidence_score": 8,
                "target_id": target_ids[0],
                "chembl_target_id": "CHEMBL4822",
                "source": "chembl",
                "source_identifier": "CHEMBL653511",
                "retrieved_at": retrieved_at,
                "dataset_version_id": chembl_version.id,
            }
        ],
    )

    compound_rows: list[dict[str, Any]] = []
    for sample in SAMPLE_COMPOUNDS:
        info = standardize_smiles(sample["smiles"])
        compound_rows.append(
            {
                "inchikey": info.inchikey,
                "canonical_smiles": info.canonical_smiles,
                "inchi": info.inchi,
                "name": sample["name"],
                "synonyms": [sample["name"].lower()],
                "chembl_id": sample["chembl_id"],
                "max_phase": sample["max_phase"],
                "molecule_type": "Small molecule",
                "is_approved_drug": sample["max_phase"] >= 4,
                "withdrawn_flag": False,
                "murcko_scaffold": info.murcko_scaffold,
                "murcko_scaffold_inchikey": info.murcko_scaffold_inchikey,
                "heavy_atom_count": info.heavy_atom_count,
                "source": "chembl",
                "source_identifier": sample["chembl_id"],
                "source_url": (
                    f"https://www.ebi.ac.uk/chembl/compound_report_card/{sample['chembl_id']}/"
                ),
                "retrieved_at": retrieved_at,
                "dataset_version_id": chembl_version.id,
            }
        )
    compound_ids = await chem_repo.upsert_compounds(session, compound_rows)

    features = featurize_many(
        [
            (compound_id, row["canonical_smiles"])
            for compound_id, row in zip(compound_ids, compound_rows, strict=True)
        ]
    )
    await chem_repo.upsert_properties(session, features.properties)
    await chem_repo.upsert_fingerprints(session, features.fingerprints)

    await chem_repo.upsert_bioactivities(
        session,
        [
            {
                "compound_id": compound_id,
                "target_id": target_ids[0],
                "assay_id": assay_ids[0],
                "activity_type": "IC50",
                "standard_value": 10 ** (9 - sample["pchembl"]),
                "standard_units": "nM",
                "standard_relation": "=",
                "pchembl_value": sample["pchembl"],
                "is_active": sample["pchembl"] >= 6,
                "potential_duplicate": False,
                "source": "chembl",
                "source_identifier": f"activity-{index}",
                "retrieved_at": retrieved_at,
                "dataset_version_id": chembl_version.id,
            }
            for index, (compound_id, sample) in enumerate(
                zip(compound_ids, SAMPLE_COMPOUNDS, strict=True)
            )
        ],
    )

    await session.commit()

    return {
        "disease_id": disease_ids[0],
        "disease_efo": "MONDO_0004975",
        "target_ids": target_ids,
        "target_keys": ["ENSG00000186318", "ENSG00000142192"],
        "gene_ids": gene_ids,
        "compound_ids": compound_ids,
        "compound_smiles": [row["canonical_smiles"] for row in compound_rows],
        "evidence_ids": evidence_ids,
        "assay_ids": assay_ids,
        "publication_ids": publication_ids,
        "dataset_version_id": dataset_version.id,
        "chembl_version_id": chembl_version.id,
    }
