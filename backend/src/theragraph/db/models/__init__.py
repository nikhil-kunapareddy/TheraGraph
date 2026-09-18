"""SQLAlchemy ORM models.

Importing this package registers every mapper on ``Base.metadata``, which is what
Alembic autogeneration and the test schema bootstrap rely on.
"""

from theragraph.db.base import Base
from theragraph.db.models.biology import (
    ClinicalTrial,
    Disease,
    Evidence,
    Gene,
    Publication,
    Target,
    TargetDiseaseAssociation,
)
from theragraph.db.models.chemistry import (
    Assay,
    Bioactivity,
    Compound,
    CompoundFingerprint,
    MolecularProperty,
)
from theragraph.db.models.jobs import Job
from theragraph.db.models.ml import (
    CandidateRanking,
    CandidateScore,
    Experiment,
    MLModel,
    ModelEvaluation,
    Prediction,
)
from theragraph.db.models.provenance import (
    DatasetVersion,
    DataSource,
    IngestionRun,
    ProvenanceRecord,
)
from theragraph.db.models.runs import (
    AgentExecution,
    Report,
    ResearchRun,
    RunCounter,
    ToolCall,
)

__all__ = [
    "AgentExecution",
    "Assay",
    "Base",
    "Bioactivity",
    "CandidateRanking",
    "CandidateScore",
    "ClinicalTrial",
    "Compound",
    "CompoundFingerprint",
    "DataSource",
    "DatasetVersion",
    "Disease",
    "Evidence",
    "Experiment",
    "Gene",
    "IngestionRun",
    "Job",
    "MLModel",
    "ModelEvaluation",
    "MolecularProperty",
    "Prediction",
    "ProvenanceRecord",
    "Publication",
    "Report",
    "ResearchRun",
    "RunCounter",
    "Target",
    "TargetDiseaseAssociation",
    "ToolCall",
]
