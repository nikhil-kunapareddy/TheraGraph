"""Turn a compound's structure into the derived rows the platform stores.

Descriptors and fingerprints are *derived* data. They are recomputed (not migrated)
whenever the descriptor set or fingerprint definition changes, and each row records the
version that produced it, so a model trained on descriptor set v1 is never silently fed
v2 features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theragraph.chem.descriptors import DESCRIPTOR_SET_VERSION, compute_descriptors
from theragraph.chem.fingerprints import compute_fingerprint
from theragraph.chem.molecule import parse_smiles
from theragraph.core.errors import ChemistryError
from theragraph.core.logging import get_logger
from theragraph.core.metrics import FEATURIZATION_FAILURES

log = get_logger(__name__)

#: Fingerprints computed for every compound at ingestion time. Morgan drives similarity
#: search and the baseline models; the pattern fingerprint is the substructure screen.
DEFAULT_FINGERPRINT_KINDS = ("morgan_r2_2048", "rdkit_pattern_2048", "maccs_167")


@dataclass
class FeaturizedCompound:
    compound_id: str
    property_row: dict[str, Any]
    fingerprint_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FeaturizationResult:
    properties: list[dict[str, Any]] = field(default_factory=list)
    fingerprints: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return len(self.properties)


def featurize(
    compound_id: str,
    smiles: str,
    *,
    fingerprint_kinds: tuple[str, ...] = DEFAULT_FINGERPRINT_KINDS,
) -> FeaturizedCompound:
    """Compute descriptors + fingerprints for one compound."""
    mol = parse_smiles(smiles)
    descriptors = compute_descriptors(mol)
    property_row = {"compound_id": compound_id, **descriptors.to_row()}
    property_row["descriptor_set_version"] = DESCRIPTOR_SET_VERSION

    fingerprint_rows = [
        compute_fingerprint(mol, kind).to_row(compound_id) for kind in fingerprint_kinds
    ]
    return FeaturizedCompound(
        compound_id=compound_id,
        property_row=property_row,
        fingerprint_rows=fingerprint_rows,
    )


def featurize_many(
    compounds: list[tuple[str, str]],
    *,
    fingerprint_kinds: tuple[str, ...] = DEFAULT_FINGERPRINT_KINDS,
) -> FeaturizationResult:
    """Featurise ``(compound_id, smiles)`` pairs, collecting per-molecule failures.

    One bad structure must not abort a 5000-compound backfill, so failures are recorded
    and reported rather than raised.
    """
    result = FeaturizationResult()
    for compound_id, smiles in compounds:
        try:
            featurized = featurize(compound_id, smiles, fingerprint_kinds=fingerprint_kinds)
        except ChemistryError as exc:
            FEATURIZATION_FAILURES.labels(reason=exc.code).inc()
            result.failures.append(
                {"compound_id": compound_id, "error": exc.message, "code": exc.code}
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive
            FEATURIZATION_FAILURES.labels(reason="unexpected").inc()
            result.failures.append(
                {"compound_id": compound_id, "error": str(exc), "code": "unexpected"}
            )
            continue
        result.properties.append(featurized.property_row)
        result.fingerprints.extend(featurized.fingerprint_rows)

    if result.failures:
        log.warning(
            "featurization.partial_failure",
            succeeded=result.succeeded,
            failed=len(result.failures),
        )
    return result
