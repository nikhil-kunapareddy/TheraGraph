"""Data-quality checks applied to every normalised batch.

Public biomedical data is messy in specific, recurring ways: missing identifiers,
unparseable structures, activity values with no units, scores outside their documented
range. Rather than discovering that during analysis, each ingestion asserts a set of
rules and stores the resulting report on the dataset version, so the UI can show
"ChEMBL activities 37: 4.1% rejected (missing units)" next to the data itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Row = dict[str, Any]


@dataclass(frozen=True)
class QualityRule:
    """One assertion over a batch of rows."""

    name: str
    description: str
    predicate: Callable[[Row], bool]
    severity: str = "error"
    """``error`` rejects the row; ``warning`` keeps it but records the issue."""


@dataclass
class DataQualityReport:
    total_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    rule_failures: dict[str, int] = field(default_factory=dict)
    field_completeness: dict[str, float] = field(default_factory=dict)
    duplicate_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Fraction of rows that passed every error-severity rule (0-1)."""
        if not self.total_rows:
            return 1.0
        return round(self.accepted_rows / self.total_rows, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "rejection_rate": round(self.rejected_rows / self.total_rows, 4)
            if self.total_rows
            else 0.0,
            "score": self.score,
            "rule_failures": self.rule_failures,
            "field_completeness": self.field_completeness,
            "duplicate_count": self.duplicate_count,
            "warnings": self.warnings[:20],
        }


def evaluate_quality(
    rows: list[Row],
    rules: list[QualityRule],
    *,
    completeness_fields: list[str] | None = None,
    dedup_key: str | None = None,
) -> tuple[list[Row], list[Row], DataQualityReport]:
    """Partition rows into accepted/rejected and build the quality report.

    Returns ``(accepted, rejected, report)``. Rejected rows carry a ``_quality_errors``
    key listing which rules they failed, so the ingestion run can store actionable
    samples instead of a bare count.
    """
    report = DataQualityReport(total_rows=len(rows))
    accepted: list[Row] = []
    rejected: list[Row] = []
    seen: set[Any] = set()

    for row in rows:
        failures: list[str] = []
        for rule in rules:
            try:
                passed = rule.predicate(row)
            except Exception:
                passed = False
            if not passed:
                report.rule_failures[rule.name] = report.rule_failures.get(rule.name, 0) + 1
                if rule.severity == "error":
                    failures.append(rule.name)
                else:
                    report.warnings.append(f"{rule.name}: {rule.description}")

        if failures:
            rejected.append({**row, "_quality_errors": failures})
            continue

        if dedup_key:
            key = row.get(dedup_key)
            if key is not None and key in seen:
                report.duplicate_count += 1
                continue
            if key is not None:
                seen.add(key)

        accepted.append(row)

    report.accepted_rows = len(accepted)
    report.rejected_rows = len(rejected)

    for field_name in completeness_fields or []:
        if not rows:
            report.field_completeness[field_name] = 0.0
            continue
        present = sum(1 for row in rows if row.get(field_name) not in (None, "", [], {}))
        report.field_completeness[field_name] = round(present / len(rows), 4)

    return accepted, rejected, report


# ------------------------------------------------------------------ common rules
def required(field_name: str, severity: str = "error") -> QualityRule:
    return QualityRule(
        name=f"required:{field_name}",
        description=f"{field_name} must be present and non-empty",
        predicate=lambda row: row.get(field_name) not in (None, "", [], {}),
        severity=severity,
    )


def in_range(field_name: str, low: float, high: float, *, allow_null: bool = True) -> QualityRule:
    def predicate(row: Row) -> bool:
        value = row.get(field_name)
        if value is None:
            return allow_null
        try:
            return low <= float(value) <= high
        except (TypeError, ValueError):
            return False

    return QualityRule(
        name=f"range:{field_name}",
        description=f"{field_name} must be within [{low}, {high}]",
        predicate=predicate,
    )


def matches_prefix(field_name: str, prefix: str) -> QualityRule:
    def predicate(row: Row) -> bool:
        value = row.get(field_name)
        return isinstance(value, str) and value.startswith(prefix)

    return QualityRule(
        name=f"prefix:{field_name}",
        description=f"{field_name} must start with {prefix!r}",
        predicate=predicate,
    )


def max_length(field_name: str, limit: int) -> QualityRule:
    def predicate(row: Row) -> bool:
        value = row.get(field_name)
        return value is None or len(str(value)) <= limit

    return QualityRule(
        name=f"max_length:{field_name}",
        description=f"{field_name} must be at most {limit} characters",
        predicate=predicate,
        severity="warning",
    )
