"""Placeholder resolution between plan steps.

A plan is written before any data exists, so its arguments contain references like
``{{step_2.compound_ids}}`` or ``{{state.target_symbol}}``. This module turns those into
concrete values from the accumulated state.

Unresolvable placeholders are a normal occurrence -- a similarity step cannot run if no
compound was found -- so resolution reports *why* it failed and the executor skips the
step with that reason recorded, instead of calling a tool with the literal string
``"{{state.target_symbol}}"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from theragraph.core.logging import get_logger

log = get_logger(__name__)

PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")


@dataclass
class ResolutionResult:
    arguments: dict[str, Any]
    resolved: dict[str, Any] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unresolved


class BindingStore:
    """Values produced by earlier steps, addressable by placeholder path."""

    def __init__(self) -> None:
        self._by_step: dict[int, dict[str, Any]] = {}
        self._state: dict[str, Any] = {}
        self._last_step: int | None = None

    # ----------------------------------------------------------------- ingestion
    def record_step(self, step: int, tool: str, data: dict[str, Any]) -> None:
        """Extract the bindings a later step could plausibly need."""
        bindings = _extract_bindings(tool, data)
        self._by_step[step] = bindings
        self._last_step = step
        # Step bindings also refresh the rolling state, so `{{state.x}}` always means
        # "the most recent value seen for x".
        for key, value in bindings.items():
            if value is not None and value != [] and value != "":
                self._state[key] = value

    def merge_state(self, patch: dict[str, Any]) -> None:
        """Fold a tool's structured state patch into the addressable state."""
        for key, value in _flatten_state(patch).items():
            if value is not None and value != [] and value != "":
                self._state.setdefault(key, value)
                self._state[key] = value

    # ------------------------------------------------------------------ lookup
    def lookup(self, path: str) -> tuple[Any, bool]:
        """Resolve one placeholder path. Returns ``(value, found)``."""
        parts = path.split(".")
        scope = parts[0]
        key = ".".join(parts[1:]) if len(parts) > 1 else ""

        if scope == "state":
            return (self._state.get(key), key in self._state)
        if scope == "prev":
            if self._last_step is None:
                return None, False
            bindings = self._by_step.get(self._last_step, {})
            if key in bindings:
                return bindings[key], True
            return (self._state.get(key), key in self._state)
        if scope.startswith("step_"):
            try:
                index = int(scope.removeprefix("step_"))
            except ValueError:
                return None, False
            bindings = self._by_step.get(index, {})
            if key in bindings:
                return bindings[key], True
            # Fall back to rolling state: a plan may reference a field the named step
            # did not produce but a later one did.
            return (self._state.get(key), key in self._state)
        return (self._state.get(path), path in self._state)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._state)


def resolve_arguments(arguments: dict[str, Any], store: BindingStore) -> ResolutionResult:
    """Substitute placeholders in a step's arguments."""
    result = ResolutionResult(arguments={})

    for name, value in arguments.items():
        resolved, unresolved = _resolve_value(value, store)
        if unresolved:
            result.unresolved.extend(f"{name}={item}" for item in unresolved)
            continue
        if resolved is None:
            continue
        result.arguments[name] = resolved
        if isinstance(value, str) and PLACEHOLDER_RE.search(value):
            result.resolved[name] = resolved

    return result


def _resolve_value(value: Any, store: BindingStore) -> tuple[Any, list[str]]:
    if isinstance(value, str):
        matches = PLACEHOLDER_RE.findall(value)
        if not matches:
            return value, []
        # A string that is exactly one placeholder resolves to the raw value (possibly a
        # list); a string with surrounding text resolves by interpolation.
        if len(matches) == 1 and PLACEHOLDER_RE.fullmatch(value.strip()):
            resolved, found = store.lookup(matches[0])
            return (resolved, []) if found else (None, [matches[0]])

        missing: list[str] = []
        interpolated = value
        for path in matches:
            resolved, found = store.lookup(path)
            if not found:
                missing.append(path)
                continue
            interpolated = interpolated.replace("{{" + path + "}}", str(resolved))
        return (interpolated, missing) if not missing else (None, missing)

    if isinstance(value, list):
        out: list[Any] = []
        missing = []
        for item in value:
            resolved, item_missing = _resolve_value(item, store)
            missing.extend(item_missing)
            if resolved is not None:
                out.extend(resolved) if isinstance(resolved, list) else out.append(resolved)
        return (out, missing) if not missing else (None, missing)

    if isinstance(value, dict):
        out_dict: dict[str, Any] = {}
        missing = []
        for key, item in value.items():
            resolved, item_missing = _resolve_value(item, store)
            missing.extend(item_missing)
            if resolved is not None:
                out_dict[key] = resolved
        return (out_dict, missing) if not missing else (None, missing)

    return value, []


def _extract_bindings(tool: str, data: dict[str, Any]) -> dict[str, Any]:
    """Pull the useful handles out of a tool's payload.

    Deliberately explicit per tool rather than a generic deep search: a plan that says
    ``{{step_2.compound_ids}}`` must get compound ids, not the first list of strings that
    happens to appear in the payload.
    """
    bindings: dict[str, Any] = {}

    if tool == "search_entities":
        hits = data.get("hits") or []
        diseases = [hit for hit in hits if hit["entity_type"] == "disease"]
        targets = [hit for hit in hits if hit["entity_type"] == "target"]
        compounds = [hit for hit in hits if hit["entity_type"] == "compound"]
        if diseases:
            bindings["top_disease"] = diseases[0]["external_id"]
            bindings["disease_efo_id"] = diseases[0]["external_id"]
            bindings["disease_name"] = diseases[0]["name"]
        if targets:
            bindings["top_target"] = (
                targets[0]["metadata"].get("symbol") or targets[0]["external_id"]
            )
            bindings["target_symbol"] = bindings["top_target"]
        if compounds:
            bindings["top_compound"] = compounds[0]["external_id"]
            bindings["compound_ids"] = [hit["id"] for hit in compounds]

    elif tool == "get_disease_targets":
        disease = data.get("disease") or {}
        targets = data.get("associated_targets") or []
        bindings["disease_efo_id"] = disease.get("efo_id")
        bindings["disease_name"] = disease.get("name")
        bindings["disease_id"] = disease.get("id")
        if targets:
            leading = targets[0]["target"]
            bindings["top_target"] = leading.get("symbol")
            bindings["target_symbol"] = leading.get("symbol")
            bindings["target_key"] = leading.get("target_key")
            bindings["target_ids"] = [item["target"]["id"] for item in targets]
            bindings["target_symbols"] = [item["target"]["symbol"] for item in targets]
            # A target with small-molecule chemistry is the useful next step; prefer it
            # over the top-scoring target when they differ.
            tractable = [
                item["target"]["symbol"]
                for item in targets
                if _is_small_molecule_tractable(item["target"].get("tractability"))
            ]
            if tractable:
                bindings["tractable_target"] = tractable[0]
                bindings["target_symbol"] = tractable[0]
                bindings["top_target"] = tractable[0]

    elif tool == "get_target_overview":
        target = data.get("target") or {}
        compounds = data.get("known_compounds") or []
        bindings["target_symbol"] = target.get("symbol")
        bindings["target_key"] = target.get("target_key")
        bindings["target_id"] = target.get("id")
        if compounds:
            bindings["candidate_compound_ids"] = [item["compound_id"] for item in compounds]
            bindings["compound_ids"] = bindings["candidate_compound_ids"]
            best = compounds[0]
            bindings["best_compound_smiles"] = best.get("smiles")
            bindings["best_compound_id"] = best.get("compound_id")
            bindings["smiles"] = best.get("smiles")

    elif tool == "get_compound":
        compound = data.get("compound") or {}
        bindings["smiles"] = compound.get("smiles")
        bindings["compound_id"] = compound.get("id")
        bindings["compound_ids"] = [compound.get("id")] if compound.get("id") else []
        bindings["best_compound_smiles"] = compound.get("smiles")

    elif tool == "find_similar_compounds":
        hits = data.get("hits") or []
        if hits:
            bindings["compound_ids"] = [hit["compound_id"] for hit in hits]
            bindings["similar_compound_ids"] = bindings["compound_ids"]
            bindings["candidate_compound_ids"] = bindings["compound_ids"]

    elif tool == "substructure_search":
        matches = data.get("matches") or []
        if matches:
            bindings["compound_ids"] = [item["compound_id"] for item in matches]
            bindings["candidate_compound_ids"] = bindings["compound_ids"]

    elif tool == "predict_properties":
        predictions = data.get("predictions") or []
        bindings["predicted_endpoints"] = sorted(
            {prediction.get("endpoint") for prediction in predictions if prediction.get("endpoint")}
        )

    elif tool == "prioritize_candidates":
        candidates = data.get("candidates") or []
        if candidates:
            bindings["ranked_compound_ids"] = [candidate["compound_id"] for candidate in candidates]
            bindings["top_candidate_id"] = candidates[0]["compound_id"]
            bindings["top_candidate_ids"] = [
                candidate["compound_id"] for candidate in candidates[:6]
            ]

    return {key: value for key, value in bindings.items() if value is not None}


def _is_small_molecule_tractable(tractability: Any) -> bool:
    """Does Open Targets report *any* small-molecule tractability bucket?"""
    if not isinstance(tractability, dict):
        return False
    for modality, value in tractability.items():
        if modality.startswith("_"):
            continue
        if "small molecule" in modality.lower() and value:
            return True
    return False


def _flatten_state(patch: dict[str, Any]) -> dict[str, Any]:
    """Derive addressable bindings from a tool's state patch."""
    flat: dict[str, Any] = {}

    disease = patch.get("disease")
    if isinstance(disease, dict):
        flat["disease_efo_id"] = disease.get("efo_id")
        flat["disease_name"] = disease.get("name")
        flat["disease_id"] = disease.get("id")

    targets = patch.get("targets")
    if isinstance(targets, list) and targets:
        flat["target_symbols"] = [item.get("symbol") for item in targets if item.get("symbol")]
        flat["target_symbol"] = flat["target_symbols"][0] if flat["target_symbols"] else None

    selected = patch.get("selected_target")
    if isinstance(selected, dict):
        flat["target_symbol"] = selected.get("symbol") or flat.get("target_symbol")
        flat["target_key"] = selected.get("target_key")
        flat["target_id"] = selected.get("id")

    known = patch.get("known_compounds")
    if isinstance(known, list) and known:
        flat["candidate_compound_ids"] = [
            item.get("compound_id") for item in known if item.get("compound_id")
        ]
        flat["best_compound_smiles"] = known[0].get("smiles")

    similar = patch.get("similar_compounds")
    if isinstance(similar, list) and similar:
        existing = flat.get("candidate_compound_ids") or []
        flat["candidate_compound_ids"] = list(
            dict.fromkeys([*existing, *(item.get("compound_id") for item in similar)])
        )

    return {key: value for key, value in flat.items() if value}
