"""
Capability-based planning for the database runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from daita.plugins import ExtensionRegistry
from daita.runtime import AccessMode, Capability
from daita.skills import SkillRuntimeEffects

from .models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbRequest,
    DbRuntimeConfig,
)

_SCHEMA_WORDS = {
    "schema",
    "schemas",
    "table",
    "tables",
    "column",
    "columns",
    "field",
    "fields",
    "foreign key",
    "relationship",
    "relationships",
}
_DATA_WORDS = {
    "count",
    "sum",
    "average",
    "avg",
    "total",
    "top",
    "list",
    "show",
    "find",
    "how many",
    "revenue",
}
_WRITE_WORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "write",
    "modify",
}
_QUALITY_WORDS = {"quality", "profile", "null", "freshness", "anomaly"}
_LINEAGE_WORDS = {"lineage", "upstream", "downstream", "impact"}
_MEMORY_WORDS = {"remember", "note", "business rule", "semantic"}
_JOIN_WORDS = {"join", "relationship", "relationships", "foreign key", "between"}
_SCHEMA_ONLY_PHRASES = {
    "schema evidence only",
    "schema only",
    "metadata only",
    "do not query rows",
    "without querying rows",
    "don't query rows",
    "no row data",
}
_SCHEMA_SEARCH_PHRASES = {
    "search the schema",
    "inspect the schema",
    "which columns",
    "which column",
    "which tables",
    "which table",
}
_CALCULATION_WORDS = {"calculate", "compute", "summarize", "aggregate"}
_COLUMN_RESOLUTION_WORDS = {
    "safest",
    "best column",
    "right column",
    "relevant column",
    "find the column",
}


@dataclass(frozen=True)
class CapabilitySelection:
    """One selected capability plus the reason it was selected."""

    capability: Capability
    reason: str


class DbIntentClassifier:
    """Small deterministic classifier for initial DB runtime contracts."""

    def classify(self, request: DbRequest) -> DbIntent:
        prompt = request.prompt.lower()
        mode = (request.mode or "").lower()
        diagnostics: dict[str, object] = {"classifier": "heuristic"}

        if mode:
            kind = _kind_from_mode(mode)
            if kind is not None:
                return DbIntent(
                    kind=kind,
                    confidence=0.95,
                    access=_access_for_intent(kind),
                    evidence_mode=_evidence_mode_for_intent(kind),
                    diagnostics={**diagnostics, "matched": f"mode:{mode}"},
                )

        if _contains_any(prompt, _MEMORY_WORDS):
            return DbIntent(
                kind=DbIntentKind.MEMORY_UPDATE,
                confidence=0.7,
                access=AccessMode.WRITE,
                evidence_mode="memory",
                diagnostics={**diagnostics, "matched": "memory_keyword"},
            )

        if _contains_any(prompt, _WRITE_WORDS):
            write_execute = any(
                word in prompt for word in ("execute", "run", "apply", "commit")
            )
            kind = (
                DbIntentKind.WRITE_EXECUTE
                if write_execute
                else DbIntentKind.WRITE_PROPOSE
            )
            return DbIntent(
                kind=kind,
                confidence=0.8,
                access=_access_for_intent(kind),
                evidence_mode=_evidence_mode_for_intent(kind),
                diagnostics={**diagnostics, "matched": "write_keyword"},
            )

        if _contains_any(prompt, _LINEAGE_WORDS):
            return DbIntent(
                kind=DbIntentKind.LINEAGE_TRACE,
                confidence=0.75,
                access=AccessMode.METADATA_READ,
                evidence_mode="lineage",
                diagnostics={**diagnostics, "matched": "lineage_keyword"},
            )

        if _contains_any(prompt, _QUALITY_WORDS):
            return DbIntent(
                kind=DbIntentKind.QUALITY_CHECK,
                confidence=0.75,
                access=AccessMode.READ,
                evidence_mode="quality",
                diagnostics={**diagnostics, "matched": "quality_keyword"},
            )

        if _is_explicit_schema_only_prompt(prompt):
            return DbIntent(
                kind=DbIntentKind.SCHEMA_QUERY,
                confidence=0.85,
                access=AccessMode.METADATA_READ,
                evidence_mode="schema",
                diagnostics={**diagnostics, "matched": "schema_only_keyword"},
            )

        if _is_schema_assisted_data_prompt(prompt):
            return DbIntent(
                kind=DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
                confidence=0.75,
                access=AccessMode.READ,
                evidence_mode="query_and_relationships",
                diagnostics={**diagnostics, "matched": "schema_assisted_data_keyword"},
            )

        if _contains_any(prompt, _JOIN_WORDS):
            return DbIntent(
                kind=DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
                confidence=0.7,
                access=AccessMode.READ,
                evidence_mode="query_and_relationships",
                diagnostics={**diagnostics, "matched": "join_keyword"},
            )

        if _contains_any(prompt, _SCHEMA_WORDS) and not _contains_any(
            prompt, _DATA_WORDS
        ):
            return DbIntent(
                kind=DbIntentKind.SCHEMA_QUERY,
                confidence=0.75,
                access=AccessMode.METADATA_READ,
                evidence_mode="schema",
                diagnostics={**diagnostics, "matched": "schema_keyword"},
            )

        if _contains_any(prompt, _DATA_WORDS) or _looks_like_question(prompt):
            return DbIntent(
                kind=DbIntentKind.DATA_QUERY,
                confidence=0.65,
                access=AccessMode.READ,
                evidence_mode="query",
                diagnostics={**diagnostics, "matched": "data_keyword"},
            )

        return DbIntent(
            kind=DbIntentKind.CONVERSATIONAL,
            confidence=0.5,
            access=AccessMode.NONE,
            evidence_mode="none",
            diagnostics={**diagnostics, "matched": "fallback"},
        )


class DbContractBuilder:
    """Build operation contracts from intents and registry capabilities."""

    def __init__(self, registry: ExtensionRegistry, config: DbRuntimeConfig) -> None:
        self.registry = registry
        self.config = config

    def build(
        self,
        request: DbRequest,
        intent: DbIntent,
        *,
        skill_effects: tuple[SkillRuntimeEffects, ...] = (),
    ) -> DbOperationContract:
        selections = self._select_capabilities(request, intent, skill_effects)
        required_evidence = _ordered_unique(
            (
                *[
                    evidence
                    for selection in selections
                    for evidence in selection.capability.output_evidence
                ],
                *[
                    evidence
                    for effect in skill_effects
                    for evidence in effect.required_evidence
                ],
            )
        )
        selected_ids = _ordered_unique(
            selection.capability.id for selection in selections
        )
        access = _max_access(
            [intent.access, *(s.capability.access for s in selections)]
        )
        policy_ids = _ordered_unique(
            (
                *self._policy_ids(intent),
                *[
                    policy_id
                    for effect in skill_effects
                    for policy_id in effect.policy_ids
                ],
            )
        )
        skill_contract_metadata = _merge_skill_metadata(
            effect.contract_metadata for effect in skill_effects
        )
        verifier_metadata = _merge_skill_metadata(
            effect.verifier_metadata for effect in skill_effects
        )
        synthesis_metadata = _merge_skill_metadata(
            effect.synthesis_metadata for effect in skill_effects
        )
        metadata = {
            "intent_kind": intent.kind.value,
            "profile": self.config.profile,
            "planned_operation": {
                "intent_kind": intent.kind.value,
                "operation_type": _operation_type_for_intent(intent.kind),
                "access": access.value,
                "admin": access is AccessMode.ADMIN
                or intent.kind is DbIntentKind.ADMIN,
                "destructive": False,
                "source": "contract_builder",
            },
            "selected_capabilities": [
                {
                    "id": selection.capability.id,
                    "owner": selection.capability.owner,
                    "executor": selection.capability.executor,
                    "reason": selection.reason,
                    "access": selection.capability.access.value,
                    "risk": selection.capability.risk.value,
                    "side_effecting": selection.capability.side_effecting,
                }
                for selection in selections
            ],
            "blocked_capabilities": self._blocked_capabilities(intent),
            "missing_capabilities": self._missing_capabilities(
                intent,
                selections,
                requested_requirements=(
                    *request.requested_capabilities,
                    *[
                        capability_id
                        for effect in skill_effects
                        for capability_id in effect.requested_capabilities
                    ],
                ),
            ),
            "approval_required": intent.kind
            in {DbIntentKind.WRITE_PROPOSE, DbIntentKind.WRITE_EXECUTE},
            "diagnostics": {
                "classifier": intent.diagnostics,
                "requested_capabilities": list(request.requested_capabilities),
                "skill_requested_capabilities": [
                    capability_id
                    for effect in skill_effects
                    for capability_id in effect.requested_capabilities
                ],
            },
            "skills": [
                {
                    "skill_id": effect.skill_id,
                    "requested_capabilities": list(effect.requested_capabilities),
                    "required_evidence": list(effect.required_evidence),
                    "policy_ids": list(effect.policy_ids),
                    "tool_view_names": list(effect.tool_view_names),
                }
                for effect in skill_effects
            ],
            "skill_contract_metadata": skill_contract_metadata,
            "skill_verifier_metadata": verifier_metadata,
            "skill_synthesis_metadata": synthesis_metadata,
        }
        metadata.update(skill_contract_metadata)
        return DbOperationContract(
            operation_type=_operation_type_for_intent(intent.kind),
            required_capabilities=tuple(selected_ids),
            required_evidence=tuple(required_evidence),
            access=access,
            limits=self.config.limits,
            policy_ids=policy_ids,
            metadata=metadata,
        )

    def _select_capabilities(
        self,
        request: DbRequest,
        intent: DbIntent,
        skill_effects: tuple[SkillRuntimeEffects, ...],
    ) -> tuple[CapabilitySelection, ...]:
        requested = set(request.requested_capabilities)
        skill_requested = {
            capability_id
            for effect in skill_effects
            for capability_id in effect.requested_capabilities
        }
        selections: list[CapabilitySelection] = []

        for capability_id in _capability_requirements_for_intent(intent.kind):
            matches = [
                capability
                for capability in self.registry.capabilities
                if capability.id == capability_id
            ]
            if not matches:
                continue
            for capability in matches:
                if _forbidden_for_intent(capability, intent):
                    continue
                selections.append(
                    CapabilitySelection(
                        capability=capability,
                        reason="required_by_intent",
                    )
                )

        for capability in self.registry.capabilities:
            if capability.id not in requested and capability.id not in skill_requested:
                continue
            if _forbidden_for_intent(capability, intent):
                continue
            selections.append(
                CapabilitySelection(
                    capability=capability,
                    reason=(
                        "skill_requested"
                        if capability.id in skill_requested
                        and capability.id not in requested
                        else "requested"
                    ),
                )
            )

        return _dedupe_selections(selections)

    def _policy_ids(self, intent: DbIntent) -> tuple[str, ...]:
        ids: list[str] = []
        if intent.kind in {DbIntentKind.WRITE_PROPOSE, DbIntentKind.WRITE_EXECUTE}:
            ids.append("runtime:approval_required_for_writes")
        return tuple(_ordered_unique(ids))

    def _blocked_capabilities(self, intent: DbIntent) -> list[dict[str, str]]:
        blocked = []
        for capability in self.registry.capabilities:
            if _forbidden_for_intent(capability, intent):
                blocked.append(
                    {
                        "id": capability.id,
                        "owner": capability.owner,
                        "reason": "forbidden_by_intent",
                    }
                )
        return blocked

    def _missing_capabilities(
        self,
        intent: DbIntent,
        selections: tuple[CapabilitySelection, ...],
        *,
        requested_requirements: tuple[str, ...] = (),
    ) -> list[str]:
        selected_ids = {selection.capability.id for selection in selections}
        return [
            capability_id
            for capability_id in _ordered_unique(
                (
                    *_capability_requirements_for_intent(intent.kind),
                    *requested_requirements,
                )
            )
            if capability_id not in selected_ids
        ]


def _kind_from_mode(mode: str) -> DbIntentKind | None:
    for kind in DbIntentKind:
        if mode == kind.value or mode == kind.name.lower():
            return kind
    aliases = {
        "schema": DbIntentKind.SCHEMA_QUERY,
        "data": DbIntentKind.DATA_QUERY,
        "write": DbIntentKind.WRITE_PROPOSE,
        "write_execute": DbIntentKind.WRITE_EXECUTE,
    }
    return aliases.get(mode)


def _operation_type_for_intent(kind: DbIntentKind) -> str:
    if kind is DbIntentKind.CATALOG_ASSISTED_DATA_QUERY:
        return DbIntentKind.DATA_QUERY.value
    return kind.value


def _capability_requirements_for_intent(kind: DbIntentKind) -> tuple[str, ...]:
    if kind is DbIntentKind.SCHEMA_QUERY:
        return ("catalog.schema.search", "catalog.asset.inspect", "db.schema.inspect")
    if kind is DbIntentKind.DATA_QUERY:
        return ("db.sql.validate", "db.sql.execute_read")
    if kind is DbIntentKind.CATALOG_ASSISTED_DATA_QUERY:
        return (
            "catalog.schema.search",
            "catalog.relationship_paths.find",
            "db.sql.validate",
            "db.sql.execute_read",
        )
    if kind is DbIntentKind.QUALITY_CHECK:
        return ("quality.profile",)
    if kind is DbIntentKind.LINEAGE_TRACE:
        return ("lineage.trace",)
    if kind is DbIntentKind.MEMORY_UPDATE:
        return ("memory.semantic.write",)
    if kind is DbIntentKind.WRITE_PROPOSE:
        return ("db.sql.validate",)
    if kind is DbIntentKind.WRITE_EXECUTE:
        return ("db.sql.validate", "db.sql.execute_write")
    return ()


def _forbidden_for_intent(capability: Capability, intent: DbIntent) -> bool:
    if intent.kind is DbIntentKind.SCHEMA_QUERY and capability.access in {
        AccessMode.READ,
        AccessMode.WRITE,
        AccessMode.ADMIN,
    }:
        return True
    if (
        intent.kind is DbIntentKind.WRITE_PROPOSE
        and capability.access is AccessMode.WRITE
    ):
        return True
    return False


def _merge_skill_metadata(values: Iterable[dict]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for value in values:
        for key, item in dict(value).items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(item, dict)
            ):
                merged[key] = {**merged[key], **item}
            else:
                merged[key] = item
    return merged


def _access_for_intent(kind: DbIntentKind) -> AccessMode:
    if kind in {
        DbIntentKind.SCHEMA_QUERY,
        DbIntentKind.LINEAGE_TRACE,
        DbIntentKind.WRITE_PROPOSE,
    }:
        return AccessMode.METADATA_READ
    if kind in {
        DbIntentKind.DATA_QUERY,
        DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
        DbIntentKind.METRIC_QUERY,
        DbIntentKind.REPORT_GENERATE,
        DbIntentKind.QUALITY_CHECK,
        DbIntentKind.ANOMALY_INVESTIGATE,
    }:
        return AccessMode.READ
    if kind in {DbIntentKind.MEMORY_UPDATE, DbIntentKind.WRITE_EXECUTE}:
        return AccessMode.WRITE
    if kind is DbIntentKind.ADMIN:
        return AccessMode.ADMIN
    return AccessMode.NONE


def _evidence_mode_for_intent(kind: DbIntentKind) -> str:
    return {
        DbIntentKind.CONVERSATIONAL: "none",
        DbIntentKind.SCHEMA_QUERY: "schema",
        DbIntentKind.DATA_QUERY: "query",
        DbIntentKind.CATALOG_ASSISTED_DATA_QUERY: "query_and_relationships",
        DbIntentKind.METRIC_QUERY: "query",
        DbIntentKind.REPORT_GENERATE: "report",
        DbIntentKind.QUALITY_CHECK: "quality",
        DbIntentKind.LINEAGE_TRACE: "lineage",
        DbIntentKind.ANOMALY_INVESTIGATE: "query_and_quality",
        DbIntentKind.MEMORY_UPDATE: "memory",
        DbIntentKind.WRITE_PROPOSE: "validation",
        DbIntentKind.WRITE_EXECUTE: "write",
        DbIntentKind.ADMIN: "admin",
    }[kind]


def _max_access(values: Iterable[AccessMode]) -> AccessMode:
    order = {
        AccessMode.NONE: 0,
        AccessMode.METADATA_READ: 1,
        AccessMode.READ: 2,
        AccessMode.WRITE: 3,
        AccessMode.ADMIN: 4,
    }
    return max(values, key=lambda value: order[AccessMode(value)])


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dedupe_selections(
    selections: Iterable[CapabilitySelection],
) -> tuple[CapabilitySelection, ...]:
    seen: set[tuple[str, str]] = set()
    out: list[CapabilitySelection] = []
    for selection in selections:
        key = (selection.capability.id, selection.capability.owner)
        if key in seen:
            continue
        seen.add(key)
        out.append(selection)
    return tuple(out)


def _contains_any(prompt: str, needles: set[str]) -> bool:
    return any(needle in prompt for needle in needles)


def _is_explicit_schema_only_prompt(prompt: str) -> bool:
    return (
        _contains_any(prompt, _SCHEMA_ONLY_PHRASES)
        or _contains_any(prompt, _SCHEMA_SEARCH_PHRASES)
    ) and _contains_any(prompt, _SCHEMA_WORDS)


def _is_schema_assisted_data_prompt(prompt: str) -> bool:
    return (
        _contains_any(prompt, _CALCULATION_WORDS)
        and _contains_any(prompt, _COLUMN_RESOLUTION_WORDS)
        and _contains_any(prompt, _DATA_WORDS)
    )


def _looks_like_question(prompt: str) -> bool:
    return bool(re.match(r"\s*(what|which|who|where|when|why|how)\b", prompt))
