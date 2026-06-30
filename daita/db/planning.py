"""
Capability-based planning for the database runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

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

_SCHEMA_OBJECT_SIGNALS = (
    ("schema", r"\bschemas?\b", 3),
    ("database_structure", r"\bdatabase\s+(structure|schema|contain|contains)\b", 4),
    ("available_data", r"\bavailable\s+(tables?|data|datasets?)\b", 3),
    ("tables", r"\btables?\b", 2),
    ("columns", r"\bcolumns?\b", 2),
    ("fields", r"\bfields?\b", 2),
    ("keys", r"\b(primary|foreign)\s+keys?\b", 2),
    ("entities", r"\bentities\b", 2),
    ("database_contains", r"\bwhat\s+(does|is)\s+this\s+database\s+contain", 4),
    ("business_questions", r"\bbusiness\s+questions?\b", 3),
    ("important_fields", r"\bimportant\s+fields?\b", 3),
)
_RELATIONSHIP_OBJECT_SIGNALS = (
    ("relationships", r"\brelationships?\b", 4),
    ("foreign_keys", r"\bforeign\s+keys?\b", 4),
    ("join_path", r"\bjoin\s+paths?\b", 4),
    ("join_relationship", r"\bjoin\b", 2),
    ("connected", r"\bconnected\b", 3),
    ("linked", r"\blink(?:ed|s)?\b", 3),
    ("related_tables", r"\brelated\s+tables?\b", 3),
    ("tables_connect", r"\bhow\s+tables?\s+connect\b", 4),
    ("connect_to", r"\bconnect(?:ed|s)?\s+to\b", 3),
)
_DATA_ACCESS_SIGNALS = (
    ("neutral_data_verb", r"\b(list|show|find)\b", 2),
    ("count_rows", r"\b(count|how\s+many|number\s+of)\b", 4),
    ("totals", r"\b(total|sum)\b", 3),
    ("averages", r"\b(avg|average|mean)\b", 3),
    ("top_n", r"\btop\s+\d+\b|\btop\b", 4),
    (
        "most_recent_records",
        r"\b(most\s+recent|latest)\s+(records?|rows?|orders?|transactions?)\b",
        4,
    ),
    (
        "filter_records",
        r"\b(where|filter|over|under|greater\s+than|less\s+than|threshold)\b",
        3,
    ),
    (
        "group_by",
        r"\b(group\s+by|by\s+(month|week|day|period|account|customer|region|status))\b",
        3,
    ),
    ("record_rows", r"\b(records?|rows?)\b", 3),
    ("transactions", r"\btransactions?\b", 2),
    (
        "revenue_by_period",
        r"\brevenue\s+by\b|\bmonthly\s+revenue\b|\brevenue\s+by\s+(month|period|account|customer)\b",
        4,
    ),
    ("calculate_metric", r"\b(calculate|compute)\b", 3),
    (
        "analysis_request",
        r"\b(multi-step|multiple\s+queries|deep\s+analysis|investigate|explain\s+why|why\s+did|root\s+cause|step\s+by\s+step\s+analysis)\b",
        4,
    ),
)
_DATA_RESOLUTION_SIGNALS = (
    ("best_table", r"\b(best|right|appropriate|relevant)\s+tables?\b", 3),
    ("best_column", r"\b(best|right|appropriate|relevant|safest)\s+columns?\b", 3),
    (
        "qualified_column",
        r"\b(best|right|appropriate|relevant|safest)\s+\w+\s+columns?\b",
        3,
    ),
    ("find_column", r"\bfind\s+the\s+columns?\b", 3),
    ("which_column", r"\bwhich\s+columns?\b", 2),
    ("which_table", r"\bwhich\s+tables?\b", 2),
)
_NEUTRAL_VERB_SIGNALS = (
    ("summarize", r"\bsummarize\b"),
    ("describe", r"\bdescribe\b"),
    ("list", r"\blist\b"),
    ("show", r"\bshow\b"),
    ("find", r"\bfind\b"),
    ("map", r"\bmap\b"),
    ("identify", r"\bidentify\b"),
    ("inspect", r"\binspect\b"),
)
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
_MEMORY_WORDS = {
    "remember",
    "note",
    "business rule",
    "semantic",
    "forget",
    "memory",
}
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
_DATA_RESULT_WORDS = {"records", "rows", "values", "result", "results", "return"}


@dataclass(frozen=True)
class CapabilitySelection:
    """One selected capability plus the reason it was selected."""

    capability: Capability
    reason: str


@dataclass(frozen=True)
class DbSafetyFrame:
    """Explicit, non-semantic boundary for planner-proposed DB actions."""

    max_access: AccessMode = AccessMode.ADMIN
    source_scope: tuple[str, ...] = ()
    explicit_mode: str | None = None
    requested_capabilities: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...] = ()
    denied_capabilities: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    runtime_limits: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_access", AccessMode(self.max_access))
        object.__setattr__(
            self, "source_scope", tuple(str(item) for item in self.source_scope)
        )
        object.__setattr__(
            self,
            "requested_capabilities",
            tuple(str(item) for item in self.requested_capabilities),
        )
        object.__setattr__(
            self,
            "allowed_capabilities",
            tuple(str(item) for item in self.allowed_capabilities),
        )
        object.__setattr__(
            self,
            "denied_capabilities",
            tuple(str(item) for item in self.denied_capabilities),
        )
        object.__setattr__(self, "constraints", dict(self.constraints))
        object.__setattr__(self, "runtime_limits", dict(self.runtime_limits))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_access": self.max_access.value,
            "source_scope": list(self.source_scope),
            "explicit_mode": self.explicit_mode,
            "requested_capabilities": list(self.requested_capabilities),
            "allowed_capabilities": list(self.allowed_capabilities),
            "denied_capabilities": list(self.denied_capabilities),
            "constraints": self.constraints,
            "runtime_limits": self.runtime_limits,
            "diagnostics": self.diagnostics,
        }


def build_safety_frame(
    registry: ExtensionRegistry,
    config: DbRuntimeConfig,
    request: DbRequest,
) -> DbSafetyFrame:
    """Build a planner boundary from explicit runtime facts only."""

    constraints = dict(request.constraints)
    max_access = _explicit_max_access(request, constraints)
    allowed_capabilities = _explicit_capability_tuple(
        constraints.get("allowed_capabilities")
        or constraints.get("capability_allowlist")
    )
    denied_capabilities = _explicit_capability_tuple(
        constraints.get("denied_capabilities") or constraints.get("capability_denylist")
    )
    requested_capabilities = tuple(request.requested_capabilities)
    missing_requested = tuple(
        capability_id
        for capability_id in requested_capabilities
        if not any(
            capability.id == capability_id for capability in registry.capabilities
        )
    )
    return DbSafetyFrame(
        max_access=max_access,
        source_scope=request.source_scope,
        explicit_mode=request.mode,
        requested_capabilities=requested_capabilities,
        allowed_capabilities=allowed_capabilities,
        denied_capabilities=denied_capabilities,
        constraints=constraints,
        runtime_limits=config.limits.to_dict(),
        diagnostics={
            "source": "explicit_runtime_facts",
            "missing_requested_capabilities": list(missing_requested),
        },
    )


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

        if request.metadata.get("sql") or request.metadata.get("query"):
            return DbIntent(
                kind=DbIntentKind.DATA_QUERY,
                confidence=0.9,
                access=AccessMode.READ,
                evidence_mode="query",
                diagnostics={
                    **diagnostics,
                    "matched": "explicit_sql_metadata",
                    "schema_score": 0,
                    "relationship_score": 0,
                    "data_score": 4,
                    "dominant_object_type": "data",
                    "neutral_verbs": _matched_signal_names(
                        prompt, _NEUTRAL_VERB_SIGNALS
                    ),
                    "data_access_requested": True,
                    "tie_break_reason": "explicit_sql_metadata",
                },
            )

        signal = _intent_signal_scores(prompt)
        diagnostics = {**diagnostics, **signal}
        if _is_explicit_schema_only_prompt(prompt):
            relationship_kind = (
                DbIntentKind.SCHEMA_RELATIONSHIP_QUERY
                if signal["relationship_score"] > signal["schema_score"]
                else DbIntentKind.SCHEMA_QUERY
            )
            return DbIntent(
                kind=relationship_kind,
                confidence=0.9,
                access=_access_for_intent(relationship_kind),
                evidence_mode=_evidence_mode_for_intent(relationship_kind),
                diagnostics={**diagnostics, "matched": "explicit_metadata_only"},
            )

        kind, confidence, reason = _kind_from_signal_scores(signal)
        if kind is not None:
            return DbIntent(
                kind=kind,
                confidence=confidence,
                access=_access_for_intent(kind),
                evidence_mode=_evidence_mode_for_intent(kind),
                diagnostics={**diagnostics, "matched": reason},
            )

        if _looks_like_question(prompt):
            return DbIntent(
                kind=DbIntentKind.DATA_QUERY,
                confidence=0.55,
                access=AccessMode.READ,
                evidence_mode="query",
                diagnostics={
                    **diagnostics,
                    "matched": "question_fallback",
                    "tie_break_reason": "question_without_metadata_signals_defaults_to_data",
                },
            )

        return DbIntent(
            kind=DbIntentKind.CONVERSATIONAL,
            confidence=0.5,
            access=AccessMode.NONE,
            evidence_mode="none",
            diagnostics={**diagnostics, "matched": "fallback"},
        )


def classify_db_request(request: DbRequest) -> DbIntent:
    """Compatibility helper for direct contract-building APIs."""

    return DbIntentClassifier().classify(request)


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
        "relationships": DbIntentKind.SCHEMA_RELATIONSHIP_QUERY,
        "relationship": DbIntentKind.SCHEMA_RELATIONSHIP_QUERY,
        "schema_relationship": DbIntentKind.SCHEMA_RELATIONSHIP_QUERY,
        "data": DbIntentKind.DATA_QUERY,
        "write": DbIntentKind.WRITE_PROPOSE,
        "write_execute": DbIntentKind.WRITE_EXECUTE,
    }
    return aliases.get(mode)


def _explicit_max_access(
    request: DbRequest,
    constraints: dict[str, Any],
) -> AccessMode:
    explicit = constraints.get("max_access") or constraints.get("max_allowed_access")
    if explicit is not None:
        try:
            return AccessMode(str(explicit))
        except ValueError:
            return AccessMode.ADMIN
    mode = (request.mode or "").lower()
    if mode in {"schema", "schema.query", "relationships", "relationship"}:
        return AccessMode.METADATA_READ
    if mode in {"data", "data.query", "query", "read"}:
        return AccessMode.READ
    if mode in {"write", "write.propose", "write.execute", "memory.update"}:
        return AccessMode.WRITE
    if mode == "admin":
        return AccessMode.ADMIN
    return AccessMode.ADMIN


def _explicit_capability_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _operation_type_for_intent(kind: DbIntentKind) -> str:
    if kind is DbIntentKind.CATALOG_ASSISTED_DATA_QUERY:
        return DbIntentKind.DATA_QUERY.value
    return kind.value


def _capability_requirements_for_intent(kind: DbIntentKind) -> tuple[str, ...]:
    if kind is DbIntentKind.SCHEMA_QUERY:
        return ("catalog.schema.search", "catalog.asset.inspect", "db.schema.inspect")
    if kind is DbIntentKind.SCHEMA_RELATIONSHIP_QUERY:
        return (
            "db.schema.inspect",
            "catalog.schema.search",
            "catalog.relationship_paths.find",
        )
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
        return ("db.memory.plan_update", "db.memory.commit_update")
    if kind is DbIntentKind.WRITE_PROPOSE:
        return ("db.sql.validate",)
    if kind is DbIntentKind.WRITE_EXECUTE:
        return ("db.sql.validate", "db.sql.execute_write")
    return ()


def _forbidden_for_intent(capability: Capability, intent: DbIntent) -> bool:
    if intent.kind in {
        DbIntentKind.SCHEMA_QUERY,
        DbIntentKind.SCHEMA_RELATIONSHIP_QUERY,
    } and capability.access in {
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
        DbIntentKind.SCHEMA_RELATIONSHIP_QUERY,
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
        DbIntentKind.SCHEMA_RELATIONSHIP_QUERY: "schema_relationships",
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
    ) and (
        _score_signal_patterns(prompt, _SCHEMA_OBJECT_SIGNALS)[0] > 0
        or _score_signal_patterns(prompt, _RELATIONSHIP_OBJECT_SIGNALS)[0] > 0
    )


def _intent_signal_scores(prompt: str) -> dict[str, object]:
    schema_score, schema_matches = _score_signal_patterns(
        prompt, _SCHEMA_OBJECT_SIGNALS
    )
    relationship_score, relationship_matches = _score_signal_patterns(
        prompt, _RELATIONSHIP_OBJECT_SIGNALS
    )
    data_score, data_matches = _score_signal_patterns(prompt, _DATA_ACCESS_SIGNALS)
    resolution_score, resolution_matches = _score_signal_patterns(
        prompt, _DATA_RESOLUTION_SIGNALS
    )
    neutral_verbs = _matched_signal_names(prompt, _NEUTRAL_VERB_SIGNALS)
    if "join_relationship" in relationship_matches and _has_data_result_object(prompt):
        data_score += 3
        data_matches.append("join_returns_data")
    if (
        data_matches
        and set(data_matches) <= {"neutral_data_verb"}
        and (schema_score > 0 or relationship_score > 0)
    ):
        data_score = 0
        data_matches = []
    dominant = "none"
    ranked = sorted(
        (
            ("schema", schema_score),
            ("relationship", relationship_score),
            ("data", data_score),
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if ranked[0][1] > 0:
        dominant = ranked[0][0]
    return {
        "schema_score": schema_score,
        "relationship_score": relationship_score,
        "data_score": data_score,
        "resolution_score": resolution_score,
        "dominant_object_type": dominant,
        "neutral_verbs": neutral_verbs,
        "schema_signals": schema_matches,
        "relationship_signals": relationship_matches,
        "data_signals": data_matches,
        "resolution_signals": resolution_matches,
        "data_access_requested": data_score > 0,
        "tie_break_reason": "none",
    }


def _kind_from_signal_scores(
    signal: dict[str, object],
) -> tuple[DbIntentKind | None, float, str]:
    schema_score = int(signal["schema_score"])
    relationship_score = int(signal["relationship_score"])
    data_score = int(signal["data_score"])
    resolution_score = int(signal["resolution_score"])
    weak_data_only = _weak_data_signal_only(signal)
    metadata_score = max(schema_score, relationship_score)
    if weak_data_only and metadata_score > 0:
        data_score = 0
    if data_score > 0 and (metadata_score > 0 or resolution_score > 0):
        if relationship_score > 0 or resolution_score > 0:
            signal["tie_break_reason"] = "data_access_with_catalog_resolution"
            return (
                DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
                0.8,
                "scored_catalog_assisted_data",
            )
        signal["tie_break_reason"] = "data_access_with_schema_terms"
        return DbIntentKind.DATA_QUERY, 0.7, "scored_data"
    if data_score > 0:
        signal["tie_break_reason"] = "data_access_requested"
        return DbIntentKind.DATA_QUERY, 0.75, "scored_data"
    if relationship_score > 0:
        signal["tie_break_reason"] = "relationship_object_without_row_access"
        return (
            DbIntentKind.SCHEMA_RELATIONSHIP_QUERY,
            0.8,
            "scored_schema_relationship",
        )
    if schema_score > 0:
        signal["tie_break_reason"] = "schema_object_without_row_access"
        return DbIntentKind.SCHEMA_QUERY, 0.8, "scored_schema"
    return None, 0.5, "fallback"


def _weak_data_signal_only(signal: dict[str, object]) -> bool:
    data_signals = set(signal.get("data_signals") or [])
    return bool(data_signals) and data_signals <= {"neutral_data_verb"}


def _score_signal_patterns(
    prompt: str,
    patterns: tuple[tuple[str, str, int], ...],
) -> tuple[int, list[str]]:
    score = 0
    matches: list[str] = []
    for name, pattern, weight in patterns:
        if re.search(pattern, prompt):
            score += weight
            matches.append(name)
    return score, matches


def _matched_signal_names(
    prompt: str,
    patterns: tuple[tuple[str, str], ...],
) -> list[str]:
    return [name for name, pattern in patterns if re.search(pattern, prompt)]


def _has_data_result_object(prompt: str) -> bool:
    words = set(re.findall(r"[a-z_]+", prompt))
    return bool(words.intersection(_DATA_RESULT_WORDS))


def _looks_like_question(prompt: str) -> bool:
    return bool(re.match(r"\s*(what|which|who|where|when|why|how)\b", prompt))
