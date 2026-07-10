"""Typed payloads and helpers for DB multi-step analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping
from uuid import uuid4

from daita.runtime import Evidence

CAPABILITY_ANALYSIS_STEP_CONTRACTS = {
    "catalog_search": {
        "owners": frozenset({"catalog"}),
        "capabilities": frozenset(
            {
                "catalog.schema.search",
                "catalog.asset.inspect",
                "catalog.relationship_paths.find",
            }
        ),
        "evidence": frozenset(
            {"schema.search_result", "schema.asset_profile", "schema.relationship_path"}
        ),
    },
    "quality_profile": {
        "owners": frozenset({"data_quality"}),
        "capabilities": frozenset({"quality.profile"}),
        "evidence": frozenset({"quality.profile"}),
    },
    "lineage_trace": {
        "owners": frozenset({"lineage"}),
        "capabilities": frozenset({"lineage.trace"}),
        "evidence": frozenset({"lineage.trace"}),
    },
    "memory_recall": {
        "owners": frozenset({"memory"}),
        "capabilities": frozenset({"memory.semantic.recall", "memory.fact.query"}),
        "evidence": frozenset({"memory.semantic.recall", "memory.fact.query"}),
    },
}
ANALYSIS_STEP_KINDS = frozenset(
    {"query", "checkpoint", "synthesis", *CAPABILITY_ANALYSIS_STEP_CONTRACTS}
)
_RegisteredCapabilities = set[str] | Mapping[str | tuple[str | None, str], Any] | None


@dataclass(frozen=True)
class DbAnalysisStepBudgets:
    """Per-step limits for Phase 3 analysis tasks."""

    max_rows: int = 200
    max_repairs: int = 1
    max_context_chars: int = 12000

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DbAnalysisStepBudgets":
        data = dict(value or {})
        return cls(
            max_rows=_positive_int(data.get("max_rows"), 200),
            max_repairs=_non_negative_int(data.get("max_repairs"), 1),
            max_context_chars=_positive_int(data.get("max_context_chars"), 12000),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_rows": self.max_rows,
            "max_repairs": self.max_repairs,
            "max_context_chars": self.max_context_chars,
        }


@dataclass(frozen=True)
class DbAnalysisBudgets:
    """Operation-level budgets for Phase 3 analysis."""

    max_steps: int = 6
    max_query_steps: int = 3
    max_checkpoint_steps: int = 3
    max_repairs: int = 1
    max_total_rows: int = 1000
    max_llm_calls: int = 6
    max_context_chars: int = 16000
    max_duration_seconds: int = 120

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DbAnalysisBudgets":
        data = dict(value or {})
        return cls(
            max_steps=_positive_int(data.get("max_steps"), 6),
            max_query_steps=_positive_int(data.get("max_query_steps"), 3),
            max_checkpoint_steps=_non_negative_int(data.get("max_checkpoint_steps"), 3),
            max_repairs=_non_negative_int(data.get("max_repairs"), 1),
            max_total_rows=_positive_int(data.get("max_total_rows"), 1000),
            max_llm_calls=_positive_int(data.get("max_llm_calls"), 6),
            max_context_chars=_positive_int(data.get("max_context_chars"), 16000),
            max_duration_seconds=_positive_int(data.get("max_duration_seconds"), 120),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_steps": self.max_steps,
            "max_query_steps": self.max_query_steps,
            "max_checkpoint_steps": self.max_checkpoint_steps,
            "max_repairs": self.max_repairs,
            "max_total_rows": self.max_total_rows,
            "max_llm_calls": self.max_llm_calls,
            "max_context_chars": self.max_context_chars,
            "max_duration_seconds": self.max_duration_seconds,
        }


@dataclass(frozen=True)
class DbAnalysisStep:
    """One declarative step in an analysis DAG."""

    id: str
    kind: str
    purpose: str
    depends_on: tuple[str, ...] = ()
    input_refs: tuple[dict[str, Any], ...] = ()
    expected_evidence: tuple[str, ...] = ()
    capability_id: str | None = None
    capability_owner: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    context_evidence_refs: tuple[dict[str, Any], ...] = ()
    budgets: DbAnalysisStepBudgets = field(default_factory=DbAnalysisStepBudgets)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DbAnalysisStep":
        data = dict(value)
        step_id = str(data.get("id") or "").strip()
        if not step_id:
            raise ValueError("analysis_step_id_required")
        kind = str(data.get("kind") or "").strip()
        purpose = str(data.get("purpose") or "").strip()
        if not purpose:
            raise ValueError(f"analysis_step_purpose_required:{step_id}")
        raw_input = data.get("input")
        return cls(
            id=step_id,
            kind=kind,
            purpose=purpose,
            depends_on=_string_tuple(data.get("depends_on")),
            input_refs=tuple(
                dict(item)
                for item in data.get("input_refs") or ()
                if isinstance(item, Mapping)
            ),
            expected_evidence=_string_tuple(data.get("expected_evidence")),
            capability_id=(
                str(data.get("capability_id")).strip()
                if data.get("capability_id")
                else None
            ),
            capability_owner=(
                str(data.get("capability_owner")).strip()
                if data.get("capability_owner")
                else None
            ),
            input=dict(raw_input) if isinstance(raw_input, Mapping) else {},
            context_evidence_refs=tuple(
                dict(item)
                for item in data.get("context_evidence_refs") or ()
                if isinstance(item, Mapping)
            ),
            budgets=DbAnalysisStepBudgets.from_mapping(data.get("budgets")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "purpose": self.purpose,
            "depends_on": list(self.depends_on),
            "input_refs": [dict(item) for item in self.input_refs],
            "expected_evidence": list(self.expected_evidence),
            "capability_id": self.capability_id,
            "capability_owner": self.capability_owner,
            "input": dict(self.input),
            "context_evidence_refs": [
                dict(item) for item in self.context_evidence_refs
            ],
            "budgets": self.budgets.to_dict(),
        }


@dataclass(frozen=True)
class DbAnalysisPlan:
    """Declarative multi-step analysis plan."""

    analysis_id: str
    goal: str
    steps: tuple[DbAnalysisStep, ...]
    budgets: DbAnalysisBudgets = field(default_factory=DbAnalysisBudgets)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DbAnalysisPlan":
        data = dict(value)
        steps_value = data.get("steps")
        if not isinstance(steps_value, list):
            raise ValueError("analysis_steps_must_be_list")
        steps = tuple(DbAnalysisStep.from_mapping(item) for item in steps_value)
        return cls(
            analysis_id=str(data.get("analysis_id") or f"analysis-{uuid4()}"),
            goal=str(data.get("goal") or "").strip(),
            steps=steps,
            budgets=DbAnalysisBudgets.from_mapping(data.get("budgets")),
            diagnostics=dict(data.get("diagnostics") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "budgets": self.budgets.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class DbAnalysisPlanValidation:
    """Deterministic validation result for an analysis DAG."""

    valid: bool
    analysis_id: str | None
    plan_evidence_id: str | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    accepted_step_ids: tuple[str, ...]
    plan_fingerprint: str | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "analysis_id": self.analysis_id,
            "plan_evidence_id": self.plan_evidence_id,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "accepted_step_ids": list(self.accepted_step_ids),
            "plan_fingerprint": self.plan_fingerprint,
            "diagnostics": dict(self.diagnostics),
        }


def parse_analysis_plan_json(content: str) -> DbAnalysisPlan:
    """Parse strict JSON analysis plan output."""
    raw = _strip_json_fence(content)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("analysis_plan_json_not_object")
    return DbAnalysisPlan.from_mapping(parsed)


def validate_analysis_plan_payload(
    payload: Mapping[str, Any],
    *,
    plan_evidence: Evidence | None = None,
    registered_capabilities: _RegisteredCapabilities = None,
) -> DbAnalysisPlanValidation:
    """Validate DAG shape, step kinds, dependency refs, and budgets."""
    errors: list[str] = []
    warnings: list[str] = []
    plan: DbAnalysisPlan | None = None
    try:
        plan = DbAnalysisPlan.from_mapping(payload)
    except Exception as exc:
        errors.append(f"invalid_payload:{type(exc).__name__}:{exc}")

    if plan is None:
        return DbAnalysisPlanValidation(
            valid=False,
            analysis_id=None,
            plan_evidence_id=getattr(plan_evidence, "id", None),
            errors=tuple(errors),
            warnings=tuple(warnings),
            accepted_step_ids=(),
            plan_fingerprint=None,
            diagnostics={},
        )

    if not plan.goal:
        errors.append("goal_required")
    if len(plan.steps) > plan.budgets.max_steps:
        errors.append("budget_max_steps_exceeded")
    query_count = sum(1 for step in plan.steps if step.kind == "query")
    checkpoint_count = sum(1 for step in plan.steps if step.kind == "checkpoint")
    if query_count > plan.budgets.max_query_steps:
        errors.append("budget_max_query_steps_exceeded")
    if checkpoint_count > plan.budgets.max_checkpoint_steps:
        errors.append("budget_max_checkpoint_steps_exceeded")

    ids = [step.id for step in plan.steps]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_step_ids")
    id_set = set(ids)
    capability_ids = _registered_capability_ids(registered_capabilities)
    for step in plan.steps:
        if step.kind not in ANALYSIS_STEP_KINDS:
            errors.append(f"unsupported_step_kind:{step.id}:{step.kind}")
        for dependency in step.depends_on:
            if dependency not in id_set:
                errors.append(f"unknown_dependency:{step.id}:{dependency}")
            if dependency == step.id:
                errors.append(f"self_dependency:{step.id}")
        if step.kind == "query" and registered_capabilities is not None:
            for capability_id in (
                "db.query.plan.validate",
                "db.sql.validate",
                "db.sql.execute_read",
            ):
                if capability_ids is not None and capability_id not in capability_ids:
                    errors.append(f"missing_capability:{capability_id}")
        if step.kind in CAPABILITY_ANALYSIS_STEP_CONTRACTS:
            _validate_capability_step(
                step,
                registered_capabilities=registered_capabilities,
                errors=errors,
            )

    errors.extend(_cycle_errors(plan.steps))
    return DbAnalysisPlanValidation(
        valid=not errors,
        analysis_id=plan.analysis_id,
        plan_evidence_id=getattr(plan_evidence, "id", None),
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(warnings),
        accepted_step_ids=tuple(ids) if not errors else (),
        plan_fingerprint=stable_fingerprint(plan.to_dict()),
        diagnostics={
            "step_count": len(plan.steps),
            "query_step_count": query_count,
            "checkpoint_step_count": checkpoint_count,
            "budgets": plan.budgets.to_dict(),
        },
    )


def capability_contract_for_step_kind(kind: str) -> Mapping[str, Any] | None:
    contract = CAPABILITY_ANALYSIS_STEP_CONTRACTS.get(kind)
    return dict(contract) if contract is not None else None


def _validate_capability_step(
    step: DbAnalysisStep,
    *,
    registered_capabilities: _RegisteredCapabilities,
    errors: list[str],
) -> None:
    contract = CAPABILITY_ANALYSIS_STEP_CONTRACTS[step.kind]
    if not step.capability_id:
        errors.append(f"capability_id_required:{step.id}")
        return
    if step.capability_id not in contract["capabilities"]:
        errors.append(f"unsupported_step_capability:{step.id}:{step.capability_id}")
    if not step.capability_owner:
        errors.append(f"capability_owner_required:{step.id}")
    elif step.capability_owner not in contract["owners"]:
        errors.append(
            f"unsupported_step_capability_owner:{step.id}:{step.capability_owner}"
        )
    expected = set(step.expected_evidence)
    if expected and not expected <= set(contract["evidence"]):
        errors.append(f"unsupported_step_expected_evidence:{step.id}")
    if registered_capabilities is None:
        return
    if isinstance(registered_capabilities, Mapping):
        capability = registered_capabilities.get(
            (step.capability_owner, step.capability_id)
        ) or registered_capabilities.get(step.capability_id)
        if capability is None:
            errors.append(f"missing_capability:{step.capability_id}")
            return
        if getattr(capability, "owner", None) != step.capability_owner:
            errors.append(f"wrong_capability_owner:{step.id}:{step.capability_id}")
        output_evidence = set(getattr(capability, "output_evidence", frozenset()))
        if expected and not expected <= output_evidence:
            errors.append(f"wrong_capability_output:{step.id}:{step.capability_id}")
        if getattr(capability, "side_effecting", False):
            errors.append(f"side_effecting_capability_not_allowed:{step.id}")
    elif step.capability_id not in registered_capabilities:
        errors.append(f"missing_capability:{step.capability_id}")


def _registered_capability_ids(
    registered_capabilities: _RegisteredCapabilities,
) -> set[str] | None:
    if registered_capabilities is None:
        return None
    ids = set()
    for key in registered_capabilities:
        if isinstance(key, tuple) and len(key) == 2:
            ids.add(str(key[1]))
        else:
            ids.add(str(key))
    return ids


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def structural_schema_fingerprint(schema: Mapping[str, Any] | None) -> str | None:
    """Fingerprint catalog-owned schema structure without incidental metadata."""
    if not schema:
        return None
    return stable_fingerprint(_structural_schema_payload(schema))


def _structural_schema_payload(schema: Mapping[str, Any]) -> dict[str, Any]:
    tables = []
    for table in schema.get("tables", []) or []:
        if not isinstance(table, Mapping):
            continue
        table_name = str(table.get("name") or "").strip()
        if not table_name:
            continue
        columns: list[dict[str, Any]] = []
        for column in table.get("columns", []) or []:
            if not isinstance(column, Mapping):
                continue
            column_name = str(column.get("name") or "").strip()
            if not column_name:
                continue
            columns.append(
                {
                    "name": column_name,
                    "data_type": column.get("data_type"),
                    "is_primary_key": bool(column.get("is_primary_key")),
                }
            )
        columns.sort(key=lambda column: str(column.get("name") or ""))
        tables.append(
            {
                "name": table_name,
                "columns": columns,
            }
        )

    foreign_keys = []
    for foreign_key in schema.get("foreign_keys", []) or []:
        if not isinstance(foreign_key, Mapping):
            continue
        foreign_keys.append(
            {
                "source_table": foreign_key.get("source_table"),
                "source_column": foreign_key.get("source_column"),
                "target_table": foreign_key.get("target_table"),
                "target_column": foreign_key.get("target_column"),
            }
        )

    return {
        "database_type": schema.get("database_type"),
        "tables": sorted(tables, key=lambda item: item["name"]),
        "foreign_keys": sorted(
            foreign_keys,
            key=lambda item: (
                str(item.get("source_table") or ""),
                str(item.get("source_column") or ""),
                str(item.get("target_table") or ""),
                str(item.get("target_column") or ""),
            ),
        ),
    }


def analysis_metadata(
    *,
    analysis_id: str,
    step_id: str | None,
    step_kind: str | None = None,
    plan_evidence_id: str | None = None,
    phase: str | None = None,
) -> dict[str, str]:
    metadata = {
        "analysis_id": analysis_id,
    }
    if step_id is not None:
        metadata["analysis_step_id"] = step_id
    if step_kind is not None:
        metadata["analysis_step_kind"] = step_kind
    if plan_evidence_id is not None:
        metadata["analysis_plan_evidence_id"] = plan_evidence_id
    if phase is not None:
        metadata["analysis_phase"] = phase
    return metadata


def with_analysis_evidence_trace(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Mark analysis metadata keys that should be copied to produced evidence."""
    copied = dict(metadata)
    trace_keys = [
        key
        for key in (
            "analysis_id",
            "analysis_step_id",
            "analysis_step_kind",
            "analysis_plan_evidence_id",
            "analysis_phase",
        )
        if copied.get(key) is not None
    ]
    if trace_keys:
        copied["evidence_trace_keys"] = trace_keys
    return copied


def evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "task_id": evidence.task_id,
        "operation_id": evidence.operation_id,
        "payload_fingerprint": evidence.metadata.get("payload_fingerprint")
        or stable_fingerprint(evidence.payload),
        "analysis_step_id": evidence.metadata.get("analysis_step_id"),
        "analysis_step_kind": evidence.metadata.get("analysis_step_kind"),
    }


def _cycle_errors(steps: tuple[DbAnalysisStep, ...]) -> list[str]:
    by_id = {step.id: step for step in steps}
    visiting: set[str] = set()
    visited: set[str] = set()
    errors: list[str] = []

    def visit(step_id: str, path: tuple[str, ...]) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            errors.append(f"cycle_detected:{'->'.join((*path, step_id))}")
            return
        visiting.add(step_id)
        step = by_id.get(step_id)
        if step is not None:
            for dependency in step.depends_on:
                if dependency in by_id:
                    visit(dependency, (*path, step_id))
        visiting.remove(step_id)
        visited.add(step_id)

    for step in steps:
        visit(step.id, ())
    return errors


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple, set)):
        raise TypeError("value must be a sequence of strings")
    return tuple(str(item) for item in value)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 1:
        raise ValueError("budget values must be positive")
    return parsed


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 0:
        raise ValueError("budget values must be non-negative")
    return parsed
