"""Evidence-backed planning context for DB query planning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from daita.runtime import Evidence, Operation

from .analysis import structural_schema_fingerprint
from .context_projection import (
    ProjectionContext,
    ProjectionMode,
    policy_summary_from_source,
    project_catalog_hints,
    project_memory_refs,
    project_memory_semantics,
    project_policy_summary,
    project_session_context,
)
from .fingerprints import persisted_fingerprint
from .memory.config import db_memory_options_from_from_db_options
from .memory.contracts import (
    db_memory_contracts_artifact_payload,
    project_db_memory_semantic_contracts,
)
from .memory.selection import (
    db_memory_refs_from_recall_evidence,
    db_memory_selection_artifact_payload,
)
from .models import DbIntent, DbRequest, DbRuntimeConfig
from .session_context import db_session_context_from_request


@dataclass(frozen=True)
class DbPlanningContext:
    """Compact context artifact used by deterministic and LLM planners."""

    operation_id: str
    prompt: str
    intent_kind: str
    source_scope: tuple[str, ...]
    dialect: str | None
    schema: dict[str, Any]
    schema_evidence_refs: tuple[str, ...] = ()
    catalog_evidence_refs: tuple[str, ...] = ()
    relationship_evidence_refs: tuple[str, ...] = ()
    relationship_evidence_details: tuple[dict[str, Any], ...] = ()
    column_value_evidence_refs: tuple[str, ...] = ()
    column_value_hints: tuple[dict[str, Any], ...] = ()
    db_memory_refs: tuple[dict[str, Any], ...] = ()
    db_memory_semantics: tuple[dict[str, Any], ...] = ()
    db_memory_selection_evidence_ref: dict[str, Any] = field(default_factory=dict)
    db_memory_contracts_evidence_ref: dict[str, Any] = field(default_factory=dict)
    db_memory_selection_artifact: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    db_memory_contracts_artifact: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    source_evidence_refs: tuple[dict[str, Any], ...] = ()
    source_fingerprints: dict[str, str] = field(default_factory=dict)
    capability_summaries: tuple[dict[str, Any], ...] = ()
    included_sections: tuple[str, ...] = ()
    omitted_sections: tuple[str, ...] = ()
    budget_usage: dict[str, Any] = field(default_factory=dict)
    context_selection_diagnostics: dict[str, Any] = field(default_factory=dict)
    policy_summary: dict[str, Any] = field(default_factory=dict)
    limit_summary: dict[str, Any] = field(default_factory=dict)
    redaction_policy: dict[str, Any] = field(default_factory=dict)
    session_context: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    rendered_context: str = ""
    schema_fingerprint: str | None = None

    def to_payload(self) -> dict[str, Any]:
        db_memory_diagnostics = _db_memory_selection_diagnostics(
            self.db_memory_selection_artifact
        )
        db_memory_contract_diagnostics = _db_memory_contract_diagnostics(
            self.db_memory_contracts_artifact
        )
        return {
            "operation_id": self.operation_id,
            "prompt": self.prompt,
            "intent_kind": self.intent_kind,
            "source_scope": list(self.source_scope),
            "dialect": self.dialect,
            "schema": self.schema,
            "schema_evidence_refs": list(self.schema_evidence_refs),
            "catalog_evidence_refs": list(self.catalog_evidence_refs),
            "relationship_evidence_refs": list(self.relationship_evidence_refs),
            "relationship_evidence_details": [
                dict(item) for item in self.relationship_evidence_details
            ],
            "column_value_evidence_refs": list(self.column_value_evidence_refs),
            "column_value_hints": [dict(item) for item in self.column_value_hints],
            "db_memory_refs": [dict(item) for item in self.db_memory_refs],
            "db_memory_semantics": [dict(item) for item in self.db_memory_semantics],
            "db_memory_selection_evidence_ref": dict(
                self.db_memory_selection_evidence_ref
            ),
            "db_memory_contracts_evidence_ref": dict(
                self.db_memory_contracts_evidence_ref
            ),
            "db_memory_diagnostics": db_memory_diagnostics,
            "db_memory_contract_diagnostics": db_memory_contract_diagnostics,
            "source_evidence_refs": [dict(item) for item in self.source_evidence_refs],
            "source_fingerprints": dict(self.source_fingerprints),
            "capability_summaries": [dict(item) for item in self.capability_summaries],
            "included_sections": list(self.included_sections),
            "omitted_sections": list(self.omitted_sections),
            "budget_usage": dict(self.budget_usage),
            "context_selection_diagnostics": dict(self.context_selection_diagnostics),
            "policy_summary": self.policy_summary,
            "limit_summary": self.limit_summary,
            "redaction_policy": self.redaction_policy,
            "session_context": self.session_context,
            "diagnostics": self.diagnostics,
            "rendered_context": self.rendered_context,
            "schema_fingerprint": self.schema_fingerprint,
        }


def authoritative_schema_identity_from_evidence(
    evidence: Iterable[Evidence],
) -> tuple[dict[str, Any], str | None]:
    """Return the operation's authoritative structural schema identity."""
    items = tuple(evidence)
    for item in reversed(items):
        if not item.accepted or item.kind != "planning.context":
            continue
        fingerprint = item.payload.get("schema_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            continue
        schema = item.payload.get("schema")
        return (
            dict(schema) if isinstance(schema, dict) else {},
            fingerprint.strip(),
        )

    connector_schema = next(
        (item for item in reversed(items) if _connector_schema_payload(item)),
        None,
    )
    schema, schema_fingerprint, _, _ = _planning_context_schema_identity(
        connector_schema,
        tuple(items),
        tuple(
            item
            for item in items
            if item.accepted
            and item.owner == "catalog"
            and item.kind == "schema.relationship_path"
        ),
    )
    return schema, schema_fingerprint


class DbPlanningContextBuilder:
    """Build bounded, auditable planning context from accepted evidence."""

    def __init__(self, config: DbRuntimeConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        request: DbRequest,
        intent: DbIntent,
        operation: Operation,
        schema_evidence: Evidence | None,
        catalog_evidence: tuple[Evidence, ...] = (),
        relationship_evidence: tuple[Evidence, ...] = (),
        memory_recall_evidence: tuple[Evidence, ...] = (),
        memory_recall_diagnostics: dict[str, Any] | None = None,
        capability_summaries: tuple[dict[str, Any], ...] = (),
        source: Any = None,
    ) -> DbPlanningContext:
        (
            schema,
            schema_fingerprint,
            schema_source,
            catalog_structural_evidence,
        ) = _planning_context_schema_identity(
            schema_evidence,
            catalog_evidence,
            relationship_evidence,
        )
        relationship_details = _relationship_evidence_details(relationship_evidence)
        dialect = (
            str(
                schema.get("database_type")
                or getattr(source, "sql_dialect", None)
                or ""
            )
            or None
        )
        options = _from_db_options(self.config.metadata)
        source_evidence = _dedupe_evidence(
            item
            for item in (schema_evidence, *catalog_evidence, *relationship_evidence)
            if item is not None and item.accepted
        )
        source_refs = tuple(_evidence_ref(item) for item in source_evidence)
        source_fingerprints = {
            str(item.id): str(
                item.metadata.get("payload_fingerprint")
                or persisted_fingerprint(item.payload)
            )
            for item in source_evidence
            if item.id
        }
        included_sections = ["schema", "limits", "policy", "capabilities"]
        if catalog_evidence:
            included_sections.append("catalog")
        if relationship_evidence:
            included_sections.append("relationships")
        policy_summary = policy_summary_from_source(source)
        projection = ProjectionContext(
            mode=ProjectionMode.PLANNER,
            operation_intent=intent.kind.value,
            safety_frame=operation.metadata.get("safety_frame"),
            policy_summary=policy_summary,
            source_identity=None,
            schema_fingerprint=schema_fingerprint,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        column_value_hints = project_catalog_hints(
            _column_value_hints(
                catalog_evidence,
                schema,
            ),
            projection,
        )
        column_value_evidence = tuple(
            item
            for item in catalog_evidence
            if item.kind
            in {
                "schema.column_value_profile",
                "schema.column_value_search_result",
                "schema.column_value_hint",
            }
        )
        if column_value_hints:
            included_sections.append("column_value_hints")
        memory_options = db_memory_options_from_from_db_options(options)
        projection = ProjectionContext(
            mode=ProjectionMode.PLANNER,
            operation_intent=intent.kind.value,
            safety_frame=operation.metadata.get("safety_frame"),
            policy_summary=policy_summary,
            source_identity=memory_options.get("source_identity"),
            schema_fingerprint=schema_fingerprint,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        memory_limit = int(memory_options.get("limit") or 3)
        memory_char_budget = int(memory_options.get("char_budget") or 800)
        memory_score_threshold = _float_option(
            memory_options,
            "score_threshold",
            0.45,
        )
        accepted_memory_recall_evidence = tuple(
            item for item in memory_recall_evidence if item.accepted
        )
        memory_recall_refs = _evidence_refs(accepted_memory_recall_evidence)
        raw_db_memory_refs, db_memory_evidence_refs, selection_diagnostics = (
            db_memory_refs_from_recall_evidence(
                accepted_memory_recall_evidence,
                prompt=request.prompt,
                schema=schema,
                source_identity=memory_options.get("source_identity"),
                schema_fingerprint=schema_fingerprint,
                limit=memory_limit,
                char_budget=memory_char_budget,
                score_threshold=memory_score_threshold,
            )
        )
        selection_diagnostics = {
            **selection_diagnostics,
            **dict(memory_recall_diagnostics or {}),
        }
        db_memory_selection_artifact = (
            db_memory_selection_artifact_payload(
                source_identity=memory_options.get("source_identity"),
                schema_fingerprint=schema_fingerprint,
                recall_evidence_refs=memory_recall_refs,
                memory_evidence_refs=db_memory_evidence_refs,
                included_refs=raw_db_memory_refs,
                diagnostics=selection_diagnostics,
                limit=memory_limit,
                char_budget=memory_char_budget,
                score_threshold=memory_score_threshold,
            )
            if accepted_memory_recall_evidence or memory_recall_diagnostics
            else {}
        )
        db_memory_refs = project_memory_refs(
            db_memory_selection_artifact.get("included_refs") or raw_db_memory_refs,
            projection,
        )
        raw_db_memory_semantics, contract_projection_diagnostics = (
            project_db_memory_semantic_contracts(
                db_memory_selection_artifact.get("included_refs") or raw_db_memory_refs,
                prompt=request.prompt,
                schema=schema,
                policy_summary=policy_summary,
                source_identity=memory_options.get("source_identity"),
            )
        )
        db_memory_contracts_artifact = (
            db_memory_contracts_artifact_payload(
                source_identity=memory_options.get("source_identity"),
                schema_fingerprint=schema_fingerprint,
                recall_evidence_refs=memory_recall_refs,
                selection_evidence_ref=None,
                contracts=raw_db_memory_semantics,
                diagnostics=contract_projection_diagnostics,
            )
            if db_memory_selection_artifact
            else {}
        )
        db_memory_semantics = project_memory_semantics(
            db_memory_contracts_artifact.get("contracts") or raw_db_memory_semantics,
            projection,
        )
        if db_memory_refs:
            included_sections.append("db_memory")
        if db_memory_semantics:
            included_sections.append("db_memory_semantics")
        diagnostics = {
            "schema_table_count": len(schema.get("tables", []) or []),
            "context_budget": options.get("planner_context_budget"),
            "omitted_sections": [],
            "schema_fingerprint": schema_fingerprint,
            "source_evidence_count": len(source_evidence),
            "capability_summary_count": len(capability_summaries),
            "column_value_hint_count": len(column_value_hints),
            "db_memory_ref_count": len(db_memory_refs),
            "db_memory_contract_count": len(db_memory_semantics),
            "structural_schema_source": schema_source,
            "catalog_structural_evidence_refs": [
                item.id for item in catalog_structural_evidence if item.id
            ],
            "relationship_evidence_ref_count": len(relationship_details),
        }
        session_context = project_session_context(
            _compact_session_context(request),
            projection,
        )
        if session_context:
            included_sections.append("session_context")
        context = DbPlanningContext(
            operation_id=operation.id,
            prompt=request.prompt,
            intent_kind=intent.kind.value,
            source_scope=request.source_scope,
            dialect=dialect,
            schema=_compact_schema(schema),
            schema_evidence_refs=_evidence_refs((schema_evidence,)),
            catalog_evidence_refs=_evidence_refs(catalog_evidence),
            relationship_evidence_refs=_evidence_refs(relationship_evidence),
            relationship_evidence_details=relationship_details,
            column_value_evidence_refs=_evidence_refs(column_value_evidence),
            column_value_hints=column_value_hints,
            db_memory_refs=db_memory_refs,
            db_memory_semantics=db_memory_semantics,
            db_memory_selection_artifact=db_memory_selection_artifact,
            db_memory_contracts_artifact=db_memory_contracts_artifact,
            source_evidence_refs=source_refs,
            source_fingerprints=source_fingerprints,
            capability_summaries=tuple(capability_summaries),
            included_sections=tuple(included_sections),
            omitted_sections=(),
            budget_usage={
                "rendered_context_chars": 0,
                "capability_summary_count": len(capability_summaries),
                "source_evidence_count": len(source_evidence),
                "db_memory_chars": int(selection_diagnostics.get("used_chars") or 0),
            },
            context_selection_diagnostics={
                "mode": "runtime_bounded",
                "capability_summary_owners": sorted(
                    {
                        str(item.get("owner"))
                        for item in capability_summaries
                        if item.get("owner")
                    }
                ),
            },
            policy_summary=project_policy_summary(policy_summary, projection),
            limit_summary={
                "max_rows": self.config.limits.max_rows,
                "timeout_seconds": self.config.limits.timeout_seconds,
                "query_default_limit": getattr(source, "query_default_limit", None),
                "query_max_rows": getattr(source, "query_max_rows", None),
                "query_max_chars": getattr(source, "query_max_chars", None),
            },
            redaction_policy={
                "redact_pii_columns": bool(getattr(source, "redact_pii_columns", True)),
                "include_sample_values": bool(
                    getattr(source, "include_sample_values", True)
                ),
            },
            session_context=session_context,
            diagnostics=diagnostics,
            schema_fingerprint=schema_fingerprint,
        )
        return DbPlanningContext(
            **{
                **context.__dict__,
                "rendered_context": _render_context_summary(context),
                "budget_usage": {
                    **context.budget_usage,
                    "rendered_context_chars": len(_render_context_summary(context)),
                },
            }
        )

    def evidence_for(self, context: DbPlanningContext) -> Evidence:
        return Evidence(
            kind="planning.context",
            owner="db_runtime",
            operation_id=context.operation_id,
            payload=context.to_payload(),
            metadata={
                "schema_fingerprint": context.schema_fingerprint,
                "payload_fingerprint": persisted_fingerprint(context.to_payload()),
            },
        )

    def memory_selection_evidence_for(
        self,
        context: DbPlanningContext,
        *,
        task_id: str | None = None,
    ) -> Evidence | None:
        if not context.db_memory_selection_artifact:
            return None
        return _artifact_evidence(
            kind="db.memory.selection",
            operation_id=context.operation_id,
            task_id=task_id,
            payload=context.db_memory_selection_artifact,
        )

    def memory_contracts_evidence_for(
        self,
        context: DbPlanningContext,
        *,
        selection_evidence_ref: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> Evidence | None:
        if not context.db_memory_contracts_artifact:
            return None
        payload = {
            **context.db_memory_contracts_artifact,
            "selection_evidence_ref": dict(
                selection_evidence_ref
                or context.db_memory_contracts_artifact.get(
                    "selection_evidence_ref",
                    {},
                )
                or {}
            ),
        }
        return _artifact_evidence(
            kind="db.memory.contracts",
            operation_id=context.operation_id,
            task_id=task_id,
            payload=payload,
        )

    def with_memory_artifact_refs(
        self,
        context: DbPlanningContext,
        *,
        selection_evidence: Evidence | None,
        contracts_evidence: Evidence | None,
    ) -> DbPlanningContext:
        return replace(
            context,
            db_memory_selection_evidence_ref=(
                _evidence_ref(selection_evidence) if selection_evidence else {}
            ),
            db_memory_contracts_evidence_ref=(
                _evidence_ref(contracts_evidence) if contracts_evidence else {}
            ),
        )


def _db_memory_selection_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact:
        return {}
    budget = artifact.get("budget_usage")
    budget_usage = dict(budget) if isinstance(budget, dict) else {}
    return {
        "candidate_count": int(
            artifact.get("raw_candidate_count")
            or budget_usage.get("raw_candidate_count")
            or 0
        ),
        "included_count": int(
            artifact.get("included_count") or budget_usage.get("included_count") or 0
        ),
        "used_chars": int(budget_usage.get("used_chars") or 0),
        "char_budget": int(budget_usage.get("char_budget") or 0),
        "limit": int(budget_usage.get("limit") or 0),
        "score_threshold": float(budget_usage.get("score_threshold") or 0.0),
        "omitted_reasons": {
            str(reason): int(count)
            for reason, count in dict(
                artifact.get("omitted_counts_by_reason") or {}
            ).items()
        },
    }


def _db_memory_contract_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact:
        return {}
    applicability = artifact.get("source_schema_applicability")
    applicability = dict(applicability) if isinstance(applicability, dict) else {}
    enforced_count = (
        applicability.get("enforced_count")
        if "enforced_count" in applicability
        else len(artifact.get("enforceable_contracts") or ())
    )
    advisory_count = (
        applicability.get("advisory_count")
        if "advisory_count" in applicability
        else len(artifact.get("advisory_contracts") or ())
    )
    return {
        "candidate_count": int(applicability.get("contract_candidate_count") or 0),
        "enforced_count": int(enforced_count or 0),
        "advisory_count": int(advisory_count or 0),
        "omitted_count": int(applicability.get("omitted_count") or 0),
        "omitted_reasons": {
            str(reason): int(count)
            for reason, count in dict(
                artifact.get("contract_omission_reasons") or {}
            ).items()
        },
    }


def _catalog_structural_evidence(
    schema_evidence: Evidence | None,
    catalog_evidence: tuple[Evidence, ...],
) -> tuple[Evidence, ...]:
    evidence = []
    for item in (schema_evidence, *catalog_evidence):
        if item is None or not item.accepted:
            continue
        if item.owner != "catalog":
            continue
        if item.payload.get("success") is False:
            continue
        if item.kind in {
            "schema.asset_profile",
            "schema.search_result",
            "catalog.source_registered",
            "catalog.profile",
        }:
            evidence.append(item)
    return _dedupe_evidence(evidence)


def _planning_context_schema_identity(
    schema_evidence: Evidence | None,
    catalog_evidence: tuple[Evidence, ...],
    relationship_evidence: tuple[Evidence, ...],
) -> tuple[dict[str, Any], str | None, str, tuple[Evidence, ...]]:
    catalog_structural_evidence = tuple(
        item
        for item in _catalog_structural_evidence(schema_evidence, catalog_evidence)
        if _catalog_evidence_is_normalizable(item)
    )
    schema = catalog_schema_from_evidence(
        catalog_structural_evidence,
        relationship_evidence,
    )
    schema_source = "catalog" if schema else "connector"
    if not schema:
        schema = _connector_schema_payload(schema_evidence)
    return (
        schema,
        structural_schema_fingerprint(schema),
        schema_source,
        catalog_structural_evidence,
    )


def _catalog_evidence_is_normalizable(evidence: Evidence) -> bool:
    try:
        catalog_schema_from_evidence((evidence,), ())
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _connector_schema_payload(evidence: Evidence | None) -> dict[str, Any]:
    if (
        evidence is None
        or not evidence.accepted
        or evidence.kind != "schema.asset_profile"
        or evidence.owner == "catalog"
        or evidence.payload.get("success") is False
    ):
        return {}
    return dict(evidence.payload)


def catalog_schema_from_evidence(
    catalog_evidence: tuple[Evidence, ...],
    relationship_evidence: tuple[Evidence, ...],
) -> dict[str, Any]:
    """Normalize accepted catalog evidence into the planner's schema contract."""
    if not catalog_evidence:
        return {}
    tables: dict[str, dict[str, Any]] = {}
    foreign_keys: list[dict[str, Any]] = []
    database_type: str | None = None
    database_name: str | None = None
    database_dialect: str | None = None
    structural_evidence_ids: list[str] = []

    for evidence in catalog_evidence:
        if (
            not evidence.accepted
            or evidence.owner != "catalog"
            or evidence.payload.get("success") is False
        ):
            continue
        if evidence.id:
            structural_evidence_ids.append(evidence.id)
        payload = dict(evidence.payload)
        database_type = database_type or _optional_string(payload.get("database_type"))
        database_name = database_name or _optional_string(payload.get("database_name"))
        database_dialect = database_dialect or _optional_string(
            payload.get("database_dialect") or payload.get("sql_dialect")
        )
        schema = payload.get("schema")
        if isinstance(schema, dict):
            database_type = database_type or _optional_string(
                schema.get("database_type")
            )
            database_name = database_name or _optional_string(
                schema.get("database_name")
            )
            database_dialect = database_dialect or _optional_string(
                schema.get("database_dialect")
                or schema.get("sql_dialect")
                or schema.get("database_type")
            )
            _merge_catalog_tables(tables, schema.get("tables"), evidence=evidence)
            foreign_keys.extend(_foreign_keys_from_payload(schema))
        if evidence.kind == "schema.asset_profile":
            table = _table_from_asset_profile(payload, evidence=evidence)
            if table is not None:
                _merge_table(tables, table)
            foreign_keys.extend(_foreign_keys_from_payload(payload))
        elif evidence.kind == "schema.search_result":
            _merge_catalog_tables(
                tables,
                payload.get("tables") or payload.get("assets"),
                evidence=evidence,
            )
        elif evidence.kind == "catalog.profile":
            _merge_catalog_tables(
                tables,
                payload.get("tables") or payload.get("assets"),
                evidence=evidence,
            )
            foreign_keys.extend(_foreign_keys_from_payload(payload))

    for evidence in relationship_evidence:
        if evidence.accepted and evidence.owner == "catalog":
            foreign_keys.extend(_foreign_keys_from_relationship_path(evidence))

    if not tables and not foreign_keys:
        return {}
    for foreign_key in foreign_keys:
        for table_name, column_name in (
            (foreign_key.get("source_table"), foreign_key.get("source_column")),
            (foreign_key.get("target_table"), foreign_key.get("target_column")),
        ):
            if not table_name:
                continue
            table = tables.setdefault(
                str(table_name).lower(),
                {"name": str(table_name), "columns": [], "metadata": {}},
            )
            if column_name:
                _merge_column(table, {"name": column_name})
    return {
        "database_type": database_type,
        "database_name": database_name,
        "database_dialect": database_dialect or database_type,
        "sql_dialect": database_dialect or database_type,
        "table_count": len(tables),
        "tables": list(tables.values()),
        "foreign_keys": _dedupe_foreign_keys(foreign_keys),
        "metadata": {
            "structural_source": "catalog",
            "catalog_structural_evidence_ids": list(
                dict.fromkeys(structural_evidence_ids)
            ),
        },
    }


def _table_from_asset_profile(
    payload: dict[str, Any],
    *,
    evidence: Evidence | None = None,
) -> dict[str, Any] | None:
    asset = payload.get("asset") if isinstance(payload.get("asset"), dict) else None
    table = payload.get("table") if isinstance(payload.get("table"), dict) else None
    source = table or asset
    if source is None and payload.get("name"):
        source = payload
    if not isinstance(source, dict):
        return None
    name = _optional_string(
        source.get("name")
        or source.get("asset_ref")
        or payload.get("table_name")
        or payload.get("asset_ref")
    )
    if not name:
        return None
    columns = payload.get("fields") or payload.get("columns") or source.get("columns")
    store_id = payload.get("store_id") or source.get("store_id")
    return {
        "name": name,
        "columns": [
            _column_from_catalog_field(
                item,
                evidence=evidence,
                table_name=name,
                store_id=store_id,
            )
            for item in columns or ()
        ],
        "metadata": {
            **dict(source.get("metadata") or {}),
            "catalog_asset_ref": source.get("asset_ref") or name,
            "catalog_store_id": store_id,
            **(
                {
                    "catalog_evidence_id": evidence.id,
                    "catalog_evidence_owner": evidence.owner,
                    "catalog_evidence_kind": evidence.kind,
                }
                if evidence is not None
                else {}
            ),
        },
    }


def _merge_catalog_tables(
    tables: dict[str, dict[str, Any]],
    raw_tables: Any,
    *,
    evidence: Evidence | None = None,
) -> None:
    if not isinstance(raw_tables, list):
        return
    for raw in raw_tables:
        if not isinstance(raw, dict):
            continue
        name = _optional_string(raw.get("name") or raw.get("asset_ref"))
        if not name:
            continue
        raw_columns = (
            raw.get("columns") or raw.get("fields") or raw.get("matched_fields")
        )
        table = {
            "name": name,
            "columns": [
                _column_from_catalog_field(
                    item,
                    evidence=evidence,
                    table_name=name,
                    store_id=(evidence.payload.get("store_id") if evidence else None),
                )
                for item in raw_columns or ()
            ],
            "metadata": {
                **dict(raw.get("metadata") or {}),
                **(
                    {
                        "catalog_evidence_id": evidence.id,
                        "catalog_evidence_owner": evidence.owner,
                        "catalog_evidence_kind": evidence.kind,
                    }
                    if evidence is not None
                    else {}
                ),
            },
        }
        _merge_table(tables, table)


def _merge_table(
    tables: dict[str, dict[str, Any]],
    table: dict[str, Any],
) -> None:
    name = _optional_string(table.get("name"))
    if not name:
        return
    key = name.lower()
    existing = tables.setdefault(
        key,
        {"name": name, "columns": [], "metadata": {}},
    )
    existing["metadata"] = {
        **dict(existing.get("metadata") or {}),
        **dict(table.get("metadata") or {}),
    }
    for column in table.get("columns", []) or []:
        _merge_column(existing, column)


def _merge_column(table: dict[str, Any], column: dict[str, Any]) -> None:
    name = _optional_string(column.get("name"))
    if not name:
        return
    columns = table.setdefault("columns", [])
    existing = next(
        (
            item
            for item in columns
            if isinstance(item, dict)
            and str(item.get("name") or "").lower() == name.lower()
        ),
        None,
    )
    normalized = _column_from_catalog_field(column)
    if existing is None:
        columns.append(normalized)
        return
    for key, value in normalized.items():
        if key == "name" or value is None:
            continue
        if key in {"logical_type_proof", "identity_proof", "catalog_evidence"}:
            if isinstance(value, dict):
                existing[key] = {**dict(existing.get(key) or {}), **value}
            continue
        if key == "is_primary_key":
            existing[key] = bool(existing.get(key)) or bool(value)
            continue
        existing[key] = value


def _column_from_catalog_field(
    value: Any,
    *,
    evidence: Evidence | None = None,
    table_name: str | None = None,
    store_id: Any = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {
        "name": value.get("name"),
        "data_type": value.get("data_type") or value.get("type"),
        "physical_type": value.get("physical_type")
        or value.get("data_type")
        or value.get("type"),
        "native_type": value.get("native_type"),
        "database_dialect": value.get("database_dialect") or value.get("dialect"),
        "is_nullable": (
            value.get("is_nullable")
            if "is_nullable" in value
            else value.get("nullable")
        ),
        "is_primary_key": value.get("is_primary_key"),
        "default_value": (
            value.get("default_value")
            if "default_value" in value
            else value.get("column_default")
        ),
        "extra": value.get("extra"),
        "logical_type": value.get("logical_type"),
        "logical_type_proof": dict(value.get("logical_type_proof", {}) or {}),
        "is_identity": value.get("is_identity"),
        "is_generated": value.get("is_generated"),
        "is_autoincrement": value.get("is_autoincrement"),
        "is_monotonic": value.get("is_monotonic"),
        "identity_proof": dict(value.get("identity_proof", {}) or {}),
        "catalog_evidence": dict(value.get("catalog_evidence", {}) or {}),
    }
    if evidence is not None:
        result["catalog_evidence"] = {
            "id": evidence.id,
            "kind": evidence.kind,
            "owner": evidence.owner,
            "accepted": evidence.accepted,
            "store_id": store_id,
            "asset_ref": table_name,
            "column": value.get("name"),
        }
    return result


def _foreign_keys_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    foreign_keys = list(payload.get("foreign_keys", []) or [])
    for relationship in payload.get("relationships", []) or []:
        if not isinstance(relationship, dict):
            continue
        foreign_key = _foreign_key_from_relationship(relationship)
        if foreign_key is not None:
            foreign_keys.append(foreign_key)
    return [
        item
        for item in (_compact_foreign_key(item) for item in foreign_keys)
        if item is not None
    ]


def _foreign_keys_from_relationship_path(evidence: Evidence) -> list[dict[str, Any]]:
    foreign_keys: list[dict[str, Any]] = []
    for path in evidence.payload.get("paths", []) or []:
        if not isinstance(path, dict):
            continue
        for join in path.get("joins", []) or path.get("relationships", []) or []:
            if not isinstance(join, dict):
                continue
            foreign_key = _foreign_key_from_relationship(join)
            if foreign_key is not None:
                foreign_keys.append(
                    {
                        **foreign_key,
                        "metadata": {
                            "relationship_evidence_id": evidence.id,
                            "relationship_evidence_owner": evidence.owner,
                        },
                    }
                )
    return foreign_keys


def _foreign_key_from_relationship(value: dict[str, Any]) -> dict[str, Any] | None:
    source_table = _optional_string(
        value.get("source_table")
        or value.get("source_asset")
        or value.get("left_table")
        or value.get("left_asset")
    )
    source_column = _optional_string(
        value.get("source_column")
        or value.get("source_field")
        or value.get("left_column")
        or value.get("left_field")
    )
    target_table = _optional_string(
        value.get("target_table")
        or value.get("target_asset")
        or value.get("right_table")
        or value.get("right_asset")
    )
    target_column = _optional_string(
        value.get("target_column")
        or value.get("target_field")
        or value.get("right_column")
        or value.get("right_field")
    )
    if not all((source_table, source_column, target_table, target_column)):
        return None
    return {
        "source_table": source_table,
        "source_column": source_column,
        "target_table": target_table,
        "target_column": target_column,
    }


def _compact_foreign_key(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    foreign_key = _foreign_key_from_relationship(value)
    if foreign_key is None:
        return None
    if isinstance(value.get("metadata"), dict):
        foreign_key["metadata"] = dict(value["metadata"])
    return foreign_key


def _dedupe_foreign_keys(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for value in values:
        key = (
            str(value.get("source_table") or "").lower(),
            str(value.get("source_column") or "").lower(),
            str(value.get("target_table") or "").lower(),
            str(value.get("target_column") or "").lower(),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _relationship_evidence_details(
    relationship_evidence: tuple[Evidence, ...],
) -> tuple[dict[str, Any], ...]:
    details = []
    for evidence in relationship_evidence:
        if not evidence.accepted or evidence.kind != "schema.relationship_path":
            continue
        details.append(
            {
                "id": evidence.id,
                "kind": evidence.kind,
                "owner": evidence.owner,
                "task_id": evidence.task_id,
                "accepted": evidence.accepted,
                "payload_fingerprint": evidence.metadata.get("payload_fingerprint")
                or persisted_fingerprint(evidence.payload),
                "reachable": evidence.payload.get("reachable"),
                "paths": [
                    {
                        "tables": list(path.get("tables") or path.get("assets") or []),
                        "joins": [
                            dict(join)
                            for join in path.get("joins", []) or []
                            if isinstance(join, dict)
                        ],
                    }
                    for path in evidence.payload.get("paths", []) or []
                    if isinstance(path, dict)
                ],
            }
        )
    return tuple(details)


def _dedupe_evidence(values: Iterable[Evidence]) -> tuple[Evidence, ...]:
    seen: set[str] = set()
    out: list[Evidence] = []
    for evidence in values:
        key = evidence.id or persisted_fingerprint(
            {
                "kind": evidence.kind,
                "owner": evidence.owner,
                "payload": evidence.payload,
            }
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(evidence)
    return tuple(out)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "database_type": schema.get("database_type"),
        "database_name": schema.get("database_name"),
        "table_count": schema.get("table_count") or len(schema.get("tables", []) or []),
        "tables": [
            {
                "name": table.get("name"),
                "columns": [
                    {
                        "name": column.get("name"),
                        "data_type": column.get("data_type"),
                        "is_primary_key": column.get("is_primary_key"),
                    }
                    for column in table.get("columns", []) or []
                    if column.get("name")
                ],
                "metadata": dict(table.get("metadata") or {}),
            }
            for table in schema.get("tables", []) or []
            if table.get("name")
        ],
        "foreign_keys": list(schema.get("foreign_keys", []) or []),
    }


def _render_context_summary(context: DbPlanningContext) -> str:
    lines = [
        f"Prompt: {context.prompt}",
        f"Intent: {context.intent_kind}",
        f"Dialect: {context.dialect or 'unknown'}",
        "Tables:",
    ]
    for table in context.schema.get("tables", []) or []:
        columns = ", ".join(
            str(column.get("name"))
            for column in table.get("columns", []) or []
            if column.get("name")
        )
        lines.append(f"- {table.get('name')}: {columns}")
    foreign_keys = context.schema.get("foreign_keys", []) or []
    if foreign_keys:
        lines.append("Relationships:")
        for item in foreign_keys[:50]:
            lines.append(
                "- "
                f"{item.get('source_table')}.{item.get('source_column')} -> "
                f"{item.get('target_table')}.{item.get('target_column')}"
            )
    if context.db_memory_refs:
        lines.append("Database memory:")
        for ref in context.db_memory_refs:
            lines.append(
                "- "
                f"{ref.get('kind')} {ref.get('key')}: "
                f"{str(ref.get('text') or '').strip()}"
            )
    if context.column_value_hints:
        lines.append("Known filter values:")
        for hint in context.column_value_hints[:20]:
            values = []
            for item in hint.get("observed_values", []) or []:
                if isinstance(item, dict):
                    label = str(item.get("value"))
                    if item.get("count") is not None:
                        label = f"{label} ({item.get('count')})"
                    values.append(label)
                else:
                    values.append(str(item))
            if values:
                lines.append(
                    f"- {hint.get('table')}.{hint.get('column')}: "
                    + ", ".join(values[:25])
                )
    referents = context.session_context.get("referents")
    if isinstance(referents, dict):
        tables = referents.get("tables") or []
        if tables:
            lines.append("Session table referents:")
            for table in tables[:20]:
                lines.append(f"- {table}")
        monitors = referents.get("monitors") or []
        if monitors:
            lines.append("Session monitor referents:")
            for monitor in monitors[:10]:
                lines.append(f"- {monitor}")
    query_scopes = context.session_context.get("query_scopes")
    if isinstance(query_scopes, list) and query_scopes:
        lines.append("Session query scopes:")
        for scope in query_scopes[:4]:
            if not isinstance(scope, dict):
                continue
            parts = []
            tables = scope.get("tables") or []
            if tables:
                parts.append("tables " + ", ".join(str(item) for item in tables[:8]))
            filter_text = _session_filter_text(scope.get("filters"))
            if filter_text:
                parts.append("filters " + filter_text)
            row_count = scope.get("result_row_count")
            if isinstance(row_count, int):
                parts.append(f"result rows {row_count}")
            if parts:
                lines.append("- " + "; ".join(parts))
    return "\n".join(lines)


def _session_filter_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    filters = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        operator = str(item.get("operator") or "").strip()
        values = item.get("values")
        if not column or not operator or not isinstance(values, list) or not values:
            continue
        filters.append(f"{column} {operator} {', '.join(str(v) for v in values[:8])}")
    return "; ".join(filters)


def _compact_session_context(request: DbRequest) -> dict[str, Any]:
    session_context = db_session_context_from_request(request)
    context: object
    if session_context is not None:
        context = session_context.to_request_dict()
    else:
        context = getattr(request, "session_context", None)
    if context is None:
        return {}
    if not isinstance(context, dict):
        return {}
    referents = context.get("referents")
    recent_operations = context.get("recent_operations")
    query_scopes = context.get("query_scopes")
    durable_ids = context.get("durable_ids")
    diagnostics = context.get("diagnostics")
    compact: dict[str, Any] = {}
    for key in ("session_id", "user_id"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            compact[key] = value.strip()
    if isinstance(referents, dict):
        compact["referents"] = {
            key: list(referents.get(key) or [])[:20]
            for key in (
                "tables",
                "columns",
                "schemas",
                "metrics",
                "monitors",
                "approvals",
                "operations",
            )
        }
    if isinstance(recent_operations, list):
        compact["recent_operations"] = recent_operations[:8]
    if isinstance(query_scopes, list):
        compact["query_scopes"] = [
            dict(item) for item in query_scopes[:4] if isinstance(item, dict)
        ]
    if isinstance(durable_ids, dict):
        compact["durable_ids"] = dict(durable_ids)
    if isinstance(diagnostics, dict):
        compact["diagnostics"] = {
            "sources": list(diagnostics.get("sources") or []),
            "referent_sources": dict(diagnostics.get("referent_sources") or {}),
            "bounded": dict(diagnostics.get("bounded") or {}),
        }
        for key in (
            "conversation_message_count",
            "recent_operation_count",
            "evidence_operation_count",
            "query_scope_count",
        ):
            if key in diagnostics:
                compact["diagnostics"][key] = diagnostics[key]
    return compact


def _column_value_hints(
    catalog_evidence: tuple[Evidence, ...],
    schema: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    hints: list[dict[str, Any]] = []
    for evidence in catalog_evidence:
        if evidence.kind == "schema.column_value_hint":
            for hint in evidence.payload.get("hints", []) or []:
                if isinstance(hint, dict):
                    hints.append(_compact_column_value_hint(hint))

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for hint in hints:
        key = (str(hint.get("table") or ""), str(hint.get("column") or ""))
        if not key[0] or not key[1] or not planner_eligible_column_value_hint(hint):
            continue
        existing = deduped.get(key)
        if existing is None or _prefer_column_value_hint(hint, existing):
            deduped[key] = hint
    return tuple(deduped.values())


def _prefer_column_value_hint(
    candidate: dict[str, Any],
    existing: dict[str, Any],
) -> bool:
    if candidate.get("candidate_mapping") and not existing.get("candidate_mapping"):
        return True
    return False


def _compact_column_value_hint(hint: dict[str, Any]) -> dict[str, Any]:
    observed = []
    for item in hint.get("observed_values", []) or []:
        if isinstance(item, dict):
            observed.append(
                {
                    "value": item.get("value"),
                    **(
                        {"count": item.get("count")}
                        if item.get("count") is not None
                        else {}
                    ),
                }
            )
        else:
            observed.append({"value": item})
    result = {
        "table": hint.get("table"),
        "column": hint.get("column"),
        "profile_ref": hint.get("profile_ref")
        or f"{hint.get('table')}.{hint.get('column')}",
        "distinct_count": hint.get("distinct_count"),
        "observed_values": observed[:25],
        "profile_status": hint.get("profile_status") or "profiled",
        "sampled": bool(hint.get("sampled", False)),
        "truncated": bool(hint.get("truncated", False)),
        "redacted": bool(hint.get("redacted", False)),
        "stale": bool(hint.get("stale", False)),
    }
    if hint.get("stale_reason"):
        result["stale_reason"] = hint.get("stale_reason")
    if hint.get("candidate_mapping"):
        result["candidate_mapping"] = dict(hint["candidate_mapping"])
    return result


def planner_eligible_column_value_hint(hint: dict[str, Any]) -> bool:
    if hint.get("profile_status") != "profiled":
        return False
    if hint.get("stale") or hint.get("redacted") or hint.get("sampled"):
        return False
    if hint.get("truncated"):
        return False
    observed = hint.get("observed_values")
    if not isinstance(observed, list) or not observed or len(observed) > 25:
        return False
    for item in observed:
        value = item.get("value") if isinstance(item, dict) else item
        if value is None:
            return False
    return True


def _evidence_refs(values: tuple[Evidence | None, ...]) -> tuple[str, ...]:
    return tuple(item.id for item in values if item is not None and item.id)


def _evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "task_id": evidence.task_id,
        "payload_fingerprint": evidence.metadata.get("payload_fingerprint")
        or persisted_fingerprint(evidence.payload),
    }


def _artifact_evidence(
    *,
    kind: str,
    operation_id: str,
    task_id: str | None,
    payload: dict[str, Any],
) -> Evidence:
    payload_fingerprint = persisted_fingerprint(payload)
    evidence_id = persisted_fingerprint(
        {
            "operation_id": operation_id,
            "task_id": task_id,
            "kind": kind,
            "payload": payload,
        }
    )
    return Evidence(
        id=f"evidence-{evidence_id}",
        kind=kind,
        owner="db_runtime",
        operation_id=operation_id,
        task_id=task_id,
        payload=payload,
        metadata={"payload_fingerprint": payload_fingerprint},
    )


def _from_db_options(metadata: dict[str, Any]) -> dict[str, Any]:
    options = metadata.get("from_db_options")
    return options if isinstance(options, dict) else {}


def _float_option(options: dict[str, Any], key: str, default: float) -> float:
    value = options.get(key)
    if value is None:
        return default
    return float(value)
