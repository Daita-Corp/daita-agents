"""Planner-facing summary helpers for the DB agent loop."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from daita.runtime import Evidence, Task

from ..fingerprints import persisted_fingerprint
from ..monitor_commands.naming import (
    MAX_MONITOR_DISPLAY_NAME_LENGTH,
    MAX_MONITOR_ID_LENGTH,
)
from ..planner_protocol import DbLoopState
from ..planning_context import (
    catalog_schema_from_evidence,
    planner_eligible_column_value_hint,
)
from .utils import (
    _dedupe_dicts,
    _dedupe_json_values,
    _optional_string,
    _safe_iterable,
    _string_list,
)

_CATALOG_CONTEXT_ASSET_LIMIT = 8
_CATALOG_CONTEXT_COLUMN_LIMIT = 20
_CATALOG_CONTEXT_EVIDENCE_LIMIT = 8
_MONITOR_SUMMARY_ITEM_LIMIT = 20
_MONITOR_SUMMARY_CANDIDATE_LIMIT = 10
_MONITOR_SUMMARY_ERROR_LIMIT = 10
_SUMMARY_STRING_LIMIT = 256
_CATALOG_STRUCTURAL_EVIDENCE_KINDS = frozenset(
    {
        "schema.asset_profile",
        "schema.search_result",
        "catalog.source_registered",
        "catalog.profile",
    }
)


def _capability_summary(capability: Any) -> dict[str, Any]:
    return {
        "id": capability.id,
        "owner": capability.owner,
        "description": capability.description,
        "access": capability.access.value,
        "risk": capability.risk.value,
        "output_evidence": sorted(capability.output_evidence),
        "runtime_only": capability.runtime_only,
    }


def _task_summary(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "capability_id": task.capability_id,
        "executor_id": task.executor_id,
        "status": task.status.value,
        "metadata": {
            key: task.metadata.get(key)
            for key in (
                "owner",
                "reason",
                "sequence",
                "planner_action_id",
                "planner_action_kind",
            )
            if key in task.metadata
        },
    }


def _task_ref(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "capability_id": task.capability_id,
        "status": task.status.value,
    }


def _approval_task_id(approval: Any) -> str | None:
    direct = getattr(approval, "task_id", None)
    if direct:
        return str(direct)
    for source in (
        getattr(approval, "metadata", None),
        getattr(approval, "proposed_action", None),
    ):
        if isinstance(source, Mapping) and source.get("task_id"):
            return str(source["task_id"])
    return None


def _catalog_context_for_state(evidence: tuple[Evidence, ...]) -> dict[str, Any]:
    """Project bounded catalog facts from the latest accepted planning context."""
    planning_context = next(
        (
            item
            for item in reversed(evidence)
            if item.accepted
            and item.kind == "planning.context"
            and isinstance(item.payload, Mapping)
        ),
        None,
    )
    if planning_context is None:
        return {}

    payload = planning_context.payload
    diagnostics = payload.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    referenced_ids = _safe_string_values(
        diagnostics.get("catalog_structural_evidence_refs")
    )
    if not referenced_ids:
        referenced_ids = _safe_string_values(payload.get("catalog_evidence_refs"))
    referenced_ids = sorted(set(referenced_ids))
    if not referenced_ids:
        return {}

    accepted_by_id = {
        str(item.id): item
        for item in evidence
        if item.id
        and item.accepted
        and item.owner == "catalog"
        and item.kind in _CATALOG_STRUCTURAL_EVIDENCE_KINDS
    }
    normalized_by_evidence_id: dict[str, dict[str, Any]] = {}
    referenced: list[Evidence] = []
    omitted_reasons: dict[str, int] = {}
    for evidence_id in referenced_ids:
        item = accepted_by_id.get(evidence_id)
        if item is None:
            omitted_reasons["not_accepted_catalog_structural_evidence"] = (
                omitted_reasons.get("not_accepted_catalog_structural_evidence", 0) + 1
            )
            continue
        try:
            item_schema = catalog_schema_from_evidence((item,), ())
        except (AttributeError, TypeError, ValueError):
            omitted_reasons["catalog_normalization_failed"] = (
                omitted_reasons.get("catalog_normalization_failed", 0) + 1
            )
            continue
        referenced.append(item)
        normalized_by_evidence_id[evidence_id] = item_schema

    normalized_schema = catalog_schema_from_evidence(tuple(referenced), ())
    raw_tables = normalized_schema.get("tables")
    tables = [item for item in _safe_iterable(raw_tables) if isinstance(item, Mapping)]
    supporting_ids = sorted(str(item.id) for item in referenced if item.id)
    included_supporting_ids = supporting_ids[:_CATALOG_CONTEXT_EVIDENCE_LIMIT]

    evidence_ids_by_asset: dict[str, set[str]] = {}
    store_ids_by_asset: dict[str, set[str]] = {}
    for item in referenced:
        if not item.id:
            continue
        item_schema = normalized_by_evidence_id[str(item.id)]
        item_store_ids = _catalog_store_ids_for_evidence(item)
        for table in _safe_iterable(item_schema.get("tables")):
            if not isinstance(table, Mapping):
                continue
            name = _safe_optional_string(table.get("name"))
            metadata = table.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            asset_ref = _safe_optional_string(metadata.get("catalog_asset_ref")) or name
            for value in (name, asset_ref):
                if value:
                    evidence_ids_by_asset.setdefault(value.casefold(), set()).add(
                        str(item.id)
                    )
                    store_ids_by_asset.setdefault(value.casefold(), set()).update(
                        item_store_ids
                    )

    assets: list[dict[str, Any]] = []
    for table in tables:
        name, name_truncated = _bounded_optional_string(table.get("name"))
        if name is None:
            continue
        metadata = table.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        asset_ref, asset_ref_truncated = _bounded_optional_string(
            metadata.get("catalog_asset_ref")
        )
        asset_ref = asset_ref or name
        columns = []
        for column in _safe_iterable(table.get("columns")):
            if not isinstance(column, Mapping):
                continue
            column_name, column_name_truncated = _bounded_optional_string(
                column.get("name")
            )
            if column_name is None:
                continue
            column_summary: dict[str, Any] = {"name": column_name}
            column_type, column_type_truncated = _bounded_optional_string(
                column.get("data_type") or column.get("type")
            )
            if column_type is not None:
                column_summary["type"] = column_type
            truncated_fields = []
            if column_name_truncated:
                truncated_fields.append("name")
            if column_type_truncated:
                truncated_fields.append("type")
            if truncated_fields:
                column_summary["truncated_fields"] = truncated_fields
            columns.append(column_summary)
        columns.sort(
            key=lambda item: (
                str(item.get("name") or "").casefold(),
                str(item.get("type") or "").casefold(),
            )
        )
        evidence_ids = sorted(
            evidence_ids_by_asset.get(name.casefold(), set())
            | evidence_ids_by_asset.get(asset_ref.casefold(), set())
        )
        if not evidence_ids:
            evidence_ids = supporting_ids
        included_evidence_ids = evidence_ids[:_CATALOG_CONTEXT_EVIDENCE_LIMIT]
        asset_store_ids = sorted(
            store_ids_by_asset.get(name.casefold(), set())
            | store_ids_by_asset.get(asset_ref.casefold(), set())
        )
        included_asset_store_ids = asset_store_ids[:_CATALOG_CONTEXT_EVIDENCE_LIMIT]
        included_columns = columns[:_CATALOG_CONTEXT_COLUMN_LIMIT]
        asset = {
            "asset_ref": asset_ref,
            "name": name,
            "columns": included_columns,
            "column_count": len(columns),
            "included_column_count": len(included_columns),
            "columns_truncated": len(columns) > len(included_columns),
            "evidence_ids": included_evidence_ids,
            "evidence_count": len(evidence_ids),
            "evidence_truncated": len(evidence_ids) > len(included_evidence_ids),
        }
        truncated_fields = []
        if name_truncated:
            truncated_fields.append("name")
        if asset_ref_truncated:
            truncated_fields.append("asset_ref")
        if truncated_fields:
            asset["truncated_fields"] = truncated_fields
        if asset_store_ids:
            asset.update(
                {
                    "catalog_store_ids": included_asset_store_ids,
                    "catalog_store_count": len(asset_store_ids),
                    "catalog_stores_truncated": len(asset_store_ids)
                    > len(included_asset_store_ids),
                }
            )
            if len(asset_store_ids) == 1:
                asset["catalog_store_id"] = asset_store_ids[0]
        assets.append(asset)
    assets.sort(
        key=lambda item: (
            str(item.get("name") or "").casefold(),
            str(item.get("asset_ref") or "").casefold(),
        )
    )
    included_assets = assets[:_CATALOG_CONTEXT_ASSET_LIMIT]
    candidate_sources, source_truncated = _catalog_candidate_metadata(tuple(referenced))
    included_candidate_sources = candidate_sources[:_CATALOG_CONTEXT_EVIDENCE_LIMIT]
    store_ids = _catalog_store_ids(tuple(referenced))
    included_store_ids = store_ids[:_CATALOG_CONTEXT_EVIDENCE_LIMIT]
    structural_source = (
        _safe_optional_string(diagnostics.get("structural_schema_source")) or "catalog"
    )
    schema_fingerprint = _safe_optional_string(
        payload.get("schema_fingerprint") or diagnostics.get("schema_fingerprint")
    )

    result: dict[str, Any] = {
        "planning_context_evidence_id": planning_context.id,
        "structural_source": structural_source,
        "database_type": normalized_schema.get("database_type"),
        "database_dialect": normalized_schema.get("database_dialect")
        or normalized_schema.get("sql_dialect")
        or normalized_schema.get("database_type"),
        "assets": included_assets,
        "candidate_count": len(assets),
        "included_candidate_count": len(included_assets),
        "truncated": source_truncated or len(assets) > len(included_assets),
        "candidate_sources": included_candidate_sources,
        "candidate_source_count": len(candidate_sources),
        "included_candidate_source_count": len(included_candidate_sources),
        "candidate_sources_truncated": len(candidate_sources)
        > len(included_candidate_sources),
        "supporting_catalog_evidence_ids": included_supporting_ids,
        "supporting_evidence_count": len(supporting_ids),
        "supporting_evidence_truncated": len(supporting_ids)
        > len(included_supporting_ids),
        "referenced_evidence_count": len(referenced_ids),
        "omitted_evidence_count": len(referenced_ids) - len(supporting_ids),
        "omitted_evidence_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(omitted_reasons.items())
        ],
    }
    if schema_fingerprint is not None:
        result["schema_fingerprint"] = schema_fingerprint
    if store_ids:
        result["catalog_store_ids"] = included_store_ids
        result["catalog_store_count"] = len(store_ids)
        result["catalog_stores_truncated"] = len(store_ids) > len(included_store_ids)
        if len(store_ids) == 1:
            result["catalog_store_id"] = store_ids[0]
    return result


def _catalog_candidate_metadata(
    evidence: tuple[Evidence, ...],
) -> tuple[list[dict[str, Any]], bool]:
    sources: list[dict[str, Any]] = []
    truncated = False
    for item in evidence:
        payload = item.payload if isinstance(item.payload, Mapping) else {}
        truncated = truncated or payload.get("truncated") is True
        if item.kind != "schema.search_result":
            continue
        item_schema = catalog_schema_from_evidence((item,), ())
        included_count = len(_safe_iterable(item_schema.get("tables")))
        value = payload.get("total_matches")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            value = included_count
        source_truncated = payload.get("truncated") is True or value > included_count
        truncated = truncated or source_truncated
        store_ids = sorted(_catalog_store_ids_for_evidence(item))
        included_store_ids = store_ids[:_CATALOG_CONTEXT_EVIDENCE_LIMIT]
        source = {
            "evidence_id": item.id,
            "candidate_count": value,
            "included_candidate_count": included_count,
            "truncated": source_truncated,
            "catalog_store_ids": included_store_ids,
            "catalog_store_count": len(store_ids),
            "catalog_stores_truncated": len(store_ids) > len(included_store_ids),
        }
        if len(store_ids) == 1:
            source["catalog_store_id"] = store_ids[0]
        sources.append(source)
    sources.sort(key=lambda item: str(item.get("evidence_id") or "").casefold())
    return sources, truncated


def _catalog_store_ids(evidence: tuple[Evidence, ...]) -> list[str]:
    store_ids: set[str] = set()
    for item in evidence:
        store_ids.update(_catalog_store_ids_for_evidence(item))
    return sorted(store_ids)


def _catalog_store_ids_for_evidence(evidence: Evidence) -> set[str]:
    payload = evidence.payload if isinstance(evidence.payload, Mapping) else {}
    sources: list[Mapping[Any, Any]] = [payload]
    for key in ("asset", "table"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    return {
        store_id
        for source in sources
        if (store_id := _safe_optional_string(source.get("store_id"))) is not None
    }


def _evidence_summary(evidence: Evidence) -> dict[str, Any]:
    summary = {
        "id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "accepted": evidence.accepted,
        "task_id": evidence.task_id,
    }
    bounded_payload = _bounded_evidence_payload_summary(evidence)
    if bounded_payload is not None:
        summary.update(bounded_payload)
        payload_fingerprint = evidence.metadata.get("payload_fingerprint")
        if payload_fingerprint:
            summary["payload_fingerprint"] = str(payload_fingerprint)
        task_input_hash = evidence.metadata.get("task_input_hash")
        if task_input_hash:
            summary["task_input_hash"] = str(task_input_hash)
        return summary
    sql = _sql_from_evidence_payload(evidence.payload)
    if sql:
        summary["sql"] = sql
    if isinstance(evidence.payload, dict):
        if evidence.kind in {
            "sql.validation",
            "query.plan.validation",
            "query.plan.proposal",
        }:
            if "valid" in evidence.payload:
                summary["valid"] = evidence.payload.get("valid") is True
            if evidence.kind == "query.plan.validation":
                for key in (
                    "plan_evidence_id",
                    "planning_context_evidence_id",
                    "planning_context_fingerprint",
                    "schema_fingerprint",
                    "session_scope_binding_fingerprint",
                    "session_context_fingerprint",
                    "contract_fingerprint",
                ):
                    value = _optional_string(evidence.payload.get(key))
                    if value is not None:
                        summary[key] = value
            validation_facts = _safe_validation_items(
                evidence.payload.get("validation_facts")
            )
            validation_warnings = _safe_validation_items(
                evidence.payload.get("warnings")
            )
            validation_errors = _safe_validation_items(evidence.payload.get("errors"))
            if validation_facts:
                summary["validation_facts"] = validation_facts
            if validation_warnings:
                summary["warnings"] = validation_warnings
            if validation_errors:
                summary["validation_warnings"] = validation_errors
            if evidence.kind == "query.plan.proposal":
                for key in (
                    "planning_context_evidence_id",
                    "planning_context_fingerprint",
                    "schema_fingerprint",
                    "session_scope_binding_fingerprint",
                    "session_context_fingerprint",
                    "contract_fingerprint",
                ):
                    value = _optional_string(evidence.payload.get(key))
                    if value is not None:
                        summary[key] = value
                structured_plan = evidence.payload.get("structured_plan")
                if isinstance(structured_plan, Mapping):
                    selected_tables = _string_list(
                        structured_plan.get("selected_tables")
                    )
                    if selected_tables:
                        summary["selected_tables"] = selected_tables
                    joins = _safe_join_summaries(structured_plan.get("joins"))
                    if joins:
                        summary["joins"] = joins
        if evidence.kind == "schema.column_value_hint":
            hints = _safe_column_value_hint_summaries(evidence.payload.get("hints"))
            if hints:
                summary["hints"] = hints
        if evidence.kind == "planning.context":
            memory_refs = evidence.payload.get("db_memory_refs")
            if isinstance(memory_refs, list):
                summary["db_memory_refs"] = [
                    dict(item) for item in memory_refs if isinstance(item, Mapping)
                ]
            session_context = evidence.payload.get("session_context")
            if isinstance(session_context, Mapping):
                summary["session_context_fingerprint"] = persisted_fingerprint(
                    session_context
                )
            for key in (
                "schema_fingerprint",
                "session_scope_binding_fingerprint",
                "contract_fingerprint",
            ):
                value = _optional_string(evidence.payload.get(key))
                if value is not None:
                    summary[key] = value
            for key in (
                "schema_evidence_refs",
                "catalog_evidence_refs",
                "relationship_evidence_refs",
            ):
                values = _string_list(evidence.payload.get(key))
                if values:
                    summary[key] = values
            relationship_joins = _planning_context_relationship_join_summaries(
                evidence.payload
            )
            if relationship_joins:
                summary["relationship_joins"] = relationship_joins
            diagnostics = evidence.payload.get("diagnostics")
            if isinstance(diagnostics, Mapping):
                structural_source = _optional_string(
                    diagnostics.get("structural_schema_source")
                )
                if structural_source is not None:
                    summary["structural_schema_source"] = structural_source
                catalog_structural_refs = _string_list(
                    diagnostics.get("catalog_structural_evidence_refs")
                )
                if catalog_structural_refs:
                    summary["catalog_structural_evidence_refs"] = (
                        catalog_structural_refs
                    )
                repair = diagnostics.get("validation_grounding_repair")
                if isinstance(repair, Mapping):
                    fingerprint = str(repair.get("fingerprint") or "").strip()
                    if fingerprint:
                        summary["validation_grounding_fingerprint"] = fingerprint
                    target_refs = [
                        str(item).strip()
                        for item in repair.get("target_refs", ())
                        if str(item).strip()
                    ]
                    if target_refs:
                        summary["validation_grounding_target_refs"] = target_refs
        if evidence.kind == "query.result":
            plan_evidence_id = _optional_string(
                evidence.payload.get("plan_evidence_id")
                or evidence.metadata.get("plan_evidence_id")
            )
            if plan_evidence_id is not None:
                summary["plan_evidence_id"] = plan_evidence_id
        if evidence.kind == "db.memory.proposal":
            proposal_fingerprint = evidence.payload.get("proposal_fingerprint")
            if isinstance(proposal_fingerprint, str) and proposal_fingerprint.strip():
                summary["proposal_fingerprint"] = proposal_fingerprint.strip()
        if evidence.kind == "query.plan.repair":
            for key in (
                "failure_evidence_id",
                "prior_plan_evidence_id",
                "planning_context_evidence_id",
            ):
                value = _optional_string(evidence.payload.get(key))
                if value is not None:
                    summary[key] = value
        if evidence.kind == "db.memory.definition":
            proposal_evidence_id = evidence.payload.get(
                "proposal_evidence_id"
            ) or evidence.metadata.get("proposal_evidence_id")
            if proposal_evidence_id:
                summary["proposal_evidence_id"] = str(proposal_evidence_id)
            proposal_fingerprint = evidence.payload.get(
                "proposal_fingerprint"
            ) or evidence.metadata.get("proposal_fingerprint")
            if isinstance(proposal_fingerprint, str) and proposal_fingerprint.strip():
                summary["proposal_fingerprint"] = proposal_fingerprint.strip()
        operation = _optional_string(evidence.payload.get("operation"))
        if operation is not None:
            summary["operation"] = operation
        if evidence.kind == "schema.relationship_path":
            summary["reachable"] = evidence.payload.get("reachable")
            joins = _relationship_path_join_summaries(evidence.payload)
            if joins:
                summary["joins"] = joins
    payload_fingerprint = evidence.metadata.get("payload_fingerprint")
    if payload_fingerprint:
        summary["payload_fingerprint"] = str(payload_fingerprint)
    task_input_hash = evidence.metadata.get("task_input_hash")
    if task_input_hash:
        summary["task_input_hash"] = str(task_input_hash)
    return summary


def _bounded_evidence_payload_summary(
    evidence: Evidence,
) -> dict[str, Any] | None:
    payload = evidence.payload if isinstance(evidence.payload, Mapping) else {}
    if evidence.kind == "monitor.listing":
        return _monitor_listing_summary(payload)
    if evidence.kind == "monitor.inspection":
        return _monitor_inspection_summary(payload)
    if evidence.kind == "monitor.proposal":
        return _monitor_proposal_summary(payload)
    if evidence.kind == "monitor.approval_state":
        return _monitor_approval_state_summary(payload)
    if evidence.kind == "monitor.approval_resolution":
        return _monitor_approval_resolution_summary(payload)
    if evidence.kind == "schema.asset_profile" and (
        not evidence.accepted or payload.get("success") is False
    ):
        return _failed_asset_profile_summary(payload)
    return None


def _monitor_listing_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_monitors, included, truncated = _project_bounded_collection(
        payload.get("monitors"),
        projector=_safe_monitor_identities,
        limit=_MONITOR_SUMMARY_ITEM_LIMIT,
    )
    return {
        "monitors": included,
        "monitor_count": len(raw_monitors),
        "included_monitor_count": len(included),
        "monitors_truncated": truncated,
    }


def _monitor_inspection_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    resolution = payload.get("resolution")
    resolution = resolution if isinstance(resolution, Mapping) else {}
    safe_resolution: dict[str, Any] = {}
    for key, max_length in (
        ("monitor_id", MAX_MONITOR_ID_LENGTH),
        ("monitor_ref", _SUMMARY_STRING_LIMIT),
        ("resolution_source", _SUMMARY_STRING_LIMIT),
        ("definition_evidence_id", _SUMMARY_STRING_LIMIT),
        ("proposal_evidence_id", _SUMMARY_STRING_LIMIT),
        ("operation_id", _SUMMARY_STRING_LIMIT),
    ):
        _set_bounded_string(
            safe_resolution,
            key,
            resolution.get(key),
            max_length=max_length,
        )
    raw_matches, included_matches, matches_truncated = _project_bounded_collection(
        resolution.get("matches"),
        projector=_safe_monitor_candidates,
        limit=_MONITOR_SUMMARY_CANDIDATE_LIMIT,
    )
    safe_resolution.update(
        {
            "matches": included_matches,
            "match_count": len(raw_matches),
            "included_match_count": len(included_matches),
            "matches_truncated": matches_truncated,
        }
    )
    for key, singular in (("warnings", "warning"), ("errors", "error")):
        raw_values = _safe_iterable(resolution.get(key))
        values, text_truncated_count = _bounded_string_values(raw_values)
        included_values = values[:_MONITOR_SUMMARY_ERROR_LIMIT]
        safe_resolution[key] = included_values
        safe_resolution[f"{singular}_count"] = len(raw_values)
        safe_resolution[f"included_{singular}_count"] = len(included_values)
        safe_resolution[f"{singular}_text_truncated_count"] = text_truncated_count
        safe_resolution[f"{key}_truncated"] = (
            len(raw_values) > len(included_values) or text_truncated_count > 0
        )
    if safe_resolution:
        result["resolution"] = safe_resolution

    inspection = payload.get("inspection")
    inspection = inspection if isinstance(inspection, Mapping) else {}
    monitor = _safe_monitor_identity(
        inspection.get("monitor") if "monitor" in inspection else payload.get("monitor")
    )
    if monitor:
        result.update(
            {
                "monitor_id": monitor.get("id"),
                "monitor_name": monitor.get("name"),
                "monitor_status": monitor.get("status"),
            }
        )
        if monitor.get("truncated_fields"):
            result["monitor_truncated_fields"] = monitor["truncated_fields"]
    elif safe_resolution.get("monitor_id"):
        result["monitor_id"] = safe_resolution["monitor_id"]
    return {key: value for key, value in result.items() if value is not None}


def _monitor_proposal_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    action = _safe_optional_string(payload.get("action"))
    if action is None:
        operation_type = _safe_optional_string(payload.get("operation_type"))
        action = operation_type.rsplit(".", 1)[-1] if operation_type else "create"
    monitor_id, monitor_id_truncated = _bounded_optional_string(
        payload.get("monitor_id"),
        max_length=MAX_MONITOR_ID_LENGTH,
    )
    if monitor_id is None:
        for key in ("after", "before"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                monitor_id, monitor_id_truncated = _bounded_optional_string(
                    value.get("id"),
                    max_length=MAX_MONITOR_ID_LENGTH,
                )
                if monitor_id is not None:
                    break
    validation = payload.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    raw_errors = _safe_iterable(validation.get("errors"))
    errors, error_text_truncated_count = _safe_error_codes(raw_errors)
    included_errors = errors[:_MONITOR_SUMMARY_ERROR_LIMIT]

    raw_candidates_value = payload.get("candidates")
    if raw_candidates_value is None:
        raw_candidates_value = payload.get("matches")
    if raw_candidates_value is None:
        diagnostics = validation.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            raw_candidates_value = diagnostics.get("candidates")
    raw_candidates, included_candidates, candidates_truncated = (
        _project_bounded_collection(
            raw_candidates_value,
            projector=_safe_monitor_candidates,
            limit=_MONITOR_SUMMARY_CANDIDATE_LIMIT,
        )
    )
    candidate_count = len(raw_candidates)
    reported_candidate_count = payload.get("candidate_count")
    if (
        isinstance(reported_candidate_count, int)
        and not isinstance(reported_candidate_count, bool)
        and reported_candidate_count >= candidate_count
    ):
        candidate_count = reported_candidate_count
    result: dict[str, Any] = {
        "action": action,
        "validation_errors": included_errors,
        "validation_error_count": len(raw_errors),
        "included_validation_error_count": len(included_errors),
        "validation_error_text_truncated_count": error_text_truncated_count,
        "validation_errors_truncated": len(raw_errors) > len(included_errors)
        or error_text_truncated_count > 0,
        "candidates": included_candidates,
        "candidate_count": candidate_count,
        "included_candidate_count": len(included_candidates),
        "candidates_truncated": candidates_truncated
        or payload.get("candidates_truncated") is True
        or candidate_count > len(included_candidates),
    }
    if monitor_id is not None:
        result["monitor_id"] = monitor_id
    if monitor_id_truncated:
        result["truncated_fields"] = ["monitor_id"]
    _set_bounded_string(result, "monitor_ref", payload.get("monitor_ref"))
    _set_bounded_string(
        result,
        "resolution_source",
        payload.get("resolution_source"),
    )
    if payload.get("monitor_ref_truncated") is True:
        truncated_fields = result.setdefault("truncated_fields", [])
        if "monitor_ref" not in truncated_fields:
            truncated_fields.append("monitor_ref")
    return result


def _monitor_approval_state_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_approvals = _safe_iterable(payload.get("approvals"))
    approvals = [_safe_approval_summary(item) for item in raw_approvals]
    approvals = [item for item in approvals if item]
    approvals = _dedupe_dicts(
        approvals,
        keys=("approval_id", "target_operation_id", "monitor_id", "policy_id"),
    )
    approvals.sort(
        key=lambda item: (
            str(item.get("approval_id") or "").casefold(),
            str(item.get("approval_id") or ""),
            str(item.get("target_operation_id") or "").casefold(),
            str(item.get("target_operation_id") or ""),
        )
    )
    included = approvals[:_MONITOR_SUMMARY_ITEM_LIMIT]
    approval_count = len(raw_approvals)
    reported_approval_count = payload.get("approval_count")
    if (
        isinstance(reported_approval_count, int)
        and not isinstance(reported_approval_count, bool)
        and reported_approval_count >= approval_count
    ):
        approval_count = reported_approval_count
    result: dict[str, Any] = {
        "approvals": included,
        "approval_count": approval_count,
        "included_approval_count": len(included),
        "approvals_truncated": payload.get("approvals_truncated") is True
        or approval_count > len(included),
    }
    read_kind = _safe_optional_string(payload.get("read_kind"))
    if read_kind == "approvals":
        result["read_kind"] = read_kind
    if isinstance(payload.get("pending_only"), bool):
        result["pending_only"] = payload["pending_only"]
    _set_bounded_string(
        result,
        "monitor_id",
        payload.get("monitor_id"),
        max_length=MAX_MONITOR_ID_LENGTH,
    )
    return result


def _monitor_approval_resolution_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    bounded = _bounded_monitor_approval_resolution_payload(payload)
    result = {
        key: value
        for key, value in bounded.items()
        if key not in {"status", "operation_id", "truncated_fields"}
    }
    if "status" in bounded:
        result["resolution_status"] = bounded["status"]
    if "operation_id" in bounded:
        result["target_operation_id"] = bounded["operation_id"]
    truncated_fields = bounded.get("truncated_fields")
    if isinstance(truncated_fields, list):
        result["truncated_fields"] = [
            {
                "status": "resolution_status",
                "operation_id": "target_operation_id",
            }.get(field, field)
            for field in truncated_fields
        ]
    return result


def _bounded_monitor_approval_resolution_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source_key, target_key, max_length in (
        ("status", "status", _SUMMARY_STRING_LIMIT),
        ("approval_id", "approval_id", _SUMMARY_STRING_LIMIT),
        ("approval_status", "approval_status", _SUMMARY_STRING_LIMIT),
        ("operation_id", "operation_id", _SUMMARY_STRING_LIMIT),
    ):
        value = payload.get(source_key)
        if source_key == "status" and value is None:
            value = payload.get("resolution_status")
        if source_key == "operation_id" and value is None:
            value = payload.get("target_operation_id")
        setter = (
            _set_bounded_exact_string
            if source_key == "approval_id"
            else _set_bounded_string
        )
        setter(
            result,
            target_key,
            value,
            max_length=max_length,
        )
    _set_bounded_string(
        result,
        "approval_action",
        payload.get("approval_action"),
    )
    _set_bounded_string(
        result,
        "monitor_id",
        payload.get("monitor_id"),
        max_length=MAX_MONITOR_ID_LENGTH,
    )
    raw_approvals = _safe_iterable(payload.get("matched_approvals"))
    approvals = [_safe_approval_summary(item) for item in raw_approvals]
    approvals = [item for item in approvals if item]
    approvals.sort(
        key=lambda item: (
            str(item.get("approval_id") or "").casefold(),
            str(item.get("approval_id") or ""),
            str(item.get("target_operation_id") or "").casefold(),
            str(item.get("target_operation_id") or ""),
        )
    )
    included = approvals[:_MONITOR_SUMMARY_ITEM_LIMIT]
    matched_approval_count = len(raw_approvals)
    reported_matched_count = payload.get("matched_approval_count")
    if (
        isinstance(reported_matched_count, int)
        and not isinstance(reported_matched_count, bool)
        and reported_matched_count >= matched_approval_count
    ):
        matched_approval_count = reported_matched_count
    result.update(
        {
            "matched_approvals": included,
            "matched_approval_count": matched_approval_count,
            "included_matched_approval_count": len(included),
            "matched_approvals_truncated": payload.get("matched_approvals_truncated")
            is True
            or matched_approval_count > len(included),
        }
    )
    return result


def _failed_asset_profile_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    requested_asset = _safe_optional_string(
        payload.get("asset_ref")
        or payload.get("requested_asset")
        or payload.get("target")
    )
    raw_candidates, included, truncated = _project_bounded_collection(
        payload.get("candidates"),
        projector=_safe_catalog_candidates,
        limit=_MONITOR_SUMMARY_CANDIDATE_LIMIT,
    )
    result: dict[str, Any] = {
        "candidates": included,
        "candidate_count": len(raw_candidates),
        "included_candidate_count": len(included),
        "candidates_truncated": truncated,
    }
    if requested_asset is not None:
        result["requested_asset"] = requested_asset
    return result


def _safe_monitor_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for source_key, target_key, max_length in (
        ("id", "id", MAX_MONITOR_ID_LENGTH),
        ("monitor_id", "id", MAX_MONITOR_ID_LENGTH),
        ("name", "name", MAX_MONITOR_DISPLAY_NAME_LENGTH),
        ("status", "status", _SUMMARY_STRING_LIMIT),
    ):
        if target_key in result:
            continue
        _set_bounded_string(
            result,
            target_key,
            value.get(source_key),
            max_length=max_length,
        )
    return result


def _safe_monitor_identities(value: Any) -> list[dict[str, Any]]:
    monitors = [
        monitor
        for item in _safe_iterable(value)
        if (monitor := _safe_monitor_identity(item))
    ]
    monitors = _dedupe_dicts(monitors, keys=("id", "name", "status"))
    monitors.sort(
        key=lambda item: (
            str(item.get("id") or "").casefold(),
            str(item.get("name") or "").casefold(),
            str(item.get("status") or "").casefold(),
        )
    )
    return monitors


def _project_bounded_collection(
    value: Any,
    *,
    projector: Callable[[Any], list[dict[str, Any]]],
    limit: int,
) -> tuple[list[Any], list[dict[str, Any]], bool]:
    raw_items = _safe_iterable(value)
    projected = projector(raw_items)
    included = projected[:limit]
    return raw_items, included, len(raw_items) > len(included)


def _safe_monitor_candidates(value: Any) -> list[dict[str, Any]]:
    candidates = []
    for item in _safe_iterable(value):
        if isinstance(item, Mapping):
            candidate_id, candidate_id_truncated = _bounded_optional_string(
                item.get("monitor_id") or item.get("id"),
                max_length=MAX_MONITOR_ID_LENGTH,
            )
            name, name_truncated = _bounded_optional_string(
                item.get("name"),
                max_length=MAX_MONITOR_DISPLAY_NAME_LENGTH,
            )
            source_truncated_fields = {
                field
                for field in _safe_iterable(item.get("truncated_fields"))
                if isinstance(field, str) and field in {"id", "name"}
            }
        elif isinstance(item, str):
            candidate_id, candidate_id_truncated = _bounded_optional_string(
                item,
                max_length=MAX_MONITOR_ID_LENGTH,
            )
            name = None
            name_truncated = False
            source_truncated_fields = set()
        else:
            continue
        candidate: dict[str, Any] = {}
        if candidate_id is not None:
            candidate["id"] = candidate_id
        if name is not None:
            candidate["name"] = name
        truncated_fields = []
        if candidate_id_truncated:
            truncated_fields.append("id")
        if name_truncated:
            truncated_fields.append("name")
        truncated_fields.extend(
            field
            for field in ("id", "name")
            if field in source_truncated_fields and field not in truncated_fields
        )
        if truncated_fields:
            candidate["truncated_fields"] = truncated_fields
        if candidate:
            candidates.append(candidate)
    candidates = _dedupe_dicts(candidates, keys=("id", "name"))
    candidates.sort(
        key=lambda item: (
            str(item.get("id") or "").casefold(),
            str(item.get("name") or "").casefold(),
        )
    )
    return candidates


def _safe_catalog_candidates(value: Any) -> list[dict[str, Any]]:
    candidates = []
    for item in _safe_iterable(value):
        if isinstance(item, Mapping):
            asset_ref, asset_ref_truncated = _bounded_optional_string(
                item.get("asset_ref") or item.get("name")
            )
            name, name_truncated = _bounded_optional_string(item.get("name"))
        elif isinstance(item, str):
            asset_ref, asset_ref_truncated = _bounded_optional_string(item)
            name = None
            name_truncated = False
        else:
            continue
        candidate: dict[str, Any] = {}
        if asset_ref is not None:
            candidate["asset_ref"] = asset_ref
        if name is not None:
            candidate["name"] = name
        truncated_fields = []
        if asset_ref_truncated:
            truncated_fields.append("asset_ref")
        if name_truncated:
            truncated_fields.append("name")
        if truncated_fields:
            candidate["truncated_fields"] = truncated_fields
        if candidate:
            candidates.append(candidate)
    candidates = _dedupe_dicts(candidates, keys=("asset_ref", "name"))
    candidates.sort(
        key=lambda item: (
            str(item.get("asset_ref") or "").casefold(),
            str(item.get("name") or "").casefold(),
        )
    )
    return candidates


def _safe_approval_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    context = value.get("context")
    context = context if isinstance(context, Mapping) else {}
    result: dict[str, Any] = {}
    for target_key, raw_value, max_length in (
        ("approval_id", value.get("approval_id"), _SUMMARY_STRING_LIMIT),
        (
            "target_operation_id",
            value.get("target_operation_id") or value.get("operation_id"),
            _SUMMARY_STRING_LIMIT,
        ),
        (
            "monitor_id",
            value.get("monitor_id") or context.get("monitor_id"),
            MAX_MONITOR_ID_LENGTH,
        ),
        (
            "policy_id",
            value.get("policy_id") or value.get("requested_by_policy_id"),
            _SUMMARY_STRING_LIMIT,
        ),
        ("status", value.get("status"), _SUMMARY_STRING_LIMIT),
    ):
        setter = (
            _set_bounded_exact_string
            if target_key == "approval_id"
            else _set_bounded_string
        )
        setter(
            result,
            target_key,
            raw_value,
            max_length=max_length,
        )
    return result


def _safe_error_codes(value: Any) -> tuple[list[str], int]:
    errors = []
    truncated_count = 0
    for item in _safe_iterable(value):
        if isinstance(item, str):
            code, truncated = _bounded_optional_string(item)
        elif isinstance(item, Mapping):
            code, truncated = _bounded_optional_string(
                item.get("code") or item.get("kind")
            )
        else:
            continue
        if code is not None:
            errors.append(code)
            truncated_count += int(truncated)
    return sorted(set(errors), key=str.casefold), truncated_count


def _bounded_string_values(value: Any) -> tuple[list[str], int]:
    values = []
    truncated_count = 0
    for item in _safe_iterable(value):
        text, truncated = _bounded_optional_string(item)
        if text is None:
            continue
        values.append(text)
        truncated_count += int(truncated)
    return sorted(set(values), key=str.casefold), truncated_count


def _safe_string_values(value: Any) -> list[str]:
    values = {
        item.strip()
        for item in _safe_iterable(value)
        if isinstance(item, str) and item.strip()
    }
    return sorted(values, key=str.casefold)


def _safe_optional_string(value: Any) -> str | None:
    return _bounded_optional_string(value)[0]


def _bounded_optional_string(
    value: Any,
    *,
    max_length: int = _SUMMARY_STRING_LIMIT,
) -> tuple[str | None, bool]:
    if not isinstance(value, str):
        return None, False
    text = value.strip()
    if not text:
        return None, False
    return text[:max_length], len(text) > max_length


def _set_bounded_string(
    result: dict[str, Any],
    key: str,
    value: Any,
    *,
    max_length: int = _SUMMARY_STRING_LIMIT,
) -> None:
    text, truncated = _bounded_optional_string(value, max_length=max_length)
    if text is None:
        return
    result[key] = text
    if truncated:
        fields = result.setdefault("truncated_fields", [])
        if key not in fields:
            fields.append(key)


def _set_bounded_exact_string(
    result: dict[str, Any],
    key: str,
    value: Any,
    *,
    max_length: int = _SUMMARY_STRING_LIMIT,
) -> None:
    if not isinstance(value, str) or value == "":
        return
    result[key] = value[:max_length]
    if len(value) > max_length:
        fields = result.setdefault("truncated_fields", [])
        if key not in fields:
            fields.append(key)


def _safe_validation_items(value: Any) -> list[Any]:
    items: list[Any] = []
    for item in _safe_iterable(value):
        if isinstance(item, Mapping):
            safe = {
                key: item[key]
                for key in (
                    "kind",
                    "table",
                    "table_name",
                    "column",
                    "column_name",
                    "operator",
                    "literal",
                    "value",
                    "filter_literal",
                    "candidates",
                    "source",
                    "reason",
                )
                if key in item
            }
            if safe:
                items.append(safe)
        elif isinstance(item, str) and item.strip():
            items.append(item.strip())
    return _dedupe_json_values(items)


def _safe_column_value_hint_summaries(value: Any) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for item in _safe_iterable(value):
        if not isinstance(item, Mapping):
            continue
        if not planner_eligible_column_value_hint(dict(item)):
            continue
        table = str(item.get("table") or "").strip()
        column = str(item.get("column") or "").strip()
        if not table or not column:
            continue
        hint: dict[str, Any] = {
            "table": table,
            "column": column,
        }
        observed_values = []
        for observed in _safe_iterable(item.get("observed_values")):
            if isinstance(observed, Mapping):
                if observed.get("value") is not None:
                    observed_values.append({"value": observed.get("value")})
            elif observed is not None:
                observed_values.append({"value": observed})
        if observed_values:
            hint["observed_values"] = observed_values[:25]
        candidate_mapping = item.get("candidate_mapping")
        if isinstance(candidate_mapping, Mapping):
            hint["candidate_mapping"] = dict(candidate_mapping)
        hints.append(hint)
    return _dedupe_dicts(hints, keys=("table", "column"))


def _safe_join_summaries(value: Any) -> list[dict[str, Any]]:
    joins: list[dict[str, Any]] = []
    for item in _safe_iterable(value):
        if not isinstance(item, Mapping):
            continue
        left_table = _optional_string(item.get("left_table") or item.get("left_asset"))
        right_table = _optional_string(
            item.get("right_table") or item.get("right_asset")
        )
        if not left_table or not right_table:
            continue
        join = {
            "left_table": left_table,
            "right_table": right_table,
        }
        left_column = _optional_string(
            item.get("left_column") or item.get("left_field") or item.get("left_key")
        )
        right_column = _optional_string(
            item.get("right_column") or item.get("right_field") or item.get("right_key")
        )
        if left_column:
            join["left_column"] = left_column.split(".")[-1]
        if right_column:
            join["right_column"] = right_column.split(".")[-1]
        joins.append(join)
    return _dedupe_dicts(
        joins, keys=("left_table", "left_column", "right_table", "right_column")
    )


def _relationship_path_join_summaries(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    joins: list[dict[str, Any]] = []
    for path in payload.get("paths", []) or []:
        if not isinstance(path, Mapping):
            continue
        joins.extend(_safe_join_summaries(path.get("joins")))
    return _dedupe_dicts(
        joins,
        keys=("left_table", "left_column", "right_table", "right_column"),
    )


def _planning_context_relationship_join_summaries(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    joins: list[dict[str, Any]] = []
    for detail in payload.get("relationship_evidence_details", []) or []:
        if not isinstance(detail, Mapping):
            continue
        joins.extend(_relationship_path_join_summaries(detail))
    return _dedupe_dicts(
        joins,
        keys=("left_table", "left_column", "right_table", "right_column"),
    )


def _evidence_ref(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "accepted": evidence.accepted,
    }


def _sql_from_evidence_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("sql", "selected_sql"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    structured_plan = payload.get("structured_plan")
    if isinstance(structured_plan, dict):
        selected_sql = structured_plan.get("selected_sql")
        if isinstance(selected_sql, str) and selected_sql.strip():
            return selected_sql.strip()
    return None


def _latest_accepted_evidence_summary(
    state: DbLoopState,
    kind: str,
) -> dict[str, Any] | None:
    for item in reversed(state.accepted_evidence_summaries):
        if item.get("kind") == kind and item.get("accepted", True) is True:
            return dict(item)
    return None


def _state_has_accepted_evidence(state: DbLoopState, kind: str) -> bool:
    return any(
        item.get("kind") == kind and item.get("accepted", True) is True
        for item in state.accepted_evidence_summaries
    )
