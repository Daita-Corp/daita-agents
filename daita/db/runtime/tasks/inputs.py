"""Executable input hydration for ``DbRuntime`` tasks."""

from __future__ import annotations

from typing import Any

from daita.runtime import Operation, Task

from ...fingerprints import persisted_fingerprint
from ...sql_evidence import blocked_scope_resources, sql_validation_facts_from_evidence
from .context import DbTaskContext
from .evidence import accepted_evidence_for_dependency, latest_accepted_evidence

_PHASE2_CATALOG_CAPABILITIES = frozenset(
    {
        "catalog.schema.search",
        "catalog.asset.inspect",
        "catalog.relationship_paths.find",
        "catalog.column_values.search",
    }
)
_LEGACY_CATALOG_CAPABILITIES = frozenset(
    {
        *_PHASE2_CATALOG_CAPABILITIES,
        "catalog.column_value_hints.resolve",
        "catalog.value_grounding.plan",
    }
)
_PHASE2_SOURCE_OWNERS = frozenset({"sqlite", "postgresql"})


async def executable_input_for_task(
    context: DbTaskContext,
    task: Task,
    operation: Operation,
) -> dict[str, Any]:
    """Hydrate DB task input from authoritative validation evidence."""
    if task.capability_id in _LEGACY_CATALOG_CAPABILITIES:
        source_owner = _catalog_source_owner(context, task, operation)
        if source_owner in _PHASE2_SOURCE_OWNERS:
            if task.capability_id not in _PHASE2_CATALOG_CAPABILITIES:
                raise RuntimeError(
                    f"{task.capability_id} is not part of the Phase 2 operation surface"
                )
            if not task.input.get("store_id"):
                raise RuntimeError(
                    "Phase 2 catalog task is missing runtime-bound input"
                )
            return task.input
        # Phase 5 deletion condition: remove this explicit unsupported-connector
        # quarantine after those connectors adopt the closed recipe owner.
        from .catalog import catalog_executable_input_for_task

        return await catalog_executable_input_for_task(context, task, operation)
    if task.capability_id not in {
        "db.sql.execute_read",
        "db.sql.execute_write",
    }:
        return task.input
    validation_dependency = next(
        (
            dependency
            for dependency in task.dependencies
            if dependency.kind_value == "evidence"
            and dependency.evidence_kind == "sql.validation"
        ),
        None,
    )
    validation = (
        await accepted_evidence_for_dependency(
            context,
            operation.id,
            validation_dependency,
        )
        if validation_dependency is not None
        else await latest_accepted_evidence(
            context,
            operation.id,
            "sql.validation",
            payload={"valid": True},
        )
    )
    if validation is None:
        return task.input
    if task.metadata.get("monitor_action_role") == "write_execution":
        proposal_fingerprint = str(task.metadata.get("proposal_fingerprint") or "")
        proposal_evidence_id = str(task.metadata.get("proposal_evidence_id") or "")
        proposal_matches = [
            item
            for item in await context.store.list_evidence(operation.id)
            if item.kind == "monitor.write_proposal" and item.id == proposal_evidence_id
        ]
        proposal = proposal_matches[-1] if proposal_matches else None
        expected_validation_id = str(task.metadata.get("validation_evidence_id") or "")
        expected_validation_fingerprint = str(
            task.metadata.get("validation_payload_fingerprint") or ""
        )
        actual_validation_fingerprint = validation.metadata.get(
            "payload_fingerprint"
        ) or persisted_fingerprint(validation.payload)
        if (
            proposal is None
            or proposal.payload.get("status") not in {"approval_required", "approved"}
            or proposal.payload.get("proposal_fingerprint") != proposal_fingerprint
            or validation.id != expected_validation_id
            or actual_validation_fingerprint != expected_validation_fingerprint
            or proposal.payload.get("validation_payload_fingerprint")
            != expected_validation_fingerprint
        ):
            raise RuntimeError("monitor_write_proposal_stale")
        facts = sql_validation_facts_from_evidence(validation)
        blocked_resources = blocked_scope_resources(
            facts.target_resources,
            tuple(task.metadata.get("source_scope") or ()),
        )
        if facts.valid is not True or blocked_resources:
            raise RuntimeError("monitor_write_validation_stale")
    sql = validation.payload.get("sql")
    if not sql:
        return task.input
    return {
        **task.input,
        "sql": sql,
        "validated_evidence_id": validation.id,
        "validated_task_id": validation.task_id,
    }


def _catalog_source_owner(
    context: DbTaskContext,
    task: Task,
    operation: Operation,
) -> str:
    candidates = (
        task.metadata.get("source_owner"),
        operation.metadata.get("source_owner"),
        (operation.metadata.get("planned_operation") or {}).get("source_owner"),
        (operation.metadata.get("loop_state") or {}).get("source_owner"),
        task.metadata.get("owner"),
    )
    for candidate in candidates:
        if candidate:
            return str(candidate)
    connector_ids = [
        manifest.id
        for manifest in context.registry.manifests
        if str(getattr(manifest.kind, "value", manifest.kind)) == "connector"
        and manifest.id not in {"catalog", "db_runtime"}
    ]
    if len(connector_ids) == 1 and connector_ids[0] not in _PHASE2_SOURCE_OWNERS:
        return connector_ids[0]
    raise RuntimeError(
        "Catalog prerequisite routing requires explicit connector identity"
    )
