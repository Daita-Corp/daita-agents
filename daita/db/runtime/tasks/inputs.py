"""Executable input hydration for ``DbRuntime`` tasks."""

from __future__ import annotations

from typing import Any

from daita.runtime import Operation, Task

from ...sql_evidence import blocked_scope_resources, sql_validation_facts_from_evidence
from .common import _payload_fingerprint


class DbRuntimeTaskInputMixin:
    async def executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        """Hydrate DB task input from authoritative validation evidence."""
        return await self._executable_input_for_task(task, operation)

    async def _executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        if task.capability_id in {
            "catalog.schema.search",
            "catalog.asset.inspect",
            "catalog.relationship_paths.find",
            "catalog.column_values.search",
            "catalog.column_value_hints.resolve",
            "catalog.value_grounding.plan",
        }:
            return await self._catalog_executable_input_for_task(task, operation)
        if task.capability_id not in {
            "db.sql.execute_read",
            "db.sql.execute_write",
        }:
            return task.input
        validation_dependency = next(
            (
                dependency
                for dependency in task.dependencies
                if dependency.kind.value == "evidence"
                and dependency.evidence_kind == "sql.validation"
            ),
            None,
        )
        validation = (
            await self._accepted_evidence_for_dependency(
                operation.id,
                validation_dependency,
            )
            if validation_dependency is not None
            else await self._latest_accepted_evidence(
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
                for item in await self.store.list_evidence(operation.id)
                if item.kind == "monitor.write_proposal"
                and item.id == proposal_evidence_id
            ]
            proposal = proposal_matches[-1] if proposal_matches else None
            expected_validation_id = str(
                task.metadata.get("validation_evidence_id") or ""
            )
            expected_validation_fingerprint = str(
                task.metadata.get("validation_payload_fingerprint") or ""
            )
            actual_validation_fingerprint = validation.metadata.get(
                "payload_fingerprint"
            ) or _payload_fingerprint(validation.payload)
            if (
                proposal is None
                or proposal.payload.get("status")
                not in {"approval_required", "approved"}
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
