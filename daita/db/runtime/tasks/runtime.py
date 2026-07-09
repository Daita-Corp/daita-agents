"""Task execution helpers for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from typing import Any, Iterable, Mapping
from uuid import uuid4

from daita.runtime import (
    AccessMode,
    ApprovalStatus,
    Capability,
    Evidence,
    Operation,
    Task,
    TaskDependency,
    TaskStatus,
)

from ...models import DbIntent, DbIntentKind
from ...sql_evidence import blocked_scope_resources, sql_validation_facts_from_evidence
from .common import (
    _evidence_payload_fingerprint,
    _payload_contains,
    _payload_fingerprint,
    _stable_hash,
)
from .execution import DbRuntimeTaskExecutionMixin
from .planning import DbRuntimeTaskPlanningMixin
from .readiness import DbRuntimeTaskReadinessMixin

_CATALOG_COLUMN_VALUE_GROUNDING_REASON = "catalog_column_value_grounding"
_SOURCE_OWNER_KEYS = (
    "source_owner",
    "db_owner",
    "connector_owner",
    "source_capability_owner",
)
_SOURCE_OWNER_OPTION_KEYS = (
    *_SOURCE_OWNER_KEYS,
    "source_plugin_id",
    "source_plugin",
)


class DbRuntimeTasksMixin(
    DbRuntimeTaskExecutionMixin,
    DbRuntimeTaskPlanningMixin,
    DbRuntimeTaskReadinessMixin,
):
    async def _persist_verification_result_evidence(
        self,
        operation: Operation,
        verification: Any,
        evidence: tuple[Evidence, ...],
    ) -> Evidence:
        existing = await self._latest_accepted_evidence(
            operation.id,
            "verification.result",
            payload={"passed": True},
        )
        if existing is not None:
            return existing
        accepted = tuple(item for item in evidence if item.accepted and item.id)
        evidence_details = [
            {
                "id": item.id,
                "kind": item.kind,
                "owner": item.owner,
                "task_id": item.task_id,
                "payload_fingerprint": item.metadata.get("payload_fingerprint")
                or _payload_fingerprint(item.payload),
            }
            for item in accepted
        ]
        payload = {
            "passed": bool(verification.passed),
            "evidence_refs": [item["id"] for item in evidence_details],
            "evidence_details": evidence_details,
            "warnings": list(verification.warnings),
            "missing_evidence": list(verification.missing_evidence),
            "diagnostics": verification.diagnostics,
            "input_fingerprint": _stable_hash(
                {
                    "operation_id": operation.id,
                    "evidence": evidence_details,
                    "warnings": list(verification.warnings),
                    "missing_evidence": list(verification.missing_evidence),
                }
            ),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        verification_evidence = Evidence(
            id=f"evidence-{uuid4()}",
            kind="verification.result",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=True,
            payload=payload,
            metadata={
                "payload_fingerprint": _payload_fingerprint(payload),
                "input_fingerprint": payload["input_fingerprint"],
            },
        )
        await self.store.save_evidence(verification_evidence)
        return verification_evidence

    async def _execute_answer_synthesis(
        self,
        *,
        operation: Operation,
        intent: DbIntent,
        outcome_evidence: tuple[Evidence, ...],
    ) -> tuple[Evidence, Task]:
        existing = await self._latest_accepted_evidence(
            operation.id,
            "answer.synthesis",
        )
        if existing is not None:
            task = next(
                (
                    item
                    for item in await self.store.list_tasks(operation.id)
                    if item.id == existing.task_id
                ),
                Task(
                    id=str(existing.task_id or f"db-task-{uuid4()}"),
                    operation_id=operation.id,
                    capability_id="db.answer.synthesize",
                    executor_id="db.answer.synthesize.runtime",
                    required_evidence=frozenset({"answer.synthesis"}),
                    metadata={"owner": "db_runtime", "reason": "answer_synthesis"},
                ),
            )
            return existing, task

        capability = self.registry.get_capability(
            "db.answer.synthesize", owner="db_runtime"
        )
        dependencies = _synthesis_dependencies(operation, intent, outcome_evidence)
        task_input = {
            "evidence_refs": [
                {
                    "id": dependency.evidence_id,
                    "kind": dependency.evidence_kind,
                    "payload_fingerprint": dependency.payload_fingerprint,
                }
                for dependency in dependencies
            ],
            "row_budget": _synthesis_context_option(
                self.config.metadata, "synthesis_row_budget", 25
            ),
            "char_budget": _synthesis_context_option(
                self.config.metadata, "synthesis_context_char_budget", 16000
            ),
        }
        input_hash = _stable_hash(task_input)
        task = Task(
            id=f"db-task-{uuid4()}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**task_input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            dependencies=dependencies,
            metadata={
                "owner": capability.owner,
                "reason": "answer_synthesis",
                "sequence": 10_000,
                "input_hash": input_hash,
                "idempotency_key": _stable_hash(
                    {
                        "operation_id": operation.id,
                        "capability_id": capability.id,
                        "evidence_refs": task_input["evidence_refs"],
                    }
                ),
                "idempotent": capability.idempotent,
                "replay_safe": capability.replay_safe,
                "side_effecting": capability.side_effecting,
            },
        )
        evidence = await self.execute_task(
            task,
            operation,
            context={"capability_owner": capability.owner},
        )
        synthesis = next(
            (
                item
                for item in evidence
                if item.kind == "answer.synthesis" and item.accepted
            ),
            None,
        )
        if synthesis is None:
            raise RuntimeError("answer.synthesis evidence was not produced")
        stored_task = await self.store.load_task(task.id)
        return synthesis, stored_task or task

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

    async def _catalog_executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        task_input = _catalog_task_input_from_metadata(task, operation)
        store_id = _catalog_store_id(task_input, self.config.metadata)
        if not store_id:
            return task_input
        schema_evidence = await self._latest_evidence(
            operation.id,
            "schema.asset_profile",
            accepted=True,
        )
        if schema_evidence is None:
            schema_evidence = await self._ensure_schema_profile_evidence(
                operation,
                parent_task=task,
                prerequisite_for=task.capability_id,
                prerequisite_reason=_CATALOG_COLUMN_VALUE_GROUNDING_REASON,
            )
        if schema_evidence is not None:
            await self._ensure_catalog_source_registered(
                operation,
                store_id=store_id,
                schema_evidence=schema_evidence,
                parent_task=task,
                prerequisite_for=task.capability_id,
                prerequisite_reason=_CATALOG_COLUMN_VALUE_GROUNDING_REASON,
            )
        if task.capability_id == "catalog.value_grounding.plan":
            task_input = await self._value_grounding_plan_executable_input(
                task_input,
                operation=operation,
                schema_evidence=schema_evidence,
                parent_task=task,
            )
        elif task.capability_id in {
            "catalog.column_values.search",
            "catalog.column_value_hints.resolve",
        }:
            grounding_plan = await self._ensure_catalog_value_grounding_plan(
                operation,
                store_id=store_id,
                task_input=task_input,
                schema_evidence=schema_evidence,
                parent_task=task,
            )
            task_input = _column_value_task_input_with_grounding_plan(
                task_input,
                grounding_plan=grounding_plan,
            )
            await self._ensure_catalog_column_values_profiled(
                operation,
                store_id=store_id,
                task_input=task_input,
                parent_task=task,
            )
        return {**task_input, "store_id": store_id}

    async def _value_grounding_plan_executable_input(
        self,
        task_input: dict[str, Any],
        *,
        operation: Operation,
        schema_evidence: Evidence | None,
        parent_task: Task,
    ) -> dict[str, Any]:
        enriched = _value_grounding_plan_input_with_operation_context(
            task_input,
            operation=operation,
        )
        return enriched

    async def _ensure_catalog_value_grounding_plan(
        self,
        operation: Operation,
        *,
        store_id: str,
        task_input: dict[str, Any],
        schema_evidence: Evidence | None,
        parent_task: Task,
    ) -> Evidence | None:
        for dependency in parent_task.dependencies:
            if (
                dependency.kind.value == "evidence"
                and dependency.evidence_kind == "catalog.value_grounding.plan"
            ):
                evidence = await self._accepted_evidence_for_dependency(
                    operation.id,
                    dependency,
                )
                if evidence is not None:
                    return evidence

        capability = self.registry.get_capability(
            "catalog.value_grounding.plan",
            owner="catalog",
        )
        prerequisite_declaration = await self._require_declared_runtime_prerequisite(
            operation,
            parent_task=parent_task,
            capability=capability,
            prerequisite_for=parent_task.capability_id,
            prerequisite_reason=_CATALOG_COLUMN_VALUE_GROUNDING_REASON,
        )
        plan_input = {
            **task_input,
            "store_id": store_id,
        }
        plan_input = await self._value_grounding_plan_executable_input(
            plan_input,
            operation=operation,
            schema_evidence=schema_evidence,
            parent_task=parent_task,
        )
        input_hash = _stable_hash(plan_input)
        prerequisite_metadata = _runtime_prerequisite_task_metadata(
            operation,
            parent_task=parent_task,
            capability=capability,
            prerequisite_for=parent_task.capability_id,
            prerequisite_reason=_CATALOG_COLUMN_VALUE_GROUNDING_REASON,
            prerequisite_declaration=prerequisite_declaration,
        )
        idempotency_key = _stable_hash(
            {
                "operation_id": operation.id,
                "capability_id": capability.id,
                "owner": capability.owner,
                "parent_task_id": parent_task.id,
                "input_hash": input_hash,
            }
        )
        task_id = f"db-task-{_stable_hash({'idempotency_key': idempotency_key})[:32]}"
        existing = await self.store.load_task(task_id)
        if existing is not None:
            plan_task = await self._merge_runtime_prerequisite_metadata(
                existing,
                prerequisite_metadata,
            )
        else:
            plan_task = await self._plan_kernel_task(
                Task(
                    id=task_id,
                    operation_id=operation.id,
                    capability_id=capability.id,
                    executor_id=capability.executor,
                    input={**plan_input, "input_hash": input_hash},
                    required_evidence=capability.output_evidence,
                    metadata={
                        **prerequisite_metadata,
                        "owner": capability.owner,
                        "reason": "runtime:catalog_value_grounding_plan_prepare",
                        "sequence": 0,
                        "input_hash": input_hash,
                        "idempotency_key": idempotency_key,
                        "deterministic_key": (
                            "runtime:catalog.value_grounding.plan:"
                            f"{parent_task.capability_id}"
                        ),
                        "idempotent": capability.idempotent,
                        "replay_safe": capability.replay_safe,
                        "side_effecting": capability.side_effecting,
                    },
                )
            )
        if plan_task.status is not TaskStatus.SUCCEEDED:
            await self.execute_task(plan_task, operation)
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation.id)
            if evidence.kind == "catalog.value_grounding.plan"
            and evidence.accepted
            and evidence.task_id == plan_task.id
        ]
        return matches[-1] if matches else None

    async def _ensure_schema_profile_evidence(
        self,
        operation: Operation,
        *,
        parent_task: Task | None = None,
        prerequisite_for: str | None = None,
        prerequisite_reason: str = _CATALOG_COLUMN_VALUE_GROUNDING_REASON,
    ) -> Evidence | None:
        existing = await self._latest_evidence(
            operation.id,
            "schema.asset_profile",
            accepted=True,
        )
        if existing is not None:
            return existing
        capability = await self._source_prerequisite_capability(
            "db.schema.inspect",
            operation,
            parent_task=parent_task,
            task_input=None,
            prerequisite_for=prerequisite_for,
            prerequisite_reason=prerequisite_reason,
        )
        if capability is None:
            return None
        prerequisite_metadata = _runtime_prerequisite_task_metadata(
            operation,
            parent_task=parent_task,
            capability=capability,
            prerequisite_for=prerequisite_for,
            prerequisite_reason=prerequisite_reason,
        )

        task_input: dict[str, Any] = {}
        input_hash = _stable_hash(task_input)
        idempotency_key = _stable_hash(
            {
                "operation_id": operation.id,
                "capability_id": capability.id,
                "owner": capability.owner,
                "reason": "runtime:schema_profile_prepare",
            }
        )
        task_id = f"db-task-{_stable_hash({'idempotency_key': idempotency_key})[:32]}"
        existing_task = await self.store.load_task(task_id)
        if existing_task is not None:
            schema_task = await self._merge_runtime_prerequisite_metadata(
                existing_task,
                prerequisite_metadata,
            )
        else:
            schema_task = await self._plan_kernel_task(
                Task(
                    id=task_id,
                    operation_id=operation.id,
                    capability_id=capability.id,
                    executor_id=capability.executor,
                    input={**task_input, "input_hash": input_hash},
                    required_evidence=capability.output_evidence,
                    metadata={
                        **prerequisite_metadata,
                        "owner": capability.owner,
                        "reason": "runtime:schema_profile_prepare",
                        "sequence": 0,
                        "input_hash": input_hash,
                        "idempotency_key": idempotency_key,
                        "deterministic_key": "runtime:db.schema.inspect",
                        "idempotent": capability.idempotent,
                        "replay_safe": capability.replay_safe,
                        "side_effecting": capability.side_effecting,
                    },
                )
            )
        if schema_task.status is not TaskStatus.SUCCEEDED:
            await self.execute_task(schema_task, operation)
        return await self._latest_evidence(
            operation.id,
            "schema.asset_profile",
            accepted=True,
        )

    async def _ensure_catalog_source_registered(
        self,
        operation: Operation,
        *,
        store_id: str,
        schema_evidence: Evidence,
        parent_task: Task | None = None,
        prerequisite_for: str | None = None,
        prerequisite_reason: str = _CATALOG_COLUMN_VALUE_GROUNDING_REASON,
    ) -> None:
        registered = await self._latest_evidence(
            operation.id,
            "catalog.source_registered",
            payload={"store_id": store_id},
            accepted=True,
        )
        if registered is not None:
            return
        capability = self.registry.get_capability(
            "catalog.source.register",
            owner="catalog",
        )
        prerequisite_metadata = _runtime_prerequisite_task_metadata(
            operation,
            parent_task=parent_task,
            capability=capability,
            prerequisite_for=prerequisite_for,
            prerequisite_reason=prerequisite_reason,
        )
        schema = dict(schema_evidence.payload)
        task_input = {
            "schema": schema,
            "store_type": schema.get("database_type"),
            "store_id": store_id,
            "persist": False,
        }
        input_hash = _stable_hash(task_input)
        schema_fingerprint = schema_evidence.metadata.get(
            "payload_fingerprint"
        ) or _payload_fingerprint(schema)
        idempotency_key = _stable_hash(
            {
                "operation_id": operation.id,
                "capability_id": capability.id,
                "owner": capability.owner,
                "store_id": store_id,
                "schema_fingerprint": schema_fingerprint,
            }
        )
        task_id = f"db-task-{_stable_hash({'idempotency_key': idempotency_key})[:32]}"
        existing = await self.store.load_task(task_id)
        if existing is not None:
            if existing.status is TaskStatus.SUCCEEDED:
                return
            register_task = await self._merge_runtime_prerequisite_metadata(
                existing,
                prerequisite_metadata,
            )
        else:
            register_task = await self._plan_kernel_task(
                Task(
                    id=task_id,
                    operation_id=operation.id,
                    capability_id=capability.id,
                    executor_id=capability.executor,
                    input={**task_input, "input_hash": input_hash},
                    required_evidence=capability.output_evidence,
                    metadata={
                        **prerequisite_metadata,
                        "owner": capability.owner,
                        "reason": "runtime:catalog_source_prepare",
                        "sequence": 0,
                        "input_hash": input_hash,
                        "idempotency_key": idempotency_key,
                        "deterministic_key": (
                            f"runtime:catalog.source.register:{store_id}"
                        ),
                        "idempotent": capability.idempotent,
                        "replay_safe": capability.replay_safe,
                        "side_effecting": capability.side_effecting,
                        "schema_evidence_id": schema_evidence.id,
                        "schema_fingerprint": schema_fingerprint,
                    },
                )
            )
        await self.execute_task(register_task, operation)

    async def _ensure_catalog_column_values_profiled(
        self,
        operation: Operation,
        *,
        store_id: str,
        task_input: dict[str, Any],
        parent_task: Task,
    ) -> None:
        pairs = _column_value_profile_read_pairs(task_input)
        if not pairs or not _operation_allows_read_profile(operation):
            return
        profile_capability = await self._source_prerequisite_capability(
            "db.column_values.profile",
            operation,
            parent_task=parent_task,
            task_input=task_input,
            prerequisite_for=parent_task.capability_id,
            prerequisite_reason=_CATALOG_COLUMN_VALUE_GROUNDING_REASON,
        )
        if profile_capability is None:
            return
        try:
            register_capability = self.registry.get_capability(
                "catalog.column_values.register",
                owner="catalog",
            )
        except KeyError:
            return
        profile_declaration = await self._require_declared_runtime_prerequisite(
            operation,
            parent_task=parent_task,
            capability=profile_capability,
            prerequisite_for=parent_task.capability_id,
            prerequisite_reason=_CATALOG_COLUMN_VALUE_GROUNDING_REASON,
        )
        register_declaration = await self._require_declared_runtime_prerequisite(
            operation,
            parent_task=parent_task,
            capability=register_capability,
            prerequisite_for=parent_task.capability_id,
            prerequisite_reason=_CATALOG_COLUMN_VALUE_GROUNDING_REASON,
        )
        for table, column in pairs[:4]:
            if await self._catalog_profile_registered(
                operation.id,
                store_id=store_id,
                table=table,
                column=column,
            ):
                continue
            profile = await self._ensure_column_value_profile(
                operation,
                capability=profile_capability,
                table=table,
                column=column,
                parent_task=parent_task,
                prerequisite_for=parent_task.capability_id,
                prerequisite_reason=_CATALOG_COLUMN_VALUE_GROUNDING_REASON,
                prerequisite_declaration=profile_declaration,
            )
            if profile is None or not profile.accepted:
                continue
            await self._ensure_catalog_column_value_profile_registered(
                operation,
                capability=register_capability,
                store_id=store_id,
                profile=profile,
                parent_task=parent_task,
                prerequisite_for=parent_task.capability_id,
                prerequisite_reason=_CATALOG_COLUMN_VALUE_GROUNDING_REASON,
                prerequisite_declaration=register_declaration,
            )

    async def _ensure_column_value_profile(
        self,
        operation: Operation,
        *,
        capability: Capability,
        table: str,
        column: str,
        parent_task: Task | None = None,
        prerequisite_for: str | None = None,
        prerequisite_reason: str = _CATALOG_COLUMN_VALUE_GROUNDING_REASON,
        prerequisite_declaration: dict[str, Any] | None = None,
    ) -> Evidence | None:
        task_input = {"table": table, "column": column, "max_values": 25}
        input_hash = _stable_hash(task_input)
        prerequisite_metadata = _runtime_prerequisite_task_metadata(
            operation,
            parent_task=parent_task,
            capability=capability,
            prerequisite_for=prerequisite_for,
            prerequisite_reason=prerequisite_reason,
            prerequisite_declaration=prerequisite_declaration,
        )
        idempotency_key = _stable_hash(
            {
                "operation_id": operation.id,
                "capability_id": capability.id,
                "owner": capability.owner,
                "table": table,
                "column": column,
            }
        )
        task_id = f"db-task-{_stable_hash({'idempotency_key': idempotency_key})[:32]}"
        existing = await self.store.load_task(task_id)
        if existing is not None:
            task = await self._merge_runtime_prerequisite_metadata(
                existing,
                prerequisite_metadata,
            )
        else:
            task = await self._plan_kernel_task(
                Task(
                    id=task_id,
                    operation_id=operation.id,
                    capability_id=capability.id,
                    executor_id=capability.executor,
                    input={**task_input, "input_hash": input_hash},
                    required_evidence=capability.output_evidence,
                    metadata={
                        **prerequisite_metadata,
                        "owner": capability.owner,
                        "reason": "runtime:column_values_profile_prepare",
                        "sequence": 0,
                        "input_hash": input_hash,
                        "idempotency_key": idempotency_key,
                        "deterministic_key": (
                            f"runtime:db.column_values.profile:{table}.{column}"
                        ),
                        "idempotent": capability.idempotent,
                        "replay_safe": capability.replay_safe,
                        "side_effecting": capability.side_effecting,
                    },
                )
            )
        if task.status is not TaskStatus.SUCCEEDED:
            await self.execute_task(task, operation)
        return await self._latest_evidence(
            operation.id,
            "column_values.profile",
            payload={"table": table, "column": column},
            accepted=True,
        )

    async def _ensure_catalog_column_value_profile_registered(
        self,
        operation: Operation,
        *,
        capability: Capability,
        store_id: str,
        profile: Evidence,
        parent_task: Task | None = None,
        prerequisite_for: str | None = None,
        prerequisite_reason: str = _CATALOG_COLUMN_VALUE_GROUNDING_REASON,
        prerequisite_declaration: dict[str, Any] | None = None,
    ) -> None:
        table = str(profile.payload.get("table") or "")
        column = str(profile.payload.get("column") or "")
        if not table or not column:
            return
        if await self._catalog_profile_registered(
            operation.id,
            store_id=store_id,
            table=table,
            column=column,
        ):
            return
        task_input = {
            "store_id": store_id,
            "profiles": [dict(profile.payload)],
            "source_evidence_id": profile.id,
            "persist": False,
        }
        input_hash = _stable_hash(task_input)
        prerequisite_metadata = _runtime_prerequisite_task_metadata(
            operation,
            parent_task=parent_task,
            capability=capability,
            prerequisite_for=prerequisite_for,
            prerequisite_reason=prerequisite_reason,
            prerequisite_declaration=prerequisite_declaration,
        )
        idempotency_key = _stable_hash(
            {
                "operation_id": operation.id,
                "capability_id": capability.id,
                "owner": capability.owner,
                "store_id": store_id,
                "table": table,
                "column": column,
                "profile_evidence_id": profile.id,
            }
        )
        task_id = f"db-task-{_stable_hash({'idempotency_key': idempotency_key})[:32]}"
        existing = await self.store.load_task(task_id)
        if existing is not None:
            task = await self._merge_runtime_prerequisite_metadata(
                existing,
                prerequisite_metadata,
            )
        else:
            task = await self._plan_kernel_task(
                Task(
                    id=task_id,
                    operation_id=operation.id,
                    capability_id=capability.id,
                    executor_id=capability.executor,
                    input={**task_input, "input_hash": input_hash},
                    required_evidence=capability.output_evidence,
                    metadata={
                        **prerequisite_metadata,
                        "owner": capability.owner,
                        "reason": "runtime:catalog_column_values_register",
                        "sequence": 0,
                        "input_hash": input_hash,
                        "idempotency_key": idempotency_key,
                        "deterministic_key": (
                            "runtime:catalog.column_values.register:"
                            f"{store_id}:{table}.{column}"
                        ),
                        "idempotent": capability.idempotent,
                        "replay_safe": capability.replay_safe,
                        "side_effecting": capability.side_effecting,
                        "profile_evidence_id": profile.id,
                    },
                )
            )
        if task.status is not TaskStatus.SUCCEEDED:
            await self.execute_task(task, operation)

    async def _merge_runtime_prerequisite_metadata(
        self,
        task: Task,
        metadata: dict[str, Any],
    ) -> Task:
        if not metadata:
            return task
        merged = {**task.metadata, **metadata}
        if merged == task.metadata:
            return task
        updated = replace(task, metadata=merged)
        await self.store.save_task(updated)
        return updated

    async def _source_prerequisite_capability(
        self,
        capability_id: str,
        operation: Operation,
        *,
        parent_task: Task | None,
        task_input: dict[str, Any] | None,
        prerequisite_for: str | None,
        prerequisite_reason: str,
    ) -> Capability | None:
        owners = _capability_owners(self.registry.capabilities, capability_id)
        owner = _source_owner_for_prerequisite(
            operation,
            metadata=self.config.metadata,
            owners=owners,
            parent_task=parent_task,
            task_input=task_input,
        )
        if owner is None:
            if len(owners) > 1 and parent_task is not None:
                await self._block_ambiguous_source_owner(
                    parent_task,
                    capability_id=capability_id,
                    owners=owners,
                    prerequisite_for=prerequisite_for,
                    prerequisite_reason=prerequisite_reason,
                )
            return None
        try:
            return self.registry.get_capability(capability_id, owner=owner)
        except KeyError:
            return None

    async def _block_ambiguous_source_owner(
        self,
        parent_task: Task,
        *,
        capability_id: str,
        owners: tuple[str, ...],
        prerequisite_for: str | None,
        prerequisite_reason: str,
    ) -> None:
        error = f"ambiguous_source_owner:{capability_id}"
        details = {
            "error": "ambiguous_source_owner",
            "capability_id": capability_id,
            "candidate_owners": list(owners),
            "prerequisite_for": prerequisite_for or parent_task.capability_id,
            "prerequisite_for_task_id": parent_task.id,
            "prerequisite_reason": prerequisite_reason,
        }
        await self.kernel.block_task(
            parent_task.id,
            message=error,
            payload={"runtime_prerequisite": details},
            metadata={"runtime_prerequisite_blocked": details},
        )
        raise RuntimeError(error)

    async def _require_declared_runtime_prerequisite(
        self,
        operation: Operation,
        *,
        parent_task: Task,
        capability: Capability,
        prerequisite_for: str,
        prerequisite_reason: str,
    ) -> dict[str, Any]:
        declaration = _runtime_prerequisite_declaration(
            operation,
            parent_task=parent_task,
            capability=capability,
            prerequisite_for=prerequisite_for,
        )
        if declaration is not None:
            return declaration
        error = f"undeclared_runtime_prerequisite:{capability.id}"
        details = {
            "error": error,
            "capability_id": capability.id,
            "owner": capability.owner,
            "prerequisite_for": prerequisite_for,
            "prerequisite_for_task_id": parent_task.id,
            "prerequisite_reason": prerequisite_reason,
        }
        await self.kernel.block_task(
            parent_task.id,
            message=error,
            payload={"runtime_prerequisite": details},
            metadata={"runtime_prerequisite_blocked": details},
        )
        raise RuntimeError(error)

    async def _catalog_profile_registered(
        self,
        operation_id: str,
        *,
        store_id: str,
        table: str,
        column: str,
    ) -> bool:
        for evidence in await self.store.list_evidence(operation_id):
            if evidence.kind != "schema.column_value_profile" or not evidence.accepted:
                continue
            if evidence.payload.get("store_id") != store_id:
                continue
            for profile in evidence.payload.get("profiles", []) or []:
                if not isinstance(profile, dict):
                    continue
                if profile.get("table") == table and profile.get("column") == column:
                    return True
        return False

    async def _accepted_evidence_for_dependency(
        self,
        operation_id: str,
        dependency: TaskDependency,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == dependency.evidence_kind
            and evidence.accepted is dependency.evidence_accepted
            and (
                dependency.evidence_id is None or evidence.id == dependency.evidence_id
            )
            and (
                dependency.evidence_owner is None
                or evidence.owner == dependency.evidence_owner
            )
            and (
                dependency.producer_task_id is None
                or evidence.task_id == dependency.producer_task_id
            )
            and (
                dependency.input_hash is None
                or evidence.metadata.get("task_input_hash") == dependency.input_hash
            )
            and _payload_contains(evidence.payload, dependency.evidence_payload)
            and (
                dependency.payload_fingerprint is None
                or dependency.payload_fingerprint
                == _evidence_payload_fingerprint(evidence)
            )
        ]
        return matches[-1] if matches else None

    async def _authoritative_validation_evidence(
        self,
        operation: Operation,
        task: Task | None,
    ) -> tuple[Evidence, ...]:
        if task is None or task.capability_id not in {
            "db.sql.execute_read",
            "db.sql.execute_write",
        }:
            return ()
        if task.metadata.get("monitor_action_role") == "write_execution":
            expected_validation_id = str(
                task.metadata.get("validation_evidence_id") or ""
            )
            if expected_validation_id:
                matches = [
                    evidence
                    for evidence in await self.store.list_evidence(operation.id)
                    if evidence.kind == "sql.validation"
                    and evidence.id == expected_validation_id
                    and evidence.accepted
                ]
                if matches:
                    return (matches[-1],)
        validation_dependency = next(
            (
                dependency
                for dependency in task.dependencies
                if dependency.kind.value == "evidence"
                and dependency.evidence_kind == "sql.validation"
            ),
            None,
        )
        if validation_dependency is None:
            return ()
        evidence = await self._accepted_evidence_for_dependency(
            operation.id,
            validation_dependency,
        )
        return (evidence,) if evidence is not None else ()

    async def _latest_accepted_evidence(
        self,
        operation_id: str,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == kind
            and evidence.accepted
            and _payload_contains(evidence.payload, payload or {})
        ]
        return matches[-1] if matches else None

    async def _latest_evidence(
        self,
        operation_id: str,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
        accepted: bool | None = None,
    ) -> Evidence | None:
        matches = [
            evidence
            for evidence in await self.store.list_evidence(operation_id)
            if evidence.kind == kind
            and (accepted is None or evidence.accepted is accepted)
            and _payload_contains(evidence.payload, payload or {})
        ]
        return matches[-1] if matches else None

    async def executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        """Hydrate DB task input from authoritative validation evidence."""
        return await self._executable_input_for_task(task, operation)


def _catalog_task_input_from_metadata(
    task: Task, operation: Operation
) -> dict[str, Any]:
    task_input = dict(task.input)
    metadata = dict(task.metadata or {})
    prompt = str(operation.request.get("prompt") or "")

    if task.capability_id == "catalog.schema.search":
        task_input.setdefault(
            "query",
            _first_metadata_string(metadata, "query", "target", "goal") or prompt,
        )
    elif task.capability_id == "catalog.asset.inspect":
        asset_ref = _first_metadata_string(
            metadata,
            "asset_ref",
            "asset",
            "table",
            "target",
        )
        if asset_ref:
            task_input.setdefault("asset_ref", asset_ref)
    elif task.capability_id == "catalog.relationship_paths.find":
        from_assets = _metadata_string_list(
            metadata,
            "from_assets",
            "from",
            "source_assets",
            "source",
        )
        to_assets = _metadata_string_list(
            metadata,
            "to_assets",
            "to",
            "target_assets",
            "target",
        )
        if from_assets:
            task_input.setdefault("from_assets", from_assets)
        if to_assets:
            task_input.setdefault("to_assets", to_assets)
    elif task.capability_id in {
        "catalog.column_values.search",
        "catalog.column_value_hints.resolve",
    }:
        task_input.setdefault(
            "query",
            _first_metadata_string(metadata, "query", "value", "literal", "goal")
            or prompt,
        )
        task_input.setdefault("prompt", task_input.get("query") or prompt)
        input_tables = _safe_string_list(task_input.get("tables"))
        if not input_tables:
            input_tables = _safe_string_list(task_input.get("table"))
        input_columns = _safe_string_list(task_input.get("columns"))
        if not input_columns:
            input_columns = _safe_string_list(task_input.get("column"))
        target_tables, target_columns = _normalize_column_value_scope(
            _safe_string_list(task_input.get("target")),
            [],
        )
        if not input_tables:
            input_tables = target_tables
        if not input_columns:
            input_columns = target_columns

        tables, columns = _normalize_column_value_scope(input_tables, input_columns)

        metadata_tables = _metadata_string_list(metadata, "tables", "table")
        metadata_columns = _metadata_string_list(metadata, "columns", "column", "field")
        metadata_target_tables, metadata_target_columns = _normalize_column_value_scope(
            _metadata_string_list(metadata, "target"),
            [],
        )
        if not metadata_tables:
            metadata_tables = metadata_target_tables
        if not metadata_columns:
            metadata_columns = metadata_target_columns
        metadata_tables, metadata_columns = _normalize_column_value_scope(
            metadata_tables,
            metadata_columns,
        )
        if not tables:
            tables = metadata_tables
        if not columns:
            columns = metadata_columns
        tables, columns = _normalize_column_value_scope(tables, columns)
        if tables:
            task_input["tables"] = tables
        if columns:
            task_input["columns"] = columns
        if tables and columns:
            task_input.pop("target", None)
    elif task.capability_id == "catalog.value_grounding.plan":
        task_input.setdefault(
            "query",
            _first_metadata_string(metadata, "query", "value", "literal", "goal")
            or prompt,
        )
        task_input.setdefault("prompt", task_input.get("query") or prompt)
        input_tables = _safe_string_list(task_input.get("tables"))
        if not input_tables:
            input_tables = _safe_string_list(task_input.get("table"))
        input_columns = _safe_string_list(task_input.get("columns"))
        if not input_columns:
            input_columns = _safe_string_list(task_input.get("column"))
        target_tables, target_columns = _normalize_column_value_scope(
            _safe_string_list(task_input.get("target")),
            [],
        )
        if not input_tables:
            input_tables = target_tables
        if not input_columns:
            input_columns = target_columns
        tables, columns = _normalize_column_value_scope(input_tables, input_columns)
        if tables:
            task_input["tables"] = tables
        if columns:
            task_input["columns"] = columns
    return task_input


def _value_grounding_plan_input_with_operation_context(
    task_input: dict[str, Any],
    *,
    operation: Operation,
) -> dict[str, Any]:
    enriched = dict(task_input)
    prompt = str(operation.request.get("prompt") or "")
    enriched.setdefault("prompt", prompt)
    enriched.setdefault("query", enriched.get("prompt") or prompt)
    if enriched.get("max_profile_budget") is not None and not enriched.get(
        "profile_budget"
    ):
        enriched["profile_budget"] = enriched["max_profile_budget"]
    session_scopes = _session_query_scopes_for_value_grounding(
        operation.request.get("session_context")
    )
    if session_scopes and not enriched.get("session_query_scopes"):
        enriched["session_query_scopes"] = session_scopes
    validation_facts, validation_warnings = _operation_validation_inputs_for_grounding(
        operation
    )
    if validation_facts and not enriched.get("validation_facts"):
        enriched["validation_facts"] = validation_facts
    if validation_warnings and not (
        enriched.get("warnings") or enriched.get("validation_warnings")
    ):
        enriched["validation_warnings"] = validation_warnings
    safety_frame = operation.metadata.get("safety_frame")
    if isinstance(safety_frame, Mapping) and not enriched.get("policy_frame"):
        enriched["policy_frame"] = dict(safety_frame)
    if not enriched.get("profile_pairs"):
        pairs = _column_value_profile_pairs(enriched)
        if pairs:
            enriched["profile_pairs"] = [
                {"table": table, "column": column} for table, column in pairs
            ]
    return enriched


def _session_query_scopes_for_value_grounding(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    scopes: list[dict[str, Any]] = []
    for item in value.get("query_scopes") or ():
        if isinstance(item, Mapping):
            scopes.append(dict(item))
    return _dedupe_json_values(scopes)


def _operation_validation_inputs_for_grounding(
    operation: Operation,
) -> tuple[list[Any], list[Any]]:
    facts: list[Any] = []
    warnings: list[Any] = []
    for item in operation.metadata.get("validation_facts") or ():
        if isinstance(item, (Mapping, str)):
            facts.append(dict(item) if isinstance(item, Mapping) else str(item))
    for item in operation.metadata.get("validation_warnings") or ():
        if isinstance(item, (Mapping, str)):
            warnings.append(dict(item) if isinstance(item, Mapping) else str(item))
    return _dedupe_json_values(facts), _dedupe_json_values(warnings)


def _validation_items_for_value_grounding(value: Any) -> list[Any]:
    items: list[Any] = []
    for item in _value_grounding_iterable(value):
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
    return items


def _value_grounding_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _dedupe_json_values(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _normalize_column_value_scope(
    tables: list[str],
    columns: list[str],
) -> tuple[list[str], list[str]]:
    normalized_tables: list[str] = []
    normalized_columns: list[str] = list(columns)
    for table in tables:
        if "." in table:
            table_name, column_name = table.split(".", 1)
            if table_name.strip():
                normalized_tables.append(table_name.strip())
            if column_name.strip() and not normalized_columns:
                normalized_columns.append(column_name.strip())
            continue
        normalized_tables.append(table)
    return _ordered_unique_strings(normalized_tables), _ordered_unique_strings(
        normalized_columns
    )


def _column_value_task_input_with_grounding_plan(
    task_input: dict[str, Any],
    *,
    grounding_plan: Evidence | None,
) -> dict[str, Any]:
    if grounding_plan is None or not isinstance(grounding_plan.payload, Mapping):
        return task_input
    targets = [
        target
        for target in grounding_plan.payload.get("targets", []) or []
        if isinstance(target, Mapping)
    ]
    if not targets:
        return {
            **task_input,
            "value_grounding_plan_evidence_id": grounding_plan.id,
            "value_grounding_profile_pairs": [],
        }
    all_pairs = _ordered_target_pairs(targets)
    profile_pairs = _ordered_target_pairs(
        target for target in targets if target.get("requires_profile_read") is True
    )
    if not all_pairs:
        return {
            **task_input,
            "value_grounding_plan_evidence_id": grounding_plan.id,
            "value_grounding_profile_pairs": [],
        }
    tables = _ordered_unique_strings([table for table, _column in all_pairs])
    columns = _ordered_unique_strings([column for _table, column in all_pairs])
    query = _query_with_grounding_terms(task_input, targets)
    return {
        **task_input,
        "query": query,
        "prompt": query,
        "profile_pairs": [
            {"table": table, "column": column} for table, column in all_pairs
        ],
        "value_grounding_profile_pairs": [
            {"table": table, "column": column} for table, column in profile_pairs
        ],
        "value_grounding_plan_evidence_id": grounding_plan.id,
        "tables": tables,
        "columns": columns,
    }


def _ordered_target_pairs(
    targets: Iterable[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for target in targets:
        table = str(target.get("table") or "").strip()
        column = str(target.get("column") or "").strip()
        if table and column:
            pairs.append((table, column))
    return list(dict.fromkeys(pairs))


def _query_with_grounding_terms(
    task_input: Mapping[str, Any],
    targets: list[Mapping[str, Any]],
) -> str:
    values = [str(task_input.get("query") or task_input.get("prompt") or "").strip()]
    for target in targets:
        values.append(str(target.get("table") or ""))
        values.append(str(target.get("column") or ""))
        source = target.get("source")
        if isinstance(source, Mapping):
            literal = source.get("literal")
            if literal is not None:
                values.append(str(literal))
            for value in source.get("values") or ():
                values.append(str(value))
    return " ".join(value for value in values if value.strip())


def _column_value_profile_pairs(task_input: dict[str, Any]) -> list[tuple[str, str]]:
    explicit_pairs = _safe_column_value_profile_pairs(task_input.get("profile_pairs"))
    if explicit_pairs:
        return explicit_pairs
    tables = _safe_string_list(task_input.get("tables"))
    columns = _safe_string_list(task_input.get("columns"))
    tables, columns = _normalize_column_value_scope(tables, columns)
    if not tables or not columns:
        return []
    return [(table, column) for table in tables for column in columns]


def _column_value_profile_read_pairs(
    task_input: dict[str, Any],
) -> list[tuple[str, str]]:
    if "value_grounding_profile_pairs" in task_input:
        return _safe_column_value_profile_pairs(
            task_input.get("value_grounding_profile_pairs")
        )
    return _column_value_profile_pairs(task_input)


def _safe_column_value_profile_pairs(value: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not isinstance(value, (list, tuple, set, frozenset)):
        return pairs
    for item in value:
        table = ""
        column = ""
        if isinstance(item, Mapping):
            table = str(item.get("table") or "").strip()
            column = str(item.get("column") or "").strip()
        elif isinstance(item, str) and "." in item:
            table, column = (part.strip() for part in item.split(".", 1))
        if table and column:
            pairs.append((table, column))
    return list(dict.fromkeys(pairs))


def _operation_allows_read_profile(operation: Operation) -> bool:
    safety_frame = operation.metadata.get("safety_frame")
    if not isinstance(safety_frame, dict):
        return True
    max_access = str(
        safety_frame.get("max_access")
        or safety_frame.get("max_allowed_access")
        or AccessMode.ADMIN.value
    )
    access_order = {
        AccessMode.NONE.value: 0,
        AccessMode.METADATA_READ.value: 1,
        AccessMode.READ.value: 2,
        AccessMode.WRITE.value: 3,
        AccessMode.ADMIN.value: 4,
    }
    return access_order.get(max_access, 4) >= access_order[AccessMode.READ.value]


def _runtime_prerequisite_task_metadata(
    operation: Operation,
    *,
    parent_task: Task | None,
    capability: Capability,
    prerequisite_for: str | None,
    prerequisite_reason: str,
    prerequisite_declaration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prerequisite_for = prerequisite_for or (
        parent_task.capability_id if parent_task is not None else ""
    )
    declaration = prerequisite_declaration
    if declaration is None:
        declaration = _runtime_prerequisite_declaration(
            operation,
            parent_task=parent_task,
            capability=capability,
            prerequisite_for=prerequisite_for,
        )
    metadata: dict[str, Any] = {
        "runtime_prerequisite": True,
        "declared_by_contract": declaration is not None,
        "prerequisite_for": prerequisite_for,
        "prerequisite_reason": prerequisite_reason,
    }
    if parent_task is not None:
        metadata["prerequisite_for_task_id"] = parent_task.id
        if parent_task.metadata.get("planner_action_id"):
            metadata["prerequisite_for_action_id"] = parent_task.metadata[
                "planner_action_id"
            ]
        if parent_task.metadata.get("planner_action_kind"):
            metadata["prerequisite_for_action_kind"] = parent_task.metadata[
                "planner_action_kind"
            ]
    if declaration is not None:
        metadata["contract_prerequisite"] = declaration
    return metadata


def _runtime_prerequisite_declaration(
    operation: Operation,
    *,
    parent_task: Task | None,
    capability: Capability,
    prerequisite_for: str | None,
) -> dict[str, Any] | None:
    contract = _latest_compiled_contract_snapshot(operation, parent_task=parent_task)
    if contract is None:
        return None
    required = {str(item) for item in contract.get("required_capabilities") or ()}
    if capability.id not in required:
        return None
    metadata = contract.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    for raw in metadata.get("runtime_prerequisites") or ():
        if not isinstance(raw, dict):
            continue
        if raw.get("capability_id") != capability.id:
            continue
        if raw.get("owner") not in {None, capability.owner}:
            continue
        expected_for = prerequisite_for or (
            parent_task.capability_id if parent_task is not None else None
        )
        if expected_for and raw.get("for_capability_id") != expected_for:
            continue
        return dict(raw)
    return None


def _latest_compiled_contract_snapshot(
    operation: Operation,
    *,
    parent_task: Task | None,
) -> dict[str, Any] | None:
    context = operation.metadata.get("resume_context")
    context = context if isinstance(context, dict) else {}
    parent_metadata = parent_task.metadata if parent_task is not None else {}
    for candidate in (
        operation.metadata.get("latest_compiled_contract_snapshot"),
        context.get("latest_compiled_contract_snapshot"),
        context.get("contract"),
        parent_metadata.get("contract"),
    ):
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return None


def _safe_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _ordered_unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _catalog_store_id(task_input: dict[str, Any], metadata: dict[str, Any]) -> str:
    explicit = task_input.get("store_id")
    if explicit:
        return str(explicit)
    options = metadata.get("from_db_options")
    if isinstance(options, dict) and options.get("catalog_store_id"):
        return str(options["catalog_store_id"])
    if metadata.get("catalog_store_id"):
        return str(metadata["catalog_store_id"])
    return ""


def _source_owner_for_prerequisite(
    operation: Operation,
    *,
    metadata: dict[str, Any],
    owners: tuple[str, ...],
    parent_task: Task | None,
    task_input: dict[str, Any] | None,
) -> str | None:
    for candidate in (
        parent_task.metadata if parent_task is not None else None,
        parent_task.input if parent_task is not None else None,
        task_input,
    ):
        owner = _source_owner_from_mapping(candidate, owners=owners)
        if owner:
            return owner

    owner = _source_owner_from_source_scope(
        operation.request.get("source_scope"),
        owners=owners,
    )
    if owner:
        return owner

    options = metadata.get("from_db_options")
    owner = _source_owner_from_mapping(
        options if isinstance(options, dict) else None,
        owners=owners,
        option_keys=True,
    )
    if owner:
        return owner

    if len(owners) == 1:
        return owners[0]
    return None


def _capability_owners(
    capabilities: Iterable[Capability],
    capability_id: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            capability.owner
            for capability in capabilities
            if capability.id == capability_id
        )
    )


def _source_owner_from_mapping(
    value: Any,
    *,
    owners: tuple[str, ...],
    option_keys: bool = False,
) -> str | None:
    if not isinstance(value, Mapping):
        return None
    keys = _SOURCE_OWNER_OPTION_KEYS if option_keys else _SOURCE_OWNER_KEYS
    for key in keys:
        owner = _normalized_source_owner(value.get(key))
        if owner:
            return owner
    owner = _normalized_source_owner(value.get("owner"))
    if owner and owner != "catalog" and (not owners or owner in owners):
        return owner
    return None


def _source_owner_from_source_scope(
    value: Any,
    *,
    owners: tuple[str, ...],
) -> str | None:
    if not owners:
        return None
    items = _source_scope_items(value)
    if len(items) != 1:
        return None
    item = items[0]
    if isinstance(item, Mapping):
        owner = _source_owner_from_mapping(item, owners=owners)
        return owner if owner in owners else None
    owner = _normalized_source_owner(item)
    return owner if owner in owners else None


def _source_scope_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


def _normalized_source_owner(value: Any) -> str | None:
    if value is None:
        return None
    owner = str(value).strip()
    return owner or None


def _first_metadata_string(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _metadata_string_list(metadata: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, (list, tuple)):
            values = [str(item).strip() for item in value if str(item).strip()]
            if values:
                return values
    return []


def _synthesis_dependencies(
    operation: Operation,
    intent: DbIntent,
    evidence: tuple[Evidence, ...],
) -> tuple[TaskDependency, ...]:
    accepted = tuple(
        item
        for item in evidence
        if item.accepted and item.operation_id == operation.id and item.id
    )
    dependencies: list[TaskDependency] = []
    if intent.kind in {
        DbIntentKind.DATA_QUERY,
        DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
    }:
        _append_dependency_for_kind(dependencies, accepted, "planning.context")
        if not any(item.evidence_kind == "planning.context" for item in dependencies):
            _append_dependency_for_any(
                dependencies,
                accepted,
                ("schema.asset_profile", "catalog.source", "schema.search_result"),
            )
        for kind in (
            "query.result",
            "query.plan.proposal",
            "query.plan.validation",
            "sql.validation",
            "verification.result",
        ):
            _append_dependency_for_kind(dependencies, accepted, kind)
    elif intent.kind is DbIntentKind.SCHEMA_QUERY:
        _append_database_schema_dependency(dependencies, accepted)
        for kind in ("planning.context", "schema.search_result"):
            _append_dependency_for_kind(dependencies, accepted, kind)
        _append_schema_asset_dependencies(dependencies, accepted)
        _append_dependency_for_kind(dependencies, accepted, "verification.result")
    elif intent.kind is DbIntentKind.SCHEMA_RELATIONSHIP_QUERY:
        _append_database_schema_dependency(dependencies, accepted)
        for kind in (
            "planning.context",
            "schema.relationship_path",
            "schema.search_result",
        ):
            _append_dependency_for_kind(dependencies, accepted, kind)
        _append_dependency_for_kind(dependencies, accepted, "verification.result")
    else:
        for item in accepted:
            if item.kind in {
                "planner.decision",
                "planner.compilation",
                "planner.observation",
                "verification.result",
                "answer.synthesis",
            }:
                continue
            dependencies.append(_dependency_for_evidence(item))
        _append_dependency_for_kind(dependencies, accepted, "verification.result")
    seen: set[tuple[str | None, str | None]] = set()
    unique: list[TaskDependency] = []
    for dependency in dependencies:
        key = (dependency.evidence_kind, dependency.evidence_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(dependency)
    return tuple(unique)


def _append_dependency_for_kind(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
    kind: str,
) -> None:
    item = next(
        (candidate for candidate in reversed(evidence) if candidate.kind == kind),
        None,
    )
    if item is not None:
        dependencies.append(_dependency_for_evidence(item))


def _append_dependency_for_any(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
    kinds: tuple[str, ...],
) -> None:
    for kind in kinds:
        item = next(
            (candidate for candidate in reversed(evidence) if candidate.kind == kind),
            None,
        )
        if item is not None:
            dependencies.append(_dependency_for_evidence(item))
            return
    catalog = next(
        (
            candidate
            for candidate in reversed(evidence)
            if candidate.kind.startswith("catalog.")
        ),
        None,
    )
    if catalog is not None:
        dependencies.append(_dependency_for_evidence(catalog))


def _append_database_schema_dependency(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
) -> None:
    item = next(
        (
            candidate
            for candidate in reversed(evidence)
            if candidate.kind == "schema.asset_profile"
            and _schema_evidence_scope(candidate) == "database"
        ),
        None,
    )
    if item is None:
        item = next(
            (
                candidate
                for candidate in evidence
                if candidate.kind == "schema.asset_profile"
                and candidate.payload.get("tables")
            ),
            None,
        )
    if item is not None:
        dependencies.append(_dependency_for_evidence(item))


def _append_schema_asset_dependencies(
    dependencies: list[TaskDependency],
    evidence: tuple[Evidence, ...],
) -> None:
    scoped = [
        item
        for item in evidence
        if item.kind == "schema.asset_profile"
        and _schema_evidence_scope(item) == "asset"
        and item.id
    ]
    for item in scoped:
        dependencies.append(_dependency_for_evidence(item))
    if not scoped:
        _append_dependency_for_kind(dependencies, evidence, "schema.asset_profile")


def _schema_evidence_scope(evidence: Evidence) -> str | None:
    metadata_scope = evidence.metadata.get("scope")
    if metadata_scope:
        return str(metadata_scope)
    payload_metadata = evidence.payload.get("metadata")
    if isinstance(payload_metadata, dict) and payload_metadata.get("scope"):
        return str(payload_metadata["scope"])
    return None


def _dependency_for_evidence(evidence: Evidence) -> TaskDependency:
    return TaskDependency(
        kind="evidence",
        evidence_kind=evidence.kind,
        evidence_id=evidence.id,
        evidence_owner=evidence.owner,
        producer_task_id=evidence.task_id,
        evidence_accepted=True,
        operation_id=evidence.operation_id,
        payload_fingerprint=evidence.metadata.get("payload_fingerprint")
        or _payload_fingerprint(evidence.payload),
    )


def _synthesis_context_option(
    metadata: dict[str, Any],
    key: str,
    default: int,
) -> int:
    options = metadata.get("from_db_options")
    if isinstance(options, dict) and options.get(key) is not None:
        try:
            return int(options[key])
        except (TypeError, ValueError):
            return default
    return default


def _combine_dependencies(
    default_dependencies: tuple[TaskDependency, ...],
    spec_dependencies: tuple[TaskDependency, ...],
) -> tuple[TaskDependency, ...]:
    combined: list[TaskDependency] = []
    seen: set[str] = set()
    for dependency in (*default_dependencies, *spec_dependencies):
        fingerprint = _stable_hash(dependency.to_dict())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        combined.append(dependency)
    return tuple(combined)


def _has_sql_validation_dependency(
    dependencies: tuple[TaskDependency, ...],
) -> bool:
    return any(
        dependency.kind.value == "evidence"
        and dependency.evidence_kind == "sql.validation"
        for dependency in dependencies
    )


def _task_dependencies_for_capability(
    operation: Operation,
    capability: Capability,
    *,
    validation_task: Task | None = None,
) -> tuple[TaskDependency, ...]:
    if capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}:
        return ()
    if capability.id == "db.sql.execute_read" and validation_task is None:
        return ()
    validation_dependency = TaskDependency(
        kind="evidence",
        evidence_kind="sql.validation",
        evidence_owner=(
            validation_task.metadata.get("owner") if validation_task else None
        ),
        producer_task_id=validation_task.id if validation_task else None,
        producer_capability_id=(
            validation_task.capability_id if validation_task else "db.sql.validate"
        ),
        producer_executor_id=(validation_task.executor_id if validation_task else None),
        evidence_payload={"valid": True},
        operation_id=operation.id,
        input_hash=(
            validation_task.metadata.get("input_hash") if validation_task else None
        ),
    )
    if capability.id == "db.sql.execute_read":
        return (validation_dependency,)
    return (
        validation_dependency,
        TaskDependency(
            kind="approval",
            approval_status=ApprovalStatus.APPROVED,
            approval_policy_id="approval_required_for_writes",
            approval_name="human",
            operation_id=operation.id,
        ),
    )
