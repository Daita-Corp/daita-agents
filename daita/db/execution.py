"""
Linear operation execution for the first DB runtime vertical slice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from daita.runtime import (
    Capability,
    Evidence,
    GovernanceResult,
    Operation,
    RuntimeEventType,
    RuntimeKernelGovernanceBlocked,
    RuntimeKernelTaskNotRunnable,
    Task,
    WorkerRuntime,
    WorkerRuntimeOptions,
)

from .evidence import DbEvidenceStore, InMemoryDbEvidenceStore
from .models import DbIntent, DbIntentKind, DbOperationContract, DbRequest
from .query_planning import DbQueryPlanner


@dataclass(frozen=True)
class DbExecutionOutcome:
    """Result of executing one DB operation plan."""

    evidence: tuple[Evidence, ...]
    tasks: tuple[Task, ...]
    diagnostics: dict[str, Any]
    warnings: tuple[str, ...] = ()


class DbOperationExecutor:
    """Execute simple DB contracts through registry capabilities.

    The executor owns linear task dispatch. Query construction, verification,
    and final answer synthesis are owned by separate runtime components.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        query_planner: DbQueryPlanner | None = None,
    ) -> None:
        self.runtime = runtime
        self.query_planner = query_planner or DbQueryPlanner()

    async def execute(
        self,
        request: DbRequest,
        intent: DbIntent,
        contract: DbOperationContract,
        operation: Operation,
    ) -> DbExecutionOutcome:
        evidence_store = DbEvidenceStore()
        tasks: list[Task] = []
        warnings: list[str] = []
        diagnostics: dict[str, Any] = {
            "planned_sql": None,
            "query_plan": None,
            "store_id": _store_id_for_request(request, self.runtime),
        }

        schema_evidence = await self._inspect_schema_if_available(
            operation, tasks, evidence_store
        )
        schema = schema_evidence.payload if schema_evidence is not None else {}

        if self._needs_catalog(contract):
            await self._register_catalog_source_if_available(
                operation,
                request,
                schema,
                tasks,
                evidence_store,
                diagnostics["store_id"],
            )

        if intent.kind is DbIntentKind.SCHEMA_QUERY:
            await self._execute_schema_steps(
                request,
                contract,
                operation,
                schema,
                tasks,
                evidence_store,
                diagnostics["store_id"],
            )
        elif intent.kind in {
            DbIntentKind.DATA_QUERY,
            DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
        }:
            relationship_payload = await self._relationship_payload_if_needed(
                request,
                intent,
                contract,
                operation,
                schema,
                tasks,
                evidence_store,
                diagnostics["store_id"],
            )
            plan = self.query_planner.plan_read_query(
                request,
                intent,
                operation,
                schema,
                relationship_payload=relationship_payload,
            )
            evidence_store.add(plan.evidence)
            warnings.extend(plan.warnings)
            diagnostics["planned_sql"] = plan.sql
            diagnostics["query_plan"] = plan.diagnostics

            if plan.sql:
                await self._execute_capability(
                    "db.sql.validate",
                    contract,
                    operation,
                    tasks,
                    evidence_store,
                    {"sql": plan.sql, "operation": "query"},
                )
                await self._execute_capability(
                    "db.sql.execute_read",
                    contract,
                    operation,
                    tasks,
                    evidence_store,
                    {"sql": plan.sql},
                )
        elif intent.kind is DbIntentKind.QUALITY_CHECK:
            await self._execute_quality_steps(
                request,
                contract,
                operation,
                schema,
                tasks,
                evidence_store,
            )
        elif intent.kind is DbIntentKind.LINEAGE_TRACE:
            await self._execute_lineage_steps(
                request,
                contract,
                operation,
                schema,
                tasks,
                evidence_store,
            )
        elif intent.kind is DbIntentKind.MEMORY_UPDATE:
            await self._execute_memory_update(
                request,
                contract,
                operation,
                tasks,
                evidence_store,
            )
        else:
            warnings.append(f"db_runtime_intent_not_executable:{intent.kind.value}")

        evidence = evidence_store.list()
        diagnostics["evidence_kinds"] = [item.kind for item in evidence]
        diagnostics["evidence_refs"] = list(evidence_store.refs())
        return DbExecutionOutcome(
            evidence=evidence,
            tasks=tuple(tasks),
            diagnostics=diagnostics,
            warnings=tuple(warnings),
        )

    async def _execute_schema_steps(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
    ) -> None:
        table = self.query_planner.best_table_for_prompt(request.prompt, schema)
        await self._execute_capability(
            "catalog.schema.search",
            contract,
            operation,
            tasks,
            evidence_store,
            {"store_id": store_id, "query": request.prompt, "limit": 10},
        )
        if table:
            await self._execute_capability(
                "catalog.asset.inspect",
                contract,
                operation,
                tasks,
                evidence_store,
                {"store_id": store_id, "asset_ref": table, "limit": 100},
            )
        await self._delegate_worker(
            "schema_specialist",
            request,
            operation,
            tasks,
            evidence_store,
            {"prompt": request.prompt, "schema": schema, "store_id": store_id},
        )

    async def _execute_quality_steps(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
    ) -> None:
        table = self.query_planner.best_table_for_prompt(request.prompt, schema)
        if not table:
            return
        await self._execute_capability(
            "quality.profile",
            contract,
            operation,
            tasks,
            evidence_store,
            {
                "table": table,
                "sample_size": request.constraints.get("sample_size")
                or request.metadata.get("sample_size"),
            },
        )

    async def _execute_lineage_steps(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
    ) -> None:
        entity_id = _lineage_entity_for_request(
            request,
            self.query_planner.best_table_for_prompt(request.prompt, schema),
        )
        await self._execute_capability(
            "lineage.trace",
            contract,
            operation,
            tasks,
            evidence_store,
            {
                "entity_id": entity_id,
                "direction": request.constraints.get("direction")
                or request.metadata.get("direction")
                or "both",
                "max_depth": request.constraints.get("max_depth")
                or request.metadata.get("max_depth")
                or 5,
            },
        )

    async def _execute_memory_update(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
    ) -> None:
        selected = _selected_capability(contract, "memory.semantic.write")
        if selected is None:
            return
        capability = self.runtime.registry.get_capability(
            selected["id"], owner=selected["owner"]
        )
        task_input = {
            "db_memory_payload": {
                **request.constraints,
                **request.metadata,
            },
            "db_memory_prompt": request.prompt,
        }
        planned = await self.runtime._planned_task_for_capability(
            operation.id, capability
        )
        task = (
            Task(
                id=f"db-task-{uuid4()}",
                operation_id=operation.id,
                capability_id=capability.id,
                executor_id=capability.executor,
                input=task_input,
                required_evidence=capability.output_evidence,
                metadata={"owner": capability.owner, "reason": "db_memory_semantics"},
            )
            if planned is None
            else replace(planned, input=task_input)
        )
        tasks.append(task)
        evidence = await self.runtime.execute_task(
            task,
            operation,
            context={"capability_owner": capability.owner},
        )
        evidence_store.add_many(evidence)

    async def _relationship_payload_if_needed(
        self,
        request: DbRequest,
        intent: DbIntent,
        contract: DbOperationContract,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
    ) -> dict[str, Any] | None:
        if intent.kind is not DbIntentKind.CATALOG_ASSISTED_DATA_QUERY:
            return None
        await self._execute_capability(
            "catalog.schema.search",
            contract,
            operation,
            tasks,
            evidence_store,
            {"store_id": store_id, "query": request.prompt, "limit": 10},
        )
        from_table, to_table = self.query_planner.relationship_tables_for_prompt(
            request.prompt, schema
        )
        if not from_table or not to_table:
            return None
        relationship_evidence = await self._execute_capability(
            "catalog.relationship_paths.find",
            contract,
            operation,
            tasks,
            evidence_store,
            {
                "store_id": store_id,
                "from_assets": [from_table],
                "to_assets": [to_table],
                "relationship_types": ["foreign_key", "references"],
                "max_hops": 3,
                "max_paths": 3,
            },
        )
        return relationship_evidence[0].payload if relationship_evidence else None

    async def _inspect_schema_if_available(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
    ) -> Evidence | None:
        cached = self.runtime.cached_schema_evidence(operation_id=operation.id)
        if cached is not None:
            evidence_store.add(cached)
            return cached
        persisted = self.runtime.persisted_schema_evidence(operation_id=operation.id)
        if persisted is not None:
            evidence_store.add(persisted)
            return persisted
        capability = self._first_capability("db.schema.inspect")
        if capability is None:
            return None
        try:
            evidence = await self._execute_direct_capability(
                capability,
                operation,
                tasks,
                evidence_store,
                {},
                reason="planning_schema_context",
            )
        except Exception as exc:
            fallback = self.runtime.stale_persisted_schema_evidence(
                operation_id=operation.id,
                error=exc,
            )
            if fallback is None:
                raise
            evidence_store.add(fallback)
            return fallback
        if evidence:
            self.runtime.remember_schema_evidence(evidence[0])
        return evidence[0] if evidence else None

    async def _register_catalog_source_if_available(
        self,
        operation: Operation,
        request: DbRequest,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
    ) -> None:
        capability = self._first_capability("catalog.source.register")
        if capability is None or not schema:
            return
        await self._execute_direct_capability(
            capability,
            operation,
            tasks,
            evidence_store,
            {
                "schema": _schema_with_catalog_metadata(
                    schema,
                    runtime=self.runtime,
                    store_id=store_id,
                ),
                "store_type": schema.get("database_type") or "db",
                "connection_string": request.metadata.get("connection_string"),
                "store_id": store_id,
                "persist": bool(
                    _runtime_from_db_option(self.runtime, "catalog_profile_key")
                ),
            },
            reason="planning_catalog_registration",
        )

    async def _execute_capability(
        self,
        capability_id: str,
        contract: DbOperationContract,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        input: dict[str, Any],
    ) -> tuple[Evidence, ...]:
        capability = _selected_capability(contract, capability_id)
        if capability is None:
            return ()
        resolved = self.runtime.registry.get_capability(
            capability["id"], owner=capability["owner"]
        )
        return await self._execute_direct_capability(
            resolved, operation, tasks, evidence_store, input
        )

    async def _execute_direct_capability(
        self,
        capability: Capability,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        input: dict[str, Any],
        *,
        reason: str | None = None,
    ) -> tuple[Evidence, ...]:
        planned = await self.runtime._planned_task_for_capability(
            operation.id, capability
        )
        task = (
            Task(
                id=f"db-task-{uuid4()}",
                operation_id=operation.id,
                capability_id=capability.id,
                executor_id=capability.executor,
                input=input,
                required_evidence=capability.output_evidence,
                metadata={"owner": capability.owner, "reason": reason or "contract"},
            )
            if planned is None
            else replace(planned, input=input)
        )
        tasks.append(task)
        evidence = await self.runtime.execute_task(
            task,
            operation,
            context={"capability_owner": capability.owner},
        )
        evidence_store.add_many(evidence)
        return evidence

    async def _delegate_worker(
        self,
        role: str,
        request: DbRequest,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        input: dict[str, Any],
    ) -> tuple[Evidence, ...]:
        for worker in self.runtime.registry.workers:
            if worker.role != role:
                continue
            for capability_id in worker.capability_ids:
                try:
                    capability = self.runtime.registry.get_capability(
                        capability_id,
                        owner=worker.owner,
                    )
                except KeyError:
                    continue
                await self.runtime.store.append_event(
                    self.runtime._runtime_event(
                        type=RuntimeEventType.WORKER_DELEGATED,
                        operation_id=operation.id,
                        capability=capability,
                        message=(
                            f"Delegated {role} work to {worker.owner}:{worker.id}."
                        ),
                        payload={
                            "worker_id": worker.id,
                            "worker_owner": worker.owner,
                            "worker_role": worker.role,
                            "capability_id": capability.id,
                            "prompt": request.prompt,
                        },
                    )
                )
                task = await self._persist_worker_task(
                    capability,
                    operation,
                    tasks,
                    input,
                    worker_id=worker.id,
                )
                worker_runtime = WorkerRuntime(
                    kernel=self.runtime.kernel,
                    options=WorkerRuntimeOptions(
                        worker_id=worker.id,
                        owner=worker.owner,
                        max_concurrency=worker.max_concurrency,
                    ),
                )
                handoff = await worker_runtime.handoff_task(
                    task.id,
                    reason=f"db_delegate:{role}",
                    metadata={"worker_role": worker.role, "prompt": request.prompt},
                )
                result = await worker_runtime.execute_handoff(
                    handoff,
                    context={"capability_owner": capability.owner},
                )
                if result.error is not None:
                    _raise_db_worker_error(result.error, result.execution)
                if result.execution is None:
                    return ()
                evidence_store.add_many(result.execution.evidence)
                return result.execution.evidence
        return ()

    async def _persist_worker_task(
        self,
        capability: Capability,
        operation: Operation,
        tasks: list[Task],
        input: dict[str, Any],
        *,
        worker_id: str,
    ) -> Task:
        planned = await self.runtime._planned_task_for_capability(
            operation.id, capability
        )
        task = (
            Task(
                id=f"db-task-{uuid4()}",
                operation_id=operation.id,
                capability_id=capability.id,
                executor_id=capability.executor,
                input=input,
                required_evidence=capability.output_evidence,
                metadata={"owner": capability.owner, "reason": f"worker:{worker_id}"},
            )
            if planned is None
            else replace(
                planned,
                input=input,
                metadata={
                    **planned.metadata,
                    "owner": capability.owner,
                    "reason": f"worker:{worker_id}",
                },
            )
        )
        tasks.append(task)
        if planned is None:
            await self.runtime._persist_planned_task(task)
        else:
            await self.runtime.store.save_task(task)
        return task

    def _first_capability(self, capability_id: str) -> Capability | None:
        for capability in self.runtime.registry.capabilities:
            if capability.id == capability_id:
                return capability
        return None

    @staticmethod
    def _needs_catalog(contract: DbOperationContract) -> bool:
        return any(
            capability["id"].startswith("catalog.")
            for capability in contract.metadata.get("selected_capabilities", [])
        )


def _selected_capability(
    contract: DbOperationContract, capability_id: str
) -> dict[str, str] | None:
    for item in contract.metadata.get("selected_capabilities", []):
        if item.get("id") == capability_id:
            return {"id": item["id"], "owner": item["owner"]}
    return None


def _store_id_for_request(request: DbRequest, runtime: Any | None = None) -> str:
    value = (
        request.metadata.get("store_id")
        or request.constraints.get("store_id")
        or (request.source_scope[0] if request.source_scope else None)
        or _runtime_from_db_option(runtime, "catalog_store_id")
    )
    return str(value or "runtime_source")


def _schema_with_catalog_metadata(
    schema: dict[str, Any],
    *,
    runtime: Any,
    store_id: str,
) -> dict[str, Any]:
    copied = dict(schema)
    copied["store_id"] = store_id
    profile_key = _runtime_from_db_option(runtime, "catalog_profile_key")
    if profile_key:
        copied["profile_key"] = str(profile_key)
        metadata = dict(copied.get("metadata") or {})
        metadata.setdefault("profile_key", str(profile_key))
        copied["metadata"] = metadata
    return copied


def _runtime_from_db_option(runtime: Any | None, key: str) -> Any | None:
    if runtime is None:
        return None
    options = getattr(getattr(runtime, "config", None), "metadata", {}).get(
        "from_db_options"
    )
    if isinstance(options, dict):
        return options.get(key)
    return None


def _raise_db_worker_error(error: BaseException, execution: Any | None) -> None:
    if isinstance(error, RuntimeKernelGovernanceBlocked):
        from .runtime import DbRuntimeGovernanceBlocked

        if execution is None:
            raise error
        raise DbRuntimeGovernanceBlocked(
            operation=execution.operation,
            task=execution.task,
            governance=(
                execution.governance
                if execution.governance is not None
                else GovernanceResult(False, False, True)
            ),
        ) from error
    if isinstance(error, RuntimeKernelTaskNotRunnable) and execution is not None:
        from .runtime import DbRuntimeTaskNotRunnable

        raise DbRuntimeTaskNotRunnable(execution.task, str(error)) from error
    raise error


def _lineage_entity_for_request(request: DbRequest, table: str | None) -> str:
    value = (
        request.metadata.get("entity_id")
        or request.constraints.get("entity_id")
        or request.metadata.get("table")
        or request.constraints.get("table")
    )
    if value:
        entity = str(value)
        return entity if ":" in entity else f"table:{entity}"
    return f"table:{table}" if table else "table:unknown"
