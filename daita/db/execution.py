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
    TaskDependency,
    WorkerRuntime,
    WorkerRuntimeOptions,
)

from .analysis import with_analysis_evidence_trace
from .evidence import DbEvidenceStore, InMemoryDbEvidenceStore
from .models import DbIntent, DbIntentKind, DbOperationContract, DbRequest
from .query_planning import DbQueryPlanner
from .query_sql_validation import sql_fingerprint


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
            "planner_strategy": None,
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
            planning_context = await self._build_planning_context(
                request,
                operation,
                tasks,
                evidence_store,
                schema_evidence=schema_evidence,
                catalog_evidence=tuple(
                    item
                    for item in evidence_store.list()
                    if item.kind.startswith("catalog.")
                    or item.kind in {"schema.search_result", "catalog.source"}
                ),
                relationship_evidence=tuple(
                    item
                    for item in evidence_store.list()
                    if item.kind == "relationship.paths"
                ),
            )
            plan_evidence, strategy_warnings, strategy_diagnostics = (
                await self._plan_query(
                    request,
                    intent,
                    operation,
                    schema,
                    relationship_payload,
                    planning_context,
                    tasks,
                    evidence_store,
                )
            )
            warnings.extend(strategy_warnings)
            diagnostics["planner_strategy"] = strategy_diagnostics.get("strategy")
            diagnostics["query_plan"] = strategy_diagnostics

            validation = await self._validate_query_plan(
                operation,
                tasks,
                evidence_store,
                plan_evidence=plan_evidence,
                planning_context=planning_context,
            )
            if (
                not validation.payload.get("valid")
                and self.runtime.db_llm_service.available
            ):
                repaired = await self._repair_query_plan(
                    operation,
                    tasks,
                    evidence_store,
                    planning_context=planning_context,
                    prior_plan=plan_evidence,
                    failure=validation,
                    repair_attempt=1,
                )
                if repaired is not None:
                    plan_evidence = repaired
                    validation = await self._validate_query_plan(
                        operation,
                        tasks,
                        evidence_store,
                        plan_evidence=plan_evidence,
                        planning_context=planning_context,
                    )

            sql = validation.payload.get("accepted_sql")
            diagnostics["planned_sql"] = sql
            if sql:
                try:
                    sql_validation = await self._execute_sql_validation(
                        contract,
                        operation,
                        tasks,
                        evidence_store,
                        sql,
                        plan_validation=validation,
                    )
                    await self._execute_validated_read(
                        contract,
                        operation,
                        tasks,
                        evidence_store,
                        sql_validation,
                    )
                except Exception as exc:
                    if self.runtime.db_llm_service.available:
                        failure = await self._record_failure_evidence(
                            operation,
                            evidence_store,
                            "query.plan.validation",
                            {
                                "valid": False,
                                "failure": "execution_failed",
                                "error": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                },
                            },
                        )
                        repaired = await self._repair_query_plan(
                            operation,
                            tasks,
                            evidence_store,
                            planning_context=planning_context,
                            prior_plan=plan_evidence,
                            failure=failure,
                            repair_attempt=1,
                        )
                        if repaired is not None:
                            validation = await self._validate_query_plan(
                                operation,
                                tasks,
                                evidence_store,
                                plan_evidence=repaired,
                                planning_context=planning_context,
                            )
                            repaired_sql = validation.payload.get("accepted_sql")
                            if repaired_sql:
                                sql_validation = await self._execute_sql_validation(
                                    contract,
                                    operation,
                                    tasks,
                                    evidence_store,
                                    repaired_sql,
                                    plan_validation=validation,
                                )
                                await self._execute_validated_read(
                                    contract,
                                    operation,
                                    tasks,
                                    evidence_store,
                                    sql_validation,
                                )
                                diagnostics["planned_sql"] = repaired_sql
                                return DbExecutionOutcome(
                                    evidence=evidence_store.list(),
                                    tasks=tuple(tasks),
                                    diagnostics={
                                        **diagnostics,
                                        "evidence_kinds": [
                                            item.kind for item in evidence_store.list()
                                        ],
                                        "evidence_refs": list(evidence_store.refs()),
                                    },
                                    warnings=tuple(warnings),
                                )
                    raise
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

    async def _build_planning_context(
        self,
        request: DbRequest,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        schema_evidence: Evidence | None,
        catalog_evidence: tuple[Evidence, ...],
        relationship_evidence: tuple[Evidence, ...],
        analysis_metadata: dict[str, Any] | None = None,
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> Evidence:
        if schema_evidence is not None and schema_evidence.id is None:
            schema_evidence = await self._persist_runtime_evidence(
                operation,
                schema_evidence,
            )
            evidence_store.add(schema_evidence)
        capability = self.runtime.registry.get_capability(
            "db.planning.context.build", owner="db_runtime"
        )
        evidence = await self._execute_direct_capability(
            capability,
            operation,
            tasks,
            evidence_store,
            {
                "prompt": request.prompt,
                "schema_evidence_id": schema_evidence.id if schema_evidence else None,
                "catalog_evidence_ids": [
                    item.id for item in catalog_evidence if item.id is not None
                ],
                "relationship_evidence_ids": [
                    item.id for item in relationship_evidence if item.id is not None
                ],
            },
            reason="planning_context",
            metadata=analysis_metadata,
            dependencies=extra_dependencies,
        )
        if not evidence:
            raise RuntimeError("planning.context evidence was not produced")
        return evidence[0]

    async def _plan_query(
        self,
        request: DbRequest,
        intent: DbIntent,
        operation: Operation,
        schema: dict[str, Any],
        relationship_payload: dict[str, Any] | None,
        planning_context: Evidence,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        analysis_metadata: dict[str, Any] | None = None,
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> tuple[Evidence, tuple[str, ...], dict[str, Any]]:
        route = _planner_route(
            request,
            schema,
            llm_available=self.runtime.db_llm_service.available,
        )
        if route["strategy"] == "llm":
            capability = self.runtime.registry.get_capability(
                "db.query.plan", owner="db_runtime"
            )
            evidence = await self._execute_direct_capability(
                capability,
                operation,
                tasks,
                evidence_store,
                {
                    "planning_context_evidence_id": planning_context.id,
                    "prompt": request.prompt,
                },
                reason="llm_query_planning",
                metadata=analysis_metadata,
                dependencies=(
                    TaskDependency(
                        kind="evidence",
                        evidence_kind="planning.context",
                        evidence_id=planning_context.id,
                        operation_id=operation.id,
                    ),
                    *extra_dependencies,
                ),
            )
            if not evidence:
                raise RuntimeError("query.plan.proposal evidence was not produced")
            return evidence[-1], (), route

        plan = self.query_planner.plan_read_query(
            request,
            intent,
            operation,
            schema,
            relationship_payload=relationship_payload,
        )
        plan_evidence = (
            replace(
                plan.evidence,
                metadata={**plan.evidence.metadata, **analysis_metadata},
            )
            if analysis_metadata
            else plan.evidence
        )
        persisted = await self._persist_runtime_evidence(operation, plan_evidence)
        evidence_store.add(persisted)
        return persisted, plan.warnings, {**route, **plan.diagnostics}

    async def _validate_query_plan(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        plan_evidence: Evidence,
        planning_context: Evidence,
        analysis_metadata: dict[str, Any] | None = None,
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> Evidence:
        capability = self.runtime.registry.get_capability(
            "db.query.plan.validate", owner="db_runtime"
        )
        evidence = await self._execute_direct_capability(
            capability,
            operation,
            tasks,
            evidence_store,
            {
                "plan_evidence_id": plan_evidence.id,
                "planning_context_evidence_id": planning_context.id,
            },
            reason="query_plan_validation",
            metadata=analysis_metadata,
            dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind="query.plan.proposal",
                    evidence_id=plan_evidence.id,
                    operation_id=operation.id,
                ),
                TaskDependency(
                    kind="evidence",
                    evidence_kind="planning.context",
                    evidence_id=planning_context.id,
                    operation_id=operation.id,
                ),
                *extra_dependencies,
            ),
        )
        if not evidence:
            raise RuntimeError("query.plan.validation evidence was not produced")
        return evidence[0]

    async def _execute_sql_validation(
        self,
        contract: DbOperationContract,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        sql: str,
        *,
        plan_validation: Evidence,
        analysis_metadata: dict[str, Any] | None = None,
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> Evidence:
        evidence = await self._execute_capability(
            "db.sql.validate",
            contract,
            operation,
            tasks,
            evidence_store,
            {"sql": sql, "operation": "query"},
            metadata=analysis_metadata,
            dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind="query.plan.validation",
                    evidence_id=plan_validation.id,
                    evidence_payload={"valid": True},
                    operation_id=operation.id,
                ),
                *extra_dependencies,
            ),
        )
        if not evidence:
            raise RuntimeError("sql.validation evidence was not produced")
        return evidence[0]

    async def _execute_validated_read(
        self,
        contract: DbOperationContract,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        validation: Evidence,
        analysis_metadata: dict[str, Any] | None = None,
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> tuple[Evidence, ...]:
        fingerprint = (
            validation.payload.get("sql_fingerprint")
            or (validation.payload.get("statement_facts") or {}).get("sql_fingerprint")
            or sql_fingerprint(str(validation.payload.get("sql") or ""))
        )
        return await self._execute_capability(
            "db.sql.execute_read",
            contract,
            operation,
            tasks,
            evidence_store,
            {
                "validated_evidence_id": validation.id,
                "sql_ref": "sql.validation",
                "sql_fingerprint": fingerprint,
            },
            metadata=analysis_metadata,
            dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind="sql.validation",
                    evidence_id=validation.id,
                    evidence_owner=validation.owner,
                    producer_task_id=validation.task_id,
                    producer_capability_id="db.sql.validate",
                    evidence_payload={"valid": True},
                    operation_id=operation.id,
                    payload_fingerprint=validation.metadata.get("payload_fingerprint"),
                ),
                *extra_dependencies,
            ),
        )

    async def _repair_query_plan(
        self,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        planning_context: Evidence,
        prior_plan: Evidence,
        failure: Evidence,
        repair_attempt: int,
    ) -> Evidence | None:
        if repair_attempt > 1:
            return None
        capability = self.runtime.registry.get_capability(
            "db.query.repair", owner="db_runtime"
        )
        evidence = await self._execute_direct_capability(
            capability,
            operation,
            tasks,
            evidence_store,
            {
                "planning_context_evidence_id": planning_context.id,
                "prior_plan_evidence_id": prior_plan.id,
                "failure_evidence_id": failure.id,
                "repair_attempt": repair_attempt,
            },
            reason="query_plan_repair",
            dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind="planning.context",
                    evidence_id=planning_context.id,
                    operation_id=operation.id,
                ),
                TaskDependency(
                    kind="evidence",
                    evidence_kind="query.plan.proposal",
                    evidence_id=prior_plan.id,
                    operation_id=operation.id,
                ),
                TaskDependency(
                    kind="evidence",
                    evidence_kind=failure.kind,
                    evidence_id=failure.id,
                    evidence_accepted=failure.accepted,
                    operation_id=operation.id,
                ),
            ),
        )
        proposals = [item for item in evidence if item.kind == "query.plan.proposal"]
        return proposals[-1] if proposals else None

    async def _persist_runtime_evidence(
        self,
        operation: Operation,
        evidence: Evidence,
    ) -> Evidence:
        evidence_id = evidence.id or f"evidence-{uuid4()}"
        persisted = replace(
            evidence,
            id=evidence_id,
            operation_id=evidence.operation_id or operation.id,
            metadata={
                **evidence.metadata,
                "payload_fingerprint": _stable_hash(evidence.payload),
            },
        )
        await self.runtime.store.save_evidence(persisted)
        return persisted

    async def _record_failure_evidence(
        self,
        operation: Operation,
        evidence_store: DbEvidenceStore,
        kind: str,
        payload: dict[str, Any],
    ) -> Evidence:
        evidence = await self._persist_runtime_evidence(
            operation,
            Evidence(
                kind=kind,
                owner="db_runtime",
                operation_id=operation.id,
                accepted=False,
                payload=payload,
            ),
        )
        evidence_store.add(evidence)
        return evidence

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
        *,
        dependencies: tuple[TaskDependency, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        capability = _selected_capability(contract, capability_id)
        if capability is None:
            return ()
        resolved = self.runtime.registry.get_capability(
            capability["id"], owner=capability["owner"]
        )
        return await self._execute_direct_capability(
            resolved,
            operation,
            tasks,
            evidence_store,
            input,
            dependencies=dependencies,
            metadata=metadata,
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
        dependencies: tuple[TaskDependency, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        planned = await self.runtime._planned_task_for_capability(
            operation.id,
            capability,
            metadata_match=metadata,
        )
        task_metadata = with_analysis_evidence_trace(
            {
                "owner": capability.owner,
                "reason": reason or "contract",
                **dict(metadata or {}),
            }
        )
        task = (
            Task(
                id=f"db-task-{uuid4()}",
                operation_id=operation.id,
                capability_id=capability.id,
                executor_id=capability.executor,
                input=input,
                required_evidence=capability.output_evidence,
                metadata=task_metadata,
                dependencies=dependencies,
            )
            if planned is None
            else replace(
                planned,
                input=input,
                dependencies=dependencies or planned.dependencies,
                metadata={
                    **with_analysis_evidence_trace(
                        {
                            **planned.metadata,
                            "owner": capability.owner,
                            "reason": reason
                            or planned.metadata.get("reason")
                            or "contract",
                            **dict(metadata or {}),
                        }
                    ),
                },
            )
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
                await self.runtime.kernel.append_event(
                    RuntimeEventType.WORKER_DELEGATED,
                    operation_id=operation.id,
                    capability=capability,
                    message=f"Delegated {role} work to {worker.owner}:{worker.id}.",
                    payload={
                        "worker_id": worker.id,
                        "worker_owner": worker.owner,
                        "worker_role": worker.role,
                        "capability_id": capability.id,
                        "prompt": request.prompt,
                        **{
                            key: value
                            for key, value in input.items()
                            if key.startswith("analysis_")
                        },
                    },
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
        analysis_trace = {
            key: value
            for key, value in input.items()
            if key
            in {
                "analysis_id",
                "analysis_step_id",
                "analysis_step_kind",
                "analysis_plan_evidence_id",
            }
        }
        planned = await self.runtime._planned_task_for_capability(
            operation.id,
            capability,
            metadata_match=analysis_trace or None,
        )
        task = (
            Task(
                id=f"db-task-{uuid4()}",
                operation_id=operation.id,
                capability_id=capability.id,
                executor_id=capability.executor,
                input=input,
                required_evidence=capability.output_evidence,
                metadata={
                    **with_analysis_evidence_trace(
                        {
                            "owner": capability.owner,
                            "reason": f"worker:{worker_id}",
                            **analysis_trace,
                        }
                    )
                },
            )
            if planned is None
            else replace(
                planned,
                input=input,
                metadata={
                    **with_analysis_evidence_trace(
                        {
                            **planned.metadata,
                            "owner": capability.owner,
                            "reason": f"worker:{worker_id}",
                            **analysis_trace,
                        }
                    ),
                },
            )
        )
        tasks.append(task)
        if planned is None:
            task = await self.runtime._plan_kernel_task(task)
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


def _planner_route(
    request: DbRequest,
    schema: dict[str, Any],
    *,
    llm_available: bool,
) -> dict[str, Any]:
    prompt = request.prompt.lower()
    explicit_sql = request.metadata.get("sql") or request.constraints.get("sql")
    if explicit_sql:
        return {
            "strategy": "deterministic",
            "reason_codes": ["explicit_sql"],
            "estimated_complexity": "low",
        }
    analytical_terms = {
        "top",
        "highest",
        "lowest",
        "average",
        "avg",
        "sum",
        "total",
        "by",
        "per",
        "rate",
        "percent",
        "revenue",
        "churn",
        "active",
        "enterprise",
    }
    reason_codes = sorted(term for term in analytical_terms if term in prompt)
    table_mentions = [
        str(table.get("name"))
        for table in schema.get("tables", []) or []
        if table.get("name") and str(table.get("name")).lower() in prompt
    ]
    if len(table_mentions) > 1:
        reason_codes.append("multiple_table_mentions")
    if llm_available and reason_codes:
        return {
            "strategy": "llm",
            "reason_codes": reason_codes,
            "estimated_complexity": "medium",
        }
    return {
        "strategy": "deterministic",
        "reason_codes": reason_codes or ["simple_fast_path"],
        "estimated_complexity": "low",
    }


def _stable_hash(value: Any) -> str:
    import hashlib
    import json

    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
