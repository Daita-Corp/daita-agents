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
    Operation,
    Task,
    TaskDependency,
)

from .analysis import with_analysis_evidence_trace
from .capabilities import (
    PLANNING_CONTEXT_EVIDENCE,
    QUERY_PLAN_PROPOSAL_EVIDENCE,
    QUERY_PLAN_VALIDATION_EVIDENCE,
    QUERY_PLAN_REPAIR_EVIDENCE,
    QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
    SCHEMA_RELATIONSHIP_PATH_EVIDENCE,
    SCHEMA_SEARCH_RESULT_EVIDENCE,
    SQL_VALIDATION_EVIDENCE,
)
from .evidence import DbEvidenceStore
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


@dataclass(frozen=True)
class _ColumnValueEvidenceState:
    key: tuple[str, str]
    profile: dict[str, Any]
    source: str
    fresh: bool
    operation_local: bool
    stale_reason: str | None = None


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

        if self._needs_catalog(contract) or intent.kind in {
            DbIntentKind.DATA_QUERY,
            DbIntentKind.CATALOG_ASSISTED_DATA_QUERY,
        }:
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
            route = _planner_route(
                request,
                schema,
                llm_available=self.runtime.db_llm_service.available,
            )
            planning_context: Evidence | None = None
            if (
                route["strategy"] == "deterministic"
                and intent.kind is DbIntentKind.DATA_QUERY
            ):
                if (
                    self._value_grounding_available()
                    and _deterministic_value_hint_search_needed(request, schema)
                ):
                    await self._search_catalog_column_values_if_available(
                        request,
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
                        catalog_evidence=_catalog_evidence_for_planning(evidence_store),
                        relationship_evidence=tuple(
                            item
                            for item in evidence_store.list()
                            if item.kind == SCHEMA_RELATIONSHIP_PATH_EVIDENCE
                        ),
                        analysis_metadata={"predicate_column_value_search": True},
                    )
                plan_evidence, validation, strategy_warnings, strategy_diagnostics = (
                    await self._prepare_deterministic_read(
                        request,
                        operation,
                        schema_evidence,
                        tasks,
                        evidence_store,
                        planning_context=planning_context,
                    )
                )
                if _plan_has_literal_predicates(plan_evidence, schema):
                    if self._value_grounding_available():
                        await self._search_catalog_column_values_if_available(
                            request,
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
                        catalog_evidence=_catalog_evidence_for_planning(evidence_store),
                        relationship_evidence=tuple(
                            item
                            for item in evidence_store.list()
                            if item.kind == SCHEMA_RELATIONSHIP_PATH_EVIDENCE
                        ),
                    )
                    planning_context = (
                        await self._resolve_predicate_value_profiles_if_available(
                            request,
                            operation,
                            schema,
                            tasks,
                            evidence_store,
                            diagnostics["store_id"],
                            schema_evidence=schema_evidence,
                            planning_context=planning_context,
                            plan_evidence=plan_evidence,
                        )
                    )
                    validation = await self._validate_query_plan(
                        operation,
                        tasks,
                        evidence_store,
                        plan_evidence=plan_evidence,
                        planning_context=planning_context,
                    )
            else:
                if route["strategy"] == "llm" and self._value_grounding_available():
                    await self._search_catalog_column_values_if_available(
                        request,
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
                    catalog_evidence=_catalog_evidence_for_planning(evidence_store),
                    relationship_evidence=tuple(
                        item
                        for item in evidence_store.list()
                        if item.kind == SCHEMA_RELATIONSHIP_PATH_EVIDENCE
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
                if (
                    route["strategy"] != "llm"
                    and _plan_has_literal_predicates(plan_evidence, schema)
                    and not _catalog_column_value_search_exists(evidence_store)
                    and self._value_grounding_available()
                ):
                    await self._search_catalog_column_values_if_available(
                        request,
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
                        catalog_evidence=_catalog_evidence_for_planning(evidence_store),
                        relationship_evidence=tuple(
                            item
                            for item in evidence_store.list()
                            if item.kind == SCHEMA_RELATIONSHIP_PATH_EVIDENCE
                        ),
                        analysis_metadata={"predicate_column_value_search": True},
                    )
                planning_context = (
                    await self._resolve_predicate_value_profiles_if_available(
                        request,
                        operation,
                        schema,
                        tasks,
                        evidence_store,
                        diagnostics["store_id"],
                        schema_evidence=schema_evidence,
                        planning_context=planning_context,
                        plan_evidence=plan_evidence,
                    )
                )
                validation = await self._validate_query_plan(
                    operation,
                    tasks,
                    evidence_store,
                    plan_evidence=plan_evidence,
                    planning_context=planning_context,
                )
            warnings.extend(strategy_warnings)
            diagnostics["planner_strategy"] = strategy_diagnostics.get("strategy")
            diagnostics["query_plan"] = strategy_diagnostics
            if (
                not validation.payload.get("valid")
                and self.runtime.db_llm_service.available
            ):
                if planning_context is None:
                    planning_context = await self._build_planning_context(
                        request,
                        operation,
                        tasks,
                        evidence_store,
                        schema_evidence=schema_evidence,
                        catalog_evidence=_catalog_evidence_for_planning(evidence_store),
                        relationship_evidence=tuple(
                            item
                            for item in evidence_store.list()
                            if item.kind == SCHEMA_RELATIONSHIP_PATH_EVIDENCE
                        ),
                        analysis_metadata={"prepare_read_repair_context": True},
                    )
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
                    read_evidence = await self._execute_validated_read(
                        contract,
                        operation,
                        tasks,
                        evidence_store,
                        sql_validation,
                    )
                    zero_row_sql = (
                        await self._repair_zero_row_result_if_available(
                            request,
                            contract,
                            operation,
                            schema,
                            tasks,
                            evidence_store,
                            planning_context=planning_context,
                            plan_evidence=plan_evidence,
                            sql_validation=sql_validation,
                            read_evidence=read_evidence,
                        )
                        if planning_context is not None
                        else None
                    )
                    if zero_row_sql:
                        diagnostics["planned_sql"] = zero_row_sql
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
            planning_context=planning_context.payload,
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

    async def _prepare_deterministic_read(
        self,
        request: DbRequest,
        operation: Operation,
        schema_evidence: Evidence | None,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        planning_context: Evidence | None = None,
    ) -> tuple[Evidence, Evidence, tuple[str, ...], dict[str, Any]]:
        if schema_evidence is not None and schema_evidence.id is None:
            schema_evidence = await self._persist_runtime_evidence(
                operation,
                schema_evidence,
            )
            evidence_store.add(schema_evidence)
        capability = self.runtime.registry.get_capability(
            "db.query.prepare_read", owner="db_runtime"
        )
        evidence = await self._execute_direct_capability(
            capability,
            operation,
            tasks,
            evidence_store,
            {
                "prompt": request.prompt,
                "schema_evidence_id": schema_evidence.id if schema_evidence else None,
                "planning_context_evidence_id": (
                    planning_context.id if planning_context is not None else None
                ),
            },
            reason="deterministic_read_prepare",
            dependencies=(
                (
                    TaskDependency(
                        kind="evidence",
                        evidence_kind=schema_evidence.kind,
                        evidence_id=schema_evidence.id,
                        evidence_accepted=schema_evidence.accepted,
                        operation_id=operation.id,
                    ),
                )
                if schema_evidence is not None and schema_evidence.id is not None
                else ()
            )
            + (
                (
                    TaskDependency(
                        kind="evidence",
                        evidence_kind=planning_context.kind,
                        evidence_id=planning_context.id,
                        evidence_accepted=planning_context.accepted,
                        operation_id=operation.id,
                    ),
                )
                if planning_context is not None and planning_context.id is not None
                else ()
            ),
        )
        plan_evidence = next(
            (item for item in evidence if item.kind == QUERY_PLAN_PROPOSAL_EVIDENCE),
            None,
        )
        validation_evidence = next(
            (item for item in evidence if item.kind == QUERY_PLAN_VALIDATION_EVIDENCE),
            None,
        )
        if plan_evidence is None or validation_evidence is None:
            raise RuntimeError(
                "prepare_read did not produce plan and validation evidence"
            )
        diagnostics = dict(plan_evidence.payload)
        structured = diagnostics.get("structured_plan")
        if isinstance(structured, dict):
            diagnostics = {
                "strategy": diagnostics.get("strategy", "deterministic"),
                "planner": "deterministic",
                "sql": diagnostics.get("sql") or structured.get("selected_sql"),
                "prepare_read": True,
            }
        return plan_evidence, validation_evidence, (), diagnostics

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
                    evidence_kind=QUERY_PLAN_PROPOSAL_EVIDENCE,
                    evidence_id=plan_evidence.id,
                    evidence_accepted=plan_evidence.accepted,
                    operation_id=operation.id,
                ),
                TaskDependency(
                    kind="evidence",
                    evidence_kind=PLANNING_CONTEXT_EVIDENCE,
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
                    evidence_kind=QUERY_PLAN_VALIDATION_EVIDENCE,
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
                    evidence_kind=SQL_VALIDATION_EVIDENCE,
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

    async def _repair_zero_row_result_if_available(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        planning_context: Evidence,
        plan_evidence: Evidence,
        sql_validation: Evidence,
        read_evidence: tuple[Evidence, ...],
    ) -> str | None:
        if not self.runtime.db_llm_service.available:
            return None
        if any(
            item.kind == QUERY_PLAN_REPAIR_EVIDENCE for item in evidence_store.list()
        ):
            return None
        query_result = _query_result_evidence(read_evidence)
        if query_result is None or not _query_result_is_zero_rows(query_result):
            return None
        sql = str(sql_validation.payload.get("sql") or "")
        if not sql or sql_validation.payload.get("is_read") is False:
            return None

        try:
            from .sql_analysis import analyze_sql

            analysis = analyze_sql(
                sql,
                dialect=str(
                    planning_context.payload.get("dialect")
                    or schema.get("database_type")
                    or ""
                ),
            )
        except Exception:
            return None
        predicates = _zero_row_literal_predicates(analysis.literal_predicates)
        if not predicates:
            return None
        hints = _column_value_hints_for_predicates(
            planning_context.payload,
            predicates,
        )
        if not hints:
            return None

        diagnosis = await self._persist_runtime_evidence(
            operation,
            Evidence(
                kind=QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
                owner="db_runtime",
                operation_id=operation.id,
                accepted=False,
                payload={
                    "valid": False,
                    "failure": "zero_row_result",
                    "prompt": request.prompt,
                    "sql": sql,
                    "sql_validation_evidence_id": sql_validation.id,
                    "query_result_evidence_id": query_result.id,
                    "prior_plan_evidence_id": plan_evidence.id,
                    "predicates": predicates,
                    "column_value_hints": hints,
                    "repair_policy": {
                        "max_attempts": 1,
                        "only_read_queries": True,
                        "only_literal_predicates": True,
                    },
                },
            ),
        )
        repaired = await self._repair_query_plan(
            operation,
            tasks,
            evidence_store,
            planning_context=planning_context,
            prior_plan=plan_evidence,
            failure=diagnosis,
            repair_attempt=1,
        )
        if repaired is None:
            return None
        validation = await self._validate_query_plan(
            operation,
            tasks,
            evidence_store,
            plan_evidence=repaired,
            planning_context=planning_context,
            analysis_metadata={"zero_row_repair": True},
            extra_dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
                    evidence_id=diagnosis.id,
                    evidence_accepted=diagnosis.accepted,
                    operation_id=operation.id,
                ),
            ),
        )
        repaired_sql = validation.payload.get("accepted_sql")
        if not validation.payload.get("valid") or not repaired_sql:
            return None
        repaired_sql_validation = await self._execute_sql_validation(
            contract,
            operation,
            tasks,
            evidence_store,
            str(repaired_sql),
            plan_validation=validation,
            analysis_metadata={"zero_row_repair": True},
            extra_dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
                    evidence_id=diagnosis.id,
                    evidence_accepted=diagnosis.accepted,
                    operation_id=operation.id,
                ),
            ),
        )
        repaired_read = await self._execute_validated_read(
            contract,
            operation,
            tasks,
            evidence_store,
            repaired_sql_validation,
            analysis_metadata={"zero_row_repair": True},
            extra_dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind=QUERY_ZERO_ROW_DIAGNOSIS_EVIDENCE,
                    evidence_id=diagnosis.id,
                    evidence_accepted=diagnosis.accepted,
                    operation_id=operation.id,
                ),
            ),
        )
        if _query_result_evidence(repaired_read) is not None:
            evidence_store.discard(query_result.id)
            return str(repaired_sql)
        return None

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
                    evidence_kind=PLANNING_CONTEXT_EVIDENCE,
                    evidence_id=planning_context.id,
                    operation_id=operation.id,
                ),
                TaskDependency(
                    kind="evidence",
                    evidence_kind=QUERY_PLAN_PROPOSAL_EVIDENCE,
                    evidence_id=prior_plan.id,
                    evidence_accepted=prior_plan.accepted,
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
        proposals = [
            item for item in evidence if item.kind == QUERY_PLAN_PROPOSAL_EVIDENCE
        ]
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
        cached = self.runtime.cached_catalog_source_evidence(
            operation_id=operation.id,
            schema=schema,
            store_id=store_id,
        )
        if cached is not None:
            persisted = await self._persist_runtime_evidence(operation, cached)
            evidence_store.add(persisted)
            return
        evidence = await self._execute_direct_capability(
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
        if evidence:
            self.runtime.remember_catalog_source_evidence(
                evidence[0],
                schema=schema,
                store_id=store_id,
            )

    async def _search_catalog_column_values_if_available(
        self,
        request: DbRequest,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
    ) -> None:
        search_capability = self._first_capability("catalog.column_values.search")
        if search_capability is None or not schema:
            return

        await self._execute_direct_capability(
            search_capability,
            operation,
            tasks,
            evidence_store,
            {
                "store_id": store_id,
                "query": request.prompt,
                "limit": 12,
                **_catalog_profile_ttl_input(self.runtime),
            },
            reason="catalog_column_value_profile_search",
        )

    async def _resolve_predicate_value_profiles_if_available(
        self,
        request: DbRequest,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
        *,
        schema_evidence: Evidence | None,
        planning_context: Evidence,
        plan_evidence: Evidence,
    ) -> Evidence:
        profile_capability = self._first_capability("db.column_values.profile")
        register_capability = self._first_capability("catalog.column_values.register")
        if profile_capability is None or register_capability is None or not schema:
            return planning_context

        sql = str(
            plan_evidence.payload.get("sql")
            or (plan_evidence.payload.get("structured_plan") or {}).get("selected_sql")
            or ""
        )
        if not sql:
            return planning_context
        try:
            from .sql_analysis import analyze_sql

            analysis = analyze_sql(
                sql,
                dialect=str(
                    planning_context.payload.get("dialect")
                    or schema.get("database_type")
                    or ""
                ),
            )
        except Exception:
            return planning_context

        evidence_index = _column_value_evidence_index(evidence_store, schema)
        candidates = _predicate_profile_candidates(
            analysis.literal_predicates,
            schema,
            evidence_index=evidence_index,
        )
        if not candidates:
            return planning_context

        registered = False
        for table, column in candidates:
            key = (table.lower(), column.lower())
            stored_state = evidence_index.get(key)
            stored_profile = stored_state.profile if stored_state is not None else None
            if (
                stored_profile is not None
                and stored_state is not None
                and not stored_state.fresh
                and _profile_capability_supports_fingerprint(profile_capability)
            ):
                fingerprint = await self._execute_direct_capability(
                    profile_capability,
                    operation,
                    tasks,
                    evidence_store,
                    _column_value_profile_input(table, column, fingerprint_only=True),
                    reason="column_value_source_fingerprint_check",
                )
                if fingerprint and _source_fingerprint_preserves_freshness(
                    stored_profile=stored_profile,
                    current_profile=fingerprint[0].payload,
                ):
                    await self._execute_direct_capability(
                        register_capability,
                        operation,
                        tasks,
                        evidence_store,
                        {
                            "store_id": store_id,
                            "profiles": [
                                _fresh_profile_from_preserved_fingerprint(
                                    stored_profile,
                                    fingerprint[0].payload,
                                )
                            ],
                            "source_evidence_id": fingerprint[0].id,
                            "persist": bool(
                                _runtime_from_db_option(
                                    self.runtime, "catalog_profile_key"
                                )
                            ),
                        },
                        reason="stale_catalog_column_value_registration",
                    )
                    registered = True
                    continue

            raw_profiles = await self._execute_direct_capability(
                profile_capability,
                operation,
                tasks,
                evidence_store,
                _column_value_profile_input(table, column),
                reason=(
                    "column_value_predicate_profile"
                    if stored_profile is None
                    else "stale_column_value_profile_refresh"
                ),
            )
            if not raw_profiles:
                continue
            await self._execute_direct_capability(
                register_capability,
                operation,
                tasks,
                evidence_store,
                {
                    "store_id": store_id,
                    "profiles": [raw_profiles[0].payload],
                    "source_evidence_id": raw_profiles[0].id,
                    "persist": bool(
                        _runtime_from_db_option(self.runtime, "catalog_profile_key")
                    ),
                },
                reason=(
                    "catalog_column_value_predicate_registration"
                    if stored_profile is None
                    else "stale_catalog_column_value_registration"
                ),
            )
            registered = True

        if not registered:
            return planning_context

        return await self._build_planning_context(
            request,
            operation,
            tasks,
            evidence_store,
            schema_evidence=schema_evidence,
            catalog_evidence=_catalog_evidence_for_planning(evidence_store),
            relationship_evidence=tuple(
                item
                for item in evidence_store.list()
                if item.kind == SCHEMA_RELATIONSHIP_PATH_EVIDENCE
            ),
            analysis_metadata={"predicate_column_value_profiles": True},
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

    def _first_capability(self, capability_id: str) -> Capability | None:
        for capability in self.runtime.registry.capabilities:
            if capability.id == capability_id:
                return capability
        return None

    def _value_grounding_available(self) -> bool:
        required = {
            "db.column_values.profile",
            "catalog.column_values.register",
            "catalog.column_values.search",
        }
        available = {capability.id for capability in self.runtime.registry.capabilities}
        return required <= available

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


def _column_value_profile_input(
    table: str,
    column: str,
    *,
    fingerprint_only: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "table": table,
        "column": column,
        "max_values": 25,
        "max_distinct_count": 100,
        "max_value_length": 80,
        "include_source_revision": True,
    }
    if fingerprint_only:
        payload["fingerprint_only"] = True
    return payload


def _predicate_profile_candidates(
    literal_predicates: tuple[Any, ...],
    schema: dict[str, Any],
    *,
    evidence_index: dict[tuple[str, str], _ColumnValueEvidenceState],
) -> tuple[tuple[str, str], ...]:
    table_columns = _schema_columns_by_table(schema)
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for predicate in literal_predicates:
        if str(getattr(predicate, "operator", "")).lower() not in {"=", "in", "like"}:
            continue
        column_ref = getattr(predicate, "column", None)
        column_name = str(getattr(column_ref, "name", "") or "")
        if not column_name:
            continue
        table_name = _predicate_table_name(
            str(getattr(column_ref, "table", "") or ""),
            column_name,
            table_columns,
        )
        if not table_name:
            continue
        key = (table_name.lower(), column_name.lower())
        if key in seen:
            continue
        state = evidence_index.get(key)
        if state is not None and (
            state.fresh or _catalog_profile_is_ineligible(state.profile)
        ):
            continue
        column = table_columns.get(table_name.lower(), {}).get(column_name.lower())
        if column is None or not _profile_candidate_shape(column):
            continue
        seen.add(key)
        candidates.append((table_name, column_name))
        if len(candidates) >= 5:
            break
    return tuple(candidates)


def _column_value_evidence_index(
    evidence_store: DbEvidenceStore,
    schema: dict[str, Any],
) -> dict[tuple[str, str], _ColumnValueEvidenceState]:
    states: dict[tuple[str, str], _ColumnValueEvidenceState] = {}
    metadata = schema.get("metadata") or {}
    schema_profiles = (
        metadata.get("column_value_profiles") if isinstance(metadata, dict) else {}
    )
    if isinstance(schema_profiles, dict):
        for profile in schema_profiles.values():
            if isinstance(profile, dict):
                _put_column_value_state(
                    states,
                    profile,
                    source="schema_metadata",
                    operation_local=False,
                    fresh=_catalog_profile_is_fresh(profile),
                )

    for evidence in evidence_store.list():
        if evidence.kind == "schema.column_value_search_result":
            for profile in _catalog_column_value_profiles(evidence):
                _put_column_value_state(
                    states,
                    profile,
                    source="catalog_search",
                    operation_local=False,
                    fresh=_catalog_profile_is_fresh(profile),
                )
        elif evidence.kind == "schema.column_value_profile":
            for profile in evidence.payload.get("profiles", []) or []:
                if isinstance(profile, dict):
                    _put_column_value_state(
                        states,
                        profile,
                        source="catalog_registration",
                        operation_local=True,
                        fresh=_operation_local_profile_is_fresh(profile),
                    )
    return states


def _put_column_value_state(
    states: dict[tuple[str, str], _ColumnValueEvidenceState],
    profile: dict[str, Any],
    *,
    source: str,
    operation_local: bool,
    fresh: bool,
) -> None:
    table = str(profile.get("table") or "").split(".")[-1].lower()
    column = str(profile.get("column") or "").lower()
    if not table or not column:
        return
    key = (table, column)
    candidate = _ColumnValueEvidenceState(
        key=key,
        profile=dict(profile),
        source=source,
        fresh=fresh,
        operation_local=operation_local,
        stale_reason=(
            str(profile.get("stale_reason")) if profile.get("stale_reason") else None
        ),
    )
    existing = states.get(key)
    if existing is None or _column_value_state_rank(
        candidate
    ) > _column_value_state_rank(existing):
        states[key] = candidate


def _column_value_state_rank(state: _ColumnValueEvidenceState) -> tuple[int, int, int]:
    return (
        1 if state.operation_local else 0,
        1 if state.fresh else 0,
        {"catalog_registration": 3, "catalog_search": 2, "schema_metadata": 1}.get(
            state.source, 0
        ),
    )


def _catalog_column_value_profiles(
    search_evidence: Evidence | None,
) -> tuple[dict[str, Any], ...]:
    if search_evidence is None:
        return ()
    return tuple(
        profile
        for profile in search_evidence.payload.get("profiles", []) or []
        if isinstance(profile, dict)
    )


def _catalog_profile_is_fresh(profile: dict[str, Any]) -> bool:
    if profile.get("stale"):
        return False
    if not _operation_local_profile_is_fresh(profile):
        return False
    return bool(profile.get("top_values") or profile.get("source_fingerprint"))


def _operation_local_profile_is_fresh(profile: dict[str, Any]) -> bool:
    if profile.get("profile_status") in {"stale", "skipped", "redacted"}:
        return False
    if profile.get("stale") or profile.get("redacted"):
        return False
    if profile.get("sampled") or profile.get("truncated"):
        return False
    return bool(profile.get("top_values"))


def _catalog_profile_is_ineligible(profile: dict[str, Any]) -> bool:
    return bool(profile.get("redacted")) or profile.get("profile_status") == "skipped"


def _source_fingerprint_preserves_freshness(
    *,
    stored_profile: dict[str, Any],
    current_profile: dict[str, Any],
) -> bool:
    stored_fingerprint = stored_profile.get("source_fingerprint")
    current_fingerprint = current_profile.get("source_fingerprint")
    if not stored_fingerprint or not current_fingerprint:
        return False
    if current_fingerprint != stored_fingerprint:
        return False
    current_status = str(
        current_profile.get("source_fingerprint_status")
        or stored_profile.get("source_fingerprint_status")
        or "best_effort"
    )
    if current_status == "unavailable":
        return False
    if stored_profile.get("stale") or stored_profile.get("profile_status") == "stale":
        if current_status == "authoritative":
            return True
        return (
            current_status == "best_effort"
            and stored_profile.get("stale_reason") == "profile_ttl_expired"
        )
    return current_status in {"authoritative", "best_effort"}


def _fresh_profile_from_preserved_fingerprint(
    stored_profile: dict[str, Any],
    current_profile: dict[str, Any],
) -> dict[str, Any]:
    fresh = {
        key: value
        for key, value in stored_profile.items()
        if key
        not in {
            "match_reasons",
            "profile_ref",
            "score",
            "stale",
            "stale_reason",
            "store_id",
        }
    }
    fresh["profile_status"] = "profiled"
    for key in (
        "source_fingerprint",
        "source_fingerprint_status",
        "source_fingerprint_reason",
        "source_revision",
    ):
        if key in current_profile:
            fresh[key] = current_profile[key]
    return fresh


def _profile_capability_supports_fingerprint(capability: Capability) -> bool:
    policy = capability.metadata.get("profile_policy")
    return isinstance(policy, dict) and bool(policy.get("fingerprint_only_supported"))


def _schema_columns_by_table(
    schema: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    for table in schema.get("tables", []) or []:
        table_name = str(table.get("name") or "")
        if not table_name:
            continue
        tables[table_name.lower()] = {
            str(column.get("name") or "").lower(): column
            for column in table.get("columns", []) or []
            if column.get("name")
        }
    return tables


def _predicate_table_name(
    table: str,
    column: str,
    table_columns: dict[str, dict[str, dict[str, Any]]],
) -> str | None:
    table_key = table.split(".")[-1].lower() if table else ""
    if table_key in table_columns:
        return table_key
    matches = [
        known_table
        for known_table, columns in table_columns.items()
        if column.lower() in columns
    ]
    return matches[0] if len(matches) == 1 else None


def _catalog_evidence_for_planning(
    evidence_store: DbEvidenceStore,
) -> tuple[Evidence, ...]:
    return tuple(
        item
        for item in evidence_store.list()
        if item.kind.startswith("catalog.")
        or item.kind
        in {
            SCHEMA_SEARCH_RESULT_EVIDENCE,
            "schema.column_value_profile",
            "schema.column_value_search_result",
            "schema.column_value_hint",
            "catalog.source",
        }
    )


def _catalog_column_value_search_exists(evidence_store: DbEvidenceStore) -> bool:
    return any(
        item.kind == "schema.column_value_search_result"
        for item in evidence_store.list()
    )


def _plan_has_literal_predicates(
    plan_evidence: Evidence,
    schema: dict[str, Any],
) -> bool:
    sql = str(
        plan_evidence.payload.get("sql")
        or (plan_evidence.payload.get("structured_plan") or {}).get("selected_sql")
        or ""
    )
    if not sql:
        return False
    try:
        from .sql_analysis import analyze_sql

        analysis = analyze_sql(sql, dialect=str(schema.get("database_type") or ""))
    except Exception:
        return False
    return bool(_zero_row_literal_predicates(analysis.literal_predicates))


def _query_result_evidence(evidence: tuple[Evidence, ...]) -> Evidence | None:
    return next((item for item in evidence if item.kind == "query.result"), None)


def _query_result_is_zero_rows(evidence: Evidence) -> bool:
    rows = evidence.payload.get("rows")
    return isinstance(rows, list) and len(rows) == 0


def _zero_row_literal_predicates(
    literal_predicates: tuple[Any, ...],
) -> list[dict[str, Any]]:
    predicates: list[dict[str, Any]] = []
    for predicate in literal_predicates:
        operator = str(getattr(predicate, "operator", "") or "").lower()
        if operator not in {"=", "in", "like"}:
            continue
        column_ref = getattr(predicate, "column", None)
        column = str(getattr(column_ref, "name", "") or "")
        if not column:
            continue
        values = [
            str(value)
            for value in getattr(predicate, "values", ()) or ()
            if value is not None
        ]
        if not values:
            continue
        predicates.append(
            {
                "table": _short_table_name(str(getattr(column_ref, "table", "") or "")),
                "column": column,
                "operator": operator,
                "values": values,
            }
        )
    return predicates


def _column_value_hints_for_predicates(
    planning_context: dict[str, Any],
    predicates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted = [
        (
            _short_table_name(str(predicate.get("table") or "")),
            str(predicate.get("column") or "").lower(),
        )
        for predicate in predicates
        if predicate.get("column")
    ]
    hint_candidates = [
        hint
        for hint in planning_context.get("column_value_hints", []) or []
        if isinstance(hint, dict)
    ]
    hints: list[dict[str, Any]] = []
    for hint in hint_candidates:
        key = (
            _short_table_name(str(hint.get("table") or "")),
            str(hint.get("column") or "").lower(),
        )
        if not _hint_key_matches_predicates(key, wanted, hint_candidates):
            continue
        observed = [
            item
            for item in hint.get("observed_values", []) or []
            if isinstance(item, dict) and item.get("value") is not None
        ]
        if not observed:
            continue
        hints.append(
            {
                "table": hint.get("table"),
                "column": hint.get("column"),
                "profile_ref": hint.get("profile_ref"),
                "observed_values": observed[:25],
                **(
                    {"candidate_mapping": dict(hint["candidate_mapping"])}
                    if isinstance(hint.get("candidate_mapping"), dict)
                    else {}
                ),
            }
        )
    return hints


def _hint_key_matches_predicates(
    key: tuple[str, str],
    wanted: list[tuple[str, str]],
    hints: list[dict[str, Any]],
) -> bool:
    for wanted_table, wanted_column in wanted:
        if wanted_column != key[1]:
            continue
        if wanted_table:
            if wanted_table == key[0]:
                return True
            continue
        matching_tables = {
            _short_table_name(str(hint.get("table") or ""))
            for hint in hints
            if str(hint.get("column") or "").lower() == wanted_column
        }
        if len(matching_tables) == 1:
            return True
    return False


def _short_table_name(table: str) -> str:
    return table.split(".")[-1].lower() if table else ""


def _profile_candidate_shape(column: dict[str, Any]) -> bool:
    if column.get("is_primary_key"):
        return False
    data_type = str(column.get("type") or column.get("data_type") or "").lower()
    return any(
        token in data_type
        for token in (
            "char",
            "text",
            "enum",
            "bool",
            "int",
        )
    )


def _runtime_from_db_option(runtime: Any | None, key: str) -> Any | None:
    if runtime is None:
        return None
    options = getattr(getattr(runtime, "config", None), "metadata", {}).get(
        "from_db_options"
    )
    if isinstance(options, dict):
        return options.get(key)
    return None


def _catalog_profile_ttl_input(runtime: Any | None) -> dict[str, Any]:
    if runtime is None:
        return {}
    value = _runtime_from_db_option(runtime, "cache_ttl")
    if value is None:
        return {}
    return {"max_profile_age_seconds": value}


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


def _deterministic_value_hint_search_needed(
    request: DbRequest, schema: dict[str, Any]
) -> bool:
    if request.metadata.get("sql") or request.metadata.get("query"):
        return False
    if request.constraints.get("sql") or request.constraints.get("query"):
        return False
    return DbQueryPlanner.needs_value_hint_context(request.prompt, schema)


def _stable_hash(value: Any) -> str:
    import hashlib
    import json

    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
