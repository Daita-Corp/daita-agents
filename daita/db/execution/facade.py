"""Public DB operation executor facade."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

from daita.runtime import Evidence, Operation, Task

from ..capabilities import SCHEMA_RELATIONSHIP_PATH_EVIDENCE
from ..evidence import DbEvidenceStore
from ..models import DbIntent, DbIntentKind, DbOperationContract, DbRequest
from ..query_planning import DbQueryPlanner
from ..session_context import db_session_context_from_request
from .catalog import (
    _ExecutionCatalogMixin,
    _catalog_column_value_search_exists,
    _catalog_evidence_for_planning,
)
from .helpers import _lineage_entity_for_request, _store_id_for_request
from .planning import (
    _ExecutionPlanningMixin,
    _accepted_sql,
    _deterministic_value_hint_search_needed,
    _planner_route,
)
from .repair import _ExecutionRepairMixin
from .tasks import _ExecutionTaskMixin, _selected_capability
from .types import DbExecutionOutcome
from .value_grounding import _ExecutionValueGroundingMixin, _plan_has_literal_predicates


class DbOperationExecutor(
    _ExecutionPlanningMixin,
    _ExecutionRepairMixin,
    _ExecutionCatalogMixin,
    _ExecutionValueGroundingMixin,
    _ExecutionTaskMixin,
):
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
        elif intent.kind is DbIntentKind.SCHEMA_RELATIONSHIP_QUERY:
            await self._execute_relationship_schema_steps(
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
            repair_attempted = False
            if (
                not validation.payload.get("valid")
                and self.runtime.db_llm_service.available
            ):
                repair_attempted = True
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
            if (
                repair_attempted
                and not _accepted_sql(validation)
                and planning_context is not None
            ):
                fallback = await self._try_deterministic_repair_fallback(
                    request,
                    intent,
                    operation,
                    schema,
                    relationship_payload,
                    tasks,
                    evidence_store,
                    planning_context=planning_context,
                    prior_plan=plan_evidence,
                    failure=validation,
                    diagnostics=diagnostics,
                )
                if fallback is not None:
                    (
                        plan_evidence,
                        validation,
                        fallback_warnings,
                        fallback_diagnostics,
                    ) = fallback
                    warnings.extend(fallback_warnings)
                    diagnostics["query_plan"] = fallback_diagnostics
                    diagnostics["planner_strategy"] = fallback_diagnostics.get(
                        "strategy"
                    )

            sql = _accepted_sql(validation)
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
                            repaired_sql = _accepted_sql(validation)
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
                            if planning_context is not None:
                                fallback = (
                                    await self._try_deterministic_repair_fallback(
                                        request,
                                        intent,
                                        operation,
                                        schema,
                                        relationship_payload,
                                        tasks,
                                        evidence_store,
                                        planning_context=planning_context,
                                        prior_plan=repaired,
                                        failure=validation,
                                        diagnostics=diagnostics,
                                    )
                                )
                                if fallback is not None:
                                    (
                                        _fallback_plan,
                                        fallback_validation,
                                        fallback_warnings,
                                        fallback_diagnostics,
                                    ) = fallback
                                    warnings.extend(fallback_warnings)
                                    fallback_sql = _accepted_sql(fallback_validation)
                                    sql_validation = await self._execute_sql_validation(
                                        contract,
                                        operation,
                                        tasks,
                                        evidence_store,
                                        fallback_sql,
                                        plan_validation=fallback_validation,
                                    )
                                    await self._execute_validated_read(
                                        contract,
                                        operation,
                                        tasks,
                                        evidence_store,
                                        sql_validation,
                                    )
                                    diagnostics["planned_sql"] = fallback_sql
                                    diagnostics["query_plan"] = fallback_diagnostics
                                    diagnostics["planner_strategy"] = (
                                        fallback_diagnostics.get("strategy")
                                    )
                                    return DbExecutionOutcome(
                                        evidence=evidence_store.list(),
                                        tasks=tuple(tasks),
                                        diagnostics={
                                            **diagnostics,
                                            "evidence_kinds": [
                                                item.kind
                                                for item in evidence_store.list()
                                            ],
                                            "evidence_refs": list(
                                                evidence_store.refs()
                                            ),
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
        tables = _schema_tables_for_request(request, schema, self.query_planner)
        await self._execute_capability(
            "catalog.schema.search",
            contract,
            operation,
            tasks,
            evidence_store,
            {"store_id": store_id, "query": request.prompt, "limit": 10},
        )
        for table in tables:
            asset_evidence = await self._execute_capability(
                "catalog.asset.inspect",
                contract,
                operation,
                tasks,
                evidence_store,
                {"store_id": store_id, "asset_ref": table, "limit": 100},
            )
            for item in asset_evidence:
                scoped = _with_schema_asset_scope(item, "asset")
                if scoped is item:
                    continue
                evidence_store.discard(item.id)
                evidence_store.add(scoped)

    async def _execute_relationship_schema_steps(
        self,
        request: DbRequest,
        contract: DbOperationContract,
        operation: Operation,
        schema: dict[str, Any],
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        store_id: str,
    ) -> None:
        await self._execute_capability(
            "catalog.schema.search",
            contract,
            operation,
            tasks,
            evidence_store,
            {"store_id": store_id, "query": request.prompt, "limit": 10},
        )
        from_assets, to_assets = _relationship_assets_for_metadata_prompt(
            request.prompt,
            schema,
            self.query_planner.relationship_tables_for_prompt(request.prompt, schema),
            self.query_planner.best_table_for_prompt(request.prompt, schema),
        )
        if not from_assets or not to_assets:
            return
        await self._execute_capability(
            "catalog.relationship_paths.find",
            contract,
            operation,
            tasks,
            evidence_store,
            {
                "store_id": store_id,
                "from_assets": from_assets,
                "to_assets": to_assets,
                "relationship_types": ["foreign_key", "references"],
                "max_hops": 4,
                "max_paths": 5,
            },
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


def _relationship_assets_for_metadata_prompt(
    prompt: str,
    schema: dict[str, Any],
    relationship_tables: tuple[str | None, str | None],
    best_table: str | None,
) -> tuple[list[str], list[str]]:
    from_table, to_table = relationship_tables
    if from_table and to_table:
        return [from_table], [to_table]
    source = from_table or to_table or best_table
    if not source:
        return [], []
    related = _related_tables_for_schema_asset(schema, source)
    if related:
        return [source], related
    all_tables = [
        str(table.get("name"))
        for table in schema.get("tables", []) or []
        if table.get("name") and str(table.get("name")) != source
    ]
    return [source], all_tables[:5]


def _schema_tables_for_request(
    request: DbRequest,
    schema: dict[str, Any],
    query_planner: DbQueryPlanner,
) -> tuple[str, ...]:
    schema_tables = _schema_table_names(schema)
    if not schema_tables:
        return ()
    if request.source_scope:
        scoped_tables = []
        for table in request.source_scope:
            if table in schema_tables:
                scoped_tables.append(table)
                continue
            short_name = table.split(".")[-1]
            if short_name in schema_tables:
                scoped_tables.append(short_name)
        return tuple(dict.fromkeys(scoped_tables))
    explicit_table = query_planner.best_table_for_prompt(request.prompt, schema)
    if explicit_table:
        return (explicit_table,)
    return _session_tables_for_request(request, schema_tables)


def _session_tables_for_request(
    request: DbRequest,
    schema_tables: set[str],
) -> tuple[str, ...]:
    session_context = db_session_context_from_request(request)
    if session_context is None:
        return ()
    referent_sources = (
        session_context.diagnostics.get("referent_sources", {}).get("tables", {})
        if isinstance(session_context.diagnostics.get("referent_sources"), dict)
        else {}
    )
    structured_referents = tuple(
        table
        for table in session_context.referents.tables
        if referent_sources.get(table) != "conversation_history"
    )
    candidate_referents = structured_referents or session_context.referents.tables
    tables = []
    for table in candidate_referents:
        if table in schema_tables:
            tables.append(table)
            continue
        short_name = table.split(".")[-1]
        if short_name in schema_tables:
            tables.append(short_name)
    return tuple(dict.fromkeys(tables))


def _schema_table_names(schema: dict[str, Any]) -> set[str]:
    return {
        str(table.get("name"))
        for table in schema.get("tables", []) or []
        if table.get("name")
    }


def _related_tables_for_schema_asset(schema: dict[str, Any], asset: str) -> list[str]:
    related: list[str] = []
    for relationship in schema.get("foreign_keys", []) or []:
        source = relationship.get("source_table") or relationship.get("source_asset")
        target = relationship.get("target_table") or relationship.get("target_asset")
        if source == asset and target:
            related.append(str(target))
        elif target == asset and source:
            related.append(str(source))
    return list(dict.fromkeys(related))


def _with_schema_asset_scope(evidence: Evidence, scope: str) -> Evidence:
    if evidence.kind != "schema.asset_profile":
        return evidence
    payload = dict(evidence.payload)
    metadata = {**evidence.metadata, "scope": scope}
    payload_metadata = dict(payload.get("metadata") or {})
    payload_metadata.setdefault("scope", scope)
    payload["metadata"] = payload_metadata
    return replace(evidence, payload=payload, metadata=metadata)
