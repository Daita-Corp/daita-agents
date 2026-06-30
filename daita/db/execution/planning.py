"""Query planning and validated-read sequencing helpers."""

from __future__ import annotations

from typing import Any

from daita.runtime import Evidence, Operation, Task, TaskDependency

from ..capabilities import (
    MEMORY_SEMANTIC_RECALL_CAPABILITY,
    PLANNING_CONTEXT_EVIDENCE,
    QUERY_PLAN_PROPOSAL_EVIDENCE,
    QUERY_PLAN_VALIDATION_EVIDENCE,
    SQL_VALIDATION_EVIDENCE,
)
from ..evidence import DbEvidenceStore
from ..memory import (
    db_memory_options_from_runtime_metadata,
    db_memory_planning_recall_decision,
)
from ..models import DbIntent, DbOperationContract, DbRequest
from ..query_sql_validation import sql_fingerprint


class _ExecutionPlanningMixin:
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
        intent = self.runtime._db_intent_from_operation(operation)
        memory_evidence, memory_diagnostics = await self._recall_db_memory_for_planning(
            request,
            intent,
            operation,
            tasks,
            evidence_store,
            schema=schema_evidence.payload if schema_evidence is not None else {},
            catalog_evidence=catalog_evidence,
            relationship_evidence=relationship_evidence,
            analysis_metadata=analysis_metadata,
        )
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
                "memory_recall_evidence_ids": [
                    item.id for item in memory_evidence if item.id is not None
                ],
                "memory_recall_diagnostics": memory_diagnostics,
            },
            reason="planning_context",
            metadata=analysis_metadata,
            dependencies=(
                *(
                    TaskDependency(
                        kind="evidence",
                        evidence_kind=item.kind,
                        evidence_id=item.id,
                        evidence_accepted=item.accepted,
                        operation_id=operation.id,
                    )
                    for item in memory_evidence
                    if item.id is not None
                ),
                *extra_dependencies,
            ),
        )
        if not evidence:
            raise RuntimeError("planning.context evidence was not produced")
        return evidence[0]

    async def _recall_db_memory_for_planning(
        self,
        request: DbRequest,
        intent: DbIntent,
        operation: Operation,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        *,
        schema: dict[str, Any],
        catalog_evidence: tuple[Evidence, ...] = (),
        relationship_evidence: tuple[Evidence, ...] = (),
        analysis_metadata: dict[str, Any] | None = None,
    ) -> tuple[tuple[Evidence, ...], dict[str, Any]]:
        options = db_memory_options_from_runtime_metadata(self.runtime.config.metadata)
        matched_schema_terms = _matched_schema_terms_from_evidence(
            (*catalog_evidence, *relationship_evidence)
        )
        decision = db_memory_planning_recall_decision(
            prompt=request.prompt,
            intent_kind=intent.kind.value,
            schema=schema,
            memory_config=options,
            matched_schema_terms=matched_schema_terms or None,
        )
        diagnostics = {
            "registered": _memory_capability_available(self.runtime),
            "queried": False,
            "decision": decision,
        }
        if not decision.get("recall"):
            return (), diagnostics
        try:
            capability = self.runtime.registry.get_capability(
                MEMORY_SEMANTIC_RECALL_CAPABILITY,
                owner="memory",
            )
        except KeyError:
            diagnostics["decision"] = {
                "recall": False,
                "reason": "memory_not_registered",
            }
            return (), diagnostics
        try:
            evidence = await self._execute_direct_capability(
                capability,
                operation,
                tasks,
                evidence_store,
                {
                    "query": str(decision.get("query") or request.prompt),
                    "category": "db_semantics",
                    "limit": int(options.get("limit") or 3) * 3,
                    "score_threshold": _float_option(
                        options,
                        "score_threshold",
                        0.45,
                    ),
                    "retrieval_mode": options.get("retrieval_mode", "structured"),
                    "source_identity": options.get("source_identity"),
                },
                reason="planning_memory_recall",
                metadata={**dict(analysis_metadata or {}), "memory_recall": "planning"},
            )
        except Exception as exc:
            diagnostics["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            return (), diagnostics
        diagnostics["queried"] = True
        diagnostics["evidence_count"] = len(evidence)
        return evidence, diagnostics

    async def _plan_query(
        self,
        request: DbRequest,
        operation: Operation,
        planning_context: Evidence,
        tasks: list[Task],
        evidence_store: DbEvidenceStore,
        analysis_metadata: dict[str, Any] | None = None,
        extra_dependencies: tuple[TaskDependency, ...] = (),
    ) -> tuple[Evidence, tuple[str, ...], dict[str, Any]]:
        if not self.runtime.db_llm_service.available:
            raise RuntimeError("DB LLM service is required for query planning")
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
        return evidence[-1], (), {"strategy": "llm"}

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

def _accepted_sql(validation: Evidence) -> str | None:
    if not validation.payload.get("valid"):
        return None
    sql = validation.payload.get("accepted_sql")
    return str(sql) if sql else None


def _memory_capability_available(runtime: Any) -> bool:
    try:
        runtime.registry.get_capability(
            MEMORY_SEMANTIC_RECALL_CAPABILITY,
            owner="memory",
        )
        return True
    except KeyError:
        return False


def _matched_schema_terms_from_evidence(
    evidence: tuple[Evidence, ...],
) -> tuple[str, ...]:
    terms: list[str] = []
    for item in evidence:
        if not item.accepted:
            continue
        payload = item.payload if isinstance(item.payload, dict) else {}
        if item.kind == "schema.search_result":
            for table in payload.get("tables", []) or []:
                if not isinstance(table, dict):
                    continue
                table_name = str(table.get("name") or "").strip()
                if table_name and (
                    table.get("matched_columns") or table.get("match_reasons")
                ):
                    terms.append(table_name)
                for column in table.get("matched_columns", []) or []:
                    if isinstance(column, dict) and column.get("name") and table_name:
                        terms.append(f"{table_name}.{column['name']}")
        elif item.kind == "schema.asset_profile":
            for table in payload.get("tables", []) or []:
                if isinstance(table, dict) and table.get("name"):
                    terms.append(str(table["name"]))
            asset_ref = payload.get("asset_ref") or payload.get("name")
            if asset_ref:
                terms.append(str(asset_ref))
        elif item.kind == "schema.relationship_path":
            for key in ("from_tables", "to_tables", "from_assets", "to_assets"):
                for value in payload.get(key, []) or []:
                    if value:
                        terms.append(str(value))
            for path in payload.get("paths", []) or []:
                if not isinstance(path, dict):
                    continue
                for key in ("from_asset", "to_asset", "source", "target"):
                    if path.get(key):
                        terms.append(str(path[key]))
    return tuple(dict.fromkeys(term for term in terms if term.strip()))


def _float_option(options: dict[str, Any], key: str, default: float) -> float:
    value = options.get(key)
    if value is None:
        return default
    return float(value)
