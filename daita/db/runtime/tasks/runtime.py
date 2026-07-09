"""Task execution helpers for ``DbRuntime``."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from daita.runtime import (
    ApprovalStatus,
    Capability,
    Evidence,
    Operation,
    Task,
    TaskDependency,
)

from ...models import DbIntent, DbIntentKind
from .catalog import DbRuntimeTaskCatalogMixin
from .common import _payload_fingerprint, _stable_hash
from .evidence import DbRuntimeTaskEvidenceMixin
from .execution import DbRuntimeTaskExecutionMixin
from .inputs import DbRuntimeTaskInputMixin
from .planning import DbRuntimeTaskPlanningMixin
from .readiness import DbRuntimeTaskReadinessMixin


class DbRuntimeTasksMixin(
    DbRuntimeTaskExecutionMixin,
    DbRuntimeTaskPlanningMixin,
    DbRuntimeTaskReadinessMixin,
    DbRuntimeTaskInputMixin,
    DbRuntimeTaskCatalogMixin,
    DbRuntimeTaskEvidenceMixin,
):
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
