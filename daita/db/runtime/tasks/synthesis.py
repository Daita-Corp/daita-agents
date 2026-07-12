"""Answer synthesis task orchestration for ``DbRuntime``."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from daita.runtime import Evidence, Operation, Task, TaskDependency

from ...fingerprints import persisted_fingerprint
from ...models import DbIntent, DbIntentKind
from .context import DbTaskContext
from .dependencies import _dependency_for_evidence
from .evidence import latest_accepted_evidence
from .execution import execute_task


async def execute_answer_synthesis(
    context: DbTaskContext,
    *,
    operation: Operation,
    intent: DbIntent,
    outcome_evidence: tuple[Evidence, ...],
) -> tuple[Evidence, Task]:
    existing = await latest_accepted_evidence(
        context,
        operation.id,
        "answer.synthesis",
    )
    if existing is not None:
        task = next(
            (
                item
                for item in await context.store.list_tasks(operation.id)
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

    capability = context.registry.get_capability(
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
            context.config.metadata, "synthesis_row_budget", 25
        ),
        "char_budget": _synthesis_context_option(
            context.config.metadata, "synthesis_context_char_budget", 16000
        ),
    }
    input_hash = persisted_fingerprint(task_input)
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
            "idempotency_key": persisted_fingerprint(
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
    evidence = await execute_task(
        context,
        task,
        operation,
        execution_context={"capability_owner": capability.owner},
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
    stored_task = await context.store.load_task(task.id)
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
