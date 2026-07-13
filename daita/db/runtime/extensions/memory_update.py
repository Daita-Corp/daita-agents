"""Runtime-owned executors for explicit DB memory commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from daita.runtime import Evidence, Operation, Task, TaskDependency, TaskStatus

from ...analysis import structural_schema_fingerprint
from ...evidence import load_evidence
from ...fingerprints import persisted_fingerprint
from ...memory.commands import DbMemoryCommandService
from ...memory.config import db_memory_options_from_runtime_metadata
from ...memory.storage import db_memory_record_ids_by_key
from ...models import DbRequest
from ..tasks.models import DbTaskSpec

_COMPLETED_TASK_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.SKIPPED,
}


@dataclass(frozen=True)
class DbMemoryPlanUpdateExecutor:
    """Executor that persists accepted or rejected DB memory proposal evidence."""

    plugin: Any
    id: str = "db_runtime.memory.plan_update"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.memory.plan_update"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        request = _request_from_task(runtime, operation, task)
        schema = task.input.get("schema")
        schema = schema if isinstance(schema, dict) else {}
        memory_options = db_memory_options_from_runtime_metadata(
            runtime.config.metadata
        )
        source_identity = str(memory_options.get("source_identity") or "").strip()
        schema_fingerprint = task.input.get("schema_fingerprint")
        if not schema_fingerprint and schema:
            schema_fingerprint = structural_schema_fingerprint(schema)

        proposal, validation = DbMemoryCommandService().plan_update(
            request,
            source_identity=source_identity,
            workspace_scope=str(memory_options.get("workspace_scope") or "source"),
            schema=schema,
            schema_fingerprint=(
                str(schema_fingerprint).strip() if schema_fingerprint else None
            ),
        )
        if _first_capability(runtime, "memory.semantic.write", owner="memory") is None:
            validation_payload = dict(proposal.get("validation") or {})
            reasons = list(validation_payload.get("reasons") or [])
            reasons.append("memory_write_capability_missing")
            validation_payload.update(
                {
                    "accepted": False,
                    "status": "rejected",
                    "reasons": list(dict.fromkeys(reasons)),
                }
            )
            proposal["validation"] = validation_payload
            proposal["proposal_fingerprint"] = persisted_fingerprint(
                {
                    key: value
                    for key, value in proposal.items()
                    if key != "proposal_fingerprint"
                }
            )
            validation_accepted = False
        else:
            validation_accepted = validation.accepted
        if validation_accepted:
            validation_accepted = await _annotate_duplicate_behavior(runtime, proposal)
        return [
            Evidence(
                kind="db.memory.proposal",
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=validation_accepted,
                payload=proposal,
                metadata={
                    "payload_fingerprint": proposal["proposal_fingerprint"],
                    "proposal_fingerprint": proposal["proposal_fingerprint"],
                    "validation_accepted": validation_accepted,
                    "source_identity": source_identity,
                },
            )
        ]


@dataclass(frozen=True)
class DbMemoryCommitUpdateExecutor:
    """Executor that commits an accepted DB memory proposal through memory tasks."""

    plugin: Any
    id: str = "db_runtime.memory.commit_update"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.memory.commit_update"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        proposal_evidence = await load_evidence(
            runtime,
            operation.id,
            task.input.get("proposal_evidence_id"),
        )
        if proposal_evidence is None:
            raise RuntimeError("DB memory proposal evidence is required")
        if not proposal_evidence.accepted:
            raise RuntimeError("DB memory proposal evidence was not accepted")

        proposal = dict(proposal_evidence.payload)
        expected_fingerprint = task.input.get("proposal_fingerprint")
        actual_fingerprint = str(proposal.get("proposal_fingerprint") or "")
        if not actual_fingerprint:
            actual_fingerprint = persisted_fingerprint(proposal)
        recomputed_fingerprint = persisted_fingerprint(
            {
                key: value
                for key, value in proposal.items()
                if key != "proposal_fingerprint"
            }
        )
        if actual_fingerprint != recomputed_fingerprint:
            raise RuntimeError("DB memory proposal fingerprint mismatch")
        if expected_fingerprint and expected_fingerprint != actual_fingerprint:
            raise RuntimeError("DB memory proposal fingerprint mismatch")

        runtime_source_identity = _runtime_source_identity(runtime)
        proposal_source_identity = str(proposal.get("source_identity") or "").strip()
        input_source_identity = str(task.input.get("source_identity") or "").strip()
        if (
            not proposal_source_identity
            or proposal_source_identity != runtime_source_identity
            or (
                input_source_identity
                and input_source_identity != proposal_source_identity
            )
        ):
            raise RuntimeError("DB memory proposal source identity mismatch")

        record = dict(proposal.get("record") or {})
        memory_capability = runtime.registry.get_capability(
            "memory.semantic.write",
            owner="memory",
        )
        write_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id=memory_capability.id,
                    owner=memory_capability.owner,
                    input={
                        "db_memory_payload": record,
                        "db_memory_prompt": str(record.get("text") or ""),
                    },
                    reason="db_memory_commit_update",
                    sequence=1,
                    dependencies=(
                        TaskDependency(
                            kind="evidence",
                            evidence_kind="db.memory.proposal",
                            evidence_id=proposal_evidence.id,
                            evidence_owner="db_runtime",
                            producer_task_id=proposal_evidence.task_id,
                            evidence_payload={
                                "proposal_fingerprint": actual_fingerprint,
                            },
                            evidence_accepted=True,
                            operation_id=operation.id,
                        ),
                    ),
                    metadata={
                        "proposal_evidence_id": proposal_evidence.id,
                        "proposal_fingerprint": actual_fingerprint,
                        "source_identity": proposal_source_identity,
                    },
                    deterministic_key=actual_fingerprint,
                ),
            ),
        )
        write_task = write_plan.tasks[0]
        write_evidence_reused = write_task.status in _COMPLETED_TASK_STATUSES
        if write_evidence_reused:
            write_evidence = await _task_evidence(
                runtime,
                operation.id,
                write_task.id,
                evidence_kind="memory.semantic.write",
            )
        else:
            write_evidence = await runtime.execute_task(
                write_task,
                operation,
                context={"capability_owner": memory_capability.owner},
            )
        write_success = bool(write_evidence) and all(
            item.accepted and item.payload.get("success") is not False
            for item in write_evidence
        )
        definition_payload = {
            "action": proposal.get("action"),
            "record": record,
            "proposal_evidence_id": proposal_evidence.id,
            "proposal_fingerprint": actual_fingerprint,
            "write_evidence_ids": [item.id for item in write_evidence],
            "source_identity": proposal_source_identity,
            "committed": write_success,
        }
        definition = Evidence(
            kind="db.memory.definition",
            owner="db_runtime",
            operation_id=operation.id,
            task_id=task.id,
            accepted=write_success,
            payload=definition_payload,
            metadata={
                "payload_fingerprint": persisted_fingerprint(definition_payload),
                "proposal_evidence_id": proposal_evidence.id,
                "proposal_fingerprint": actual_fingerprint,
                "source_identity": proposal_source_identity,
            },
        )
        if write_evidence_reused:
            return [definition]
        return [definition, *write_evidence]


def _request_from_task(runtime: Any, operation: Operation, task: Task) -> DbRequest:
    payload = task.input.get("request")
    if isinstance(payload, dict):
        return DbRequest(
            prompt=str(payload.get("prompt") or ""),
            mode=payload.get("mode"),
            metadata=dict(payload.get("metadata") or {}),
            constraints=dict(payload.get("constraints") or {}),
            source_scope=tuple(payload.get("source_scope") or ()),
            requested_capabilities=tuple(payload.get("requested_capabilities") or ()),
        )
    return runtime._db_request_from_operation(operation)


async def _task_evidence(
    runtime: Any,
    operation_id: str,
    task_id: str,
    *,
    evidence_kind: str,
) -> tuple[Evidence, ...]:
    return tuple(
        evidence
        for evidence in await runtime.store.list_evidence(operation_id)
        if evidence.task_id == task_id and evidence.kind == evidence_kind
    )


def _first_capability(runtime: Any, capability_id: str, *, owner: str | None = None):
    for capability in runtime.registry.capabilities:
        if capability.id != capability_id:
            continue
        if owner is not None and capability.owner != owner:
            continue
        return capability
    return None


async def _annotate_duplicate_behavior(runtime: Any, proposal: dict[str, Any]) -> bool:
    try:
        memory_plugin = runtime.registry.get_plugin("memory")
        existing_record_ids = await db_memory_record_ids_by_key(
            memory_plugin,
            dict(proposal.get("record") or {}),
        )
    except Exception as exc:
        validation_payload = dict(proposal.get("validation") or {})
        diagnostics = dict(validation_payload.get("diagnostics") or {})
        diagnostics["duplicate_check_error"] = str(exc)
        reasons = list(validation_payload.get("reasons") or [])
        reasons.append("duplicate_check_failed")
        validation_payload["accepted"] = False
        validation_payload["status"] = "rejected"
        validation_payload["reasons"] = list(dict.fromkeys(reasons))
        validation_payload["diagnostics"] = diagnostics
        proposal["validation"] = validation_payload
        proposal["commit_behavior"] = "unknown"
        accepted = False
    else:
        validation_payload = dict(proposal.get("validation") or {})
        diagnostics = dict(validation_payload.get("diagnostics") or {})
        diagnostics["existing_chunk_ids"] = list(existing_record_ids)
        diagnostics["commit_behavior"] = "update" if existing_record_ids else "create"
        validation_payload["diagnostics"] = diagnostics
        proposal["validation"] = validation_payload
        proposal["existing_chunk_ids"] = list(existing_record_ids)
        proposal["commit_behavior"] = "update" if existing_record_ids else "create"
        accepted = True
    proposal["proposal_fingerprint"] = persisted_fingerprint(
        {key: value for key, value in proposal.items() if key != "proposal_fingerprint"}
    )
    return accepted


def _runtime_source_identity(runtime: Any) -> str:
    memory_options = db_memory_options_from_runtime_metadata(runtime.config.metadata)
    return str(memory_options.get("source_identity") or "").strip()
