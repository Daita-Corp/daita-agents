"""Unregistered revision-1 to draft revision-2 conversion harness.

This module is deliberately absent from the migration registry and package exports.
It exists so Phase 2 can prove the complete conversion before revision 2 is frozen
or published.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..autonomy import AutonomousFollowup, FollowupDisposition
from ..jobs.graph.models import (
    AttemptState,
    BudgetAmount,
    BudgetLimit,
    ControlKind,
    ControlState,
    EdgeKind,
    GraphAdmission,
    GraphAuthority,
    GraphDesiredState,
    GraphJob,
    GraphJobSpecification,
    GraphState,
    GraphTask,
    GraphTaskSpecification,
    JobGraph,
    TaskAttempt,
    TaskControl,
    TaskDependency,
    TaskExecutionKind,
    TaskResult,
    TaskRole,
    TaskState,
    canonical_digest,
    topology_digest,
)
from ..jobs.graph.validation import require_authority_subset, validate_graph_topology
from ..jobs.models import (
    JobAttempt,
    JobAttemptStatus,
    JobDesiredState,
    JobExecutionMode,
    JobRun,
    JobStatus,
)
from .draft_graph_codecs import (
    decode_draft_delivery,
    decode_graph_mutation,
    decode_task_checkpoint,
    decode_task_comment,
    decode_task_control,
    encode_graph_job,
    encode_graph_task,
    encode_job_graph,
    encode_task_attempt,
    encode_task_control,
    encode_task_result,
)
from .draft_graph_schema import (
    connect_draft_graph,
    create_draft_graph_database,
    require_draft_graph_schema,
)
from .sqlite import validate_current_state_database
from .sqlite_codecs.autonomy import decode_autonomous_followup
from .sqlite_codecs.distribution import decode_delivery
from .sqlite_codecs.jobs import decode_job_run
from .sqlite_graph import admit_graph, datetime_to_us, inspect_graph

DraftPhaseHook = Callable[[str], None]
_DIGEST_ZERO = "sha256:" + "0" * 64


@dataclass(frozen=True, slots=True)
class ExecutabilityLedgerEntry:
    source_kind: str
    source_id: str
    decoded_state: str
    evidence_classification: str
    target_job_id: str
    target_graph_state: GraphState
    runnable: bool
    executor_kind: str | None
    target_task_id: str | None
    reason: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "decoded_state": self.decoded_state,
            "evidence_classification": self.evidence_classification,
            "target_job_id": self.target_job_id,
            "target_graph_state": self.target_graph_state.value,
            "runnable": self.runnable,
            "executor_kind": self.executor_kind,
            "target_task_id": self.target_task_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DraftConversionReport:
    source_revision: int
    target_revision: int
    entries: tuple[ExecutabilityLedgerEntry, ...]
    source_database_sha256: str
    target_database_sha256: str

    @property
    def ledger_digest(self) -> str:
        return canonical_digest(
            {
                "source_revision": self.source_revision,
                "target_revision": self.target_revision,
                "entries": [entry.to_mapping() for entry in self.entries],
            }
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _decode_revision_1_job(data: str, *, agent_id: str, job_id: str) -> JobRun:
    """Migration-owned entry point for the frozen revision-1 job shape."""

    return decode_job_run(data, agent_id=agent_id, job_id=job_id)


def _decode_revision_1_followup(
    data: str, *, agent_id: str, followup_id: str
) -> AutonomousFollowup:
    """Migration-owned entry point for the frozen revision-1 follow-up shape."""

    return decode_autonomous_followup(data, agent_id=agent_id, followup_id=followup_id)


def _copy_home_files(source_home: Path, target_home: Path) -> None:
    target_home.mkdir(parents=True, exist_ok=True)
    for source in source_home.rglob("*"):
        relative = source.relative_to(source_home)
        if relative.parts[0] in {".home-upgrade", ".home-rollbacks"}:
            continue
        if relative.as_posix() in {"state.db", "state.db-wal", "state.db-shm"}:
            continue
        target = target_home / relative
        if source.is_symlink():
            raise ValueError("draft conversion refuses symlinked home content")
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        else:
            raise ValueError("draft conversion encountered unsupported home content")


_COPY_TABLES = (
    "metadata",
    "sources",
    "syncs",
    "snapshots",
    "runs",
    "messages",
    "semantic_annotations",
    "learning_candidates",
    "agent_home_migrations",
    "source_read_scopes",
    "relational_write_scopes",
    "mcp_server_bindings",
    "deliveries",
    "scheduled_routines",
    "routine_occurrences",
)


def _copy_table(
    source: sqlite3.Connection, target: sqlite3.Connection, table: str
) -> None:
    columns = tuple(
        str(row[1]) for row in target.execute(f"PRAGMA table_info({table})")
    )
    source_columns = {
        str(row[1]) for row in source.execute(f"PRAGMA table_info({table})")
    }
    if not set(columns).issubset(source_columns):
        raise ValueError(f"revision-1 table cannot map to draft target: {table}")
    projection = ", ".join(f'"{column}"' for column in columns)
    rows = tuple(source.execute(f'SELECT {projection} FROM "{table}"'))
    placeholders = ", ".join("?" for _ in columns)
    target.executemany(
        f'INSERT INTO "{table}" ({projection}) VALUES ({placeholders})', rows
    )


def _copy_unchanged_state(
    source: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    for table in _COPY_TABLES:
        _copy_table(source, target, table)
    receipt_columns = (
        "agent_id",
        "id",
        "run_id",
        "call_id",
        "operation_key",
        "routine_id",
        "occurrence_id",
        "grant_digest",
        "unresolved",
        "data",
    )
    projection = ", ".join(receipt_columns)
    rows = tuple(source.execute(f"SELECT {projection} FROM effect_receipts"))
    target.executemany(
        """INSERT INTO effect_receipts(
               agent_id, id, run_id, call_id, operation_key, routine_id,
               occurrence_id, grant_digest, unresolved, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    delivery_rows = tuple(
        target.execute("SELECT agent_id, delivery_id, data FROM deliveries")
    )
    for agent_id, delivery_id, data in delivery_rows:
        decode_delivery(str(data), agent_id=str(agent_id), delivery_id=str(delivery_id))
        payload = json.loads(str(data))
        fields = payload.get("fields")
        if not isinstance(fields, dict):
            raise ValueError("stored revision-1 delivery is invalid")
        fields["migration_provenance"] = {}
        target.execute(
            "UPDATE deliveries SET data = ? WHERE agent_id = ? AND delivery_id = ?",
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                agent_id,
                delivery_id,
            ),
        )


def _authority(job: JobRun) -> GraphAuthority:
    bindings = {
        "execution": {
            "capability_id": job.specification.execution_capability_id,
            "contract_digest": job.specification.execution_contract_digest,
        },
        "resources": [
            {
                "source_id": binding.source_id,
                "source_revision": binding.source_revision,
                "resource_id": binding.resource_id,
                "resource_revision": binding.resource_revision,
                "adapter_id": binding.adapter_id,
            }
            for binding in job.specification.resource_bindings
        ],
    }
    return GraphAuthority(
        source_ids=tuple(
            sorted({item.source_id for item in job.specification.resource_bindings})
        ),
        resource_ids=tuple(
            sorted({item.resource_id for item in job.specification.resource_bindings})
        ),
        capability_ids=(job.specification.execution_capability_id,),
        access_modes=("read",),
        operational_effects=("none",),
        sensitivity=job.specification.sensitivity,
        contract_bindings=bindings,
    )


def _attempt_has_external_evidence(attempt: JobAttempt) -> bool:
    return bool(attempt.external_intents or attempt.external_observations)


def _is_safe_profile(job: JobRun, *, now: datetime) -> bool:
    return (
        job.specification.execution_mode is JobExecutionMode.DAITA
        and job.specification.job_kind == "data_profile"
        and job.specification.deadline_at > now
        and len(job.attempts) < 3
        and not any(_attempt_has_external_evidence(item) for item in job.attempts)
    )


def _migration_provenance(job: JobRun, *, classification: str) -> dict[str, object]:
    """Project bounded revision-1 lifecycle evidence without retaining its codec."""

    return {
        "source_revision": 1,
        "source_job_revision": job.revision,
        "source_specification_digest": job.specification_digest,
        "source_specification": job.specification.digest_material(),
        "source_status": job.status.value,
        "source_desired_state": job.desired_state.value,
        "source_fencing_epoch": job.fencing_epoch,
        "cancel_requested_at": (
            None
            if job.cancel_requested_at is None
            else job.cancel_requested_at.isoformat()
        ),
        "terminal_observed_at": (
            None
            if job.terminal_observed_at is None
            else job.terminal_observed_at.isoformat()
        ),
        "completion_binding": (
            None
            if job.completion_binding is None
            else {
                "owner_kind": job.completion_binding.owner_kind.value,
                "owner_id": job.completion_binding.owner_id,
                "terminal_event_id": job.completion_binding.terminal_event_id,
                "bound_at": job.completion_binding.bound_at.isoformat(),
            }
        ),
        "attempt_evidence": [
            {
                "number": attempt.number,
                "fencing_epoch": attempt.fencing_epoch,
                "claim_token": attempt.claim_token,
                "execution_run_id": attempt.execution_run_id,
                "reserved_artifact_id": attempt.reserved_artifact_id,
                "status": attempt.status.value,
                "claimed_at": attempt.claimed_at.isoformat(),
                "lease_expires_at": attempt.lease_expires_at.isoformat(),
                "renewals": attempt.renewals,
                "completed_at": (
                    None
                    if attempt.completed_at is None
                    else attempt.completed_at.isoformat()
                ),
                "error_code": attempt.error_code,
                "external_intents": [
                    {
                        "kind": intent.kind.value,
                        "idempotency_key": intent.idempotency_key,
                        "requested_at": intent.requested_at.isoformat(),
                        "disposition": intent.disposition.value,
                        "completed_at": (
                            None
                            if intent.completed_at is None
                            else intent.completed_at.isoformat()
                        ),
                        "external_job_id": intent.external_job_id,
                        "reason_code": intent.reason_code,
                    }
                    for intent in attempt.external_intents
                ],
                "external_observations": [
                    {
                        "sequence": observation.sequence,
                        "observed_at": observation.observed_at.isoformat(),
                        "status": observation.status.value,
                        "observation_digest": observation.observation_digest,
                        "external_job_id": observation.external_job_id,
                    }
                    for observation in attempt.external_observations
                ],
            }
            for attempt in job.attempts
        ],
        "classification": classification,
    }


def _classify(
    job: JobRun,
    *,
    now: datetime,
    followups: tuple[AutonomousFollowup, ...],
) -> tuple[GraphState, bool, str, str]:
    pending_followup = any(
        item.disposition is not FollowupDisposition.COMPLETED for item in followups
    )
    if pending_followup:
        return (
            GraphState.NEEDS_ATTENTION,
            False,
            "followup_unmappable",
            "revision-1 undelivered follow-up requires attention",
        )
    if job.status is JobStatus.SUCCEEDED:
        return (
            GraphState.SUCCEEDED,
            False,
            "accepted_result",
            "terminal success preserved",
        )
    if job.status is JobStatus.FAILED:
        return (
            GraphState.FAILED,
            False,
            "terminal_failure",
            "terminal failure preserved",
        )
    if job.status is JobStatus.CANCELLED:
        return (
            GraphState.CANCELLED,
            False,
            "terminal_cancel",
            "terminal cancellation preserved",
        )
    if job.status is JobStatus.NEEDS_ATTENTION:
        return (
            GraphState.NEEDS_ATTENTION,
            False,
            "attention_evidence",
            "revision-1 attention state preserved",
        )
    if (
        job.desired_state is JobDesiredState.CANCEL
        or job.status is JobStatus.CANCEL_REQUESTED
    ):
        return GraphState.CANCELLED, False, "cancel_intent", "cancellation reconciled"
    if _is_safe_profile(job, now=now):
        return (
            GraphState.QUEUED,
            True,
            "safe_effect_free_profile",
            "exact data-profile task remains executable",
        )
    if job.specification.execution_mode is JobExecutionMode.CONNECTED_EXECUTOR:
        return (
            GraphState.NEEDS_ATTENTION,
            False,
            "connected_executor_intent",
            "revision 2 has no connected-executor branch",
        )
    return (
        GraphState.NEEDS_ATTENTION,
        False,
        "unmappable_nonterminal",
        "nonterminal revision-1 work is not deterministically executable",
    )


def _task_result(
    *,
    job: JobRun,
    task_id: str,
    attempt_id: str,
    run_id: str,
    result_id: str,
    payload: dict[str, object],
    summary: str,
    provenance: dict[str, object],
    artifact_ids: tuple[str, ...],
    completed_at: datetime,
) -> TaskResult:
    schema_digest = canonical_digest({"kind": "migrated_revision_1"})
    material = {
        "agent_id": job.agent_id,
        "job_id": job.job_id,
        "task_id": task_id,
        "result_id": result_id,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "result_kind": "migrated_revision_1",
        "schema_digest": schema_digest,
        "payload": payload,
        "summary": summary,
        "sensitivity": job.specification.sensitivity.value,
        "provenance": provenance,
        "artifact_ids": artifact_ids,
        "effect_receipt_ids": (),
        "verification": {"migration_owned": True},
        "residual_risk": None,
        "downstream_constraints": {},
        "completed_at": completed_at.isoformat(),
    }
    return TaskResult(
        agent_id=job.agent_id,
        job_id=job.job_id,
        task_id=task_id,
        result_id=result_id,
        attempt_id=attempt_id,
        run_id=run_id,
        result_kind="migrated_revision_1",
        schema_digest=schema_digest,
        payload=payload,
        summary=summary,
        sensitivity=job.specification.sensitivity,
        provenance=provenance,
        artifact_ids=artifact_ids,
        effect_receipt_ids=(),
        verification={"migration_owned": True},
        residual_risk=None,
        downstream_constraints={},
        completed_at=completed_at,
        result_digest=canonical_digest(material),
    )


def _map_attempt_state(value: JobAttemptStatus) -> AttemptState:
    return {
        JobAttemptStatus.CLAIMED: AttemptState.FENCED,
        JobAttemptStatus.SUCCEEDED: AttemptState.SUCCEEDED,
        JobAttemptStatus.FAILED: AttemptState.FAILED,
        JobAttemptStatus.CANCELLED: AttemptState.CANCELLED,
        JobAttemptStatus.NEEDS_ATTENTION: AttemptState.FAILED,
        JobAttemptStatus.FENCED: AttemptState.FENCED,
    }[value]


def _insert_attempt(
    connection: sqlite3.Connection,
    *,
    attempt: TaskAttempt,
    settled_units: int,
) -> None:
    connection.execute(
        """INSERT INTO job_task_attempts(
               agent_id, job_id, task_id, attempt_id, ordinal, fencing_epoch,
               state, claim_token, run_id, lease_expires_at_us,
               absolute_deadline_at_us, started_at_us, heartbeat_at_us,
               ended_at_us, active_slot, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL, ?)""",
        (
            attempt.agent_id,
            attempt.job_id,
            attempt.task_id,
            attempt.attempt_id,
            attempt.ordinal,
            attempt.fencing_epoch,
            attempt.state.value,
            attempt.claim_token,
            attempt.run_id,
            datetime_to_us(attempt.absolute_deadline_at),
            datetime_to_us(attempt.started_at),
            datetime_to_us(attempt.heartbeat_at),
            datetime_to_us(attempt.ended_at),
            encode_task_attempt(attempt),
        ),
    )
    connection.execute(
        """INSERT INTO job_attempt_budget_reservations(
               agent_id, job_id, task_id, attempt_id, dimension,
               reserved, settled, updated_at_us
           ) VALUES (?, ?, ?, ?, 'work_units', ?, ?, ?)""",
        (
            attempt.agent_id,
            attempt.job_id,
            attempt.task_id,
            attempt.attempt_id,
            settled_units,
            settled_units,
            datetime_to_us(attempt.ended_at),
        ),
    )


def _replace_projection_rows(
    connection: sqlite3.Connection,
    *,
    job: GraphJob,
    graph: JobGraph,
    tasks: tuple[GraphTask, ...],
) -> None:
    connection.execute(
        """UPDATE job_runs
           SET state = ?, desired_state = ?, updated_at_us = ?, terminal_at_us = ?,
               finalizer_task_id = ?, data = ?
           WHERE agent_id = ? AND job_id = ?""",
        (
            job.state.value,
            job.desired_state.value,
            datetime_to_us(job.updated_at),
            datetime_to_us(job.terminal_at),
            job.finalizer_task_id,
            encode_graph_job(job),
            job.agent_id,
            job.job_id,
        ),
    )
    connection.execute(
        """UPDATE job_graphs
           SET revision = ?, task_count = ?, edge_count = ?, mutation_count = ?,
               active_attempt_count = ?, next_ready_at_us = ?,
               finalization_attempt_id = ?, finalization_started_revision = ?,
               updated_at_us = ?, data = ?
           WHERE agent_id = ? AND job_id = ?""",
        (
            graph.revision,
            graph.task_count,
            graph.edge_count,
            graph.mutation_count,
            graph.active_attempt_count,
            datetime_to_us(graph.next_ready_at),
            graph.finalization_attempt_id,
            graph.finalization_started_revision,
            datetime_to_us(graph.updated_at),
            encode_job_graph(graph),
            graph.agent_id,
            graph.job_id,
        ),
    )
    for task in tasks:
        connection.execute(
            """UPDATE job_tasks
               SET state = ?, current_attempt_id = ?, task_revision = ?,
                   supersedes_task_id = ?, superseded_by_task_id = ?,
                   latest_result_id = ?, latest_control_id = ?,
                   latest_checkpoint_id = ?, updated_at_us = ?, terminal_at_us = ?,
                   data = ?
               WHERE agent_id = ? AND job_id = ? AND task_id = ?""",
            (
                task.state.value,
                task.current_attempt_id,
                task.task_revision,
                task.supersedes_task_id,
                task.superseded_by_task_id,
                task.latest_result_id,
                task.latest_control_id,
                task.latest_checkpoint_id,
                datetime_to_us(task.updated_at),
                datetime_to_us(task.terminal_at),
                encode_graph_task(task),
                task.agent_id,
                task.job_id,
                task.task_id,
            ),
        )


def _convert_job(
    connection: sqlite3.Connection,
    job: JobRun,
    *,
    now: datetime,
    followups: tuple[AutonomousFollowup, ...],
) -> ExecutabilityLedgerEntry:
    state, runnable, evidence, reason = _classify(job, now=now, followups=followups)
    authority = _authority(job)
    work_id = f"{job.job_id}:work"
    finalizer_id = f"{job.job_id}:finalizer"
    terminal_at = (
        job.terminal_at or job.updated_at
        if state
        in {
            GraphState.SUCCEEDED,
            GraphState.FAILED,
            GraphState.CANCELLED,
        }
        else None
    )
    effective_now = max(job.updated_at, now if runnable else job.updated_at)
    specification = GraphJobSpecification(
        principal_id=job.agent_id,
        objective=f"Migrated revision-1 {job.specification.job_kind} job",
        outcome_contract={"kind": "migrated_revision_1_job"},
        authority=authority,
        distribution_plan_digest=canonical_digest(
            {"conversation_id": job.conversation_id, "kind": "conversation_inbox"}
        ),
        budgets=(BudgetLimit("work_units", 4, 1),),
        deadline_at=job.specification.deadline_at,
        retry_policy={"max_attempts": 3},
        cancellation_policy={"preserve_evidence": True},
        finalizer_task_template={"kind": "internal_promotion"},
    )
    work_spec = GraphTaskSpecification(
        title=f"Migrated {job.specification.job_kind}",
        description="Execute the exact frozen revision-1 job specification.",
        expected_result_contract={"kind": "migrated_revision_1_result"},
        authority=authority,
        budgets=(BudgetAmount("work_units", 3),),
        max_steps=1,
        max_wall_time_seconds=min(300, int(job.specification.max_wall_time_seconds)),
        created_by="migration:revision_1",
    )
    final_spec = GraphTaskSpecification(
        title="Finalize migrated job",
        description="Promote the authenticated migrated work result.",
        expected_result_contract={"kind": "migrated_revision_1_final"},
        authority=GraphAuthority(
            sensitivity=job.specification.sensitivity,
            operational_effects=("none",),
        ),
        budgets=(BudgetAmount("work_units", 1),),
        max_steps=1,
        max_wall_time_seconds=300,
        created_by="migration:revision_1",
    )
    old_result = job.result
    work_result_id = None if old_result is None else f"{job.job_id}:work-result"
    final_result_id = (
        f"{job.job_id}:final-result" if state is GraphState.SUCCEEDED else None
    )
    attention = state is GraphState.NEEDS_ATTENTION
    if state is GraphState.SUCCEEDED:
        work_state = TaskState.SUCCEEDED
        final_state = TaskState.SUCCEEDED
    elif state is GraphState.FAILED:
        work_state = TaskState.FAILED
        final_state = TaskState.CANCELLED
    elif state is GraphState.CANCELLED:
        work_state = TaskState.CANCELLED
        final_state = TaskState.CANCELLED
    elif runnable:
        work_state = TaskState.READY
        final_state = TaskState.PENDING
    elif old_result is not None:
        work_state = TaskState.SUCCEEDED
        final_state = TaskState.BLOCKED
    else:
        work_state = TaskState.BLOCKED
        final_state = TaskState.BLOCKED
    work_terminal = (
        terminal_at
        if work_state
        in {
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
        else (
            old_result.completed_at
            if work_state is TaskState.SUCCEEDED and old_result is not None
            else None
        )
    )
    final_terminal = (
        terminal_at
        if final_state
        in {
            TaskState.SUCCEEDED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
        else None
    )
    work = GraphTask(
        agent_id=job.agent_id,
        job_id=job.job_id,
        task_id=work_id,
        state=work_state,
        role=TaskRole.INTERNAL,
        execution_kind=TaskExecutionKind.INTERNAL_CAPABILITY,
        priority=0,
        not_before=None,
        current_attempt_id=None,
        task_revision=1,
        specification=work_spec,
        task_spec_digest=work_spec.digest,
        task_scope_digest=work_spec.authority.digest,
        attempt_count=len(job.attempts),
        failure_streak=sum(
            item.status is not JobAttemptStatus.SUCCEEDED for item in job.attempts
        ),
        fencing_epoch=job.fencing_epoch
        + (1 if job.status in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED} else 0),
        created_at=job.created_at,
        updated_at=effective_now,
        terminal_at=work_terminal,
        latest_result_id=work_result_id if work_state is TaskState.SUCCEEDED else None,
        latest_control_id=(
            f"{job.job_id}:migration-attention"
            if attention and work_state is TaskState.BLOCKED
            else None
        ),
    )
    finalizer = GraphTask(
        agent_id=job.agent_id,
        job_id=job.job_id,
        task_id=finalizer_id,
        state=final_state,
        role=TaskRole.FINALIZER,
        execution_kind=TaskExecutionKind.INTERNAL_CAPABILITY,
        priority=1_000,
        not_before=None,
        current_attempt_id=None,
        task_revision=1,
        specification=final_spec,
        task_spec_digest=final_spec.digest,
        task_scope_digest=final_spec.authority.digest,
        attempt_count=1 if state is GraphState.SUCCEEDED else 0,
        failure_streak=0,
        fencing_epoch=1 if state is GraphState.SUCCEEDED else 0,
        created_at=job.created_at,
        updated_at=effective_now,
        terminal_at=final_terminal,
        latest_result_id=final_result_id,
        latest_control_id=(
            f"{job.job_id}:migration-attention"
            if attention and final_state is TaskState.BLOCKED
            else None
        ),
    )
    dependency = TaskDependency(
        agent_id=job.agent_id,
        job_id=job.job_id,
        upstream_task_id=work_id,
        downstream_task_id=finalizer_id,
        edge_kind=EdgeKind.REQUIRES_ACCEPTED_SUCCESS,
        created_at=job.created_at,
        creator_key="migration:revision_1",
    )
    graph_job = GraphJob(
        agent_id=job.agent_id,
        job_id=job.job_id,
        conversation_id=job.conversation_id,
        origin_run_id=job.origin_run_id,
        origin_call_id=job.origin_call_id,
        state=state,
        desired_state=(
            GraphDesiredState.CANCEL
            if state is GraphState.CANCELLED
            else GraphDesiredState.RUN
        ),
        created_at=job.created_at,
        updated_at=effective_now,
        deadline_at=job.specification.deadline_at,
        specification=specification,
        specification_digest=specification.digest,
        finalizer_task_id=finalizer_id,
        terminal_at=terminal_at,
        terminal_result_id=final_result_id,
        failure_code=job.failure_code,
        migration_provenance=_migration_provenance(job, classification=evidence),
    )
    graph = JobGraph(
        agent_id=job.agent_id,
        job_id=job.job_id,
        revision=0,
        task_count=2,
        edge_count=1,
        mutation_count=0,
        active_attempt_count=0,
        next_ready_at=effective_now if runnable else None,
        finalization_attempt_id=None,
        finalization_started_revision=None,
        created_at=job.created_at,
        updated_at=effective_now,
        topology_digest=topology_digest((work, finalizer), (dependency,)),
    )
    admit_graph(
        connection,
        GraphAdmission(
            job=graph_job,
            graph=graph,
            tasks=(work, finalizer),
            dependencies=(dependency,),
        ),
    )
    settled_work = 0
    for old_attempt in job.attempts:
        attempt_state = _map_attempt_state(old_attempt.status)
        ended_at = old_attempt.completed_at or effective_now
        migrated = TaskAttempt(
            agent_id=job.agent_id,
            job_id=job.job_id,
            task_id=work_id,
            attempt_id=f"{job.job_id}:attempt:{old_attempt.number}",
            ordinal=old_attempt.number,
            fencing_epoch=old_attempt.fencing_epoch,
            state=attempt_state,
            claim_token=old_attempt.claim_token,
            run_id=old_attempt.execution_run_id,
            lease_expires_at=None,
            absolute_deadline_at=job.specification.deadline_at,
            started_at=old_attempt.claimed_at,
            heartbeat_at=old_attempt.claimed_at,
            ended_at=ended_at,
            execution_scope_digest=work.task_scope_digest,
            executor_id=job.specification.execution_capability_id,
            error_code=(
                "migration_fenced"
                if old_attempt.status is JobAttemptStatus.CLAIMED
                else old_attempt.error_code
            ),
            reserved_budgets=(BudgetAmount("work_units", 1),),
            measured_usage=(BudgetAmount("work_units", 1),),
            result_id=(
                work_result_id
                if old_result is not None
                and old_attempt.number == job.attempts[-1].number
                else None
            ),
            artifact_ids=(
                tuple(sorted(ref.artifact_id for ref in old_result.artifact_refs))
                if old_result is not None
                and old_attempt.number == job.attempts[-1].number
                else ()
            ),
        )
        _insert_attempt(connection, attempt=migrated, settled_units=1)
        settled_work += 1
    if old_result is not None:
        source_attempt = job.attempts[-1]
        attempt_id = f"{job.job_id}:attempt:{source_attempt.number}"
        result = _task_result(
            job=job,
            task_id=work_id,
            attempt_id=attempt_id,
            run_id=source_attempt.execution_run_id,
            result_id=work_result_id or f"{job.job_id}:work-result",
            payload=dict(old_result.summary),
            summary=f"Migrated revision-1 result {old_result.result_id}",
            provenance={
                **dict(old_result.provenance),
                "revision_1_result_id": old_result.result_id,
            },
            artifact_ids=tuple(
                sorted(ref.artifact_id for ref in old_result.artifact_refs)
            ),
            completed_at=old_result.completed_at,
        )
        connection.execute(
            """INSERT INTO job_task_results(
                   agent_id, job_id, task_id, result_id, attempt_id,
                   completed_at_us, sensitivity, result_digest, data
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.agent_id,
                result.job_id,
                result.task_id,
                result.result_id,
                result.attempt_id,
                datetime_to_us(result.completed_at),
                result.sensitivity.value,
                result.result_digest,
                encode_task_result(result),
            ),
        )
    final_settled = 0
    if state is GraphState.SUCCEEDED:
        final_attempt_id = f"{job.job_id}:finalizer-attempt"
        final_run_id = f"{job.job_id}:finalizer-run"
        final_time = terminal_at or effective_now
        final_attempt = TaskAttempt(
            agent_id=job.agent_id,
            job_id=job.job_id,
            task_id=finalizer_id,
            attempt_id=final_attempt_id,
            ordinal=1,
            fencing_epoch=1,
            state=AttemptState.SUCCEEDED,
            claim_token=f"{job.job_id}:migration-finalizer-claim",
            run_id=final_run_id,
            lease_expires_at=None,
            absolute_deadline_at=job.specification.deadline_at,
            started_at=final_time,
            heartbeat_at=final_time,
            ended_at=final_time,
            execution_scope_digest=finalizer.task_scope_digest,
            executor_id="migration:internal_finalizer",
            reserved_budgets=(BudgetAmount("work_units", 1),),
            measured_usage=(BudgetAmount("work_units", 1),),
            result_id=final_result_id,
            artifact_ids=(
                ()
                if old_result is None
                else tuple(sorted(ref.artifact_id for ref in old_result.artifact_refs))
            ),
        )
        _insert_attempt(connection, attempt=final_attempt, settled_units=1)
        final_settled = 1
        final_result = _task_result(
            job=job,
            task_id=finalizer_id,
            attempt_id=final_attempt_id,
            run_id=final_run_id,
            result_id=final_result_id or f"{job.job_id}:final-result",
            payload={"work_result_id": work_result_id},
            summary="Migrated revision-1 terminal outcome",
            provenance={
                "source_revision": 1,
                "followup_ids": tuple(sorted(item.followup_id for item in followups)),
            },
            artifact_ids=(
                ()
                if old_result is None
                else tuple(sorted(ref.artifact_id for ref in old_result.artifact_refs))
            ),
            completed_at=final_time,
        )
        connection.execute(
            """INSERT INTO job_task_results(
                   agent_id, job_id, task_id, result_id, attempt_id,
                   completed_at_us, sensitivity, result_digest, data
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                final_result.agent_id,
                final_result.job_id,
                final_result.task_id,
                final_result.result_id,
                final_result.attempt_id,
                datetime_to_us(final_result.completed_at),
                final_result.sensitivity.value,
                final_result.result_digest,
                encode_task_result(final_result),
            ),
        )
    if attention:
        control_task = finalizer if final_state is TaskState.BLOCKED else work
        control = TaskControl(
            agent_id=job.agent_id,
            job_id=job.job_id,
            task_id=control_task.task_id,
            control_id=f"{job.job_id}:migration-attention",
            kind=ControlKind.CAPABILITY_UNAVAILABLE,
            state=ControlState.OPEN,
            requesting_attempt_id=None,
            payload={"reason": reason, "classification": evidence},
            created_at=effective_now,
            payload_digest=canonical_digest(
                {"reason": reason, "classification": evidence}
            ),
        )
        connection.execute(
            """INSERT INTO job_task_controls(
                   agent_id, job_id, task_id, control_id, kind, state,
                   requesting_attempt_id, created_at_us, resolved_at_us,
                   resolved_by_kind, resolved_by_id, payload_digest, data
               ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, NULL, ?, ?)""",
            (
                control.agent_id,
                control.job_id,
                control.task_id,
                control.control_id,
                control.kind.value,
                control.state.value,
                datetime_to_us(control.created_at),
                control.payload_digest,
                encode_task_control(control),
            ),
        )
    connection.execute(
        """UPDATE job_budget_ledger SET settled = ?, updated_at_us = ?
           WHERE agent_id = ? AND job_id = ? AND dimension = 'work_units'""",
        (
            settled_work + final_settled,
            datetime_to_us(effective_now),
            job.agent_id,
            job.job_id,
        ),
    )
    connection.execute(
        """UPDATE job_task_budget_ledger SET settled = ?, updated_at_us = ?
           WHERE agent_id = ? AND job_id = ? AND task_id = ?
             AND dimension = 'work_units'""",
        (
            settled_work,
            datetime_to_us(effective_now),
            job.agent_id,
            job.job_id,
            work_id,
        ),
    )
    connection.execute(
        """UPDATE job_task_budget_ledger SET settled = ?, updated_at_us = ?
           WHERE agent_id = ? AND job_id = ? AND task_id = ?
             AND dimension = 'work_units'""",
        (
            final_settled,
            datetime_to_us(effective_now),
            job.agent_id,
            job.job_id,
            finalizer_id,
        ),
    )
    return ExecutabilityLedgerEntry(
        source_kind="job",
        source_id=job.job_id,
        decoded_state=job.status.value,
        evidence_classification=evidence,
        target_job_id=job.job_id,
        target_graph_state=state,
        runnable=runnable,
        executor_kind=("data_profile" if runnable else None),
        target_task_id=(work_id if runnable else None),
        reason=reason,
    )


def _convert_deliveries_and_followups(
    connection: sqlite3.Connection,
    followups: tuple[AutonomousFollowup, ...],
    job_states: dict[str, GraphState],
) -> tuple[ExecutabilityLedgerEntry, ...]:
    entries: list[ExecutabilityLedgerEntry] = []
    for followup in followups:
        completed = (
            followup.disposition is FollowupDisposition.COMPLETED
            and followup.delivery_id is not None
        )
        if completed:
            row = connection.execute(
                """SELECT data FROM deliveries
                   WHERE agent_id = ? AND delivery_id = ?""",
                (followup.agent_id, followup.delivery_id),
            ).fetchone()
            if row is None:
                raise ValueError("completed follow-up delivery is missing")
            payload = json.loads(str(row[0]))
            if (
                not isinstance(payload, dict)
                or payload.get("__record__") != "Delivery"
                or not isinstance(payload.get("fields"), dict)
            ):
                raise ValueError("stored revision-1 delivery is invalid")
            fields = payload["fields"]
            former_subject_kind = fields.get("subject_kind")
            former_subject_id = fields.get("subject_id")
            former_logical_key = fields.get("logical_key")
            fields["subject_kind"] = "graph_job"
            fields["subject_id"] = followup.job_id
            fields["logical_key"] = f"graph_job/{followup.job_id}"
            fields["migration_provenance"] = {
                "source_revision": 1,
                "former_subject_kind": former_subject_kind,
                "former_subject_id": former_subject_id,
                "former_logical_key": former_logical_key,
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            connection.execute(
                """UPDATE deliveries
                   SET subject_kind = 'graph_job', subject_id = ?, logical_key = ?,
                       data = ?
                   WHERE agent_id = ? AND delivery_id = ?""",
                (
                    followup.job_id,
                    f"graph_job/{followup.job_id}",
                    encoded,
                    followup.agent_id,
                    followup.delivery_id,
                ),
            )
        entries.append(
            ExecutabilityLedgerEntry(
                source_kind="followup",
                source_id=followup.followup_id,
                decoded_state=followup.disposition.value,
                evidence_classification=(
                    "delivered_followup" if completed else "unmappable_followup"
                ),
                target_job_id=followup.job_id,
                target_graph_state=job_states[followup.job_id],
                runnable=False,
                executor_kind=None,
                target_task_id=None,
                reason=(
                    "completed delivery provenance preserved"
                    if completed
                    else "follow-up preserved as graph attention provenance"
                ),
            )
        )
    return tuple(entries)


def stage_draft_revision_2_home(
    source_home: Path,
    target_home: Path,
    *,
    now: datetime,
    phase_hook: DraftPhaseHook | None = None,
) -> DraftConversionReport:
    """Convert one revision-1 home into an isolated, unstamped draft home."""

    source_home = source_home.resolve()
    target_home = target_home.resolve()
    if target_home.exists() and any(target_home.iterdir()):
        raise ValueError("draft conversion target must be absent or empty")
    validate_current_state_database(source_home / "state.db")
    _copy_home_files(source_home, target_home)
    backup = target_home / ".revision-1-source.db"
    _backup_database(source_home / "state.db", backup)
    if phase_hook is not None:
        phase_hook("after_backup")
    target_path = target_home / "state.db"
    target = sqlite3.connect(target_path)
    source = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    try:
        source.execute("PRAGMA query_only = ON")
        source.execute("PRAGMA foreign_keys = ON")
        create_draft_graph_database(target)
        if phase_hook is not None:
            phase_hook("after_schema")
        target.execute("BEGIN IMMEDIATE")
        _copy_unchanged_state(source, target)
        if phase_hook is not None:
            phase_hook("after_unchanged_copy")
        job_rows = tuple(
            source.execute(
                "SELECT agent_id, job_id, data FROM job_runs ORDER BY agent_id, job_id"
            )
        )
        followup_rows = tuple(
            source.execute(
                """SELECT agent_id, followup_id, data FROM autonomous_followups
                   ORDER BY agent_id, followup_id"""
            )
        )
        jobs = tuple(
            _decode_revision_1_job(
                str(data), agent_id=str(agent_id), job_id=str(job_id)
            )
            for agent_id, job_id, data in job_rows
        )
        followups = tuple(
            _decode_revision_1_followup(
                str(data), agent_id=str(agent_id), followup_id=str(followup_id)
            )
            for agent_id, followup_id, data in followup_rows
        )
        by_job: dict[str, list[AutonomousFollowup]] = defaultdict(list)
        for followup in followups:
            by_job[followup.job_id].append(followup)
        entries = tuple(
            _convert_job(
                target,
                job,
                now=now,
                followups=tuple(by_job.get(job.job_id, ())),
            )
            for job in jobs
        )
        if phase_hook is not None:
            phase_hook("after_jobs")
        job_states = {
            entry.target_job_id: entry.target_graph_state for entry in entries
        }
        followup_entries = _convert_deliveries_and_followups(
            target, followups, job_states
        )
        if phase_hook is not None:
            phase_hook("after_followups")
        target.commit()
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException:
        target.rollback()
        raise
    finally:
        source.close()
        target.close()
        backup.unlink(missing_ok=True)
    os.chmod(target_path, 0o600)
    report = DraftConversionReport(
        source_revision=1,
        target_revision=2,
        entries=tuple(
            sorted(
                (*entries, *followup_entries),
                key=lambda item: (item.source_kind, item.source_id),
            )
        ),
        source_database_sha256=_sha256(source_home / "state.db"),
        target_database_sha256=_sha256(target_path),
    )
    validate_draft_revision_2_home(source_home, target_home, report=report)
    if phase_hook is not None:
        phase_hook("after_validation")
    return report


def _source_record_keys(path: Path) -> set[tuple[str, str]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        jobs = {
            ("job", str(row[0]))
            for row in connection.execute("SELECT job_id FROM job_runs")
        }
        followups = {
            ("followup", str(row[0]))
            for row in connection.execute(
                "SELECT followup_id FROM autonomous_followups"
            )
        }
        return jobs | followups
    finally:
        connection.close()


def _validate_ledger(
    source_home: Path,
    connection: sqlite3.Connection,
    report: DraftConversionReport,
) -> None:
    source_keys = _source_record_keys(source_home / "state.db")
    ledger_keys = {(entry.source_kind, entry.source_id) for entry in report.entries}
    if source_keys != ledger_keys or len(ledger_keys) != len(report.entries):
        raise ValueError("draft executability ledger coverage is incomplete")
    target_jobs = {
        str(row[0]) for row in connection.execute("SELECT job_id FROM job_runs")
    }
    source_jobs = {
        entry.target_job_id for entry in report.entries if entry.source_kind == "job"
    }
    if target_jobs != source_jobs:
        raise ValueError("draft target graph lacks source or migration provenance")
    for entry in report.entries:
        if not entry.runnable:
            continue
        if entry.executor_kind != "data_profile" or entry.target_task_id is None:
            raise ValueError("runnable ledger row has no current graph executor")
        row = connection.execute(
            """SELECT state, task_kind FROM job_tasks
               WHERE job_id = ? AND task_id = ?""",
            (entry.target_job_id, entry.target_task_id),
        ).fetchone()
        if row != ("ready", "internal_capability"):
            raise ValueError("runnable ledger target is not executable")


def _validate_graph_records(connection: sqlite3.Connection) -> None:
    job_keys = tuple(
        (str(row[0]), str(row[1]))
        for row in connection.execute("SELECT agent_id, job_id FROM job_runs")
    )
    for agent_id, job_id in job_keys:
        inspection = inspect_graph(connection, agent_id, job_id)
        if inspection is None:
            raise ValueError("draft graph disappeared during validation")
        job = inspection.job
        graph = inspection.graph
        limits = job.specification.limits
        validate_graph_topology(
            inspection.tasks,
            inspection.dependencies,
            graph=graph,
            max_tasks=limits.max_tasks,
            max_edges=limits.max_edges,
            max_depth=limits.max_depth,
            max_direct_parents=limits.max_direct_parents,
        )
        finalizers = [
            task for task in inspection.tasks if task.role is TaskRole.FINALIZER
        ]
        if len(finalizers) != 1 or finalizers[0].task_id != job.finalizer_task_id:
            raise ValueError("draft graph finalizer invariant failed")
        active = [
            attempt
            for attempt in inspection.attempts
            if attempt.state in {AttemptState.CLAIMED, AttemptState.RUNNING}
        ]
        if len(active) != graph.active_attempt_count:
            raise ValueError("draft graph active-attempt counter is invalid")
        mutation_rows = tuple(
            decode_graph_mutation(str(row[0]))
            for row in connection.execute(
                """SELECT data FROM job_graph_mutations
                   WHERE agent_id = ? AND job_id = ? ORDER BY created_at_us, mutation_id""",
                (agent_id, job_id),
            )
        )
        if len(mutation_rows) != graph.mutation_count:
            raise ValueError("draft graph mutation counter is invalid")
        committed_revisions = sorted(
            item.committed_revision
            for item in mutation_rows
            if item.decision.value == "committed"
        )
        if committed_revisions != list(range(1, graph.revision + 1)):
            raise ValueError("draft graph revision history is invalid")
        event_count = int(
            connection.execute(
                """SELECT COUNT(*) FROM job_graph_events
                   WHERE agent_id = ? AND job_id = ?""",
                (agent_id, job_id),
            ).fetchone()[0]
        )
        if event_count > limits.max_events:
            raise ValueError("draft graph event bound is exceeded")
        attempts_by_task: dict[str, list[TaskAttempt]] = defaultdict(list)
        for attempt in inspection.attempts:
            attempts_by_task[attempt.task_id].append(attempt)
        results_by_task = {result.task_id: result for result in inspection.results}
        tasks_by_id = {task.task_id: task for task in inspection.tasks}
        for task in inspection.tasks:
            require_authority_subset(
                task.specification.authority, job.specification.authority
            )
            attempts = sorted(
                attempts_by_task[task.task_id], key=lambda item: item.ordinal
            )
            if [item.ordinal for item in attempts] != list(range(1, len(attempts) + 1)):
                raise ValueError("draft task attempt ordinals are invalid")
            if task.attempt_count != len(attempts):
                raise ValueError("draft task attempt counter is invalid")
            fences = [item.fencing_epoch for item in attempts]
            if fences != sorted(set(fences)):
                raise ValueError("draft task fences are not monotonic")
            if fences and task.fencing_epoch < max(fences):
                raise ValueError("draft task fencing projection regressed")
            task_active = [
                item
                for item in attempts
                if item.state in {AttemptState.CLAIMED, AttemptState.RUNNING}
            ]
            if len(task_active) > 1:
                raise ValueError("draft task has multiple active attempts")
            projected_attempt = None if not task_active else task_active[0].attempt_id
            if task.current_attempt_id != projected_attempt:
                raise ValueError("draft task current-attempt projection is invalid")
            for attempt in attempts:
                budget_rows = tuple(
                    connection.execute(
                        """SELECT dimension, reserved, settled
                           FROM job_attempt_budget_reservations
                           WHERE agent_id = ? AND job_id = ? AND task_id = ?
                             AND attempt_id = ? ORDER BY dimension""",
                        (agent_id, job_id, task.task_id, attempt.attempt_id),
                    )
                )
                if (
                    tuple(BudgetAmount(str(row[0]), int(row[1])) for row in budget_rows)
                    != attempt.reserved_budgets
                ):
                    raise ValueError("draft attempt budget reservation is invalid")
                projected_usage = tuple(
                    BudgetAmount(str(row[0]), int(row[2]))
                    for row in budget_rows
                    if row[2] is not None
                )
                if projected_usage != attempt.measured_usage:
                    raise ValueError("draft attempt measured usage is invalid")
                checkpoint_ids = tuple(
                    sorted(
                        str(row[0])
                        for row in connection.execute(
                            """SELECT checkpoint_id FROM job_task_checkpoints
                               WHERE agent_id = ? AND job_id = ? AND task_id = ?
                                 AND attempt_id = ?""",
                            (agent_id, job_id, task.task_id, attempt.attempt_id),
                        )
                    )
                )
                if checkpoint_ids != attempt.checkpoint_ids:
                    raise ValueError("draft attempt checkpoint references are invalid")
                control_ids = tuple(
                    sorted(
                        str(row[0])
                        for row in connection.execute(
                            """SELECT control_id FROM job_task_controls
                               WHERE agent_id = ? AND job_id = ? AND task_id = ?
                                 AND requesting_attempt_id = ?""",
                            (agent_id, job_id, task.task_id, attempt.attempt_id),
                        )
                    )
                )
                if control_ids != attempt.control_ids:
                    raise ValueError("draft attempt control references are invalid")
            if task.supersedes_task_id is not None:
                prior = tasks_by_id.get(task.supersedes_task_id)
                if prior is None or prior.superseded_by_task_id != task.task_id:
                    raise ValueError("draft task supersession is not reciprocal")
            if task.superseded_by_task_id is not None:
                replacement = tasks_by_id.get(task.superseded_by_task_id)
                if (
                    replacement is None
                    or replacement.supersedes_task_id != task.task_id
                ):
                    raise ValueError("draft task supersession is not reciprocal")
            if task.state is TaskState.SUCCEEDED:
                result = results_by_task.get(task.task_id)
                if result is None or result.result_id != task.latest_result_id:
                    raise ValueError("draft successful task result is invalid")
                matching = [
                    item for item in attempts if item.attempt_id == result.attempt_id
                ]
                if (
                    len(matching) != 1
                    or matching[0].state is not AttemptState.SUCCEEDED
                    or matching[0].run_id != result.run_id
                    or matching[0].result_id != result.result_id
                    or matching[0].artifact_ids != result.artifact_ids
                    or matching[0].effect_receipt_ids != result.effect_receipt_ids
                ):
                    raise ValueError("draft task result attempt is invalid")
            elif task.task_id in results_by_task:
                raise ValueError("draft non-successful task has an accepted result")
        if graph.finalization_attempt_id is not None:
            finalizer_attempt = next(
                (
                    item
                    for item in active
                    if item.attempt_id == graph.finalization_attempt_id
                    and item.task_id == job.finalizer_task_id
                ),
                None,
            )
            if (
                finalizer_attempt is None
                or graph.finalization_started_revision is None
                or graph.finalization_started_revision > graph.revision
            ):
                raise ValueError("draft graph finalization seal is invalid")
        if job.state is GraphState.SUCCEEDED:
            finalizer = finalizers[0]
            if (
                finalizer.state is not TaskState.SUCCEEDED
                or finalizer.latest_result_id != job.terminal_result_id
            ):
                raise ValueError("draft successful job finalizer result is invalid")
            unresolved = connection.execute(
                """SELECT 1 FROM effect_receipts
                   WHERE agent_id = ? AND job_id = ? AND unresolved = 1 LIMIT 1""",
                (agent_id, job_id),
            ).fetchone()
            if unresolved is not None:
                raise ValueError("draft successful graph has an unresolved effect")
        root_rows = {
            str(row[0]): (int(row[1]), int(row[2]), int(row[3]))
            for row in connection.execute(
                """SELECT dimension, ceiling, settled, reserved
                   FROM job_budget_ledger WHERE agent_id = ? AND job_id = ?""",
                (agent_id, job_id),
            )
        }
        if {item.dimension: item.ceiling for item in job.specification.budgets} != {
            dimension: values[0] for dimension, values in root_rows.items()
        }:
            raise ValueError("draft root budget ledger shape is invalid")
        reservations = tuple(
            connection.execute(
                """SELECT dimension, reserved, settled
                   FROM job_attempt_budget_reservations
                   WHERE agent_id = ? AND job_id = ?""",
                (agent_id, job_id),
            )
        )
        for dimension, (ceiling, settled, reserved) in root_rows.items():
            projected_settled = sum(
                int(row[2])
                for row in reservations
                if row[0] == dimension and row[2] is not None
            )
            projected_reserved = sum(
                int(row[1])
                for row in reservations
                if row[0] == dimension and row[2] is None
            )
            if (settled, reserved) != (projected_settled, projected_reserved):
                raise ValueError("draft root budget ledger is not conserved")
            if settled + reserved > ceiling:
                raise ValueError("draft root budget ceiling is exceeded")
        task_rows = tuple(
            connection.execute(
                """SELECT task_id, dimension, ceiling, settled, reserved
                   FROM job_task_budget_ledger
                   WHERE agent_id = ? AND job_id = ?""",
                (agent_id, job_id),
            )
        )
        expected_task_ledgers = {
            (task.task_id, item.dimension): item.amount
            for task in inspection.tasks
            for item in task.specification.budgets
        }
        actual_task_ledgers = {
            (str(row[0]), str(row[1])): int(row[2]) for row in task_rows
        }
        if actual_task_ledgers != expected_task_ledgers:
            raise ValueError("draft task budget ledger shape is invalid")
        for task_id, dimension, ceiling, settled, reserved in task_rows:
            task_reservations = tuple(
                connection.execute(
                    """SELECT reserved, settled
                       FROM job_attempt_budget_reservations
                       WHERE agent_id = ? AND job_id = ? AND task_id = ?
                         AND dimension = ?""",
                    (agent_id, job_id, task_id, dimension),
                )
            )
            projected_settled = sum(
                int(row[1]) for row in task_reservations if row[1] is not None
            )
            projected_reserved = sum(
                int(row[0]) for row in task_reservations if row[1] is None
            )
            if (int(settled), int(reserved)) != (
                projected_settled,
                projected_reserved,
            ):
                raise ValueError("draft task budget ledger is not conserved")
            if int(settled) + int(reserved) > int(ceiling):
                raise ValueError("draft task budget ceiling is exceeded")
    for (data,) in connection.execute("SELECT data FROM job_graph_mutations"):
        decode_graph_mutation(str(data))
    for (data,) in connection.execute("SELECT data FROM job_task_checkpoints"):
        decode_task_checkpoint(str(data))
    for (data,) in connection.execute("SELECT data FROM job_task_comments"):
        decode_task_comment(str(data))
    for (data,) in connection.execute("SELECT data FROM job_task_controls"):
        decode_task_control(str(data))
    for row in connection.execute(
        """SELECT agent_id, delivery_id, conversation_id, subject_kind,
                  subject_id, logical_key, state, data
           FROM deliveries"""
    ):
        decode_draft_delivery(
            str(row[7]),
            agent_id=str(row[0]),
            delivery_id=str(row[1]),
            conversation_id=str(row[2]),
            subject_kind=str(row[3]),
            subject_id=str(row[4]),
            logical_key=str(row[5]),
            state=str(row[6]),
        )


def _validate_home_files(source_home: Path, candidate_home: Path) -> None:
    for source in source_home.rglob("*"):
        relative = source.relative_to(source_home)
        if relative.parts[0] in {".home-upgrade", ".home-rollbacks"}:
            continue
        if relative.as_posix() in {"state.db", "state.db-wal", "state.db-shm"}:
            continue
        candidate = candidate_home / relative
        if source.is_dir():
            if not candidate.is_dir():
                raise ValueError("draft whole-home directory is missing")
        elif not candidate.is_file() or source.read_bytes() != candidate.read_bytes():
            raise ValueError("draft whole-home non-database content changed")


def validate_draft_revision_2_home(
    source_home: Path,
    candidate_home: Path,
    *,
    report: DraftConversionReport,
) -> None:
    """Validate the complete staged draft home and its semantic conversion."""

    _validate_home_files(source_home, candidate_home)
    with connect_draft_graph(candidate_home / "state.db", read_only=True) as connection:
        require_draft_graph_schema(connection)
        journal = tuple(connection.execute("""SELECT revision, migration_id, checksum
                   FROM agent_home_migrations ORDER BY revision"""))
        if len(journal) != 1 or journal[0][0] != 1:
            raise ValueError("draft home must retain only the revision-1 stamp")
        if (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'autonomous_followups'"
            ).fetchone()
            is not None
        ):
            raise ValueError("draft target retained the autonomous follow-up table")
        _validate_graph_records(connection)
        _validate_ledger(source_home, connection, report)
        legacy_delivery = connection.execute(
            "SELECT 1 FROM deliveries WHERE subject_kind = 'autonomous_followup' LIMIT 1"
        ).fetchone()
        if legacy_delivery is not None:
            raise ValueError("draft target retained an autonomous follow-up delivery")
    if report.source_database_sha256 != _sha256(source_home / "state.db"):
        raise ValueError("draft conversion source changed after inventory")
    if report.target_database_sha256 != _sha256(candidate_home / "state.db"):
        raise ValueError("draft conversion target changed after validation")


def publish_draft_revision_2_for_test(
    active_home: Path,
    *,
    now: datetime,
    phase_hook: DraftPhaseHook | None = None,
) -> DraftConversionReport:
    """Exercise state-last publication and rollback without registering revision 2."""

    active_home = active_home.resolve()
    work = Path(tempfile.mkdtemp(prefix="daita-draft-r2-", dir=active_home.parent))
    source_snapshot = work / "source"
    stage = work / "stage"
    backup = work / "state.db.before"
    published = False
    report: DraftConversionReport | None = None
    try:
        _copy_home_files(active_home, source_snapshot)
        _backup_database(active_home / "state.db", source_snapshot / "state.db")
        report = stage_draft_revision_2_home(
            source_snapshot, stage, now=now, phase_hook=phase_hook
        )
        shutil.copyfile(active_home / "state.db", backup)
        if phase_hook is not None:
            phase_hook("before_publication")
        temporary = active_home / ".state.db.draft-r2"
        shutil.copyfile(stage / "state.db", temporary)
        os.replace(temporary, active_home / "state.db")
        published = True
        if phase_hook is not None:
            phase_hook("after_state_publication")
        validate_draft_revision_2_home(source_snapshot, active_home, report=report)
        return report
    except BaseException:
        if published:
            restore = active_home / ".state.db.restore-r1"
            shutil.copyfile(backup, restore)
            os.replace(restore, active_home / "state.db")
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


__all__ = [
    "DraftConversionReport",
    "ExecutabilityLedgerEntry",
    "publish_draft_revision_2_for_test",
    "stage_draft_revision_2_home",
    "validate_draft_revision_2_home",
]
