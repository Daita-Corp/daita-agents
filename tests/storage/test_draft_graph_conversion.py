from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita.capabilities import ExecutionContractBindings
from daita.distribution.models import (
    ConversationInboxTarget,
    DeliveryState,
    OutcomeConclusionKind,
    OutcomeReference,
    OutcomeState,
)
from daita.jobs.graph.models import GraphState
from daita.llm.models import ModelSensitivity
from daita.loop.models import LoopLimits
from daita.storage.graph_schema import (
    GRAPH_TABLE_NAMES,
    connect_graph,
)
from daita.storage.home_migrations.revision_0001 import REVISION_1
from daita.storage.home_migrations.revision_0001_schema import (
    REVISION_1_DATABASE_SQL,
)
from daita.storage.home_migrations.revision_0002_conversion import (
    publish_revision_2_for_test,
    stage_revision_2_home,
    validate_revision_2_home,
)
from daita.storage.home_migrations.revision_0002_legacy_autonomy import (
    FollowupConclusionEvidence,
    create_terminal_job_followup,
)
from daita.storage.home_migrations.revision_0002_legacy_autonomy_codecs import (
    encode_autonomous_followup,
)
from daita.storage.home_migrations.revision_0002_legacy_delivery import (
    Revision1Delivery,
    Revision1DeliverySubjectKind,
    encode_revision_1_delivery,
    revision_1_logical_delivery_key,
)
from daita.storage.home_migrations.revision_0002_legacy_job_codecs import (
    encode_job_run,
)
from daita.storage.home_migrations.revision_0002_legacy_jobs import (
    ConnectedExecutorBinding,
    ExternalIntent,
    ExternalIntentDisposition,
    ExternalIntentKind,
    JobAttempt,
    JobAttemptStatus,
    JobDesiredState,
    JobExecutionMode,
    JobResourceBinding,
    JobResult,
    JobRun,
    JobSpecification,
    JobStatus,
)
from daita.storage.sqlite import SQLiteStateStore
from tests.support.graph import GRAPH_NOW

pytestmark = pytest.mark.integration

ROOT = Path(__file__).parents[2]
REVISION_1_GOLDEN = ROOT / "tests/fixtures/agent-home-revisions/revision-1"
DRAFT_GOLDEN = ROOT / "tests/fixtures/draft-agent-home-revisions/revision-2-candidate"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_revision_1_golden(target: Path) -> None:
    target.mkdir(parents=True)
    for source in REVISION_1_GOLDEN.rglob("*"):
        relative = source.relative_to(REVISION_1_GOLDEN)
        if relative.as_posix() == "state.sql":
            continue
        destination = target / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    connection = sqlite3.connect(target / "state.db")
    try:
        connection.executescript(
            (REVISION_1_GOLDEN / "state.sql").read_text(encoding="utf-8")
        )
    finally:
        connection.close()


def _create_empty_revision_1_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(REVISION_1_DATABASE_SQL)
        connection.execute(
            "INSERT INTO agent_home_migrations(revision, migration_id, checksum) "
            "VALUES (1, ?, ?)",
            (REVISION_1.migration_id, REVISION_1.checksum),
        )


def _specification(
    *, execution_mode: JobExecutionMode = JobExecutionMode.DAITA
) -> JobSpecification:
    binding = JobResourceBinding(
        source_id="source-1",
        source_revision="source-revision-1",
        resource_id="resource-1",
        resource_revision="sha256:" + "1" * 64,
        adapter_id="sqlite",
        sensitivity=ModelSensitivity.INTERNAL,
    )
    external = (
        None
        if execution_mode is JobExecutionMode.DAITA
        else ConnectedExecutorBinding(
            profile_id="profile-1",
            binding_id="binding-1",
            execution_identity="identity-1",
            contract_digest="sha256:" + "2" * 64,
            revision=1,
            maximum_sensitivity=ModelSensitivity.INTERNAL,
        )
    )
    return JobSpecification(
        job_kind="data_profile",
        arguments={"resource_ids": ("resource-1",), "sample_rows": 10},
        resource_bindings=(binding,),
        execution_capability_id="jobs.data_profile.execute",
        execution_contract_digest="sha256:" + "3" * 64,
        execution_mode=execution_mode,
        sensitivity=ModelSensitivity.INTERNAL,
        deadline_at=GRAPH_NOW + timedelta(hours=1),
        max_wall_time_seconds=60,
        external_executor=external,
    )


def _job(job_id: str, *, execution_mode=JobExecutionMode.DAITA) -> JobRun:
    specification = _specification(execution_mode=execution_mode)
    return JobRun(
        job_id=job_id,
        agent_id="agent-1",
        conversation_id="conversation-1",
        origin_run_id=f"{job_id}:origin-run",
        origin_call_id=f"{job_id}:origin-call",
        specification=specification,
        specification_digest=specification.digest,
        status=JobStatus.QUEUED,
        desired_state=JobDesiredState.RUN,
        created_at=GRAPH_NOW,
        updated_at=GRAPH_NOW,
    )


@pytest.mark.asyncio
async def test_revision_one_jobs_convert_with_complete_executability_ledger(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "USER.md").write_text("owner data\n", encoding="utf-8")
    _create_empty_revision_1_database(source / "state.db")
    succeeded_attempt = JobAttempt(
        number=1,
        fencing_epoch=1,
        claim_token="claim-1",
        execution_run_id="run-1",
        reserved_artifact_id="artifact-1",
        status=JobAttemptStatus.SUCCEEDED,
        claimed_at=GRAPH_NOW + timedelta(seconds=1),
        lease_expires_at=GRAPH_NOW + timedelta(seconds=31),
        completed_at=GRAPH_NOW + timedelta(seconds=2),
    )
    result = JobResult(
        result_id="job-result-1",
        summary={"profiled_resources": 1},
        sensitivity=ModelSensitivity.INTERNAL,
        provenance={"kind": "test"},
        artifact_refs=(),
        completed_at=GRAPH_NOW + timedelta(seconds=2),
    )
    succeeded = replace(
        _job("a-succeeded"),
        status=JobStatus.SUCCEEDED,
        updated_at=GRAPH_NOW + timedelta(seconds=2),
        revision=3,
        fencing_epoch=1,
        attempts=(succeeded_attempt,),
        terminal_at=GRAPH_NOW + timedelta(seconds=2),
        result=result,
    )
    connected_attempt = JobAttempt(
        number=1,
        fencing_epoch=1,
        claim_token="claim-2",
        execution_run_id="run-2",
        reserved_artifact_id="artifact-2",
        status=JobAttemptStatus.CLAIMED,
        claimed_at=GRAPH_NOW + timedelta(seconds=2),
        lease_expires_at=GRAPH_NOW + timedelta(seconds=32),
    )
    connected_attempt = replace(
        connected_attempt,
        external_intents=(
            ExternalIntent(
                kind=ExternalIntentKind.START,
                idempotency_key="external-start-1",
                requested_at=GRAPH_NOW + timedelta(seconds=2),
                disposition=ExternalIntentDisposition.ACCEPTED,
                completed_at=GRAPH_NOW + timedelta(seconds=2),
                external_job_id="remote-job-1",
            ),
        ),
    )
    connected_with_evidence = replace(
        _job("connected", execution_mode=JobExecutionMode.CONNECTED_EXECUTOR),
        status=JobStatus.RUNNING,
        attempts=(connected_attempt,),
        revision=3,
        fencing_epoch=1,
        updated_at=GRAPH_NOW + timedelta(seconds=2),
    )
    # The connected job intentionally remains nonterminal and maps to attention.
    queued_attempt = JobAttempt(
        number=1,
        fencing_epoch=1,
        claim_token="claim-3",
        execution_run_id="run-3",
        reserved_artifact_id="artifact-3",
        status=JobAttemptStatus.CLAIMED,
        claimed_at=GRAPH_NOW + timedelta(seconds=3),
        lease_expires_at=GRAPH_NOW + timedelta(seconds=33),
    )
    queued = replace(
        _job("queued"),
        status=JobStatus.RUNNING,
        updated_at=GRAPH_NOW + timedelta(seconds=3),
        revision=2,
        fencing_epoch=1,
        attempts=(queued_attempt,),
    )
    # Running native work has exact restart-safe evidence and remains runnable.

    terminal_time = GRAPH_NOW + timedelta(seconds=5)
    failed = replace(
        _job("failed"),
        status=JobStatus.FAILED,
        updated_at=terminal_time,
        revision=2,
        terminal_at=terminal_time,
        failure_code="test_failure",
    )
    cancelled = replace(
        _job("cancelled"),
        status=JobStatus.CANCELLED,
        updated_at=terminal_time,
        revision=2,
        terminal_at=terminal_time,
    )
    attention = replace(
        _job("attention"),
        status=JobStatus.NEEDS_ATTENTION,
        updated_at=terminal_time,
        revision=2,
        terminal_at=terminal_time,
        failure_code="test_attention",
    )
    cancel_intent = replace(
        _job("cancel-intent"),
        desired_state=JobDesiredState.CANCEL,
        updated_at=terminal_time,
        revision=2,
        cancel_requested_at=terminal_time,
    )
    expiring_specification = replace(
        _specification(), deadline_at=GRAPH_NOW + timedelta(seconds=30)
    )
    expired = replace(
        _job("expired"),
        specification=expiring_specification,
        specification_digest=expiring_specification.digest,
    )
    followup_job = replace(
        _job("followup-pending"),
        status=JobStatus.FAILED,
        updated_at=terminal_time,
        revision=2,
        terminal_at=terminal_time,
        failure_code="test_followup_failure",
    )
    pending_followup = create_terminal_job_followup(
        followup_job,
        followup_id="followup-1",
        grant_id="grant-1",
        scope_id="scope-1",
        received_at=terminal_time + timedelta(seconds=1),
        allowed_capability_ids=("jobs.inspect", "jobs.read_results"),
        eligible_model_routes=("mock:scripted",),
        limits=LoopLimits(max_estimated_cost_usd=Decimal("1")),
        contract_bindings=ExecutionContractBindings(
            capability_contracts={
                "jobs.inspect": "sha256:" + "4" * 64,
                "jobs.read_results": "sha256:" + "5" * 64,
            },
            resource_revisions={"resource-1": "sha256:" + "1" * 64},
            model_routes={"mock:scripted": "sha256:" + "6" * 64},
        ),
    )
    completed_job = replace(
        _job("followup-completed"),
        status=JobStatus.FAILED,
        updated_at=terminal_time,
        revision=2,
        terminal_at=terminal_time,
        failure_code="test_completed_followup_failure",
    )
    completed_template = create_terminal_job_followup(
        completed_job,
        followup_id="followup-2",
        grant_id="grant-2",
        scope_id="scope-2",
        received_at=terminal_time + timedelta(seconds=1),
        allowed_capability_ids=("jobs.inspect", "jobs.read_results"),
        eligible_model_routes=("mock:scripted",),
        limits=LoopLimits(max_estimated_cost_usd=Decimal("1")),
        contract_bindings=pending_followup.execution_scope.contract_bindings,
    )
    evidence = FollowupConclusionEvidence(
        run_id="followup-run-2",
        job_id=completed_job.job_id,
        job_revision=completed_job.revision,
        inspection_call_id="inspect-call",
        inspection_result_digest="sha256:" + "7" * 64,
        result_call_id="result-call",
        result_result_digest="sha256:" + "8" * 64,
        job_result_id=None,
        report_digest="sha256:" + "9" * 64,
    )
    completed_followup = replace(
        completed_template,
        disposition=completed_template.disposition.COMPLETED,
        revision=2,
        attempt_count=1,
        reserved_run_id=evidence.run_id,
        updated_at=terminal_time + timedelta(seconds=2),
        grant_consumed_at=terminal_time + timedelta(seconds=2),
        conclusion_evidence=evidence,
        delivery_id="delivery-1",
    )
    target_binding = completed_followup.grant.distribution_plan.targets[0]
    assert isinstance(target_binding, ConversationInboxTarget)
    outcome = OutcomeReference(
        conclusion_kind=OutcomeConclusionKind.TERMINAL_RUN,
        conclusion_state=OutcomeState.FAILED,
        conclusion_id=evidence.run_id,
        conclusion_digest=evidence.report_digest,
        conclusion_preview="Terminal follow-up report.",
        conclusion_preview_truncated=False,
        resulting_run_id=evidence.run_id,
        artifact_references=(),
        effective_sensitivity=ModelSensitivity.INTERNAL,
        provenance_digest="sha256:" + "a" * 64,
        failure_code="test_completed_followup_failure",
        observed_at=terminal_time + timedelta(seconds=2),
    )
    old_logical_key = revision_1_logical_delivery_key(
        agent_id="agent-1",
        subject_kind=Revision1DeliverySubjectKind.AUTONOMOUS_FOLLOWUP,
        subject_id=completed_followup.followup_id,
        target_fingerprint=target_binding.target_fingerprint,
    )
    delivery = Revision1Delivery(
        delivery_id="delivery-1",
        agent_id="agent-1",
        conversation_id="conversation-1",
        subject_kind=Revision1DeliverySubjectKind.AUTONOMOUS_FOLLOWUP,
        subject_id=completed_followup.followup_id,
        logical_key=old_logical_key,
        target=target_binding,
        outcome=outcome,
        visibility_state=DeliveryState.AVAILABLE,
        acknowledged_at=None,
        blocked_reason_code=None,
        created_at=terminal_time + timedelta(seconds=2),
        updated_at=terminal_time + timedelta(seconds=2),
    )
    with sqlite3.connect(source / "state.db") as connection:
        connection.executemany(
            "INSERT INTO job_runs(agent_id, job_id, data) VALUES (?, ?, ?)",
            (
                (job.agent_id, job.job_id, encode_job_run(job))
                for job in (
                    succeeded,
                    connected_with_evidence,
                    queued,
                    failed,
                    cancelled,
                    attention,
                    cancel_intent,
                    expired,
                    followup_job,
                    completed_job,
                )
            ),
        )
        connection.execute(
            "INSERT INTO autonomous_followups("
            "agent_id, followup_id, job_id, event_id, data"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                pending_followup.agent_id,
                pending_followup.followup_id,
                pending_followup.job_id,
                pending_followup.event_id,
                encode_autonomous_followup(pending_followup),
            ),
        )
        connection.execute(
            "INSERT INTO autonomous_followups("
            "agent_id, followup_id, job_id, event_id, data"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                completed_followup.agent_id,
                completed_followup.followup_id,
                completed_followup.job_id,
                completed_followup.event_id,
                encode_autonomous_followup(completed_followup),
            ),
        )
        connection.execute(
            """INSERT INTO deliveries(
                   agent_id, delivery_id, conversation_id, subject_kind, subject_id,
                   logical_key, target_kind, target_fingerprint, state,
                   created_at_us, data
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                delivery.agent_id,
                delivery.delivery_id,
                delivery.conversation_id,
                delivery.subject_kind.value,
                delivery.subject_id,
                delivery.logical_key,
                "conversation_inbox",
                delivery.target.target_fingerprint,
                delivery.visibility_state.value,
                int(delivery.created_at.timestamp() * 1_000_000),
                encode_revision_1_delivery(delivery),
            ),
        )

    target = tmp_path / "target"
    report = stage_revision_2_home(source, target, now=GRAPH_NOW + timedelta(minutes=1))

    assert len(report.entries) == 12
    entries = {item.source_id: item for item in report.entries}
    assert entries["connected"].target_graph_state is GraphState.NEEDS_ATTENTION
    assert entries["connected"].runnable is False
    assert entries["queued"].target_graph_state is GraphState.QUEUED
    assert entries["queued"].runnable is True
    assert entries["a-succeeded"].target_graph_state is GraphState.SUCCEEDED
    assert entries["a-succeeded"].runnable is False
    assert entries["failed"].target_graph_state is GraphState.FAILED
    assert entries["cancelled"].target_graph_state is GraphState.CANCELLED
    assert entries["attention"].target_graph_state is GraphState.NEEDS_ATTENTION
    assert entries["cancel-intent"].target_graph_state is GraphState.CANCELLED
    assert entries["expired"].target_graph_state is GraphState.NEEDS_ATTENTION
    assert entries["followup-pending"].target_graph_state is GraphState.NEEDS_ATTENTION
    followup_entry = next(
        item
        for item in report.entries
        if item.source_kind == "followup" and item.source_id == "followup-1"
    )
    assert followup_entry.evidence_classification == "unmappable_followup"
    completed_entry = next(
        item
        for item in report.entries
        if item.source_kind == "followup" and item.source_id == "followup-2"
    )
    assert completed_entry.evidence_classification == "delivered_followup"
    assert (target / "USER.md").read_bytes() == (source / "USER.md").read_bytes()
    validate_revision_2_home(source, target, report=report)

    draft_store = await SQLiteStateStore.open(
        target / "state.db", current_home_validated=True
    )
    succeeded_graph = await draft_store.inspect_graph("agent-1", "a-succeeded")
    assert succeeded_graph is not None
    assert succeeded_graph.job.terminal_result_id == "a-succeeded:final-result"
    assert len(succeeded_graph.results) == 2
    connected = await draft_store.inspect_graph("agent-1", "connected")
    assert connected is not None
    assert "remote-job-1" in str(connected.job.migration_provenance["attempt_evidence"])
    with connect_graph(target / "state.db", read_only=True) as connection:
        migrated_delivery = connection.execute(
            "SELECT subject_kind, subject_id, logical_key, data "
            "FROM deliveries WHERE delivery_id = 'delivery-1'"
        ).fetchone()
    assert migrated_delivery is not None
    assert migrated_delivery[:3] == (
        "graph_job",
        "followup-completed",
        "graph_job/followup-completed",
    )
    assert old_logical_key in str(migrated_delivery[3])


def test_revision_one_golden_whole_home_converts_to_frozen_draft_manifest(
    tmp_path: Path,
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _create_revision_1_golden(source)
    expected = json.loads((DRAFT_GOLDEN / "expected.json").read_text())

    report = stage_revision_2_home(source, target, now=GRAPH_NOW)
    files = sorted(
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
        and path.name not in {"state.db", "state.db-wal", "state.db-shm"}
    )
    with connect_graph(target / "state.db", read_only=True) as connection:
        table_count = len(
            tuple(
                connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            )
        )

    assert {
        "source_revision": report.source_revision,
        "target_revision": report.target_revision,
        "entry_count": len(report.entries),
        "ledger_digest": report.ledger_digest,
        "table_count": table_count,
        "non_database_paths": files,
    } == expected
    assert table_count == len(GRAPH_TABLE_NAMES)


@pytest.mark.parametrize(
    "interruption",
    (
        "after_backup",
        "after_schema",
        "after_unchanged_copy",
        "after_jobs",
        "after_followups",
        "after_validation",
        "before_publication",
        "after_state_publication",
    ),
)
def test_interrupted_publication_restores_exact_revision_one_state(
    tmp_path: Path, interruption: str
):
    active = tmp_path / "active"
    _create_revision_1_golden(active)
    original_database = _sha256(active / "state.db")
    original_files = {
        path.relative_to(active).as_posix(): path.read_bytes()
        for path in active.rglob("*")
        if path.is_file()
    }

    def interrupt(phase: str) -> None:
        if phase == interruption:
            raise RuntimeError(f"interrupted at {phase}")

    with pytest.raises(RuntimeError, match="interrupted"):
        publish_revision_2_for_test(
            active,
            now=GRAPH_NOW,
            phase_hook=interrupt,
        )

    assert _sha256(active / "state.db") == original_database
    assert {
        path.relative_to(active).as_posix(): path.read_bytes()
        for path in active.rglob("*")
        if path.is_file()
    } == original_files
