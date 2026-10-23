"""Revision 4 turns graph jobs into checklist jobs and drops the task graph."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from daita import Agent
from daita.distribution import DeliverySubjectKind
from daita.distribution.models import (
    CONVERSATION_INBOX_DESTINATION_REVISION,
    ConversationInboxTarget,
    DeliveryState,
    GraphJobDelivery,
    OutcomeConclusionKind,
    OutcomeReference,
    OutcomeState,
    conversation_inbox_destination_id,
    graph_job_delivery_key,
    target_fingerprint,
)
from daita.distribution.owner import construct_logical_delivery
from daita.jobs import JobState
from daita.jobs.graph.models import GraphState
from daita.jobs.models import UPGRADED_JOB_SUMMARY
from daita.llm.models import ModelSensitivity
from daita.storage.home_migrations.revision_0004 import REVISION_4
from daita.storage.sqlite_codecs.distribution import encode_delivery
from daita.storage.sqlite_codecs.graph import (
    encode_graph_job,
    encode_graph_job_delivery,
    encode_task_result,
)
from daita.storage.sqlite_codecs.receipts import encode_receipt
from daita.storage.sqlite_graph import admit_graph, datetime_to_us
from daita.storage.sqlite_records import EffectReceipt, effect_receipt_id
from tests.support.graph import GRAPH_NOW, graph_admission, task_result
from tests.support.home_revisions import undo_revision_4_schema
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.integration

NAME = "legacy-jobs"
CONVERSATION = "conversation-1"
DIGEST = "sha256:" + "4" * 64


def _target() -> ConversationInboxTarget:
    destination = conversation_inbox_destination_id(CONVERSATION)
    return ConversationInboxTarget(
        conversation_id=CONVERSATION,
        destination_id=destination,
        destination_revision=CONVERSATION_INBOX_DESTINATION_REVISION,
        sensitivity_ceiling=ModelSensitivity.RESTRICTED,
        target_fingerprint=target_fingerprint(
            conversation_id=CONVERSATION,
            destination_id=destination,
            destination_revision=CONVERSATION_INBOX_DESTINATION_REVISION,
            sensitivity_ceiling=ModelSensitivity.RESTRICTED,
        ),
    )


def _insert_delivery(connection: sqlite3.Connection, **row: object) -> None:
    connection.execute(
        "INSERT INTO deliveries(agent_id, delivery_id, conversation_id, "
        "subject_kind, subject_id, logical_key, target_kind, target_fingerprint, "
        "state, created_at_us, data) VALUES (:agent_id, :delivery_id, "
        ":conversation_id, :subject_kind, :subject_id, :logical_key, "
        "'conversation_inbox', :target_fingerprint, :state, :created_at_us, :data)",
        row,
    )


def _seed_revision_3_graph(home: Path, agent_id: str) -> str:
    """A released revision-3 home with a finished, a failed and a queued graph job."""

    result = task_result(job_id="job-done", agent_id=agent_id)
    terminal_at = GRAPH_NOW + timedelta(seconds=3)
    with sqlite3.connect(home / "state.db") as connection:
        undo_revision_4_schema(connection)
        connection.execute("DELETE FROM agent_home_migrations WHERE revision = 4")
        jobs = {
            job_id: admit_graph(
                connection, graph_admission(job_id=job_id, agent_id=agent_id)
            )
            for job_id in ("job-done", "job-failed", "job-queued")
        }
        connection.execute(
            "INSERT INTO job_task_attempts(agent_id, job_id, task_id, attempt_id, "
            "ordinal, fencing_epoch, state, claim_token, run_id, "
            "absolute_deadline_at_us, ended_at_us, data) "
            "VALUES (?, 'job-done', 'worker', 'attempt-1', 1, 1, 'succeeded', "
            "'claim', 'run-1', ?, ?, '{}')",
            (agent_id, datetime_to_us(terminal_at), datetime_to_us(terminal_at)),
        )
        connection.execute(
            "INSERT INTO job_task_results(agent_id, job_id, task_id, result_id, "
            "attempt_id, completed_at_us, sensitivity, result_digest, data) "
            "VALUES (?, 'job-done', 'worker', ?, 'attempt-1', ?, 'restricted', ?, ?)",
            (
                agent_id,
                result.result_id,
                datetime_to_us(result.completed_at),
                result.result_digest,
                encode_task_result(result),
            ),
        )
        finished = {
            "job-done": replace(
                jobs["job-done"],
                state=GraphState.SUCCEEDED,
                terminal_at=terminal_at,
                terminal_result_id=result.result_id,
                updated_at=terminal_at,
            ),
            "job-failed": replace(
                jobs["job-failed"],
                state=GraphState.FAILED,
                terminal_at=terminal_at,
                failure_code="worker_failed",
                updated_at=terminal_at,
            ),
        }
        for job_id, job in finished.items():
            connection.execute(
                "UPDATE job_runs SET state = ?, terminal_at_us = ?, updated_at_us = ?, "
                "data = ? WHERE job_id = ?",
                (
                    job.state.value,
                    datetime_to_us(terminal_at),
                    datetime_to_us(terminal_at),
                    encode_graph_job(job),
                    job_id,
                ),
            )
        target = _target()
        outcome = OutcomeReference(
            conclusion_kind=OutcomeConclusionKind.NO_MODEL_OCCURRENCE,
            conclusion_state=OutcomeState.SUCCEEDED,
            conclusion_id=result.result_id,
            conclusion_digest=result.result_digest,
            conclusion_preview=result.summary,
            conclusion_preview_truncated=False,
            resulting_run_id=None,
            artifact_references=(),
            effective_sensitivity=ModelSensitivity.RESTRICTED,
            provenance_digest=DIGEST,
            failure_code=None,
            observed_at=terminal_at,
        )
        graph_delivery = GraphJobDelivery(
            delivery_id="delivery-done",
            agent_id=agent_id,
            conversation_id=CONVERSATION,
            job_id="job-done",
            logical_key=graph_job_delivery_key(
                job_id="job-done", outcome_digest=result.result_digest
            ),
            target=target,
            outcome=outcome,
            visibility_state=DeliveryState.AVAILABLE,
            blocked_reason_code=None,
            created_at=terminal_at,
            updated_at=terminal_at,
        )
        _insert_delivery(
            connection,
            agent_id=agent_id,
            delivery_id="delivery-done",
            conversation_id=CONVERSATION,
            subject_kind="graph_job",
            subject_id="job-done",
            logical_key=graph_delivery.logical_key,
            target_fingerprint=target.target_fingerprint,
            state="available",
            created_at_us=datetime_to_us(terminal_at),
            data=encode_graph_job_delivery(graph_delivery),
        )
        notice = construct_logical_delivery(
            delivery_id="delivery-notice",
            agent_id=agent_id,
            conversation_id=CONVERSATION,
            subject_kind=DeliverySubjectKind.JOB,
            subject_id="job-queued/event/1/control_opened/control-1",
            target=target,
            conclusion_kind=OutcomeConclusionKind.NO_MODEL_OCCURRENCE,
            conclusion_state=OutcomeState.SUCCEEDED,
            conclusion_id="job-queued/event/1/control_opened/control-1",
            conclusion_digest=DIGEST,
            conclusion_preview="A task needs your input.",
            conclusion_preview_truncated=False,
            resulting_run_id=None,
            artifact_references=(),
            effective_sensitivity=ModelSensitivity.RESTRICTED,
            provenance_digest=DIGEST,
            failure_code=None,
            observed_at=GRAPH_NOW,
        )
        _insert_delivery(
            connection,
            agent_id=agent_id,
            delivery_id="delivery-notice",
            conversation_id=CONVERSATION,
            subject_kind="graph_attention",
            subject_id=notice.subject_id,
            logical_key="delivery:" + DIGEST,
            target_fingerprint=target.target_fingerprint,
            state="available",
            created_at_us=datetime_to_us(GRAPH_NOW),
            data=encode_delivery(notice).replace('"job"', '"graph_attention"', 1),
        )
        receipt = EffectReceipt(
            receipt_id=effect_receipt_id(
                agent_id=agent_id,
                run_id="run-1",
                call_id="call-1",
                operation_key=DIGEST,
            ),
            receipt_kind="test.effect",
            agent_id=agent_id,
            run_id="run-1",
            call_id="call-1",
            capability_id="test.effect",
            domain_owner_id="test",
            capability_contract_digest=DIGEST,
            operation_key=DIGEST,
            argument_fingerprint=DIGEST,
            sensitivity=ModelSensitivity.INTERNAL,
            started_at=GRAPH_NOW,
        )
        connection.execute(
            "INSERT INTO effect_receipts(agent_id, id, run_id, call_id, operation_key, "
            "unresolved, job_id, task_id, task_attempt_id, fencing_epoch, "
            "task_spec_digest, data) VALUES (?, ?, 'run-1', 'call-1', ?, 1, "
            "'job-done', 'worker', 'attempt-1', 1, ?, ?)",
            (agent_id, receipt.receipt_id, DIGEST, DIGEST, encode_receipt(receipt)),
        )
    return receipt.receipt_id


async def test_graph_jobs_become_checklist_jobs_and_the_graph_tables_go(
    tmp_path: Path,
) -> None:
    created = await Agent.create(NAME, root=tmp_path, workspace=workspace_for(tmp_path))
    home, agent_id = created.home, created.id
    await created.close()
    receipt_id = _seed_revision_3_graph(home, agent_id)
    before = await Agent.inspect_home(NAME, root=tmp_path)
    assert before.found_revision == 3 and before.upgrade_required

    agent = await Agent.open(NAME, root=tmp_path, workspace=workspace_for(tmp_path))
    try:
        jobs = {job.job_id: job for job in await agent.list_jobs()}
        done, failed, queued = jobs["job-done"], jobs["job-failed"], jobs["job-queued"]
        # A finished job keeps its accepted result, so its history stays readable.
        assert done.state is JobState.DONE
        assert done.final_summary == "Accepted deterministic result."
        assert done.objective == "Complete the deterministic graph test."
        assert done.conversation_id == CONVERSATION
        assert failed.state is JobState.FAILED
        assert failed.failure_code == "worker_failed"
        # Unfinished work cannot continue: it fails with the upgrade message.
        assert queued.state is JobState.FAILED
        assert queued.failure_code == "job_upgraded"
        assert queued.final_summary == UPGRADED_JOB_SUMMARY
        assert all(job.checklist == () for job in jobs.values())

        (delivery,) = await agent.list_deliveries(include_acknowledged=True)
        assert delivery.subject_kind is DeliverySubjectKind.JOB
        assert delivery.subject_id == "job-done"
        assert delivery.conclusion_preview == "Accepted deterministic result."
        receipt = await agent.inspect_effect(receipt_id)
        assert receipt is not None
    finally:
        await agent.close()

    with sqlite3.connect(home / "state.db") as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert not {name for name in tables if name.startswith("job_")}
        assert connection.execute(
            "SELECT job_id FROM effect_receipts WHERE id = ?", (receipt_id,)
        ).fetchone() == ("job-done",)
        (journal,) = connection.execute(
            "SELECT migration_id, checksum FROM agent_home_migrations WHERE revision = 4"
        )
        assert journal == (REVISION_4.migration_id, REVISION_4.checksum)
        (encoded,) = connection.execute(
            "SELECT data FROM jobs WHERE job_id = 'job-done'"
        ).fetchone()
        assert json.loads(encoded)["fields"]["sensitivity"] == "restricted"
