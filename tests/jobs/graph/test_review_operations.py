"""Phase 7 review-task and attention-delivery invariants."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from daita.distribution.models import DeliverySubjectKind
from daita.jobs.graph.capabilities import REVIEW_CAPABILITY_IDS
from daita.jobs.graph.models import (
    BudgetAmount,
    ControlKind,
    ControlState,
    GraphState,
    TaskControl,
    TaskRole,
    TaskState,
    canonical_digest,
)
from daita.jobs.owner import JobError, JobOwner
from daita.llm.models import FinishReason, ModelResponse, ModelUsage, ToolCall
from daita.llm.pricing import CostEstimate
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_graph import GraphStoreConflictError
from tests.support.graph import GRAPH_NOW, graph_admission
from tests.support.model_graph_integration import AGENT_ID, ModelGraphIntegration
from tests.support.static_graph_integration import DeterministicIds


def _response(call_id: str, name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0))),
    )


def _review_script(*, decision: str = "accept") -> tuple[ModelResponse, ...]:
    responses = [
        _response(
            "read-call",
            "graph_read",
            {"resource_ids": tuple(f"resource-{index}" for index in range(5))},
        ),
        _response(
            "review-request",
            "task_request_review",
            {
                "result_kind": "test.graph.read",
                "summary": "Candidate awaiting an independent review.",
                "payload": {"value": 42},
                "evidence_call_ids": ("read-call",),
                "artifact_ids": (),
                "residual_risk": None,
                "downstream_constraints": {"review_required": True},
                "message": "Check the authenticated read candidate.",
            },
        ),
        _response("review-inspect", "review_inspect_candidate", {}),
    ]
    if decision == "accept":
        responses.append(
            _response(
                "review-accept",
                "review_accept",
                {"rationale": "The candidate matches its authenticated evidence."},
            )
        )
    else:
        responses.append(
            _response(
                "review-changes",
                "review_request_changes",
                {
                    "rationale": "The candidate needs a bounded replacement.",
                    "replacement_guidance": "Repeat within the unchanged task authority.",
                },
            )
        )
    return tuple(responses)


async def _wait_for_review(integration: ModelGraphIntegration, job_id: str):
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        inspection = await integration.owner.inspect_graph(job_id)
        assert inspection is not None
        review = next(
            (
                item
                for item in inspection.controls
                if item.kind is ControlKind.REVIEW_REQUESTED
            ),
            None,
        )
        if review is not None:
            return inspection, review
        await asyncio.sleep(0.005)
    raise AssertionError("review control was not opened")


async def test_separate_reviewer_accepts_exact_candidate_once(tmp_path: Path) -> None:
    integration = await ModelGraphIntegration.open(
        tmp_path, script=_review_script(), namespace="review-accept"
    )
    try:
        admission = integration.build()
        worker_before = next(
            task for task in admission.tasks if task.role is TaskRole.WORKER
        )
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)

        assert terminal.job.state is GraphState.SUCCEEDED
        worker = next(task for task in terminal.tasks if task.role is TaskRole.WORKER)
        reviewer = next(
            task for task in terminal.tasks if task.role is TaskRole.REVIEWER
        )
        assert worker.state is TaskState.SUCCEEDED
        assert reviewer.state is TaskState.SUCCEEDED
        assert worker.specification.authority == worker_before.specification.authority
        assert reviewer.specification.authority.capability_ids == tuple(
            sorted(REVIEW_CAPABILITY_IDS)
        )
        assert reviewer.specification.authority.source_ids == ()
        assert reviewer.specification.authority.resource_ids == ()
        assert reviewer.specification.authority.operational_effects == ("none",)

        review = next(
            item
            for item in terminal.controls
            if item.kind is ControlKind.REVIEW_REQUESTED
        )
        accepted = tuple(
            item for item in terminal.results if item.task_id == worker.task_id
        )
        assert len(accepted) == 1
        assert review.state is ControlState.RESOLVED
        candidate = review.payload["candidate"]
        assert isinstance(candidate, Mapping)
        assert dict(candidate) == accepted[0].candidate_material()
        assert review.payload["candidate_digest"] == accepted[0].result_digest
        assert review.resolution is not None
        assert review.resolution["decision"] == "accepted"
        assert (
            len(
                tuple(
                    item
                    for item in terminal.results
                    if item.task_id == reviewer.task_id
                )
            )
            == 1
        )

        attention = await integration.store.list_deliveries(AGENT_ID, limit=10)
        assert attention == ()
    finally:
        await integration.close()


async def test_review_changes_preserve_candidate_and_emit_one_new_attention(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(
        tmp_path, script=_review_script(decision="changes"), namespace="review-changes"
    )
    try:
        admission = integration.build()
        await integration.admit_and_start(admission)
        terminal = await integration.wait_terminal(admission.job.job_id)
        worker = next(task for task in terminal.tasks if task.role is TaskRole.WORKER)
        review = next(
            item
            for item in terminal.controls
            if item.kind is ControlKind.REVIEW_REQUESTED
        )
        changes = next(
            item
            for item in terminal.controls
            if item.kind is ControlKind.CHANGES_REQUESTED
        )

        assert terminal.job.state is GraphState.BLOCKED
        assert worker.state is TaskState.BLOCKED
        assert not any(item.task_id == worker.task_id for item in terminal.results)
        assert review.state is ControlState.REJECTED
        assert changes.state is ControlState.OPEN
        assert review.payload["candidate_digest"] == changes.payload["candidate_digest"]
        candidate_before = review.payload["candidate"]

        reviewer = next(
            task for task in terminal.tasks if task.role is TaskRole.REVIEWER
        )
        replay = await integration.owner.request_graph_task_review_changes(
            admission.job.job_id,
            worker.task_id,
            review.control_id,
            reviewer_task_id=reviewer.task_id,
            rationale="The candidate needs a bounded replacement.",
            replacement_guidance="Repeat within the unchanged task authority.",
            idempotency_key="review-changes",
            changes_control_id="ignored-on-replay",
            resolved_by_kind="reviewer_attempt",
            resolved_by_id=next(
                attempt.attempt_id
                for attempt in terminal.attempts
                if attempt.task_id == reviewer.task_id
            ),
        )
        assert replay == changes
        current = await integration.owner.inspect_graph(admission.job.job_id)
        assert current is not None
        persisted_review = next(
            item for item in current.controls if item.control_id == review.control_id
        )
        assert persisted_review.payload["candidate"] == candidate_before
        await integration.supervisor.close()
        mutation = await integration.owner.replace_graph_task_by_policy(
            admission.job.job_id,
            worker.task_id,
            principal_id=admission.job.specification.principal_id,
            advisory_note="Use the reviewer guidance without changing authority.",
            idempotency_key="review-policy-replacement",
            expected_revision=current.graph.revision,
        )
        replaced = await integration.owner.inspect_graph(admission.job.job_id)
        assert replaced is not None
        old = next(item for item in replaced.tasks if item.task_id == worker.task_id)
        replacement = next(
            item
            for item in replaced.tasks
            if item.task_id in mutation.resulting_task_ids
        )
        resolved_changes = next(
            item for item in replaced.controls if item.control_id == changes.control_id
        )
        assert old.state is TaskState.SUPERSEDED
        assert replacement.specification.authority == worker.specification.authority
        assert replacement.supersedes_task_id == worker.task_id
        assert resolved_changes.state is ControlState.RESOLVED
        assert (
            resolved_changes.resolved_by_id == admission.job.specification.principal_id
        )
        attention = await integration.store.list_deliveries(AGENT_ID, limit=10)
        assert [item.subject_kind for item in attention] == [
            DeliverySubjectKind.GRAPH_ATTENTION
        ]
        assert changes.control_id in attention[0].subject_id
    finally:
        await integration.close()


async def test_human_review_resolution_is_fenced_and_restart_safe(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(
        tmp_path,
        script=_review_script()[:2],
        namespace="review-restart",
    )
    admission = integration.build()
    await integration.admit_and_start(admission)
    _inspection, review = await _wait_for_review(integration, admission.job.job_id)
    await integration.close()

    store = await SQLiteStateStore.open(tmp_path / "state.sqlite")
    owner = JobOwner(
        agent_id=AGENT_ID,
        store=store,
        clock=lambda: datetime.now(UTC),
        id_factory=DeterministicIds("reopened-review"),
    )
    try:
        current = await owner.inspect_graph(admission.job.job_id)
        assert current is not None
        subject = next(task for task in current.tasks if task.task_id == review.task_id)
        reviewer = next(
            task for task in current.tasks if task.role is TaskRole.REVIEWER
        )
        with pytest.raises(JobError, match="typed review decision"):
            await owner.replace_graph_task_by_policy(
                admission.job.job_id,
                subject.task_id,
                principal_id=admission.job.specification.principal_id,
                advisory_note="Attempt to bypass the exact review candidate.",
                idempotency_key="review-bypass",
                expected_revision=current.graph.revision,
            )
        with pytest.raises(GraphStoreConflictError, match="review fence is stale"):
            await store.accept_graph_review(
                agent_id=AGENT_ID,
                job_id=admission.job.job_id,
                subject_task_id=subject.task_id,
                control_id=review.control_id,
                reviewer_task_id=reviewer.task_id,
                resolved_at=datetime.now(UTC),
                resolved_by_kind="principal",
                resolved_by_id=admission.job.specification.principal_id,
                rationale="Exact principal review.",
                idempotency_key="human-review-once",
                expected_control_digest=review.payload_digest,
                expected_subject_revision=subject.task_revision + 1,
            )

        accepted = await owner.accept_graph_task_review_by_principal(
            admission.job.job_id,
            subject.task_id,
            review.control_id,
            principal_id=admission.job.specification.principal_id,
            rationale="Exact principal review.",
            idempotency_key="human-review-once",
        )
        replay = await owner.accept_graph_task_review_by_principal(
            admission.job.job_id,
            subject.task_id,
            review.control_id,
            principal_id=admission.job.specification.principal_id,
            rationale="Exact principal review.",
            idempotency_key="human-review-once",
        )
        assert replay == accepted
        settled = await owner.inspect_graph(admission.job.job_id)
        assert settled is not None
        assert (
            len(
                tuple(
                    item for item in settled.results if item.task_id == subject.task_id
                )
            )
            == 1
        )
        assert (
            next(
                item
                for item in settled.controls
                if item.control_id == review.control_id
            ).resolved_by_id
            == admission.job.specification.principal_id
        )
        assert (
            subject.specification.authority
            == next(
                item for item in settled.tasks if item.task_id == subject.task_id
            ).specification.authority
        )
    finally:
        await store.close()


async def test_authorized_new_job_outcome_is_atomic_idempotent_and_separate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replacement.sqlite"
    store = await SQLiteStateStore.open(path)
    owner = JobOwner(
        agent_id="agent-1",
        store=store,
        clock=lambda: GRAPH_NOW,
        id_factory=DeterministicIds("replacement-job"),
    )
    original = graph_admission()
    await owner.admit(original)
    attempt = await store.claim_graph_task(
        "agent-1",
        "job-1",
        "worker",
        attempt_id="authorization-attempt",
        claim_token="authorization-claim",
        run_id="authorization-run",
        executor_id="executor-1",
        claimed_at=GRAPH_NOW,
        lease_seconds=30,
        absolute_deadline_at=original.job.deadline_at,
        budget_reservations=(BudgetAmount("work_units", 1),),
    )
    assert attempt is not None
    started = await store.start_graph_attempt(
        "agent-1",
        "job-1",
        "worker",
        "authorization-attempt",
        claim_token="authorization-claim",
        fencing_epoch=attempt.fencing_epoch,
        started_at=GRAPH_NOW,
    )
    assert started is not None
    payload = {
        "message": "A separately authorized job is required.",
        "expires_at": original.job.deadline_at.isoformat(),
        "default_behavior": "remain_blocked",
    }
    control = TaskControl(
        agent_id="agent-1",
        job_id="job-1",
        task_id="worker",
        control_id="authorization-control",
        kind=ControlKind.NEEDS_AUTHORIZATION,
        state=ControlState.OPEN,
        requesting_attempt_id=started.attempt_id,
        payload=payload,
        created_at=GRAPH_NOW,
        payload_digest=canonical_digest(payload),
    )
    await store.open_graph_control(
        control,
        claim_token="authorization-claim",
        fencing_epoch=started.fencing_epoch,
    )
    original_before = await owner.inspect_graph("job-1")
    assert original_before is not None
    original_authority = original_before.job.specification.authority
    replacement = graph_admission(job_id="job-2")

    admitted = await owner.admit_authorized_replacement_graph(
        replacement,
        replaces_job_id="job-1",
        replaces_task_id="worker",
        control_id=control.control_id,
        principal_id="agent-1",
        idempotency_key="authorize-job-2",
    )
    assert admitted.job_id == "job-2"
    await store.close()

    reopened_store = await SQLiteStateStore.open(path)
    reopened_owner = JobOwner(
        agent_id="agent-1",
        store=reopened_store,
        clock=lambda: GRAPH_NOW,
        id_factory=DeterministicIds("replacement-job-reopened"),
    )
    try:
        replay = await reopened_owner.admit_authorized_replacement_graph(
            replacement,
            replaces_job_id="job-1",
            replaces_task_id="worker",
            control_id=control.control_id,
            principal_id="agent-1",
            idempotency_key="authorize-job-2",
        )
        assert replay.job_id == admitted.job_id
        old = await reopened_owner.inspect_graph("job-1")
        new = await reopened_owner.inspect_graph("job-2")
        assert old is not None and new is not None
        resolved = next(
            item for item in old.controls if item.control_id == control.control_id
        )
        assert resolved.state is ControlState.RESOLVED
        assert resolved.resolved_by_kind == "principal"
        assert resolved.resolved_by_id == "agent-1"
        assert resolved.resolution is not None
        assert dict(resolved.resolution) == {
            "action": "authorize_replacement_job",
            "replacement_job_id": "job-2",
            "idempotency_key": "authorize-job-2",
        }
        assert old.job.specification.authority == original_authority
        assert old.job.desired_state.value == "cancel"
        assert new.job.migration_provenance["replaces_job_id"] == "job-1"
        assert (
            new.job.specification.authority == replacement.job.specification.authority
        )
    finally:
        await reopened_store.close()
