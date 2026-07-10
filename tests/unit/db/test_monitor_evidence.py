from types import SimpleNamespace

from daita.db.runtime.extensions.monitor_evidence import (
    evidence_matches_dependency,
    load_monitor_proposal_evidence,
)
from daita.runtime import Evidence, Operation, Task, TaskDependency


class _EvidenceStore:
    def __init__(self, evidence):
        self.evidence = tuple(evidence)
        self.operation_ids = []

    async def list_evidence(self, operation_id):
        self.operation_ids.append(operation_id)
        return self.evidence


def _proposal(evidence_id, *, accepted=True, owner="db_runtime", task_id="plan"):
    return Evidence(
        id=evidence_id,
        kind="monitor.proposal",
        owner=owner,
        operation_id="operation-1",
        task_id=task_id,
        accepted=accepted,
        payload={"monitor_id": evidence_id},
    )


def _task(*dependencies):
    return Task(
        id="commit",
        operation_id="operation-1",
        capability_id="db.monitor.commit_create",
        executor_id="db_runtime.monitor.commit_create",
        dependencies=dependencies,
    )


def test_monitor_evidence_matches_complete_dependency_contract():
    evidence = _proposal("proposal-1", owner="planner", task_id="plan-1")
    dependency = TaskDependency(
        kind="evidence",
        evidence_kind="monitor.proposal",
        evidence_id="proposal-1",
        evidence_owner="planner",
        producer_task_id="plan-1",
        evidence_accepted=True,
        evidence_payload={"monitor_id": "proposal-1"},
    )

    assert evidence_matches_dependency(evidence, dependency) is True
    assert (
        evidence_matches_dependency(
            evidence,
            TaskDependency(
                kind="evidence",
                evidence_kind="monitor.proposal",
                evidence_id="other",
            ),
        )
        is False
    )
    assert (
        evidence_matches_dependency(
            evidence,
            TaskDependency(
                kind="evidence",
                evidence_kind="monitor.proposal",
                evidence_payload={"monitor_id": "other"},
            ),
        )
        is False
    )


async def test_monitor_proposal_loader_prefers_explicit_reference():
    explicit = _proposal("explicit", accepted=False)
    fallback = _proposal("fallback")
    store = _EvidenceStore((explicit, fallback))
    runtime = SimpleNamespace(store=store)
    operation = Operation(id="operation-1", operation_type="db.run")

    loaded = await load_monitor_proposal_evidence(
        runtime, operation, _task(), "explicit"
    )

    assert loaded is explicit
    assert store.operation_ids == ["operation-1"]


async def test_monitor_proposal_loader_preserves_dependency_and_evidence_order():
    first_old = _proposal("first-old", owner="first", task_id="plan-1")
    second = _proposal("second", owner="second", task_id="plan-2")
    first_latest = _proposal("first-latest", owner="first", task_id="plan-1")
    store = _EvidenceStore((first_old, second, first_latest))
    runtime = SimpleNamespace(store=store)
    operation = Operation(id="operation-1", operation_type="db.run")
    task = _task(
        TaskDependency(
            kind="evidence",
            evidence_kind="monitor.proposal",
            evidence_owner="first",
            producer_task_id="plan-1",
        ),
        TaskDependency(
            kind="evidence",
            evidence_kind="monitor.proposal",
            evidence_owner="second",
            producer_task_id="plan-2",
        ),
    )

    loaded = await load_monitor_proposal_evidence(runtime, operation, task, None)

    assert loaded is first_latest
    assert store.operation_ids == ["operation-1"]


async def test_monitor_proposal_loader_uses_latest_accepted_fallback_or_none():
    rejected = _proposal("rejected", accepted=False)
    accepted = _proposal("accepted")
    latest = _proposal("latest")
    store = _EvidenceStore((accepted, rejected, latest))
    runtime = SimpleNamespace(store=store)
    operation = Operation(id="operation-1", operation_type="db.run")

    assert (
        await load_monitor_proposal_evidence(runtime, operation, _task(), "missing")
        is latest
    )
    assert store.operation_ids == ["operation-1", "operation-1"]

    empty_runtime = SimpleNamespace(store=_EvidenceStore((rejected,)))
    assert (
        await load_monitor_proposal_evidence(empty_runtime, operation, _task(), None)
        is None
    )
