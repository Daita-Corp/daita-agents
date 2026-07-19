from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita.events.models import RuntimeEvent
from daita.identity import AgentIdentity
from daita.learning import (
    LearningCandidateCategory,
    LearningDecision,
    LearningProposal,
    LearningProposalKind,
    LearningProposalState,
    LearningProvenance,
    LearningSourceOutcome,
)
from daita.loop.models import LoopBudgets, LoopPhase, LoopState
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)
from daita.sessions import Session
from daita.skills import (
    Skill,
    SkillActivation,
    SkillActivationMode,
    SkillChangeCommit,
    SkillChangeConflictError,
    SkillIndex,
    SkillSource,
    SkillVersion,
)
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
AGENT_ID = "agent-skill-change"
SESSION_ID = "session-skill-change"
SKILL_ID = "skill:reconcile-customers"


def _operation(number: int) -> OperationSnapshot:
    created_at = NOW + timedelta(minutes=number)
    trigger = AgentTrigger(
        id=f"trigger-skill-{number}",
        agent_id=AGENT_ID,
        kind=TriggerKind.USER,
        source_id="user:owner",
        session_id=SESSION_ID,
        payload={"message": f"Propose reviewed skill version {number}."},
        created_at=created_at,
    )
    operation = Operation(
        id=f"operation-skill-{number}",
        agent_id=AGENT_ID,
        trigger_id=trigger.id,
        status=OperationStatus.SUCCEEDED,
        session_id=SESSION_ID,
        final_text=f"Reviewed skill source {number} completed.",
        terminal_reason="completed",
        created_at=created_at,
        updated_at=created_at,
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(phase=LoopPhase.TERMINAL),
        budgets=LoopBudgets(),
        turns=(),
        model_calls=(),
        readiness=(),
        tasks=(),
        evidence=(),
        observations=(),
        events=(
            RuntimeEvent(
                id=f"event-skill-{number}",
                type="operation.completed",
                agent_id=AGENT_ID,
                operation_id=operation.id,
                session_id=SESSION_ID,
                payload={"status": "succeeded"},
                created_at=created_at,
            ),
        ),
    )


async def _open_store(
    path: Path,
    *,
    operations: int,
) -> SQLiteOperationStore:
    store = await SQLiteOperationStore.open(path)
    await store.initialize_identity(
        AgentIdentity(
            id=AGENT_ID,
            display_name="Skill Change Contract",
            created_at=NOW,
        )
    )
    await store.create_session(
        Session(
            id=SESSION_ID,
            agent_id=AGENT_ID,
            title="Skill change contract",
            created_at=NOW,
            updated_at=NOW,
        )
    )
    for number in range(1, operations + 1):
        await store.create(_operation(number))
    return store


def _candidate(version: str, instructions: str) -> dict[str, object]:
    return {
        "activation_mode": "always",
        "description": "Reconcile customer records from bounded evidence.",
        "domains": ["data"],
        "instructions": instructions,
        "policy_notes": None,
        "required_capability_ids": ["data.sqlite.query"],
        "resource_kinds": ["table"],
        "sensitivity_notes": "Do not expose raw rows.",
        "stable_name": "reconcile-customers",
        "version": version,
    }


def _proposal(
    number: int,
    *,
    version: str,
    instructions: str,
) -> LearningProposal:
    return LearningProposal.create(
        proposal_id=f"proposal-skill-{number}",
        kind=LearningProposalKind.SKILL,
        category=LearningCandidateCategory.SKILL_CHANGE,
        provenance=LearningProvenance(
            agent_id=AGENT_ID,
            operation_id=f"operation-skill-{number}",
            trigger_id=f"trigger-skill-{number}",
            source_outcome=LearningSourceOutcome.SUCCEEDED,
            source_hash="sha256:" + f"{number:064x}",
        ),
        candidate=_candidate(version, instructions),
        created_at=NOW + timedelta(minutes=number, seconds=1),
    )


def _commit(
    proposal: LearningProposal,
    *,
    version: str,
    instructions: str,
    expected_active_version_id: str | None,
    expected_skill_version_count: int,
) -> SkillChangeCommit:
    number = int(proposal.id.rsplit("-", 1)[1])
    skill = Skill(
        id=SKILL_ID,
        agent_id=AGENT_ID,
        stable_name="reconcile-customers",
        source=SkillSource.LEARNED_PROPOSAL,
        created_at=NOW + timedelta(minutes=1, seconds=1),
    )
    skill_version = SkillVersion(
        id=f"skill-version:{proposal.candidate_hash.removeprefix('sha256:')}",
        agent_id=AGENT_ID,
        skill_id=SKILL_ID,
        stable_name="reconcile-customers",
        version=version,
        description="Reconcile customer records from bounded evidence.",
        domains=("data",),
        resource_kinds=("table",),
        required_capability_ids=("data.sqlite.query",),
        activation_mode=SkillActivationMode.ALWAYS,
        sensitivity_notes="Do not expose raw rows.",
        policy_notes=None,
        source=SkillSource.LEARNED_PROPOSAL,
        content_hash=proposal.candidate_hash,
        instructions=instructions,
        source_path=None,
        created_at=proposal.created_at,
    )
    activated_at = proposal.created_at + timedelta(seconds=1)
    activation = SkillActivation(
        id=f"skill-change-activation-{number}",
        agent_id=AGENT_ID,
        skill_id=SKILL_ID,
        version_id=skill_version.id,
        previous_version_id=expected_active_version_id,
        actor_id="user:owner",
        reason=f"Explicitly accept reviewed skill proposal {number}.",
        activated_at=activated_at,
    )
    decision = LearningDecision(
        proposal_id=proposal.id,
        idempotency_key=proposal.idempotency_key,
        candidate_hash=proposal.candidate_hash,
        state=LearningProposalState.COMMITTED,
        decided_at=activated_at,
        result_skill_id=SKILL_ID,
        result_skill_version=expected_skill_version_count + 1,
    )
    return SkillChangeCommit(
        proposal=proposal,
        skill=skill,
        version=skill_version,
        staged_index=SkillIndex.from_version(
            skill_version,
            active_version_id=expected_active_version_id,
        ),
        activation=activation,
        decision=decision,
        expected_active_version_id=expected_active_version_id,
        expected_skill_version_count=expected_skill_version_count,
    )


async def test_sqlite_skill_change_commit_replays_and_reopens_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await _open_store(path, operations=1)
    proposal = _proposal(
        1,
        version="1.0.0",
        instructions="Use accepted evidence and cite every result.",
    )
    assert proposal.state is LearningProposalState.PROPOSED
    assert await store.create_proposal(proposal) == proposal
    request = _commit(
        proposal,
        version="1.0.0",
        instructions="Use accepted evidence and cite every result.",
        expected_active_version_id=None,
        expected_skill_version_count=0,
    )

    accepted = await store.commit_skill_change(request)
    replay = await store.commit_skill_change(request)

    assert accepted.replayed is False
    assert accepted.proposal.state is LearningProposalState.COMMITTED
    assert accepted.proposal.result_skill_id == SKILL_ID
    assert accepted.proposal.result_skill_version == 1
    assert accepted.inspection.index.active_version_id == request.version.id
    assert accepted.inspection.versions == (request.version,)
    assert accepted.inspection.activations == (request.activation,)
    assert replay.replayed is True
    assert replay.proposal == accepted.proposal
    assert replay.inspection == accepted.inspection
    await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        durable_proposal = await reopened.load_proposal(AGENT_ID, proposal.id)
        durable_skill = await reopened.inspect_skill(AGENT_ID, SKILL_ID)
    finally:
        await reopened.close()
    assert durable_proposal is not None
    assert durable_skill is not None
    assert durable_proposal == accepted.proposal
    assert durable_skill == accepted.inspection


async def test_sqlite_skill_change_cas_conflicts_roll_back_across_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await _open_store(path, operations=3)
    first = _proposal(
        1,
        version="1.0.0",
        instructions="Use accepted evidence and cite every result.",
    )
    stale_pointer = _proposal(
        2,
        version="2.0.0",
        instructions="Compare accepted evidence before citing each result.",
    )
    stale_count = _proposal(
        3,
        version="3.0.0",
        instructions="Inspect, compare, and cite accepted evidence.",
    )
    for proposal in (first, stale_pointer, stale_count):
        await store.create_proposal(proposal)
    first_request = _commit(
        first,
        version="1.0.0",
        instructions="Use accepted evidence and cite every result.",
        expected_active_version_id=None,
        expected_skill_version_count=0,
    )
    first_result = await store.commit_skill_change(first_request)

    pointer_request = _commit(
        stale_pointer,
        version="2.0.0",
        instructions="Compare accepted evidence before citing each result.",
        expected_active_version_id=None,
        expected_skill_version_count=1,
    )
    with pytest.raises(SkillChangeConflictError):
        await store.commit_skill_change(pointer_request)

    count_request = _commit(
        stale_count,
        version="3.0.0",
        instructions="Inspect, compare, and cite accepted evidence.",
        expected_active_version_id=first_request.version.id,
        expected_skill_version_count=2,
    )
    with pytest.raises(SkillChangeConflictError):
        await store.commit_skill_change(count_request)
    await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        durable = await reopened.inspect_skill(AGENT_ID, SKILL_ID)
        pointer_proposal = await reopened.load_proposal(
            AGENT_ID,
            stale_pointer.id,
        )
        count_proposal = await reopened.load_proposal(AGENT_ID, stale_count.id)
        missing_pointer_version = await reopened.load_skill_version(
            AGENT_ID,
            pointer_request.version.id,
        )
        missing_count_version = await reopened.load_skill_version(
            AGENT_ID,
            count_request.version.id,
        )
    finally:
        await reopened.close()

    assert durable is not None
    assert durable == first_result.inspection
    assert len(durable.versions) == 1
    assert len(durable.activations) == 1
    assert pointer_proposal is not None
    assert pointer_proposal.state is LearningProposalState.PROPOSED
    assert count_proposal is not None
    assert count_proposal.state is LearningProposalState.PROPOSED
    assert missing_pointer_version is None
    assert missing_count_version is None
