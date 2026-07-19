from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3

import pytest

from daita._json import canonical_json
from daita.adapters.models import SourceRegistration
from daita.events.models import RuntimeEvent
from daita.identity import AgentIdentity
from daita.learning import (
    LearningCandidateCategory,
    LearningDecision,
    LearningProposal,
    LearningProposalKind,
    LearningProposalState,
    LearningProvenance,
    LearningRejectionCategory,
    LearningSourceOutcome,
    LearningStoreConflictError,
)
from daita.loop.models import LoopBudgets, LoopPhase, LoopState
from daita.llm.models import ModelProfile
from daita.llm.protocols import ModelProfileConflictError
from daita.memory.learning import (
    ExplicitCorrectionCommit,
    ExplicitCorrectionStoreConflictError,
)
from daita.memory.models import (
    MemoryCreator,
    MemoryKind,
    MemoryProvenance,
    MemoryProvenanceKind,
    MemoryRecord,
    MemoryRestoreRequest,
    MemoryScope,
    MemorySensitivity,
    MemorySnapshot,
    MemoryState,
    MemorySupersessionRequest,
    MemoryVersion,
)
from daita.memory.protocols import MemoryStoreConflictError
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)
from daita.sessions import Session, SessionCompressionCheckpoint
from daita.skills.models import (
    Skill,
    SkillActivation,
    SkillActivationMode,
    SkillIndex,
    SkillSource,
    SkillVersion,
)
from daita.skills.service import SkillActivationConflictError
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import (
    SQLiteCorruptionError,
    SQLiteOperationStore,
    SQLiteStoreError,
)

NOW = datetime(2026, 7, 18, 18, 0, tzinfo=timezone.utc)
AGENT_ID = "agent-phase5"
SESSION_ID = "session-phase5"
SHA_A = "sha256:" + ("a" * 64)
SHA_B = "sha256:" + ("b" * 64)


def _operation(number: int) -> OperationSnapshot:
    operation_id = f"operation-{number}"
    trigger = AgentTrigger(
        id=f"trigger-{number}",
        agent_id=AGENT_ID,
        kind=TriggerKind.USER,
        source_id="user-phase5",
        session_id=SESSION_ID,
        payload={"message": f"request {number}"},
        created_at=NOW + timedelta(minutes=number),
    )
    operation = Operation(
        id=operation_id,
        agent_id=AGENT_ID,
        trigger_id=trigger.id,
        session_id=SESSION_ID,
        status=OperationStatus.SUCCEEDED,
        final_text=f"answer {number}",
        terminal_reason="completed",
        created_at=trigger.created_at,
        updated_at=trigger.created_at,
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
                id=f"event-{number}",
                type="operation.completed",
                agent_id=AGENT_ID,
                operation_id=operation_id,
                session_id=SESSION_ID,
                payload={"status": "succeeded"},
                created_at=trigger.created_at,
            ),
        ),
    )


async def _open_home(
    path: Path,
    *,
    operations: int = 0,
    source: bool = False,
) -> tuple[SQLiteOperationStore, SourceRegistration | None]:
    store = await SQLiteOperationStore.open(path)
    await store.initialize_identity(
        AgentIdentity(id=AGENT_ID, display_name="Phase 5", created_at=NOW)
    )
    await store.create_session(
        Session(
            id=SESSION_ID,
            agent_id=AGENT_ID,
            title="Phase 5",
            created_at=NOW,
            updated_at=NOW,
        )
    )
    registration = None
    if source:
        registration = SourceRegistration.build(
            agent_id=AGENT_ID,
            adapter_id="sqlite",
            native_identity="phase5.db",
            display_name="Phase 5 DB",
            configuration={},
            attached_at=NOW,
        )
        await store.register_source(registration)
    for number in range(1, operations + 1):
        await store.create(_operation(number))
    return store, registration


def _import_memory(
    memory_id: str,
    *,
    scope: MemoryScope,
    logical_key: str,
    content: str,
    sensitivity: MemorySensitivity = MemorySensitivity.PUBLIC,
    created_at: datetime = NOW,
    expires_at: datetime | None = None,
) -> MemorySnapshot:
    record = MemoryRecord(
        id=memory_id,
        scope=scope,
        kind=MemoryKind.SEMANTIC_FACT,
        logical_key=logical_key,
        current_version=1,
        state=MemoryState.ACTIVE,
        created_at=created_at,
        updated_at=created_at,
    )
    version = MemoryVersion(
        memory_id=memory_id,
        version=1,
        content=content,
        creator=MemoryCreator.IMPORT,
        confidence=1.0,
        sensitivity=sensitivity,
        provenance=MemoryProvenance(
            kind=MemoryProvenanceKind.IMPORT,
            content_hash=SHA_A,
            external_ref=f"import:{memory_id}",
        ),
        created_at=created_at,
        expires_at=expires_at,
    )
    return MemorySnapshot(record, version)


def _proposal(
    operation_number: int,
    *,
    proposal_id: str,
    candidate: Mapping[str, object],
    created_at: datetime,
) -> LearningProposal:
    return LearningProposal.create(
        proposal_id=proposal_id,
        kind=LearningProposalKind.MEMORY,
        category=LearningCandidateCategory.EXPLICIT_CORRECTION,
        provenance=LearningProvenance(
            agent_id=AGENT_ID,
            operation_id=f"operation-{operation_number}",
            trigger_id=f"trigger-{operation_number}",
            source_outcome=LearningSourceOutcome.SUCCEEDED,
            source_hash=SHA_A if operation_number == 1 else SHA_B,
        ),
        candidate=candidate,
        created_at=created_at,
    )


def _alias_snapshot(
    *,
    source_id: str,
    operation_number: int,
    version: int,
    content: str,
    created_at: datetime,
    record: MemoryRecord | None = None,
) -> MemorySnapshot:
    scope = MemoryScope(
        agent_id=AGENT_ID,
        source_id=source_id,
        resource_id="resource-customers",
    )
    if record is None:
        record = MemoryRecord(
            id="memory-alias",
            scope=scope,
            kind=MemoryKind.RESOURCE_ALIAS,
            logical_key="status:completed",
            current_version=version,
            state=MemoryState.ACTIVE,
            created_at=created_at,
            updated_at=created_at,
        )
    else:
        record = replace(
            record,
            current_version=version,
            updated_at=created_at,
        )
    memory_version = MemoryVersion(
        memory_id=record.id,
        version=version,
        content=content,
        creator=MemoryCreator.LEARNING_SERVICE,
        confidence=1.0,
        sensitivity=MemorySensitivity.INTERNAL,
        provenance=MemoryProvenance(
            kind=MemoryProvenanceKind.USER_STATEMENT,
            content_hash=SHA_A if operation_number == 1 else SHA_B,
            operation_id=f"operation-{operation_number}",
            trigger_id=f"trigger-{operation_number}",
            session_id=SESSION_ID,
        ),
        attributes={"business_term": "completed", "stored_value": content},
        resource_revision=SHA_A,
        supersedes_version=None if version == 1 else version - 1,
        created_at=created_at,
    )
    return MemorySnapshot(record, memory_version)


async def test_migration_nine_and_session_compression_are_versioned_and_exact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "phase5.db"
    store, _ = await _open_home(path, operations=2)

    assert sqlite_owner._MIGRATIONS[-1].version == 9
    assert sqlite_owner._MIGRATIONS[-1].name == (
        "add_context_memory_learning_and_skills"
    )
    facts = await store.load_session_operation("operation-1")
    assert facts is not None
    assert facts.session_id == SESSION_ID
    assert facts.revision == "1"

    summary = canonical_json(
        {
            "approval_ids": [],
            "evidence_ids": [],
            "kind": "extractive_session_history",
            "operation_ids": ["operation-1"],
            "resource_ids": [],
            "schema_version": 1,
            "trust": "untrusted_external",
        }
    )
    checkpoint = SessionCompressionCheckpoint(
        id="checkpoint-1",
        agent_id=AGENT_ID,
        session_id=SESSION_ID,
        version=1,
        through_position=0,
        through_operation_id="operation-1",
        source_fingerprint=SHA_A,
        summary=summary,
        operation_ids=("operation-1",),
        created_at=NOW + timedelta(hours=1),
    )
    assert (
        await store.commit_session_compression(
            checkpoint,
            expected_version=0,
        )
        == checkpoint
    )
    assert await store.load_session_compression(AGENT_ID, SESSION_ID) == checkpoint
    assert await store.load_session_compression("agent-other", SESSION_ID) is None

    with pytest.raises(SQLiteStoreError, match="version changed"):
        await store.commit_session_compression(
            replace(checkpoint, id="checkpoint-stale"),
            expected_version=0,
        )
    wrong_prefix = replace(
        checkpoint,
        id="checkpoint-2",
        version=2,
        through_operation_id="operation-2",
        operation_ids=("operation-2",),
    )
    with pytest.raises(SQLiteStoreError, match="exact session prefix"):
        await store.commit_session_compression(wrong_prefix, expected_version=1)
    await store.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (9,)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE session_compression_checkpoints SET summary = 'x'"
            )
    finally:
        connection.close()


async def test_model_profile_binding_is_normalized_immutable_and_reopen_exact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-profile.db"
    profile = ModelProfile(
        id="mock:phase5",
        context_window_tokens=131_072,
        max_output_tokens=8_192,
        supports_tools=True,
        supports_parallel_tools=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=True,
        supports_vision=True,
        supports_documents=True,
        supports_prompt_caching=True,
        supports_native_continuation=True,
        input_cost_per_million_usd=Decimal("2.50"),
        output_cost_per_million_usd=Decimal("10.125"),
        data_routing_classification="restricted",
    )
    store, _ = await _open_home(path)
    assert await store.load_model_profile(AGENT_ID) is None
    assert await store.bind_model_profile(AGENT_ID, profile) == profile
    assert await store.bind_model_profile(AGENT_ID, profile) == profile
    with pytest.raises(ModelProfileConflictError, match="different model profile"):
        await store.bind_model_profile(
            AGENT_ID,
            replace(profile, max_output_tokens=4_096),
        )
    await store.close()

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT profile_id, context_window_tokens, max_output_tokens, "
            "supports_tools, supports_parallel_tools, input_cost_per_million_usd, "
            "output_cost_per_million_usd, data_routing_classification, available, "
            "healthy FROM agent_model_profiles WHERE agent_id = ?",
            (AGENT_ID,),
        ).fetchone()
        assert row == (
            "mock:phase5",
            131_072,
            8_192,
            1,
            1,
            "2.50",
            "10.125",
            "restricted",
            1,
            1,
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE agent_model_profiles SET healthy = 0 WHERE agent_id = ?",
                (AGENT_ID,),
            )

    reopened = await SQLiteOperationStore.open(path)
    assert await reopened.load_model_profile(AGENT_ID) == profile
    await reopened.close()


async def test_memory_filters_history_cas_restore_and_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.db"
    store, _ = await _open_home(path)
    broad = _import_memory(
        "memory-broad",
        scope=MemoryScope(agent_id=AGENT_ID),
        logical_key="customer status",
        content="Completed customer status",
    )
    private = _import_memory(
        "memory-private",
        scope=MemoryScope(agent_id=AGENT_ID, user_id="user-1"),
        logical_key="customer secret",
        content="Completed restricted status",
        sensitivity=MemorySensitivity.RESTRICTED,
    )
    await store.create_memory(broad.record, broad.version)
    await store.create_memory(private.record, private.version)

    hits = await store.recall_candidates(
        query="completed status",
        scope=MemoryScope(agent_id=AGENT_ID, user_id="user-2"),
        states=(MemoryState.ACTIVE,),
        sensitivities=(MemorySensitivity.PUBLIC,),
        unexpired_at=NOW,
        limit=10,
    )
    assert tuple(hit.record.id for hit in hits) == ("memory-broad",)
    assert await store.load_history("agent-other", broad.record.id) is None

    version_two = replace(
        broad.version,
        version=2,
        content="Corrected customer status",
        created_at=NOW + timedelta(minutes=1),
        supersedes_version=1,
    )
    history = await store.supersede(
        MemorySupersessionRequest(
            agent_id=AGENT_ID,
            memory_id=broad.record.id,
            expected_version=1,
            replacement=version_two,
        )
    )
    assert history.record.current_version == 2
    with pytest.raises(MemoryStoreConflictError, match="head changed"):
        await store.supersede(
            MemorySupersessionRequest(
                agent_id=AGENT_ID,
                memory_id=broad.record.id,
                expected_version=1,
                replacement=version_two,
            )
        )
    restored = replace(
        broad.version,
        version=3,
        created_at=NOW + timedelta(minutes=2),
        supersedes_version=2,
    )
    history = await store.restore(
        MemoryRestoreRequest(
            agent_id=AGENT_ID,
            memory_id=broad.record.id,
            expected_version=2,
            restore_version=1,
            replacement=restored,
        )
    )
    assert history.current.content == broad.version.content
    await store.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE memory_records SET scope_fingerprint = ? WHERE id = ?",
            (SHA_B, broad.record.id),
        )
        connection.commit()
    finally:
        connection.close()
    reopened = await SQLiteOperationStore.open(path)
    with pytest.raises(SQLiteCorruptionError, match="scope fingerprint"):
        await reopened.load_history(AGENT_ID, broad.record.id)
    await reopened.close()


async def test_learning_is_redacted_idempotent_scoped_and_state_guarded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "learning.db"
    store, _ = await _open_home(path, operations=1)
    proposal = _proposal(
        1,
        proposal_id="proposal-safe",
        candidate={"business_definition": "completed means complete"},
        created_at=NOW + timedelta(hours=1),
    )
    assert proposal.state is LearningProposalState.PROPOSED
    assert await store.create_proposal(proposal) == proposal
    replay = await store.create_proposal(
        replace(proposal, id="proposal-retry", created_at=NOW + timedelta(hours=2))
    )
    assert replay == proposal

    decision = LearningDecision(
        proposal_id=proposal.id,
        idempotency_key=proposal.idempotency_key,
        candidate_hash=proposal.candidate_hash,
        state=LearningProposalState.REJECTED,
        decided_at=NOW + timedelta(hours=2),
        rejection_category=LearningRejectionCategory.OUT_OF_BOUNDS,
        rejection_reason="not durable enough",
    )
    rejected = await store.resolve_proposal(
        decision,
        expected_state=LearningProposalState.PROPOSED,
    )
    assert rejected.state is LearningProposalState.REJECTED
    assert rejected.candidate_payload is None
    assert await store.create_proposal(proposal) == rejected
    assert (
        await store.resolve_proposal(
            decision,
            expected_state=LearningProposalState.PROPOSED,
        )
        == rejected
    )
    assert await store.load_proposal("agent-other", proposal.id) is None

    other = replace(
        decision,
        rejection_reason="another decision",
    )
    with pytest.raises(LearningStoreConflictError, match="another decision"):
        await store.resolve_proposal(
            other,
            expected_state=LearningProposalState.PROPOSED,
        )
    await store.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT candidate_payload_json FROM learning_proposals WHERE id = ?",
            (proposal.id,),
        ).fetchone() == (None,)
    finally:
        connection.close()


def _skill_version(number: int) -> tuple[Skill, SkillVersion, SkillIndex]:
    skill = Skill(
        id="skill-customers",
        agent_id=AGENT_ID,
        stable_name="customer-lookup",
        source=SkillSource.USER,
        created_at=NOW,
    )
    version = SkillVersion(
        id=f"skill-version-{number}",
        agent_id=AGENT_ID,
        skill_id=skill.id,
        stable_name=skill.stable_name,
        version=f"{number}.0.0",
        description="Look up customers safely",
        domains=("customers",),
        resource_kinds=("table",),
        required_capability_ids=("data.read",),
        activation_mode=SkillActivationMode.EXPLICIT,
        sensitivity_notes=None,
        policy_notes=None,
        source=skill.source,
        content_hash=SHA_A if number == 1 else SHA_B,
        instructions=f"Version {number} instructions",
        source_path="customer-lookup/SKILL.md",
        created_at=NOW + timedelta(minutes=number),
    )
    return skill, version, SkillIndex.from_version(version)


async def test_skill_versions_and_activations_are_append_only_and_cas_guarded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "skills.db"
    store, _ = await _open_home(path)
    skill, version_one, index_one = _skill_version(1)
    assert await store.record_discovery(skill, version_one, index_one) == index_one
    _, version_two, index_two = _skill_version(2)
    indexed = await store.record_discovery(skill, version_two, index_two)
    assert indexed.version_id == version_two.id

    activation = SkillActivation(
        id="activation-1",
        agent_id=AGENT_ID,
        skill_id=skill.id,
        version_id=version_one.id,
        previous_version_id=None,
        actor_id="user-1",
        reason="approved",
        activated_at=NOW + timedelta(hours=1),
    )
    inspection = await store.activate_skill(
        activation,
        expected_active_version_id=None,
    )
    assert inspection.index.active_version_id == version_one.id
    assert inspection.index.version_id == version_one.id
    assert inspection.index.matches(version_one)
    assert len(inspection.versions) == 2
    assert await store.load_skill_index("agent-other", skill.id) is None

    _, version_three, index_three = _skill_version(3)
    after_inactive_discovery = await store.record_discovery(
        skill,
        version_three,
        index_three,
    )
    assert after_inactive_discovery == inspection.index
    refreshed_inspection = await store.inspect_skill(AGENT_ID, skill.id)
    assert refreshed_inspection is not None
    assert refreshed_inspection.index == inspection.index
    assert tuple(item.version for item in refreshed_inspection.versions) == (
        "1.0.0",
        "2.0.0",
        "3.0.0",
    )

    stale = SkillActivation(
        id="activation-stale",
        agent_id=AGENT_ID,
        skill_id=skill.id,
        version_id=version_two.id,
        previous_version_id=None,
        actor_id="user-1",
        reason="stale",
        activated_at=NOW + timedelta(hours=2),
    )
    with pytest.raises(SkillActivationConflictError, match="changed"):
        await store.activate_skill(stale, expected_active_version_id=None)
    await store.close()

    connection = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE skill_versions SET instructions = 'changed' WHERE id = ?",
                (version_one.id,),
            )
    finally:
        connection.close()


async def test_explicit_correction_commit_is_atomic_and_old_key_replays_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "correction.db"
    store, source = await _open_home(path, operations=2, source=True)
    assert source is not None
    candidate_one = {"alias": "complete", "business_term": "completed"}
    proposal_one = _proposal(
        1,
        proposal_id="proposal-one",
        candidate=candidate_one,
        created_at=NOW + timedelta(hours=1),
    )
    intended_one = _alias_snapshot(
        source_id=source.id,
        operation_number=1,
        version=1,
        content="complete",
        created_at=NOW + timedelta(hours=1),
    )
    decision_one = LearningDecision(
        proposal_id=proposal_one.id,
        idempotency_key=proposal_one.idempotency_key,
        candidate_hash=proposal_one.candidate_hash,
        state=LearningProposalState.COMMITTED,
        decided_at=NOW + timedelta(hours=1),
        result_memory_id=intended_one.record.id,
        result_memory_version=1,
    )
    first = await store.commit_explicit_correction(
        ExplicitCorrectionCommit(
            proposal=proposal_one,
            intended_memory=intended_one,
            decision=decision_one,
        )
    )
    assert not first.replayed
    assert first.memory is not None and first.memory.record.current_version == 1

    proposal_two = _proposal(
        2,
        proposal_id="proposal-two",
        candidate={"alias": "complete-v2", "business_term": "completed"},
        created_at=NOW + timedelta(hours=2),
    )
    intended_two = _alias_snapshot(
        source_id=source.id,
        operation_number=2,
        version=2,
        content="complete-v2",
        created_at=NOW + timedelta(hours=2),
        record=intended_one.record,
    )
    decision_two = LearningDecision(
        proposal_id=proposal_two.id,
        idempotency_key=proposal_two.idempotency_key,
        candidate_hash=proposal_two.candidate_hash,
        state=LearningProposalState.COMMITTED,
        decided_at=NOW + timedelta(hours=2),
        result_memory_id=intended_two.record.id,
        result_memory_version=2,
    )
    second = await store.commit_explicit_correction(
        ExplicitCorrectionCommit(
            proposal=proposal_two,
            expected_memory_version=1,
            intended_memory=intended_two,
            decision=decision_two,
        )
    )
    assert second.memory is not None and second.memory.record.current_version == 2

    retry_intended = _alias_snapshot(
        source_id=source.id,
        operation_number=1,
        version=3,
        content="complete",
        created_at=NOW + timedelta(hours=3),
        record=intended_two.record,
    )
    retry_decision = replace(decision_one, result_memory_version=3)
    replay = await store.commit_explicit_correction(
        ExplicitCorrectionCommit(
            proposal=proposal_one,
            expected_memory_version=2,
            intended_memory=retry_intended,
            decision=retry_decision,
        )
    )
    assert replay.replayed
    assert replay.proposal.result_memory_version == 1
    assert replay.memory is not None and replay.memory.record.current_version == 2

    stale_proposal = _proposal(
        1,
        proposal_id="proposal-stale",
        candidate={"alias": "stale", "business_term": "completed"},
        created_at=NOW + timedelta(hours=4),
    )
    stale_intended = _alias_snapshot(
        source_id=source.id,
        operation_number=1,
        version=2,
        content="stale",
        created_at=NOW + timedelta(hours=4),
        record=intended_one.record,
    )
    stale_decision = LearningDecision(
        proposal_id=stale_proposal.id,
        idempotency_key=stale_proposal.idempotency_key,
        candidate_hash=stale_proposal.candidate_hash,
        state=LearningProposalState.COMMITTED,
        decided_at=NOW + timedelta(hours=4),
        result_memory_id=stale_intended.record.id,
        result_memory_version=2,
    )
    with pytest.raises(ExplicitCorrectionStoreConflictError, match="head changed"):
        await store.commit_explicit_correction(
            ExplicitCorrectionCommit(
                proposal=stale_proposal,
                expected_memory_version=1,
                intended_memory=stale_intended,
                decision=stale_decision,
            )
        )
    assert await store.load_proposal(AGENT_ID, stale_proposal.id) is None
    await store.close()
