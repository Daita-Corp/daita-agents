from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita._json import canonical_json
from daita.catalog import (
    CatalogResource,
    CatalogResourceRevision,
    ResourceKind,
    Sensitivity,
    catalog_resource_id,
)
from daita.learning import (
    LearningCandidateCategory,
    LearningProposal,
    LearningProposalState,
    LearningRejectionCategory,
    resolve_learning_proposal,
)
from daita.loop.models import LoopBudgets, LoopPhase, LoopState
from daita.memory import (
    RESOURCE_ALIAS_CORRECTION_PREFIX,
    ExplicitCorrectionCommit,
    ExplicitCorrectionFormatError,
    ExplicitCorrectionLearningService,
    ExplicitCorrectionNotEligibleError,
    ExplicitCorrectionResult,
    ExplicitCorrectionStoreConflictError,
    MemoryCreator,
    MemoryHistory,
    MemoryKind,
    MemoryProvenanceKind,
    MemoryScope,
    MemoryState,
    ResourceAliasCorrection,
)
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)

NOW = datetime(2026, 7, 18, 15, 0, tzinfo=timezone.utc)


class FakeCatalogReader:
    def __init__(self, resource: CatalogResource | None) -> None:
        self.resource = resource
        self.calls: list[tuple[str, str]] = []

    async def load_resource(
        self,
        agent_id: str,
        resource_id: str,
    ) -> CatalogResource | None:
        self.calls.append((agent_id, resource_id))
        return self.resource


class FakeExplicitCorrectionStore:
    def __init__(self) -> None:
        self.histories: dict[tuple[str, MemoryKind, str], MemoryHistory] = {}
        self.results: dict[str, ExplicitCorrectionResult] = {}
        self.proposals: dict[str, LearningProposal] = {}
        self.requests: list[ExplicitCorrectionCommit] = []
        self.applied_count = 0

    @staticmethod
    def _key(
        scope: MemoryScope,
        logical_key: str,
        kind: MemoryKind = MemoryKind.RESOURCE_ALIAS,
    ) -> tuple[str, MemoryKind, str]:
        return scope.fingerprint, kind, logical_key

    async def load_resource_alias(
        self,
        scope: MemoryScope,
        logical_key: str,
    ) -> MemoryHistory | None:
        return self.histories.get(self._key(scope, logical_key))

    async def load_learning_memory(
        self,
        scope: MemoryScope,
        kind: MemoryKind,
        logical_key: str,
    ) -> MemoryHistory | None:
        return self.histories.get(self._key(scope, logical_key, kind))

    async def create_proposal(
        self,
        proposal: LearningProposal,
    ) -> LearningProposal:
        existing = self.proposals.get(proposal.idempotency_key)
        if existing is not None:
            return existing
        self.proposals[proposal.idempotency_key] = proposal
        return proposal

    async def load_proposal(
        self,
        agent_id: str,
        proposal_id: str,
    ) -> LearningProposal | None:
        for proposal in self.proposals.values():
            if proposal.provenance.agent_id == agent_id and proposal.id == proposal_id:
                return proposal
        for result in self.results.values():
            proposal = result.proposal
            if proposal.provenance.agent_id == agent_id and proposal.id == proposal_id:
                return proposal
        return None

    async def commit_explicit_correction(
        self,
        request: ExplicitCorrectionCommit,
    ) -> ExplicitCorrectionResult:
        self.requests.append(request)
        prior = self.results.get(request.proposal.idempotency_key)
        if prior is not None:
            if prior.memory is None:
                return replace(prior, replayed=True)
            assert request.intended_memory is not None
            current = self.histories[
                self._key(
                    request.intended_memory.record.scope,
                    request.intended_memory.record.logical_key,
                    request.intended_memory.record.kind,
                )
            ]
            return ExplicitCorrectionResult(
                proposal=prior.proposal,
                memory=current,
                replayed=True,
            )

        if request.proposal.state is LearningProposalState.REJECTED:
            result = ExplicitCorrectionResult(
                proposal=request.proposal,
                memory=None,
            )
        else:
            assert request.intended_memory is not None
            assert request.decision is not None
            intended = request.intended_memory
            key = self._key(
                intended.record.scope,
                intended.record.logical_key,
                intended.record.kind,
            )
            current_head = self.histories.get(key)
            actual_version = (
                None if current_head is None else current_head.record.current_version
            )
            if actual_version != request.expected_memory_version:
                raise ExplicitCorrectionStoreConflictError(
                    "resource alias head changed"
                )
            versions = (
                (intended.version,)
                if current_head is None
                else (*current_head.versions, intended.version)
            )
            history = MemoryHistory(intended.record, versions)
            self.histories[key] = history
            result = ExplicitCorrectionResult(
                proposal=resolve_learning_proposal(
                    request.proposal,
                    request.decision,
                ),
                memory=history,
            )
        self.applied_count += 1
        self.results[request.proposal.idempotency_key] = result
        return result


def _resource(
    *,
    agent_id: str = "agent-1",
    source_id: str = "source-1",
    sensitivity: Sensitivity = Sensitivity.INTERNAL,
) -> CatalogResource:
    resource_id = catalog_resource_id(source_id, ResourceKind.TABLE, "customers")
    revision = CatalogResourceRevision.build(
        resource_id=resource_id,
        sync_id="sync-1",
        observed_at=NOW,
    )
    return CatalogResource.build(
        agent_id=agent_id,
        source_id=source_id,
        native_identity="customers",
        external_uri="sqlite://customers",
        kind=ResourceKind.TABLE,
        name="customers",
        sensitivity=sensitivity,
        revision=revision,
        first_observed_at=NOW,
        last_observed_at=NOW,
    )


def _correction(
    resource: CatalogResource,
    *,
    source_id: str | None = None,
    revision: str | None = None,
    stored_value: str = "complete",
) -> ResourceAliasCorrection:
    return ResourceAliasCorrection(
        source_id=resource.source_id if source_id is None else source_id,
        resource_id=resource.id,
        resource_revision=(resource.current_revision if revision is None else revision),
        field="status",
        business_term="completed",
        stored_value=stored_value,
    )


def _snapshot(
    message: str,
    *,
    operation_id: str = "operation-1",
    trigger_id: str = "trigger-1",
    status: OperationStatus = OperationStatus.SUCCEEDED,
    phase: LoopPhase = LoopPhase.TERMINAL,
    trigger_kind: TriggerKind = TriggerKind.USER,
    terminal_reason: str = "completed",
) -> OperationSnapshot:
    trigger = AgentTrigger(
        id=trigger_id,
        agent_id="agent-1",
        kind=trigger_kind,
        source_id="user-1",
        session_id="session-origin",
        payload={"message": message},
        created_at=NOW,
    )
    is_terminal = status in {
        OperationStatus.SUCCEEDED,
        OperationStatus.FAILED,
        OperationStatus.CANCELLED,
        OperationStatus.INTERRUPTED,
    }
    operation = Operation(
        id=operation_id,
        agent_id=trigger.agent_id,
        trigger_id=trigger.id,
        status=status,
        session_id=trigger.session_id,
        final_text=(
            "Correction processed." if status is OperationStatus.SUCCEEDED else None
        ),
        terminal_reason=terminal_reason if is_terminal else None,
        created_at=NOW,
        updated_at=NOW,
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(phase=phase),
        budgets=LoopBudgets(),
        turns=(),
        model_calls=(),
        readiness=(),
        tasks=(),
        evidence=(),
        observations=(),
        events=(),
    )


def _service(
    resource: CatalogResource | None,
    store: FakeExplicitCorrectionStore | None = None,
) -> tuple[
    ExplicitCorrectionLearningService,
    FakeCatalogReader,
    FakeExplicitCorrectionStore,
]:
    catalog = FakeCatalogReader(resource)
    resolved_store = store or FakeExplicitCorrectionStore()
    return (
        ExplicitCorrectionLearningService(
            catalog=catalog,
            store=resolved_store,
            clock=lambda: NOW,
        ),
        catalog,
        resolved_store,
    )


@pytest.mark.parametrize(
    ("status", "phase", "terminal_reason"),
    [
        (OperationStatus.RUNNING, LoopPhase.PREPARING_CONTEXT, "unused"),
        (OperationStatus.SUCCEEDED, LoopPhase.SYNTHESIZING, "completed"),
        (OperationStatus.FAILED, LoopPhase.TERMINAL, "failed"),
        (OperationStatus.FAILED, LoopPhase.TERMINAL, "policy_blocked"),
    ],
)
async def test_learning_consumes_only_completed_succeeded_operations(
    status: OperationStatus,
    phase: LoopPhase,
    terminal_reason: str,
) -> None:
    resource = _resource()
    service, catalog, store = _service(resource)
    snapshot = _snapshot(
        _correction(resource).to_trigger_message(),
        status=status,
        phase=phase,
        terminal_reason=terminal_reason,
    )

    with pytest.raises(
        ExplicitCorrectionNotEligibleError,
        match="completed SUCCEEDED",
    ):
        await service.learn(snapshot)

    assert catalog.calls == []
    assert store.requests == []


async def test_original_user_trigger_requires_the_one_canonical_record_pattern() -> (
    None
):
    resource = _resource()
    service, catalog, store = _service(resource)
    correction = _correction(resource)

    with pytest.raises(ExplicitCorrectionFormatError, match="supported"):
        await service.learn(_snapshot("Store this hidden memory."))
    noncanonical = RESOURCE_ALIAS_CORRECTION_PREFIX + canonical_json(
        correction.candidate
    ).replace(":", ": ", 1)
    with pytest.raises(ExplicitCorrectionFormatError, match="canonical"):
        await service.learn(_snapshot(noncanonical))
    with pytest.raises(ExplicitCorrectionNotEligibleError, match="USER"):
        await service.learn(
            _snapshot(
                correction.to_trigger_message(),
                trigger_kind=TriggerKind.INTERNAL,
            )
        )

    assert catalog.calls == []
    assert store.requests == []


async def test_natural_remember_commits_safe_agent_scoped_semantic_fact() -> None:
    service, catalog, store = _service(None)
    snapshot = _snapshot("Remember that fiscal weeks start on Monday.")

    result = await service.learn(snapshot)

    assert result.proposal.state is LearningProposalState.COMMITTED
    assert result.memory is not None
    assert result.memory.record.kind is MemoryKind.SEMANTIC_FACT
    assert result.memory.record.scope == MemoryScope(agent_id="agent-1")
    assert result.memory.current.content == "Fiscal weeks start on Monday."
    assert result.memory.current.attributes["statement"] == (
        "Fiscal weeks start on Monday."
    )
    assert result.memory.current.provenance.kind is (
        MemoryProvenanceKind.USER_STATEMENT
    )
    assert result.memory.current.provenance.operation_id == snapshot.operation.id
    assert result.memory.current.resource_revision is None
    assert catalog.calls == []
    assert store.applied_count == 1


async def test_natural_alias_shape_without_current_read_binding_fails_closed() -> None:
    service, catalog, store = _service(None)

    with pytest.raises(
        ExplicitCorrectionNotEligibleError,
        match="accepted current read binding",
    ):
        await service.learn(
            _snapshot("Remember that completed status is stored as complete.")
        )

    assert catalog.calls == []
    assert store.histories == {}
    assert store.requests == []


async def test_evidence_fact_without_accepted_current_read_is_redacted() -> None:
    service, catalog, store = _service(None)
    raw = "Every customer is active."

    proposal = await service.propose_evidence_fact(
        _snapshot(f"Propose a fact from the evidence: {raw}")
    )

    assert proposal is not None
    assert proposal.category is LearningCandidateCategory.EVIDENCE_BACKED_FACT
    assert proposal.state is LearningProposalState.REJECTED
    assert proposal.rejection_category is LearningRejectionCategory.INELIGIBLE_SOURCE
    assert proposal.rejection_reason == "accepted_current_read_evidence_required"
    assert proposal.candidate_payload is None
    assert raw not in repr(proposal)
    assert proposal.provenance.evidence_id is None
    assert proposal.provenance.evidence_accepted is False
    assert catalog.calls == []
    assert store.histories == {}


@pytest.mark.parametrize("catalog_case", ["missing", "other_agent"])
async def test_catalog_resource_must_be_current_and_owned(
    catalog_case: str,
) -> None:
    resource = _resource()
    returned = (
        None
        if catalog_case == "missing"
        else replace(
            resource,
            agent_id="agent-other",
        )
    )
    service, _, store = _service(returned)

    with pytest.raises(ExplicitCorrectionNotEligibleError):
        await service.learn(_snapshot(_correction(resource).to_trigger_message()))
    assert store.requests == []


@pytest.mark.parametrize(
    "correction",
    [
        "stale_revision",
        "other_source",
    ],
)
async def test_catalog_revision_and_source_are_exact_bindings(
    correction: str,
) -> None:
    resource = _resource()
    service, _, store = _service(resource)
    selected = (
        _correction(resource, revision="sha256:" + "f" * 64)
        if correction == "stale_revision"
        else _correction(resource, source_id="source-other")
    )

    with pytest.raises(ExplicitCorrectionNotEligibleError):
        await service.learn(_snapshot(selected.to_trigger_message()))
    assert store.requests == []


async def test_safe_correction_creates_exact_resource_alias_and_provenance() -> None:
    resource = _resource()
    correction = _correction(resource)
    service, _, store = _service(resource)
    snapshot = _snapshot(correction.to_trigger_message())

    result = await service.learn(snapshot)

    assert result.replayed is False
    assert result.proposal.state is LearningProposalState.COMMITTED
    assert result.memory is not None
    history = result.memory
    assert history.record.kind is MemoryKind.RESOURCE_ALIAS
    assert history.record.state is MemoryState.ACTIVE
    assert history.record.scope == MemoryScope(
        agent_id="agent-1",
        source_id=resource.source_id,
        resource_id=resource.id,
    )
    assert history.record.logical_key == "status:completed"
    assert history.record.current_version == 1
    assert len(history.versions) == 1
    version = history.current
    assert version.creator is MemoryCreator.LEARNING_SERVICE
    assert version.confidence == 1.0
    assert version.resource_revision == resource.current_revision
    assert version.provenance.kind is MemoryProvenanceKind.USER_STATEMENT
    assert version.provenance.operation_id == snapshot.operation.id
    assert version.provenance.trigger_id == snapshot.trigger.id
    assert version.provenance.session_id == snapshot.operation.session_id
    assert version.provenance.content_hash == result.proposal.provenance.source_hash
    assert dict(version.attributes) == {
        "business_term": "completed",
        "field": "status",
        "stored_value": "complete",
    }
    assert result.proposal.result_memory_id == history.record.id
    assert result.proposal.result_memory_version == 1
    assert store.requests[0].expected_memory_version is None
    assert store.applied_count == 1


@pytest.mark.parametrize("unsafe_kind", ["pii", "raw_row"])
async def test_unsafe_candidates_persist_only_a_redacted_rejection(
    unsafe_kind: str,
) -> None:
    resource = _resource()
    correction = _correction(
        resource,
        stored_value=("ada@example.com" if unsafe_kind == "pii" else "complete"),
    )
    if unsafe_kind == "pii":
        message = correction.to_trigger_message()
        forbidden = "ada@example.com"
    else:
        candidate = correction.candidate.to_dict()
        candidate["rows"] = [{"customer_id": 42, "status": "complete"}]
        message = RESOURCE_ALIAS_CORRECTION_PREFIX + canonical_json(candidate)
        forbidden = "customer_id"
    service, _, store = _service(resource)

    result = await service.learn(_snapshot(message))

    assert result.proposal.state is LearningProposalState.REJECTED
    assert (
        result.proposal.rejection_category is LearningRejectionCategory.RAW_ROW_OR_PII
    )
    assert result.proposal.candidate_payload is None
    assert result.memory is None
    assert store.histories == {}
    assert store.applied_count == 1
    assert forbidden not in repr(store.requests[0])


async def test_retry_is_idempotent_and_does_not_advance_memory_twice() -> None:
    resource = _resource()
    service, _, store = _service(resource)
    snapshot = _snapshot(_correction(resource).to_trigger_message())

    first = await service.learn(snapshot)
    replay = await service.learn(snapshot)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.proposal == first.proposal
    assert replay.memory is not None
    assert replay.memory.record.current_version == 1
    assert len(replay.memory.versions) == 1
    assert store.applied_count == 1
    assert len(store.requests) == 1


async def test_new_correction_supersedes_version_without_deleting_history() -> None:
    resource = _resource()
    service, _, store = _service(resource)
    first = _snapshot(
        _correction(resource).to_trigger_message(),
        operation_id="operation-a",
        trigger_id="trigger-a",
    )
    second = _snapshot(
        _correction(resource, stored_value="closed").to_trigger_message(),
        operation_id="operation-b",
        trigger_id="trigger-b",
    )

    first_result = await service.learn(first)
    second_result = await service.learn(second)

    assert first_result.memory is not None
    assert second_result.memory is not None
    assert second_result.memory.record.id == first_result.memory.record.id
    assert second_result.memory.record.current_version == 2
    assert [version.version for version in second_result.memory.versions] == [1, 2]
    assert second_result.memory.versions[0].attributes["stored_value"] == "complete"
    assert second_result.memory.versions[1].attributes["stored_value"] == "closed"
    assert second_result.memory.versions[1].supersedes_version == 1
    assert second_result.memory.versions[1].provenance.operation_id == "operation-b"
    assert store.requests[1].expected_memory_version == 1
    assert store.applied_count == 2


async def test_replaying_older_operation_uses_its_historical_version_not_new_head() -> (
    None
):
    resource = _resource()
    service, _, store = _service(resource)
    correction_a = _snapshot(
        _correction(resource).to_trigger_message(),
        operation_id="operation-a",
        trigger_id="trigger-a",
    )
    correction_b = _snapshot(
        _correction(resource, stored_value="closed").to_trigger_message(),
        operation_id="operation-b",
        trigger_id="trigger-b",
    )

    first = await service.learn(correction_a)
    second = await service.learn(correction_b)
    replay = await service.learn(correction_a)

    assert first.proposal.result_memory_version == 1
    assert second.proposal.result_memory_version == 2
    assert replay.replayed is True
    assert replay.proposal == first.proposal
    assert replay.memory is not None
    assert replay.memory.record.current_version == 2
    assert replay.memory.versions[0].attributes["stored_value"] == "complete"
    assert replay.memory.current.attributes["stored_value"] == "closed"
    assert store.applied_count == 2
    durable = next(iter(store.histories.values()))
    assert durable.record.current_version == 2
    assert len(durable.versions) == 2


async def test_terminal_replay_precedes_mutable_catalog_revision_validation() -> None:
    resource = _resource()
    service, catalog, store = _service(resource)
    original = _snapshot(_correction(resource).to_trigger_message())
    first = await service.learn(original)
    catalog.resource = replace(
        resource,
        current_revision="sha256:" + "f" * 64,
        current_sync_id="sync-later",
    )
    restarted = ExplicitCorrectionLearningService(
        catalog=catalog,
        store=store,
        clock=lambda: NOW + timedelta(days=1),
    )

    replay = await restarted.learn(original)

    assert replay.replayed is True
    assert replay.proposal == first.proposal
    assert replay.memory == first.memory
    assert catalog.calls == [("agent-1", resource.id)]
    assert store.applied_count == 1


async def test_new_stale_correction_still_validates_current_catalog_revision() -> None:
    resource = _resource()
    service, catalog, store = _service(resource)
    await service.learn(_snapshot(_correction(resource).to_trigger_message()))
    catalog.resource = replace(
        resource,
        current_revision="sha256:" + "f" * 64,
        current_sync_id="sync-later",
    )
    new_stale_operation = _snapshot(
        _correction(resource, stored_value="closed").to_trigger_message(),
        operation_id="operation-later",
        trigger_id="trigger-later",
    )

    with pytest.raises(ExplicitCorrectionNotEligibleError, match="not current"):
        await service.learn(new_stale_operation)

    assert store.applied_count == 1


def test_learning_service_imports_no_executor_and_has_no_execution_call() -> None:
    path = Path(__file__).parents[3] / "src" / "daita" / "memory" / "learning.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    execute_calls: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute"
        ):
            execute_calls.append(node.lineno)

    assert not any("executor" in name.casefold().split(".") for name in imports)
    assert execute_calls == []
