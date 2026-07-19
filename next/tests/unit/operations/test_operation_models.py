from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daita._json import FrozenJsonObject
from daita.loop.models import LoopPhase, LoopState
from daita.operations.models import (
    ActionProposal,
    ActionRejection,
    ActionValidationFacts,
    AgentTrigger,
    Observation,
    Operation,
    OperationStatus,
    TriggerKind,
)

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def test_operation_status_and_loop_checkpoint_use_distinct_records() -> None:
    checkpoint = LoopState(phase=LoopPhase.CREATED)
    operation = Operation(
        id="op-1",
        agent_id="agent-1",
        trigger_id="trigger-1",
        session_id=None,
        status=OperationStatus.PENDING,
        created_at=NOW,
        updated_at=NOW,
    )

    assert operation.status is OperationStatus.PENDING
    assert checkpoint.phase is LoopPhase.CREATED
    assert operation.session_id is None


def test_terminal_status_requires_reason_and_nonterminal_status_rejects_one() -> None:
    with pytest.raises(ValueError, match="terminal reason"):
        Operation(
            id="op-1",
            agent_id="agent-1",
            trigger_id="trigger-1",
            status=OperationStatus.CANCELLED,
            created_at=NOW,
            updated_at=NOW,
        )

    with pytest.raises(ValueError, match="nonterminal"):
        Operation(
            id="op-1",
            agent_id="agent-1",
            trigger_id="trigger-1",
            status=OperationStatus.RUNNING,
            created_at=NOW,
            updated_at=NOW,
            terminal_reason="not-terminal",
        )


def test_succeeded_operation_requires_final_text_and_monotonic_timestamps() -> None:
    with pytest.raises(ValueError, match="final text"):
        Operation(
            id="op-1",
            agent_id="agent-1",
            trigger_id="trigger-1",
            status=OperationStatus.SUCCEEDED,
            created_at=NOW,
            updated_at=NOW,
            terminal_reason="completed",
        )

    with pytest.raises(ValueError, match="updated_at"):
        Operation(
            id="op-1",
            agent_id="agent-1",
            trigger_id="trigger-1",
            status=OperationStatus.RUNNING,
            created_at=NOW,
            updated_at=NOW - timedelta(seconds=1),
        )


def test_failed_operation_may_retain_an_honest_partial_user_result() -> None:
    operation = Operation(
        id="op-1",
        agent_id="agent-1",
        trigger_id="trigger-1",
        status=OperationStatus.FAILED,
        created_at=NOW,
        updated_at=NOW,
        terminal_reason="evidence_incomplete",
        final_text="Partial result: one requested side had no accepted evidence.",
    )

    assert operation.final_text is not None
    assert operation.final_text.startswith("Partial result")


def test_trigger_is_typed_timezone_aware_and_mutation_isolated() -> None:
    source_ids = ["source-1"]
    payload: dict[str, object] = {
        "message": "hello",
        "scope": {"source_ids": source_ids},
    }
    trigger = AgentTrigger(
        id="trigger-1",
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        session_id="session-1",
        payload=payload,
        created_at=NOW,
    )
    source_ids.append("source-2")

    assert isinstance(trigger.payload, FrozenJsonObject)
    assert trigger.payload.to_dict() == {
        "message": "hello",
        "scope": {"source_ids": ["source-1"]},
    }

    with pytest.raises(ValueError, match="timezone-aware"):
        AgentTrigger(
            id="trigger-2",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={},
            created_at=datetime(2026, 7, 16, 12, 0),
        )


def test_action_proposal_freezes_validated_arguments() -> None:
    arguments = {"key": "alpha"}
    proposal = ActionProposal(
        operation_id="op-1",
        turn_id="turn-1",
        call_id="call-1",
        capability_id="fake.read",
        arguments=arguments,
        proposed_at=NOW,
    )
    arguments["key"] = "mutated"

    assert isinstance(proposal.arguments, FrozenJsonObject)
    assert proposal.arguments.to_dict() == {"key": "alpha"}


def test_explicit_action_validation_facts_freeze_scope_impact_and_evidence() -> None:
    impact = {"affected_rows": 2, "bounded": True}
    facts = ActionValidationFacts(
        schema_version=1,
        validation_passed=True,
        in_scope=True,
        destructive=False,
        sensitivity_class="confidential",
        source_id="source-sqlite",
        resource_ids=("resource-orders",),
        resource_revisions=(("resource-orders", "sha256:" + ("a" * 64)),),
        source_revision="sqlite:data-version:7",
        impact=impact,
        evidence_ids=("evidence-impact",),
    )
    impact["affected_rows"] = 999

    assert facts.fingerprint is not None
    assert isinstance(facts.impact, FrozenJsonObject)
    assert facts.impact.to_dict() == {"affected_rows": 2, "bounded": True}
    assert facts.audit_projection()["validation_fingerprint"] == facts.fingerprint

    proposal = ActionProposal(
        operation_id="op-1",
        turn_id="turn-1",
        call_id="call-1",
        capability_id="sqlite.update",
        proposed_at=NOW,
        arguments={"source_id": "source-sqlite"},
        validation_facts=facts,
    )
    assert proposal.validation_facts is facts


def test_legacy_validation_defaults_preserve_no_explicit_authority() -> None:
    facts = ActionValidationFacts()
    assert facts.schema_version == 0
    assert facts.fingerprint is None

    with pytest.raises(ValueError, match="legacy"):
        ActionValidationFacts(schema_version=0, source_id="source-sqlite")
    with pytest.raises(ValueError, match="source_id"):
        ActionValidationFacts(schema_version=1)
    with pytest.raises(ValueError, match="schema_version"):
        ActionValidationFacts(schema_version=2, source_id="source-sqlite")
    with pytest.raises(ValueError, match="cover every"):
        ActionValidationFacts(
            schema_version=1,
            source_id="source-sqlite",
            resource_ids=("resource-orders", "resource-users"),
            resource_revisions=(("resource-orders", "sha256:" + ("a" * 64)),),
        )


def test_action_rejection_is_structured_immutable_and_bounded() -> None:
    details = {"field": "key", "expected": "string"}
    rejection = ActionRejection(
        code="invalid_arguments",
        message="The key argument must be a string.",
        details=details,
    )
    details["field"] = "mutated"

    assert isinstance(rejection.details, FrozenJsonObject)
    assert rejection.details.to_dict() == {
        "expected": "string",
        "field": "key",
    }

    with pytest.raises(ValueError, match="bounded"):
        ActionRejection(
            code="invalid_arguments",
            message="x" * 513,
        )

    with pytest.raises(ValueError, match="bounded"):
        ActionRejection(
            code="invalid_arguments",
            message="Invalid arguments.",
            details={"value": "x" * 2048},
        )


def test_observation_links_evidence_to_task_and_preserves_bounded_facts() -> None:
    observation = Observation(
        operation_id="op-1",
        turn_id="turn-1",
        code="fake.read.succeeded",
        message="Read completed.",
        payload={"value": 42},
        success=True,
        task_id="task-1",
        evidence_id="evidence-1",
        truncated=False,
        created_at=NOW,
    )

    assert isinstance(observation.payload, FrozenJsonObject)
    assert observation.payload.to_dict() == {"value": 42}

    with pytest.raises(ValueError, match="task_id"):
        Observation(
            operation_id="op-1",
            turn_id="turn-1",
            code="invalid",
            message="Invalid linkage.",
            payload={},
            success=False,
            evidence_id="evidence-1",
            created_at=NOW,
        )
