from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from daita._json import FrozenJsonObject
from daita.events.models import RuntimeEvent
from daita.operations import runtime as operation_runtime

NOW = datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc)


def _event() -> RuntimeEvent:
    return RuntimeEvent(
        id="event-1",
        type="model_call.started",
        agent_id="agent-1",
        operation_id="operation-1",
        session_id="session-1",
        turn_id="turn-1",
        model_call_id="model-call-1",
        call_id="call-1",
        task_id="task-1",
        evidence_id="evidence-1",
        monitor_id="monitor-1",
        capability_id="fake.read",
        executor_id="fake.executor",
        payload={"nested": {"values": [1, 2]}},
        created_at=NOW,
    )


def test_runtime_event_is_provider_neutral_correlated_and_mutation_isolated() -> None:
    values = [1, 2]
    payload: dict[str, object] = {"nested": {"values": values}}

    event = replace(_event(), payload=payload)
    values.append(3)

    assert event.id == "event-1"
    assert event.type == "model_call.started"
    assert event.agent_id == "agent-1"
    assert event.operation_id == "operation-1"
    assert event.session_id == "session-1"
    assert event.turn_id == "turn-1"
    assert event.model_call_id == "model-call-1"
    assert event.call_id == "call-1"
    assert event.task_id == "task-1"
    assert event.evidence_id == "evidence-1"
    assert event.monitor_id == "monitor-1"
    assert event.capability_id == "fake.read"
    assert event.executor_id == "fake.executor"
    assert event.created_at == NOW
    assert isinstance(event.payload, FrozenJsonObject)
    assert event.payload.to_dict() == {"nested": {"values": [1, 2]}}


def test_runtime_event_requires_stable_identity_and_aware_creation_time() -> None:
    event = _event()

    with pytest.raises(ValueError, match="event id.*non-empty|id.*non-empty"):
        replace(event, id=" ")
    with pytest.raises(ValueError, match="event type.*non-empty|type.*non-empty"):
        replace(event, type="")
    with pytest.raises(ValueError, match="agent_id.*non-empty"):
        replace(event, agent_id=" ")
    assert replace(event, operation_id=None).operation_id is None
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(event, created_at=datetime(2026, 7, 17, 15, 0))


@pytest.mark.parametrize(
    "field_name",
    (
        "operation_id",
        "session_id",
        "turn_id",
        "model_call_id",
        "call_id",
        "task_id",
        "evidence_id",
        "monitor_id",
        "capability_id",
        "executor_id",
    ),
)
def test_runtime_event_rejects_blank_optional_correlations(field_name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        replace(_event(), **{field_name: " "})  # type: ignore[arg-type]


def test_runtime_event_rejects_implicit_non_json_payload_coercion() -> None:
    with pytest.raises(TypeError, match="Unsupported JSON value"):
        replace(_event(), payload={"unsafe": object()})


def test_operation_runtime_reexports_the_canonical_event_for_compatibility() -> None:
    assert operation_runtime.RuntimeEvent is RuntimeEvent
