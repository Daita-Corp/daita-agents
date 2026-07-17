from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any

import pytest

from daita.capabilities import AccessMode, RiskLevel
import daita.operations.models as operation_models

NOW = datetime(2026, 7, 17, 18, 0, tzinfo=timezone.utc)
CAPABILITY_FINGERPRINT = "sha256:" + ("a" * 64)
ARGUMENTS_HASH = "sha256:" + ("b" * 64)
TASK_ARGUMENTS_HASH = "sha256:" + hashlib.sha256(b'{"key":"alpha"}').hexdigest()


def _model_type(name: str) -> type[Any]:
    model_type = getattr(operation_models, name, None)
    assert model_type is not None, f"daita.operations.models must define {name}"
    return model_type


def _lease_type() -> type[Any]:
    try:
        leases = importlib.import_module("daita.operations.leases")
    except ModuleNotFoundError as exc:
        pytest.fail(f"daita.operations.leases must exist: {exc}")
    lease_type = getattr(leases, "TaskLease", None)
    assert lease_type is not None, "daita.operations.leases must define TaskLease"
    return lease_type


def _facts(**overrides: object) -> Any:
    values: dict[str, object] = {
        "capability_fingerprint": CAPABILITY_FINGERPRINT,
        "arguments_hash": ARGUMENTS_HASH,
        "access_mode": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "side_effecting": False,
        "idempotent": True,
        "replay_safe": True,
        "idempotency_key": None,
    }
    values.update(overrides)
    return _model_type("TaskExecutionFacts")(**values)


def _task(**overrides: object) -> Any:
    values: dict[str, object] = {
        "id": "task-1",
        "operation_id": "operation-1",
        "turn_id": "turn-1",
        "call_id": "call-1",
        "capability_id": "fake.read",
        "executor_id": "fake.read.executor",
        "status": operation_models.TaskStatus.READY,
        "attempt": 1,
        "arguments": {"key": "alpha"},
        "execution_facts": _facts(arguments_hash=TASK_ARGUMENTS_HASH),
        "created_at": NOW,
        "updated_at": NOW,
        "manual_recovery_reason": None,
    }
    values.update(overrides)
    task_type: Any = operation_models.Task
    return task_type(**values)


def _lease(**overrides: object) -> Any:
    values: dict[str, object] = {
        "operation_id": "operation-1",
        "task_id": "task-1",
        "attempt": 1,
        "fencing_token": 1,
        "holder_id": "runtime-1",
        "acquired_at": NOW,
        "expires_at": NOW + timedelta(seconds=30),
        "started_at": None,
        "renewed_at": None,
        "released_at": None,
        "release_reason": None,
    }
    values.update(overrides)
    return _lease_type()(**values)


def test_task_status_has_the_exact_durable_execution_vocabulary() -> None:
    assert [status.value for status in operation_models.TaskStatus] == [
        "pending",
        "ready",
        "claimed",
        "running",
        "waiting_for_approval",
        "succeeded",
        "failed",
        "cancelled",
        "manual_recovery_required",
    ]


def test_task_execution_facts_are_typed_frozen_materialization_facts() -> None:
    facts = _facts()

    assert facts.capability_fingerprint == CAPABILITY_FINGERPRINT
    assert facts.arguments_hash == ARGUMENTS_HASH
    assert facts.access_mode is AccessMode.READ
    assert facts.risk is RiskLevel.LOW
    assert facts.side_effecting is False
    assert facts.idempotent is True
    assert facts.replay_safe is True
    assert facts.idempotency_key is None

    with pytest.raises(FrozenInstanceError):
        setattr(facts, "replay_safe", False)


@pytest.mark.parametrize("field_name", ["capability_fingerprint", "arguments_hash"])
@pytest.mark.parametrize(
    "invalid_hash",
    [
        "sha256:" + ("a" * 63),
        "sha256:" + ("A" * 64),
        "sha256:" + ("g" * 64),
        "md5:" + ("a" * 64),
        "a" * 64,
        "",
    ],
)
def test_task_execution_facts_require_canonical_sha256_hashes(
    field_name: str,
    invalid_hash: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _facts(**{field_name: invalid_hash})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("access_mode", "read"),
        ("risk", "low"),
        ("side_effecting", 0),
        ("idempotent", 1),
        ("replay_safe", None),
    ],
)
def test_task_execution_facts_reject_untyped_safety_facts(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        _facts(**{field_name: invalid_value})


def test_task_execution_facts_reject_impossible_read_and_replay_claims() -> None:
    with pytest.raises(ValueError, match="read.*side effect"):
        _facts(side_effecting=True)

    with pytest.raises(ValueError, match="replay_safe.*idempotent"):
        _facts(idempotent=False, replay_safe=True)


def test_side_effect_replay_requires_a_stable_idempotency_key() -> None:
    with pytest.raises(ValueError, match="idempotency_key"):
        _facts(
            access_mode=AccessMode.WRITE,
            side_effecting=True,
            idempotent=True,
            replay_safe=True,
        )

    facts = _facts(
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.HIGH,
        side_effecting=True,
        idempotent=True,
        replay_safe=True,
        idempotency_key="operation-1:task-1",
    )
    assert facts.idempotency_key == "operation-1:task-1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"idempotency_key": "read-key"},
        {
            "access_mode": AccessMode.WRITE,
            "side_effecting": False,
            "idempotency_key": "write-key",
        },
        {
            "access_mode": AccessMode.WRITE,
            "side_effecting": True,
            "idempotent": False,
            "replay_safe": False,
            "idempotency_key": "unsafe-key",
        },
        {
            "access_mode": AccessMode.WRITE,
            "side_effecting": True,
            "idempotent": True,
            "replay_safe": False,
            "idempotency_key": "   ",
        },
    ],
)
def test_idempotency_keys_are_only_valid_for_idempotent_side_effects(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="idempotency_key"):
        _facts(**overrides)


def test_task_persists_execution_facts_and_is_immutable() -> None:
    facts = _facts(arguments_hash=TASK_ARGUMENTS_HASH)
    task = _task(execution_facts=facts)

    assert task.execution_facts is facts
    assert task.manual_recovery_reason is None

    with pytest.raises(FrozenInstanceError):
        setattr(task, "execution_facts", _facts(replay_safe=False))

    with pytest.raises(TypeError, match="execution_facts"):
        _task(execution_facts=None)


def test_task_binds_nonlegacy_arguments_hash_to_canonical_arguments() -> None:
    with pytest.raises(ValueError, match="arguments_hash.*arguments"):
        _task(execution_facts=_facts(arguments_hash=ARGUMENTS_HASH))


def test_manual_recovery_task_requires_an_explicit_reason() -> None:
    manual = _task(
        status=operation_models.TaskStatus.MANUAL_RECOVERY_REQUIRED,
        manual_recovery_reason="unknown_side_effect_outcome",
    )
    assert manual.manual_recovery_reason == "unknown_side_effect_outcome"

    with pytest.raises(ValueError, match="manual recovery reason"):
        _task(
            status=operation_models.TaskStatus.MANUAL_RECOVERY_REQUIRED,
            manual_recovery_reason=None,
        )

    with pytest.raises(ValueError, match="manual recovery reason"):
        _task(
            status=operation_models.TaskStatus.MANUAL_RECOVERY_REQUIRED,
            manual_recovery_reason="   ",
        )


def test_only_manual_recovery_tasks_may_carry_a_manual_recovery_reason() -> None:
    with pytest.raises(ValueError, match="manual recovery"):
        _task(
            status=operation_models.TaskStatus.READY,
            manual_recovery_reason="not-applicable",
        )


def test_existing_terminal_task_payload_rules_apply_to_new_statuses() -> None:
    succeeded = _task(
        status=operation_models.TaskStatus.SUCCEEDED,
        evidence_ids=("evidence-1",),
    )
    failed = _task(
        status=operation_models.TaskStatus.FAILED,
        error_code="executor_failed",
    )
    cancelled = _task(status=operation_models.TaskStatus.CANCELLED)

    assert succeeded.evidence_ids == ("evidence-1",)
    assert failed.error_code == "executor_failed"
    assert cancelled.evidence_ids == ()

    with pytest.raises(ValueError, match="accepted evidence"):
        _task(status=operation_models.TaskStatus.SUCCEEDED)
    with pytest.raises(ValueError, match="error_code"):
        _task(status=operation_models.TaskStatus.FAILED)
    with pytest.raises(ValueError, match="only failed"):
        _task(status=operation_models.TaskStatus.CANCELLED, error_code="cancelled")


def test_task_dependency_is_a_strict_immutable_link() -> None:
    dependency_type = _model_type("TaskDependency")
    dependency = dependency_type(
        operation_id="operation-1",
        task_id="task-2",
        prerequisite_task_id="task-1",
    )

    assert dependency.operation_id == "operation-1"
    assert dependency.task_id == "task-2"
    assert dependency.prerequisite_task_id == "task-1"
    with pytest.raises(FrozenInstanceError):
        setattr(dependency, "prerequisite_task_id", "task-3")


@pytest.mark.parametrize(
    "field_name",
    ["operation_id", "task_id", "prerequisite_task_id"],
)
def test_task_dependency_requires_every_link_identity(field_name: str) -> None:
    values = {
        "operation_id": "operation-1",
        "task_id": "task-2",
        "prerequisite_task_id": "task-1",
    }
    values[field_name] = "   "
    with pytest.raises(ValueError, match=field_name):
        _model_type("TaskDependency")(**values)


def test_task_dependency_rejects_self_dependency() -> None:
    with pytest.raises(ValueError, match="depend.*itself|self"):
        _model_type("TaskDependency")(
            operation_id="operation-1",
            task_id="task-1",
            prerequisite_task_id="task-1",
        )


def test_task_lease_is_a_typed_immutable_attempt_record() -> None:
    lease = _lease()

    assert lease.operation_id == "operation-1"
    assert lease.task_id == "task-1"
    assert lease.attempt == 1
    assert lease.fencing_token == 1
    assert lease.holder_id == "runtime-1"
    assert lease.acquired_at == NOW
    assert lease.expires_at == NOW + timedelta(seconds=30)
    assert lease.started_at is None
    assert lease.renewed_at is None
    assert lease.released_at is None
    assert lease.release_reason is None

    with pytest.raises(FrozenInstanceError):
        setattr(lease, "holder_id", "runtime-2")


@pytest.mark.parametrize("field_name", ["operation_id", "task_id", "holder_id"])
def test_task_lease_requires_every_identity(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _lease(**{field_name: "   "})


@pytest.mark.parametrize("field_name", ["attempt", "fencing_token"])
@pytest.mark.parametrize("invalid_value", [0, -1, True, 1.5])
def test_task_lease_requires_positive_integer_attempt_and_fence(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _lease(**{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("acquired_at", datetime(2026, 7, 17, 18, 0)),
        ("expires_at", datetime(2026, 7, 17, 18, 1)),
        ("started_at", datetime(2026, 7, 17, 18, 0)),
        ("renewed_at", datetime(2026, 7, 17, 18, 0)),
        ("released_at", datetime(2026, 7, 17, 18, 0)),
    ],
)
def test_task_lease_timestamps_must_be_timezone_aware(
    field_name: str,
    invalid_value: datetime,
) -> None:
    overrides: dict[str, object] = {field_name: invalid_value}
    if field_name == "released_at":
        overrides["release_reason"] = "completed"
    with pytest.raises(ValueError, match=field_name):
        _lease(**overrides)


def test_task_lease_requires_strict_acquisition_and_expiry_chronology() -> None:
    with pytest.raises(ValueError, match="expires_at"):
        _lease(expires_at=NOW)

    with pytest.raises(ValueError, match="expires_at"):
        _lease(expires_at=NOW - timedelta(microseconds=1))


@pytest.mark.parametrize("field_name", ["started_at", "renewed_at"])
def test_active_lease_checkpoints_must_fall_inside_the_live_interval(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _lease(**{field_name: NOW - timedelta(microseconds=1)})

    with pytest.raises(ValueError, match=field_name):
        _lease(**{field_name: NOW + timedelta(seconds=30)})


def test_task_lease_release_fields_are_paired_and_chronological() -> None:
    released = _lease(
        started_at=NOW + timedelta(seconds=1),
        renewed_at=NOW + timedelta(seconds=10),
        released_at=NOW + timedelta(seconds=20),
        release_reason="succeeded",
    )
    assert released.release_reason == "succeeded"

    with pytest.raises(ValueError, match="release"):
        _lease(released_at=NOW + timedelta(seconds=1))
    with pytest.raises(ValueError, match="release"):
        _lease(release_reason="succeeded")
    with pytest.raises(ValueError, match="release_reason"):
        _lease(
            released_at=NOW + timedelta(seconds=1),
            release_reason="   ",
        )
    with pytest.raises(ValueError, match="released_at"):
        _lease(
            started_at=NOW + timedelta(seconds=2),
            released_at=NOW + timedelta(seconds=1),
            release_reason="released_before_start",
        )
    with pytest.raises(ValueError, match="released_at"):
        _lease(
            renewed_at=NOW + timedelta(seconds=2),
            released_at=NOW + timedelta(seconds=1),
            release_reason="released_before_renewal",
        )


def test_expired_attempt_may_record_recovery_release_after_expiry() -> None:
    released = _lease(
        expires_at=NOW + timedelta(seconds=5),
        released_at=NOW + timedelta(seconds=10),
        release_reason="expired_reclaimed",
    )

    assert released.released_at > released.expires_at


def test_claimed_work_may_renew_before_executor_start() -> None:
    lease = _lease(
        renewed_at=NOW + timedelta(seconds=2),
        started_at=NOW + timedelta(seconds=3),
    )

    assert lease.renewed_at < lease.started_at
