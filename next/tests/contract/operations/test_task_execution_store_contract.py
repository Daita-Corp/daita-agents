from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
import inspect
from typing import Any, cast

import pytest

import daita.operations.leases as leases_module
import daita.operations.store as store_module
from daita.events.models import RuntimeEvent
from daita.operations.leases import TaskClaimRequest, TaskLeaseGuard
from daita.operations.models import TaskStatus
from daita.operations.store import (
    ExpiredTaskLeaseError,
    OperationStore,
    OperationStoreError,
    StaleTaskFenceError,
    TaskClaimConflictError,
    TaskClaimResult,
    TaskDependenciesNotReadyError,
    TaskExecutionStore,
    TaskNotClaimableError,
    TaskNotFoundError,
)

NOW = datetime(2026, 7, 17, 18, 0, tzinfo=timezone.utc)
EXPIRES_AT = NOW + timedelta(seconds=30)
_replace: object = replace


def _dynamic_replace(record: object, **changes: object) -> object:
    replace_callable = cast(object, _replace)
    assert callable(replace_callable)
    return replace_callable(record, **changes)


def _claim_event(
    *,
    operation_id: str = "operation-1",
    task_id: str = "task-1",
) -> RuntimeEvent:
    return RuntimeEvent(
        id="event-task-claimed-1",
        type="task.claimed",
        agent_id="agent-1",
        operation_id=operation_id,
        task_id=task_id,
        capability_id="fake.read",
        executor_id="fake.executor",
        payload={"holder_id": "worker-1"},
        created_at=NOW,
    )


def _claim_request() -> TaskClaimRequest:
    return TaskClaimRequest(
        operation_id="operation-1",
        task_id="task-1",
        holder_id="worker-1",
        acquired_at=NOW,
        expires_at=EXPIRES_AT,
        event=_claim_event(),
    )


def _lease_guard() -> TaskLeaseGuard:
    return TaskLeaseGuard(
        operation_id="operation-1",
        task_id="task-1",
        holder_id="worker-1",
        attempt=2,
        fencing_token=7,
        checked_at=NOW + timedelta(seconds=1),
    )


def test_claim_request_is_an_immutable_portable_atomic_claim_candidate() -> None:
    request = _claim_request()

    assert is_dataclass(TaskClaimRequest)
    assert tuple(field.name for field in fields(TaskClaimRequest)) == (
        "operation_id",
        "task_id",
        "holder_id",
        "acquired_at",
        "expires_at",
        "event",
    )
    assert request.operation_id == "operation-1"
    assert request.task_id == "task-1"
    assert request.holder_id == "worker-1"
    assert request.acquired_at == NOW
    assert request.expires_at == EXPIRES_AT
    assert request.event == _claim_event()
    with pytest.raises(FrozenInstanceError):
        setattr(request, "holder_id", "worker-2")


def test_claim_request_rejects_ambiguous_identity_and_chronology() -> None:
    request = _claim_request()

    for field_name in ("operation_id", "task_id", "holder_id"):
        for value in ("", " "):
            with pytest.raises(ValueError, match=f"{field_name}.*non-empty"):
                _dynamic_replace(request, **{field_name: value})
        with pytest.raises((TypeError, ValueError), match=field_name):
            _dynamic_replace(request, **{field_name: cast(str, object())})

    for field_name in ("acquired_at", "expires_at"):
        with pytest.raises(ValueError, match=f"{field_name}.*timezone-aware"):
            _dynamic_replace(request, **{field_name: NOW.replace(tzinfo=None)})

    for invalid_expiry in (NOW, NOW - timedelta(microseconds=1)):
        with pytest.raises(ValueError, match="expires_at.*after.*acquired_at"):
            _dynamic_replace(request, expires_at=invalid_expiry)


def test_claim_request_requires_an_event_bound_to_the_exact_task() -> None:
    request = _claim_request()

    with pytest.raises(TypeError, match="event.*RuntimeEvent"):
        replace(request, event=cast(RuntimeEvent, object()))
    with pytest.raises(ValueError, match="event.*operation|operation.*event"):
        replace(request, event=replace(request.event, operation_id="operation-2"))
    with pytest.raises(ValueError, match="event.*task|task.*event"):
        replace(request, event=replace(request.event, task_id="task-2"))


def test_lease_guard_is_an_immutable_strict_fencing_proof() -> None:
    guard = _lease_guard()

    assert is_dataclass(TaskLeaseGuard)
    assert tuple(field.name for field in fields(TaskLeaseGuard)) == (
        "operation_id",
        "task_id",
        "holder_id",
        "attempt",
        "fencing_token",
        "checked_at",
    )
    assert guard.operation_id == "operation-1"
    assert guard.task_id == "task-1"
    assert guard.holder_id == "worker-1"
    assert guard.attempt == 2
    assert guard.fencing_token == 7
    assert guard.checked_at == NOW + timedelta(seconds=1)
    with pytest.raises(FrozenInstanceError):
        setattr(guard, "fencing_token", 8)


def test_lease_guard_rejects_ambiguous_identity_time_attempt_and_fence() -> None:
    guard = _lease_guard()

    for field_name in ("operation_id", "task_id", "holder_id"):
        for invalid_identity in ("", " "):
            with pytest.raises(ValueError, match=f"{field_name}.*non-empty"):
                _dynamic_replace(guard, **{field_name: invalid_identity})
        with pytest.raises((TypeError, ValueError), match=field_name):
            _dynamic_replace(guard, **{field_name: cast(str, object())})

    with pytest.raises(ValueError, match="checked_at.*timezone-aware"):
        _dynamic_replace(guard, checked_at=guard.checked_at.replace(tzinfo=None))

    for field_name in ("attempt", "fencing_token"):
        for invalid_number in (0, -1, True):
            with pytest.raises(ValueError, match=f"{field_name}.*positive integer"):
                _dynamic_replace(guard, **{field_name: invalid_number})
        with pytest.raises((TypeError, ValueError), match=f"{field_name}.*positive"):
            _dynamic_replace(guard, **{field_name: cast(int, object())})


def test_task_claim_result_is_an_immutable_three_record_envelope() -> None:
    task_claim_result_type: Any = TaskClaimResult
    assert is_dataclass(TaskClaimResult)
    assert tuple(field.name for field in fields(TaskClaimResult)) == (
        "commit_result",
        "task",
        "lease",
    )
    dataclass_parameters = getattr(TaskClaimResult, "__dataclass_params__", None)
    assert dataclass_parameters is not None
    assert dataclass_parameters.frozen
    assert tuple(task_claim_result_type.__slots__) == (
        "commit_result",
        "task",
        "lease",
    )
    assert tuple(inspect.signature(task_claim_result_type).parameters) == (
        "commit_result",
        "task",
        "lease",
    )


def test_task_execution_store_extends_only_the_portable_operation_contract() -> None:
    task_execution_store_type = cast(type[object], TaskExecutionStore)
    assert OperationStore in task_execution_store_type.__mro__[1:]
    assert getattr(TaskExecutionStore, "_is_protocol", False)

    declared_methods = {
        name
        for name, value in TaskExecutionStore.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert declared_methods == {
        "claim_task",
        "renew_task_lease",
        "commit_fenced",
        "recover_expired_task",
    }

    expected_parameters = {
        "claim_task": ("self", "request", "expected_revision"),
        "renew_task_lease": ("self", "snapshot", "expected_revision", "guard"),
        "commit_fenced": ("self", "snapshot", "expected_revision", "guard"),
        "recover_expired_task": (
            "self",
            "snapshot",
            "expected_revision",
            "guard",
        ),
    }
    for method_name, parameter_names in expected_parameters.items():
        method = getattr(TaskExecutionStore, method_name)
        assert inspect.iscoroutinefunction(method)
        signature = inspect.signature(method)
        assert tuple(signature.parameters) == parameter_names
        assert signature.parameters[parameter_names[1]].kind is (
            inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        for parameter_name in parameter_names[2:]:
            parameter = signature.parameters[parameter_name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is inspect.Parameter.empty


def test_portable_task_execution_contract_has_no_sqlite_dependency() -> None:
    for module in (leases_module, store_module):
        source = inspect.getsource(module)
        tree = ast.parse(source)
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.append(node.module or "")
                imported_names.extend(alias.name for alias in node.names)
        assert not any("sqlite" in name.lower() for name in imported_names)

    portable_annotations = (
        TaskClaimRequest.__annotations__,
        TaskLeaseGuard.__annotations__,
        TaskClaimResult.__annotations__,
        *(
            getattr(TaskExecutionStore, method_name).__annotations__
            for method_name in (
                "claim_task",
                "renew_task_lease",
                "commit_fenced",
                "recover_expired_task",
            )
        ),
    )
    assert "sqlite" not in repr(portable_annotations).lower()


def test_task_execution_errors_preserve_stable_machine_readable_facts() -> None:
    dependency_ids = ("task-parent-a", "task-parent-b")
    errors = (
        TaskNotFoundError("operation-1", "task-1"),
        TaskDependenciesNotReadyError(
            "operation-1",
            "task-1",
            dependency_ids,
        ),
        TaskClaimConflictError(
            "operation-1",
            "task-1",
            "worker-current",
            7,
            EXPIRES_AT,
        ),
        TaskNotClaimableError(
            "operation-1",
            "task-1",
            TaskStatus.RUNNING,
        ),
        StaleTaskFenceError("operation-1", "task-1", 7, 8),
        ExpiredTaskLeaseError(
            "operation-1",
            "task-1",
            7,
            EXPIRES_AT,
            EXPIRES_AT,
        ),
    )

    assert all(isinstance(error, OperationStoreError) for error in errors)
    assert all(error.operation_id == "operation-1" for error in errors)
    assert all(error.task_id == "task-1" for error in errors)
    assert all("operation-1" in str(error) for error in errors)
    assert all("task-1" in str(error) for error in errors)

    dependency_error = cast(TaskDependenciesNotReadyError, errors[1])
    assert dependency_error.dependency_ids == dependency_ids

    conflict_error = cast(TaskClaimConflictError, errors[2])
    assert conflict_error.holder_id == "worker-current"
    assert conflict_error.fencing_token == 7
    assert conflict_error.expires_at == EXPIRES_AT

    not_claimable_error = cast(TaskNotClaimableError, errors[3])
    assert not_claimable_error.status is TaskStatus.RUNNING

    stale_error = cast(StaleTaskFenceError, errors[4])
    assert stale_error.expected_fencing_token == 7
    assert stale_error.actual_fencing_token == 8

    expired_error = cast(ExpiredTaskLeaseError, errors[5])
    assert expired_error.fencing_token == 7
    assert expired_error.expires_at == EXPIRES_AT
    assert expired_error.checked_at == EXPIRES_AT


def test_task_execution_error_constructors_require_the_stable_facts() -> None:
    expected_parameters = {
        TaskNotFoundError: ("operation_id", "task_id"),
        TaskDependenciesNotReadyError: (
            "operation_id",
            "task_id",
            "dependency_ids",
        ),
        TaskClaimConflictError: (
            "operation_id",
            "task_id",
            "holder_id",
            "fencing_token",
            "expires_at",
        ),
        TaskNotClaimableError: ("operation_id", "task_id", "status"),
        StaleTaskFenceError: (
            "operation_id",
            "task_id",
            "expected_fencing_token",
            "actual_fencing_token",
        ),
        ExpiredTaskLeaseError: (
            "operation_id",
            "task_id",
            "fencing_token",
            "expires_at",
            "checked_at",
        ),
    }

    for error_type, parameter_names in expected_parameters.items():
        signature = inspect.signature(error_type)
        assert tuple(signature.parameters) == parameter_names
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and parameter.default is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        )
