from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    Capability,
    CapabilityInputError,
    CapabilityRegistry,
    ToolExecution,
    ToolOutput,
    ToolView,
)
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import ModelSensitivity, ToolCall
from daita.loop.models import LoopLimits, RunInput, ToolBatchOutcome
from _capability_runtime_support import StaticTestDomain, static_registry


class _CountingExecutor:
    executor_id = "test.stage_m1.executor"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        self.calls += 1
        return ToolOutput(kind="test.stage_m1.output")


class _UnavailableDomain(StaticTestDomain):
    async def project(self, run: RunInput) -> tuple[str, ...]:
        del run
        return ()


class _CurrentAdmissionDomain(StaticTestDomain):
    def __init__(
        self,
        capabilities: tuple[Capability, ...],
        tool_views: tuple[ToolView, ...],
    ) -> None:
        super().__init__(capabilities, tool_views)
        self.prepare_calls = 0

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        del run, call, capability, arguments, request_sensitivity
        self.prepare_calls += 1
        raise CapabilityInputError(
            "current_admission_denied",
            "The current domain admission no longer permits this call.",
        )


class _OversizedFailureDomain(StaticTestDomain):
    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        del run, call, capability, arguments, request_sensitivity
        raise CapabilityInputError(
            "typed_boundary_failure",
            "The bounded typed failure occurred.",
            {"remote_diagnostic": "SECRET-REMOTE-DIAGNOSTIC" * 100},
        )


def _declaration() -> tuple[Capability, ToolView]:
    capability = Capability(
        id="test.stage_m1",
        description="Exercise the common capability runtime boundary.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_kind="test.stage_m1.output",
        output_schema={"type": "object", "properties": {}},
        executor_id=_CountingExecutor.executor_id,
    )
    return capability, ToolView(
        name="test_stage_m1",
        capability_id=capability.id,
        description=capability.description,
    )


def _run() -> RunInput:
    return RunInput(
        id="run-stage-m1",
        agent_id="agent-stage-m1",
        message="exercise runtime ownership",
        created_at=datetime(2026, 8, 19, tzinfo=UTC),
    )


def _error_code(outcome: ToolBatchOutcome) -> str:
    error = outcome.ordered_results[0].output["error"]
    assert isinstance(error, Mapping)
    code = error["code"]
    assert isinstance(code, str)
    return code


async def test_unprojected_capability_is_rejected_before_executor_io() -> None:
    capability, view = _declaration()
    executor = _CountingExecutor()
    domain = _UnavailableDomain((capability,), (view,))
    runtime = CapabilityRuntime(static_registry(domain, (executor,)), (domain,))

    outcome = await runtime.execute_all(
        _run(),
        (ToolCall(id="call-unavailable", name=view.name, arguments={"value": "x"}),),
        sensitivity=ModelSensitivity.INTERNAL,
    )

    assert _error_code(outcome) == "tool_not_available"
    assert executor.calls == 0


async def test_schema_and_domain_current_admission_precede_executor_io() -> None:
    capability, view = _declaration()
    executor = _CountingExecutor()
    domain = _CurrentAdmissionDomain((capability,), (view,))
    runtime = CapabilityRuntime(static_registry(domain, (executor,)), (domain,))

    invalid_schema = await runtime.execute_all(
        _run(),
        (ToolCall(id="call-schema", name=view.name),),
        sensitivity=ModelSensitivity.INTERNAL,
    )
    assert _error_code(invalid_schema) == "missing_arguments"
    assert domain.prepare_calls == 0
    assert executor.calls == 0

    denied = await runtime.execute_all(
        _run(),
        (ToolCall(id="call-current", name=view.name, arguments={"value": "x"}),),
        sensitivity=ModelSensitivity.INTERNAL,
    )
    assert _error_code(denied) == "current_admission_denied"
    assert domain.prepare_calls == 1
    assert executor.calls == 0


def test_registry_execution_and_owner_identity_are_immutable() -> None:
    capability, view = _declaration()
    executor = _CountingExecutor()
    domain = StaticTestDomain((capability,), (view,))
    registry = static_registry(domain, (executor,))

    executors = cast(dict[str, object], registry._executors)
    owners = cast(dict[str, str], registry._domain_owners)
    with pytest.raises(TypeError):
        executors[executor.executor_id] = object()
    with pytest.raises(TypeError):
        owners[capability.id] = "replacement"

    resolved_capability, resolved_executor = registry.resolve_execution(capability.id)
    assert resolved_capability is capability
    assert resolved_executor is executor
    assert registry.resolve_domain_owner(capability.id) == domain.domain_owner_id

    executor.executor_id = "test.stage_m1.changed"
    with pytest.raises(ValueError, match="executor identity changed"):
        registry.resolve_execution(capability.id)


async def test_generic_result_bounds_cover_typed_failures_and_redact_payload() -> None:
    capability, view = _declaration()
    executor = _CountingExecutor()
    domain = _OversizedFailureDomain((capability,), (view,))
    runtime = CapabilityRuntime(
        static_registry(domain, (executor,)),
        (domain,),
        limits=LoopLimits(max_tool_result_bytes=128),
    )

    outcome = await runtime.execute_all(
        _run(),
        (ToolCall(id="call-bounded", name=view.name, arguments={"value": "x"}),),
        sensitivity=ModelSensitivity.INTERNAL,
    )

    assert _error_code(outcome) == "tool_result_too_large"
    assert "SECRET-REMOTE-DIAGNOSTIC" not in repr(outcome.ordered_results[0])
    assert executor.calls == 0


@pytest.mark.parametrize(
    "limits, message",
    (
        (LoopLimits(max_tool_result_bytes=128), None),
        ({"max_tool_result_bytes": 127}, "max_tool_result_bytes must be at least 128"),
        ({"max_tool_result_depth": 2}, "max_tool_result_depth must be at least 3"),
    ),
)
def test_loop_limits_can_represent_the_fixed_bounded_error(limits, message) -> None:
    if message is None:
        assert isinstance(limits, LoopLimits)
        return
    with pytest.raises(ValueError, match=message):
        LoopLimits(**limits)
