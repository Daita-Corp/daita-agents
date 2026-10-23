"""Execution limits, durable availability, and model-facing recovery descriptors."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest

from daita import capability_runtime
from daita.capabilities import (
    CallTargetLane,
    CapabilityInputError,
    ExecutionAdmissionPolicy,
    ExecutionPreference,
)
from daita.capability_runtime import (
    APPROVED_TARGET_CHANGED,
    _require_call_target_ceiling,
    call_target_lane,
)
from daita.loop.models import RunOrigin
from tests.capabilities._toolbox_support import (
    AccessMode,
    ApprovalDecision,
    Capability,
    CapabilityRegistry,
    CapabilityRuntime,
    FrozenJsonObject,
    OperationalEffect,
    RunInput,
    StaticTestDomain,
    ToolboxId,
    ToolCall,
    ToolLoadMode,
    ToolResultBlock,
    ToolTextTrust,
    ToolView,
    _append_results,
    _data,
    _error_code,
    _execute,
    _Executor,
    _load,
    _presentation,
    _run,
)

_POLICY = ExecutionAdmissionPolicy(
    shape="test.bounded_read",
    inline_eligible=True,
    graph_v1_eligible=True,
    target_count_argument="tables",
    inline_max_targets=4,
    graph_max_targets=16,
)
_UNADMITTED = ExecutionAdmissionPolicy(
    shape="test.unadmitted",
    inline_eligible=False,
    graph_v1_eligible=False,
    target_count_argument=None,
    inline_max_targets=0,
    graph_max_targets=0,
)
_TABLES = tuple(f"table_{index}" for index in range(8))


class _ApplicabilityDomain(StaticTestDomain):
    """Static domain whose current local applicability can change mid-run."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden: set[str] = set()

    async def project(self, run: RunInput) -> tuple[str, ...]:
        return tuple(
            name for name in await super().project(run) if name not in self.hidden
        )


class _VersionedExecutor(_Executor):
    """Effectful executor whose preflight binding changes on demand."""

    def __init__(self, name: str) -> None:
        super().__init__(name, effectful=True)
        self.version = 0

    async def preflight(self, request) -> FrozenJsonObject:
        self.preflight_calls += 1
        return FrozenJsonObject.from_mapping({"version": self.version})


def _capability(
    name: str,
    executor: _Executor,
    *,
    policy: ExecutionAdmissionPolicy | None = None,
    durable: bool = False,
    effect: OperationalEffect = OperationalEffect.NONE,
) -> Capability:
    properties: dict[str, object] = {"value": {"type": "string", "minLength": 1}}
    if policy is not None:
        properties["tables"] = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "maxItems": 64,
        }
    return Capability(
        id=f"test.recovery.{name}",
        description=f"Execute {name}.",
        input_schema={
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        },
        output_kind=f"test.toolbox.{name}.output",
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        executor_id=executor.executor_id,
        access_mode=(
            AccessMode.NONE
            if durable
            else (
                AccessMode.WRITE
                if effect is not OperationalEffect.NONE
                else AccessMode.READ
            )
        ),
        operational_effect=effect,
        execution_admission_policy=policy,
        durable_foreground_entrypoint=durable,
    )


def _recovery_runtime(
    *,
    durable: bool,
    approval_handler=None,
) -> tuple[CapabilityRuntime, _ApplicabilityDomain, dict[str, _Executor]]:
    executors: dict[str, _Executor] = {
        "bounded_read": _Executor("bounded_read"),
        "guarded_write": _VersionedExecutor("guarded_write"),
        "unadmitted_read": _Executor("unadmitted_read"),
    }
    capabilities = [
        _capability("bounded_read", executors["bounded_read"], policy=_POLICY),
        _capability(
            "guarded_write",
            executors["guarded_write"],
            effect=OperationalEffect.CHANGE_ADVISORY_CONTEXT,
        ),
        _capability(
            "unadmitted_read", executors["unadmitted_read"], policy=_UNADMITTED
        ),
    ]
    load_modes = {
        "bounded_read": ToolLoadMode.PINNED,
        "guarded_write": ToolLoadMode.ON_DEMAND,
        "unadmitted_read": ToolLoadMode.PINNED,
    }
    if durable:
        executors["submit_durable"] = _Executor("submit_durable")
        capabilities.append(
            _capability("submit_durable", executors["submit_durable"], durable=True)
        )
        load_modes["submit_durable"] = ToolLoadMode.ON_DEMAND
    views = tuple(
        ToolView(
            name=capability.id.rsplit(".", 1)[1],
            capability_id=capability.id,
            description=capability.description,
            presentation=_presentation(
                capability.id.rsplit(".", 1)[1],
                (
                    ToolboxId.JOBS
                    if capability.durable_foreground_entrypoint
                    else ToolboxId.SOURCES
                ),
                load_modes[capability.id.rsplit(".", 1)[1]],
                text_trust=ToolTextTrust.CODE,
            ),
        )
        for capability in capabilities
    )
    domain = _ApplicabilityDomain(
        tuple(capabilities), views, domain_owner_id="recovery_test"
    )
    registry = CapabilityRegistry(
        declarations=(domain.declarations,),
        executors=tuple(executors.values()),
    )
    return (
        CapabilityRuntime(registry, (domain,), approval_handler=approval_handler),
        domain,
        executors,
    )


def _details(result: ToolResultBlock) -> Mapping[str, object]:
    error = result.output.get("error")
    assert isinstance(error, Mapping)
    details = error.get("details")
    assert isinstance(details, Mapping)
    return details


def _recovery(result: ToolResultBlock) -> Mapping[str, object]:
    recovery = _details(result).get("recovery")
    assert isinstance(recovery, Mapping)
    return recovery


def _wide_read(call_id: str = "wide") -> ToolCall:
    return ToolCall(id=call_id, name="bounded_read", arguments={"tables": _TABLES})


def _dispatches(executors: Mapping[str, _Executor]) -> int:
    return sum(item.execute_calls + item.preflight_calls for item in executors.values())


async def test_wide_query_without_durable_path_names_bound_and_no_tool():
    runtime, _, executors = _recovery_runtime(durable=False)
    run = _run("run-recovery-no-durable")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())

    result = (await _execute(runtime, run, projection, _wide_read())).ordered_results[0]

    assert _error_code(result) == "durable_execution_required"
    assert _dispatches(executors) == 0
    message = str(result.output["error"]["message"])
    assert message.startswith("No query ran")
    assert "8 tables exceed the per-call bound of 4" in message
    details = _details(result)
    assert details["execution_disposition"] == "not_dispatched"
    assert details["durable_available"] is False
    limits = details["execution_limits"]
    assert isinstance(limits, Mapping)
    assert limits["inline_max_targets"] == 4
    recovery = _recovery(result)
    assert recovery["kind"] == "reduce_scope"
    assert "tool_name" not in recovery
    rendered = str(result.output)
    for name in ("submit_durable", "start_graph_job", "toolbox_load"):
        assert name not in rendered


async def test_limits_appear_once_in_description_and_once_in_guidance():
    runtime, _, _ = _recovery_runtime(durable=True)
    run = _run("run-recovery-limits")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())

    definition = next(
        item for item in projection.provider_definitions if item.name == "bounded_read"
    )
    assert definition.description.endswith("Limits: at most 4 tables per call.")
    assert projection.limits_guidance == (
        "Per-call limits: bounded_read at most 4 tables "
        "(up to 16 through submit_durable after toolbox_load)."
    )


def test_rendered_and_enforced_target_bounds_read_one_lane_policy():
    lanes = {origin: call_target_lane(origin) for origin in RunOrigin}
    assert lanes == {
        RunOrigin.USER: CallTargetLane.FOREGROUND,
        RunOrigin.JOB: CallTargetLane.JOB,
        RunOrigin.JOB_SUBRUN: CallTargetLane.JOB,
        RunOrigin.JOB_TASK: None,
        RunOrigin.JOB_EVENT: None,
        RunOrigin.SCHEDULED_ROUTINE: None,
    }
    assert _POLICY.call_target_ceiling(lanes[RunOrigin.USER]) == 4
    assert _POLICY.call_target_ceiling(lanes[RunOrigin.JOB]) == 16
    for origin, lane in lanes.items():
        ceiling = _POLICY.call_target_ceiling(lane)
        sentence = _POLICY.limits_sentence(lane)
        if ceiling is None:
            assert sentence is None
            continue
        assert sentence == f"Limits: at most {ceiling} tables per call."
        fits = {"tables": tuple(f"t{i}" for i in range(ceiling))}
        over = {"tables": tuple(f"t{i}" for i in range(ceiling + 1))}
        if origin is RunOrigin.USER:
            # Foreground admission: the rendered bound is the inline boundary.
            assert _POLICY.admission_error(fits, ExecutionPreference.AUTO) is None
            assert _POLICY.admission_error(over, ExecutionPreference.AUTO) is not None
        else:
            _require_call_target_ceiling(_POLICY, lane, fits)
            with pytest.raises(CapabilityInputError) as raised:
                _require_call_target_ceiling(_POLICY, lane, over)
            recovery = raised.value.details["recovery"]
            assert isinstance(recovery, Mapping)
            assert recovery["kind"] == "reduce_scope"
            assert recovery["constraint"] == (
                f"Select at most {ceiling} tables per call."
            )


async def test_durable_available_on_demand_names_the_exact_entrypoint():
    runtime, _, executors = _recovery_runtime(durable=True)
    run = _run("run-recovery-durable-on-demand")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())

    result = (await _execute(runtime, run, projection, _wide_read())).ordered_results[0]

    assert _dispatches(executors) == 0
    assert _details(result)["durable_available"] is True
    recovery = _recovery(result)
    assert recovery["kind"] == "reduce_scope"
    assert recovery["tool_name"] == "submit_durable"
    assert "after loading it with toolbox_load" in str(recovery["constraint"])


async def test_durable_loaded_entrypoint_is_named_without_load_advice():
    runtime, _, executors = _recovery_runtime(durable=True)
    run = _run("run-recovery-durable-loaded")
    catalog = await runtime.prepare_run(run)
    load, messages, projection = await _load(
        runtime, run, catalog, (), ("submit_durable",), call_id="load"
    )
    assert not load.is_error

    result = (
        await _execute(runtime, run, projection, _wide_read(), messages=messages)
    ).ordered_results[0]

    recovery = _recovery(result)
    assert recovery["tool_name"] == "submit_durable"
    assert "toolbox_load" not in str(recovery["constraint"])
    assert executors["submit_durable"].execute_calls == 0
    assert "through submit_durable)" in projection.limits_guidance


async def test_registered_but_not_applicable_durable_entrypoint_is_never_suggested():
    runtime, domain, executors = _recovery_runtime(durable=True)
    run = _run("run-recovery-durable-hidden")
    domain.hidden.add("submit_durable")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    assert "submit_durable" not in projection.limits_guidance

    result = (await _execute(runtime, run, projection, _wide_read())).ordered_results[0]

    assert _dispatches(executors) == 0
    assert _details(result)["durable_available"] is False
    assert "tool_name" not in _recovery(result)
    assert "submit_durable" not in str(result.output)

    search = ToolCall(
        id="search", name="toolbox_search", arguments={"query": "submit_durable"}
    )
    found = (await _execute(runtime, run, projection, search)).ordered_results[0]
    assert _data(found)["exact_tool_name"] == "submit_durable"
    assert _data(found)["exact_tool_available"] is False
    assert "submit_durable" not in str(_data(found)["matches"])


async def test_durable_entrypoint_revoked_after_preparation_is_not_suggested():
    runtime, domain, executors = _recovery_runtime(durable=True)
    run = _run("run-recovery-durable-revoked")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    domain.hidden.add("submit_durable")

    result = (await _execute(runtime, run, projection, _wide_read())).ordered_results[0]
    assert _details(result)["durable_available"] is False
    assert "submit_durable" not in str(result.output)

    load_call = ToolCall(
        id="load",
        name="toolbox_load",
        arguments={"tool_names": ("submit_durable",)},
    )
    load = (await _execute(runtime, run, projection, load_call)).ordered_results[0]
    assert _error_code(load) == "toolbox_tool_not_available"
    assert _recovery(load)["kind"] == "terminal"
    names = _details(load)["tool_names"]
    assert isinstance(names, tuple | list)
    assert tuple(names) == ("submit_durable",)
    after = runtime.project(catalog, _append_results((), (load_call,), (load,)))
    assert "submit_durable" not in {item.name for item in after.provider_definitions}
    assert _dispatches(executors) == 0


async def test_disabled_durable_entrypoint_is_absent_from_search_and_recovery():
    runtime, _, _ = _recovery_runtime(durable=False)
    run = _run("run-recovery-durable-disabled")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())

    search = ToolCall(
        id="search", name="toolbox_search", arguments={"query": "submit_durable"}
    )
    found = (await _execute(runtime, run, projection, search)).ordered_results[0]

    assert "exact_tool_available" not in _data(found)
    assert (
        projection.limits_guidance == "Per-call limits: bounded_read at most 4 tables."
    )


async def test_exact_search_reports_available_tool():
    runtime, _, _ = _recovery_runtime(durable=True)
    run = _run("run-recovery-exact-search")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())

    search = ToolCall(
        id="search", name="toolbox_search", arguments={"query": "submit_durable"}
    )
    found = (await _execute(runtime, run, projection, search)).ordered_results[0]

    assert _data(found)["exact_tool_available"] is True


async def test_correct_arguments_recovery_for_invalid_arguments():
    runtime, _, executors = _recovery_runtime(durable=False)
    run = _run("run-recovery-correct-arguments")
    projection = runtime.project(await runtime.prepare_run(run), ())

    bad = ToolCall(id="bad", name="bounded_read", arguments={"unknown": "x"})
    result = (await _execute(runtime, run, projection, bad)).ordered_results[0]

    assert _error_code(result) == "unexpected_arguments"
    assert _recovery(result)["kind"] == "correct_arguments"
    assert _details(result)["execution_disposition"] == "not_dispatched"
    assert _dispatches(executors) == 0


async def test_correct_arguments_recovery_for_unloaded_tool_names_toolbox_load():
    runtime, _, executors = _recovery_runtime(durable=True)
    run = _run("run-recovery-unloaded")
    projection = runtime.project(await runtime.prepare_run(run), ())

    call = ToolCall(id="early", name="submit_durable", arguments={"value": "x"})
    result = (await _execute(runtime, run, projection, call)).ordered_results[0]

    assert _error_code(result) == "tool_not_available"
    assert dict(_recovery(result)) == {
        "kind": "correct_arguments",
        "constraint": "Load submit_durable with toolbox_load, then call it in "
        "the next step.",
        "tool_name": "toolbox_load",
    }
    assert _details(result)["execution_disposition"] == "not_dispatched"
    assert _dispatches(executors) == 0


async def test_drift_during_approval_is_final_stop_and_report():
    executors: dict[str, _Executor] = {}

    async def approve(request):
        del request
        executor = executors["guarded_write"]
        assert isinstance(executor, _VersionedExecutor)
        executor.version += 1
        return ApprovalDecision.APPROVE

    runtime, _, built = _recovery_runtime(durable=False, approval_handler=approve)
    executors.update(built)
    run = _run("run-recovery-reinspect")
    result = await _write_after_load(runtime, run)

    assert _error_code(result) == "state_changed"
    assert dict(_recovery(result)) == APPROVED_TARGET_CHANGED.to_details()
    assert _details(result)["execution_disposition"] == "not_dispatched"
    assert executors["guarded_write"].execute_calls == 0


def test_post_approval_drift_recovery_never_suggests_repreview():
    """Snapshot: every post-approval ``state_changed`` shares one final recovery.

    The runtime owns this recovery for every capability that rechecks after
    approval (native update/upsert, MCP actions and any other), so no write
    path can tell the model to preview or request approval again.
    """

    assert APPROVED_TARGET_CHANGED.to_details() == {
        "kind": "terminal",
        "constraint": (
            "The approved target changed while approval was pending; nothing "
            "was written. Report the conflict and the target's current value "
            "to the user. Do not preview again or request approval again "
            "unless the user asks."
        ),
    }
    tree = ast.parse(Path(capability_runtime.__file__).read_text(encoding="utf-8"))
    drift_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(
            isinstance(argument, ast.Constant) and argument.value == "state_changed"
            for argument in node.args
        )
    ]
    assert len(drift_sites) >= 4
    for node in drift_sites:
        names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
        assert "APPROVED_TARGET_CHANGED" in names, ast.unparse(node)
    reinspect = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "REINSPECT_BINDING"
    ]
    assert reinspect == []


async def _write_after_load(runtime, run: RunInput) -> ToolResultBlock:
    catalog = await runtime.prepare_run(run)
    load, messages, projection = await _load(
        runtime, run, catalog, (), ("guarded_write",), call_id="load"
    )
    assert not load.is_error
    call = ToolCall(id="write", name="guarded_write", arguments={"value": "x"})
    return (
        await _execute(runtime, run, projection, call, messages=messages)
    ).ordered_results[0]


async def test_requires_human_permission_recovery_when_denied():
    async def deny(request):
        del request
        return ApprovalDecision.DENY

    runtime, _, executors = _recovery_runtime(durable=False, approval_handler=deny)
    run = _run("run-recovery-denied")
    result = await _write_after_load(runtime, run)

    assert _error_code(result) == "approval_denied"
    assert _recovery(result)["kind"] == "requires_human_permission"
    assert _details(result)["execution_disposition"] == "not_dispatched"
    assert executors["guarded_write"].execute_calls == 0


async def test_reduce_scope_beyond_every_bound_names_no_tool():
    runtime, _, executors = _recovery_runtime(durable=True)
    run = _run("run-recovery-beyond-graph")
    projection = runtime.project(await runtime.prepare_run(run), ())

    huge = ToolCall(
        id="huge",
        name="bounded_read",
        arguments={"tables": tuple(f"t{index}" for index in range(20))},
    )
    result = (await _execute(runtime, run, projection, huge)).ordered_results[0]

    assert _error_code(result) == "execution_not_supported_for_requested_shape"
    assert _recovery(result)["kind"] == "reduce_scope"
    assert "tool_name" not in _recovery(result)
    assert _details(result)["execution_disposition"] == "not_dispatched"
    assert _dispatches(executors) == 0


async def test_terminal_recovery_when_no_path_admits_the_shape():
    runtime, _, executors = _recovery_runtime(durable=True)
    run = _run("run-recovery-terminal")
    projection = runtime.project(await runtime.prepare_run(run), ())

    call = ToolCall(id="never", name="unadmitted_read", arguments={"value": "x"})
    result = (await _execute(runtime, run, projection, call)).ordered_results[0]

    assert _error_code(result) == "execution_not_supported_for_requested_shape"
    assert _recovery(result)["kind"] == "terminal"
    assert "tool_name" not in _recovery(result)
    assert _details(result)["execution_disposition"] == "not_dispatched"
    assert _dispatches(executors) == 0


async def test_successful_bounded_call_is_dispatched_without_recovery():
    runtime, _, executors = _recovery_runtime(durable=False)
    run = _run("run-recovery-success")
    projection = runtime.project(await runtime.prepare_run(run), ())

    call = ToolCall(id="ok", name="bounded_read", arguments={"tables": _TABLES[:4]})
    result = (await _execute(runtime, run, projection, call)).ordered_results[0]

    assert not result.is_error
    assert executors["bounded_read"].execute_calls == 1
