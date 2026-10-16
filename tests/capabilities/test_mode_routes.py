from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from daita.capabilities import (
    Capability,
    CapabilityDeclarations,
    CapabilityRegistry,
    ToolExecution,
    ToolOutput,
    ToolView,
    omit_unset_arguments,
)
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import ToolCall
from daita.loop.models import RunInput, ToolBatchOutcome
from tests.support.capability_runtime import (
    StaticTestDomain,
    execute_projected,
    presentation_metadata,
)


class _RecordingExecutor:
    def __init__(self, executor_id: str, kind: str) -> None:
        self.executor_id = executor_id
        self._kind = kind
        self.arguments: list[dict[str, object]] = []

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.arguments.append(dict(request.arguments))
        return ToolOutput(kind=self._kind, data={"received": len(self.arguments)})


def _schema(*extra: str) -> dict[str, object]:
    properties: dict[str, object] = {"target": {"type": "string"}}
    properties.update({name: {"type": "string"} for name in extra})
    return {
        "type": "object",
        "properties": properties,
        "required": ["target", *extra],
        "additionalProperties": False,
    }


def _capabilities() -> tuple[Capability, Capability]:
    preview = Capability(
        id="test.mode.preview",
        description="Preview the target.",
        input_schema=_schema(),
        output_kind="test.mode.preview.output",
        output_schema={"type": "object", "properties": {}},
        executor_id="test.mode.preview.executor",
    )
    apply = Capability(
        id="test.mode.apply",
        description="Apply the previewed target.",
        input_schema=_schema("fingerprint"),
        output_kind="test.mode.apply.output",
        output_schema={"type": "object", "properties": {}},
        executor_id="test.mode.apply.executor",
    )
    return preview, apply


def _view(**changes: Any) -> ToolView:
    view = ToolView(
        name="test_mode_rows",
        capability_id="test.mode.apply",
        description="Preview, then apply, one target.",
        presentation=presentation_metadata(),
        mode_capability_ids=(
            ("preview", "test.mode.preview"),
            ("apply", "test.mode.apply"),
        ),
    )
    return replace(view, **changes)


def _fixture() -> tuple[
    CapabilityRegistry,
    CapabilityRuntime,
    _RecordingExecutor,
    _RecordingExecutor,
]:
    preview, apply = _capabilities()
    domain = StaticTestDomain((preview, apply), (_view(),))
    preview_executor = _RecordingExecutor(preview.executor_id, preview.output_kind)
    apply_executor = _RecordingExecutor(apply.executor_id, apply.output_kind)
    registry = CapabilityRegistry(
        declarations=(domain.declarations,),
        executors=(preview_executor, apply_executor),
    )
    return (
        registry,
        CapabilityRuntime(registry, (domain,)),
        preview_executor,
        apply_executor,
    )


def _run() -> RunInput:
    return RunInput(
        id="run-mode-routes",
        agent_id="agent-mode-routes",
        message="exercise mode routes",
        created_at=datetime(2026, 10, 15, tzinfo=UTC),
    )


def _error_code(outcome: ToolBatchOutcome) -> str:
    error = outcome.ordered_results[0].output["error"]
    assert isinstance(error, Mapping)
    code = error["code"]
    assert isinstance(code, str)
    return code


def test_mode_routed_tool_exposes_one_merged_schema_over_existing_capabilities():
    registry, _runtime, _preview, _apply = _fixture()

    definition = registry.tool_definition("test_mode_rows")
    schema = definition.input_schema
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    assert dict(properties["mode"]) == {"type": "string", "enum": ("preview", "apply")}
    assert set(properties) == {"mode", "target", "fingerprint"}
    assert schema["required"] == ("mode", "target")
    assert definition.description == "Preview, then apply, one target."

    assert registry.tool_capability_ids("test_mode_rows") == (
        "test.mode.apply",
        "test.mode.preview",
    )
    assert registry.resolve_tool("test_mode_rows")[1].id == "test.mode.apply"
    assert (
        registry.resolve_tool_mode("test_mode_rows", "preview")[1].id
        == "test.mode.preview"
    )
    assert (
        registry.call_capability("test_mode_rows", {"mode": "preview"}).id
        == "test.mode.preview"
    )
    assert registry.target_arguments(
        "test_mode_rows", {"mode": "apply", "target": "t"}
    ) == {"target": "t"}
    with pytest.raises(KeyError):
        registry.resolve_tool_mode("test_mode_rows", "delete")
    with pytest.raises(KeyError):
        registry.resolve_tool("test_mode_rows", capability_id="test.other")


@pytest.mark.parametrize(
    "changes,message",
    [
        (
            {"mode_capability_ids": (("preview", "test.mode.preview"),)},
            "tool mode routes are invalid",
        ),
        (
            {
                "mode_capability_ids": (
                    ("preview", "test.mode.preview"),
                    ("dry_run", "test.mode.preview"),
                )
            },
            "tool mode routes are invalid",
        ),
        ({"capability_id": "test.mode.other"}, "tool mode routes are invalid"),
    ],
)
def test_mode_routes_reject_ambiguous_declarations(changes, message):
    with pytest.raises(ValueError, match=message):
        _view(**changes)


def test_mode_routes_require_compatible_closed_route_schemas():
    preview, apply = _capabilities()
    conflicting = replace(
        preview,
        input_schema={
            **_schema(),
            "properties": {"target": {"type": "integer"}},
        },
    )
    executors = (
        _RecordingExecutor(preview.executor_id, preview.output_kind),
        _RecordingExecutor(apply.executor_id, apply.output_kind),
    )
    with pytest.raises(ValueError, match="disagree on argument target"):
        CapabilityRegistry(
            declarations=(
                CapabilityDeclarations(
                    domain_owner_id="test",
                    capabilities=(conflicting, apply),
                    executor_ids=(preview.executor_id, apply.executor_id),
                    tool_views=(_view(),),
                ),
            ),
            executors=executors,
        )
    open_schema = replace(
        preview, input_schema={"type": "object", "properties": {"target": {}}}
    )
    with pytest.raises(ValueError, match="closed object schemas"):
        CapabilityRegistry(
            declarations=(
                CapabilityDeclarations(
                    domain_owner_id="test",
                    capabilities=(open_schema, apply),
                    executor_ids=(preview.executor_id, apply.executor_id),
                    tool_views=(_view(),),
                ),
            ),
            executors=executors,
        )


async def test_mode_selects_the_existing_capability_and_strips_the_selector():
    _registry, runtime, preview_executor, apply_executor = _fixture()

    outcome = await execute_projected(
        runtime,
        _run(),
        (
            ToolCall(
                id="preview",
                name="test_mode_rows",
                arguments={"mode": "preview", "target": "row-1"},
            ),
            ToolCall(
                id="apply",
                name="test_mode_rows",
                arguments={"mode": "apply", "target": "row-1", "fingerprint": "f"},
            ),
        ),
    )

    preview_result, apply_result = outcome.ordered_results
    assert not preview_result.is_error, preview_result.output
    assert not apply_result.is_error, apply_result.output
    assert preview_result.capability_id == "test.mode.preview"
    assert preview_result.output["kind"] == "test.mode.preview.output"
    assert apply_result.capability_id == "test.mode.apply"
    assert apply_result.output["kind"] == "test.mode.apply.output"
    assert preview_executor.arguments == [{"target": "row-1"}]
    assert apply_executor.arguments == [{"target": "row-1", "fingerprint": "f"}]


@pytest.mark.parametrize(
    "arguments,code",
    [
        ({"target": "row-1"}, "invalid_argument_value"),
        ({"mode": "delete", "target": "row-1"}, "invalid_argument_value"),
        ({"mode": "apply", "target": "row-1"}, "missing_arguments"),
    ],
)
async def test_invalid_mode_or_route_arguments_never_reach_an_executor(arguments, code):
    _registry, runtime, preview_executor, apply_executor = _fixture()

    outcome = await execute_projected(
        runtime,
        _run(),
        (ToolCall(id="call", name="test_mode_rows", arguments=arguments),),
    )

    assert _error_code(outcome) == code
    assert preview_executor.arguments == apply_executor.arguments == []


async def test_unset_null_of_another_mode_reaches_the_selected_route_as_omitted():
    _registry, runtime, preview_executor, apply_executor = _fixture()

    # Models often send every listed key, writing null for the ones unused.
    outcome = await execute_projected(
        runtime,
        _run(),
        (
            ToolCall(
                id="preview",
                name="test_mode_rows",
                arguments={"mode": "preview", "target": "row-1", "fingerprint": None},
            ),
        ),
    )

    assert not outcome.ordered_results[0].is_error, outcome.ordered_results[0].output
    assert preview_executor.arguments == [{"target": "row-1"}]
    assert apply_executor.arguments == []


def test_only_nulls_no_schema_accepts_are_omitted_at_every_level():
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "value": {"type": ["string", "null"]},
            "choice": {"enum": ["a", None]},
            "schedule": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "weekdays": {"type": "array"},
                },
            },
        },
    }
    arguments = {
        "id": None,
        "value": None,
        "choice": None,
        "unknown": None,
        "schedule": {"kind": "calendar", "weekdays": None},
    }

    assert omit_unset_arguments(arguments, (schema,)) == {
        "value": None,
        "choice": None,
        # An undeclared key is left for the route check to reject by name.
        "unknown": None,
        "schedule": {"kind": "calendar"},
    }
    # A null that any candidate route accepts is kept.
    nullable = {"type": "object", "properties": {"id": {"type": ["string", "null"]}}}
    assert omit_unset_arguments({"id": None}, (schema, nullable)) == {"id": None}
    unchanged = {"id": "x"}
    assert omit_unset_arguments(unchanged, (schema,)) is unchanged
