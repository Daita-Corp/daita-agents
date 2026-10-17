"""Model-visible result references and their exact canonical resolution."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from daita import Agent
from daita._json import canonical_json
from daita.capabilities import (
    AccessMode,
    CallTargetLane,
    Capability,
    CapabilityRegistry,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolTextTrust,
    ToolView,
)
from daita.capability_runtime import CapabilityRuntime
from daita.domains.data.export_capabilities import ARTIFACT_CREATE_TOOL_NAME
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    model_tool_result,
    tool_result_refs,
)
from daita.llm.providers.anthropic.messages import _message_input
from daita.llm.providers.gemini.messages import _gemini_contents
from daita.llm.providers.openai.messages import _response_input
from daita.llm.providers.openai_compatible.messages import _chat_messages
from daita.llm.providers.subscription_cli.envelope import _project_message
from tests.capabilities._toolbox_support import (
    StaticTestDomain,
    _append_results,
    _execute,
    _presentation,
    _run,
)
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

_CANONICAL = ("call-canonical-a", "call-canonical-b")


def _conversation() -> tuple[CanonicalMessage, ...]:
    return (
        CanonicalMessage(MessageRole.USER, content=(TextBlock("look it up"),)),
        CanonicalMessage(
            MessageRole.ASSISTANT,
            tool_calls=(ToolCall(_CANONICAL[0], "lookup", {"query": "a"}),),
        ),
        CanonicalMessage(
            MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=_CANONICAL[0],
                    output={"kind": "test.lookup", "data": {"value": "a"}},
                ),
            ),
        ),
        CanonicalMessage(
            MessageRole.ASSISTANT,
            tool_calls=(ToolCall(_CANONICAL[1], "lookup", {"query": "b"}),),
        ),
        CanonicalMessage(
            MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=_CANONICAL[1],
                    output={"error": {"code": "lookup_failed", "message": "no"}},
                    is_error=True,
                ),
            ),
        ),
    )


def _rendered_refs(name: str, request: ModelRequest) -> list[str]:
    refs = request.tool_result_refs
    if name == "openai":
        return [
            json.loads(str(item["output"]))["result_ref"]
            for item in _response_input(request.messages, "openai", refs)
            if item.get("type") == "function_call_output"
        ]
    if name == "openai_compatible":
        return [
            json.loads(str(item["content"]))["result_ref"]
            for item in _chat_messages(request.messages, "compatible", refs)
            if item.get("role") == "tool"
        ]
    if name == "anthropic":
        _, messages = _message_input(request.messages, "anthropic", refs)
        return [
            json.loads(str(block["content"]))["result_ref"]
            for message in messages
            for block in message["content"]  # type: ignore[attr-defined]
            if block.get("type") == "tool_result"
        ]
    if name == "gemini":
        contents, _ = _gemini_contents(request.messages, "gemini", refs)
        return [
            part["function_response"]["response"]["result_ref"]
            for content in contents
            for part in content["parts"]  # type: ignore[attr-defined]
            if "function_response" in part
        ]
    assert name == "subscription_cli"
    return [
        block["result_ref"]
        for message in request.messages
        for block in _project_message(message, refs)["content"]  # type: ignore[attr-defined]
        if block.get("type") == "tool_result"
    ]


def test_every_provider_renders_a_result_ref_that_resolves_to_its_call():
    messages = _conversation()
    stored = canonical_json([repr(message) for message in messages])
    refs = tool_result_refs(messages)
    request = ModelRequest(messages=messages, tool_result_refs=refs.by_call_id)

    assert refs.citable == ("r1",)
    for name in (
        "openai",
        "openai_compatible",
        "anthropic",
        "gemini",
        "subscription_cli",
    ):
        rendered = _rendered_refs(name, request)
        assert rendered == ["r1", "r2"], name
        assert tuple(refs.call_ids[ref] for ref in rendered) == _CANONICAL
    assert canonical_json([repr(message) for message in messages]) == stored
    assert all(
        "result_ref" not in block.output
        for message in messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    )


def test_shared_rendering_omits_ref_only_when_none_was_assigned():
    block = _conversation()[2].content[0]
    assert isinstance(block, ToolResultBlock)
    assert model_tool_result(block, "r4") == {
        "result_ref": "r4",
        "is_error": False,
        "output": {"kind": "test.lookup", "data": {"value": "a"}},
    }
    assert "result_ref" not in model_tool_result(block, None)


async def test_agent_requests_carry_refs_while_the_transcript_stays_exact(
    tmp_path: Path,
) -> None:
    provider = ToolboxAwareMockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "document",
                        ARTIFACT_CREATE_TOOL_NAME,
                        {"from": "document", "format": "txt", "content": "alpha\n"},
                    ),
                ),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        ),
        provider_id="mock:result-refs",
    )
    agent = await Agent.create(
        "result-refs",
        root=tmp_path,
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=32_000,
            max_output_tokens=4_000,
            supports_tools=True,
        ),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Write one document.")
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    final = provider.requests[-1]
    expected = tool_result_refs(transcript.messages).by_call_id
    assert dict(final.tool_result_refs) == dict(expected)
    assert "document" in final.tool_result_refs
    rendered = _rendered_refs("anthropic", final)
    assert final.tool_result_refs["document"] in rendered
    assert all(
        "result_ref" not in block.output
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    )


class _RecordingExecutor:
    def __init__(self, name: str) -> None:
        self.executor_id = f"test.refs.{name}.executor"
        self.requests: list[ToolExecution] = []

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.requests.append(request)
        return ToolOutput(kind="test.refs.output", data={"value": "ok"})


def _ref_runtime() -> tuple[CapabilityRuntime, dict[str, _RecordingExecutor]]:
    executors = {
        name: _RecordingExecutor(name)
        for name in ("read_value", "cite_one", "cite_many", "cite_nested")
    }
    string = {"type": "string", "minLength": 1, "maxLength": 256}
    schemas: dict[str, tuple[dict[str, object], tuple[str, ...]]] = {
        "read_value": ({"value": string}, ()),
        "cite_one": ({"result_ref": string}, ("result_ref",)),
        "cite_many": (
            {"evidence_refs": {"type": "array", "items": string, "maxItems": 4}},
            ("evidence_refs[]",),
        ),
        "cite_nested": (
            {
                "inputs": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"result_ref": string},
                        "additionalProperties": False,
                    },
                }
            },
            ("inputs.*.result_ref",),
        ),
    }
    capabilities = tuple(
        Capability(
            id=f"test.refs.{name}",
            description=f"Execute {name}.",
            input_schema={
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            },
            output_kind="test.refs.output",
            output_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            executor_id=executors[name].executor_id,
            access_mode=AccessMode.READ,
            result_ref_arguments=paths,
        )
        for name, (properties, paths) in schemas.items()
    )
    views = tuple(
        ToolView(
            name=capability.id.rsplit(".", 1)[1],
            capability_id=capability.id,
            description=capability.description,
            presentation=_presentation(
                capability.id.rsplit(".", 1)[1],
                ToolboxId.SOURCES,
                ToolLoadMode.PINNED,
                text_trust=ToolTextTrust.CODE,
            ),
        )
        for capability in capabilities
    )
    domain = StaticTestDomain(capabilities, views, domain_owner_id="refs_test")
    registry = CapabilityRegistry(
        declarations=(domain.declarations,), executors=tuple(executors.values())
    )
    return CapabilityRuntime(registry, (domain,)), executors


async def _after_read(runtime: CapabilityRuntime, run):
    catalog = await runtime.prepare_run(run)
    read = ToolCall("read-1", "read_value", {"value": "x"})
    first = runtime.project(catalog, ())
    outcome = await _execute(runtime, run, first, read)
    messages = _append_results((), (read,), outcome.ordered_results)
    return catalog, messages


def _recovery(result: ToolResultBlock) -> Mapping[str, object]:
    error = result.output["error"]
    assert isinstance(error, Mapping)
    assert error["code"] == "result_ref_invalid"
    details = error["details"]
    assert isinstance(details, Mapping)
    assert details["execution_disposition"] == "not_dispatched"
    recovery = details["recovery"]
    assert isinstance(recovery, Mapping)
    assert recovery["kind"] == "correct_arguments"
    return recovery


async def test_declared_ref_fields_resolve_to_canonical_calls_before_validation():
    runtime, executors = _ref_runtime()
    run = _run("run-refs-resolve")
    catalog, messages = await _after_read(runtime, run)
    projection = runtime.project(catalog, messages)
    calls = (
        ToolCall("one", "cite_one", {"result_ref": "r1"}),
        ToolCall("many", "cite_many", {"evidence_refs": ["r1"]}),
        ToolCall(
            "nested",
            "cite_nested",
            {"inputs": {"orders": {"result_ref": "r1"}}},
        ),
    )

    outcome = await _execute(runtime, run, projection, *calls, messages=messages)

    assert not any(result.is_error for result in outcome.ordered_results)
    assert executors["cite_one"].requests[0].arguments["result_ref"] == "read-1"
    cited = executors["cite_many"].requests[0].arguments["evidence_refs"]
    assert isinstance(cited, tuple) and cited == ("read-1",)
    nested = executors["cite_nested"].requests[0].arguments["inputs"]
    assert isinstance(nested, Mapping)
    assert dict(nested["orders"]) == {"result_ref": "read-1"}
    # The model's call keeps exactly what it sent.
    assert calls[0].arguments["result_ref"] == "r1"


async def test_unknown_empty_raw_and_uncitable_refs_are_rejected_before_dispatch():
    runtime, executors = _ref_runtime()
    run = _run("run-refs-reject")
    catalog, messages = await _after_read(runtime, run)
    search = ToolCall("search", "toolbox_search", {"query": "read"})
    failed = ToolCall("failed", "read_value", {"value": ""})
    projection = runtime.project(catalog, messages)
    outcome = await _execute(
        runtime, run, projection, search, failed, messages=messages
    )
    assert outcome.ordered_results[1].is_error
    messages = _append_results(messages, (search, failed), outcome.ordered_results)
    refs = tool_result_refs(messages)
    assert refs.by_call_id == {"read-1": "r1", "search": "r2", "failed": "r3"}
    projection = runtime.project(catalog, messages)

    for value in ("r9", "", "read-1", "r2", "r3", "R1"):
        call = ToolCall(f"cite-{value or 'empty'}", "cite_one", {"result_ref": value})
        result = (
            await _execute(runtime, run, projection, call, messages=messages)
        ).ordered_results[0]
        recovery = _recovery(result)
        assert recovery["constraint"] == "Use one of these result_ref values: r1."
    nested = ToolCall(
        "nested-bad", "cite_nested", {"inputs": {"orders": {"result_ref": "r9"}}}
    )
    result = (
        await _execute(runtime, run, projection, nested, messages=messages)
    ).ordered_results[0]
    _recovery(result)
    assert all(
        not executor.requests
        for name, executor in executors.items()
        if name.startswith("cite")
    )


async def test_refs_from_another_run_are_rejected():
    runtime, executors = _ref_runtime()
    _, earlier = await _after_read(runtime, _run("run-refs-earlier"))
    assert tool_result_refs(earlier).citable == ("r1",)
    run = _run("run-refs-later")
    projection = runtime.project(await runtime.prepare_run(run), ())

    call = ToolCall("cite", "cite_one", {"result_ref": "r1"})
    result = (await _execute(runtime, run, projection, call)).ordered_results[0]

    recovery = _recovery(result)
    assert recovery["constraint"] == (
        "No earlier successful result exists in this run yet."
    )
    assert not executors["cite_one"].requests


_CALL_ID_FIELD = re.compile(r"(^|_)call_ids?$")
_CALL_ID_TEXT = re.compile(r"ToolCall\.id|\bcall[ _-]?ids?\b", re.IGNORECASE)


def _schema_violations(tool: str, schema: object, path: str) -> list[str]:
    found: list[str] = []
    if isinstance(schema, Mapping):
        description = schema.get("description")
        if isinstance(description, str) and _CALL_ID_TEXT.search(description):
            found.append(f"{tool}:{path} description")
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for key, value in properties.items():
                if _CALL_ID_FIELD.search(str(key)):
                    found.append(f"{tool}:{path}.{key}")
                found.extend(_schema_violations(tool, value, f"{path}.{key}"))
        for key in ("items", "additionalProperties", "not"):
            found.extend(_schema_violations(tool, schema.get(key), f"{path}[{key}]"))
        for key in ("anyOf", "oneOf", "allOf"):
            for index, item in enumerate(schema.get(key) or ()):
                found.extend(_schema_violations(tool, item, f"{path}[{key}{index}]"))
    return found


async def test_no_model_facing_schema_asks_for_a_tool_call_id(tmp_path: Path) -> None:
    agent = await Agent.create(
        "result-ref-inventory", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        registry = agent._embedded._capabilities
        violations: list[str] = []
        for name in sorted(registry.tool_names):
            definition = registry.tool_definition(name, lane=CallTargetLane.FOREGROUND)
            if _CALL_ID_TEXT.search(definition.description):
                violations.append(f"{name}: description")
            violations.extend(
                _schema_violations(name, definition.input_schema, "input")
            )
        assert {"task_complete", "python_execute", "data_upsert_rows"} <= set(
            registry.tool_names
        )
    finally:
        await agent.close()

    assert violations == []
