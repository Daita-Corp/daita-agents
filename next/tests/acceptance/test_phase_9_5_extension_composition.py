from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from daita import Agent, AgentHost, SQLiteSource
from daita.capabilities import AccessMode, RiskLevel
from daita.extensions import (
    ConfiguredExtension,
    ExtensionKind,
    ExtensionLoadError,
    ExtensionManifest,
    ExtensionRegistration,
    tool,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
)

NOW = datetime(2026, 7, 20, 21, 0, tzinfo=timezone.utc)
PROFILE = ModelProfile(
    id="mock:phase-9-5-extensions",
    context_window_tokens=32_768,
    max_output_tokens=4_096,
    supports_tools=True,
)


class ScriptedProvider:
    provider_id = PROFILE.id

    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected extension model call")
        return self.responses.pop(0)


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (id INTEGER PRIMARY KEY, status TEXT NOT NULL);
            INSERT INTO customers (status) VALUES ('active'), ('inactive');
            """)


def _configured_extension(
    calls: list[Mapping[str, object]],
    *,
    version: str = "1.0.0",
    description: str = "Look up one configured extension value.",
    capability_id: str = "example.lookup",
) -> ConfiguredExtension:
    @tool(
        id=capability_id,
        owner="example",
        name="example_lookup",
        description=description,
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
    )
    async def lookup(arguments: Mapping[str, object]) -> Mapping[str, object]:
        calls.append(arguments)
        return {"value": f"configured:{arguments['key']}"}

    manifest = ExtensionManifest(
        id="example",
        version=version,
        kind=ExtensionKind.CAPABILITY_PROVIDER,
        declarations=lookup.declarations(),
    )
    return ConfiguredExtension(
        id="example",
        factory=lambda: ExtensionRegistration(
            manifest=manifest,
            executors=(lookup.executor,),
        ),
    )


def _tool_response(
    call_id: str,
    name: str,
    arguments: Mapping[str, object],
) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _script(
    source_id: str, *, suffix: str, evidence_id: str
) -> tuple[ModelResponse, ...]:
    return (
        _tool_response(
            f"query-{suffix}",
            "data_query_sqlite",
            {
                "source_id": source_id,
                "sql": "SELECT COUNT(*) AS total FROM customers",
            },
        ),
        _tool_response(
            f"extension-{suffix}",
            "example_lookup",
            {"key": suffix},
        ),
        ModelResponse(
            text=f"The built-in count and extension lookup completed. [evidence:{evidence_id}]",
            finish_reason=FinishReason.STOP,
        ),
    )


async def test_configured_extension_is_additive_bound_and_runtime_executed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    root = tmp_path / "state"
    ids = _ids()
    calls: list[Mapping[str, object]] = []
    configured = _configured_extension(calls)
    provider = ScriptedProvider()
    agent = await Agent.create(
        "extended",
        root=root,
        model=provider,
        model_profile=PROFILE,
        extensions=(configured,),
        clock=lambda: NOW,
        id_factory=ids,
    )
    registration = await agent.attach(SQLiteSource(database))
    provider.responses.extend(
        _script(registration.id, suffix="first", evidence_id="evidence-1")
    )
    first = await agent.run("Use the customer count and configured lookup.")
    first_snapshot = await agent.inspect(first.operation_id)

    assert [task.capability_id for task in first_snapshot.tasks] == [
        "data.sqlite.query",
        "example.lookup",
    ]
    assert [evidence.task_id for evidence in first_snapshot.evidence] == [
        task.id for task in first_snapshot.tasks
    ]
    assert all(evidence.accepted for evidence in first_snapshot.evidence)
    assert len(calls) == 1
    assert calls[0]["key"] == "first"
    projected_tools = {tool.name for tool in provider.requests[0].tools}
    assert "data_query_sqlite" in projected_tools
    assert "example_lookup" in projected_tools
    assert agent.extension_bindings[0].id == "example"
    assert agent.extension_bindings[0].version == "1.0.0"
    await agent.close()

    missing_provider = ScriptedProvider()
    with pytest.raises(ExtensionLoadError) as missing:
        await Agent.open(
            "extended",
            root=root,
            model=missing_provider,
            clock=lambda: NOW,
            id_factory=ids,
        )
    assert missing.value.diagnostic.code == "extension.configuration_missing"
    assert missing_provider.requests == []

    drift_provider = ScriptedProvider()
    with pytest.raises(ExtensionLoadError) as drift:
        await Agent.open(
            "extended",
            root=root,
            model=drift_provider,
            extensions=(_configured_extension([], version="2.0.0"),),
            clock=lambda: NOW,
            id_factory=ids,
        )
    assert drift.value.diagnostic.code == "extension.configuration_drift"
    assert drift_provider.requests == []

    changed_provider = ScriptedProvider()
    with pytest.raises(ExtensionLoadError) as changed:
        await Agent.open(
            "extended",
            root=root,
            model=changed_provider,
            extensions=(
                _configured_extension([], description="A changed tool projection."),
            ),
            clock=lambda: NOW,
            id_factory=ids,
        )
    assert changed.value.diagnostic.code == "extension.configuration_drift"
    assert changed_provider.requests == []

    reopened_calls: list[Mapping[str, object]] = []
    reopened_provider = ScriptedProvider()
    reopened = await Agent.open(
        "extended",
        root=root,
        model=reopened_provider,
        extensions=(_configured_extension(reopened_calls),),
        clock=lambda: NOW,
        id_factory=ids,
    )
    reopened_provider.responses.extend(
        _script(registration.id, suffix="second", evidence_id="evidence-3")
    )
    second = await reopened.run("Use both configured capabilities again.")
    second_snapshot = await reopened.inspect(second.operation_id)
    assert [task.capability_id for task in second_snapshot.tasks] == [
        "data.sqlite.query",
        "example.lookup",
    ]
    assert len(reopened_calls) == 1
    await reopened.close()

    host = await AgentHost.open(
        "extended",
        root=root,
        model=ScriptedProvider(),
        extensions=(_configured_extension([]),),
        clock=lambda: NOW,
        id_factory=ids,
        cadence_seconds=3_600,
    )
    try:
        assert host.extension_bindings == agent.extension_bindings
        await host.start()
    finally:
        await host.stop(drain=False)


async def test_builtin_collision_fails_before_agent_home_publication_or_io(
    tmp_path: Path,
) -> None:
    calls: list[Mapping[str, object]] = []
    colliding = _configured_extension(
        calls,
        capability_id="data.sqlite.query",
    )

    with pytest.raises(ExtensionLoadError) as failure:
        await Agent.create(
            "colliding",
            root=tmp_path,
            extensions=(colliding,),
            clock=lambda: NOW,
            id_factory=_ids(),
        )

    assert failure.value.diagnostic.code == "extension.declaration_collision"
    assert failure.value.diagnostic.declaration_kind == "capability"
    assert calls == []
    assert not (tmp_path / "colliding").exists()
