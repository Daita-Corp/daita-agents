from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from daita import Agent, SQLiteSource
from daita.hosting.embedded import AgentNotConfiguredError
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ToolResultBlock,
    ToolCall,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter
from daita.loop.models import LoopExitKind

NOW = datetime(2026, 7, 19, 18, 0, tzinfo=timezone.utc)


def _ids():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}-{counters[prefix]}"

    return factory


def _profile(provider_id: str) -> ModelProfile:
    return ModelProfile(
        id=provider_id,
        context_window_tokens=32_768,
        max_output_tokens=4_096,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _registration(
    provider: MockModelProvider,
    sensitivities: frozenset[ModelSensitivity] = frozenset({ModelSensitivity.INTERNAL}),
) -> ModelProviderRegistration:
    return ModelProviderRegistration(
        provider=provider,
        profile=_profile(provider.provider_id),
        allowed_sensitivities=sensitivities,
    )


def _router(
    primary: MockModelProvider,
    fallback: MockModelProvider,
) -> ModelRouter:
    return ModelRouter(_registration(primary), (_registration(fallback),))


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (id INTEGER PRIMARY KEY, active INTEGER NOT NULL);
            INSERT INTO customers (active) VALUES (1), (1), (0);
            """)


async def test_provider_switch_uses_canonical_context_and_survives_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "customers.db"
    _database(database)
    primary = MockModelProvider((), provider_id="openai:primary")
    fallback = MockModelProvider((), provider_id="anthropic:fallback")
    route = _router(primary, fallback)
    agent = await Agent.create(
        "portable-route",
        root=tmp_path / "state",
        model=route,
        model_profile=route.profile,
        clock=lambda: NOW,
        id_factory=_ids(),
    )
    source = await agent.attach(SQLiteSource(database))
    primary._script = (
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="canonical-query-call",
                    provider_call_id="openai-native-call",
                    name="data_query_sqlite",
                    arguments={
                        "source_id": source.id,
                        "sql": "SELECT COUNT(*) AS active_count FROM customers WHERE active = ?",
                        "parameters": [1],
                    },
                ),
            ),
            provider_metadata={"openai_replay_items": [{"type": "reasoning"}]},
        ),
        ModelProviderError(ProviderErrorCode.RATE_LIMIT_ERROR),
    )
    fallback._script = (
        ModelResponse(
            finish_reason=FinishReason.STOP,
            text="There are 2 active customers. [evidence:evidence-1]",
        ),
    )

    result = await agent.run(
        "How many active customers are there?",
        session_id="session-portable-route",
    )
    snapshot = await agent.inspect(result.operation_id)
    await agent.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert [call.provider_id for call in snapshot.model_calls] == [
        route.provider_id,
        route.provider_id,
    ]
    assert [
        call.response.provider_id for call in snapshot.model_calls if call.response
    ] == [
        "openai:primary",
        "anthropic:fallback",
    ]
    second = snapshot.model_calls[1].response
    assert second is not None and second.routing is not None
    assert [item.provider_id for item in second.routing.attempts] == [
        "openai:primary",
        "anthropic:fallback",
    ]
    portable_request = fallback.requests[0]
    prior_assistant = next(
        message
        for message in portable_request.messages
        if message.role is MessageRole.ASSISTANT and message.tool_calls
    )
    assert prior_assistant.tool_calls[0].id == "canonical-query-call"
    assert prior_assistant.tool_calls[0].provider_call_id is None
    assert prior_assistant.provider_id is None
    assert not prior_assistant.provider_metadata
    assert any(
        message.role is MessageRole.TOOL
        and isinstance(message.content[0], ToolResultBlock)
        and message.content[0].call_id == "canonical-query-call"
        for message in portable_request.messages
    )
    routed_events = [
        event
        for event in snapshot.events
        if event.type == "model_response.recorded" and "routing" in event.payload
    ]
    assert len(routed_events) == 2
    assert routed_events[-1].payload["selected_provider_id"] == ("anthropic:fallback")

    reopened_primary = MockModelProvider((), provider_id="openai:primary")
    reopened_fallback = MockModelProvider((), provider_id="anthropic:fallback")
    reopened_route = _router(reopened_primary, reopened_fallback)
    assert reopened_route.provider_id == route.provider_id
    reopened = await Agent.open(
        "portable-route",
        root=tmp_path / "state",
        model=reopened_route,
        model_profile=reopened_route.profile,
        clock=lambda: NOW,
    )
    assert await reopened.inspect(result.operation_id) == snapshot
    transcript = await reopened.transcript("session-portable-route")
    await reopened.close()
    assert tuple(
        message.provider_id
        for message in transcript.messages
        if message.role is MessageRole.ASSISTANT
    ) == ("openai:primary", "anthropic:fallback")


async def test_reopen_rejects_changed_route_sensitivity_policy(tmp_path: Path) -> None:
    primary = MockModelProvider((), provider_id="openai:primary")
    fallback = MockModelProvider((), provider_id="anthropic:fallback")
    route = _router(primary, fallback)
    agent = await Agent.create(
        "route-drift",
        root=tmp_path,
        model=route,
        model_profile=route.profile,
        clock=lambda: NOW,
    )
    await agent.close()

    changed = ModelRouter(
        _registration(
            MockModelProvider((), provider_id="openai:primary"),
            frozenset({ModelSensitivity.PUBLIC}),
        ),
        (
            _registration(
                MockModelProvider((), provider_id="anthropic:fallback"),
                frozenset({ModelSensitivity.PUBLIC}),
            ),
        ),
    )
    with pytest.raises(AgentNotConfiguredError, match="provider differs"):
        await Agent.open(
            "route-drift",
            root=tmp_path,
            model=changed,
            model_profile=changed.profile,
            clock=lambda: NOW,
        )
