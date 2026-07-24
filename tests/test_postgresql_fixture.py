from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import io
import os
from pathlib import Path
import re

import pytest

from daita import Agent
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference
from daita.terminal import run_terminal_application
import daita.hosting.embedded as embedded

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_POSTGRES_FIXTURE") != "1",
        reason=(
            "set DAITA_RUN_POSTGRES_FIXTURE=1 after starting "
            "tests/fixtures/postgresql/compose.yaml"
        ),
    ),
]


class _FakeKeychain:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.events: list[tuple[str, str]] = []

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]

    async def set(self, reference: SecretReference, value: str) -> None:
        self.events.append(("set", reference.name))
        self.values[reference.name] = value

    async def delete(self, reference: SecretReference) -> None:
        self.events.append(("delete", reference.name))
        self.values.pop(reference.name, None)


class _GroundedFixtureProvider:
    """Fake model boundary that grounds its answer in the real tool result."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.grounded_region: str | None = None

    @property
    def provider_id(self) -> str:
        return "openai:fixture-model"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="find-regions",
                        name="catalog_search",
                        arguments={"query": "regions", "limit": 1},
                    ),
                ),
                provider_id=self.provider_id,
            )

        tool_results = tuple(
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        if len(self.requests) == 2:
            if len(tool_results) != 1 or tool_results[0].is_error:
                raise AssertionError("expected one successful catalog search result")
            catalog_data = tool_results[0].output["data"]
            if not isinstance(catalog_data, Mapping):
                raise AssertionError("catalog search data must be a mapping")
            hits = catalog_data["hits"]
            if not isinstance(hits, tuple) or len(hits) != 1:
                raise AssertionError("catalog search must return the regions table")
            hit = hits[0]
            if not isinstance(hit, Mapping):
                raise AssertionError("catalog hit must be a mapping")
            source_id = hit["source_id"]
            if (
                not isinstance(source_id, str)
                or re.fullmatch(
                    r"source:sha256:[0-9a-f]{64}",
                    source_id,
                )
                is None
            ):
                raise AssertionError("catalog hit did not expose the source ID")
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="paid-revenue",
                        name="data_query_postgresql",
                        arguments={
                            "source_id": source_id,
                            "sql": (
                                "SELECT c.region_code, "
                                "SUM(o.total_amount) AS paid_revenue "
                                "FROM analytics.customers AS c "
                                "JOIN analytics.orders AS o "
                                "ON o.customer_id = c.customer_id "
                                "WHERE o.status = $1 "
                                "GROUP BY c.region_code "
                                "ORDER BY paid_revenue DESC "
                                "LIMIT 1"
                            ),
                            "parameters": ["paid"],
                        },
                    ),
                ),
                provider_id=self.provider_id,
            )

        query_results = tuple(
            block for block in tool_results if block.call_id == "paid-revenue"
        )
        if len(query_results) != 1 or query_results[0].is_error:
            raise AssertionError("expected one successful PostgreSQL tool result")
        data = query_results[0].output["data"]
        if not isinstance(data, Mapping):
            raise AssertionError("tool data must be a mapping")
        rows = data["rows"]
        if not isinstance(rows, tuple) or len(rows) != 1:
            raise AssertionError("query must return one leading region")
        row = rows[0]
        if not isinstance(row, Mapping):
            raise AssertionError("query row must be a mapping")
        region = row["region_code"]
        paid_revenue = row["paid_revenue"]
        if not isinstance(region, str) or not isinstance(paid_revenue, Mapping):
            raise AssertionError("query result has unexpected value types")
        if paid_revenue["type"] != "decimal":
            raise AssertionError("paid revenue must retain decimal typing")
        if Decimal(str(paid_revenue["value"])) <= 0:
            raise AssertionError("paid revenue must be positive")
        self.grounded_region = region
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text=f"{region} has the most paid revenue.",
            provider_id=self.provider_id,
        )


def _terminal_onboarding_input(port: int) -> str:
    choices = (
        "postgresql-fixture",
        "1",  # OpenAI
        "4",  # Enter a model ID manually
        "fixture-model",
        "128000",  # Explicit context window for an unreviewed model
        "4096",  # Explicit maximum output
        "3",  # PostgreSQL
        "Fixture PostgreSQL",
        "127.0.0.1",
        str(port),
        "daita_fixture",
        "daita_reader",
        "disable",
        "1",  # analytics
        "Which region has the most paid revenue?",
        "/exit",
    )
    return "\n".join(choices) + "\n"


async def test_zero_argument_onboarding_through_grounded_postgresql_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    password = os.environ.get(
        "DAITA_FIXTURE_POSTGRES_PASSWORD",
        "daita_fixture_password",
    )
    port = int(os.environ.get("DAITA_FIXTURE_POSTGRES_PORT", "55432"))
    keychain = _FakeKeychain()
    validator = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="validation-call",
                        name="daita_validate_tool_support",
                        arguments={},
                    ),
                ),
                provider_id="openai:fixture-model",
            ),
        ),
        provider_id="openai:fixture-model",
    )
    provider = _GroundedFixtureProvider()
    monkeypatch.setattr(
        embedded,
        "create_model_route_provider",
        lambda route, *, secret_provider=None: provider,
    )
    hidden_values = iter(("fake-provider-key", password))
    hidden_prompts: list[str] = []

    def hidden_input(prompt: str) -> str:
        hidden_prompts.append(prompt)
        return next(hidden_values)

    output = io.StringIO()
    code = await run_terminal_application(
        root=tmp_path,
        input_stream=io.StringIO(_terminal_onboarding_input(port)),
        output_stream=output,
        hidden_input=hidden_input,
        keychain=keychain,
        model_validator=validator,
    )

    assert code == 0
    assert hidden_prompts == ["API key: ", "Password: "]
    assert len([event for event in keychain.events if event[0] == "set"]) == 2
    assert sorted(keychain.values.values()) == sorted(["fake-provider-key", password])
    text = output.getvalue()
    assert "✓ Connection validated" in text
    assert "analytics · base tables" in text
    assert "✓ Schemas selected: analytics" in text
    assert "✓ Catalog ready: 8 tables" in text
    assert provider.grounded_region in {"AMER", "EMEA", "APAC"}
    assert f"{provider.grounded_region} has the most paid revenue." in text
    assert "fake-provider-key" not in text
    assert password not in text
    assert len(provider.requests) == 3
    validator.assert_consumed()

    reopened = await Agent.open(
        "postgresql-fixture",
        root=tmp_path,
        keychain=keychain,
    )
    try:
        sources = await reopened.list_sources()
        summary = await reopened.catalog_summary()
        assert len(sources) == 1
        assert sources[0].display_name == "Fixture PostgreSQL"
        assert sources[0].configuration["schemas"] == ("analytics",)
        assert summary.active_source_count == 1
        assert summary.resource_count == 8
        assert summary.relationship_count >= 7
        assert summary.is_empty is False
    finally:
        await reopened.close()
