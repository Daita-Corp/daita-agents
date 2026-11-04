from __future__ import annotations

import os
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

import daita.hosting.configuration as configuration
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
from tests.support.workspace import workspace_for

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
    """Fake model boundary that grounds its answer in the real tool results."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.grounded_region: str | None = None
        self.grounded_margin: Decimal | None = None
        self.search_data: Mapping[str, object] | None = None
        self.described: Mapping[str, object] | None = None

    @property
    def provider_id(self) -> str:
        return "openai:fixture-model"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def close(self, *, deadline: float | None = None) -> None:
        return None

    def _call(self, call_id: str, tool: str, /, **arguments: object) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(ToolCall(id=call_id, name=tool, arguments=arguments),),
            provider_id=self.provider_id,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        tool_results = tuple(
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        if len(self.requests) == 1:
            return self._call(
                "find-tables",
                "catalog_search",
                query="customers regions orders order_items products",
                limit=8,
            )
        if len(self.requests) == 2:
            self.search_data = self._data(tool_results, "find-tables")
            return self._call(
                "describe-items",
                "catalog_describe",
                source="Fixture PostgreSQL",
                name="analytics.order_items",
            )
        if len(self.requests) == 3:
            self.described = self._data(tool_results, "describe-items")
            return self._call(
                "paid-revenue",
                "data_query",
                source="Fixture PostgreSQL",
                sql=(
                    "SELECT c.region_code, "
                    "SUM(o.total_amount) AS paid_revenue, "
                    "SUM(oi.line_total - "
                    "(oi.quantity * p.unit_cost)) AS gross_margin "
                    "FROM analytics.customers AS c "
                    "JOIN analytics.orders AS o "
                    "ON o.customer_id = c.customer_id "
                    "JOIN analytics.order_items AS oi "
                    "ON oi.order_id = o.order_id "
                    "JOIN analytics.products AS p "
                    "ON p.product_id = oi.product_id "
                    "WHERE o.status = $1 "
                    "GROUP BY c.region_code "
                    "ORDER BY paid_revenue DESC "
                    "LIMIT 1"
                ),
                parameters=["paid"],
            )

        data = self._data(tool_results, "paid-revenue")
        rows = data["rows"]
        if not isinstance(rows, tuple) or len(rows) != 1:
            raise AssertionError("query must return one leading region")
        row = rows[0]
        if not isinstance(row, Mapping):
            raise AssertionError("query row must be a mapping")
        region = row["region_code"]
        paid_revenue = row["paid_revenue"]
        gross_margin = row["gross_margin"]
        if (
            not isinstance(region, str)
            or not isinstance(paid_revenue, Mapping)
            or not isinstance(gross_margin, Mapping)
        ):
            raise AssertionError("query result has unexpected value types")
        if paid_revenue["type"] != "decimal" or gross_margin["type"] != "decimal":
            raise AssertionError("financial results must retain decimal typing")
        if Decimal(str(paid_revenue["value"])) <= 0:
            raise AssertionError("paid revenue must be positive")
        if Decimal(str(gross_margin["value"])) <= 0:
            raise AssertionError("gross margin must be positive")
        self.grounded_region = region
        self.grounded_margin = Decimal(str(gross_margin["value"]))
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text=f"{region} has the most paid revenue with positive gross margin.",
            provider_id=self.provider_id,
        )

    @staticmethod
    def _data(
        results: tuple[ToolResultBlock, ...], call_id: str
    ) -> Mapping[str, object]:
        matching = tuple(block for block in results if block.call_id == call_id)
        if len(matching) != 1 or matching[0].is_error:
            raise AssertionError(f"expected one successful {call_id} result")
        data = matching[0].output["data"]
        if not isinstance(data, Mapping):
            raise AssertionError("tool data must be a mapping")
        return data


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
        "Which region has the most paid revenue and gross margin?",
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
        configuration,
        "create_model_route_provider",
        lambda route, *, secret_provider=None: provider,
    )
    agent = await Agent.create(
        "postgresql-fixture",
        root=tmp_path,
        keychain=keychain,
        model_validator=validator,
        workspace=workspace_for(tmp_path),
    )
    try:
        await agent.configure_model(
            provider="openai",
            model="fixture-model",
            api_key="fake-provider-key",
            context_window_tokens=128000,
            max_output_tokens=4096,
        )
        credential = await agent.store_postgresql_password(password)
        await agent.attach_postgresql(
            host="127.0.0.1",
            port=port,
            database="daita_fixture",
            username="daita_reader",
            credential=credential,
            schemas=("analytics",),
            ssl_mode="disable",
            name="Fixture PostgreSQL",
        )
        await agent.close()
        agent = await Agent.open(
            "postgresql-fixture",
            root=tmp_path,
            keychain=keychain,
            workspace=workspace_for(tmp_path),
        )
        result = await agent.run(
            "Which region has the most paid revenue and gross margin?"
        )
    finally:
        await agent.close()

    assert result.final_text is not None
    assert provider.grounded_region in {"AMER", "EMEA", "APAC"}
    assert provider.grounded_margin is not None and provider.grounded_margin > 0
    assert provider.grounded_region in result.final_text
    assert "fake-provider-key" not in result.final_text
    assert password not in result.final_text
    assert len(provider.requests) == 4
    assert provider.search_data is not None
    hits = provider.search_data["hits"]
    assert isinstance(hits, tuple)
    assert {
        "analytics.customers",
        "analytics.orders",
        "analytics.order_items",
        "analytics.products",
        "analytics.regions",
    } <= {item["name"] for item in hits if isinstance(item, Mapping)}
    assert provider.described is not None
    assert provider.described["answered_from"] == "live"
    assert provider.described["primary_key"] == ("order_item_id",)
    assert tuple(
        (column["name"], column["type"], column["nullable"])
        for column in provider.described["columns"]  # type: ignore[attr-defined]
    ) == (
        ("order_item_id", "bigint", False),
        ("order_id", "bigint", False),
        ("product_id", "bigint", False),
        ("quantity", "integer", False),
        ("unit_price", "numeric(12,2)", False),
        ("discount_percent", "numeric(5,2)", False),
        ("line_total", "numeric(14,2)", True),
    )
    assert {
        (item["from"], item["to"])
        for item in provider.described["relationships"]  # type: ignore[attr-defined]
    } == {
        ("analytics.order_items", "analytics.orders"),
        ("analytics.order_items", "analytics.products"),
    }
    validator.assert_consumed()

    reopened = await Agent.open(
        "postgresql-fixture",
        root=tmp_path,
        keychain=keychain,
        workspace=workspace_for(tmp_path),
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
