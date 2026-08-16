from __future__ import annotations

import io
import os
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

import daita.hosting.embedded as embedded
from daita import Agent
from daita._json import canonical_json
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference

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
        self.grounded_margin: Decimal | None = None
        self.catalog_tool_call_count = 0
        self.schema_result_bytes = 0
        self.projected_resource_count = 0
        self.duplicated_relationship_id_count = 0
        self.truncation: Mapping[str, object] | None = None
        self.schema_data: Mapping[str, object] | None = None

    @property
    def provider_id(self) -> str:
        return "openai:fixture-model"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            self.catalog_tool_call_count += 1
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="regional-schema",
                        name="catalog_schema",
                        arguments={
                            "query": (
                                "customers regions orders order_items products "
                                "region_code total_amount unit_cost unit_price"
                            ),
                            "limit": 8,
                            "include_relationships": True,
                        },
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
                raise AssertionError("expected one successful catalog schema result")
            catalog_data = tool_results[0].output["data"]
            if not isinstance(catalog_data, Mapping):
                raise AssertionError("catalog schema data must be a mapping")
            resources = catalog_data["resources"]
            relationships = catalog_data["relationships"]
            sources = catalog_data["sources"]
            truncation = catalog_data["truncation"]
            if (
                not isinstance(resources, tuple)
                or not isinstance(relationships, tuple)
                or not isinstance(sources, tuple)
                or len(sources) != 1
                or not isinstance(sources[0], Mapping)
                or not isinstance(truncation, Mapping)
            ):
                raise AssertionError("catalog schema result has invalid structure")
            names = {item["name"] for item in resources if isinstance(item, Mapping)}
            if (
                not {
                    "analytics.customers",
                    "analytics.regions",
                    "analytics.orders",
                    "analytics.order_items",
                    "analytics.products",
                }
                <= names
            ):
                raise AssertionError("schema slice omitted a required join resource")
            relationship_ids = tuple(
                item["relationship_id"]
                for item in relationships
                if isinstance(item, Mapping)
            )
            self.schema_result_bytes = len(canonical_json(catalog_data).encode("utf-8"))
            self.schema_data = catalog_data
            self.projected_resource_count = len(resources)
            self.duplicated_relationship_id_count = len(relationship_ids) - len(
                set(relationship_ids)
            )
            self.truncation = truncation
            source_id = sources[0]["source_id"]
            if not isinstance(source_id, str):
                raise AssertionError("schema slice did not expose the source ID")
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
        embedded,
        "create_model_route_provider",
        lambda route, *, secret_provider=None: provider,
    )
    agent = await Agent.create(
        "postgresql-fixture",
        root=tmp_path,
        keychain=keychain,
        model_validator=validator,
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
    assert len(provider.requests) == 3
    assert provider.catalog_tool_call_count == 1
    assert provider.projected_resource_count == 8
    assert provider.duplicated_relationship_id_count == 0
    assert provider.schema_result_bytes > 0
    assert provider.truncation is not None
    assert provider.truncation["resources"] is False
    assert provider.truncation["relationships"] is False
    assert provider.schema_data is not None
    projected_resources = provider.schema_data["resources"]
    assert isinstance(projected_resources, tuple)
    by_name = {
        item["name"]: item for item in projected_resources if isinstance(item, Mapping)
    }
    expected_columns = {
        "analytics.regions": (
            ("region_code", "pg_catalog.text|text", False),
            ("region_name", "pg_catalog.text|text", False),
            ("currency_code", "pg_catalog.text|text", False),
        ),
        "analytics.customers": (
            ("customer_id", "pg_catalog.int8|bigint", False),
            ("customer_name", "pg_catalog.text|text", False),
            ("email", "pg_catalog.text|text", False),
            ("region_code", "pg_catalog.text|text", False),
            ("segment", "pg_catalog.text|text", False),
            (
                "signed_up_at",
                "pg_catalog.timestamptz|timestamp with time zone",
                False,
            ),
            ("is_active", "pg_catalog.bool|boolean", False),
        ),
        "analytics.products": (
            ("product_id", "pg_catalog.int8|bigint", False),
            ("sku", "pg_catalog.text|text", False),
            ("product_name", "pg_catalog.text|text", False),
            ("category", "pg_catalog.text|text", False),
            ("unit_price", "pg_catalog.numeric|numeric(12,2)", False),
            ("unit_cost", "pg_catalog.numeric|numeric(12,2)", False),
            ("is_active", "pg_catalog.bool|boolean", False),
            (
                "created_at",
                "pg_catalog.timestamptz|timestamp with time zone",
                False,
            ),
        ),
        "analytics.orders": (
            ("order_id", "pg_catalog.int8|bigint", False),
            ("customer_id", "pg_catalog.int8|bigint", False),
            (
                "ordered_at",
                "pg_catalog.timestamptz|timestamp with time zone",
                False,
            ),
            ("status", "pg_catalog.text|text", False),
            ("sales_channel", "pg_catalog.text|text", False),
            ("subtotal", "pg_catalog.numeric|numeric(14,2)", False),
            ("tax_amount", "pg_catalog.numeric|numeric(14,2)", False),
            ("total_amount", "pg_catalog.numeric|numeric(14,2)", False),
        ),
        "analytics.order_items": (
            ("order_item_id", "pg_catalog.int8|bigint", False),
            ("order_id", "pg_catalog.int8|bigint", False),
            ("product_id", "pg_catalog.int8|bigint", False),
            ("quantity", "pg_catalog.int4|integer", False),
            ("unit_price", "pg_catalog.numeric|numeric(12,2)", False),
            (
                "discount_percent",
                "pg_catalog.numeric|numeric(5,2)",
                False,
            ),
            ("line_total", "pg_catalog.numeric|numeric(14,2)", True),
        ),
        "analytics.payments": (
            ("payment_id", "pg_catalog.int8|bigint", False),
            ("order_id", "pg_catalog.int8|bigint", False),
            (
                "processed_at",
                "pg_catalog.timestamptz|timestamp with time zone",
                False,
            ),
            ("payment_method", "pg_catalog.text|text", False),
            ("payment_status", "pg_catalog.text|text", False),
            ("amount", "pg_catalog.numeric|numeric(14,2)", False),
        ),
        "analytics.shipments": (
            ("shipment_id", "pg_catalog.int8|bigint", False),
            ("order_id", "pg_catalog.int8|bigint", False),
            ("warehouse_code", "pg_catalog.text|text", False),
            ("shipment_status", "pg_catalog.text|text", False),
            (
                "shipped_at",
                "pg_catalog.timestamptz|timestamp with time zone",
                True,
            ),
            (
                "delivered_at",
                "pg_catalog.timestamptz|timestamp with time zone",
                True,
            ),
        ),
        "analytics.support_tickets": (
            ("ticket_id", "pg_catalog.int8|bigint", False),
            ("customer_id", "pg_catalog.int8|bigint", False),
            ("order_id", "pg_catalog.int8|bigint", True),
            (
                "opened_at",
                "pg_catalog.timestamptz|timestamp with time zone",
                False,
            ),
            ("category", "pg_catalog.text|text", False),
            ("priority", "pg_catalog.text|text", False),
            ("ticket_status", "pg_catalog.text|text", False),
            ("satisfaction_score", "pg_catalog.int4|integer", True),
        ),
    }
    assert set(by_name) == set(expected_columns)
    for name, expected in expected_columns.items():
        columns = by_name[name]["columns"]
        assert isinstance(columns, tuple)
        expected_display = tuple(
            (column_name, native_type.rpartition("|")[2], nullable)
            for column_name, native_type, nullable in expected
        )
        assert (
            tuple(
                (column["name"], column["type"], column["nullable"])
                for column in columns
                if isinstance(column, Mapping)
            )
            == expected_display
        )
    expected_primary_keys = {
        "analytics.regions": ("region_code",),
        "analytics.customers": ("customer_id",),
        "analytics.products": ("product_id",),
        "analytics.orders": ("order_id",),
        "analytics.order_items": ("order_item_id",),
        "analytics.payments": ("payment_id",),
        "analytics.shipments": ("shipment_id",),
        "analytics.support_tickets": ("ticket_id",),
    }
    expected_unique_keys = {
        "analytics.regions": (("region_name",),),
        "analytics.customers": (("email",),),
        "analytics.products": (("sku",),),
        "analytics.orders": (),
        "analytics.order_items": (),
        "analytics.payments": (("order_id",),),
        "analytics.shipments": (("order_id",),),
        "analytics.support_tickets": (),
    }
    assert {
        name: item["primary_key_fields"] for name, item in by_name.items()
    } == expected_primary_keys
    assert {
        name: item["unique_key_fields"] for name, item in by_name.items()
    } == expected_unique_keys
    relationships = provider.schema_data["relationships"]
    assert isinstance(relationships, tuple)
    names_by_id = {item["resource_id"]: item["name"] for item in by_name.values()}
    relationship_pairs = {
        (
            names_by_id[item["from_resource_id"]],
            names_by_id[item["to_resource_id"]],
            tuple(
                (pair["source_field"], pair["target_field"])
                for pair in item["field_pairs"]
                if isinstance(pair, Mapping)
            ),
        )
        for item in relationships
        if isinstance(item, Mapping)
    }
    assert relationship_pairs == {
        ("analytics.customers", "analytics.regions", (("region_code", "region_code"),)),
        ("analytics.orders", "analytics.customers", (("customer_id", "customer_id"),)),
        ("analytics.order_items", "analytics.orders", (("order_id", "order_id"),)),
        (
            "analytics.order_items",
            "analytics.products",
            (("product_id", "product_id"),),
        ),
        ("analytics.payments", "analytics.orders", (("order_id", "order_id"),)),
        ("analytics.shipments", "analytics.orders", (("order_id", "order_id"),)),
        (
            "analytics.support_tickets",
            "analytics.customers",
            (("customer_id", "customer_id"),),
        ),
        (
            "analytics.support_tickets",
            "analytics.orders",
            (("order_id", "order_id"),),
        ),
    }
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
        resources = await reopened.list_catalog_resources(source_id=sources[0].id)
        inspections = tuple(
            [
                await reopened._embedded._catalog_service.inspect_resource(
                    reopened.id,
                    resource.id,
                )
                for resource in resources
            ]
        )
        inspection_bytes = sum(
            len(canonical_json(inspection).encode("utf-8"))
            for inspection in inspections
        )
        assert provider.schema_result_bytes < inspection_bytes * 0.7
    finally:
        await reopened.close()
