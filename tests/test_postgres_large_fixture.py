from __future__ import annotations

import os
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

from daita import Agent
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import SecretReference

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "postgres-large"
ATTACHED_SCHEMAS = (
    "analytics",
    "archive",
    "billing",
    "catalog",
    "core",
    "sales",
    "support",
)


class _Secrets:
    def __init__(self, password: str) -> None:
        self.password = password
        self.references: list[SecretReference] = []

    async def resolve(self, reference: SecretReference) -> str:
        self.references.append(reference)
        return self.password


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=4_096,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _tool_results(provider: MockModelProvider) -> tuple[ToolResultBlock, ...]:
    return tuple(
        block
        for request in provider.requests
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    )


def test_postgres_large_fixture_declares_its_bounded_contract():
    compose = (FIXTURE / "compose.yaml").read_text(encoding="utf-8")
    init = (FIXTURE / "init.sql").read_text(encoding="utf-8")
    readme = (FIXTURE / "README.md").read_text(encoding="utf-8")

    assert "name: daita-postgres-large-fixture" in compose
    assert "${DAITA_LARGE_POSTGRES_PORT:-55433}" in compose
    assert "private.fixture_status" in compose
    for schema in (*ATTACHED_SCHEMAS, "private", "staging"):
        assert f"CREATE SCHEMA {schema};" in init
    assert "CREATE TABLE sales.orders" in init
    assert "CREATE TABLE archive.orders" in init
    assert "REFERENCES sales.orders(order_id)" in init
    assert "REFERENCES core.customers(customer_id)" in init
    assert "CREATE TYPE catalog.lifecycle_state AS ENUM" in init
    assert "CREATE VIEW analytics.monthly_revenue" in init
    assert "daita_large_reader" in init
    assert "34 supported base tables and 47" in readme
    assert "DAITA_RUN_POSTGRES_LARGE_FIXTURE=1" in readme


@pytest.mark.acceptance
@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.skipif(
    os.environ.get("DAITA_RUN_POSTGRES_LARGE_FIXTURE") != "1",
    reason=(
        "set DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 after starting "
        "tests/fixtures/postgres-large/compose.yaml"
    ),
)
async def test_daita_catalogs_and_queries_large_multi_schema_postgresql(tmp_path: Path):
    password = os.environ.get(
        "DAITA_LARGE_POSTGRES_PASSWORD",
        "daita_large_fixture_password",
    )
    port = int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433"))
    credential = SecretReference.keychain("fixture:postgres-large:credential")
    secrets = _Secrets(password)
    provider = MockModelProvider(())
    agent = await Agent.create(
        "postgres-large-fixture",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        secret_provider=secrets,
    )
    try:
        probe = await agent.probe_postgresql(
            host="127.0.0.1",
            port=port,
            database="daita_large_fixture",
            username="daita_large_reader",
            credential=credential,
            ssl_mode="disable",
        )
        probe_by_name = {
            schema.name: schema.has_base_tables for schema in probe.schemas
        }
        assert all(probe_by_name[schema] for schema in ATTACHED_SCHEMAS)
        assert probe_by_name["private"] is False
        assert probe_by_name["staging"] is False

        source = await agent.attach_postgresql(
            host="127.0.0.1",
            port=port,
            database="daita_large_fixture",
            username="daita_large_reader",
            credential=credential,
            schemas=ATTACHED_SCHEMAS,
            ssl_mode="disable",
            name="Large multi-schema PostgreSQL",
        )
        summary = await agent.catalog_summary()
        resources = await agent.list_catalog_resources(source_id=source.id)
        by_native_identity = {
            resource.native_identity: resource for resource in resources
        }

        assert summary.resource_count == 34
        assert summary.relationship_count == 47
        assert len(resources) == 34
        assert "sales.orders" in by_native_identity
        assert "archive.orders" in by_native_identity
        assert "catalog.unsupported_type_probe" not in by_native_identity
        assert "analytics.monthly_revenue" not in by_native_identity
        assert all(not name.startswith("private.") for name in by_native_identity)

        schema_resources = (
            "core.regions",
            "core.organizations",
            "core.customers",
            "sales.orders",
            "billing.invoices",
            "archive.orders",
        )
        provider._script = (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="multi-schema-catalog",
                        name="catalog_schema",
                        arguments={
                            "resource_ids": [
                                by_native_identity[name].id for name in schema_resources
                            ],
                            "include_relationships": True,
                        },
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="regional-invoiced-revenue",
                        name="data_query_postgresql",
                        arguments={
                            "source_id": source.id,
                            "sql": (
                                "SELECT r.region_code, "
                                "COUNT(DISTINCT o.order_id) AS paid_order_count, "
                                "SUM(i.total_amount) AS invoiced_total "
                                "FROM core.customers AS c "
                                "JOIN core.organizations AS org "
                                "ON org.organization_id = c.organization_id "
                                "JOIN core.regions AS r "
                                "ON r.region_code = org.region_code "
                                "JOIN sales.orders AS o "
                                "ON o.customer_id = c.customer_id "
                                "JOIN billing.invoices AS i "
                                "ON i.order_id = o.order_id "
                                "WHERE i.status = $1 "
                                "GROUP BY r.region_code "
                                "ORDER BY invoiced_total DESC "
                                "LIMIT 1"
                            ),
                            "parameters": ["paid"],
                        },
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="The leading region was calculated from current invoice data.",
            ),
        )

        result = await agent.run("Which region has the most paid invoiced revenue?")

        assert result.final_text == (
            "The leading region was calculated from current invoice data."
        )
        tool_results = _tool_results(provider)
        schema_result = next(
            block for block in tool_results if block.call_id == "multi-schema-catalog"
        )
        assert schema_result.is_error is False
        schema_data = schema_result.output["data"]
        assert isinstance(schema_data, Mapping)
        projected = schema_data["resources"]
        assert isinstance(projected, tuple)
        assert {item["name"] for item in projected if isinstance(item, Mapping)} == set(
            schema_resources
        )
        relationships = schema_data["relationships"]
        assert isinstance(relationships, tuple)
        assert len(relationships) >= 5

        query_result = next(
            block
            for block in tool_results
            if block.call_id == "regional-invoiced-revenue"
        )
        assert query_result.is_error is False
        query_data = query_result.output["data"]
        assert isinstance(query_data, Mapping)
        rows = query_data["rows"]
        assert isinstance(rows, tuple)
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, Mapping)
        assert row["region_code"] in {"AMER", "EMEA", "APAC", "LATAM", "CAN"}
        assert isinstance(row["paid_order_count"], int)
        assert row["paid_order_count"] > 0
        invoiced_total = row["invoiced_total"]
        assert isinstance(invoiced_total, Mapping)
        assert invoiced_total["type"] == "decimal"
        assert Decimal(str(invoiced_total["value"])) > 0
        provider.assert_consumed()
    finally:
        await agent.close()
