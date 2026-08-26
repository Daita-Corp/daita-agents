from __future__ import annotations

from _workspace_support import workspace_for

import os
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

from daita import Agent, ApprovalDecision, ApprovalRequest
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.protocols import ModelProvider
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


def _profile(provider: ModelProvider) -> ModelProfile:
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


class _BulkUpdateProvider:
    def __init__(self) -> None:
        self._source_id = ""
        self._resource_id = ""
        self._desired_priority = ""
        self._phase = 0
        self._matched_rows = 0
        self._requests: list[ModelRequest] = []

    @property
    def provider_id(self) -> str:
        return "mock:postgres-large-bulk-update"

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        return tuple(self._requests)

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return isinstance(request, ModelRequest)

    def configure(
        self,
        *,
        source_id: str,
        resource_id: str,
        desired_priority: str,
    ) -> None:
        self._source_id = source_id
        self._resource_id = resource_id
        self._desired_priority = desired_priority
        self._phase = 0
        self._matched_rows = 0

    def _plan(self) -> dict[str, object]:
        return {
            "source_id": self._source_id,
            "resource_id": self._resource_id,
            "where": [
                {
                    "column": "ticket_status",
                    "operator": "eq",
                    "value": "waiting",
                },
                {"column": "category", "operator": "eq", "value": "billing"},
            ],
            "assignments": [{"column": "priority", "value": self._desired_priority}],
        }

    @staticmethod
    def _latest_result(request: ModelRequest, call_id: str) -> ToolResultBlock:
        for message in reversed(request.messages):
            for block in reversed(message.content):
                if isinstance(block, ToolResultBlock) and block.call_id == call_id:
                    return block
        raise AssertionError(f"missing tool result {call_id}")

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self._requests.append(request)
        if self._phase == 0:
            self._phase = 1
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="bulk-preview",
                        name="data_preview_postgresql_update",
                        arguments=self._plan(),
                    ),
                ),
            )
        if self._phase == 1:
            preview = self._latest_result(request, "bulk-preview")
            assert preview.is_error is False
            data = preview.output["data"]
            assert isinstance(data, Mapping)
            matched_rows = data["matched_rows"]
            preview_fingerprint = data["preview_fingerprint"]
            assert isinstance(matched_rows, int)
            assert matched_rows > 1
            assert isinstance(preview_fingerprint, str)
            self._matched_rows = matched_rows
            self._phase = 2
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="bulk-update",
                        name="data_update_postgresql",
                        arguments={
                            **self._plan(),
                            "preview_fingerprint": preview_fingerprint,
                            "expected_affected_rows": matched_rows,
                        },
                    ),
                ),
            )
        if self._phase == 2:
            update = self._latest_result(request, "bulk-update")
            assert update.is_error is False
            data = update.output["data"]
            assert isinstance(data, Mapping)
            assert data["outcome"] == "committed"
            assert data["affected_rows"] == self._matched_rows
            self._phase = 3
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="bulk-readback",
                        name="data_query_postgresql",
                        arguments={
                            "source_id": self._source_id,
                            "sql": (
                                "SELECT priority, COUNT(*) AS matched_rows "
                                "FROM support.tickets "
                                "WHERE ticket_status = $1 AND category = $2 "
                                "GROUP BY priority ORDER BY priority"
                            ),
                            "parameters": ["waiting", "billing"],
                        },
                    ),
                ),
            )
        if self._phase == 3:
            readback = self._latest_result(request, "bulk-readback")
            assert readback.is_error is False
            data = readback.output["data"]
            assert isinstance(data, Mapping)
            rows = data["rows"]
            assert isinstance(rows, tuple)
            assert len(rows) == 1
            row = rows[0]
            assert isinstance(row, Mapping)
            assert row["priority"] == self._desired_priority
            assert row["matched_rows"] == self._matched_rows
            self._phase = 4
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                text=(f"Committed and verified {self._matched_rows} ticket updates."),
            )
        raise AssertionError("bulk update provider received an unexpected model call")


async def _restore_bulk_priority(*, port: int, password: str) -> None:
    import asyncpg  # type: ignore[import-untyped]

    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=port,
        database="daita_large_fixture",
        user="daita_large_writer",
        password=password,
        ssl=False,
    )
    try:
        await connection.execute(
            "UPDATE support.tickets SET priority = 'low' "
            "WHERE ticket_status = 'waiting' AND category = 'billing'"
        )
    finally:
        await connection.close()


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
    assert "CREATE ROLE daita_large_writer" in init
    assert "NOBYPASSRLS" in init
    assert "REVOKE ALL PRIVILEGES ON DATABASE daita_large_fixture FROM PUBLIC" in init
    assert "GRANT SELECT ON support.tickets TO daita_large_writer" in init
    assert "GRANT UPDATE (priority) ON support.tickets TO daita_large_writer" in init
    assert "daita_large_writer_fixture_password" in readme
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
        workspace=workspace_for(tmp_path),
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


@pytest.mark.acceptance
@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.skipif(
    os.environ.get("DAITA_RUN_POSTGRES_LARGE_FIXTURE") != "1",
    reason=(
        "set DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 after recreating "
        "tests/fixtures/postgres-large/compose.yaml"
    ),
)
async def test_terminal_write_permissions_use_large_support_tickets(
    tmp_path: Path,
) -> None:
    password = os.environ.get(
        "DAITA_LARGE_POSTGRES_WRITER_PASSWORD",
        "daita_large_writer_fixture_password",
    )
    port = int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433"))
    credential = SecretReference.keychain("fixture:postgres-large:writer-credential")
    secrets = _Secrets(password)
    provider = MockModelProvider(())
    agent = await Agent.create(
        "postgres-large-terminal-write-permissions",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        secret_provider=secrets,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach_postgresql(
            host="127.0.0.1",
            port=port,
            database="daita_large_fixture",
            username="daita_large_writer",
            credential=credential,
            schemas=("support",),
            ssl_mode="disable",
            name="Large PostgreSQL write canary",
        )
        inspection = await agent.inspect_source_permissions(source.id)
        tickets = next(
            resource
            for resource in inspection.resources
            if resource.display_name == "support.tickets"
        )
        assert "priority" in tickets.eligible_assignment_columns
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode=inspection.state.read_scope.mode,
            read_resource_ids=inspection.state.read_scope.resource_ids,
            postgresql_update_scopes={tickets.resource_id: ("priority",)},
        )
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )

        after = await agent.inspect_source_permissions(source.id)
        assert len(after.state.postgresql_update_scopes) == 1
        scope = after.state.postgresql_update_scopes[0]
        assert scope.resource_id == tickets.resource_id
        assert scope.allowed_assignment_columns == ("priority",)
        readiness = await agent.postgresql_update_readiness(
            source.id,
            tickets.resource_id,
            ("priority",),
        )
        assert readiness.ready_for_preview is True
        assert provider.requests == ()
    finally:
        await agent.close()


@pytest.mark.acceptance
@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.skipif(
    os.environ.get("DAITA_RUN_POSTGRES_LARGE_FIXTURE") != "1",
    reason=(
        "set DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 after recreating "
        "tests/fixtures/postgres-large/compose.yaml"
    ),
)
async def test_bulk_update_uses_exact_preview_approval_commit_and_readback(
    tmp_path: Path,
) -> None:
    password = os.environ.get(
        "DAITA_LARGE_POSTGRES_WRITER_PASSWORD",
        "daita_large_writer_fixture_password",
    )
    port = int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433"))
    credential = SecretReference.keychain("fixture:postgres-large:bulk-credential")
    secrets = _Secrets(password)
    provider = _BulkUpdateProvider()
    approvals: list[ApprovalRequest] = []

    async def approve_once(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "postgres-large-bulk-update",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        secret_provider=secrets,
        approval_handler=approve_once,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach_postgresql(
            host="127.0.0.1",
            port=port,
            database="daita_large_fixture",
            username="daita_large_writer",
            credential=credential,
            schemas=("support",),
            ssl_mode="disable",
            name="Large PostgreSQL bulk update",
        )
        inspection = await agent.inspect_source_permissions(source.id)
        tickets = next(
            resource
            for resource in inspection.resources
            if resource.display_name == "support.tickets"
        )
        permissions = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="all",
            read_resource_ids=(),
            postgresql_update_scopes={tickets.resource_id: ("priority",)},
        )
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=permissions.confirmation_fingerprint,
        )

        provider.configure(
            source_id=source.id,
            resource_id=tickets.resource_id,
            desired_priority="high",
        )
        applied = await agent.run(
            "Set waiting billing tickets to high priority and verify the result."
        )
        assert applied.final_text is not None
        assert applied.final_text.startswith("Committed and verified ")

        provider.configure(
            source_id=source.id,
            resource_id=tickets.resource_id,
            desired_priority="low",
        )
        restored = await agent.run(
            "Restore waiting billing tickets to low priority and verify the result."
        )
        assert restored.final_text is not None
        assert restored.final_text.startswith("Committed and verified ")

        assert len(approvals) == 2
        assert all(item.tool_name == "data_update_postgresql" for item in approvals)
        for request in approvals:
            expected_rows = request.arguments["expected_affected_rows"]
            preview_fingerprint = request.arguments["preview_fingerprint"]
            assert isinstance(expected_rows, int)
            assert expected_rows > 1
            assert isinstance(preview_fingerprint, str)
            assert preview_fingerprint.startswith("sha256:")
    finally:
        await agent.close()
        await _restore_bulk_priority(port=port, password=password)
