"""Component-owned tests split from ``test_large_fixture_contract.py``."""

from __future__ import annotations

from tests.data.postgresql._large_fixture_support import (
    ATTACHED_SCHEMAS,
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    Decimal,
    FinishReason,
    Mapping,
    MockModelProvider,
    ModelResponse,
    Path,
    SecretReference,
    ToolCall,
    _BulkUpdateProvider,
    _profile,
    _restore_bulk_priority,
    _Secrets,
    _tool_results,
    os,
    pytest,
    update_constraints,
    workspace_for,
)


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
                        name="data_query",
                        arguments={
                            "source_id": source.id,
                            "resource_ids": tuple(
                                by_native_identity[name].id
                                for name in (
                                    "core.customers",
                                    "core.organizations",
                                    "core.regions",
                                    "sales.orders",
                                    "billing.invoices",
                                )
                            ),
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
            relational_write_scopes={
                tickets.resource_id: update_constraints(
                    ("priority",), key_columns=tickets.key_columns
                )
            },
        )
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )

        after = await agent.inspect_source_permissions(source.id)
        assert len(after.state.relational_write_scopes) == 1
        scope = after.state.relational_write_scopes[0]
        assert scope.resource_id == tickets.resource_id
        assert scope.allowed_update_columns == ("priority",)
        readiness = await agent.relational_update_readiness(
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
            relational_write_scopes={
                tickets.resource_id: update_constraints(
                    ("priority",), key_columns=tickets.key_columns
                )
            },
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
        assert all(item.tool_name == "data_update_rows" for item in approvals)
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
