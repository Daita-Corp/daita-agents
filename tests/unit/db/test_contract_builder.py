from daita.db import DbIntentKind, DbRequest, DbRuntime
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import AccessMode


def _runtime() -> DbRuntime:
    return DbRuntime(
        plugins=(
            CatalogPlugin(auto_persist=False),
            SQLitePlugin(path=":memory:"),
        )
    )


def test_schema_question_contract_forbids_sql_execution_capabilities():
    runtime = _runtime()
    request = DbRequest("What columns are in the customers table?")

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.SCHEMA_QUERY
    assert contract.operation_type == "schema.query"
    assert contract.access is AccessMode.METADATA_READ
    assert "catalog.schema.search" in contract.required_capabilities
    assert "catalog.asset.inspect" in contract.required_capabilities
    assert "db.schema.inspect" in contract.required_capabilities
    assert "db.sql.execute_read" not in contract.required_capabilities
    assert "query.result" not in contract.required_evidence
    blocked = {item["id"] for item in contract.metadata["blocked_capabilities"]}
    assert "db.sql.execute_read" in blocked
    assert "db.sql.execute_write" in blocked


def test_schema_search_with_metric_terms_stays_schema_only():
    runtime = _runtime()
    request = DbRequest(
        "Search the schema for columns that might represent price, amount, "
        "total, revenue, invoice, payment, or subscription. Tell me which "
        "columns are safest to use and why."
    )

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.SCHEMA_QUERY
    assert intent.evidence_mode == "schema"
    assert contract.operation_type == "schema.query"
    assert "db.sql.execute_read" not in contract.required_capabilities
    assert "query.result" not in contract.required_evidence
    blocked = {item["id"] for item in contract.metadata["blocked_capabilities"]}
    assert "db.sql.execute_read" in blocked


def test_schema_evidence_only_prompt_forbids_row_query_capabilities():
    runtime = _runtime()
    request = DbRequest(
        "Which tables look most relevant for understanding customer billing "
        "or revenue? Use schema evidence only; do not query rows."
    )

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.SCHEMA_QUERY
    assert contract.access is AccessMode.METADATA_READ
    assert "catalog.schema.search" in contract.required_capabilities
    assert "db.sql.execute_read" not in contract.required_capabilities
    blocked = {item["id"] for item in contract.metadata["blocked_capabilities"]}
    assert "db.sql.execute_read" in blocked


def test_data_question_contract_requires_validation_and_query_evidence():
    runtime = _runtime()
    request = DbRequest("How many orders are there?")

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.DATA_QUERY
    assert contract.operation_type == "data.query"
    assert contract.access is AccessMode.READ
    assert contract.required_capabilities == (
        "db.sql.validate",
        "db.sql.execute_read",
    )
    assert "sql.validation" in contract.required_evidence
    assert "query.result" in contract.required_evidence
    assert contract.metadata["missing_capabilities"] == []


def test_schema_assisted_calculation_uses_catalog_assisted_data_contract():
    runtime = _runtime()
    request = DbRequest(
        "Find the safest revenue column, then calculate total revenue by "
        "billing period."
    )

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.CATALOG_ASSISTED_DATA_QUERY
    assert intent.evidence_mode == "query_and_relationships"
    assert {
        "catalog.schema.search",
        "catalog.relationship_paths.find",
        "db.sql.validate",
        "db.sql.execute_read",
    } <= set(contract.required_capabilities)
    assert "schema.relationship_path" in contract.required_evidence
    assert "query.result" in contract.required_evidence


def test_catalog_assisted_data_contract_selects_catalog_and_query_capabilities():
    runtime = _runtime()
    request = DbRequest("Join orders to customers using their relationship")

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.CATALOG_ASSISTED_DATA_QUERY
    assert contract.operation_type == "data.query"
    assert {
        "catalog.schema.search",
        "catalog.relationship_paths.find",
        "db.sql.validate",
        "db.sql.execute_read",
    } <= set(contract.required_capabilities)
    assert "schema.relationship_path" in contract.required_evidence
    assert "query.result" in contract.required_evidence
    selected = contract.metadata["selected_capabilities"]
    assert all("executor" in item for item in selected)


def test_write_proposal_contract_requires_validation_and_approval_without_execution():
    runtime = _runtime()
    request = DbRequest("Update the orders total to 50")

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.WRITE_PROPOSE
    assert contract.operation_type == "write.propose"
    assert contract.access is AccessMode.METADATA_READ
    assert contract.required_capabilities == ("db.sql.validate",)
    assert "db.sql.execute_write" not in contract.required_capabilities
    assert contract.metadata["approval_required"] is True
    assert "runtime:approval_required_for_writes" in contract.policy_ids


def test_write_execution_contract_selects_write_capability_and_requires_approval():
    runtime = _runtime()
    request = DbRequest("Execute update orders set total = 50 where id = 10")

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.WRITE_EXECUTE
    assert contract.operation_type == "write.execute"
    assert contract.access is AccessMode.WRITE
    assert contract.required_capabilities == (
        "db.sql.validate",
        "db.sql.execute_write",
    )
    assert "write.execution" in contract.required_evidence
    assert contract.metadata["approval_required"] is True
    assert "runtime:approval_required_for_writes" in contract.policy_ids


def test_missing_capability_diagnostics_explain_unavailable_services():
    runtime = DbRuntime(plugins=(SQLitePlugin(path=":memory:"),))
    request = DbRequest("Check data quality for the orders table", mode="quality.check")

    intent = runtime.classify_request(request)
    contract = runtime.build_contract(request, intent)

    assert intent.kind is DbIntentKind.QUALITY_CHECK
    assert contract.required_capabilities == ()
    assert contract.required_evidence == ()
    assert contract.metadata["missing_capabilities"] == ["quality.profile"]


def test_lineage_and_memory_contracts_report_missing_declared_services():
    runtime = _runtime()

    lineage = runtime.build_contract(DbRequest("Trace lineage for orders"))
    memory = runtime.build_contract(DbRequest("Remember that revenue excludes tax"))

    assert lineage.operation_type == "lineage.trace"
    assert lineage.metadata["missing_capabilities"] == ["lineage.trace"]
    assert memory.operation_type == "memory.update"
    assert memory.access is AccessMode.WRITE
    assert memory.metadata["missing_capabilities"] == ["memory.semantic.write"]


def test_conversational_contract_requires_no_capabilities_or_evidence():
    runtime = _runtime()

    contract = runtime.build_contract(DbRequest("Hello there"))

    assert contract.operation_type == "conversational"
    assert contract.access is AccessMode.NONE
    assert contract.required_capabilities == ()
    assert contract.required_evidence == ()
    assert contract.metadata["missing_capabilities"] == []


def test_requested_capabilities_are_included_when_allowed():
    runtime = _runtime()
    request = DbRequest(
        "How many orders are there?",
        requested_capabilities=("catalog.schema.search",),
    )

    contract = runtime.build_contract(request)

    assert "catalog.schema.search" in contract.required_capabilities
    selected = {
        (item["id"], item["reason"])
        for item in contract.metadata["selected_capabilities"]
    }
    assert ("catalog.schema.search", "requested") in selected


async def test_run_uses_classified_intent_and_contract_metadata():
    runtime = _runtime()

    try:
        result = await runtime.run("How many orders are there?")
    finally:
        await runtime.teardown()

    assert result.intent.kind is DbIntentKind.DATA_QUERY
    assert result.contract.operation_type == "data.query"
    assert result.contract.required_capabilities == (
        "db.sql.validate",
        "db.sql.execute_read",
    )
    assert result.diagnostics["contract"]["missing_capabilities"] == []
