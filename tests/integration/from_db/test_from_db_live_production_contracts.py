"""Live production contract tests for ``Agent.from_db``.

Run:
    DAITA_RUN_LIVE_LLM=1 pytest \
        tests/integration/from_db/test_from_db_live_production_contracts.py \
        -m "requires_llm and integration" -q -rs -s
"""

from __future__ import annotations

import asyncio
import re
import sys

import pytest

from tests.integration.from_db.live_production_helpers import (
    assert_loop_evidence,
    assert_no_unexpected_write_execution,
    assert_sql_is_read_only,
    assert_successful_prompt_run,
    assert_synthesized_answer,
    create_live_postgres_from_db_agent,
    create_live_sqlite_from_db_agent,
    evidence_kinds,
    latest_evidence,
    query_rows,
    require_live_postgres_enabled,
    row_values,
    seed_rich_postgres_schema,
    seed_rich_sqlite_schema,
    sql_from_result,
    task_capabilities,
    write_failure_artifacts,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


@pytest.fixture(scope="module")
def seeded_rich_postgres_url():
    require_live_postgres_enabled()
    from tests.integration._harness import start_container

    container = start_container(
        "postgres:16-alpine",
        container_port=5432,
        env={
            "POSTGRES_USER": "daita",
            "POSTGRES_PASSWORD": "daita_test_pw",
            "POSTGRES_DB": "daita_from_db_bucket1",
        },
        tag_prefix="daita-from-db-bucket1-pg",
    )
    url = (
        "postgresql://daita:daita_test_pw"
        f"@{container.host}:{container.host_port}/daita_from_db_bucket1"
    )
    try:
        asyncio.run(seed_rich_postgres_schema(url))
        yield url
    finally:
        container.remove()


@pytest.mark.integration
@pytest.mark.requires_llm
async def test_live_sqlite_simple_query_full_loop_contract(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LiveFromDbProductionSmoke",
    )
    result = None
    snapshot = None

    try:
        result = await agent.run_detailed("How many customers are there?")
        snapshot = await agent.runtime.inspect_operation(result.operation_id)

        assert snapshot is not None
        assert_successful_prompt_run(result, snapshot=snapshot)
        assert_loop_evidence(result)
        assert_loop_evidence(snapshot)
        assert_synthesized_answer(result)
        assert "db.answer.synthesize" in task_capabilities(result)
        assert "db.answer.synthesize" in task_capabilities(snapshot)

        query_result = latest_evidence(result, "query.result")
        assert query_result is not None
        assert 4 in row_values(result)
        assert re.search(r"\b4\b", result.answer or "")

        sql = sql_from_result(result) or sql_from_result(snapshot)
        assert_sql_is_read_only(sql)
        assert re.search(r"(?i)\bcustomers\b", sql), sql
        assert re.search(r"(?i)\bcount\s*\(", sql), sql

        assert {
            "schema.asset_profile",
            "query.plan.proposal",
            "sql.validation",
        } <= evidence_kinds(result)
        assert_no_unexpected_write_execution(result)
        assert_no_unexpected_write_execution(snapshot)
    except AssertionError:
        artifact_dir = write_failure_artifacts(
            tmp_path / "from_db_failure_artifacts",
            result=result,
            snapshot=snapshot,
        )
        print(f"[from_db artifacts] {artifact_dir}", file=sys.stderr)
        raise
    finally:
        await agent.stop()


@pytest.mark.requires_db
async def test_live_postgres_simple_query_full_loop_contract(
    tmp_path,
    seeded_rich_postgres_url,
):
    agent = await create_live_postgres_from_db_agent(
        seeded_rich_postgres_url,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LiveFromDbPostgresProductionSmoke",
    )
    result = None
    snapshot = None

    try:
        result = await agent.run_detailed("How many customers are there?")
        snapshot = await agent.runtime.inspect_operation(result.operation_id)

        assert snapshot is not None
        assert_successful_prompt_run(result, snapshot=snapshot)
        assert_loop_evidence(result)
        assert_loop_evidence(snapshot)
        assert_synthesized_answer(result)
        assert "db.answer.synthesize" in task_capabilities(result)
        assert "db.answer.synthesize" in task_capabilities(snapshot)

        assert 4 in row_values(result)
        assert re.search(r"\b4\b", result.answer or "")

        sql = sql_from_result(result) or sql_from_result(snapshot)
        assert_sql_is_read_only(sql)
        assert re.search(r"(?i)\bcustomers\b", sql), sql
        assert re.search(r"(?i)\bcount\s*\(", sql), sql

        assert {
            "schema.asset_profile",
            "query.plan.proposal",
            "sql.validation",
        } <= evidence_kinds(result)
        assert_no_unexpected_write_execution(result)
        assert_no_unexpected_write_execution(snapshot)
    except AssertionError:
        artifact_dir = write_failure_artifacts(
            tmp_path / "from_db_failure_artifacts",
            result=result,
            snapshot=snapshot,
        )
        print(f"[from_db artifacts] {artifact_dir}", file=sys.stderr)
        raise
    finally:
        await agent.stop()


async def test_live_catalog_relationship_join_uses_catalog_paths(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LiveFromDbCatalogRelationshipJoin",
    )
    result = None
    snapshot = None

    try:
        result = await agent.run_detailed(
            "Join orders to customers using their relationship and return records"
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)

        assert snapshot is not None
        assert_successful_prompt_run(result, snapshot=snapshot)
        assert_loop_evidence(result)
        assert_loop_evidence(snapshot)
        assert_synthesized_answer(result)

        assert {
            "catalog.source_registered",
            "schema.relationship_path",
            "query.plan.proposal",
            "sql.validation",
            "query.result",
            "verification.result",
            "answer.synthesis",
        } <= evidence_kinds(result)
        assert "catalog.relationship_paths.find" in task_capabilities(result)

        relationship_path = latest_evidence(result, "schema.relationship_path")
        assert relationship_path is not None
        assert relationship_path.payload

        rows = query_rows(result)
        values = {str(value) for row in rows for value in row.values()}
        assert len(rows) >= 5
        assert "Ada Lovelace" in values
        assert "100" in values

        sql = sql_from_result(result) or sql_from_result(snapshot)
        assert_sql_is_read_only(sql)
        assert re.search(r"(?i)\borders\b", sql), sql
        assert re.search(r"(?i)\bcustomers\b", sql), sql
        assert re.search(r"(?i)\bjoin\b", sql), sql

        assert_no_unexpected_write_execution(result)
        assert_no_unexpected_write_execution(snapshot)
    except AssertionError:
        artifact_dir = write_failure_artifacts(
            tmp_path / "from_db_failure_artifacts",
            result=result,
            snapshot=snapshot,
        )
        print(f"[from_db artifacts] {artifact_dir}", file=sys.stderr)
        raise
    finally:
        await agent.stop()


async def test_live_literal_value_grounding_completed_vs_complete(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LiveFromDbLiteralValueGrounding",
    )
    result = None
    snapshot = None

    try:
        result = await agent.run_detailed(
            "Search catalog column values for orders.status, then show completed "
            "orders by status."
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)

        assert snapshot is not None
        assert_successful_prompt_run(result, snapshot=snapshot)
        assert_loop_evidence(result)
        assert_loop_evidence(snapshot)
        assert_synthesized_answer(result)
        assert "catalog.column_values.search" in task_capabilities(result)
        assert "schema.column_value_search_result" in evidence_kinds(result)

        rows = query_rows(result)
        statuses = {row.get("status") for row in rows if "status" in row}
        values = row_values(result)
        assert rows
        assert statuses == {"complete"} or 3 in values

        sql = sql_from_result(result) or sql_from_result(snapshot)
        assert_sql_is_read_only(sql)
        lowered_sql = sql.lower()
        assert "orders" in lowered_sql
        assert "status" in lowered_sql
        assert re.search(r"(?i)['\"]complete['\"]", sql), sql
        assert not re.search(r"(?i)['\"]completed['\"]", sql), sql

        assert_no_unexpected_write_execution(result)
        assert_no_unexpected_write_execution(snapshot)
    except AssertionError:
        artifact_dir = write_failure_artifacts(
            tmp_path / "from_db_failure_artifacts",
            result=result,
            snapshot=snapshot,
        )
        print(f"[from_db artifacts] {artifact_dir}", file=sys.stderr)
        raise
    finally:
        await agent.stop()
