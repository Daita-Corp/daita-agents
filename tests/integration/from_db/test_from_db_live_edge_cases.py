"""Live edge-case contract tests for ``Agent.from_db`` Bucket 1 behavior.

Run:
    DAITA_RUN_LIVE_LLM=1 pytest \
        tests/integration/from_db/test_from_db_live_edge_cases.py \
        -m "requires_llm and integration" -q -rs -s
"""

from __future__ import annotations

from collections import Counter
import re
import sys

import pytest

from daita.runtime import OperationStatus
from tests.integration.from_db.live_production_helpers import (
    all_sql_strings,
    assert_loop_evidence,
    assert_no_unexpected_write_execution,
    assert_sql_is_read_only,
    assert_successful_prompt_run,
    assert_synthesized_answer,
    create_live_sqlite_from_db_agent,
    diagnostic_text,
    evidence_kinds,
    latest_evidence,
    query_rows,
    seed_rich_sqlite_schema,
    sql_from_result,
    task_capabilities,
    write_failure_artifacts,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


async def test_live_schema_query_disambiguates_near_collision_tables(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LiveFromDbSchemaNearCollision",
    )
    result = None
    snapshot = None

    try:
        result = await agent.run_detailed(
            "can you tell me about the operations table?",
            mode="schema.query",
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)

        assert snapshot is not None
        assert_successful_prompt_run(result, snapshot=snapshot)
        assert_synthesized_answer(snapshot, public_result=result)
        assert "schema.asset_profile" in evidence_kinds(result)
        assert "query.result" not in evidence_kinds(result)

        synthesis = latest_evidence(snapshot, "answer.synthesis")
        assert synthesis is not None
        diagnostics = synthesis.payload.get("diagnostics") or {}
        context = diagnostics.get("context") or {}
        scope = context.get("schema_answer_scope") or {}
        assert scope.get("mode") == "asset"
        assert scope.get("selected_table_names") == ["operations"]

        answer = (result.answer or "").lower()
        assert "operations" in answer
        assert "operation_id" in answer
        assert "runtime_operation_id" not in answer
        assert "worker_id" not in answer
        assert "monitor_id" not in answer
        assert "enabled" not in answer
        assert "audit_logs" not in answer

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


async def test_live_zero_row_result_is_truthful(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LiveFromDbZeroRows",
    )
    result = None
    snapshot = None

    try:
        result = await agent.run_detailed(
            "Return order_id and status for orders whose status is pending "
            "and total is greater than 1000."
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)

        assert snapshot is not None
        assert_successful_prompt_run(result, snapshot=snapshot)
        assert_loop_evidence(result)
        assert_loop_evidence(snapshot)
        assert_synthesized_answer(snapshot, public_result=result)

        rows = query_rows(snapshot)
        assert rows == []

        sql = sql_from_result(snapshot)
        assert_sql_is_read_only(sql)
        assert re.search(r"(?i)\borders\b", sql), sql
        assert re.search(r"(?i)\bstatus\b", sql), sql
        assert re.search(r"(?i)\btotal\b", sql), sql

        answer = (result.answer or "").lower()
        assert any(
            phrase in answer
            for phrase in (
                "no rows",
                "no matching",
                "no canceled",
                "no cancelled",
                "no orders",
                "no pending",
                "no data",
                "0 rows",
                "zero rows",
            )
        ), result.answer

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


async def test_live_sql_repair_loop_recovers_from_bad_literal_or_column(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LiveFromDbRepairBadLiteral",
    )
    result = None
    snapshot = None

    try:
        warmup = await agent.run_detailed(
            "Search catalog column values for support_tickets.status and "
            "support_tickets.severity."
        )
        warmup_snapshot = await agent.runtime.inspect_operation(warmup.operation_id)
        assert warmup_snapshot is not None
        assert "schema.column_value_search_result" in evidence_kinds(warmup_snapshot)
        result = await agent.run_detailed(
            "Using observed catalog column values, answer: how many support tickets "
            "have status='unresolved' and severity='urgent'?"
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)

        assert snapshot is not None
        assert result.status in {
            OperationStatus.SUCCEEDED,
            OperationStatus.BLOCKED,
            OperationStatus.FAILED,
        }

        capabilities = task_capabilities(snapshot)
        assert capabilities.count("db.query.repair") <= 2

        sql_values = all_sql_strings(snapshot)
        for sql in sql_values:
            assert_sql_is_read_only(sql)

        failing_sql = [
            item.payload.get("sql")
            for item in snapshot.evidence
            if item.kind in {"sql.validation", "query.plan.validation"}
            and item.accepted is False
            and isinstance(item.payload.get("sql"), str)
        ]
        repeated = [
            sql for sql, count in Counter(failing_sql).items() if sql and count > 2
        ]
        assert not repeated, f"Repeated identical failing SQL: {repeated}"

        repair_attempts = [
            item
            for item in snapshot.evidence
            if item.kind in {"query.plan.repair", "query.plan.proposal"}
            and item.payload.get("repair_attempt") is not None
        ]
        if result.status is OperationStatus.SUCCEEDED:
            assert_successful_prompt_run(result, snapshot=snapshot)
            assert_loop_evidence(result)
            assert_synthesized_answer(snapshot, public_result=result)

            sql = sql_from_result(snapshot)
            assert re.search(r"(?i)\bsupport_tickets\b", sql), sql
            assert not re.search(r"(?i)['\"]unresolved['\"]", sql), sql
            assert not re.search(r"(?i)['\"]urgent['\"]", sql), sql
            assert re.search(r"(?i)['\"](open|closed)['\"]", sql), sql
            assert re.search(r"(?i)['\"](high|low)['\"]", sql), sql
            assert (
                repair_attempts
                or "schema.column_value_search_result" in evidence_kinds(snapshot)
            )
        else:
            text = diagnostic_text(snapshot).lower()
            assert any(
                token in text
                for token in (
                    "unobserved_filter_literal",
                    "repair",
                    "validation",
                    "unresolved",
                    "urgent",
                )
            ), text

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


async def test_live_ambiguous_prompt_does_not_loop_or_overquery(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LiveFromDbAmbiguousBounded",
    )
    result = None
    snapshot = None

    try:
        result = await agent.run_detailed("Show me its status from the database.")
        snapshot = await agent.runtime.inspect_operation(result.operation_id)

        assert snapshot is not None
        assert result.status in {OperationStatus.BLOCKED, OperationStatus.FAILED}
        assert snapshot.operation.status is result.status

        decisions = [
            item for item in snapshot.evidence if item.kind == "planner.decision"
        ]
        assert 1 <= len(decisions) <= 5

        kinds = evidence_kinds(snapshot)
        assert "planner.decision" in kinds
        assert "query.result" not in kinds
        assert not any(
            capability == "db.sql.execute_read"
            for capability in task_capabilities(snapshot)
        )
        assert len(snapshot.tasks) <= 8

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
