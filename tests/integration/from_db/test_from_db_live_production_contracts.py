"""Live production contract tests for ``Agent.from_db``.

Run:
    DAITA_RUN_LIVE_LLM=1 pytest \
        tests/integration/from_db/test_from_db_live_production_contracts.py \
        -m "requires_llm and integration" -q -rs -s
"""

from __future__ import annotations

import re
import sys

import pytest

from tests.integration.from_db.live_production_helpers import (
    assert_loop_evidence,
    assert_no_unexpected_write_execution,
    assert_sql_is_read_only,
    assert_successful_prompt_run,
    assert_synthesized_answer,
    create_live_sqlite_from_db_agent,
    evidence_kinds,
    latest_evidence,
    seed_rich_sqlite_schema,
    sql_from_result,
    task_capabilities,
    write_failure_artifacts,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


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
        rows = query_result.payload.get("rows") or []
        assert any(4 in row.values() for row in rows if isinstance(row, dict))
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
