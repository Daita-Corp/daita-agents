"""Bucket 3 live contracts for DB session-scoped structured memory.

Run:
    DAITA_RUN_LIVE_LLM=1 pytest \
        tests/integration/from_db/test_from_db_live_memory_contracts.py \
        -m "requires_llm and integration" -q -rs -s
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from daita.agents.agent import Agent
from daita.runtime import OperationStatus

from tests.integration.from_db.live_production_helpers import (
    assert_loop_evidence,
    assert_synthesized_answer,
    diagnostic_text,
    latest_evidence,
    require_live_openai_kwargs,
)
from tests.integration.from_db.test_from_db_memory_live import (
    _db_memory_keys,
    _memory_backend,
    _memory_option,
    _planning_context,
    _rows,
    _run_live,
    _seed_revenue_db,
    _source_identity,
    _sql,
    _write_memory,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


async def test_live_memory_metric_definition_changes_future_planning(tmp_path):
    db_path = tmp_path / "recognized-revenue.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryRecognizedRevenueContract",
        memory=_memory_option(tmp_path, "recognized-revenue-memory"),
        cache_ttl=0,
        **require_live_openai_kwargs(),
    )

    try:
        before = await _run_live(
            agent,
            "Calculate recognized revenue from orders.total.",
        )
        await _write_memory(
            agent,
            kind="metric_definition",
            key="metric:recognized_revenue",
            text=(
                "Recognized revenue is SUM(orders.total) only for orders whose "
                "status is complete."
            ),
            metadata={
                "aliases": ["recognized revenue"],
                "schema_refs": ["orders.total", "orders.status"],
            },
        )
        after = await _run_live(
            agent,
            "Calculate recognized revenue from orders.total.",
        )
    finally:
        await agent.stop()

    assert "metric:recognized_revenue" not in _db_memory_keys(before)
    assert after.status is OperationStatus.SUCCEEDED
    assert_loop_evidence(after)
    assert_synthesized_answer(after)
    assert "metric:recognized_revenue" in _db_memory_keys(after)
    sql = _sql(after).lower()
    assert "orders" in sql
    assert "total" in sql
    assert "complete" in sql
    assert _has_numeric_value(_rows(after), 345.0)


async def test_live_memory_source_scope_and_stale_filters(tmp_path):
    db_path = tmp_path / "source-scope.sqlite"
    await _seed_revenue_db(db_path)
    other_db_path = tmp_path / "source-scope-other.sqlite"
    await _seed_revenue_db(other_db_path)
    shared_backend = _memory_backend(tmp_path, "bucket3-source-scope-memory")

    other_source = await Agent.from_db(
        str(other_db_path),
        name="LiveMemoryBucket3OtherSource",
        memory={"backend": shared_backend, "score_threshold": 0.0},
        cache_ttl=0,
        **require_live_openai_kwargs(),
    )
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryBucket3SourceScope",
        memory={"backend": shared_backend, "score_threshold": 0.0},
        cache_ttl=0,
        **require_live_openai_kwargs(),
    )

    try:
        source_identity = await _source_identity(agent)
        other_source_identity = await _source_identity(other_source)
        assert source_identity != other_source_identity

        await _write_memory(
            agent,
            kind="metric_definition",
            key="metric:active_memory",
            text="Active memory revenue is SUM(orders.total) for complete orders.",
            source_identity=source_identity,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total", "orders.status"],
            },
        )
        await _write_memory(
            agent,
            kind="metric_definition",
            key="metric:inactive_memory",
            text="Inactive memory revenue should never affect planning.",
            source_identity=source_identity,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total", "orders.status"],
            },
            active=False,
        )
        await _write_memory(
            agent,
            kind="metric_definition",
            key="metric:stale_memory",
            text="Stale memory revenue should never affect planning.",
            source_identity=source_identity,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total", "orders.status"],
            },
            stale=True,
        )
        await _write_memory(
            agent,
            kind="metric_definition",
            key="metric:expired_memory",
            text="Expired memory revenue should never affect planning.",
            source_identity=source_identity,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total", "orders.status"],
            },
            expires_at="2000-01-01T00:00:00+00:00",
        )
        await _write_memory(
            agent,
            kind="metric_definition",
            key="metric:stale_schema_memory",
            text="Stale schema memory revenue should never affect planning.",
            source_identity=source_identity,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total", "orders.status"],
                "schema_fingerprint": "stale-schema-fingerprint",
            },
        )
        await _write_memory(
            other_source,
            kind="metric_definition",
            key="metric:cross_source_memory",
            text="Cross source memory revenue should never affect planning.",
            source_identity=other_source_identity,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total", "orders.status"],
            },
        )

        result = await _run_live(
            agent,
            "Calculate active memory revenue from orders.total.",
        )
    finally:
        await other_source.stop()
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert_loop_evidence(result)
    assert_synthesized_answer(result)
    context = _planning_context(result)
    keys = _db_memory_keys(result)
    assert keys == ["metric:active_memory"]
    assert "metric:inactive_memory" not in keys
    assert "metric:stale_memory" not in keys
    assert "metric:expired_memory" not in keys
    assert "metric:stale_schema_memory" not in keys
    assert "metric:cross_source_memory" not in keys
    diagnostics = context.get("db_memory_diagnostics") or {}
    assert diagnostics.get("included_count") == 1
    assert _has_numeric_value(_rows(result), 345.0)
    assert "complete" in _sql(result).lower()


async def test_live_memory_pii_candidate_rejected(tmp_path):
    db_path = tmp_path / "pii-candidate.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryBucket3PiiCandidate",
        memory=_memory_option(tmp_path, "bucket3-pii-memory"),
        cache_ttl=0,
        **require_live_openai_kwargs(),
    )
    blocked_value = "ada@example.com"

    try:
        evidence = await agent.runtime.execute_capability(
            "memory.semantic.write",
            owner="memory",
            operation_type="memory.update",
            input={
                "db_memory_payload": {
                    "kind": "business_rule",
                    "key": f"business_rule:customer_email_{blocked_value}",
                    "text": f"Ada customer email is {blocked_value}.",
                    "metadata": {
                        "source_identity": await _source_identity(agent),
                        "workspace_scope": "source",
                    },
                },
                "db_memory_prompt": f"Ada customer email is {blocked_value}.",
            },
        )
        memory = agent.runtime.registry.get_plugin("memory")
        records = await memory.backend.list_db_records(
            category="db_semantics",
            limit=10,
        )
        result = await _run_live(
            agent, "Calculate recognized revenue from orders.total."
        )
    finally:
        await agent.stop()

    assert evidence[0].kind == "memory.semantic.write"
    assert evidence[0].payload["success"] is False
    assert (
        "PII" in evidence[0].payload["error"]
        or "row-level" in evidence[0].payload["error"]
    )
    assert blocked_value not in json.dumps(evidence[0].payload, default=str)
    assert records == []

    assert result.status is OperationStatus.SUCCEEDED
    assert_loop_evidence(result)
    assert_synthesized_answer(result)
    context = _planning_context(result)
    assert not context.get("db_memory_refs")
    assert blocked_value not in diagnostic_text(result)
    synthesis = latest_evidence(result, "answer.synthesis")
    assert synthesis is not None
    assert blocked_value not in json.dumps(synthesis.payload, default=str)


def _has_numeric_value(rows: list[dict[str, Any]], expected: float) -> bool:
    for row in rows:
        for value in row.values():
            try:
                if float(str(value).replace(",", "")) == expected:
                    return True
            except (TypeError, ValueError):
                continue
    return False
