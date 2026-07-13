"""Live-gated integration tests for structured DB semantic memory.

Run:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/from_db/test_from_db_memory_live.py \
        -m "requires_llm and integration" -v -s

These tests exercise the local structured DB memory lane with live LLM planning.
Managed backend parity is covered separately.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from daita.agents.agent import Agent
from daita.db import DbLLMConfig, DbMemoryConfig, DbRequest, DbSourceOptions
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus, WorkerRuntime, WorkerRuntimeOptions
from tests.db_evidence_helpers import assert_no_invalid_accepted_query_plans
from tests.integration.from_db.live_production_helpers import (
    latest_evidence,
    query_rows,
    sql_from_result,
)

load_dotenv(Path.cwd() / ".env")

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


def _require_live_openai() -> dict[str, object]:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live DB memory tests")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return {
        "llm": DbLLMConfig(
            provider="openai",
            model=os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
            api_key=api_key,
            temperature=0,
        )
    }


async def _seed_revenue_db(path: Path) -> None:
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer TEXT NOT NULL,
            status TEXT NOT NULL,
            total REAL NOT NULL
        );
        CREATE TABLE refunds (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL REFERENCES orders(id),
            amount REAL NOT NULL
        );
        INSERT INTO orders (id, customer, status, total) VALUES
            (1, 'Ada', 'complete', 120.00),
            (2, 'Ada', 'pending', 80.00),
            (3, 'Linus', 'complete', 50.00),
            (4, 'Grace', 'complete', 175.00);
        INSERT INTO refunds (id, order_id, amount) VALUES
            (1, 4, 35.00);
        """)
    await plugin.disconnect()


async def _seed_cents_db(path: Path) -> None:
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            total_cents INTEGER NOT NULL
        );
        INSERT INTO orders (id, status, total_cents) VALUES
            (1, 'complete', 1234),
            (2, 'complete', 500),
            (3, 'pending', 800);
        """)
    await plugin.disconnect()


def _board_revenue_semantic_contract(source_identity: str | None = None) -> dict:
    return {
        "version": 1,
        "contract_kind": "metric_definition",
        "subject": {
            "type": "metric",
            "key": "metric:board_revenue",
            "aliases": ["board revenue"],
        },
        "requirements": {
            "refs": [
                {"kind": "column", "ref": "orders.total", "role": "measure"},
                {"kind": "column", "ref": "refunds.amount", "role": "adjustment"},
                {"kind": "column", "ref": "orders.status", "role": "filter"},
            ],
            "relationships": [
                {"from": "refunds.order_id", "to": "orders.id", "role": "join"}
            ],
            "filters": [
                {
                    "ref": "orders.status",
                    "operator": "semantic_equals",
                    "value": "complete",
                    "value_source": "literal_or_catalog_value",
                }
            ],
            "aggregations": [
                {"function": "sum", "ref": "orders.total", "role": "base_measure"},
                {
                    "function": "sum",
                    "ref": "refunds.amount",
                    "role": "subtractive_adjustment",
                },
            ],
            "result_shape": {"grain": "single_aggregate"},
        },
        "grounding": {
            "source_identity": source_identity,
            "catalog_refs": [],
            "evidence_refs": [],
        },
        "enforcement": {"mode": "required_when_recalled", "min_confidence": 0.8},
    }


def _board_revenue_metric_metadata() -> dict[str, Any]:
    contract = _board_revenue_semantic_contract()
    return {
        "aliases": ["board revenue"],
        "schema_refs": [
            {"table": "orders", "column": "total"},
            {"table": "orders", "column": "status"},
            {"table": "refunds", "column": "amount"},
            {"table": "refunds", "column": "order_id"},
            {"table": "orders", "column": "id"},
        ],
        "subject": contract["subject"],
        "requirements": contract["requirements"],
        "enforcement": contract["enforcement"],
    }


def _memory_backend(tmp_path: Path, workspace: str) -> LocalMemoryBackend:
    return LocalMemoryBackend(
        workspace=workspace,
        agent_id=workspace,
        scope="project",
        base_dir=tmp_path / "memory",
        embedder=None,
    )


def _memory_option(
    tmp_path: Path,
    workspace: str,
    *,
    limit: int = 3,
    char_budget: int = 800,
    score_threshold: float = 0.0,
) -> DbMemoryConfig:
    return DbMemoryConfig(
        backend=_memory_backend(tmp_path, workspace),
        limit=limit,
        char_budget=char_budget,
        score_threshold=score_threshold,
    )


async def _source_identity(agent) -> str:
    inspection = await agent.describe()
    return inspection.metadata["from_db_options"]["memory"]["source_identity"]


async def _write_memory(
    agent,
    *,
    kind: str,
    key: str,
    text: str,
    source_identity: str | None = None,
    metadata: dict[str, Any] | None = None,
    importance: float = 0.9,
    confidence: float = 0.95,
    active: bool = True,
    stale: bool = False,
    expires_at: str | None = None,
) -> tuple[Any, ...]:
    source_identity = source_identity or await _source_identity(agent)
    payload_metadata = {
        "source_identity": source_identity,
        "workspace_scope": "source",
        "active": active,
        "stale": stale,
        "confidence": confidence,
        **dict(metadata or {}),
    }
    if expires_at:
        payload_metadata["expires_at"] = expires_at
    evidence = await agent.runtime.execute_capability(
        "memory.semantic.write",
        owner="memory",
        operation_type="memory.update",
        input={
            "db_memory_payload": {
                "kind": kind,
                "key": key,
                "text": text,
                "importance": importance,
                "active": active,
                "stale": stale,
                "expires_at": expires_at,
                "metadata": payload_metadata,
            },
            "db_memory_prompt": text,
        },
    )
    assert evidence
    assert evidence[0].kind == "memory.semantic.write"
    assert evidence[0].payload["success"] is True
    return tuple(evidence)


def _evidence(result, kind: str):
    return next(item for item in result.evidence if item.kind == kind)


def _maybe_evidence(result, kind: str):
    return next((item for item in result.evidence if item.kind == kind), None)


def _sql(snapshot) -> str:
    """Return raw SQL from an operation snapshot."""
    return sql_from_result(snapshot)


def _public_planning_context(result) -> dict[str, Any]:
    """Return the latest accepted planning context on the supplied surface."""
    evidence = latest_evidence(result, "planning.context")
    if evidence is None:
        return {}
    return dict(evidence.payload)


def _raw_planning_context(snapshot) -> dict[str, Any]:
    """Return the latest accepted raw planning context from a snapshot."""
    return _public_planning_context(snapshot)


def _db_memory_keys(result_or_snapshot) -> list[str]:
    context = _public_planning_context(result_or_snapshot)
    return [str(item.get("key")) for item in context.get("db_memory_refs", [])]


async def _run_live(agent, prompt: str):
    result = None
    for _ in range(2):
        result = await agent.run_detailed(prompt)
        if "Connection error" not in str(result.answer):
            return result
    return result


async def _recall_memory(
    agent,
    query: str,
    *,
    source_identity: str | None = None,
    kinds: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "category": "db_semantics",
        "limit": 10,
        "score_threshold": 0.0,
        "retrieval_mode": "structured",
    }
    if source_identity is not None:
        payload["source_identity"] = source_identity
    if kinds is not None:
        payload["kinds"] = kinds
    evidence = await agent.runtime.execute_capability(
        "memory.semantic.recall",
        owner="memory",
        operation_type="memory.recall",
        input=payload,
    )
    assert evidence
    assert evidence[0].kind == "memory.semantic.recall"
    return dict(evidence[0].payload)


async def _run_memory_worker(agent) -> tuple[Any, ...]:
    worker = WorkerRuntime(
        kernel=agent.runtime.kernel,
        options=WorkerRuntimeOptions(
            worker_id="db.memory.learner",
            owner="db_runtime",
            queues=("memory_learning",),
        ),
    )
    run = await worker.run_once()
    assert run is not None
    assert run.error is None
    return tuple(await agent.runtime.store.list_evidence(run.handoff.operation_id))


async def _remember_board_revenue_contract(agent) -> Any:
    result = await agent.runtime.run(
        DbRequest(
            "Remember the board revenue metric definition",
            mode="memory.update",
            metadata={
                "kind": "metric_definition",
                "key": "metric:board_revenue",
                "text": (
                    "Board revenue SQL must return one aggregate: "
                    "SUM(orders.total) for orders whose status is complete minus "
                    "COALESCE(SUM(refunds.amount), 0). Join refunds on "
                    "refunds.order_id = orders.id."
                ),
                "schema_refs": _board_revenue_metric_metadata()["schema_refs"],
                "metadata": _board_revenue_metric_metadata(),
            },
        )
    )
    assert result.status is OperationStatus.SUCCEEDED
    snapshot = await agent.runtime.inspect_operation(result.operation_id)
    assert snapshot is not None
    proposal = _evidence(snapshot, "db.memory.proposal")
    assert proposal.payload["validation"]["diagnostics"]["semantic_contract"]["created"]
    public_proposal = _evidence(result, "db.memory.proposal")
    assert "validation" not in public_proposal.payload
    return result


def _rows(snapshot) -> list[dict[str, Any]]:
    """Return raw query rows from an operation snapshot."""
    return query_rows(snapshot)


# ---------------------------------------------------------------------------
# Core live tests
# ---------------------------------------------------------------------------


async def test_live_explicit_db_memory_changes_future_planning(tmp_path):
    db_path = tmp_path / "core-memory.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryCoreExplicit",
        memory=_memory_option(tmp_path, "core-explicit-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        source_identity = await _source_identity(agent)
        before = await _run_live(
            agent, "Calculate greenlight revenue from orders.total."
        )
        before_snapshot = await agent.runtime.inspect_operation(before.operation_id)
        await _write_memory(
            agent,
            kind="metric_definition",
            key="metric:greenlight_revenue",
            text=(
                "Greenlight revenue is the sum of orders.total only for orders "
                "whose status is complete."
            ),
            metadata={
                "aliases": ["greenlight revenue"],
                "schema_refs": ["orders.total", "orders.status"],
            },
            source_identity=source_identity,
        )
        after = await _run_live(
            agent, "Calculate greenlight revenue from orders.total."
        )
        after_snapshot = await agent.runtime.inspect_operation(after.operation_id)
    finally:
        await agent.stop()

    assert before_snapshot is not None
    assert after_snapshot is not None
    assert "metric:greenlight_revenue" not in _db_memory_keys(before_snapshot)
    assert after.status is OperationStatus.SUCCEEDED
    assert "metric:greenlight_revenue" in _db_memory_keys(after_snapshot)
    assert "complete" in _sql(after_snapshot).lower()
    assert _rows(after_snapshot)[0]


async def test_live_default_structured_memory_uses_no_embedder_or_vector_recall(
    tmp_path,
):
    db_path = tmp_path / "structured-no-embedder.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryNoEmbedder",
        memory=_memory_option(tmp_path, "structured-no-embedder-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        memory = agent.runtime.registry.get_plugin("memory")
        await _write_memory(
            agent,
            kind="business_rule",
            key="business_rule:recognized_revenue",
            text="Recognized revenue excludes orders that are not complete.",
            metadata={
                "aliases": ["recognized revenue"],
                "schema_refs": ["orders.status", "orders.total"],
            },
        )
        result = await _run_live(agent, "How should recognized revenue be calculated?")
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
        inspection = await agent.describe()
    finally:
        await agent.stop()

    memory_options = inspection.metadata["from_db_options"]["memory"]
    assert snapshot is not None
    recall = _evidence(snapshot, "memory.semantic.recall")
    assert memory._embedder is None
    assert getattr(memory.backend, "embedding_available") is False
    assert memory_options["retrieval_mode"] == "structured"
    assert memory_options["embedding_available"] is False
    assert memory_options["structured_index"] == "sqlite_fts5"
    assert recall.payload["diagnostics"]["retrieval_mode"] == "structured"
    assert recall.payload["diagnostics"]["embedding_available"] is False
    assert recall.payload["diagnostics"]["structured_candidate_count"] >= 1
    assert recall.payload["diagnostics"]["embedding_candidate_count"] == 0


async def test_live_source_and_activity_filters_apply_before_planning(tmp_path):
    db_path = tmp_path / "source-filter.sqlite"
    await _seed_revenue_db(db_path)
    shared_backend = _memory_backend(tmp_path, "shared-source-filter-memory")
    first = await Agent.from_db(
        str(db_path),
        name="LiveMemorySourceA",
        memory=DbMemoryConfig(backend=shared_backend, score_threshold=0.0),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )
    other_db = tmp_path / "source-filter-other.sqlite"
    await _seed_revenue_db(other_db)
    second = await Agent.from_db(
        str(other_db),
        name="LiveMemorySourceB",
        memory=DbMemoryConfig(backend=shared_backend, score_threshold=0.0),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        source_a = await _source_identity(first)
        source_b = await _source_identity(second)
        await _write_memory(
            first,
            kind="metric_definition",
            key="metric:source_a_only",
            text="Source A only revenue uses complete orders.",
            source_identity=source_a,
            metadata={
                "aliases": ["source a only revenue"],
                "schema_refs": ["orders.total"],
            },
        )
        await _write_memory(
            second,
            kind="metric_definition",
            key="metric:active_memory",
            text="Active memory revenue uses complete orders.",
            source_identity=source_b,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total"],
            },
        )
        await _write_memory(
            second,
            kind="metric_definition",
            key="metric:inactive_memory",
            text="Inactive memory revenue should never be used.",
            source_identity=source_b,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total"],
            },
            active=False,
        )
        await _write_memory(
            second,
            kind="metric_definition",
            key="metric:stale_memory",
            text="Stale memory revenue should never be used.",
            source_identity=source_b,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total"],
            },
            stale=True,
        )
        await _write_memory(
            second,
            kind="metric_definition",
            key="metric:expired_memory",
            text="Expired memory revenue should never be used.",
            source_identity=source_b,
            metadata={
                "aliases": ["active memory revenue"],
                "schema_refs": ["orders.total"],
            },
            expires_at="2000-01-01T00:00:00+00:00",
        )
        cross_source = await _recall_memory(
            second,
            "Calculate source a only revenue from orders.total.",
            source_identity=source_b,
        )
        active = await _recall_memory(
            second,
            "Calculate active memory revenue from orders.total.",
            source_identity=source_b,
        )
    finally:
        await first.stop()
        await second.stop()

    assert source_a != source_b
    assert "metric:source_a_only" not in [
        item["metadata"]["db_memory"]["key"] for item in cross_source["results"]
    ]
    assert [item["metadata"]["db_memory"]["key"] for item in active["results"]] == [
        "metric:active_memory"
    ]


async def test_live_alias_and_schema_ref_memory_ranks_above_broad_match(tmp_path):
    db_path = tmp_path / "ranking.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryRanking",
        memory=_memory_option(tmp_path, "ranking-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        await _write_memory(
            agent,
            kind="business_rule",
            key="business_rule:broad_revenue",
            text="Revenue appears in several dashboards and order reports.",
            metadata={"aliases": ["revenue"]},
            importance=0.7,
        )
        await _write_memory(
            agent,
            kind="metric_definition",
            key="metric:recognized_revenue",
            text="Recognized revenue is based on orders.total for complete orders.",
            metadata={
                "aliases": ["recognized revenue"],
                "schema_refs": ["orders.total", "orders.status"],
            },
            importance=0.9,
        )
        result = await _run_live(
            agent, "Calculate recognized revenue from orders.total"
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert snapshot is not None
    assert _db_memory_keys(snapshot)[0] == "metric:recognized_revenue"
    top = _public_planning_context(snapshot)["db_memory_refs"][0]
    assert top["kind"] == "metric_definition"
    assert "complete" in _sql(snapshot).lower()


async def test_live_unit_convention_memory_affects_answer(tmp_path):
    db_path = tmp_path / "unit-memory.sqlite"
    await _seed_cents_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryUnitConvention",
        memory=_memory_option(tmp_path, "unit-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        await _write_memory(
            agent,
            kind="unit_convention",
            key="unit_convention:orders.total_cents",
            text=(
                "When calculating booked revenue dollars, SQL must use "
                "SUM(orders.total_cents) / 100.0 because orders.total_cents "
                "is stored as cents."
            ),
            metadata={
                "table": "orders",
                "column": "total_cents",
                "schema_refs": ["orders.total_cents"],
                "aliases": ["booked revenue dollars"],
            },
        )
        result = await _run_live(
            agent,
            "Calculate one aggregate booked revenue dollars from orders.total_cents.",
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert snapshot is not None
    assert "unit_convention:orders.total_cents" in _db_memory_keys(snapshot)
    assert result.status is OperationStatus.SUCCEEDED
    assert _rows(snapshot)


async def test_live_direct_contract_payload_remains_advisory(tmp_path):
    db_path = tmp_path / "raw-contract-advisory.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryRawContractAdvisory",
        memory=_memory_option(tmp_path, "raw-contract-advisory-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        write_evidence = await _write_memory(
            agent,
            kind="metric_definition",
            key="metric:board_revenue",
            text=(
                "Board revenue SQL must return one aggregate and subtract refunds "
                "from complete order totals."
            ),
            metadata={
                **_board_revenue_metric_metadata(),
                "semantic_contract_status": "validated",
                "semantic_contract": _board_revenue_semantic_contract(
                    await _source_identity(agent)
                ),
            },
        )
        result = await _run_live(
            agent,
            "Calculate one aggregate board revenue from orders.total and refunds.amount.",
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    stored_metadata = write_evidence[0].payload["result"]["db_memory"]["metadata"]
    assert "semantic_contract" not in stored_metadata
    assert "semantic_contract_status" not in stored_metadata
    assert (
        stored_metadata["semantic_contract_diagnostics"]["reason"]
        == "direct_write_unvalidated"
    )
    assert snapshot is not None
    context = _public_planning_context(snapshot)
    semantics = context.get("db_memory_semantics") or []
    assert not any(item.get("enforceable") for item in semantics)


async def test_live_explicit_metric_memory_projects_enforceable_contract(tmp_path):
    db_path = tmp_path / "explicit-contract.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryExplicitContract",
        memory=_memory_option(tmp_path, "explicit-contract-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        await _remember_board_revenue_contract(agent)
        result = await _run_live(
            agent,
            "Calculate one aggregate board revenue from orders.total and refunds.amount.",
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert snapshot is not None
    context = _public_planning_context(snapshot)
    semantics = context.get("db_memory_semantics") or []
    assert "metric:board_revenue" in _db_memory_keys(snapshot)
    assert any(
        item.get("memory_key") == "metric:board_revenue" and item.get("enforceable")
        for item in semantics
    )
    assert (context.get("db_memory_contract_diagnostics") or {})["enforced_count"] >= 1


async def test_live_policy_blocked_contract_is_not_enforceable(tmp_path):
    db_path = tmp_path / "policy-blocked-contract.sqlite"
    await _seed_revenue_db(db_path)
    backend = _memory_backend(tmp_path, "policy-blocked-contract-memory")
    writer = await Agent.from_db(
        str(db_path),
        name="LiveMemoryPolicyBlockedWriter",
        memory=DbMemoryConfig(backend=backend, score_threshold=0.0),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )
    try:
        await _remember_board_revenue_contract(writer)
    finally:
        await writer.stop()

    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryPolicyBlockedContract",
        memory=DbMemoryConfig(backend=backend, score_threshold=0.0),
        source_options=DbSourceOptions(
            blocked_columns=("refunds.amount",), cache_ttl=0
        ),
        **_require_live_openai(),
    )

    try:
        result = await _run_live(
            agent,
            "Calculate one aggregate board revenue from orders.total and refunds.amount.",
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
        assert snapshot is not None
        assert_no_invalid_accepted_query_plans(snapshot.evidence)
        raw_context = _raw_planning_context(snapshot)
    finally:
        await agent.stop()

    public_context = _public_planning_context(result)
    public_dumped = str(public_context)
    assert public_context.get("redacted") is True
    assert "db_memory_refs" not in public_context
    assert "db_memory_semantics" not in public_context
    assert "db_memory_contract_diagnostics" not in public_context
    assert "refunds.amount" not in public_dumped
    assert "blocked_by_policy" not in public_dumped

    semantics = raw_context.get("db_memory_semantics") or []
    diagnostics = raw_context.get("db_memory_contract_diagnostics") or {}
    assert semantics
    assert not any(item.get("enforceable") for item in semantics)
    omitted = diagnostics.get("omitted_reasons") or {}
    assert omitted.get("blocked_by_policy") or omitted.get("schema_scope_mismatch")


async def test_live_explicit_unit_convention_projects_and_enforces_conversion(tmp_path):
    db_path = tmp_path / "explicit-unit-contract.sqlite"
    await _seed_cents_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryExplicitUnitContract",
        memory=_memory_option(tmp_path, "explicit-unit-contract-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        memory_result = await agent.runtime.run(
            DbRequest(
                "Remember orders total cents unit convention",
                mode="memory.update",
                metadata={
                    "kind": "unit_convention",
                    "key": "unit_convention:orders.total_cents",
                    "text": "orders.total_cents is stored as cents.",
                    "schema_refs": [
                        {"table": "orders", "column": "total_cents"},
                    ],
                    "metadata": {
                        "table": "orders",
                        "column": "total_cents",
                        "unit": "cents",
                        "aliases": ["booked revenue dollars"],
                    },
                },
            )
        )
        result = await _run_live(
            agent,
            "Calculate one aggregate SUM booked revenue dollars from orders.total_cents.",
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert memory_result.status is OperationStatus.SUCCEEDED
    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    context = _public_planning_context(snapshot)
    semantics = context.get("db_memory_semantics") or []
    assert any(
        item.get("memory_key") == "unit_convention:orders.total_cents"
        and item.get("enforceable")
        for item in semantics
    )
    assert "/ 100" in _sql(snapshot).lower() or "/100" in _sql(snapshot).lower()


# ---------------------------------------------------------------------------
# Learning / worker-oriented live tests
# ---------------------------------------------------------------------------


async def test_live_worker_promotes_unit_convention_from_accepted_evidence(tmp_path):
    db_path = tmp_path / "worker-unit.sqlite"
    await _seed_cents_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryWorkerUnit",
        memory=_memory_option(tmp_path, "worker-unit-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        source = await _run_live(agent, "How many orders are there?")
        worker_evidence = await _run_memory_worker(agent)
        memory = agent.runtime.registry.get_plugin("memory")
        records = await memory.backend.list_db_records(
            category="db_semantics",
            key="unit_convention:orders.total_cents",
            source_identity=await _source_identity(agent),
            limit=10,
        )
    finally:
        await agent.stop()

    assert source.status is OperationStatus.SUCCEEDED
    assert {item.kind for item in worker_evidence} >= {
        "db.memory.candidate",
        "db.memory.promotion",
        "memory.semantic.write",
    }
    assert records
    assert records[0]["metadata"]["db_memory"]["kind"] == "unit_convention"


async def test_live_catalog_cited_value_alias_write_path(tmp_path):
    db_path = tmp_path / "worker-alias.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryWorkerAlias",
        memory=_memory_option(tmp_path, "worker-alias-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        result = await _run_live(agent, "Show completed orders by status")
        catalog_evidence = _maybe_evidence(result, "catalog.source_registered")
        assert catalog_evidence is not None
        await _write_memory(
            agent,
            kind="value_alias",
            key="value_alias:orders.status:completed",
            text=(
                "When users say completed orders, consult the catalog profile "
                "for orders.status."
            ),
            metadata={
                "table": "orders",
                "column": "status",
                "alias": "completed orders",
                "catalog_profile_ref": "orders.status",
                "catalog_evidence_id": catalog_evidence.id,
                "schema_refs": ["orders.status"],
            },
        )
        memory = agent.runtime.registry.get_plugin("memory")
        records = await memory.backend.list_db_records(
            category="db_semantics",
            key="value_alias:orders.status:completed",
            source_identity=await _source_identity(agent),
            limit=20,
        )
        recalled = await _recall_memory(
            agent,
            "completed orders status orders.status",
            source_identity=await _source_identity(agent),
            kinds=["value_alias"],
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert records
    assert records[0]["metadata"]["db_memory"]["kind"] == "value_alias"
    assert [item["metadata"]["db_memory"]["key"] for item in recalled["results"]] == [
        "value_alias:orders.status:completed"
    ]
    assert "observed_values" not in str(
        [
            record["metadata"]["db_memory"]
            for record in records
            if record["metadata"]["db_memory"]["kind"] == "value_alias"
        ]
    )


async def test_live_pii_db_memory_candidate_is_rejected(tmp_path):
    db_path = tmp_path / "pii-rejection.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryPiiRejection",
        memory=_memory_option(tmp_path, "pii-rejection-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        evidence = await agent.runtime.execute_capability(
            "memory.semantic.write",
            owner="memory",
            operation_type="memory.update",
            input={
                "db_memory_payload": {
                    "kind": "business_rule",
                    "key": "business_rule:customer_email_ada@example.com",
                    "text": "Ada customer email is ada@example.com.",
                    "metadata": {
                        "source_identity": await _source_identity(agent),
                        "workspace_scope": "source",
                    },
                },
                "db_memory_prompt": "Ada customer email is ada@example.com.",
            },
        )
        memory = agent.runtime.registry.get_plugin("memory")
        records = await memory.backend.list_db_records(
            category="db_semantics", limit=10
        )
    finally:
        await agent.stop()

    assert evidence[0].kind == "memory.semantic.write"
    assert evidence[0].payload["success"] is False
    assert (
        "PII" in evidence[0].payload["error"]
        or "row-level" in evidence[0].payload["error"]
    )
    assert records == []


# ---------------------------------------------------------------------------
# End-to-end golden live tests
# ---------------------------------------------------------------------------


async def test_live_golden_memory_improves_metric_answer(tmp_path):
    db_path = tmp_path / "golden-improves.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryGoldenImproves",
        memory=_memory_option(tmp_path, "golden-improves-memory"),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        before = await _run_live(
            agent,
            "Calculate board revenue from orders.total and refunds.amount.",
        )
        before_snapshot = await agent.runtime.inspect_operation(before.operation_id)
        await _remember_board_revenue_contract(agent)
        after = await _run_live(
            agent,
            "Calculate one aggregate board revenue from orders.total and refunds.amount.",
        )
        after_snapshot = await agent.runtime.inspect_operation(after.operation_id)
    finally:
        await agent.stop()

    assert before_snapshot is not None
    assert after_snapshot is not None
    assert "metric:board_revenue" not in _db_memory_keys(before_snapshot)
    assert after.status is OperationStatus.SUCCEEDED
    assert "metric:board_revenue" in _db_memory_keys(after_snapshot)
    sql = _sql(after_snapshot).lower()
    assert "refund" in sql
    assert "sum" in sql
    assert "complete" in sql
    assert _rows(after_snapshot)


async def test_live_golden_context_budget_under_memory_load(tmp_path):
    db_path = tmp_path / "golden-budget.sqlite"
    await _seed_revenue_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveMemoryGoldenBudget",
        memory=_memory_option(
            tmp_path,
            "golden-budget-memory",
            limit=3,
            char_budget=280,
        ),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        source_identity = await _source_identity(agent)
        for idx in range(30):
            await _write_memory(
                agent,
                kind="business_rule",
                key=f"business_rule:revenue_load_{idx:02d}",
                text=(
                    f"Revenue load rule {idx:02d} mentions orders.total and "
                    "complete orders for stress testing bounded memory context."
                ),
                source_identity=source_identity,
                metadata={
                    "aliases": ["revenue load"],
                    "schema_refs": ["orders.total", "orders.status"],
                },
                importance=0.5 + min(idx, 10) / 20,
            )
        result = await _run_live(agent, "How should revenue load be calculated?")
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
    finally:
        await agent.stop()

    assert snapshot is not None
    context = _raw_planning_context(snapshot)
    rendered = str(context.get("rendered_context") or "")
    diagnostics = context.get("db_memory_diagnostics") or {}
    assert len(context.get("db_memory_refs") or []) <= 3
    assert diagnostics["used_chars"] <= 280
    assert diagnostics["included_count"] <= 3
    assert diagnostics["candidate_count"] >= 3
    assert "Database memory:" in rendered

    public_context = _public_planning_context(result)
    public_diagnostics = public_context.get("db_memory_diagnostics") or {}
    assert "rendered_context" not in public_context
    assert "used_chars" not in public_diagnostics
    assert "included_count" not in public_diagnostics


async def test_live_golden_structured_memory_persists_after_restart(tmp_path):
    db_path = tmp_path / "golden-persist.sqlite"
    await _seed_revenue_db(db_path)
    workspace = "golden-persist-memory"
    memory_dir = tmp_path / "memory"
    first = await Agent.from_db(
        str(db_path),
        name="LiveMemoryGoldenPersistFirst",
        memory=DbMemoryConfig(
            backend=LocalMemoryBackend(
                workspace=workspace,
                agent_id=workspace,
                scope="project",
                base_dir=memory_dir,
                embedder=None,
            ),
            score_threshold=0.0,
        ),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        await _write_memory(
            first,
            kind="business_rule",
            key="business_rule:persistent_revenue",
            text="Persistent revenue uses complete orders only.",
            metadata={
                "aliases": ["persistent revenue"],
                "schema_refs": ["orders.total", "orders.status"],
            },
        )
    finally:
        await first.stop()

    second = await Agent.from_db(
        str(db_path),
        name="LiveMemoryGoldenPersistSecond",
        memory=DbMemoryConfig(
            backend=LocalMemoryBackend(
                workspace=workspace,
                agent_id=workspace,
                scope="project",
                base_dir=memory_dir,
                embedder=None,
            ),
            score_threshold=0.0,
        ),
        source_options=DbSourceOptions(cache_ttl=0),
        **_require_live_openai(),
    )

    try:
        result = await _run_live(second, "How should persistent revenue be calculated?")
        snapshot = await second.runtime.inspect_operation(result.operation_id)
    finally:
        await second.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    assert "business_rule:persistent_revenue" in _db_memory_keys(snapshot)
    assert "complete" in _sql(snapshot).lower()
