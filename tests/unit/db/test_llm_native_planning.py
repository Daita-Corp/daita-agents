import json

from daita.agents.agent import Agent
from daita.db import DbRuntime
from daita.db.llm_service import DbLLMResponse
from daita.db.query_plan import DbQueryPlan as StructuredDbQueryPlan
from daita.db.query_planning import DbQueryPlan as DeterministicDbQueryPlan
from daita.db.query_planning import DbQueryPlanner
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import Evidence, OperationStatus


class FakeDbLLMService:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.safe_metadata = {"provider": "fake", "model": "planner-test"}

    @property
    def available(self):
        return True

    async def generate_json(self, messages):
        self.calls.append(messages)
        content = self.responses.pop(0)
        return DbLLMResponse(
            content=content,
            diagnostics={
                "provider": "fake",
                "model": "planner-test",
                "tokens": {"total_tokens": 42},
                "estimated_cost_usd": 0.001,
                "latency_ms": 1.5,
            },
        )


async def _runtime_with_llm(tmp_path, responses):
    db_path = tmp_path / "llm_planning.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total REAL NOT NULL,
            status TEXT NOT NULL
        );
        INSERT INTO customers (id, name) VALUES (1, 'Ada'), (2, 'Linus');
        INSERT INTO orders (customer_id, total, status)
        VALUES
            (1, 10.0, 'complete'),
            (1, 30.0, 'complete'),
            (2, 20.0, 'pending');
        """)
    runtime = DbRuntime(
        plugins=(CatalogPlugin(auto_persist=False), sqlite),
        db_llm_service=FakeDbLLMService(responses),
    )
    await runtime.setup()
    return runtime, sqlite


def _plan(sql, *, tables=("orders", "customers"), confidence=0.92, operation="read"):
    return json.dumps(
        {
            "operation": operation,
            "selected_sql": sql,
            "candidates": [
                {
                    "sql": sql,
                    "purpose": "answer analytical question",
                    "confidence": confidence,
                    "tables": list(tables),
                    "columns": [],
                    "expected_columns": ["name", "total"],
                    "assumptions": [],
                    "risk_notes": [],
                }
            ],
            "selected_tables": list(tables),
            "joins": [],
            "filters": [],
            "aggregations": [{"function": "sum", "column": "orders.total"}],
            "group_by": ["customers.name"],
            "order_by": ["total desc"],
            "limit": 5,
            "assumptions": [],
            "clarification_question": None,
            "confidence": confidence,
            "planner": "llm",
        }
    )


def _join_plan(*, operation="read"):
    sql = (
        "SELECT o.id, o.customer_id, o.total, c.id AS customer_id, c.name "
        "FROM orders o JOIN customers c ON o.customer_id = c.id"
    )
    return json.dumps(
        {
            "operation": operation,
            "selected_sql": sql,
            "candidates": [
                {
                    "sql": sql,
                    "purpose": "join orders to customers",
                    "confidence": 0.9,
                    "tables": ["orders", "customers"],
                }
            ],
            "selected_tables": ["orders", "customers"],
            "joins": [
                {
                    "left_table": "orders",
                    "left_column": "customer_id",
                    "right_table": "customers",
                    "right_column": "id",
                    "type": "INNER",
                    "relationship": "orders.customer_id -> customers.id",
                }
            ],
            "filters": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": None,
            "assumptions": [],
            "clarification_question": None,
            "confidence": 0.9,
            "planner": "llm",
        }
    )


def _status_plan(sql, *, value, confidence=0.9, operator="="):
    return json.dumps(
        {
            "operation": "read",
            "selected_sql": sql,
            "candidates": [
                {
                    "sql": sql,
                    "purpose": "filter orders by status",
                    "confidence": confidence,
                    "tables": ["orders"],
                    "columns": ["orders.status"],
                }
            ],
            "selected_tables": ["orders"],
            "joins": [],
            "filters": [
                {
                    "column": "orders.status",
                    "operator": operator,
                    "value": value,
                }
            ],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": 10,
            "assumptions": [],
            "clarification_question": None,
            "confidence": confidence,
            "planner": "llm",
        }
    )


async def test_from_db_model_config_registers_llm_planning_capability(tmp_path):
    db_path = tmp_path / "from_db_llm.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script("CREATE TABLE customers (id INTEGER PRIMARY KEY)")
    await sqlite.disconnect()

    agent = await Agent.from_db(str(db_path), model="mock-model", llm_provider="mock")
    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert "db_runtime:db.query.plan" in inspection.capability_ids
    assert "db_runtime:query.plan.proposal" in inspection.evidence_schema_kinds
    assert inspection.metadata["from_db_options"]["model"] == "mock-model"
    assert "api_key" not in inspection.metadata["from_db_options"]


async def test_analytical_prompt_routes_to_llm_planner_and_executes_validated_read(
    tmp_path,
):
    sql = (
        "SELECT c.name, SUM(o.total) AS total "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "GROUP BY c.name ORDER BY total DESC LIMIT 5"
    )
    runtime, sqlite = await _runtime_with_llm(tmp_path, [_plan(sql)])
    try:
        result = await runtime.run("Show top customers by total revenue")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.diagnostics["execution"]["planner_strategy"] == "llm"
    assert {"planning.context", "query.plan.proposal", "query.plan.validation"} <= {
        item.kind for item in result.evidence
    }
    proposal = next(
        item for item in result.evidence if item.kind == "query.plan.proposal"
    )
    assert proposal.payload["planner_diagnostics"]["model"] == "planner-test"
    assert proposal.payload["planner_diagnostics"]["tokens"]["total_tokens"] == 42
    assert snapshot is not None
    read_task = next(
        task for task in snapshot.tasks if task.capability_id == "db.sql.execute_read"
    )
    assert read_task.input["sql_ref"] == "sql.validation"
    assert "validated_evidence_id" in read_task.input
    dependency = next(
        dep for dep in read_task.dependencies if dep.evidence_kind == "sql.validation"
    )
    assert dependency.evidence_id == read_task.input["validated_evidence_id"]
    assert dependency.payload_fingerprint


async def test_catalog_relationship_evidence_reaches_llm_planning_context(tmp_path):
    runtime, sqlite = await _runtime_with_llm(
        tmp_path, [_join_plan(operation="query_planning")]
    )
    try:
        result = await runtime.run("Join orders to customers using their relationship")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.answer == "Returned 3 rows."
    assert snapshot is not None
    planning_context = [
        item for item in snapshot.evidence if item.kind == "planning.context"
    ][-1]
    assert planning_context.payload["relationship_evidence_refs"]
    proposal = next(
        item for item in snapshot.evidence if item.kind == "query.plan.proposal"
    )
    assert proposal.payload["structured_plan"]["operation"] == "read"
    assert "schema.relationship_path" in {item.kind for item in snapshot.evidence}


async def test_llm_validation_failure_creates_repair_with_fresh_tasks(tmp_path):
    bad_sql = "SELECT SUM(total) AS total FROM payments LIMIT 5"
    good_sql = (
        "SELECT c.name, SUM(o.total) AS total "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "GROUP BY c.name ORDER BY total DESC LIMIT 5"
    )
    runtime, sqlite = await _runtime_with_llm(
        tmp_path,
        [
            _plan(bad_sql, tables=("payments",), confidence=0.8),
            _plan(good_sql, confidence=0.7),
        ],
    )
    try:
        result = await runtime.run("Show top customers by total revenue")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    validation_tasks = [
        task
        for task in snapshot.tasks
        if task.capability_id == "db.query.plan.validate"
    ]
    assert len(validation_tasks) == 2
    assert len({task.id for task in validation_tasks}) == 2
    assert any(item.kind == "query.plan.repair" for item in snapshot.evidence)
    failed_validations = [
        item
        for item in snapshot.evidence
        if item.kind == "query.plan.validation" and not item.accepted
    ]
    assert failed_validations
    repair_task = next(
        task for task in snapshot.tasks if task.capability_id == "db.query.repair"
    )
    failure_dependency = next(
        dep
        for dep in repair_task.dependencies
        if dep.evidence_kind == "query.plan.validation"
        and dep.evidence_id == failed_validations[0].id
    )
    assert failure_dependency.evidence_accepted is False


async def test_invalid_repair_uses_deterministic_fallback_before_read(tmp_path):
    bad_sql = "SELECT id, status FROM orders WHERE status = 'completed' LIMIT 10"
    runtime, sqlite = await _runtime_with_llm(
        tmp_path,
        [
            _status_plan(bad_sql, value="completed", confidence=0.82),
            "not json",
        ],
    )
    try:
        result = await runtime.run("Show completed orders by status")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    capability_sequence = [task.capability_id for task in snapshot.tasks]
    assert capability_sequence.index(
        "db.query.plan.validate"
    ) < capability_sequence.index("db.query.repair")
    assert "db.sql.validate" in capability_sequence
    assert "db.sql.execute_read" in capability_sequence
    fallback_proposal = next(
        item
        for item in snapshot.evidence
        if item.kind == "query.plan.proposal"
        and item.metadata.get("repair_fallback") is True
    )
    fallback_validation = next(
        item
        for item in snapshot.evidence
        if item.kind == "query.plan.validation"
        and item.payload.get("plan_evidence_id") == fallback_proposal.id
    )
    assert fallback_validation.payload["accepted_sql"]
    query_result = next(item for item in result.evidence if item.kind == "query.result")
    assert query_result.payload["rows"] == [
        {"status": "complete"},
        {"status": "complete"},
    ]


async def test_repair_exhaustion_records_terminal_diagnostics_without_sql_execution(
    tmp_path,
    monkeypatch,
):
    def invalid_plan(self, request, intent, operation, schema, **kwargs):
        structured = StructuredDbQueryPlan.deterministic(
            sql=None,
            tables=(),
            confidence=0.0,
            strategy="forced_invalid_fallback",
        )
        return DeterministicDbQueryPlan(
            sql=None,
            evidence=Evidence(
                kind="query.plan.proposal",
                owner="db_runtime",
                operation_id=operation.id,
                payload={
                    "sql": None,
                    "structured_plan": structured.to_dict(),
                    "strategy": "forced_invalid_fallback",
                    "valid": False,
                    "warnings": ["forced_invalid_fallback"],
                    "plan_fingerprint": "forced-invalid",
                    "sql_fingerprint": None,
                },
            ),
            diagnostics={
                "planner": "deterministic",
                "strategy": "forced_invalid_fallback",
                "sql": None,
                "schema_table_count": len(schema.get("tables", []) or []),
            },
            warnings=("forced_invalid_fallback",),
        )

    monkeypatch.setattr(DbQueryPlanner, "plan_read_query", invalid_plan)
    runtime, sqlite = await _runtime_with_llm(
        tmp_path,
        [
            _plan(
                "SELECT SUM(total) AS total FROM payments LIMIT 5",
                tables=("payments",),
                confidence=0.8,
            ),
            "not json",
        ],
    )
    try:
        result = await runtime.run("Show top customers by total revenue")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.FAILED
    assert snapshot is not None
    assert not any(item.kind == "sql.validation" for item in snapshot.evidence)
    assert not any(item.kind == "query.result" for item in snapshot.evidence)
    terminal = next(
        item
        for item in snapshot.evidence
        if item.kind == "query.plan.validation"
        and item.payload.get("failure") == "repair_exhausted"
    )
    assert terminal.payload["repair_exhausted"] is True
    assert result.diagnostics["execution"]["repair_exhausted"] is True
    assert result.diagnostics["execution"]["accepted_sql_missing"] is True


async def test_value_profile_validation_drives_repair_to_observed_literal(tmp_path):
    bad_sql = "SELECT id, status FROM orders WHERE status = 'completed' LIMIT 10"
    good_sql = "SELECT id, status FROM orders WHERE status = 'complete' LIMIT 10"
    runtime, sqlite = await _runtime_with_llm(
        tmp_path,
        [
            _status_plan(bad_sql, value="completed", confidence=0.82),
            _status_plan(good_sql, value="complete", confidence=0.76),
        ],
    )
    try:
        result = await runtime.run("Show completed orders by status")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    planning_context = [
        item for item in snapshot.evidence if item.kind == "planning.context"
    ][-1]
    assert (
        "orders.status: complete (2), pending (1)"
        in planning_context.payload["rendered_context"]
    )
    failed_validation = next(
        item
        for item in snapshot.evidence
        if item.kind == "query.plan.validation" and not item.accepted
    )
    assert any(
        error.startswith("unobserved_filter_literal:orders.status=completed")
        for error in failed_validation.payload["errors"]
    )
    assert any(item.kind == "query.plan.repair" for item in snapshot.evidence)
    query_result = next(item for item in result.evidence if item.kind == "query.result")
    assert query_result.payload["rows"] == [
        {"id": 1, "status": "complete"},
        {"id": 2, "status": "complete"},
    ]


async def test_predicate_profile_fills_missing_value_hint_before_repair(tmp_path):
    bad_sql = "SELECT id, status FROM orders WHERE status = 'completed' LIMIT 10"
    good_sql = "SELECT id, status FROM orders WHERE status = 'complete' LIMIT 10"
    runtime, sqlite = await _runtime_with_llm(
        tmp_path,
        [
            _status_plan(bad_sql, value="completed", confidence=0.82),
            _status_plan(good_sql, value="complete", confidence=0.76),
        ],
    )
    try:
        result = await runtime.run("Show completed rows by status")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    predicate_profile_task = next(
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
        and task.metadata.get("reason") == "column_value_predicate_profile"
    )
    assert predicate_profile_task.input["table"] == "orders"
    assert predicate_profile_task.input["column"] == "status"
    planning_contexts = [
        item for item in snapshot.evidence if item.kind == "planning.context"
    ]
    assert len(planning_contexts) >= 2
    assert planning_contexts[-1].payload["diagnostics"]["column_value_hint_count"] >= 1
    assert (
        "orders.status: complete (2), pending (1)"
        in planning_contexts[-1].payload["rendered_context"]
    )
    failed_validation = next(
        item
        for item in snapshot.evidence
        if item.kind == "query.plan.validation" and not item.accepted
    )
    assert any(
        error.startswith("unobserved_filter_literal:orders.status=completed")
        for error in failed_validation.payload["errors"]
    )
    query_result = next(item for item in result.evidence if item.kind == "query.result")
    assert query_result.payload["rows"] == [
        {"id": 1, "status": "complete"},
        {"id": 2, "status": "complete"},
    ]


async def test_zero_row_backstop_repairs_text_predicate_once(tmp_path):
    bad_sql = "SELECT id, status FROM orders WHERE status LIKE 'completed%' LIMIT 10"
    good_sql = "SELECT id, status FROM orders WHERE status = 'complete' LIMIT 10"
    runtime, sqlite = await _runtime_with_llm(
        tmp_path,
        [
            _status_plan(
                bad_sql,
                value="completed%",
                confidence=0.82,
                operator="like",
            ),
            _status_plan(good_sql, value="complete", confidence=0.76),
        ],
    )
    try:
        result = await runtime.run("Show completed rows by status")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.diagnostics["execution"]["planned_sql"] == good_sql
    assert snapshot is not None
    diagnosis = next(
        item for item in snapshot.evidence if item.kind == "query.zero_row_diagnosis"
    )
    assert diagnosis.accepted is False
    assert diagnosis.payload["failure"] == "zero_row_result"
    assert diagnosis.payload["predicates"][0]["operator"] == "like"
    assert diagnosis.payload["column_value_hints"][0]["observed_values"][0] == {
        "value": "complete",
        "count": 2,
    }
    repair = next(
        item for item in snapshot.evidence if item.kind == "query.plan.repair"
    )
    assert repair.payload["failure_evidence_id"] == diagnosis.id
    query_results = [item for item in result.evidence if item.kind == "query.result"]
    assert len(query_results) == 1
    assert query_results[0].payload["sql"] == good_sql
    assert query_results[0].payload["rows"] == [
        {"id": 1, "status": "complete"},
        {"id": 2, "status": "complete"},
    ]


async def test_invalid_llm_proposal_artifact_can_reach_repair(tmp_path):
    good_sql = (
        "SELECT c.name, SUM(o.total) AS total "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "GROUP BY c.name ORDER BY total DESC LIMIT 5"
    )
    runtime, sqlite = await _runtime_with_llm(
        tmp_path,
        [
            "not json",
            _plan(good_sql, confidence=0.7),
        ],
    )
    try:
        result = await runtime.run("Show top customers by total revenue")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    failed_proposals = [
        item
        for item in snapshot.evidence
        if item.kind == "query.plan.proposal"
        and item.payload.get("failure") == "planner_json_invalid"
    ]
    assert failed_proposals
    assert failed_proposals[0].accepted is True
    repair_task = next(
        task for task in snapshot.tasks if task.capability_id == "db.query.repair"
    )
    proposal_dependency = next(
        dep
        for dep in repair_task.dependencies
        if dep.evidence_kind == "query.plan.proposal"
        and dep.evidence_id == failed_proposals[0].id
    )
    assert proposal_dependency.evidence_accepted is True
