import json

from daita.agents.agent import Agent
from daita.db import DbRuntime
from daita.db.llm_service import DbLLMResponse
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus


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
            total REAL NOT NULL
        );
        INSERT INTO customers (id, name) VALUES (1, 'Ada'), (2, 'Linus');
        INSERT INTO orders (customer_id, total)
        VALUES (1, 10.0), (1, 30.0), (2, 20.0);
        """)
    runtime = DbRuntime(
        plugins=(CatalogPlugin(auto_persist=False), sqlite),
        db_llm_service=FakeDbLLMService(responses),
    )
    await runtime.setup()
    return runtime, sqlite


def _plan(sql, *, tables=("orders", "customers"), confidence=0.92):
    return json.dumps(
        {
            "operation": "read",
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
