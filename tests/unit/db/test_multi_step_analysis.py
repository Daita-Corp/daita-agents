import json
from pathlib import Path
from uuid import uuid4

from daita.agents.agent import Agent
from daita.db import DbLLMConfig, DbRequest, DbRuntime, DbRuntimeConfig
from daita.db.analysis import DbAnalysisPlan, analysis_metadata
from daita.db.llm_service import DbLLMResponse
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import Evidence, GovernanceResult, OperationStatus, TaskStatus


class FakeAnalysisLLMService:
    def __init__(self, *, planner_responses, synthesis_response=None):
        self.planner_responses = list(planner_responses)
        self.synthesis_response = synthesis_response
        self.calls = []
        self.synthesis_calls = []
        self.safe_metadata = {"provider": "fake", "model": "analysis-test"}

    @property
    def available(self):
        return True

    async def generate_json(self, messages):
        self.calls.append(messages)
        content = self.planner_responses.pop(0)
        if callable(content):
            content = content(messages)
        return DbLLMResponse(
            content=content,
            diagnostics={
                "provider": "fake",
                "model": "analysis-test",
                "tokens": {"total_tokens": 17},
                "latency_ms": 1.0,
            },
        )

    async def generate_synthesis_json(self, messages):
        self.synthesis_calls.append(messages)
        if self.synthesis_response is None:
            content = _analysis_synthesis_response(messages)
        else:
            content = self.synthesis_response(messages)
        return DbLLMResponse(
            content=content,
            diagnostics={
                "provider": "fake",
                "model": "analysis-test",
                "tokens": {"total_tokens": 19},
                "latency_ms": 1.0,
            },
        )


async def _runtime(tmp_path, llm_service, *, config=None):
    db_path = tmp_path / f"multi_step_{uuid4().hex}.sqlite"
    sqlite = SQLitePlugin(path=str(db_path), query_default_limit=20)
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            month TEXT NOT NULL,
            region TEXT NOT NULL,
            total REAL NOT NULL
        );
        INSERT INTO orders (id, month, region, total)
        VALUES
            (1, 'February', 'East', 100.0),
            (2, 'March', 'East', 40.0),
            (3, 'March', 'West', 30.0),
            (4, 'April', 'East', 90.0);
        """)
    runtime = DbRuntime(
        config=config,
        plugins=(CatalogPlugin(auto_persist=False), sqlite),
        db_llm_service=llm_service,
    )
    await runtime.setup()
    return runtime, sqlite


def _analysis_plan(*, steps=None, budgets=None):
    return json.dumps(
        {
            "analysis_id": "analysis-test-1",
            "goal": "Investigate revenue changes",
            "steps": steps
            or [
                {
                    "id": "step_1",
                    "kind": "query",
                    "purpose": "Calculate revenue by month",
                    "depends_on": [],
                    "input_refs": [],
                    "expected_evidence": ["query.result"],
                    "budgets": {"max_rows": 200, "max_repairs": 1},
                },
                {
                    "id": "step_2",
                    "kind": "query",
                    "purpose": "Break March revenue down by region",
                    "depends_on": ["step_1"],
                    "input_refs": [],
                    "expected_evidence": ["query.result"],
                    "budgets": {"max_rows": 200, "max_repairs": 1},
                },
                {
                    "id": "step_3",
                    "kind": "synthesis",
                    "purpose": "Summarize likely drivers",
                    "depends_on": ["step_1", "step_2"],
                    "input_refs": [],
                    "expected_evidence": ["analysis.synthesis"],
                    "budgets": {"max_context_chars": 12000},
                },
            ],
            "budgets": budgets
            or {
                "max_steps": 6,
                "max_query_steps": 3,
                "max_checkpoint_steps": 3,
                "max_repairs": 1,
                "max_total_rows": 1000,
                "max_llm_calls": 6,
                "max_context_chars": 16000,
                "max_duration_seconds": 120,
            },
            "diagnostics": {"mode": "llm", "model": "analysis-test"},
        }
    )


def _query_plan(sql, *, tables=("orders",), columns=None):
    return json.dumps(
        {
            "operation": "read",
            "selected_sql": sql,
            "candidates": [
                {
                    "sql": sql,
                    "purpose": "analysis query",
                    "confidence": 0.9,
                    "tables": list(tables),
                    "columns": columns or [],
                    "expected_columns": columns or [],
                    "assumptions": [],
                    "risk_notes": [],
                }
            ],
            "selected_tables": list(tables),
            "joins": [],
            "filters": [],
            "aggregations": [{"function": "sum", "column": "orders.total"}],
            "group_by": [],
            "order_by": [],
            "limit": 10,
            "assumptions": [],
            "clarification_question": None,
            "confidence": 0.9,
            "planner": "llm",
        }
    )


def _analysis_synthesis_response(messages):
    context = json.loads(messages[-1]["content"])
    evidence = context["evidence"]
    citations = [
        {
            "id": item["id"],
            "kind": item["kind"],
            "purpose": "accepted analysis evidence",
        }
        for item in evidence
        if item["kind"]
        in {"query.result", "analysis.checkpoint", "verification.result"}
    ]
    return json.dumps(
        {
            "answer": "March revenue dropped, with the East region contributing most of the decline.",
            "reasoning_summary": "Compared monthly totals and March regional totals.",
            "cited_evidence_refs": citations,
            "assumptions": [],
            "limitations": [],
            "warnings": [],
            "sufficiency": "answered",
            "confidence": 0.87,
        }
    )


def test_phase3_analysis_materialization_has_no_executor_dependency():
    source = Path("daita/db/runtime/analysis/materialization.py").read_text()

    assert "DbOperationExecutor" not in source
    assert "_execute_sql_validation" not in source
    assert "_execute_validated_read" not in source


async def test_from_db_model_config_registers_analysis_capabilities(tmp_path):
    db_path = tmp_path / "from_db_analysis.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    await sqlite.disconnect()

    agent = await Agent.from_db(
        str(db_path), llm=DbLLMConfig(provider="mock", model="mock-model")
    )
    try:
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert {
        "db_runtime:db.analysis.plan",
        "db_runtime:db.analysis.plan.validate",
        "db_runtime:db.analysis.checkpoint",
        "db_runtime:db.analysis.summarize",
        "db_runtime:db.analysis.replan",
    } <= set(inspection.capability_ids)
    assert {
        "db_runtime:analysis.plan",
        "db_runtime:analysis.plan.validation",
        "db_runtime:analysis.checkpoint",
        "db_runtime:analysis.synthesis",
        "db_runtime:analysis.plan.revision",
    } <= set(inspection.evidence_schema_kinds)


def test_ready_step_selector_is_serial_by_default_and_dependency_aware():
    runtime = DbRuntime()
    plan = DbAnalysisPlan.from_mapping(
        json.loads(
            _analysis_plan(
                steps=[
                    {
                        "id": "step_1",
                        "kind": "query",
                        "purpose": "A",
                        "depends_on": [],
                    },
                    {
                        "id": "step_2",
                        "kind": "query",
                        "purpose": "B",
                        "depends_on": [],
                    },
                    {
                        "id": "step_3",
                        "kind": "synthesis",
                        "purpose": "C",
                        "depends_on": ["step_1", "step_2"],
                    },
                ]
            )
        )
    )

    ready_serial = runtime._select_ready_analysis_steps(plan, {}, serial=True)
    ready_set = runtime._select_ready_analysis_steps(plan, {}, serial=False)

    assert [step.id for step in ready_serial] == ["step_1"]
    assert [step.id for step in ready_set] == ["step_1", "step_2"]

    completed_by_step = {
        "step_1": (
            Evidence(
                id="evidence-step-1",
                kind="query.result",
                owner="sqlite",
                operation_id="operation-1",
                accepted=True,
                payload={},
                metadata={"analysis_step_id": "step_1"},
            ),
        )
    }
    ready_after_one = runtime._select_ready_analysis_steps(
        plan,
        completed_by_step,
        serial=False,
    )

    assert [step.id for step in ready_after_one] == ["step_2"]
