import json
from uuid import uuid4

from daita.agents.agent import Agent
from daita.db import DbRequest, DbRuntime, DbRuntimeConfig
from daita.db.analysis import DbAnalysisPlan, analysis_metadata, stable_fingerprint
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


async def test_from_db_model_config_registers_analysis_capabilities(tmp_path):
    db_path = tmp_path / "from_db_analysis.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    await sqlite.disconnect()

    agent = await Agent.from_db(str(db_path), model="mock-model", llm_provider="mock")
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


async def test_multi_step_plan_validation_precedes_query_steps(tmp_path):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month ORDER BY month LIMIT 10",
                columns=["month", "revenue"],
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders WHERE month = 'March' GROUP BY region ORDER BY revenue DESC LIMIT 10",
                columns=["region", "revenue"],
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate why revenue changed using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    kinds = [item.kind for item in snapshot.evidence]
    assert kinds.index("analysis.plan") < kinds.index("analysis.plan.validation")
    assert kinds.index("analysis.plan.validation") < kinds.index("query.plan.proposal")
    assert (
        result.answer
        == next(
            item for item in snapshot.evidence if item.kind == "analysis.synthesis"
        ).payload["answer"]
    )


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


async def test_invalid_dag_rejected_before_query_materialization(tmp_path):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(
                steps=[
                    {
                        "id": "step_1",
                        "kind": "query",
                        "purpose": "A",
                        "depends_on": ["step_2"],
                    },
                    {
                        "id": "step_2",
                        "kind": "python",
                        "purpose": "B",
                        "depends_on": ["step_1"],
                    },
                ],
                budgets={
                    "max_steps": 1,
                    "max_query_steps": 1,
                    "max_checkpoint_steps": 0,
                    "max_repairs": 1,
                    "max_total_rows": 1000,
                    "max_llm_calls": 6,
                    "max_context_chars": 16000,
                    "max_duration_seconds": 120,
                },
            )
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.BLOCKED
    validation = next(
        item for item in snapshot.evidence if item.kind == "analysis.plan.validation"
    )
    assert validation.accepted is False
    assert any(
        "unsupported_step_kind" in error for error in validation.payload["errors"]
    )
    assert any("cycle_detected" in error for error in validation.payload["errors"])
    assert "budget_max_steps_exceeded" in validation.payload["errors"]
    assert not any(
        task.capability_id == "db.sql.execute_read" for task in snapshot.tasks
    )


async def test_capability_step_wrong_owner_is_rejected_before_materialization(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(
                steps=[
                    {
                        "id": "catalog_1",
                        "kind": "catalog_search",
                        "purpose": "Search catalog assets",
                        "depends_on": [],
                        "capability_id": "catalog.schema.search",
                        "capability_owner": "db_runtime",
                        "input": {"store_id": "missing", "query": "orders"},
                        "expected_evidence": ["schema.search_result"],
                    }
                ]
            )
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.BLOCKED
    validation = next(
        item for item in snapshot.evidence if item.kind == "analysis.plan.validation"
    )
    assert validation.accepted is False
    assert any(
        "unsupported_step_capability_owner" in error
        for error in validation.payload["errors"]
    )
    assert not any(
        task.capability_id == "catalog.schema.search" for task in snapshot.tasks
    )


async def test_capability_backed_catalog_step_executes_through_runtime_task(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(
                steps=[
                    {
                        "id": "catalog_1",
                        "kind": "catalog_search",
                        "purpose": "Search catalog assets",
                        "depends_on": [],
                        "capability_id": "catalog.schema.search",
                        "capability_owner": "catalog",
                        "input": {"store_id": "missing", "query": "orders"},
                        "expected_evidence": ["schema.search_result"],
                    },
                    {
                        "id": "step_2",
                        "kind": "synthesis",
                        "purpose": "Summarize catalog search",
                        "depends_on": ["catalog_1"],
                        "expected_evidence": ["analysis.synthesis"],
                    },
                ]
            )
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    catalog_task = next(
        task for task in snapshot.tasks if task.capability_id == "catalog.schema.search"
    )
    assert catalog_task.status is TaskStatus.SUCCEEDED
    assert catalog_task.metadata["analysis_step_id"] == "catalog_1"
    search = next(
        item for item in snapshot.evidence if item.kind == "schema.search_result"
    )
    assert search.task_id == catalog_task.id
    synthesis_task = next(
        task for task in snapshot.tasks if task.capability_id == "db.analysis.summarize"
    )
    assert search.id in {
        dependency.evidence_id for dependency in synthesis_task.dependencies
    }


async def test_planning_context_records_source_refs_and_capability_summaries(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders WHERE month = 'March' GROUP BY region LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    context = next(
        item for item in snapshot.evidence if item.kind == "planning.context"
    )
    payload = context.payload
    assert "schema" in payload["included_sections"]
    assert "capabilities" in payload["included_sections"]
    assert payload["source_evidence_refs"]
    assert payload["source_fingerprints"]
    assert payload["budget_usage"]["rendered_context_chars"] > 0
    assert any(
        item["owner"] == "catalog" and item["id"] == "catalog.schema.search"
        for item in payload["capability_summaries"]
    )


async def test_plan_referencing_missing_context_evidence_fails_validation(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(
                steps=[
                    {
                        "id": "step_1",
                        "kind": "query",
                        "purpose": "Query with missing context",
                        "depends_on": [],
                        "context_evidence_refs": [
                            {
                                "id": "evidence-missing",
                                "payload_fingerprint": "nope",
                            }
                        ],
                        "expected_evidence": ["query.result"],
                    }
                ]
            )
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.BLOCKED
    validation = next(
        item for item in snapshot.evidence if item.kind == "analysis.plan.validation"
    )
    assert validation.accepted is False
    assert (
        "context_evidence_missing:step_1:evidence-missing"
        in validation.payload["errors"]
    )
    assert not any(
        task.capability_id == "db.sql.execute_read" for task in snapshot.tasks
    )


async def test_query_steps_use_existing_validated_execute_path_and_exact_refs(tmp_path):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month ORDER BY month LIMIT 10"
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders WHERE month = 'March' GROUP BY region LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    read_tasks = [
        task for task in snapshot.tasks if task.capability_id == "db.sql.execute_read"
    ]
    assert len(read_tasks) == 2
    for task in read_tasks:
        dependency = next(
            dep for dep in task.dependencies if dep.evidence_kind == "sql.validation"
        )
        assert task.input["validated_evidence_id"] == dependency.evidence_id
        assert dependency.payload_fingerprint
        assert task.metadata["analysis_step_id"] in {"step_1", "step_2"}

    results = [item for item in snapshot.evidence if item.kind == "query.result"]
    assert {item.metadata["analysis_step_id"] for item in results} == {
        "step_1",
        "step_2",
    }
    synthesis_task = next(
        task for task in snapshot.tasks if task.capability_id == "db.analysis.summarize"
    )
    result_dependencies = [
        dep
        for dep in synthesis_task.dependencies
        if dep.evidence_kind == "query.result"
    ]
    assert {dep.evidence_id for dep in result_dependencies} == {
        item.id for item in results
    }
    by_id = {item.id: item for item in results}
    for dep in result_dependencies:
        assert (
            dep.payload_fingerprint
            == by_id[dep.evidence_id].metadata["payload_fingerprint"]
        )


async def test_query_step_context_refs_become_exact_task_dependencies(tmp_path):
    def plan_with_context_ref(messages):
        content = messages[-1]["content"]
        refs_json = content.split("Source evidence refs: ", 1)[1].split(
            "\nCapability summaries:",
            1,
        )[0]
        context_ref = json.loads(refs_json)[0]
        return _analysis_plan(
            steps=[
                {
                    "id": "step_1",
                    "kind": "query",
                    "purpose": "Query with exact context ref",
                    "depends_on": [],
                    "context_evidence_refs": [context_ref],
                    "expected_evidence": ["query.result"],
                },
                {
                    "id": "step_2",
                    "kind": "synthesis",
                    "purpose": "Summarize",
                    "depends_on": ["step_1"],
                    "expected_evidence": ["analysis.synthesis"],
                },
            ]
        )

    llm = FakeAnalysisLLMService(
        planner_responses=[
            plan_with_context_ref,
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    plan = next(item for item in snapshot.evidence if item.kind == "analysis.plan")
    context_ref = plan.payload["steps"][0]["context_evidence_refs"][0]
    query_step_tasks = [
        task
        for task in snapshot.tasks
        if task.metadata.get("analysis_step_id") == "step_1"
    ]
    assert query_step_tasks
    for task in query_step_tasks:
        dependency = next(
            (dep for dep in task.dependencies if dep.evidence_id == context_ref["id"]),
            None,
        )
        assert dependency is not None
        assert dependency.payload_fingerprint == context_ref["payload_fingerprint"]


async def test_parallel_enabled_runs_independent_query_steps_with_distinct_traces(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(
                steps=[
                    {
                        "id": "step_1",
                        "kind": "query",
                        "purpose": "Calculate revenue by month",
                        "depends_on": [],
                        "expected_evidence": ["query.result"],
                    },
                    {
                        "id": "step_2",
                        "kind": "query",
                        "purpose": "Calculate revenue by region",
                        "depends_on": [],
                        "expected_evidence": ["query.result"],
                    },
                    {
                        "id": "step_3",
                        "kind": "synthesis",
                        "purpose": "Summarize both independent results",
                        "depends_on": ["step_1", "step_2"],
                        "expected_evidence": ["analysis.synthesis"],
                    },
                ]
            ),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10",
                columns=["month", "revenue"],
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders GROUP BY region LIMIT 10",
                columns=["region", "revenue"],
            ),
        ]
    )
    runtime, sqlite = await _runtime(
        tmp_path,
        llm,
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "analysis_parallel_enabled": True,
                    "analysis_max_concurrency": 2,
                }
            }
        ),
    )
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.SUCCEEDED
    read_tasks = [
        task for task in snapshot.tasks if task.capability_id == "db.sql.execute_read"
    ]
    assert len(read_tasks) == 2
    assert len({task.id for task in read_tasks}) == 2
    assert {task.metadata["analysis_step_id"] for task in read_tasks} == {
        "step_1",
        "step_2",
    }
    query_results = [item for item in snapshot.evidence if item.kind == "query.result"]
    assert len(query_results) == 2
    assert len({item.id for item in query_results}) == 2
    assert {item.metadata["analysis_step_id"] for item in query_results} == {
        "step_1",
        "step_2",
    }
    checkpoints = [
        item
        for item in snapshot.evidence
        if item.kind == "analysis.checkpoint"
        and item.payload["diagnostics"].get("parallel_batch")
    ]
    assert {item.metadata["analysis_step_id"] for item in checkpoints} == {
        "step_1_checkpoint",
        "step_2_checkpoint",
    }


async def test_checkpoints_preserve_diagnostics_and_budget_usage(tmp_path):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders WHERE month = 'March' GROUP BY region LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    checkpoints = [
        item for item in snapshot.evidence if item.kind == "analysis.checkpoint"
    ]
    assert checkpoints
    assert all(
        item.metadata["analysis_step_kind"] == "checkpoint" for item in checkpoints
    )
    assert checkpoints[0].payload["completed_step_evidence_refs"]
    assert "query_result_rows" in checkpoints[0].payload["budget_usage"]
    progress = checkpoints[0].payload["diagnostics"]["progress"]
    assert progress["operation_id"] == result.operation_id
    assert progress["latest_checkpoint_id"] is None
    assert progress["task_status_counts"]["succeeded"] >= 1
    assert result.diagnostics["analysis"]["diagnostics"]["dependency_evidence_refs"]


async def test_budget_exhaustion_with_replan_flag_persists_revision_evidence(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(
                budgets={
                    "max_steps": 6,
                    "max_query_steps": 3,
                    "max_checkpoint_steps": 3,
                    "max_repairs": 1,
                    "max_total_rows": 1,
                    "max_llm_calls": 6,
                    "max_context_chars": 16000,
                    "max_duration_seconds": 120,
                }
            ),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(
        tmp_path,
        llm,
        config=DbRuntimeConfig(
            metadata={"from_db_options": {"analysis_replan_enabled": True}}
        ),
    )
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.BLOCKED
    revision = next(
        item for item in snapshot.evidence if item.kind == "analysis.plan.revision"
    )
    plan = next(item for item in snapshot.evidence if item.kind == "analysis.plan")
    budget_checkpoint = next(
        item
        for item in snapshot.evidence
        if item.kind == "analysis.checkpoint"
        and item.metadata["analysis_step_id"] == "budget_checkpoint"
    )
    assert revision.accepted
    assert revision.payload["parent_plan_evidence_id"] == plan.id
    assert revision.payload["revision_number"] == 1
    assert revision.payload["trigger_evidence_refs"][0]["id"] == budget_checkpoint.id
    assert revision.payload["failed_or_invalidated_step_ids"] == ["step_2", "step_3"]
    assert revision.payload["budget_usage"]["total_rows"] > 1
    assert revision.payload["retry_rationale"] == "budget_exhausted"
    synthesis = [
        item for item in snapshot.evidence if item.kind == "analysis.synthesis"
    ][-1]
    assert synthesis.payload["partial"] is True
    assert synthesis.payload["pause_reason"] == "budget_exhausted"
    assert synthesis.payload["remaining_step_ids"] == ["step_2", "step_3"]
    progress = await runtime.inspect_analysis_operation(result.operation_id)
    assert progress["latest_synthesis_id"] == synthesis.id
    assert progress["latest_synthesis_partial"] is True
    assert progress["final_synthesis_id"] is None
    replan_task = next(
        task for task in snapshot.tasks if task.capability_id == "db.analysis.replan"
    )
    dependency_ids = {dependency.evidence_id for dependency in replan_task.dependencies}
    assert {plan.id, budget_checkpoint.id} <= dependency_ids


async def test_resume_consumes_latest_accepted_revision_and_skips_completed_sql(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(
                budgets={
                    "max_steps": 6,
                    "max_query_steps": 3,
                    "max_checkpoint_steps": 3,
                    "max_repairs": 1,
                    "max_total_rows": 1,
                    "max_llm_calls": 6,
                    "max_context_chars": 16000,
                    "max_duration_seconds": 120,
                }
            ),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10",
                columns=["month", "revenue"],
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders GROUP BY region LIMIT 10",
                columns=["region", "revenue"],
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        before = await runtime.inspect_operation(result.operation_id)
        plan = next(item for item in before.evidence if item.kind == "analysis.plan")
        checkpoint = next(
            item
            for item in before.evidence
            if item.kind == "analysis.checkpoint"
            and item.metadata["analysis_step_id"] == "budget_checkpoint"
        )
        revision_payload = {
            "analysis_id": "analysis-test-1",
            "parent_plan_evidence_id": plan.id,
            "parent_plan_evidence_ref": {
                "id": plan.id,
                "kind": plan.kind,
                "owner": plan.owner,
                "task_id": plan.task_id,
                "payload_fingerprint": plan.metadata["payload_fingerprint"],
            },
            "revision_number": 1,
            "trigger_evidence_refs": [
                {
                    "id": checkpoint.id,
                    "kind": checkpoint.kind,
                    "owner": checkpoint.owner,
                    "task_id": checkpoint.task_id,
                    "payload_fingerprint": checkpoint.metadata["payload_fingerprint"],
                }
            ],
            "failed_or_invalidated_step_ids": ["step_2", "step_3"],
            "unchanged_step_ids": ["step_1"],
            "replacement_steps": [
                {
                    "id": "step_2_revised",
                    "kind": "query",
                    "purpose": "Break revenue down by region within revised budget",
                    "depends_on": ["step_1"],
                    "expected_evidence": ["query.result"],
                },
                {
                    "id": "step_3_revised",
                    "kind": "synthesis",
                    "purpose": "Summarize revised analysis",
                    "depends_on": ["step_1", "step_2_revised"],
                    "expected_evidence": ["analysis.synthesis"],
                },
            ],
            "budgets": {
                "max_steps": 6,
                "max_query_steps": 3,
                "max_checkpoint_steps": 3,
                "max_repairs": 1,
                "max_total_rows": 1000,
                "max_llm_calls": 6,
                "max_context_chars": 16000,
                "max_duration_seconds": 120,
            },
            "budget_usage": {"total_rows": 3},
            "retry_rationale": "budget_increased_after_review",
            "prior_sql_fingerprints": [],
            "diagnostics": {"mode": "test"},
        }
        await runtime.store.save_evidence(
            Evidence(
                id="evidence-revision-test",
                kind="analysis.plan.revision",
                owner="db_runtime",
                operation_id=result.operation_id,
                accepted=True,
                payload=revision_payload,
                metadata={
                    **analysis_metadata(
                        analysis_id="analysis-test-1",
                        step_id="analysis_replan",
                        phase="replan",
                        plan_evidence_id=plan.id,
                    ),
                    "payload_fingerprint": stable_fingerprint(revision_payload),
                },
            )
        )
        resumed = await runtime.resume_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.BLOCKED
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    read_tasks = [
        task for task in resumed.tasks if task.capability_id == "db.sql.execute_read"
    ]
    assert {task.metadata["analysis_step_id"] for task in read_tasks} == {
        "step_1",
        "step_2_revised",
    }
    assert (
        len(
            [
                task
                for task in read_tasks
                if task.metadata["analysis_step_id"] == "step_1"
            ]
        )
        == 1
    )
    final = [
        item
        for item in resumed.evidence
        if item.kind == "analysis.synthesis" and not item.payload.get("partial")
    ][-1]
    assert final.metadata["analysis_plan_evidence_id"] == "evidence-revision-test"


async def test_inspect_analysis_operation_reports_progress_and_final_synthesis(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders WHERE month = 'March' GROUP BY region LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        progress = await runtime.inspect_analysis_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert progress["operation_id"] == result.operation_id
    assert progress["operation_status"] == OperationStatus.SUCCEEDED.value
    assert {"step_1", "step_2", "step_3"} <= set(progress["completed_step_ids"])
    assert progress["remaining_step_ids"] == []
    assert progress["latest_checkpoint_id"]
    assert progress["latest_synthesis_id"] == progress["final_synthesis_id"]
    assert progress["latest_synthesis_partial"] is False
    assert progress["next_resumable_task_ids"] == []


async def test_no_llm_or_invalid_json_returns_clarification_without_db_work(tmp_path):
    runtime, sqlite = await _runtime(tmp_path, llm_service=None)
    try:
        no_llm = await runtime.run(
            DbRequest(
                "Show revenue using multiple queries",
                metadata={"analysis_mode": "multi_step"},
            )
        )
        no_llm_snapshot = await runtime.inspect_operation(no_llm.operation_id)
    finally:
        await sqlite.disconnect()

    assert no_llm.status is OperationStatus.BLOCKED
    assert "no DB LLM planner" in no_llm.answer
    assert not any(
        task.capability_id == "db.sql.execute_read" for task in no_llm_snapshot.tasks
    )

    llm = FakeAnalysisLLMService(planner_responses=["not json"])
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        invalid = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        invalid_snapshot = await runtime.inspect_operation(invalid.operation_id)
    finally:
        await sqlite.disconnect()

    assert invalid.status is OperationStatus.BLOCKED
    assert "clarify" in invalid.answer.lower()
    assert not any(
        task.capability_id == "db.sql.execute_read" for task in invalid_snapshot.tasks
    )


async def test_resume_skips_completed_analysis_tasks(tmp_path):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders WHERE month = 'March' GROUP BY region LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        before = await runtime.inspect_operation(result.operation_id)
        resumed = await runtime.resume_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert resumed.operation.status is OperationStatus.SUCCEEDED
    before_ids = [task.id for task in before.tasks]
    after_ids = [task.id for task in resumed.tasks]
    assert before_ids == after_ids
    assert all(task.status is TaskStatus.SUCCEEDED for task in resumed.tasks)
    assert len(llm.calls) == 3


async def test_resume_materializes_incomplete_ready_analysis_steps(tmp_path):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders WHERE month = 'March' GROUP BY region LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    original_synthesis = runtime._execute_analysis_synthesis_task

    async def fail_once(*args, **kwargs):
        runtime._execute_analysis_synthesis_task = original_synthesis
        await original_synthesis(
            *args,
            **{
                **kwargs,
                "partial": True,
                "pause_reason": "interrupted",
                "remaining_step_ids": ("step_3",),
            },
        )
        raise RuntimeError("synthesis interrupted")

    runtime._execute_analysis_synthesis_task = fail_once
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        before = await runtime.inspect_operation(result.operation_id)
        progress_before = await runtime.inspect_analysis_operation(result.operation_id)
        resumed = await runtime.resume_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.FAILED
    assert before is not None
    partial = [item for item in before.evidence if item.kind == "analysis.synthesis"][
        -1
    ]
    assert partial.payload["partial"] is True
    assert partial.payload["pause_reason"] == "interrupted"
    assert progress_before["latest_synthesis_id"] == partial.id
    assert progress_before["latest_synthesis_partial"] is True
    assert progress_before["final_synthesis_id"] is None
    before_sql_task_ids = [
        task.id for task in before.tasks if task.capability_id == "db.sql.execute_read"
    ]
    after_sql_task_ids = [
        task.id for task in resumed.tasks if task.capability_id == "db.sql.execute_read"
    ]
    assert after_sql_task_ids == before_sql_task_ids
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    final_synthesis = [
        item
        for item in resumed.evidence
        if item.kind == "analysis.synthesis" and not item.payload.get("partial")
    ]
    assert final_synthesis


async def test_governance_blocked_analysis_emits_checkpoint_and_partial_synthesis(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    original_query_step = runtime._execute_analysis_query_step

    # Import through the module to avoid broadening the public test surface.
    from daita.db.runtime import DbRuntimeGovernanceBlocked

    async def block_second_query(*args, **kwargs):
        if kwargs["step_metadata"]["analysis_step_id"] == "step_2":
            raise DbRuntimeGovernanceBlocked(
                operation=args[4],
                task=None,
                governance=GovernanceResult(
                    allowed=False,
                    blocked=False,
                    pending_approval=True,
                    metadata={"reason": "unit_test_approval"},
                ),
            )
        return await original_query_step(*args, **kwargs)

    runtime._execute_analysis_query_step = block_second_query
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    assert result.status is OperationStatus.BLOCKED
    checkpoint = next(
        item
        for item in snapshot.evidence
        if item.kind == "analysis.checkpoint"
        and item.metadata["analysis_step_id"] == "analysis_blocked_checkpoint"
    )
    assert checkpoint.payload["diagnostics"]["checkpoint_reason"] == "blocked"
    partial = next(
        item
        for item in snapshot.evidence
        if item.kind == "analysis.synthesis"
        and item.payload.get("pause_reason") == "approval_required"
    )
    assert partial.payload["partial"] is True
    assert result.diagnostics["analysis_checkpoint_id"] == checkpoint.id
    assert result.diagnostics["analysis_partial_synthesis_id"] == partial.id


async def test_analysis_tasks_and_evidence_stay_in_same_operation_with_metadata(
    tmp_path,
):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders WHERE month = 'March' GROUP BY region LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await sqlite.disconnect()

    analysis_tasks = [
        task for task in snapshot.tasks if task.metadata.get("analysis_id")
    ]
    analysis_evidence = [
        item for item in snapshot.evidence if item.metadata.get("analysis_id")
    ]
    assert analysis_tasks
    assert analysis_evidence
    assert {task.operation_id for task in analysis_tasks} == {result.operation_id}
    assert {item.operation_id for item in analysis_evidence} == {result.operation_id}
    assert {
        item.metadata.get("analysis_step_id")
        for item in analysis_evidence
        if item.kind == "query.result"
    } == {"step_1", "step_2"}
    assert {
        item.metadata["analysis_step_kind"]
        for item in analysis_evidence
        if item.metadata.get("analysis_step_kind")
    } <= {"query", "checkpoint", "synthesis"}
    assert all(
        item.metadata.get("analysis_plan_evidence_id")
        for item in analysis_evidence
        if item.kind not in {"analysis.plan", "planning.context"}
    )


async def test_final_answer_comes_from_accepted_analysis_synthesis(tmp_path):
    llm = FakeAnalysisLLMService(
        planner_responses=[
            _analysis_plan(),
            _query_plan(
                "SELECT month, SUM(total) AS revenue FROM orders GROUP BY month LIMIT 10"
            ),
            _query_plan(
                "SELECT region, SUM(total) AS revenue FROM orders WHERE month = 'March' GROUP BY region LIMIT 10"
            ),
        ]
    )
    runtime, sqlite = await _runtime(tmp_path, llm)
    try:
        result = await runtime.run(
            "Show a multi-step analysis to investigate revenue using multiple queries"
        )
    finally:
        await sqlite.disconnect()

    synthesis = next(
        item for item in result.evidence if item.kind == "analysis.synthesis"
    )
    assert synthesis.accepted
    assert result.answer == synthesis.payload["answer"]
    assert synthesis.payload["cited_evidence_refs"]
