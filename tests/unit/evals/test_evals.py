import json

from daita.evals import EvalSuite, EvalSuiteConfig
from daita.evals.analysis import RunEvidence, extract_run_evidence, summarize_stability
from daita.evals.assertions.runtime import (
    capability_assertions,
    evidence_assertions,
    task_assertions,
)
from daita.evals.assertions.stability import stability_assertions
from daita.evals.config import JudgeCriterion, JudgeExpectations
from daita.evals.judges import build_judge_result
from daita.evals.models import (
    RunMetrics,
    RuntimeEvidenceRecord,
    RuntimeTaskEvidence,
)
from daita.evals.reporters import render_junit, render_pretty
from daita.db.models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from daita.runtime import Evidence, OperationStatus


def test_load_yaml_config(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("""
name: sales-evals
version: 1
agent:
  factory: "tests.fixtures.eval_agents:create_passing_agent"
cases:
  - id: top-products
    prompt: What were the top products?
    expectations:
      answer:
        contains: ["Widget A"]
""")

    config = EvalSuiteConfig.from_file(path)

    assert config.name == "sales-evals"
    assert config.cases[0].expectations.answer.contains == ["Widget A"]


def test_load_json_config(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "name": "json-evals",
                "agent": {"factory": "tests.fixtures.eval_agents:create_passing_agent"},
                "cases": [{"id": "case-1", "prompt": "hello"}],
            }
        )
    )

    assert EvalSuiteConfig.from_file(path).name == "json-evals"


async def test_suite_runs_factory_agent_and_writes_artifacts(tmp_path):
    config = EvalSuiteConfig(
        name="sales-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "top-products",
                "prompt": "What were the top products?",
                "expectations": {
                    "answer": {
                        "contains": ["Widget A"],
                        "numeric": [
                            {
                                "label": "revenue",
                                "expected": 12840.50,
                                "tolerance": 0.01,
                            }
                        ],
                    },
                    "capabilities": {
                        "required": ["db.sql.execute_read"],
                        "max_calls": 2,
                    },
                    "evidence": {
                        "required_kinds": ["sql.validation", "query.result"],
                    },
                    "sql": {
                        "require_limit": True,
                        "must_include": ["GROUP BY"],
                        "must_not_include": ["DELETE"],
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)

    assert report.status == "passed"
    assert report.summary.cases_passed == 1
    assert (tmp_path / report.run_id / "report.json").exists()
    assert (
        tmp_path / report.run_id / "cases" / "top-products" / "run-001.json"
    ).exists()


def test_extracts_real_db_operation_result_dataclasses():
    result = DbOperationResult(
        operation_id="operation-real",
        request=DbRequest("How many?"),
        intent=DbIntent(DbIntentKind.DATA_QUERY),
        contract=DbOperationContract(
            operation_type="db.query",
            required_evidence=("query.result",),
        ),
        status=OperationStatus.SUCCEEDED,
        answer="The count is 1.",
        evidence=(
            Evidence(
                id="evidence-real",
                kind="query.result",
                owner="sqlite",
                operation_id="operation-real",
                task_id="task-real",
                payload={"sql": "SELECT COUNT(*) AS count FROM customers", "rows": []},
            ),
        ),
        diagnostics={
            "execution": {
                "task_count": 1,
                "tasks": [
                    {
                        "id": "task-real",
                        "operation_id": "operation-real",
                        "capability_id": "db.sql.execute_read",
                        "executor_id": "db.sql.execute_read",
                        "input": {},
                        "status": "succeeded",
                        "required_evidence": ["query.result"],
                        "dependencies": [],
                        "metadata": {"owner": "sqlite"},
                    }
                ],
            },
            "governance": {
                "allowed": True,
                "blocked": False,
                "pending_approval": False,
            },
        },
    )

    evidence = extract_run_evidence("How many?", {"runtime_result": result})

    assert evidence.operation_type == "db.query"
    assert evidence.intent == "data.query"
    assert evidence.tasks[0].capability_id == "db.sql.execute_read"
    assert evidence.evidence[0].kind == "query.result"


async def test_sql_assertions_fail_with_stable_codes():
    config = EvalSuiteConfig(
        name="sql-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_sql_failure_agent"},
        cases=[
            {
                "id": "safe-readonly",
                "prompt": "Show users",
                "expectations": {
                    "sql": {
                        "require_limit": True,
                        "forbidden_tables": ["users_pii"],
                    }
                },
            }
        ],
    )

    report = await EvalSuite(config).run(write_artifacts=False)

    assert report.status == "failed"
    assert {failure.code for failure in report.failures} == {
        "sql_missing_limit",
        "sql_forbidden_table",
    }


async def test_sql_required_table_and_row_count_assertions_fail():
    config = EvalSuiteConfig(
        name="sql-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "table-and-row-limit",
                "prompt": "Show users",
                "expectations": {
                    "sql": {
                        "required_tables": ["users"],
                        "max_rows_returned": 0,
                    }
                },
            }
        ],
    )

    report = await EvalSuite(config).run(write_artifacts=False)

    assert report.status == "failed"
    assert {failure.code for failure in report.failures} == {
        "sql_required_table_missing",
        "sql_too_many_rows",
    }


async def test_query_result_content_assertions_pass_and_fail():
    passing = EvalSuiteConfig(
        name="result-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "gold-row",
                "prompt": "Which product won?",
                "expectations": {
                    "result": {
                        "required_columns": ["product", "revenue"],
                        "required_rows": [{"product": "Widget A", "revenue": 12840.50}],
                        "min_rows": 1,
                        "max_rows": 1,
                    }
                },
            }
        ],
    )
    failing = EvalSuiteConfig(
        name="result-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "missing-gold-row",
                "prompt": "Which product won?",
                "expectations": {
                    "result": {
                        "required_columns": ["margin"],
                        "required_rows": [{"product": "Widget B"}],
                    }
                },
            }
        ],
    )

    passing_report = await EvalSuite(passing).run(write_artifacts=False)
    failing_report = await EvalSuite(failing).run(write_artifacts=False)

    assert passing_report.status == "passed"
    assert failing_report.status == "failed"
    assert {failure.code for failure in failing_report.failures} == {
        "query_result_required_column_missing",
        "query_result_required_row_missing",
    }


async def test_legacy_tool_call_results_are_rejected():
    config = EvalSuiteConfig(
        name="legacy-rejected",
        agent={"factory": "tests.fixtures.eval_agents:create_legacy_agent"},
        cases=[{"id": "legacy", "prompt": "Use the old shape"}],
    )

    report = await EvalSuite(config).run(write_artifacts=False)

    assert report.status == "failed"
    assert report.failures[0].code == "run_error"
    assert "no longer accepts legacy tool_calls results" in report.failures[0].message


async def test_runtime_capability_and_evidence_assertions_pass():
    config = EvalSuiteConfig(
        name="runtime-contracts",
        agent={"factory": "tests.fixtures.eval_agents:create_runtime_capability_agent"},
        cases=[
            {
                "id": "runtime-capabilities",
                "prompt": "Load customer context",
                "expectations": {
                    "capabilities": {
                        "required": ["catalog.schema.search", "db.sql.execute_read"],
                        "required_owners": ["catalog", "sqlite"],
                        "forbidden": ["memory.semantic.write"],
                        "max_calls": 3,
                    },
                    "evidence": {
                        "required_kinds": [
                            "schema.search_result",
                            "sql.validation",
                            "query.result",
                        ],
                        "required_owners": ["catalog", "sqlite"],
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(write_artifacts=False)

    assert report.status == "passed"
    assert report.summary.runs_passed == 1


async def test_governance_assertions_and_runtime_artifacts(tmp_path):
    config = EvalSuiteConfig(
        name="governance-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_governance_agent"},
        cases=[
            {
                "id": "governed-run",
                "prompt": "Inspect schema and query sales.",
                "expectations": {
                    "capabilities": {
                        "required": ["db.sql.validate"],
                        "required_owners": ["sqlite"],
                    },
                    "governance": {
                        "allowed": True,
                        "blocked": False,
                        "pending_approval": False,
                        "required_policies": ["read_only_sql"],
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    run = report.cases[0].runs[0]
    payload = json.loads(
        (
            tmp_path / report.run_id / "cases" / "governed-run" / "run-001.json"
        ).read_text()
    )

    assert report.status == "passed"
    assert run.tasks[0].capability_id == "db.sql.validate"
    assert run.governance.allowed is True
    assert payload["tasks"]
    assert payload["governance"]["allowed"] is True
    assert "capabilities: db.sql.validate" in render_pretty(report)
    assert "owners: sqlite" in render_pretty(report)


async def test_repeat_runs_evaluate_stability():
    config = EvalSuiteConfig(
        name="repeat-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_unstable_agent"},
        defaults={"runs": 2},
        cases=[
            {
                "id": "stable-answer",
                "prompt": "Answer consistently",
                "expectations": {
                    "stability": {
                        "require_same_capabilities": True,
                        "max_answer_variants": 1,
                    }
                },
            }
        ],
    )

    report = await EvalSuite(config).run(write_artifacts=False)

    assert report.status == "failed"
    assert {failure.code for failure in report.failures} == {
        "unstable_capabilities",
        "unstable_answer",
    }


async def test_reporters_render_output():
    config = EvalSuiteConfig(
        name="report-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[{"id": "case-1", "prompt": "hello"}],
    )
    report = await EvalSuite(config).run(write_artifacts=False)

    pretty = render_pretty(report)
    assert "Daita Eval: report-evals" in pretty
    assert "Cases" in pretty
    assert "PASSED case-1" in pretty
    assert "<testsuite" in render_junit(report)


async def test_artifact_privacy_flags_strip_full_answer_and_evidence_payloads(tmp_path):
    config = EvalSuiteConfig(
        name="privacy-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        artifacts={"include_full_answers": False, "include_evidence_payloads": False},
        cases=[{"id": "case-1", "prompt": "hello"}],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    run_path = tmp_path / report.run_id / "cases" / "case-1" / "run-001.json"
    payload = json.loads(run_path.read_text())

    assert payload["final_answer"] is None
    assert payload["evidence"][0]["payload"] is None


async def test_dataset_records_expand_into_cases(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "dataset-case",
                        "prompt": "Which product won?",
                        "expected": {"contains": ["Widget A"]},
                    }
                )
            ]
        )
    )
    config = EvalSuiteConfig(
        name="dataset-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        dataset={"path": str(dataset_path)},
        case_template={
            "expectations": {
                "capabilities": {"required": ["db.sql.execute_read"]},
                "sql": {"require_limit": True},
            }
        },
    )

    report = await EvalSuite(config).run(write_artifacts=False)

    assert report.status == "passed"
    assert report.summary.cases_total == 1
    assert report.cases[0].case_id == "dataset-case"


async def test_llm_judge_adds_assertion_result_and_artifacts(tmp_path):
    config = EvalSuiteConfig(
        name="judge-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "judged-case",
                "prompt": "Which product won?",
                "expectations": {
                    "answer": {"contains": ["Widget A"]},
                    "judge": {
                        "provider": "mock",
                        "model": "mock-model",
                        "pass_score": 0,
                        "criteria": [
                            {
                                "id": "mentions_winner",
                                "description": "The answer mentions the winning product.",
                            }
                        ],
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    assertion_codes = {
        assertion.code for assertion in report.cases[0].runs[0].assertions
    }
    run = report.cases[0].runs[0]
    judge_dir = tmp_path / report.run_id / "cases" / "judged-case" / "run-001"

    assert report.status == "passed"
    assert report.summary.judge_calls == 1
    assert "judge_passed" in assertion_codes
    assert run.judges[0].criteria[0].id == "mentions_winner"
    assert (judge_dir / "judge-001-input.json").exists()
    assert (judge_dir / "judge-001-output.json").exists()


def test_structured_judge_result_requires_required_criteria_to_pass():
    config = JudgeExpectations(
        provider="mock",
        model="mock-model",
        pass_score=0.8,
        criteria=[
            JudgeCriterion(
                id="grounding",
                description="Answer is grounded.",
                required=True,
            )
        ],
    )

    result = build_judge_result(
        config,
        {
            "score": 0.95,
            "reasoning": "Mostly good, but missing grounding.",
            "criteria": [
                {
                    "id": "grounding",
                    "passed": False,
                    "score": 0.2,
                    "evidence": "No runtime evidence cited.",
                }
            ],
        },
        raw_response="{}",
        metrics={"latency_ms": 12, "cost": 0.001, "tokens_total": 50},
    )

    assert result.passed is False
    assert result.score == 0.95
    assert result.criteria[0].required is True
    assert result.cost == 0.001


async def test_record_and_compare_baseline_detects_new_failure(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    passing = EvalSuiteConfig(
        name="baseline-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "top-products",
                "prompt": "Which product won?",
                "expectations": {"answer": {"contains": ["Widget A"]}},
            }
        ],
    )
    failing = EvalSuiteConfig(
        name="baseline-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_sql_failure_agent"},
        baselines={"fail_on": {"new_failures": True}},
        cases=[
            {
                "id": "top-products",
                "prompt": "Which product won?",
                "expectations": {"answer": {"contains": ["Widget A"]}},
            }
        ],
    )

    baseline_report = await EvalSuite(passing).run(
        write_artifacts=False,
        record_baseline=True,
        baseline_path=baseline_path,
    )
    report = await EvalSuite(failing).run(
        write_artifacts=False,
        compare_baseline=True,
        baseline_path=baseline_path,
    )

    assert baseline_report.status == "passed"
    assert baseline_path.exists()
    assert report.status == "failed"
    assert report.baseline_comparison["status"] == "failed"
    assert any(failure.code == "baseline_regression" for failure in report.failures)


def test_exact_capability_sequence_assertion_fails_on_order_drift():
    config = EvalSuiteConfig(
        name="sequence-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "warm-read",
                "prompt": "How many customers?",
                "expectations": {
                    "capabilities": {
                        "exact_sequence": [
                            "db.query.plan",
                            "db.query.plan.validate",
                            "db.sql.validate",
                            "db.sql.execute_read",
                            "db.answer.synthesize",
                        ]
                    }
                },
            }
        ],
    )
    evidence = _run_evidence(
        capabilities=[
            "db.sql.execute_read",
            "db.query.plan",
            "db.sql.validate",
            "db.query.plan.validate",
            "db.answer.synthesize",
        ]
    )

    results = capability_assertions(
        config.cases[0].expectations,
        evidence,
        config.defaults,
    )

    assert {result.code for result in results} == {"capability_sequence_mismatch"}


def test_allowed_capability_sequences_accept_only_declared_traces():
    config = EvalSuiteConfig(
        name="allowed-sequence-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "warm-read",
                "prompt": "How many customers?",
                "expectations": {
                    "capabilities": {
                        "allowed_sequences": [
                            [
                                "db.query.plan",
                                "db.query.plan.validate",
                                "db.sql.validate",
                                "db.sql.execute_read",
                                "db.answer.synthesize",
                            ]
                        ]
                    }
                },
            }
        ],
    )
    evidence = _run_evidence(
        capabilities=[
            "db.query.plan",
            "db.sql.validate",
            "db.sql.execute_read",
            "db.answer.synthesize",
        ]
    )

    results = capability_assertions(
        config.cases[0].expectations,
        evidence,
        config.defaults,
    )

    assert {result.code for result in results} == {"capability_sequence_not_allowed"}


def test_forbidden_capability_subsequence_assertion_reports_matching_tasks():
    config = EvalSuiteConfig(
        name="forbidden-subsequence-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "warm-read",
                "prompt": "How many customers?",
                "expectations": {
                    "capabilities": {
                        "forbidden_subsequences": [
                            ["db.schema.inspect", "catalog.source.register"]
                        ]
                    }
                },
            }
        ],
    )
    evidence = _run_evidence(
        capabilities=[
            "db.schema.inspect",
            "catalog.source.register",
            "db.sql.execute_read",
        ]
    )

    results = capability_assertions(
        config.cases[0].expectations,
        evidence,
        config.defaults,
    )

    assert {result.code for result in results} == {"forbidden_capability_subsequence"}
    assert results[0].related_task_ids == ["task-1", "task-2"]


def test_max_per_capability_assertions_fail_when_a_capability_repeats():
    config = EvalSuiteConfig(
        name="max-capability-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "warm-read",
                "prompt": "How many customers?",
                "expectations": {
                    "capabilities": {"max_per_capability": {"db.query.plan": 0}},
                    "tasks": {"max_per_capability": {"db.query.plan": 0}},
                },
            }
        ],
    )
    evidence = _run_evidence(
        capabilities=["db.query.plan", "db.sql.validate", "db.sql.execute_read"]
    )

    capability_results = capability_assertions(
        config.cases[0].expectations,
        evidence,
        config.defaults,
    )
    task_results = task_assertions(config.cases[0].expectations, evidence)

    assert {result.code for result in capability_results} == {
        "too_many_capability_calls_by_id"
    }
    assert {result.code for result in task_results} == {"too_many_tasks_by_capability"}


def test_evidence_max_per_kind_assertion_fails_on_extra_records():
    config = EvalSuiteConfig(
        name="evidence-count-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "warm-read",
                "prompt": "How many customers?",
                "expectations": {
                    "evidence": {"max_per_kind": {"catalog.column_values.search": 0}}
                },
            }
        ],
    )
    evidence = _run_evidence(
        capabilities=["catalog.column_values.search"],
        evidence_kinds=["catalog.column_values.search"],
    )

    results = evidence_assertions(config.cases[0].expectations, evidence)

    assert {result.code for result in results} == {"too_many_evidence_records_by_kind"}


def test_stability_latency_percentile_gates_fail_independently():
    config = EvalSuiteConfig(
        name="latency-percentile-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        cases=[
            {
                "id": "warm-count-50-runs",
                "prompt": "How many customers?",
                "expectations": {
                    "stability": {
                        "max_latency_p50_ms": 10,
                        "max_latency_p95_ms": 10,
                        "max_latency_p99_ms": 10,
                    }
                },
            }
        ],
    )
    runs = [
        _run_evidence(capabilities=["db.sql.execute_read"], latency_ms=latency)
        for latency in [5, 10, 20, 30]
    ]

    results = stability_assertions(config.cases[0], summarize_stability(runs))

    assert {result.code for result in results} == {
        "latency_p50_too_high",
        "latency_p95_too_high",
        "latency_p99_too_high",
    }


def _run_evidence(
    *,
    capabilities: list[str],
    evidence_kinds: list[str] | None = None,
    latency_ms: float | None = None,
) -> RunEvidence:
    tasks = [
        RuntimeTaskEvidence(
            id=f"task-{index}",
            capability_id=capability_id,
            executor_id=capability_id,
            owner="sqlite",
            status="succeeded",
        )
        for index, capability_id in enumerate(capabilities, start=1)
    ]
    records = [
        RuntimeEvidenceRecord(
            id=f"evidence-{index}",
            kind=kind,
            owner="sqlite",
            task_id=tasks[min(index - 1, len(tasks) - 1)].id if tasks else None,
        )
        for index, kind in enumerate(evidence_kinds or [], start=1)
    ]
    return RunEvidence(
        answer="ok",
        prompt_hash="prompt",
        answer_hash="answer",
        operation_id="operation-1",
        operation_status="succeeded",
        operation_type="db.query",
        intent="query",
        tasks=tasks,
        evidence=records,
        governance=None,
        approvals=[],
        warnings=[],
        metrics=RunMetrics(latency_ms=latency_ms),
    )
