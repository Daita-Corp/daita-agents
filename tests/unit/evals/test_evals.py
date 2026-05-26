import json

from daita.evals import EvalSuite, EvalSuiteConfig
from daita.evals.config import JudgeCriterion, JudgeExpectations
from daita.evals.judges import build_judge_result
from daita.evals.reporters import render_junit, render_pretty


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
                    "tools": {"required": ["sqlite_query"], "max_calls": 1},
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


async def test_non_sql_data_operation_inspectors_pass():
    config = EvalSuiteConfig(
        name="data-ops",
        agent={"factory": "tests.fixtures.eval_agents:create_data_ops_agent"},
        cases=[
            {
                "id": "data-services",
                "prompt": "Load customer context",
                "expectations": {
                    "operations": {
                        "required_categories": ["file", "api", "storage", "vector"],
                        "forbidden_categories": ["workflow"],
                        "max_write_operations": 0,
                        "max_delete_operations": 0,
                    },
                    "files": {"required_read": ["sales.csv"]},
                    "api": {
                        "required_methods": ["GET"],
                        "required_hosts": ["api.example.com"],
                        "forbidden_methods": ["POST", "DELETE"],
                    },
                    "storage": {
                        "required_buckets": ["analytics"],
                        "forbidden_buckets": ["prod-secrets"],
                        "forbidden_write": True,
                    },
                    "vector": {
                        "max_top_k": 10,
                        "required_filters": ["tenant_id"],
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(write_artifacts=False)

    assert report.status == "passed"
    assert report.summary.runs_passed == 1


async def test_skill_and_plugin_execution_assertions_and_artifacts(tmp_path):
    config = EvalSuiteConfig(
        name="execution-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_execution_agent"},
        cases=[
            {
                "id": "skill-plugin-run",
                "prompt": "Inspect schema and query sales.",
                "expectations": {
                    "skills": {
                        "required": ["schema_discovery"],
                        "forbidden": ["web_search"],
                        "max_calls": 2,
                        "max_latency_ms": 100,
                        "max_errors": 0,
                    },
                    "plugins": {
                        "required": ["sqlite"],
                        "forbidden": ["s3"],
                        "max_calls": 2,
                        "max_latency_ms": 100,
                        "max_errors": 0,
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    run = report.cases[0].runs[0]
    payload = json.loads(
        (
            tmp_path / report.run_id / "cases" / "skill-plugin-run" / "run-001.json"
        ).read_text()
    )

    assert report.status == "passed"
    assert {span.kind for span in run.execution_spans} == {"skill", "plugin", "tool"}
    assert any(span.name == "schema_discovery" for span in run.execution_spans)
    assert any(span.name == "sqlite" for span in run.execution_spans)
    assert payload["execution_spans"]
    assert "skills: schema_discovery.inspect 25ms" in render_pretty(report)
    assert "plugins: sqlite.query 18ms" in render_pretty(report)


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
                        "require_same_tools": True,
                        "max_answer_variants": 1,
                    }
                },
            }
        ],
    )

    report = await EvalSuite(config).run(write_artifacts=False)

    assert report.status == "failed"
    assert {failure.code for failure in report.failures} == {
        "unstable_tools",
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


async def test_artifact_privacy_flags_strip_full_answer_and_tool_outputs(tmp_path):
    config = EvalSuiteConfig(
        name="privacy-evals",
        agent={"factory": "tests.fixtures.eval_agents:create_passing_agent"},
        artifacts={"include_full_answers": False, "include_tool_outputs": False},
        cases=[{"id": "case-1", "prompt": "hello"}],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    run_path = tmp_path / report.run_id / "cases" / "case-1" / "run-001.json"
    payload = json.loads(run_path.read_text())

    assert payload["final_answer"] is None
    assert payload["tool_calls"][0]["result"] is None


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
                "tools": {"required": ["sqlite_query"]},
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
                    "evidence": "No tool output evidence cited.",
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
