"""Live integration tests for the Daita eval engine.

Examples:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/test_evals_live.py \
        -m "requires_llm and integration" -v -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from daita.evals import EvalSuite, EvalSuiteConfig
from daita.evals.reporters import render_junit, render_pretty

load_dotenv(Path.cwd() / ".env")


def _require_live_openai() -> None:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live eval integration tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


def _show_report(report) -> None:
    print()
    print(render_pretty(report))
    print()


async def test_eval_plain_live_agent_smoke(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-plain-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_plain_live_agent"
        },
        cases=[
            {
                "id": "simple-answer",
                "prompt": "Reply with this exact token and no explanation: daita-eval-ok",
                "expectations": {
                    "answer": {"contains": ["daita-eval-ok"]},
                    "budgets": {"max_iterations": 1},
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    _show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert (tmp_path / report.run_id / "report.json").exists()


async def test_eval_live_tool_calling_agent(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-tool-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_tool_live_agent"
        },
        cases=[
            {
                "id": "multiply",
                "prompt": (
                    "Use the multiply tool exactly once to compute 17 * 19. "
                    "Then answer with the number 323."
                ),
                "expectations": {
                    "answer": {"contains": ["323"]},
                    "tools": {"required": ["multiply"], "max_calls": 1},
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    _show_report(report)

    assert report.status == "passed", render_pretty(report)


async def test_eval_live_skill_and_plugin_spans(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-skill-plugin-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_skill_plugin_live_agent"
        },
        cases=[
            {
                "id": "skill-plugin-multiply",
                "prompt": (
                    "Use the multiply tool exactly once to compute 12 * 13. "
                    "Answer exactly: product-156"
                ),
                "expectations": {
                    "answer": {"contains": ["product-156"]},
                    "tools": {"required": ["multiply"], "max_calls": 1},
                    "skills": {
                        "required": ["math_reasoning"],
                        "forbidden": ["web_search"],
                        "max_calls": 2,
                        "max_latency_ms": 100,
                        "max_errors": 0,
                    },
                    "plugins": {
                        "required": ["calculator"],
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
    _show_report(report)

    run = report.cases[0].runs[0]
    assert report.status == "passed", render_pretty(report)
    assert any(
        span.kind == "skill" and span.name == "math_reasoning"
        for span in run.execution_spans
    )
    assert any(
        span.kind == "plugin" and span.name == "calculator"
        for span in run.execution_spans
    )
    assert "skills: math_reasoning.plan 35ms" in render_pretty(report)
    assert "plugins: calculator.multiply 12ms" in render_pretty(report)


async def test_eval_live_expected_failure_report_for_skills_and_plugins(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-expected-failure-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_skill_plugin_live_agent"
        },
        cases=[
            {
                "id": "forbidden-execution-path",
                "prompt": (
                    "Use the multiply tool exactly once to compute 5 * 6. "
                    "Answer exactly: product-30"
                ),
                "expectations": {
                    "answer": {"contains": ["product-30"]},
                    "tools": {"required": ["multiply"], "max_calls": 1},
                    "skills": {
                        "forbidden": ["math_reasoning"],
                        "max_latency_ms": 10,
                    },
                    "plugins": {
                        "forbidden": ["calculator"],
                        "max_latency_ms": 5,
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    _show_report(report)
    codes = {failure.code for failure in report.failures}

    assert report.status == "failed"
    assert "forbidden_skill_called" in codes
    assert "forbidden_plugin_called" in codes
    assert "skill_latency_over_budget" in codes
    assert "plugin_latency_over_budget" in codes


async def test_eval_live_sqlite_data_agent(tmp_path):
    _require_live_openai()
    db_path = tmp_path / "sales.db"
    config = EvalSuiteConfig(
        name="live-sqlite-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_sqlite_sales_agent",
            "kwargs": {"db_path": str(db_path)},
        },
        cases=[
            {
                "id": "top-products",
                "prompt": (
                    "Using sqlite_query, what are the top 3 products by total revenue? "
                    "Use SUM(revenue), GROUP BY product, ORDER BY revenue descending, and LIMIT 3. "
                    "Include Widget A's revenue as 12840.50 in the answer."
                ),
                "expectations": {
                    "answer": {
                        "contains": ["Widget A"],
                        "numeric": [
                            {
                                "label": "Widget A",
                                "expected": 12840.50,
                                "tolerance": 0.01,
                            }
                        ],
                    },
                    "tools": {"required": ["sqlite_query"]},
                    "sql": {
                        "read_only": True,
                        "require_limit": True,
                        "must_include": ["SUM", "GROUP BY"],
                        "must_not_include": ["DELETE", "DROP", "SELECT *"],
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    _show_report(report)

    assert report.status == "passed", render_pretty(report)


async def test_eval_live_non_sql_data_operation_inspectors(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-data-ops-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_data_ops_live_agent"
        },
        cases=[
            {
                "id": "data-services",
                "prompt": (
                    "Call these tools exactly once with these arguments: "
                    "file_read(path='data/sales.csv'); "
                    "rest_request(method='GET', url='https://api.example.com/customers'); "
                    "s3_get_object(bucket='analytics', key='sales.csv'); "
                    "vector_search(index='docs', top_k=5, filters={'tenant_id':'acme'}). "
                    "Then summarize briefly."
                ),
                "expectations": {
                    "operations": {
                        "required_categories": ["file", "api", "storage", "vector"],
                        "max_write_operations": 0,
                        "max_delete_operations": 0,
                    },
                    "files": {"required_read": ["sales.csv"]},
                    "api": {
                        "required_methods": ["GET"],
                        "required_hosts": ["api.example.com"],
                        "forbidden_methods": ["POST", "PUT", "PATCH", "DELETE"],
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

    report = await EvalSuite(config).run(output_dir=tmp_path)
    _show_report(report)

    assert report.status == "passed", render_pretty(report)


async def test_eval_live_repeat_run_stability(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-repeat-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_tool_live_agent"
        },
        defaults={"runs": 3},
        cases=[
            {
                "id": "stable-tool-path",
                "prompt": (
                    "Use the multiply tool exactly once to compute 2 * 5. "
                    "Answer exactly: policy-code-10"
                ),
                "expectations": {
                    "answer": {"contains": ["policy-code-10"]},
                    "tools": {"required": ["multiply"], "max_calls": 1},
                    "stability": {
                        "require_same_tools": True,
                        "max_answer_variants": 1,
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    _show_report(report)

    assert report.status == "passed", render_pretty(report)
    case = report.cases[0]
    assert case.runs_requested == 3
    assert case.stability.tool_sequence_variants == 1


async def test_eval_live_multi_case_suite_with_repeated_case(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-multi-case-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_tool_live_agent"
        },
        defaults={"runs": 1},
        cases=[
            {
                "id": "multiply-17-19",
                "prompt": (
                    "Use the multiply tool exactly once to compute 17 * 19. "
                    "Answer exactly: product-323"
                ),
                "expectations": {
                    "answer": {"contains": ["product-323"]},
                    "tools": {"required": ["multiply"], "max_calls": 1},
                },
            },
            {
                "id": "multiply-8-9-stability",
                "runs": 2,
                "prompt": (
                    "Use the multiply tool exactly once to compute 8 * 9. "
                    "Answer exactly: product-72"
                ),
                "expectations": {
                    "answer": {"contains": ["product-72"]},
                    "tools": {"required": ["multiply"], "max_calls": 1},
                    "stability": {
                        "require_same_tools": True,
                        "max_answer_variants": 1,
                    },
                },
            },
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    _show_report(report)

    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == 2
    assert report.summary.runs_total == 3
    assert [case.case_id for case in report.cases] == [
        "multiply-17-19",
        "multiply-8-9-stability",
    ]
    assert report.cases[1].runs_requested == 2
    assert report.cases[1].stability.tool_sequence_variants == 1


async def test_eval_live_dataset_judge_and_baseline_flow(tmp_path):
    _require_live_openai()
    dataset_path = tmp_path / "math-cases.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "dataset-multiply-3-7",
                        "prompt": (
                            "Use the multiply tool exactly once to compute 3 * 7. "
                            "Answer exactly: product-21"
                        ),
                        "expected": {"contains": ["product-21"]},
                    }
                ),
                json.dumps(
                    {
                        "id": "dataset-multiply-4-6",
                        "prompt": (
                            "Use the multiply tool exactly once to compute 4 * 6. "
                            "Answer exactly: product-24"
                        ),
                        "expected": {"contains": ["product-24"]},
                    }
                ),
            ]
        )
    )
    baseline_path = tmp_path / "baseline.json"
    config = EvalSuiteConfig(
        name="live-dataset-judge-baseline-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_tool_live_agent"
        },
        dataset={"path": str(dataset_path)},
        case_template={
            "expectations": {
                "tools": {"required": ["multiply"], "max_calls": 1},
                "judge": {
                    "provider": "mock",
                    "model": "mock-model",
                    "pass_score": 0,
                    "rubric": [
                        "The answer contains the requested product token.",
                        "The agent used the available multiplication tool.",
                    ],
                },
            }
        },
        baselines={
            "fail_on": {
                "new_failures": True,
                "score_drop_gt": 0.01,
                "tool_sequence_changed": True,
            }
        },
    )

    baseline_report = await EvalSuite(config).run(
        output_dir=tmp_path / "baseline-run",
        record_baseline=True,
        baseline_path=baseline_path,
    )
    _show_report(baseline_report)

    report = await EvalSuite(config).run(
        output_dir=tmp_path / "comparison-run",
        compare_baseline=True,
        baseline_path=baseline_path,
    )
    _show_report(report)

    assert baseline_report.status == "passed", render_pretty(baseline_report)
    assert baseline_path.exists()
    assert report.status == "passed", render_pretty(report)
    assert report.summary.cases_total == 2
    assert report.summary.runs_total == 2
    assert report.baseline_comparison is not None
    assert report.baseline_comparison["status"] == "passed"
    assert (Path(report.artifact_path) / "baseline-comparison.json").exists()

    for case in report.cases:
        assertion_codes = {
            assertion.code for run in case.runs for assertion in run.assertions
        }
        assert "judge_passed" in assertion_codes


async def test_eval_live_openai_judge(tmp_path):
    _require_live_openai()
    judge_model = os.environ.get("OPENAI_JUDGE_TEST_MODEL") or os.environ.get(
        "OPENAI_TEST_MODEL", "gpt-5.4-mini"
    )
    config = EvalSuiteConfig(
        name="live-openai-judge-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_tool_live_agent"
        },
        cases=[
            {
                "id": "judge-tool-answer",
                "prompt": (
                    "Use the multiply tool exactly once to compute 6 * 7. "
                    "Answer exactly: product-42"
                ),
                "expectations": {
                    "answer": {"contains": ["product-42"]},
                    "tools": {"required": ["multiply"], "max_calls": 1},
                    "judge": {
                        "provider": "openai",
                        "model": judge_model,
                        "api_key": os.environ["OPENAI_API_KEY"],
                        "pass_score": 0.8,
                        "require_all_criteria_pass": True,
                        "criteria": [
                            {
                                "id": "exact_answer",
                                "description": "The final answer contains the exact token product-42.",
                                "required": True,
                                "weight": 0.4,
                            },
                            {
                                "id": "tool_grounding",
                                "description": "The tool call evidence shows the multiply tool was used.",
                                "required": True,
                                "weight": 0.4,
                            },
                            {
                                "id": "no_extra_claims",
                                "description": "The response does not introduce unrelated facts.",
                                "required": True,
                                "weight": 0.2,
                            },
                        ],
                        "include_tool_outputs": True,
                    },
                },
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    _show_report(report)

    run = report.cases[0].runs[0]
    judge_assertions = [
        assertion for assertion in run.assertions if assertion.id == "judge"
    ]

    assert report.status == "passed", render_pretty(report)
    assert judge_assertions, "Expected the OpenAI judge assertion to run"
    assert judge_assertions[0].code == "judge_passed"
    assert run.judges[0].score >= 0.8
    assert all(criterion.passed for criterion in run.judges[0].criteria)
    print(f"OpenAI judge score: {run.judges[0].score}")
    print(f"OpenAI judge reasoning: {judge_assertions[0].message}")
    for criterion in run.judges[0].criteria:
        print(
            f"Criterion {criterion.id}: passed={criterion.passed} "
            f"score={criterion.score} evidence={criterion.evidence}"
        )


async def test_eval_live_artifact_contract(tmp_path):
    _require_live_openai()
    config = EvalSuiteConfig(
        name="live-artifact-evals",
        agent={
            "factory": "tests.integration.eval_live_factories:create_plain_live_agent"
        },
        cases=[
            {
                "id": "artifact-smoke",
                "prompt": "Reply with this exact token and no explanation: artifact-ok",
                "expectations": {"answer": {"contains": ["artifact-ok"]}},
            }
        ],
    )

    report = await EvalSuite(config).run(output_dir=tmp_path)
    _show_report(report)
    run_dir = tmp_path / report.run_id

    assert report.status == "passed", render_pretty(report)
    assert (run_dir / "report.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "junit.xml").exists()
    assert (run_dir / "cases" / "artifact-smoke" / "case.json").exists()
    assert (run_dir / "cases" / "artifact-smoke" / "run-001.json").exists()
    assert "Daita Eval: live-artifact-evals" in render_pretty(report)
    assert "<testsuite" in render_junit(report)
