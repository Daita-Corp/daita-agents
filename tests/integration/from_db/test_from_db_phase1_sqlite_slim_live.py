"""Live Phase 1 gate for the representative SQLite slim read slice.

Run with ``DAITA_RUN_LIVE_LLM=1`` and ``OPENAI_API_KEY``. Set
``DAITA_SLIM_PHASE1_ARTIFACT_ROOT`` to retain neutral-harness artifacts.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sqlite3

import pytest
from dotenv import load_dotenv

from daita.agents.agent import Agent
from daita.db import DbLLMConfig, DbSourceOptions
from daita.runtime import OperationStatus
from tests.integration.from_db.live_production_helpers import (
    seed_rich_sqlite_schema,
)
from tests.performance.from_db.scale_runner import (
    ScaleBenchmarkParameters,
    default_environment_metadata,
    measure_agent_operation,
    run_scale_benchmark,
)

load_dotenv(Path.cwd() / ".env")

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]

_BASELINE_SHA = "b87df31873d33fffbf50498f5dc4d8892115e8f8"
_FIXTURE_REVISION = f"rich-sqlite-production-contract@{_BASELINE_SHA}"


def _require_live_config() -> DbLLMConfig:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run the Phase 1 live gate")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return DbLLMConfig(
        provider="openai",
        model=os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        api_key=api_key,
        temperature=0,
    )


async def _seed_fixture(path: Path) -> None:
    await seed_rich_sqlite_schema(path)


async def _agent(path: Path, *, name: str):
    return await Agent.from_db(
        str(path),
        name=name,
        llm=_require_live_config(),
        source_options=DbSourceOptions(
            cache_ttl=None,
            read_only=True,
            redact_pii_columns=True,
            query_default_limit=50,
            query_max_rows=100,
            query_max_chars=10_000,
        ),
    )


async def _run_case(agent, prompt: str, *, scenario: str, tmp_path: Path):
    captured = {}
    control_label = os.environ.get("DAITA_SLIM_EXPERIMENT_LABEL", "phase1-provisional")

    def correctness(result, snapshot):
        query_result = _raw_evidence(snapshot, "query.result")
        validation = _raw_evidence(snapshot, "sql.validation")
        return {
            "answer": {
                "passed": result.status is OperationStatus.SUCCEEDED,
                "answer": result.answer,
            },
            "sql": {
                "passed": bool(
                    validation
                    and validation.payload.get("is_read") is True
                    and query_result
                ),
                "safety": "read_only",
            },
        }

    async def operation(_index: int):
        envelope = await measure_agent_operation(
            agent,
            prompt,
            measurement={
                "scenario": scenario,
                "run_id": "run-001",
                "control_label": control_label,
                "provider": "openai",
                "model": _require_live_config().model,
                "model_parameters": {"temperature": 0},
                "database": {
                    "type": "sqlite",
                    "version": sqlite3.sqlite_version,
                    "fixture": (
                        str(Path(agent.runtime.source).name)
                        if isinstance(agent.runtime.source, str)
                        else "phase1.sqlite"
                    ),
                },
                "fixture_revision": _FIXTURE_REVISION,
                "state": "warm",
                "concurrency": 1,
            },
            correctness_evaluator=correctness,
        )
        captured["envelope"] = envelope
        return envelope

    output_root = Path(
        os.environ.get(
            "DAITA_SLIM_ARTIFACT_ROOT",
            os.environ.get(
                "DAITA_SLIM_PHASE1_ARTIFACT_ROOT",
                str(tmp_path / "phase1-neutral-artifacts"),
            ),
        )
    )
    artifact = await run_scale_benchmark(
        suite=f"from-db-{control_label.removesuffix('-provisional')}-sqlite-slim-read",
        parameters=ScaleBenchmarkParameters(
            concurrency=1,
            operations=1,
            scenario=scenario,
        ),
        operation_factory=operation,
        output_dir=output_root,
        artifact_name=f"{scenario}.json",
        environment=default_environment_metadata(
            control_label=control_label,
            database_type="sqlite",
            database_version=sqlite3.sqlite_version,
            dataset="rich_sqlite_production_contract",
            fixture_revision=_FIXTURE_REVISION,
            model=_require_live_config().model,
            provider="openai",
            model_parameters={"temperature": 0},
        ),
    )
    envelope = captured["envelope"]
    return (
        envelope["runtime_result"],
        envelope["operation_snapshot"],
        artifact,
    )


def _raw_evidence(snapshot, kind: str):
    return next((item for item in snapshot.evidence if item.kind == kind), None)


def _task_capabilities(snapshot) -> list[str]:
    return [task.capability_id for task in snapshot.tasks]


def _query_rows(snapshot) -> list[dict[str, object]]:
    evidence = _raw_evidence(snapshot, "query.result")
    return list(evidence.payload.get("rows") or []) if evidence else []


class _RecordingProvider:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.provider_name = delegate.provider_name
        self.model = delegate.model
        self.model_name = delegate.model_name
        self.default_params = delegate.default_params
        self.calls = []

    async def generate(self, messages, tools=None, stream=False, **kwargs):
        self.calls.append(deepcopy(messages))
        return await self.delegate.generate(
            messages,
            tools=tools,
            stream=stream,
            **kwargs,
        )

    def _get_last_token_usage(self):
        return self.delegate._get_last_token_usage()

    def _estimate_cost(self, usage):
        return self.delegate._estimate_cost(usage)

    def get_accumulated_tokens(self):
        return self.delegate.get_accumulated_tokens()

    def get_accumulated_cost(self):
        return self.delegate.get_accumulated_cost()

    async def aclose(self):
        await self.delegate.aclose()


class _FirstInvalidQueryProvider(_RecordingProvider):
    async def generate(self, messages, tools=None, stream=False, **kwargs):
        response = await super().generate(
            messages,
            tools=tools,
            stream=stream,
            **kwargs,
        )
        if len(self.calls) == 1:
            return {
                "tool_calls": [
                    {
                        "id": "forced-invalid-query",
                        "name": "query",
                        "arguments": {
                            "sql": "SELECT missing_customer_name FROM customers",
                            "params": [],
                        },
                    }
                ]
            }
        return response


class _FirstEmailQueryProvider(_RecordingProvider):
    async def generate(self, messages, tools=None, stream=False, **kwargs):
        response = await super().generate(
            messages,
            tools=tools,
            stream=stream,
            **kwargs,
        )
        if len(self.calls) == 1:
            return {
                "tool_calls": [
                    {
                        "id": "forced-email-query",
                        "name": "query",
                        "arguments": {
                            "sql": "SELECT name, email FROM customers ORDER BY name",
                            "params": [],
                        },
                    }
                ]
            }
        return response


async def test_phase1_live_simple_customer_count(tmp_path):
    db_path = tmp_path / "phase1.sqlite"
    await _seed_fixture(db_path)
    agent = await _agent(db_path, name="Phase1SlimSimpleCount")
    try:
        result, snapshot, artifact = await _run_case(
            agent,
            "How many customers are there?",
            scenario="simple-customer-count",
            tmp_path=tmp_path,
        )
    finally:
        await agent.stop()

    operation = artifact["operations"][0]
    assert result.status is OperationStatus.SUCCEEDED
    assert any(4 in row.values() for row in _query_rows(snapshot))
    assert re.search(r"\b4\b", result.answer or "")
    assert _task_capabilities(snapshot) == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert {item.kind for item in snapshot.evidence} == {
        "sql.validation",
        "query.result",
    }
    assert operation["model_call_summary"]["call_count"] == 2
    assert operation["task_count"] == 2
    assert operation["evidence_count"] == 2
    assert operation["catalog"]["task_count"] == 0
    assert operation["repair_count"] == 0
    assert operation["sql"]["fingerprint_preserved"] is True
    assert "planner" not in result.diagnostics


async def test_phase1_live_filtered_read_uses_bound_parameters(tmp_path):
    db_path = tmp_path / "phase1.sqlite"
    await _seed_fixture(db_path)
    agent = await _agent(db_path, name="Phase1SlimFilteredRead")
    try:
        result, snapshot, _artifact = await _run_case(
            agent,
            (
                "Return the names of enterprise customers in region NA. "
                "Use SQLite ? placeholders and typed bound params for both filters."
            ),
            scenario="filtered-read",
            tmp_path=tmp_path,
        )
    finally:
        await agent.stop()

    execute = next(
        task for task in snapshot.tasks if task.capability_id == "db.sql.execute_read"
    )
    assert result.status is OperationStatus.SUCCEEDED
    assert {row.get("name") for row in _query_rows(snapshot)} == {
        "Ada Lovelace",
        "Grace Hopper",
    }
    assert execute.input["params"] == ["enterprise", "NA"] or execute.input[
        "params"
    ] == ["NA", "enterprise"]
    assert len(execute.input.get("param_specs") or ()) == 2


async def test_phase1_live_invalid_sql_repairs_before_connector_io(tmp_path):
    db_path = tmp_path / "phase1.sqlite"
    await _seed_fixture(db_path)
    agent = await _agent(db_path, name="Phase1SlimInvalidRepair")
    provider = _FirstInvalidQueryProvider(agent.runtime.db_llm_service.provider)
    agent.runtime.db_llm_service._provider = provider
    sqlite = next(
        plugin
        for plugin in agent.runtime.config.plugins
        if getattr(getattr(plugin, "manifest", None), "id", None) == "sqlite"
    )
    query_calls = []
    original_query = sqlite.query

    async def spy_query(sql, params=None):
        query_calls.append(sql)
        return await original_query(sql, params)

    sqlite.query = spy_query
    try:
        result, snapshot, artifact = await _run_case(
            agent,
            "Return customer names ordered by name.",
            scenario="invalid-sql-repair",
            tmp_path=tmp_path,
        )
    finally:
        await agent.stop()

    operation = artifact["operations"][0]
    assert result.status is OperationStatus.SUCCEEDED
    assert {row.get("name") for row in _query_rows(snapshot)} == {
        "Ada Lovelace",
        "Grace Hopper",
        "Katherine Johnson",
        "Linus Torvalds",
    }
    assert len(query_calls) == 1
    assert "missing_customer_name" not in query_calls[0]
    assert operation["repair_count"] == 1
    assert result.diagnostics["loop"]["turn_count"] >= 3


async def test_phase1_live_public_and_model_projection_redacts_pii(tmp_path):
    db_path = tmp_path / "phase1.sqlite"
    await _seed_fixture(db_path)
    agent = await _agent(db_path, name="Phase1SlimPublicProjection")
    provider = _FirstEmailQueryProvider(agent.runtime.db_llm_service.provider)
    agent.runtime.db_llm_service._provider = provider
    try:
        result, snapshot, _artifact = await _run_case(
            agent,
            "Query customer names and email addresses, then summarize the rows.",
            scenario="public-projection-redaction",
            tmp_path=tmp_path,
        )
    finally:
        await agent.stop()

    pii_values = {
        "ada@example.com",
        "grace@example.com",
        "katherine@example.com",
        "linus@example.com",
    }
    raw_values = {
        value
        for row in _query_rows(snapshot)
        for value in row.values()
        if isinstance(value, str)
    }
    public_dump = json.dumps(
        {
            "answer": result.answer,
            "evidence": [item.to_dict() for item in result.evidence],
            "diagnostics": result.diagnostics,
        },
        sort_keys=True,
        default=str,
    )
    model_dump = json.dumps(provider.calls, sort_keys=True, default=str)

    assert result.status is OperationStatus.SUCCEEDED
    assert pii_values <= raw_values
    assert not any(value in public_dump for value in pii_values)
    assert not any(value in model_dump for value in pii_values)
    public_query = next(item for item in result.evidence if item.kind == "query.result")
    assert "rows" not in public_query.payload
    assert "sql" not in public_query.payload
