"""Real runtime/count adapter contracts over an independently seeded fixture."""

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject
from daita.adapters.models import SourceRegistration
from daita.catalog.models import CatalogSchemaRequest, ResourceKind
from daita.llm.models import ToolCall
from daita.loop.models import LoopLimits, RunInput
from tests.support.capability_runtime import execute_projected
from tests.support.count_recovery import MANIFEST, TOTALS, seed_sqlite
from tests.support.workspace import workspace_for


def _comparison_selections(sizes=(8, 38)):
    return tuple(
        {
            "source_id": f"source:{i}",
            "source_revision": f"revision:{i}",
            "complete_source_enumeration": True,
            "resources": tuple(
                {
                    "source_id": f"source:{i}",
                    "resource_id": f"resource:{i}:{j}",
                    "revision": f"revision:{i}:{j}",
                    "kind": "table",
                }
                for j in range(size)
            ),
        }
        for i, size in enumerate(sizes)
    )


def test_count_comparison_freezes_complete_coverage_and_total_batch_bounds():
    from daita.domains.data.row_counts import CountComparisonPlan

    plan = CountComparisonPlan.create(_comparison_selections(), batch_width=4)
    assert len(plan.batches) == 12
    assert plan.limits["maximum_batches"] == 16
    assert plan.limits["automatic_source_retries"] == 0
    assert plan.limits["work_units"] == 15
    assert CountComparisonPlan.from_material(plan.material) == plan
    assert (
        CountComparisonPlan.create(
            tuple(reversed(_comparison_selections())), batch_width=4
        ).digest
        == plan.digest
    )
    assert (
        len(
            CountComparisonPlan.create(
                _comparison_selections((1,) * 16), batch_width=4
            ).batches
        )
        == 16
    )
    with pytest.raises(ValueError):
        CountComparisonPlan.from_material({**plan.material, "include_views": True})
    for selections in (
        (),
        _comparison_selections((1,)),
        _comparison_selections((0, 1)),
        _comparison_selections((50, 50)),
        _comparison_selections((1,) * 17),
        (_comparison_selections()[0],) * 2,
        (
            {**_comparison_selections()[0], "complete_source_enumeration": False},
            _comparison_selections()[1],
        ),
    ):
        with pytest.raises(ValueError):
            CountComparisonPlan.create(selections, batch_width=4)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "duplicate",
        "foreign",
        "revision",
        "negative",
        "boolean",
        "subtotal",
        "digest",
        "executor",
        "substituted",
    ],
)
@pytest.mark.parametrize("totals", [(0, 5), (0, 0)])
def test_comparison_reducer_authenticates_values_coverage_and_ties(fault, totals):
    from copy import deepcopy

    from daita.capabilities import ToolOutputValidationError
    from daita.domains.data.row_counts import (
        CountComparisonPlan,
        reduce_count_comparison,
    )
    from daita.jobs.graph.models import canonical_digest

    plan = CountComparisonPlan.create(_comparison_selections((1, 1)), batch_width=4)
    mapping = tuple(
        {
            "batch_id": f"batch:{i + 1}",
            "task_id": f"task:{i}",
            "task_spec_digest": "sha256:" + str(i) * 64,
            "schema_digest": "sha256:" + "c" * 64,
            "contract_digest": "sha256:" + "d" * 64,
        }
        for i in range(2)
    )
    results: list[dict[str, Any]] = []
    for i, total in enumerate(totals):
        payload = {
            "source_id": f"source:{i}",
            "source_revision": f"revision:{i}",
            "counts": (
                {
                    "resource_id": f"resource:{i}:0",
                    "revision": f"revision:{i}:0",
                    "rows": total,
                },
            ),
            "total_rows": total,
            "counted_resources": 1,
            "requested_resources": 1,
            "coverage": "complete_requested_batch",
            "snapshot_scope": "one_source_one_statement",
            "observed_at": "2026-10-14T00:00:00+00:00",
            "duration_ms": 0,
            "source_name": f"Source {i}",
            "adapter_id": "sqlite",
            "scope_limitation": "Admitted tables only",
            "sql_fingerprint": "sha256:" + "0" * 64,
            "trust_classification": "untrusted_external_data",
        }
        material = {
            "agent_id": "agent:test",
            "job_id": "job:test",
            "task_id": f"task:{i}",
            "attempt_id": f"attempt:{i}",
            "run_id": f"run:{i}",
            "result_id": f"result:attempt:{i}",
            "result_kind": "data.row_counts",
            "schema_digest": mapping[i]["schema_digest"],
            "payload": payload,
            "summary": "Count observation.",
            "sensitivity": "restricted",
            "provenance": {
                "authority": "authenticated_internal_capability_output",
                "plan_digest": plan.digest,
                "batch_id": mapping[i]["batch_id"],
                "task_spec_digest": mapping[i]["task_spec_digest"],
            },
            "verification": {
                "capability_id": "data.count_batch",
                "executor_id": "data.count_rows.executor",
                "contract_digest": mapping[i]["contract_digest"],
                "output_sha256": canonical_digest(payload),
            },
            "artifact_ids": (),
            "effect_receipt_ids": (),
            "residual_risk": None,
            "downstream_constraints": {},
            "completed_at": "2026-10-14T00:00:00+00:00",
        }
        results.append({**material, "result_digest": canonical_digest(material)})
    if fault == "missing":
        results.pop()
    elif fault == "duplicate":
        results.append(deepcopy(results[0]))
    elif fault:
        item = results[0]
        if fault == "foreign":
            item["job_id"] = "job:other"
        elif fault == "substituted":
            item["task_id"] = "task:other"
        elif fault == "executor":
            item["verification"]["executor_id"] = "data.query.executor"
        elif fault == "revision":
            item["payload"]["counts"][0]["revision"] = "revision:other"
        elif fault == "boolean":
            item["payload"]["counts"][0]["rows"] = True
        elif fault == "negative":
            item["payload"]["counts"][0]["rows"] = -1
        elif fault == "subtotal":
            item["payload"]["total_rows"] = 10
        item["verification"]["output_sha256"] = canonical_digest(item["payload"])
        item["result_digest"] = canonical_digest(
            {k: v for k, v in item.items() if k != "result_digest"}
        )
        if fault == "digest":
            item["result_digest"] = "sha256:" + "0" * 64
    frozen_results = tuple(FrozenJsonObject.from_mapping(r) for r in reversed(results))
    kwargs: dict[str, Any] = {
        "job_id": "job:test",
        "agent_id": "agent:test",
        "batch_tasks": mapping,
        "results": frozen_results,
    }
    if fault:
        with pytest.raises((ToolOutputValidationError, ValueError)):
            reduce_count_comparison(plan, **kwargs)
    else:
        result = reduce_count_comparison(plan, **kwargs)
        assert [s["total_rows"] for s in cast(Any, result["sources"])] == list(totals)
        assert result["winner_source_ids"] == (
            ("source:1",) if totals[1] else ("source:0", "source:1")
        )
        empty = reduce_count_comparison(
            plan, **{**kwargs, "results": ()}, complete=False
        )
        assert empty["winner_source_ids"] == ()
        assert all(
            s["total_rows"] is None
            and s["observed_total_rows"] == 0
            and s["counted_resources"] == 0
            for s in cast(Any, empty["sources"])
        )


def _results(request):
    from daita.llm.models import ToolResultBlock

    return {
        block.call_id: block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def _cursor_model(respond):
    from daita.llm.providers.mock import MockModelProvider

    class CursorModel(MockModelProvider):
        async def generate(self, request):
            self._requests.append(request)
            return await respond(request)

    return CursorModel(())


@pytest.mark.parametrize("include_views", [False, True])
@pytest.mark.parametrize("mixed_arguments", [False, True])
async def test_count_operation_executes_fixed_selection_without_model_page_turns(
    tmp_path, include_views, mixed_arguments
):
    from daita.llm.models import FinishReason, ModelResponse
    from daita.loop.models import validate_completed_transcript

    sources: dict[str, SourceRegistration] = {}

    async def respond(request):
        results = _results(request)
        if not results:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "load",
                        "toolbox_load",
                        {"tool_names": ("data_count_selection", "data_count_inspect")},
                    ),
                    ToolCall(
                        "inspect-contract",
                        "toolbox_inspect",
                        {"tool_name": "data_count_selection"},
                    ),
                    ToolCall(
                        "inspect-plan",
                        "toolbox_inspect",
                        {"tool_name": "data_count_plan"},
                    ),
                    *(
                        ToolCall(
                            name,
                            "catalog_schema",
                            {"source_id": source.id, "detail": "identity", "limit": 50},
                        )
                        for name, source in sources.items()
                    ),
                ),
            )
        if len(model.requests) == 2:
            definition = next(
                t for t in request.tools if t.name == "data_count_selection"
            )
            assert set(definition.input_schema["properties"]) == {
                "selection_ref",
                "include_views",
                "source_id",
                "coverage",
            }
            assert "data_count_plan" not in {t.name for t in request.tools}
            assert results["inspect-plan"].is_error
            from daita._json import canonical_json

            contract = results["inspect-contract"].output["data"]["value"]
            assert canonical_json(contract["input_schema"]) == canonical_json(
                definition.input_schema
            )
            from hashlib import sha256

            assert (
                contract["input_schema_digest"]
                == "sha256:"
                + sha256(canonical_json(definition.input_schema).encode()).hexdigest()
            )
            loaded = next(
                item
                for item in results["load"].output["data"]["contracts"]
                if item["tool_name"] == "data_count_selection"
            )
            assert loaded["input_schema_digest"] == contract["input_schema_digest"]
            assert loaded["contract_digest"] == contract["contract_digest"]
            if mixed_arguments:
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=tuple(
                        ToolCall(
                            name + "-mixed",
                            "data_count_selection",
                            {
                                "selection_ref": results[name].output["data"][
                                    "selection"
                                ]["count_selection_ref"],
                                "source_id": source.id,
                            },
                        )
                        for name, source in sources.items()
                    ),
                )
        operation_step = 3 if mixed_arguments else 2
        if len(model.requests) == operation_step:
            if mixed_arguments:
                for name in sources:
                    error = results[name + "-mixed"].output["error"]
                    assert results[name + "-mixed"].is_error
                    assert error["code"] == "invalid_argument_value"
                    assert error["details"]["constraint"] == "oneOf"
                    recovery = error["details"]["recovery"]
                    assert recovery["kind"] == "correct_arguments"
                    assert set(recovery["allowed_fields"]) == {
                        "selection_ref",
                        "include_views",
                        "source_id",
                        "coverage",
                    }
                    assert error["details"]["execution_disposition"] == "not_dispatched"
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=tuple(
                    ToolCall(
                        name + "-operation",
                        "data_count_selection",
                        {
                            "selection_ref": results[name].output["data"]["selection"][
                                "count_selection_ref"
                            ],
                            "include_views": include_views,
                        },
                    )
                    for name in sources
                ),
            )
        if len(model.requests) == operation_step + 1:
            for name in sources:
                result = results[name + "-operation"]
                assert not result.is_error, result.output
                data = result.output["data"]
                expected = TOTALS[name] + (
                    int(MANIFEST[name][2]["rows"]) if include_views else 0
                )
                assert data["status"] == "completed"
                assert data["total_rows"] == data["observed_total_rows"] == expected
                assert (
                    data["counted_resources"]
                    == data["selected_resources"]
                    == len(MANIFEST[name]) + include_views
                )
                assert data["unexecuted_resources"] == data["failed_resources"] == 0
                assert data["snapshot_scope"] == "separate_batch_snapshots"
                assert data["coverage"] == (
                    "complete_catalog_selection"
                    if include_views
                    else "complete_admitted_base_tables"
                )
                assert "counts" not in data
            if mixed_arguments:
                return ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="Exact count operation completed.",
                )
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=tuple(
                    ToolCall(
                        name + "-inspect",
                        "data_count_inspect",
                        {
                            "operation_ref": results[name + "-operation"].output[
                                "data"
                            ]["operation_ref"],
                            "limit": 50,
                        },
                    )
                    for name in sources
                ),
            )
        for name in sources:
            result = results[name + "-inspect"]
            assert not result.is_error, result.output
            assert (
                len(result.output["data"]["counts"])
                == len(MANIFEST[name]) + include_views
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP, text="Exact count operation completed."
        )

    model = _cursor_model(respond)
    agent = await Agent.create(
        "count-operation",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        workspace=workspace_for(tmp_path),
        limits=LoopLimits(max_steps=4),
    )
    try:
        for name, path in seed_sqlite(tmp_path).items():
            sources[name] = await agent.attach(SQLiteSource(path, name=name))
        result = await agent.run(
            "Compare admitted source row totals",
            source_scope_ids=tuple(s.id for s in sources.values()),
        )
        assert result.reason == "completed", (result.reason, result.steps)
        assert len(model.requests) == 4
        transcript = await agent.transcript(result.run_id)
        validate_completed_transcript(transcript, result)
        evidence = await agent.analysis_evidence(result.run_id)
        children = [r for r in evidence if r.kind == "child"]
        assert len(children) == (13 if include_views else 12)
        assert all(
            r.facts["status"] == "succeeded"
            and len(cast(Any, r.facts["arguments"])["resource_ids"]) <= 4
            for r in children
        )
        actual = {
            row["resource_id"]: row["rows"]
            for r in children
            for row in cast(Any, r.facts["result"])["data"]["counts"]
        }
        assert len(actual) == 46 + 2 * include_views
        for name, source in sources.items():
            for resource in await agent.list_catalog_resources(source_id=source.id):
                if resource.kind.value == "view":
                    if include_views:
                        assert actual[resource.id] == int(MANIFEST[name][2]["rows"])
                    continue
                expected = next(
                    t["rows"] for t in MANIFEST[name] if t["table"] == resource.name
                )
                assert actual[resource.id] == expected
        assert await agent.list_effects() == ()
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "fault,authoring",
    [
        (fault, authoring)
        for fault in [
            "unknown",
            "truncated",
            "revocation",
            "catalog_drift",
            "queued_revocation",
            "queued_catalog_drift",
            "native_drift",
            "timeout",
            "deadline",
            "run_budget",
        ]
        for authoring in ["reference", "whole_source"]
        if authoring == "reference" or fault not in {"unknown", "truncated"}
    ],
)
async def test_count_operation_retains_fixed_coverage_and_honest_failures(
    tmp_path, monkeypatch, fault, authoring
):
    import asyncio
    import sqlite3
    from dataclasses import replace

    from daita.adapters.sqlite_query import SQLiteQueryBackend, SQLiteQueryError
    from daita.domains.data import capabilities as data_capabilities
    from daita.domains.data.controller import DataCapabilityDomain
    from daita.llm.models import FinishReason, ModelResponse
    from daita.loop.models import LoopLimits, validate_completed_transcript
    from daita.storage.sqlite_records import SourceReadMode, SourceReadScope

    source = None
    child_bindings = 0
    dispatches = []
    original_bind = DataCapabilityDomain.prepare_call
    original_read = SQLiteQueryBackend.execute_read
    paths = seed_sqlite(tmp_path)

    async def bind(self, run, call, capability, arguments, **kwargs):
        nonlocal child_bindings
        assert source is not None
        if (
            capability.id == "data.count_rows"
            and "_count_operation_binding" in arguments
        ):
            child_bindings += 1
            if child_bindings == 2:
                if fault == "revocation":
                    await agent._embedded._store.replace_source_permission_scopes(
                        SourceReadScope(
                            agent_id=agent.id,
                            source_id=source.id,
                            mode=SourceReadMode.SELECTED,
                            resource_ids=(
                                arguments["_count_operation_binding"]["resources"][0][
                                    "resource_id"
                                ],
                            ),
                        ),
                        (),
                    )
                if fault == "catalog_drift":
                    with sqlite3.connect(paths["Juniper"]) as connection:
                        connection.execute(
                            'ALTER TABLE "records_00" ADD COLUMN drift TEXT'
                        )
                    await agent.refresh_source(source.id)
                if fault == "native_drift":
                    resource = next(
                        r
                        for r in await agent.list_catalog_resources(source_id=source.id)
                        if r.id == arguments["resource_ids"][0]
                    )
                    with sqlite3.connect(paths["Juniper"]) as connection:
                        connection.execute(
                            'ALTER TABLE "'
                            + resource.name.replace('"', '""')
                            + '" ADD COLUMN drift TEXT'
                        )
        return await original_bind(self, run, call, capability, arguments, **kwargs)

    async def read(self, **kwargs):
        dispatches.append(kwargs)
        if len(dispatches) == 2 and fault == "timeout":
            raise SQLiteQueryError("sqlite_query_timeout", "Controlled read timeout.")
        if len(dispatches) == 2 and fault == "deadline":
            await asyncio.Event().wait()
        result = await original_read(self, **kwargs)
        if len(dispatches) == 1 and fault.startswith("queued_"):
            assert source is not None
            if fault == "queued_revocation":
                await agent._embedded._store.replace_source_permission_scopes(
                    SourceReadScope(
                        agent_id=agent.id,
                        source_id=source.id,
                        mode=SourceReadMode.SELECTED,
                        resource_ids=tuple(
                            pair[0] for pair in kwargs["expected_resource_revisions"]
                        ),
                    ),
                    (),
                )
            else:
                with sqlite3.connect(paths["Juniper"]) as connection:
                    connection.execute('ALTER TABLE "records_00" ADD COLUMN drift TEXT')
                await agent.refresh_source(source.id)
        return result

    monkeypatch.setattr(DataCapabilityDomain, "prepare_call", bind)
    monkeypatch.setattr(
        DataCapabilityDomain, "prepare_session_call", bind, raising=False
    )
    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", read)
    if fault == "deadline":
        monkeypatch.setattr(
            data_capabilities,
            "COUNT_OPERATION_POLICY",
            replace(
                data_capabilities.COUNT_OPERATION_POLICY, inline_max_wall_seconds=1
            ),
        )

    async def respond(request):
        results = _results(request)
        assert source is not None
        if not results:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "load",
                        "toolbox_load",
                        {"tool_names": ("data_count_selection",)},
                    ),
                    ToolCall(
                        "catalog",
                        "catalog_schema",
                        {
                            "source_id": source.id,
                            "detail": "identity",
                            "limit": 6 if fault == "truncated" else 50,
                        },
                    ),
                ),
            )
        if len(model.requests) == 2:
            ref = (
                "sel_0000000000000000"
                if fault == "unknown"
                else results["catalog"].output["data"]["selection"][
                    "count_selection_ref"
                ]
            )
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "operation",
                        "data_count_selection",
                        (
                            {"selection_ref": ref}
                            if authoring == "reference"
                            else {
                                "source_id": source.id,
                                "coverage": "complete_admitted_base_tables",
                            }
                        ),
                    ),
                ),
            )
        assert (
            fault != "run_budget"
        ), "No model request is permitted after tool-budget exhaustion"
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text="Retained exact evidence and honest fixed-selection coverage.",
        )

    model = _cursor_model(respond)
    limits = (
        LoopLimits(
            max_parallel_reads_per_source=1,
            max_tool_calls_per_run=4,
            max_tool_calls_per_response=4,
        )
        if fault == "run_budget"
        else LoopLimits(
            max_parallel_reads_per_source=4 if fault.startswith("queued_") else 1
        )
    )
    agent = await Agent.create(
        "count-failure",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        limits=limits,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(SQLiteSource(paths["Juniper"], name="Juniper"))
        result = await agent.run(
            "Count admitted table rows", source_scope_ids=(source.id,)
        )
        transcript = await agent.transcript(result.run_id)
        if fault != "run_budget":
            validate_completed_transcript(transcript, result)
        output = next(r for c, r in transcript.tool_pairs if c.id == "operation")
        assert output is not None
        if fault == "unknown":
            assert output.is_error and not dispatches
            assert (
                cast(Any, output.output)["error"]["details"]["execution_disposition"]
                == "not_dispatched"
            )
            return
        assert not output.is_error, output.output
        data = cast(Any, output.output["data"])
        records = [
            r for r in await agent.analysis_evidence(result.run_id) if r.kind == "child"
        ]
        success = [
            cast(Any, r.facts)["result"]["data"]
            for r in records
            if r.facts["status"] == "succeeded"
        ]
        assert data["observed_total_rows"] == sum(
            row["rows"] for page in success for row in page["counts"]
        )
        assert data["selected_resources"] == (5 if fault == "truncated" else 8)
        if fault == "truncated":
            assert data["status"] == "completed"
            assert data["coverage"] == "complete_catalog_selection"
            assert data["catalog_enumeration_complete"] is False
            assert data["counted_resources"] == 5
        else:
            assert data["total_rows"] is None
            assert data["coverage"] == "partial_catalog_selection"
            assert data["counted_resources"] == (0 if fault == "run_budget" else 4)
            assert (
                data["failed_resources"]
                + data["interrupted_resources"]
                + data["unexecuted_resources"]
                == 8 - data["counted_resources"]
            )
            assert data["failures"]
        assert len(dispatches) == (
            0
            if fault == "run_budget"
            else (
                1
                if fault
                in {
                    "revocation",
                    "catalog_drift",
                    "queued_revocation",
                    "queued_catalog_drift",
                }
                else 2
            )
        )
        if fault == "run_budget":
            assert result.reason == "tool_calls_per_run_exceeded"
            assert len(model.requests) == 2
        if fault == "deadline":
            assert data["deadline_reached"] is True
            assert data["interrupted_resources"] == 4
        assert await agent.list_effects() == ()
    finally:
        await agent.close()


@pytest.mark.parametrize("include_views", [False, True])
async def test_count_selection_cursors_own_batches_totals_and_coverage(
    tmp_path, include_views
):
    from daita.llm.models import FinishReason, ModelResponse
    from daita.loop.models import validate_completed_transcript

    sources: dict[str, SourceRegistration] = {}
    completed = {}
    observed = {}

    async def respond(request):
        results = _results(request)
        if not results:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "load", "toolbox_load", {"tool_names": ("data_count_rows",)}
                    ),
                    *(
                        ToolCall(
                            name,
                            "catalog_schema",
                            {"source_id": source.id, "detail": "identity", "limit": 50},
                        )
                        for name, source in sources.items()
                    ),
                ),
            )
        calls = []
        for name in sources:
            previous = next(
                (
                    r
                    for key, r in reversed(tuple(results.items()))
                    if key.startswith(name + "-count-")
                ),
                None,
            )
            if previous is not None:
                assert not previous.is_error, previous.output
                data = previous.output["data"]
                for row in data["counts"]:
                    observed[row["resource_id"]] = row["rows"]
                assert data["progress"]["total_rows"] >= 0
                assert data["counted_resources"] <= 4
                if data["next_cursor"] is None:
                    completed[name] = data
                    continue
                arguments = {"cursor": data["next_cursor"]}
            else:
                selection = results[name].output["data"]["selection"]
                assert selection["complete_source_enumeration"] is True
                assert selection["base_tables"] == len(MANIFEST[name])
                arguments = {"selection_ref": selection["count_selection_ref"]}
                if include_views:
                    arguments["include_views"] = True
            calls.append(
                ToolCall(
                    f"{name}-count-{len(model.requests)}", "data_count_rows", arguments
                )
            )
        if calls:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS, tool_calls=tuple(calls)
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            text="Counts complete from bounded framework selections.",
        )

    model = _cursor_model(respond)
    agent = await Agent.create(
        "cursor-counts",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        for name, path in seed_sqlite(tmp_path).items():
            sources[name] = await agent.attach(SQLiteSource(path, name=name))
        result = await agent.run(
            "Compare admitted source row totals",
            source_scope_ids=tuple(s.id for s in sources.values()),
        )
        assert result.reason == "completed", (
            result.reason,
            result.steps,
            result.usage.input_tokens,
            result.usage.output_tokens,
            len(model.requests),
        )
        for name, data in completed.items():
            assert data["source_name"] == name
            progress = data["progress"]
            assert progress["total_rows"] == TOTALS[name] + (
                int(MANIFEST[name][2]["rows"]) if include_views else 0
            )
            assert (
                progress["counted_resources"]
                == progress["selected_resources"]
                == len(MANIFEST[name]) + include_views
            )
            assert progress["remaining_resources"] == 0
            assert progress["coverage"] == (
                "complete_catalog_selection"
                if include_views
                else "complete_admitted_base_tables"
            )
            assert progress["snapshot_scope"] == "separate_batch_snapshots"
        assert set(completed) == set(TOTALS) and len(observed) == 46 + 2 * include_views
        transcript = await agent.transcript(result.run_id)
        validate_completed_transcript(transcript, result)
        assert all(
            set(c.arguments) <= {"selection_ref", "cursor", "include_views"}
            for c, _ in transcript.tool_pairs
            if c.name == "data_count_rows"
        )
        assert await agent.list_effects() == ()
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "mixed",
        "cross_selection",
        "revocation",
        "catalog_drift",
        "native_drift",
        "truncated",
        "repeat",
        "data_changed",
        "unknown_cursor",
        "tampered_result",
        "failed_page",
    ],
)
async def test_count_cursor_failures_and_repeated_reads_preserve_selection(
    tmp_path, monkeypatch, fault
):
    import sqlite3

    from daita.adapters.sqlite_query import SQLiteQueryBackend
    from daita.llm.models import FinishReason, ModelResponse
    from daita.storage.sqlite_records import SourceReadMode, SourceReadScope

    source = None
    invocations = []
    original = SQLiteQueryBackend.execute_read

    async def recorded(self, **kwargs):
        invocations.append(kwargs)
        if fault == "failed_page" and len(invocations) == 2:
            from daita.adapters.sqlite_query import SQLiteQueryError

            raise SQLiteQueryError("sqlite_query_timeout", "Read timed out.")
        return await original(self, **kwargs)

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", recorded)

    async def respond(request):
        results = _results(request)
        assert source is not None
        step = len(model.requests)
        if step == 1:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "load", "toolbox_load", {"tool_names": ("data_count_rows",)}
                    ),
                    ToolCall(
                        "catalog",
                        "catalog_schema",
                        {
                            "source_id": source.id,
                            "detail": "identity",
                            "limit": 6 if fault == "truncated" else 50,
                        },
                    ),
                    ToolCall(
                        "other",
                        "catalog_schema",
                        {"source_id": source.id, "detail": "identity", "limit": 1},
                    ),
                ),
            )
        ref = results["catalog"].output["data"]["selection"]["count_selection_ref"]
        if step == 2:
            args = {
                "selection_ref": "sel_0000000000000000" if fault == "unknown" else ref
            }
            if fault == "mixed":
                args["source_id"] = source.id
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("first", "data_count_rows", args),),
            )
        if fault in {"unknown", "mixed"}:
            assert results["first"].is_error and not invocations
            return ModelResponse(
                finish_reason=FinishReason.STOP, text="Invalid reference rejected."
            )
        first = results["first"]
        assert not first.is_error, first.output
        if step == 3:
            cursor = first.output["data"]["next_cursor"]
            assert cursor
            if fault == "revocation":
                await agent._embedded._store.replace_source_permission_scopes(
                    SourceReadScope(
                        agent_id=agent.id,
                        source_id=source.id,
                        mode=SourceReadMode.SELECTED,
                        resource_ids=(
                            first.output["data"]["counts"][0]["resource_id"],
                        ),
                    ),
                    (),
                )
            if fault in {"catalog_drift", "native_drift"}:
                with sqlite3.connect(paths["Juniper"]) as connection:
                    connection.execute(
                        "ALTER TABLE records_00 ADD COLUMN changed INTEGER"
                    )
                if fault == "catalog_drift":
                    await agent.refresh_source(source.id)
            args = {"cursor": cursor}
            if fault == "unknown_cursor":
                args["cursor"] = "cnt_0000000000000000"
            if fault == "data_changed":
                with sqlite3.connect(paths["Juniper"]) as connection:
                    connection.execute("INSERT INTO records_07 DEFAULT VALUES")
            if fault == "tampered_result":
                from dataclasses import replace

                from daita.llm.models import ToolResultBlock

                domain = agent._embedded._capability_runtime._domains["data"]
                load_transcript = domain._transcript_loader

                async def tampered(run_id):
                    transcript = await load_transcript(run_id)
                    return replace(
                        transcript,
                        messages=tuple(
                            replace(
                                message,
                                content=tuple(
                                    (
                                        replace(block, executor_id="forged.executor")
                                        if isinstance(block, ToolResultBlock)
                                        and block.call_id == "first"
                                        else block
                                    )
                                    for block in message.content
                                ),
                            )
                            for message in transcript.messages
                        ),
                    )

                monkeypatch.setattr(domain, "_transcript_loader", tampered)
            if fault == "cross_selection":
                args["selection_ref"] = results["other"].output["data"]["selection"][
                    "count_selection_ref"
                ]
            calls = (ToolCall("next", "data_count_rows", args),)
            if fault in {"repeat", "data_changed"}:
                calls += (ToolCall("repeat", "data_count_rows", args),)
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls
            )
        next_result = results["next"]
        if fault == "failed_page" and step == 4:
            assert next_result.is_error and "data" not in next_result.output
            assert len(invocations) == 2  # No automatic replay or skipped page.
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "recovered",
                        "data_count_rows",
                        {"cursor": first.output["data"]["next_cursor"]},
                    ),
                ),
            )
        if fault == "failed_page":
            next_result = results["recovered"]
            assert not next_result.is_error, next_result.output
            assert (
                next_result.output["data"]["progress"]["total_rows"]
                == TOTALS["Juniper"]
            )
            assert next_result.output["data"]["progress"]["counted_resources"] == 8
            assert len(invocations) == 3
        elif fault in {
            "cross_selection",
            "revocation",
            "catalog_drift",
            "native_drift",
            "unknown_cursor",
            "tampered_result",
        }:
            assert next_result.is_error and "data" not in next_result.output
            assert len(invocations) == (2 if fault == "native_drift" else 1)
        else:
            assert not next_result.is_error, next_result.output
            data = next_result.output["data"]
            assert data["next_cursor"] is None
            assert data["progress"]["remaining_resources"] == 0
            if fault in {"repeat", "data_changed"}:
                repeated = results["repeat"].output["data"]
                assert (
                    data["progress"]["total_rows"]
                    == repeated["progress"]["total_rows"]
                    == TOTALS["Juniper"] + (fault == "data_changed")
                )
                assert data["progress"]["counted_resources"] == 8
                assert len(invocations) == 3
            else:
                assert data["progress"]["coverage"] == "complete_catalog_selection"
                assert data["progress"]["catalog_enumeration_complete"] is False
                assert data["progress"]["selected_resources"] < 8
        return ModelResponse(
            finish_reason=FinishReason.STOP, text="Bounded count evidence retained."
        )

    model = _cursor_model(respond)
    agent = await Agent.create(
        "cursor-boundaries",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        paths = seed_sqlite(tmp_path)
        source = await agent.attach(SQLiteSource(paths["Juniper"], name="Juniper"))
        result = await agent.run("Count admitted tables", source_scope_ids=(source.id,))
        assert result.reason == "completed"
    finally:
        await agent.close()


@pytest.mark.parametrize("limit", ["steps", "tokens", "cross_run"])
@pytest.mark.parametrize("count_tool", ["data_count_rows", "data_count_selection"])
async def test_count_cursor_references_do_not_extend_run_or_budgets(
    tmp_path, limit, count_tool
):
    from daita import AgentConfig
    from daita.llm.models import FinishReason, ModelResponse, ModelUsage
    from daita.loop.models import LoopLimits

    source = None
    saved: dict[str, Any] = {}

    async def respond(request):
        results = _results(request)
        assert source is not None
        if "load" not in results:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("load", "toolbox_load", {"tool_names": (count_tool,)}),
                    ToolCall(
                        "catalog",
                        "catalog_schema",
                        {"detail": "identity", "source_id": source.id, "limit": 50},
                    ),
                ),
            )
        ref = results["catalog"].output["data"]["selection"]["count_selection_ref"]
        if "page" not in results:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("page", count_tool, {"selection_ref": ref}),),
                usage=ModelUsage(
                    input_tokens=100_000 if limit == "tokens" else 1, output_tokens=1
                ),
            )
        saved.update(results["page"].output["data"])
        return ModelResponse(
            finish_reason=FinishReason.STOP, text="Partial count evidence retained."
        )

    model = _cursor_model(respond)
    agent = await Agent.create(
        "cursor-limits",
        root=tmp_path,
        model=model,
        model_profile=model.model_profile,
        config=AgentConfig(limits=LoopLimits(max_steps=2 if limit == "steps" else 24)),
        workspace=workspace_for(tmp_path),
    )
    try:
        paths = seed_sqlite(tmp_path)
        source = await agent.attach(SQLiteSource(paths["Juniper"]))
        result = await agent.run("Count tables", source_scope_ids=(source.id,))
        transcript = await agent.transcript(result.run_id)
        pages = [(c, r) for c, r in transcript.tool_pairs if c.name == count_tool]
        if pages:
            assert pages[0][1] is not None
        if limit == "steps":
            assert result.reason == "step_limit_reached" and len(model.requests) == 2
            assert len(pages) == 1
            page_result = pages[0][1]
            assert page_result is not None and not page_result.is_error
            page_data = page_result.output["data"]
            assert isinstance(page_data, FrozenJsonObject)
            progress = (
                page_data["progress"] if count_tool == "data_count_rows" else page_data
            )
            assert isinstance(progress, FrozenJsonObject)
            assert progress["counted_resources"] == (
                4 if count_tool == "data_count_rows" else 8
            )
        elif limit == "tokens":
            assert result.reason == "token_limit_reached" and len(model.requests) == 2
            assert result.usage.total_tokens == 100_001
            # The requested call still receives its ordered budget result; no
            # count evidence or source dispatch occurs after returned exhaustion.
            assert len(pages) == 1
            page_result = pages[0][1]
            assert page_result is not None and page_result.is_error
            assert "data" not in page_result.output
        else:
            assert result.reason == "completed"
            reference_arguments = (
                {"cursor": saved["next_cursor"]}
                if count_tool == "data_count_rows"
                else {"selection_ref": saved["selection_ref"]}
            )

            async def next_run(request):
                results = _results(request)
                if not results:
                    return ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS,
                        tool_calls=(
                            ToolCall(
                                "load-next",
                                "toolbox_load",
                                {"tool_names": (count_tool,)},
                            ),
                        ),
                    )
                if "cross-run" not in results:
                    return ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS,
                        tool_calls=(
                            ToolCall(
                                "cross-run",
                                count_tool,
                                reference_arguments,
                            ),
                        ),
                    )
                assert results["cross-run"].is_error
                assert (
                    results["cross-run"].output["error"]["code"]
                    == "count_selection_invalid"
                )
                return ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="A fresh run needs a fresh catalog selection.",
                )

            # The same provider remains installed; its next journey has no access
            # to another run's count references, even on the same agent/source.
            respond = next_run
            second = await agent.run(
                "Continue prior count", source_scope_ids=(source.id,)
            )
            assert second.reason == "completed"
    finally:
        await agent.close()


async def test_count_batches_are_exact_revision_bound_and_compact(tmp_path):
    paths = seed_sqlite(tmp_path)
    agent = await Agent.create(
        "counts", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        for name, path in paths.items():
            source = await agent.attach(SQLiteSource(path, name=name))
            resources = [
                r
                for r in await agent.list_catalog_resources(source_id=source.id)
                if r.kind is ResourceKind.TABLE
            ]
            expected = {f'main.{t["table"]}': t["rows"] for t in MANIFEST[name]}
            run = RunInput(
                "run-" + name,
                agent.id,
                "count tables",
                datetime.now(UTC),
                source_scope_ids=(source.id,),
            )
            observed = {}
            for offset in range(0, len(resources), 4):
                batch = resources[offset : offset + 4]
                call = ToolCall(
                    "count-" + str(offset),
                    "data_count_rows",
                    {
                        "source_id": source.id,
                        "resource_ids": tuple(r.id for r in batch),
                    },
                )
                result = (
                    await execute_projected(
                        agent._embedded._capability_runtime, run, (call,)
                    )
                ).ordered_results[0]
                assert not result.is_error, result.output
                data = result.output["data"]
                assert data["source_name"] == name
                assert data["coverage"] == "complete_requested_batch"
                assert data["total_rows"] == sum(
                    int(expected[r.native_identity]) for r in batch
                )
                assert data["observed_at"] and data["duration_ms"] >= 0
                assert data["snapshot_scope"] == "one_source_one_statement"
                for item in data["counts"]:
                    r = next(r for r in batch if r.id == item["resource_id"])
                    assert item["revision"] == r.current_revision
                    observed[r.native_identity] = item["rows"]
            assert observed == expected and sum(observed.values()) == TOTALS[name]
            full = await agent._embedded._catalog_service.schema_slice(
                CatalogSchemaRequest(
                    agent.id, resource_ids=tuple(r.id for r in resources), limit=50
                )
            )
            compact = await agent._embedded._catalog_service.schema_slice(
                CatalogSchemaRequest(
                    agent.id,
                    resource_ids=tuple(r.id for r in resources),
                    limit=50,
                    detail="identity",
                )
            )
            from daita._json import canonical_json

            assert len(canonical_json(compact).encode()) <= 0.4 * len(
                canonical_json(full).encode()
            )
    finally:
        await agent.close()


async def test_public_run_warns_on_repeated_failure_and_preserves_correction(tmp_path):
    from daita.llm.models import (
        FinishReason,
        MessageRole,
        ModelResponse,
        TextBlock,
        ToolResultBlock,
    )
    from daita.llm.providers.mock import MockModelProvider

    paths = seed_sqlite(tmp_path)
    agent = await Agent.create(
        "count-progress", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    source = await agent.attach(SQLiteSource(paths["Juniper"]))
    ids = tuple(
        r.id
        for r in await agent.list_catalog_resources(source_id=source.id)
        if r.kind is ResourceKind.TABLE
    )
    await agent.close()
    provider = MockModelProvider(
        [
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "load", "toolbox_load", {"tool_names": ("data_count_rows",)}
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "bad-1",
                        "data_count_rows",
                        {"source_id": source.id, "resource_ids": ids},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "bad-2",
                        "data_count_rows",
                        {"source_id": source.id, "resource_ids": ids},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "fixed",
                        "data_count_rows",
                        {"source_id": source.id, "resource_ids": ids[:4]},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Partial counts obtained; four tables remain.",
            ),
        ]
    )
    agent = await Agent.open(
        "count-progress",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run(
            "Count the admitted tables", source_scope_ids=(source.id,)
        )
        assert result.reason == "completed"
        systems = [
            b.text
            for m in provider.requests[3].messages
            if m.role is MessageRole.SYSTEM
            for b in m.content
            if isinstance(b, TextBlock)
        ]
        assert any("Repeated invocation failure" in text for text in systems)
        assert not any(
            "Repeated invocation failure" in b.text
            for m in provider.requests[2].messages
            if m.role is MessageRole.SYSTEM
            for b in m.content
            if isinstance(b, TextBlock)
        )
        evidence = [
            b
            for m in provider.requests[4].messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
        assert [b.call_id for b in evidence] == ["load", "bad-1", "bad-2", "fixed"]
        assert [b.is_error for b in evidence] == [False, True, True, False]
        assert evidence[1] == next(
            b
            for m in provider.requests[2].messages
            for b in m.content
            if isinstance(b, ToolResultBlock) and b.call_id == "bad-1"
        )
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "fault", ["oversized", "duplicates", "other_source", "view", "drift"]
)
async def test_count_rejections_never_manufacture_zero_or_partial_batches(
    tmp_path, monkeypatch, fault
):
    import sqlite3

    from daita.adapters.sqlite_query import SQLiteQueryBackend

    paths = seed_sqlite(tmp_path)
    agent = await Agent.create(
        "count-failures", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    calls = []
    original = SQLiteQueryBackend.execute_read

    async def observed(self, **kwargs):
        calls.append(kwargs)
        return await original(self, **kwargs)

    monkeypatch.setattr(SQLiteQueryBackend, "execute_read", observed)
    try:
        first = await agent.attach(SQLiteSource(paths["Juniper"]))
        other = await agent.attach(SQLiteSource(paths["Meridian"]))
        items = await agent.list_catalog_resources(source_id=first.id)
        tables = [r for r in items if r.kind is ResourceKind.TABLE]
        ids = tuple(r.id for r in tables[:2])
        if fault == "oversized":
            ids = tuple(r.id for r in tables)
        if fault == "duplicates":
            ids = (ids[0], ids[0])
        if fault == "other_source":
            ids = ((await agent.list_catalog_resources(source_id=other.id))[0].id,)
        if fault == "view":
            ids = (next(r.id for r in items if r.kind is ResourceKind.VIEW),)
        if fault == "drift":
            with sqlite3.connect(paths["Juniper"]) as connection:
                connection.execute("ALTER TABLE records_00 ADD COLUMN changed INTEGER")
        run = RunInput(
            "count-fault",
            agent.id,
            "count",
            datetime.now(UTC),
            source_scope_ids=(first.id,),
        )
        call = ToolCall(
            "count", "data_count_rows", {"source_id": first.id, "resource_ids": ids}
        )
        result = (
            await execute_projected(agent._embedded._capability_runtime, run, (call,))
        ).ordered_results[0]
        assert result.is_error and "data" not in result.output
        assert len(calls) == (1 if fault == "drift" else 0)
        assert await agent.list_effects() == ()
    finally:
        await agent.close()
