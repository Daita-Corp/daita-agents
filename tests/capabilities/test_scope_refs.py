"""Sources and tables are addressed by name; the runtime resolves exact IDs."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    ApprovalRequest,
    AutomationEligibility,
    CallTargetLane,
    Capability,
    CapabilityInputError,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    EffectReceiptPolicy,
    OperationalEffect,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolView,
    resource_id_schema,
    source_id_schema,
)
from daita.capability_runtime import CapabilityRuntime
from daita.catalog import ResourceKind, catalog_resource_id
from daita.distribution import outcome_contract_projection
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
    model_tool_result,
)
from daita.llm.providers.anthropic.messages import _message_input
from daita.llm.providers.mock import MockModelProvider
from daita.llm.providers.openai.messages import _response_input
from daita.loop.models import RunInput, Transcript
from daita.scope import EffectiveSourceScope, scope_refs
from tests.capabilities._effect_receipt_support import STARTED_AT, _store
from tests.support.capability_runtime import (
    StaticTestDomain,
    execute_projected,
    presentation_metadata,
    static_registry,
)
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.workspace import workspace_for

SOURCE_A = "source:sha256:" + "a" * 64
SOURCE_B = "source:sha256:" + "b" * 64
SOURCE_C = "source:sha256:" + "c" * 64
ORDERS_A = catalog_resource_id(SOURCE_A, ResourceKind.TABLE, "orders")
CUSTOMERS_A = catalog_resource_id(SOURCE_A, ResourceKind.VIEW, "customers")
ORDERS_B = catalog_resource_id(SOURCE_B, ResourceKind.TABLE, "orders")
SCOPE = EffectiveSourceScope(
    frozenset({SOURCE_B, SOURCE_A}),
    frozenset({ORDERS_B, ORDERS_A, CUSTOMERS_A}),
    {SOURCE_A: ("Sales", "postgresql"), SOURCE_B: ("HR", "sqlite")},
)
NAMES = {
    SOURCE_A: "Sales",
    SOURCE_B: "HR",
    ORDERS_A: "orders",
    CUSTOMERS_A: "customers",
}
_RUN = RunInput(
    id="run-effect",
    agent_id="agent-effect",
    message="exact names",
    created_at=STARTED_AT,
    conversation_id="conversation-effect",
)
_frozen = FrozenJsonObject.from_mapping


def _object(**properties: object) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: object, maximum: int) -> dict[str, object]:
    return {"type": "array", "items": items, "minItems": 1, "maxItems": maximum}


_IDENTITY = {
    "source_id": source_id_schema(),
    "resource_ids": _array(resource_id_schema(), 4),
}
_IDENTITY_INPUT = _object(**_IDENTITY)
_OK_OUTPUT = _object(ok={"type": "boolean"})


def test_stored_handles_still_number_the_frozen_scope():
    """Older transcripts carry ``s1``/``t1`` text; it resolves as it always did."""

    refs = scope_refs(SCOPE)
    assert dict(refs.handles) == {
        SOURCE_A: "s1",
        SOURCE_B: "s2",
        **{
            identity: f"t{index}"
            for index, identity in enumerate(sorted(SCOPE.resource_ids), 1)
        },
    }
    assert refs.resolve("s", "s3") is None
    assert not scope_refs(None).handles


def test_source_labels_are_frozen_with_the_scope():
    assert SCOPE.sources_named("sales") == (SOURCE_A,)
    assert SCOPE.sources_named("Finance") == ()
    assert SCOPE.to_mapping()["source_labels"] == {
        SOURCE_A: ["Sales", "postgresql"],
        SOURCE_B: ["HR", "sqlite"],
    }
    assert (
        "source_labels"
        not in EffectiveSourceScope(frozenset(), frozenset()).to_mapping()
    )
    with pytest.raises(ValueError):
        EffectiveSourceScope(frozenset({SOURCE_A}), frozenset(), {SOURCE_C: ("x", "y")})


@dataclass
class _Executor:
    executor_id: str
    effect: bool
    requests: list[ToolExecution] = field(default_factory=list)

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        return _frozen({"current": True})

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.requests.append(request)
        if not self.effect:
            return ToolOutput(kind="test.scope.output", data={"ok": True})
        payload = _frozen({key: request.arguments[key] for key in _IDENTITY})
        return ToolOutput(
            kind="test.scope.write",
            data=payload,
            effect_observation=EffectObservation(
                EffectOutcome.SUCCEEDED, EffectEvidenceBasis.SERVER_REPORTED, payload
            ),
        )


def _capability(name: str, input_schema: dict[str, object], **kw) -> Capability:
    capability_id = f"test.scope.{name}"
    if kw.pop("effect", False):
        kw |= dict(
            output_kind="test.scope.write",
            output_schema=_IDENTITY_INPUT,
            access_mode=AccessMode.NONE,
            operational_effect=OperationalEffect.EXTERNAL_ACTION,
            automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
            effect_receipt_policy=EffectReceiptPolicy(
                capability_id, _IDENTITY_INPUT, EffectEvidenceBasis.SERVER_REPORTED
            ),
        )
    return Capability(
        id=capability_id,
        description=f"Exercise exact identities through {name}.",
        input_schema=input_schema,
        executor_id=f"{capability_id}.executor",
        **{"output_kind": "test.scope.output", "output_schema": _OK_OUTPUT, **kw},
    )


_CAPABILITIES = (
    _capability("read", _IDENTITY_INPUT, access_mode=AccessMode.READ),
    # Names a table without naming its source.
    _capability(
        "tables",
        _object(resource_ids=_array(resource_id_schema(), 4)),
        access_mode=AccessMode.READ,
    ),
    # Retains another capability's exact arguments.
    _capability(
        "propose",
        _object(
            task=_object(capability_id={"type": "string"}, arguments={"type": "object"})
        ),
        delegated_arguments=(("task", "arguments", "input"),),
    ),
    _capability("write", _IDENTITY_INPUT, effect=True),
    _capability(
        "cite",
        _object(**_IDENTITY, evidence_refs=_array({"type": "string"}, 2)),
        effect=True,
        result_ref_arguments=("evidence_refs[]",),
    ),
)


class _Approvals(list[ApprovalRequest]):
    """An approval handler that approves and records every request."""

    async def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        self.append(request)
        return ApprovalDecision.APPROVE


@pytest.fixture
async def store(tmp_path: Path):
    store = await _store(tmp_path / "state.db")
    yield store
    await store.close()


@pytest.fixture
def rig(store):
    """A runtime over read/tables/propose/write/cite, its executors and approvals."""

    executors, views = {}, []
    for capability in _CAPABILITIES:
        name = capability.id.rsplit(".", 1)[1]
        effect = capability.effect_receipt_policy is not None
        executors[name] = _Executor(capability.executor_id, effect)
        load = ToolLoadMode.ON_DEMAND if effect else ToolLoadMode.PINNED
        presentation = presentation_metadata(load_mode=load)
        views.append(
            ToolView(name, capability.id, capability.description, presentation)
        )
    domain = StaticTestDomain(_CAPABILITIES, tuple(views))
    approvals = _Approvals()

    async def resolve_scope(run: RunInput) -> EffectiveSourceScope:
        return SCOPE

    runtime = CapabilityRuntime(
        static_registry(domain, tuple(executors.values())),
        (domain,),
        effect_receipts=store,
        approval_handler=approvals,
        clock=lambda: STARTED_AT,
        source_scope_resolver=resolve_scope,
    )
    return runtime, {name: item.requests for name, item in executors.items()}, approvals


def _call(call_id: str, tool: str, source: str, *tables: str) -> ToolCall:
    """One call naming a source and tables; ``propose`` nests it as a read task."""

    if tool != "propose":
        return ToolCall(call_id, tool, {"source": source, "tables": tables})
    arguments = {"source": source, "tables": list(tables)}
    task = {"capability_id": "test.scope.read", "arguments": arguments}
    return ToolCall(call_id, tool, {"task": task})


async def test_names_reach_execution_approval_and_receipt_as_full_ids(store, rig):
    runtime, requests, approvals = rig
    sent = (
        _call("read", "read", "Sales", "orders", "customers"),
        _call("propose", "propose", "hr", "orders"),
        _call("write", "write", "HR", "orders"),
        ToolCall("tables", "tables", {"tables": ["customers"]}),
    )
    outcome = await execute_projected(runtime, _RUN, sent)
    assert not any(result.is_error for result in outcome.ordered_results)
    assert requests["read"][0].arguments == _frozen(
        {"source_id": SOURCE_A, "resource_ids": (ORDERS_A, CUSTOMERS_A)}
    )
    task = requests["propose"][0].arguments["task"]
    assert isinstance(task, Mapping)
    full = _frozen({"source_id": SOURCE_B, "resource_ids": (ORDERS_B,)})
    assert task["arguments"] == full
    assert requests["write"][0].arguments == full
    assert [request.arguments for request in approvals] == [full]
    (receipt,) = await store.list_effect_receipts("agent-effect")
    assert receipt.payload == full
    # A table name unique across the scope needs no source.
    assert requests["tables"][0].arguments == _frozen({"resource_ids": (CUSTOMERS_A,)})
    # The model's calls keep exactly what it sent.
    assert sent[2].arguments["source"] == "HR"

    # The same operation named by full IDs is the same reserved effect.
    again = await execute_projected(
        runtime, _RUN, (ToolCall("write-again", "write", full),)
    )
    reference = outcome.ordered_results[2].output["effect_receipt"]
    assert again.ordered_results[0].output["effect_receipt"] == reference
    assert len(requests["write"]) == 1


async def test_unknown_and_ambiguous_names_fail_before_dispatch(store, rig):
    runtime, requests, approvals = rig
    cases = (  # (call id, tool, source, table, error code, details name)
        ("source", "read", "Finance", "orders", "source_unknown", "source"),
        ("table", "read", "Sales", "invoices", "table_unknown", "tables[0]"),
        ("other", "read", "HR", "customers", "table_unknown", "tables[0]"),
        (
            "nested",
            "propose",
            "Finance",
            "orders",
            "source_unknown",
            "task.arguments.source",
        ),
        ("effect", "write", "Finance", "orders", "source_unknown", "source"),
    )
    calls = tuple(_call(*case[:4]) for case in cases) + (
        ToolCall("ambiguous", "tables", {"tables": ["orders"]}),
    )
    results = (await execute_projected(runtime, _RUN, calls)).ordered_results
    expected = [case[4:] for case in cases] + [("table_ambiguous", "tables[0]")]
    details = []
    for result, (code, name) in zip(results, expected, strict=True):
        error = result.output["error"]
        assert result.is_error and isinstance(error, Mapping)
        assert error["code"] == code
        assert isinstance(detail := error["details"], Mapping)
        assert detail["execution_disposition"] == "not_dispatched"
        assert detail["name"] == name
        details.append(detail)
    assert details[0]["recovery"] == _frozen(
        {
            "kind": "correct_arguments",
            "constraint": "Use one of this run's source names: 'HR', 'Sales'.",
        }
    )
    assert (
        details[-1]["recovery"]["constraint"] == "Name the source too: 'Sales', 'HR'."
    )
    assert "sha256" not in canonical_json(
        [result.output for result in results]
    ), "no error payload shows an ID"
    assert not any(requests.values()) and not approvals
    assert not await store.list_effect_receipts("agent-effect")


def test_two_sources_with_one_display_name_are_refused_as_ambiguous(rig):
    runtime, _, _ = rig
    scope = EffectiveSourceScope(
        frozenset({SOURCE_A, SOURCE_C}),
        frozenset({ORDERS_A}),
        {SOURCE_A: ("Sales", "postgresql"), SOURCE_C: ("Sales", "sqlite")},
    )
    with pytest.raises(CapabilityInputError) as caught:
        runtime._registry.resolve_call(
            _call("ambiguous", "read", "Sales", "orders"),
            result_refs=None,
            source_scope=scope,
        )
    assert caught.value.code == "source_ambiguous"
    assert canonical_json(caught.value.details["candidates"]) == canonical_json(
        [
            {"display_name": "Sales", "adapter": "postgresql"},
            {"display_name": "Sales", "adapter": "sqlite"},
        ]
    )
    assert "sha256" not in canonical_json(caught.value.details)


def test_rendered_results_show_names_and_never_ids():
    hits = [
        {"source_id": SOURCE_A, "resource_id": CUSTOMERS_A, "name": "customers"},
        {"source_id": "source:sha256:" + "f" * 64, "name": "outside"},
    ]
    data = {
        "hits": hits,
        "resource_ids": [ORDERS_A, "catalog-resource:sha256:" + "9" * 64],
        "resource_revisions": [
            {"resource_id": ORDERS_A, "revision": "sha256:" + "0" * 64}
        ],
        "evidence_call_ids": ["call-1"],
        "id": SOURCE_B,
    }
    block = ToolResultBlock("call-1", {"kind": "test.scope.output", "data": data})
    rendered = model_tool_result(block, "r1", NAMES)["output"]["data"]  # type: ignore[index]
    assert rendered["hits"][0] == {
        "source": "Sales",
        "table": "customers",
        "name": "customers",
    }
    # An ID outside the run's scope is no name the model can use.
    assert rendered["hits"][1] == {"source": None, "name": "outside"}
    assert rendered["tables"] == ["orders", None]
    assert rendered["resource_revisions"] == [
        {"table": "orders", "revision": "sha256:" + "0" * 64}
    ]
    assert rendered["evidence_call_ids"] == ["call-1"]
    assert rendered["id"] == "HR"
    assert "sha256" not in canonical_json(rendered).replace("sha256:" + "0" * 64, "")
    # Stored output is unchanged; only the rendering carries names.
    assert ORDERS_A in canonical_json(block.output)

    call = ToolCall("call-1", "read", {"source": "Sales"})
    request = ModelRequest(
        messages=(
            CanonicalMessage(MessageRole.ASSISTANT, tool_calls=(call,)),
            CanonicalMessage(MessageRole.TOOL, content=(block,)),
        ),
        tool_result_refs={"call-1": "r1"},
        scope_names=NAMES,
    )
    refs = (request.tool_result_refs, request.scope_names)
    _, anthropic = _message_input(request.messages, "anthropic", *refs)
    openai = _response_input(request.messages, "openai", *refs)
    for text in (
        str(anthropic[-1]["content"][0]["content"]),  # type: ignore[index]
        str(next(item["output"] for item in openai if "output" in item)),
    ):
        assert '"table":"customers"' in text and '"source":"Sales"' in text
        assert "catalog-resource:" not in text and "source:sha256" not in text


async def test_every_model_facing_identity_argument_takes_a_name(tmp_path: Path):
    agent = await Agent.create(
        "scope-ref-inventory", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    names = ("source_id", "source_ids", "resource_id", "resource_ids")
    identity_keys: set[str] = set()
    id_patterns: set[str] = set()

    def visit(tool: str, rule: object, path: str) -> None:
        if not isinstance(rule, Mapping):
            return
        pattern = rule.get("pattern")
        if isinstance(pattern, str) and (
            "source:" in pattern or "catalog-resource" in pattern
        ):
            id_patterns.add(f"{tool}:{path}")
        properties = rule.get("properties")
        for key, child in properties.items() if isinstance(properties, Mapping) else ():
            assert not str(key).endswith(names), f"{tool}:{path}.{key} is an ID key"
            leaf = child.get("items", child) if isinstance(child, Mapping) else None
            if isinstance(leaf, Mapping) and str(
                leaf.get("description", "")
            ).startswith(("Source display name", "Exact table")):
                identity_keys.add(f"{tool}:{path}.{key}")
            visit(tool, child, f"{path}.{key}")
        for key in ("items", "additionalProperties"):
            visit(tool, rule.get(key), f"{path}[{key}]")

    try:
        registry = agent._embedded._capabilities
        for name in sorted(registry.tool_names):
            definition = registry.tool_definition(name, lane=CallTargetLane.FOREGROUND)
            visit(name, definition.input_schema, "input")
    finally:
        await agent.close()

    assert id_patterns == set()
    assert {
        "data_query:input.source",
        "catalog_describe:input.tables",
        "data_update_rows:input.table",
        "routine_save:input.allowed_tables",
        "semantic_save:input.subject.fields[items].table",
    } <= identity_keys


async def test_routine_save_persists_full_ids_named_by_name(tmp_path: Path) -> None:
    database = tmp_path / "current.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE current_value (value INTEGER NOT NULL)")
    connection.close()
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="Authorize it."),)
    )
    approvals = _Approvals()
    agent = await Agent.create(
        "scope-ref-routine",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=64_000,
            max_output_tokens=1_000,
            supports_tools=True,
            supports_parallel_tools=True,
        ),
        approval_handler=approvals,
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Current"))
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        origin = await agent.run("Create a scheduled report.")
        assert origin.conversation_id is not None
        destinations = await agent.distribution_destinations(origin.conversation_id)
        run = (await agent._embedded._store.load(origin.run_id)).run
        # The stored run freezes the display names its transcript resolves by.
        assert run.resolved_source_scope is not None
        assert run.resolved_source_scope.source_labels[source.id][0] == "Current"
        now = datetime.now(UTC)
        arguments = {
            "title": "Report",
            "authorized_instruction": "Describe the current_value resource.",
            "schedule": {
                "kind": "interval",
                "interval_seconds": 3600,
                "anchor_at": (now + timedelta(hours=1)).isoformat(),
            },
            "misfire_policy": "latest_only",
            "reporting_mode": "always",
            "allowed_sources": ("Current",),
            "allowed_tables": (resource.native_identity,),
            "allowed_connector_binding_ids": (),
            "allowed_capability_ids": ("catalog.describe",),
            "sensitivity_ceiling": "internal",
            "outcome_contract": outcome_contract_projection(
                no_artifact_outcome_contract()
            ),
            "distribution_destination_id": destinations[0].destination_id,
            "eligible_model_routes": (provider.provider_id,),
            "per_run_max_tokens": 1000,
            "per_run_max_cost_usd": "0.01",
            "cumulative_max_tokens": 10000,
            "cumulative_max_cost_usd": "0.10",
            "cumulative_max_attempts": 10,
            "cumulative_max_occurrences": 10,
            "maximum_consecutive_failures": 3,
            "expires_at": (now + timedelta(days=30)).isoformat(),
            "skill_names": (),
            "run_immediately": False,
        }
        created = await execute_projected(
            agent._embedded._capability_runtime,
            run,
            (ToolCall("named-routine", "routine_save", arguments),),
        )
        assert not created.ordered_results[0].is_error, canonical_json(
            created.ordered_results[0].output
        )
        (summary,) = await agent.list_routines()
        inspection = await agent.inspect_routine(summary.routine_id)
        assert inspection is not None
        assert inspection.routine.allowed_source_ids == (source.id,)
        assert inspection.routine.allowed_resource_ids == (resource.id,)
        assert approvals
        approved = canonical_json(approvals[-1].arguments)
        assert source.id in approved and resource.id in approved
        assert '"Current"' not in approved
    finally:
        await agent.close()


async def test_executed_record_is_exactly_what_execution_approval_and_receipt_saw(
    store, rig
):
    """One resolver derives the executed record every reader shares.

    A call naming a result ref, a source and tables is persisted as the model
    sent it; the registry's executed record of that persisted call equals the
    executor's request, the approval request and the receipt payload.
    """

    runtime, requests, approvals = rig
    registry = runtime._registry
    run = await runtime.freeze_source_scope(_RUN)
    assert run.resolved_source_scope == SCOPE
    load = ToolCall("load", "toolbox_load", {"tool_names": ("cite",)})
    sent_read = _call("read", "read", "HR", "orders")
    # The control result is numbered r1 but is not citable; the read is r2.
    cited = {"source": "Sales", "tables": ("customers", ORDERS_A)}
    sent_cite = ToolCall("cite", "cite", {**cited, "evidence_refs": ("r2",)})
    catalog = await runtime.prepare_run(run)
    messages: tuple[CanonicalMessage, ...] = (run.start_message(),)
    for call in (load, sent_read, sent_cite):
        outcome = await runtime.execute_all(
            run,
            (call,),
            projection=runtime.project(catalog, messages),
            messages=messages,
            sensitivity=ModelSensitivity.INTERNAL,
        )
        messages = (
            *messages,
            CanonicalMessage(MessageRole.ASSISTANT, tool_calls=(call,)),
            CanonicalMessage(MessageRole.TOOL, content=outcome.ordered_results),
        )
    transcript = Transcript(run, messages)
    results = [result for _, result in transcript.tool_pairs]
    assert all(result is not None and not result.is_error for result in results)

    load_pair, read_pair, cite_pair = registry.executed_tool_pairs(transcript)
    assert [load_pair[1], read_pair[1], cite_pair[1]] == results
    assert load_pair[0] == load
    assert read_pair[0].arguments == requests["read"][0].arguments
    assert read_pair[0].arguments == _frozen(
        {"source_id": SOURCE_B, "resource_ids": (ORDERS_B,)}
    )
    full = {"source_id": SOURCE_A, "resource_ids": (CUSTOMERS_A, ORDERS_A)}
    executed = cite_pair[0].arguments
    assert executed == _frozen({**full, "evidence_refs": ("read",)})
    assert executed == requests["cite"][0].arguments
    assert [request.arguments for request in approvals] == [executed]
    (receipt,) = await store.list_effect_receipts("agent-effect")
    assert (receipt.run_id, receipt.call_id) == (run.id, cite_pair[0].id)
    assert receipt.payload == _frozen(full)
    # The transcript keeps the model's exact text for provider replay.
    assert [call for call, _ in transcript.tool_pairs] == [load, sent_read, sent_cite]

    # A call that never resolved failed before execution and is unchanged.
    for call, history in (
        (_call("unknown", "read", "Finance", "orders"), messages),
        (ToolCall("unknown", "not_a_tool", {"source": "Sales"}), ()),
    ):
        assert (
            registry.executed_call(call, messages=history, source_scope=SCOPE) == call
        )


def test_stored_transcripts_still_resolve_their_handles(rig):
    """Runs recorded before names were used carry ``s#``/``t#`` text; it resolves."""

    runtime, _, _ = rig
    registry = runtime._registry
    handles = scope_refs(SCOPE).handles
    old = ToolCall(
        "old",
        "read",
        {"source_id": "s2", "resource_ids": (handles[ORDERS_B], ORDERS_A)},
    )
    executed = registry.executed_call(old, messages=(), source_scope=SCOPE)
    assert executed.arguments == _frozen(
        {"source_id": SOURCE_B, "resource_ids": (ORDERS_B, ORDERS_A)}
    )
    nested = ToolCall(
        "old-nested",
        "propose",
        {
            "task": {
                "capability_id": "test.scope.read",
                "arguments": {"source_id": "s1", "resource_ids": [handles[ORDERS_A]]},
            }
        },
    )
    task = registry.executed_call(nested, messages=(), source_scope=SCOPE).arguments[
        "task"
    ]
    assert isinstance(task, Mapping)
    assert task["arguments"] == _frozen(
        {"source_id": SOURCE_A, "resource_ids": (ORDERS_A,)}
    )
    # A handle this scope never issued is no name either: the call never ran.
    unissued = ToolCall(
        "unissued", "read", {"source_id": "s9", "resource_ids": ("t1",)}
    )
    assert registry.executed_call(unissued, messages=(), source_scope=SCOPE) == unissued
