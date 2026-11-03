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
    resolve_identity_refs,
    resource_id_schema,
    source_id_schema,
)
from daita.capability_runtime import CapabilityRuntime
from daita.catalog import ResourceKind, catalog_resource_id
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
# The shape runs stored before names were used: no labels, numbered handles.
STORED_SCOPE = EffectiveSourceScope(SCOPE.source_ids, SCOPE.resource_ids)
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

    refs = scope_refs(STORED_SCOPE)
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
    # A scope that froze display names was never shown a handle.
    assert not scope_refs(SCOPE).handles


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


def test_a_handle_shaped_value_in_a_new_run_is_only_a_name(rig):
    """A source called ``s2`` is that source, and ``t1`` names no numbered table."""

    runtime, _, _ = rig
    # SOURCE_A sorts first, so as a handle ``s2`` would mean the other source.
    table = catalog_resource_id(SOURCE_A, ResourceKind.TABLE, "t1")
    scope = EffectiveSourceScope(
        frozenset({SOURCE_A, SOURCE_B}),
        frozenset({ORDERS_B, table}),
        {SOURCE_A: ("s2", "sqlite"), SOURCE_B: ("HR", "sqlite")},
    )
    resolve = runtime._registry.resolve_call
    named = resolve(
        _call("n", "read", "s2", "t1"), result_refs=None, source_scope=scope
    )
    assert named.arguments == _frozen({"source_id": SOURCE_A, "resource_ids": (table,)})
    for call, code in (
        (_call("s", "read", "s1", "orders"), "source_unknown"),
        (ToolCall("t", "tables", {"tables": ("t2",)}), "table_unknown"),
    ):
        with pytest.raises(CapabilityInputError) as refused:
            resolve(call, result_refs=None, source_scope=scope)
        assert refused.value.code == code


def test_nested_tables_resolve_within_the_sources_their_parent_names():
    """Two SQLite sources can share ``main.orders``; the subject says which it means."""

    orders_a = catalog_resource_id(SOURCE_A, ResourceKind.TABLE, "main.orders")
    orders_b = catalog_resource_id(SOURCE_B, ResourceKind.TABLE, "main.orders")
    scope = EffectiveSourceScope(
        frozenset({SOURCE_A, SOURCE_B}),
        frozenset({orders_a, orders_b}),
        {SOURCE_A: ("Sales", "sqlite"), SOURCE_B: ("HR", "sqlite")},
    )
    schema = _object(
        subject=_object(
            fields=_array(_object(resource_id=resource_id_schema()), 4),
            resource_ids=_array(resource_id_schema(), 4),
            source_ids=_array(source_id_schema(), 4),
        )
    )
    field = {"table": "main.orders", "field_name": "total"}
    named = {"sources": ["HR"], "tables": ["main.orders"], "fields": [field]}
    resolved = resolve_identity_refs(schema, {"subject": named}, scope)["subject"]
    assert isinstance(resolved, Mapping)
    assert resolved["source_ids"] == [SOURCE_B]
    assert resolved["resource_ids"] == [orders_b]
    assert list(resolved["fields"]) == [
        {"resource_id": orders_b, "field_name": "total"}
    ]
    # Naming no source leaves the shared name ambiguous rather than guessing.
    with pytest.raises(CapabilityInputError) as ambiguous:
        resolve_identity_refs(schema, {"subject": {"fields": [field]}}, scope)
    assert ambiguous.value.code == "table_ambiguous"


def test_table_names_prefer_their_exact_case_then_fold(rig):
    """``sales.Customers`` and ``sales.customers`` are two PostgreSQL tables, and
    an unquoted ``SALES.CUSTOMERS`` is the second of them, as the server reads it."""

    runtime, _, _ = rig
    quoted = catalog_resource_id(SOURCE_A, ResourceKind.TABLE, "sales.Customers")
    folded = catalog_resource_id(SOURCE_A, ResourceKind.TABLE, "sales.customers")
    scope = EffectiveSourceScope(
        frozenset({SOURCE_A}),
        frozenset({quoted, folded}),
        {SOURCE_A: ("Sales", "postgresql")},
    )
    resolve = runtime._registry.resolve_call
    for name, identity in (
        ("sales.Customers", quoted),
        ("sales.customers", folded),
        ("SALES.CUSTOMERS", folded),
    ):
        call = _call("c", "read", "Sales", name)
        resolved = resolve(call, result_refs=None, source_scope=scope)
        assert resolved.arguments["resource_ids"] == (identity,)
    with pytest.raises(CapabilityInputError) as refused:
        resolve(
            _call("c", "read", "Sales", "sales.absent"),
            result_refs=None,
            source_scope=scope,
        )
    assert refused.value.code == "table_unknown"


def test_argument_problems_name_the_keys_the_model_sends(rig):
    runtime, _, _ = rig
    registry = runtime._registry
    with pytest.raises(CapabilityInputError) as missing:
        registry.validate_arguments("test.scope.read", {"resource_ids": (ORDERS_A,)})
    assert "arguments.source is required" in str(missing.value)
    assert "source_id" not in canonical_json(missing.value.details)
    with pytest.raises(CapabilityInputError) as invalid:
        registry.validate_arguments(
            "test.scope.read", {"source_id": SOURCE_A, "resource_ids": (7,)}
        )
    assert "resource_ids" not in str(invalid.value)
    assert "resource_ids" not in canonical_json(invalid.value.details)


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

    outside = "source:sha256:" + "f" * 64
    relationship = "catalog-relationship:sha256:" + "1" * 64
    link = "catalog-link:sha256:" + "2" * 64
    embedded = model_tool_result(
        ToolResultBlock(
            "call-2",
            {
                "note": f"- Sources: {SOURCE_A}, {outside}",
                "uri": f"sqlite://{SOURCE_B}/main/orders",
                "by_table": {ORDERS_A: 1, outside: 2},
                # A query result's own columns are data, whatever they are called.
                "rows": [
                    {"source": "web", "source_id": 7, "resource_ids": None},
                    {"source_id": "unrelated text", "relationship_id": 9},
                ],
                # A relationship shows its tables and keys, and needs no identity.
                "relationships": [
                    {"relationship_id": relationship, "from_resource_id": ORDERS_A},
                    {"relationship_id": link, "to_resource_id": CUSTOMERS_A},
                ],
                "paths": [{"relationship_ids": [relationship, link]}],
                "matched_fields": [f"relationship:{relationship}", "name"],
            },
        ),
        None,
        NAMES,
    )["output"]
    assert embedded == {
        "note": "- Sources: Sales, a source outside this run",
        "uri": "sqlite://HR/main/orders",
        "by_table": {"orders": 1},
        "rows": [
            {"source": "web", "source_id": 7, "resource_ids": None},
            {"source_id": "unrelated text", "relationship_id": 9},
        ],
        "relationships": [{"from_table": "orders"}, {"to_table": "customers"}],
        "paths": [{}],
        "matched_fields": ["relationship", "name"],
    }

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
        "catalog_describe:input.source",
        "data_update_rows:input.table",
        "semantic_save:input.subject.fields[items].table",
    } <= identity_keys


async def test_routine_save_persists_full_source_ids_named_by_name(
    tmp_path: Path,
) -> None:
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
        origin = await agent.run("Create a scheduled report.")
        assert origin.conversation_id is not None
        run = (await agent._embedded._store.load(origin.run_id)).run
        # The stored run freezes the display names its transcript resolves by.
        assert run.resolved_source_scope is not None
        assert run.resolved_source_scope.source_labels[source.id][0] == "Current"
        now = datetime.now(UTC)
        arguments = {
            "title": "Report",
            "prompt": "Describe the current_value resource.",
            "schedule": {
                "kind": "interval",
                "interval_seconds": 3600,
                "anchor_at": (now + timedelta(hours=1)).isoformat(),
            },
            "sources": ("current",),
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
        assert inspection.routine.source_ids == (source.id,)
        assert approvals
        # The confirmation card names the source; the record keeps its full id.
        assert approvals[-1].arguments["sources"] == ("Current",)
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
    handles = scope_refs(STORED_SCOPE).handles
    old = ToolCall(
        "old",
        "read",
        {"source_id": "s2", "resource_ids": (handles[ORDERS_B], ORDERS_A)},
    )
    executed = registry.executed_call(old, messages=(), source_scope=STORED_SCOPE)
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
    task = registry.executed_call(
        nested, messages=(), source_scope=STORED_SCOPE
    ).arguments["task"]
    assert isinstance(task, Mapping)
    assert task["arguments"] == _frozen(
        {"source_id": SOURCE_A, "resource_ids": (ORDERS_A,)}
    )
    # A handle this scope never issued is no name either: the call never ran.
    unissued = ToolCall(
        "unissued", "read", {"source_id": "s9", "resource_ids": ("t1",)}
    )
    assert (
        registry.executed_call(unissued, messages=(), source_scope=STORED_SCOPE)
        == unissued
    )


def test_a_source_read_in_full_resolves_names_by_hash_alone():
    scope = EffectiveSourceScope(
        frozenset({SOURCE_A}),
        frozenset(),
        {SOURCE_A: ("Sales", "postgresql")},
        frozenset({SOURCE_A}),
    )
    schema = {
        "type": "object",
        "properties": {
            "source_id": source_id_schema(),
            "resource_id": resource_id_schema(),
        },
    }
    resolved = resolve_identity_refs(
        schema, {"source": "Sales", "table": "sales.Orders"}, scope
    )
    # Exact, so sales.Orders and sales.orders stay two tables, and the same on replay.
    assert resolved["resource_id"] == catalog_resource_id(
        SOURCE_A, ResourceKind.TABLE, "sales.Orders"
    )
    assert scope.to_mapping()["resource_ids"] == ()


def test_a_stored_scope_that_lists_tables_still_resolves_a_view_as_a_view():
    stored = EffectiveSourceScope(
        frozenset({SOURCE_A}),
        frozenset({CUSTOMERS_A}),
        {SOURCE_A: ("Sales", "postgresql")},
        frozenset({SOURCE_A}),
    )
    schema = {
        "type": "object",
        "properties": {
            "source_id": source_id_schema(),
            "resource_id": resource_id_schema(),
        },
    }
    resolved = resolve_identity_refs(
        schema, {"source": "Sales", "table": "customers"}, stored
    )
    assert resolved["resource_id"] == CUSTOMERS_A
