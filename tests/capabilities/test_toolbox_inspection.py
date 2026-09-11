"""Component-owned tests split from ``test_toolbox_contracts.py``."""

from __future__ import annotations

from tests.capabilities._toolbox_support import (
    Any,
    CanonicalMessage,
    CapabilityRegistry,
    CapabilityRuntime,
    LoopLimits,
    Mapping,
    OperationalEffect,
    StaticTestDomain,
    ToolboxId,
    ToolCall,
    ToolLoadMode,
    ToolTextTrust,
    _append_results,
    _data,
    _declaration,
    _error_code,
    _execute,
    _load,
    _run,
    _runtime,
    canonical_json,
    cast,
    pytest,
    replace,
)


async def test_exact_contract_inspection_never_activates_or_executes():
    runtime, registry, _, executors = _runtime()
    run = _run()
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    inspected = (
        await _execute(
            runtime,
            run,
            projection,
            ToolCall("inspect", "toolbox_inspect", {"tool_name": "on_demand_a"}),
        )
    ).ordered_results[0]
    assert not inspected.is_error, inspected
    contract = cast(Mapping[str, Any], _data(inspected))["value"]
    assert (
        contract["input_schema"] == registry.tool_definition("on_demand_a").input_schema
    )
    assert _data(inspected)["complete"] is True
    call = ToolCall("inspect", "toolbox_inspect", {"tool_name": "on_demand_a"})
    messages = _append_results((), (call,), (inspected,))
    after = runtime.project(catalog, messages)
    assert after.loaded_entries == ()
    assert after.provider_definitions == projection.provider_definitions
    blocked = (
        await _execute(
            runtime,
            run,
            after,
            ToolCall("attempt", "on_demand_a", {}),
            messages=messages,
        )
    ).ordered_results[0]
    assert _error_code(blocked) == "tool_not_available"
    assert all(e.execute_calls == e.preflight_calls == 0 for e in executors)


async def test_exact_inspection_is_available_for_a_pinned_only_domain():
    domain, executors = _declaration(
        "pinned_only",
        (
            (
                "pinned_contract",
                ToolboxId.SOURCES,
                ToolLoadMode.PINNED,
                OperationalEffect.NONE,
                ToolTextTrust.CODE,
            ),
        ),
    )
    registry = CapabilityRegistry(
        declarations=(domain.declarations,), executors=executors
    )
    runtime = CapabilityRuntime(registry, (domain,))
    run = _run()
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    result = (
        await _execute(
            runtime,
            run,
            projection,
            ToolCall("inspect", "toolbox_inspect", {"tool_name": "pinned_contract"}),
        )
    ).ordered_results[0]
    assert not result.is_error
    assert [tool.name for tool in catalog.control_definitions] == ["toolbox_inspect"]
    assert all(e.execute_calls == 0 for e in executors)


@pytest.mark.parametrize("shape", ["wide", "deep", "long_text"])
async def test_inspection_retrieves_large_contract_exactly_with_bounded_pages(shape):
    domain, executors = _declaration(
        "inspection",
        (
            (
                "inspect_unfamiliar",
                ToolboxId.SOURCES,
                ToolLoadMode.ON_DEMAND,
                OperationalEffect.NONE,
                ToolTextTrust.CODE,
            ),
        ),
    )
    schema = {
        "type": "object",
        "properties": {"queue_key": {"type": "string"}},
        "required": ["queue_key"],
    }
    if shape == "wide":
        schema["properties"] = {
            f"field/{i}~value": {"type": "string", "description": "É" * 180}
            for i in range(35)
        }
        schema["required"] = list(schema["properties"])
    elif shape == "deep":
        for _ in range(6):
            schema = {"type": "object", "properties": {"nested": schema}}
    else:
        schema["description"] = "Aé🧭" * 2500
    capability = replace(domain.declarations.capabilities[0], input_schema=schema)
    domain = StaticTestDomain(
        (capability,), domain.declarations.tool_views, domain_owner_id="inspection"
    )
    registry = CapabilityRegistry(
        declarations=(domain.declarations,), executors=executors
    )
    limits = replace(
        LoopLimits(), max_toolbox_load_result_bytes=1800, max_tool_result_depth=10
    )
    runtime = CapabilityRuntime(registry, (domain,), limits=limits)
    run = _run()
    catalog = await runtime.prepare_run(run)
    messages: tuple[CanonicalMessage, ...] = ()
    digest: str | None = None
    count = 0

    async def retrieve(path="") -> Any:
        nonlocal messages, digest, count
        offset = 0
        reconstructed: Any = None
        while True:
            count += 1
            assert count < 300
            arguments: dict[str, object] = {"tool_name": "inspect_unfamiliar"}
            if digest is not None:
                arguments.update(contract_digest=digest, path=path, offset=offset)
            call = ToolCall(f"inspect-{count}", "toolbox_inspect", arguments)
            result = (
                await _execute(
                    runtime,
                    run,
                    runtime.project(catalog, messages),
                    call,
                    messages=messages,
                )
            ).ordered_results[0]
            assert not result.is_error, result
            data = cast(Mapping[str, Any], _data(result))
            assert (
                len(canonical_json(data).encode())
                <= limits.max_toolbox_load_result_bytes
            )
            digest = data["contract_digest"]
            messages = _append_results(messages, (call,), (result,))
            if data["complete"]:
                return data["value"]
            if data["value_type"] == "string":
                reconstructed = (reconstructed or "") + data["text"]
            else:
                if reconstructed is None:
                    reconstructed = {} if data["value_type"] == "object" else []
                for child in data["children"]:
                    child_value = (
                        child["value"]
                        if child["complete"]
                        else await retrieve(child["path"])
                    )
                    if isinstance(reconstructed, list):
                        reconstructed.append(child_value)
                    else:
                        token = (
                            child["path"]
                            .rsplit("/", 1)[1]
                            .replace("~1", "/")
                            .replace("~0", "~")
                        )
                        reconstructed[token] = child_value
            if data["next_offset"] is None:
                return reconstructed
            assert data["next_offset"] > offset
            offset = data["next_offset"]

    full = await retrieve()
    assert canonical_json(full["input_schema"]) == canonical_json(
        capability.input_schema
    )
    assert runtime.project(catalog, messages).loaded_entries == ()
    assert all(e.execute_calls == e.preflight_calls == 0 for e in executors)


async def test_omitted_load_contract_has_exact_inspection_reference():
    runtime, registry, _, _ = _runtime()
    run = _run()
    catalog = await runtime.prepare_run(run)
    full, _, _ = await _load(
        runtime, run, catalog, (), ("on_demand_a",), call_id="full"
    )
    runtime._limits = replace(
        runtime._limits,
        max_toolbox_load_result_bytes=len(canonical_json(_data(full)).encode()) - 1,
    )
    omitted, messages, projection = await _load(
        runtime, run, catalog, (), ("on_demand_a",), call_id="small"
    )
    assert not omitted.is_error
    ref = cast(Mapping[str, Any], _data(omitted))["contracts"][0]
    assert ref["complete"] is False
    assert "input_schema" not in ref
    assert {entry.view.name for entry in projection.loaded_entries} == {"on_demand_a"}
    inspected = (
        await _execute(
            runtime,
            run,
            projection,
            ToolCall(
                "exact",
                "toolbox_inspect",
                {
                    "tool_name": "on_demand_a",
                    "contract_digest": ref["contract_digest"],
                    "path": "/input_schema",
                },
            ),
            messages=messages,
        )
    ).ordered_results[0]
    assert not inspected.is_error
    assert (
        _data(inspected)["value"]
        == registry.tool_definition("on_demand_a").input_schema
    )


@pytest.mark.parametrize(
    "arguments,code",
    [
        ({"tool_name": "not_prepared"}, "toolbox_tool_not_available"),
        (
            {"tool_name": "on_demand_a", "contract_digest": "sha256:" + "0" * 64},
            "toolbox_inspect_stale",
        ),
        (
            {"tool_name": "on_demand_a", "path": "/input_schema"},
            "toolbox_inspect_reference_required",
        ),
        (
            {"tool_name": "on_demand_a", "offset": 1},
            "toolbox_inspect_reference_required",
        ),
    ],
)
async def test_inspection_rejects_inexact_or_unprepared_references(arguments, code):
    runtime, _, _, executors = _runtime()
    run = _run()
    catalog = await runtime.prepare_run(run)
    result = (
        await _execute(
            runtime,
            run,
            runtime.project(catalog, ()),
            ToolCall("inspect", "toolbox_inspect", arguments),
        )
    ).ordered_results[0]
    assert _error_code(result) == code
    assert all(e.execute_calls == 0 for e in executors)


async def test_inspection_admission_failure_preserves_independent_ordered_results(
    monkeypatch,
):
    runtime, _, domain, executors = _runtime()
    run = _run()
    catalog = await runtime.prepare_run(run)

    async def unavailable(_run):
        raise RuntimeError("private storage exception must not reach the model")

    monkeypatch.setattr(domain, "project", unavailable)
    results = (
        await _execute(
            runtime,
            run,
            runtime.project(catalog, ()),
            ToolCall("inspect", "toolbox_inspect", {"tool_name": "on_demand_a"}),
            ToolCall("read", "pinned_read", {}),
        )
    ).ordered_results
    assert [item.call_id for item in results] == ["inspect", "read"]
    assert _error_code(results[0]) == "toolbox_inspect_unavailable"
    assert "private storage" not in canonical_json(results[0].output)
    assert not results[1].is_error
    assert sum(e.execute_calls for e in executors) == 1
