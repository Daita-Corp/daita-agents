"""Toolbox, catalog-contract and step-projection helpers for the runtime."""

from __future__ import annotations

import re
from collections.abc import Mapping
from hashlib import sha256
from typing import TYPE_CHECKING, cast

from ._json import canonical_json
from .capabilities import (
    CallTargetLane,
    CapabilityRegistry,
    ToolLoadMode,
    model_identity_schema,
    model_output_schema,
)
from .capability_results import _json_depth
from .llm.errors import ToolSurfaceLimitExceeded
from .llm.models import (
    CanonicalMessage,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from .loop.models import LoopLimits
from .scope import EffectiveSourceScope

_TOKEN = re.compile(r"[a-z0-9]+")

if TYPE_CHECKING:
    from .capability_runtime import RunToolCatalogEntry, ToolboxManifestEntry


def _sha256_digest(value: object) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _definition_material(definition: ToolDefinition) -> dict[str, object]:
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": definition.input_schema,
    }


def _definition_bytes(definitions: tuple[ToolDefinition, ...]) -> int:
    return len(
        canonical_json([_definition_material(item) for item in definitions]).encode(
            "utf-8"
        )
    )


def _validate_pinned_surface(
    definitions: tuple[ToolDefinition, ...],
    limits: LoopLimits,
) -> None:
    definition_bytes = _definition_bytes(definitions)
    if (
        len(definitions) > limits.max_pinned_tools
        or definition_bytes > limits.max_pinned_tool_definition_bytes
    ):
        raise ToolSurfaceLimitExceeded(
            observed_tools=len(definitions),
            maximum_tools=limits.max_pinned_tools,
            observed_definition_bytes=definition_bytes,
            maximum_definition_bytes=limits.max_pinned_tool_definition_bytes,
        )


def _validate_step_surface(
    definitions: tuple[ToolDefinition, ...],
    limits: LoopLimits,
) -> None:
    definition_bytes = _definition_bytes(definitions)
    if (
        len(definitions) > limits.max_step_tools
        or definition_bytes > limits.max_step_tool_definition_bytes
    ):
        raise ToolSurfaceLimitExceeded(
            observed_tools=len(definitions),
            maximum_tools=limits.max_step_tools,
            observed_definition_bytes=definition_bytes,
            maximum_definition_bytes=limits.max_step_tool_definition_bytes,
        )


def _entry_resolves_exactly(
    entry: RunToolCatalogEntry,
    registry: CapabilityRegistry,
) -> bool:
    if entry.routes and not all(
        route.view == entry.view and _entry_resolves_exactly(route, registry)
        for _, route in entry.routes
    ):
        return False
    try:
        view, capability, owner_id = registry.resolve_tool_owner(
            entry.view.name, capability_id=entry.capability.id
        )
    except KeyError:
        return False
    return all(
        (
            view == entry.view,
            capability == entry.capability,
            owner_id == entry.domain_owner_id,
            capability.executor_id == entry.executor_id,
            _sha256_digest(capability.input_schema) == entry.input_schema_digest,
            view.presentation.toolbox_id is entry.toolbox_id,
            view.presentation.load_mode is entry.load_mode,
        )
    )


def _catalog_entry_material(entry: RunToolCatalogEntry) -> dict[str, object]:
    return {
        "tool_name": entry.view.name,
        "capability_id": entry.capability.id,
        "domain_owner_id": entry.domain_owner_id,
        "executor_id": entry.executor_id,
        "description": entry.view.description,
        "presentation_sensitivity": entry.view.presentation_sensitivity.value,
        "connector_presentation": entry.view.connector_presentation,
        "input_schema": model_identity_schema(entry.capability.input_schema),
        "input_schema_digest": entry.input_schema_digest,
        "output_kind": entry.capability.output_kind,
        "data_access": entry.capability.access_mode.value,
        "operational_effect": entry.capability.operational_effect.value,
        "presentation": {
            "toolbox_id": entry.toolbox_id.value,
            "load_mode": entry.load_mode.value,
            "text_trust": entry.view.presentation.text_trust.value,
            "summary": entry.view.presentation.summary,
            "when_to_use": entry.view.presentation.when_to_use,
            "keywords": entry.view.presentation.keywords,
        },
        "origin_revision_digest": entry.origin_revision_digest,
        "mode_routes": [
            {"mode": mode, "entry": _catalog_entry_material(route)}
            for mode, route in entry.routes
        ],
    }


def _tool_contract_reference(entry: RunToolCatalogEntry) -> dict[str, object]:
    return {
        "tool_name": entry.view.name,
        "capability_id": entry.capability.id,
        "contract_digest": _sha256_digest(
            {
                "entry": _catalog_entry_material(entry),
                "output_schema": entry.capability.output_schema,
            }
        ),
        "input_schema_digest": entry.input_schema_digest,
        "origin_revision_digest": entry.origin_revision_digest,
        "complete": False,
        "inspection_tool": "toolbox_inspect",
    }


def _tool_contract(
    entry: RunToolCatalogEntry, *, include_schemas: bool = True
) -> dict[str, object]:
    """One registered declaration projection shared by search, load and inspection."""
    receipt = entry.capability.effect_receipt_policy
    contract = {
        "effect_evidence_basis": (
            None if receipt is None else receipt.success_evidence_basis.value
        ),
        "connector": entry.view.connector_presentation,
        **_tool_contract_reference(entry),
    }
    if include_schemas:
        contract.update(
            complete=True,
            input_schema=model_identity_schema(entry.capability.input_schema),
            output_schema=model_output_schema(entry.capability.output_schema),
            output_kind=entry.capability.output_kind,
            data_access=entry.capability.access_mode.value,
            operational_effect=entry.capability.operational_effect.value,
        )
        if entry.routes:
            # A routed tool's top-level schema is its primary route; each other
            # admitted route states only how its input differs from that schema.
            contract.update(
                route_argument=entry.view.mode_argument,
                routes=[
                    {
                        "route": mode,
                        "capability_id": route.capability.id,
                        **_route_input_contract(
                            model_identity_schema(entry.capability.input_schema),
                            model_identity_schema(route.capability.input_schema),
                        ),
                        "output_kind": route.capability.output_kind,
                        "data_access": route.capability.access_mode.value,
                        "operational_effect": route.capability.operational_effect.value,
                    }
                    for mode, route in entry.routes
                ],
            )
    return contract


def _route_input_contract(
    primary: Mapping[str, object], schema: Mapping[str, object]
) -> dict[str, object]:
    """Describe a route's input as changes to the top-level schema, never a copy."""
    if schema == primary:
        return {"input_schema": "same_as_top_level"}
    primary_properties = primary.get("properties")
    properties = schema.get("properties")
    if not isinstance(primary_properties, Mapping) or not isinstance(
        properties, Mapping
    ):
        return {"input_schema": schema}
    changes: dict[str, object] = {
        key: value
        for key, value in schema.items()
        if key != "properties" and primary.get(key) != value
    }
    removed_keywords = sorted(
        key for key in primary if key != "properties" and key not in schema
    )
    if removed_keywords:
        changes["removed_keywords"] = removed_keywords
    changed = {
        name: value
        for name, value in properties.items()
        if primary_properties.get(name) != value
    }
    if changed:
        changes["properties"] = changed
    removed = sorted(name for name in primary_properties if name not in properties)
    if removed:
        changes["removed_properties"] = removed
    return {"input_schema_changes": changes}


def _fit_load_contracts(
    data: dict[str, object],
    entries: tuple[RunToolCatalogEntry, ...],
    limits: LoopLimits,
) -> None:
    """Omit whole schemas deterministically; verification uses the same projection."""
    contracts = cast(list[dict[str, object]], data["contracts"])
    remaining = list(range(len(contracts)))
    while remaining and (
        len(canonical_json(data).encode("utf-8")) > limits.max_toolbox_load_result_bytes
        or _json_depth(data) >= limits.max_tool_result_depth
    ):
        index = max(
            remaining,
            key=lambda i: (
                (
                    _json_depth(contracts[i])
                    if _json_depth(data) >= limits.max_tool_result_depth
                    else 0
                ),
                len(canonical_json(contracts[i]).encode("utf-8")),
            ),
        )
        contracts[index] = _tool_contract_reference(entries[index])
        remaining.remove(index)


def _inspection_page(
    contract: dict[str, object],
    path: str,
    offset: int,
    *,
    maximum_bytes: int,
    maximum_children: int,
    maximum_depth: int,
) -> dict[str, object]:
    """Read a bounded part of one immutable contract; paths cannot address state."""
    value: object = contract
    if path:
        if not path.startswith("/"):
            raise KeyError(path)
        for encoded in path[1:].split("/"):
            if re.search(r"~(?![01])", encoded):
                raise KeyError(path)
            token = encoded.replace("~1", "/").replace("~0", "~")
            if isinstance(value, Mapping) and token in value:
                value = value[token]
            elif isinstance(value, (tuple, list)) and re.fullmatch(
                r"0|[1-9][0-9]*", token
            ):
                index = int(token)
                if index >= len(value):
                    raise KeyError(path)
                value = value[index]
            else:
                raise KeyError(path)
    base = {
        "tool_name": contract["tool_name"],
        "contract_digest": contract["contract_digest"],
        "path": path,
        "offset": offset,
    }
    full = {**base, "complete": True, "value": value}
    if (
        offset == 0
        and len(canonical_json(full).encode("utf-8")) <= maximum_bytes
        and _json_depth(full) <= maximum_depth
    ):
        return full
    if isinstance(value, str):
        if offset >= len(value):
            raise KeyError(path)
        end = min(len(value), offset + maximum_bytes)
        while end > offset:
            page = {
                **base,
                "complete": False,
                "value_type": "string",
                "text": value[offset:end],
                "next_offset": end if end < len(value) else None,
                "total_characters": len(value),
            }
            if (
                len(canonical_json(page).encode("utf-8")) <= maximum_bytes
                and _json_depth(page) <= maximum_depth
            ):
                return page
            end = offset + (end - offset) // 2
        raise ValueError("inspection string fragment cannot fit")
    if isinstance(value, Mapping):
        children = sorted(value.items())
        kind = "object"
    elif isinstance(value, (tuple, list)):
        children = [(str(index), child) for index, child in enumerate(value)]
        kind = "array"
    else:
        raise ValueError("inspection scalar cannot fit")
    if offset >= len(children):
        raise KeyError(path)
    items: list[dict[str, object]] = []
    page = {
        **base,
        "complete": False,
        "value_type": kind,
        "children": items,
        "total_children": len(children),
        "next_offset": offset,
    }
    for key, child in children[offset : offset + maximum_children]:
        child_path = path + "/" + key.replace("~", "~0").replace("/", "~1")
        item = {"path": child_path, "complete": True, "value": child}
        items.append(item)
        page["next_offset"] = (
            offset + len(items) if offset + len(items) < len(children) else None
        )
        if (
            len(canonical_json(page).encode("utf-8")) > maximum_bytes
            or _json_depth(page) > maximum_depth
        ):
            items[-1] = {"path": child_path, "complete": False}
        if (
            len(canonical_json(page).encode("utf-8")) > maximum_bytes
            or _json_depth(page) > maximum_depth
        ):
            items.pop()
            page["next_offset"] = offset + len(items)
            break
    if not items:
        raise ValueError("inspection child reference cannot fit")
    return page


def _manifest_material(
    manifest: tuple[ToolboxManifestEntry, ...],
) -> list[dict[str, object]]:
    return [
        {
            "toolbox_id": item.toolbox_id.value,
            "label": item.label,
            "summary": item.summary,
            "pinned_count": item.pinned_count,
            "on_demand_count": item.on_demand_count,
            "access_modes": tuple(mode.value for mode in item.access_modes),
            "operational_effects": tuple(
                effect.value for effect in item.operational_effects
            ),
        }
        for item in manifest
    ]


def _loaded_entry_material(entry: RunToolCatalogEntry) -> dict[str, object]:
    return {
        "tool_name": entry.view.name,
        "capability_id": entry.capability.id,
        "domain_owner_id": entry.domain_owner_id,
        "executor_id": entry.executor_id,
        "input_schema_digest": entry.input_schema_digest,
        "origin_revision_digest": entry.origin_revision_digest,
        "toolbox_id": entry.toolbox_id.value,
        "load_mode": entry.load_mode.value,
        "text_trust": entry.view.presentation.text_trust.value,
    }


def _activation_digest(
    run_id: str,
    catalog_digest: str,
    entries: tuple[RunToolCatalogEntry, ...],
) -> str:
    return _sha256_digest(
        {
            "run_id": run_id,
            "catalog_digest": catalog_digest,
            "loaded_tools": [_loaded_entry_material(entry) for entry in entries],
        }
    )


def _loaded_tool_receipt(
    *,
    run_id: str,
    catalog_digest: str,
    catalog_entries: tuple[RunToolCatalogEntry, ...],
    messages: tuple[object, ...],
    limits: LoopLimits,
    registry: CapabilityRegistry,
    lane: CallTargetLane | None,
) -> tuple[tuple[RunToolCatalogEntry, ...], str, str]:
    exchanges = _ordered_tool_exchanges(messages)
    loaded: tuple[RunToolCatalogEntry, ...] = ()
    activation_digest = _activation_digest(run_id, catalog_digest, ())
    for call, block in exchanges:
        if call.name != "toolbox_load" or block.is_error:
            continue
        names = call.arguments.get("tool_names")
        if not isinstance(names, (tuple, list)) or not all(
            isinstance(name, str) for name in names
        ):
            continue
        if block.output.get("kind") != "toolbox_load_receipt":
            continue
        data = block.output.get("data")
        provenance = block.sensitivity_provenance
        if not isinstance(data, Mapping) or (
            provenance.get("authority") != "tool_catalog_control"
            or provenance.get("run_id") != run_id
            or provenance.get("catalog_digest") != catalog_digest
            or provenance.get("control_name") != "toolbox_load"
        ):
            continue
        candidate = _verified_loaded_entries(
            data,
            requested_names=tuple(names),
            run_id=run_id,
            catalog_digest=catalog_digest,
            catalog_entries=catalog_entries,
            limits=limits,
            registry=registry,
            lane=lane,
        )
        if candidate is None:
            continue
        loaded = candidate
        activation_digest = _activation_digest(run_id, catalog_digest, loaded)
    return loaded, activation_digest, _toolbox_exchanges_digest(exchanges)


def _ordered_tool_exchanges(
    messages: tuple[object, ...],
) -> tuple[tuple[ToolCall, ToolResultBlock], ...]:
    pending: list[ToolCall] = []
    exchanges: list[tuple[ToolCall, ToolResultBlock]] = []
    for message in messages:
        if not isinstance(message, CanonicalMessage):
            raise TypeError("step transcript must contain CanonicalMessage records")
        if message.role is MessageRole.ASSISTANT:
            if pending:
                raise ValueError("step transcript has incomplete ordered tool results")
            pending = list(message.tool_calls)
            continue
        if message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise TypeError("tool message must contain ToolResultBlock records")
                if not pending:
                    raise ValueError(
                        "step transcript contains an unexpected tool result"
                    )
                call = pending.pop(0)
                if block.call_id != call.id:
                    raise ValueError("step transcript tool results are out of order")
                exchanges.append((call, block))
            continue
        if pending:
            raise ValueError("step transcript has incomplete ordered tool results")
    if pending:
        raise ValueError("step transcript has incomplete ordered tool results")
    return tuple(exchanges)


def _toolbox_transcript_digest(messages: tuple[object, ...]) -> str:
    """Digest ordered toolbox-load exchanges without treating call IDs as global."""

    return _toolbox_exchanges_digest(_ordered_tool_exchanges(messages))


def _toolbox_exchanges_digest(
    exchanges: tuple[tuple[ToolCall, ToolResultBlock], ...],
) -> str:
    return _sha256_digest(
        [
            {
                "call": {
                    "id": call.id,
                    "provider_call_id": call.provider_call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                },
                "result": {
                    "call_id": result.call_id,
                    "output": result.output,
                    "is_error": result.is_error,
                    "sensitivity": (
                        None if result.sensitivity is None else result.sensitivity.value
                    ),
                    "sensitivity_provenance": result.sensitivity_provenance,
                    "capability_id": result.capability_id,
                    "executor_id": result.executor_id,
                },
            }
            for call, result in exchanges
            if call.name == "toolbox_load"
        ]
    )


def _verified_loaded_entries(
    data: Mapping[str, object],
    *,
    requested_names: tuple[str, ...],
    run_id: str,
    catalog_digest: str,
    catalog_entries: tuple[RunToolCatalogEntry, ...],
    limits: LoopLimits,
    registry: CapabilityRegistry,
    lane: CallTargetLane | None,
) -> tuple[RunToolCatalogEntry, ...] | None:
    receipt_keys = {
        "run_id",
        "catalog_digest",
        "loaded_names",
        "definition_bytes",
        "activation_digest",
        "contracts",
    }
    if set(data) not in (receipt_keys, receipt_keys | {"already_pinned_names"}):
        return None
    by_name = {entry.view.name: entry for entry in catalog_entries}
    pinned_names = tuple(
        sorted(
            name
            for name in requested_names
            if name in by_name and by_name[name].load_mode is not ToolLoadMode.ON_DEMAND
        )
    )
    pinned_value = data.get("already_pinned_names", ())
    if (
        not isinstance(pinned_value, (tuple, list))
        or tuple(pinned_value) != pinned_names
        or ("already_pinned_names" in data and not pinned_names)
    ):
        return None
    names_value = data.get("loaded_names")
    if not isinstance(names_value, (tuple, list)):
        return None
    names = tuple(name for name in names_value if isinstance(name, str))
    if (
        len(names) != len(names_value)
        or names
        != tuple(sorted(name for name in requested_names if name not in pinned_names))
        or len(names) != len(set(names))
        or len(names) > limits.max_loaded_tools
        or data.get("run_id") != run_id
        or data.get("catalog_digest") != catalog_digest
    ):
        return None
    try:
        entries = tuple(by_name[name] for name in names)
    except KeyError:
        return None
    if any(
        entry.load_mode is not ToolLoadMode.ON_DEMAND
        or not _entry_resolves_exactly(entry, registry)
        for entry in entries
    ):
        return None
    definitions = tuple(_entry_definition(registry, entry, lane) for entry in entries)
    definition_bytes = _definition_bytes(definitions)
    expected_activation = _activation_digest(run_id, catalog_digest, entries)
    expected_data: dict[str, object] = dict(data)
    expected_data["contracts"] = [
        _tool_contract(entry, include_schemas=False) for entry in entries
    ]
    _fit_load_contracts(expected_data, entries, limits)
    if (
        data.get("definition_bytes") != definition_bytes
        or canonical_json(data.get("contracts"))
        != canonical_json(expected_data["contracts"])
        or definition_bytes > limits.max_loaded_tool_definition_bytes
        or data.get("activation_digest") != expected_activation
        or len(canonical_json(data).encode("utf-8"))
        > limits.max_toolbox_load_result_bytes
    ):
        return None
    return entries


def _toolbox_search_score(query: str, entry: RunToolCatalogEntry) -> int:
    normalized = query.strip().lower()
    terms = tuple(_TOKEN.findall(normalized))
    if not terms:
        return 0
    score = 0
    if normalized == entry.view.name:
        score += 10_000
    phrases = (
        entry.view.name.replace("_", " "),
        entry.view.presentation.summary.lower(),
        entry.view.presentation.when_to_use.lower(),
        *entry.view.presentation.keywords,
    )
    if normalized in phrases:
        score += 8_000
    weighted_fields = (
        (entry.view.name, 100),
        (entry.capability.id, 50),
        (entry.view.presentation.summary, 20),
        (entry.view.presentation.when_to_use, 10),
        (" ".join(entry.view.presentation.keywords), 30),
        (" ".join(entry.parameter_names), 15),
        (
            (
                canonical_json(entry.view.connector_presentation)
                if entry.view.connector_presentation is not None
                else ""
            ),
            20,
        ),
    )
    for text, weight in weighted_fields:
        tokens = set(_TOKEN.findall(text.lower()))
        score += weight * sum(term in tokens for term in terms)
    return score


def _toolbox_search_match(
    score: int,
    entry: RunToolCatalogEntry,
    *,
    loaded_names: set[str],
) -> dict[str, object]:
    load_state = (
        ToolLoadMode.PINNED.value
        if entry.load_mode is ToolLoadMode.PINNED
        else ("loaded" if entry.view.name in loaded_names else "on_demand")
    )
    match: dict[str, object] = {
        "tool_name": entry.view.name,
        "toolbox_id": entry.toolbox_id.value,
        "capability_id": entry.capability.id,
        "summary": entry.view.presentation.summary,
        "when_to_use": entry.view.presentation.when_to_use,
        "text_trust": entry.view.presentation.text_trust.value,
        "load_state": load_state,
        "data_access": entry.capability.access_mode.value,
        "operational_effect": entry.capability.operational_effect.value,
        "score": score,
        "match_status": "matched" if score > 0 else "unmatched_fallback",
    }
    match["inspection_tool"] = "toolbox_inspect"
    return match


def _projection_digest(
    *,
    run_id: str,
    registry_digest: str,
    catalog_digest: str,
    transcript_digest: str,
    provider_definitions: tuple[ToolDefinition, ...],
    callable_entries: tuple[RunToolCatalogEntry, ...],
    loaded_entries: tuple[RunToolCatalogEntry, ...],
    activation_digest: str,
    call_target_lane: CallTargetLane | None,
    source_scope: EffectiveSourceScope | None = None,
) -> str:
    return _sha256_digest(
        {
            "run_id": run_id,
            "registry_digest": registry_digest,
            "catalog_digest": catalog_digest,
            "transcript_digest": transcript_digest,
            "provider_definitions": [
                _definition_material(item) for item in provider_definitions
            ],
            "callable_tools": [entry.view.name for entry in callable_entries],
            "loaded_tools": [entry.view.name for entry in loaded_entries],
            "activation_digest": activation_digest,
            "call_target_lane": (
                None if call_target_lane is None else call_target_lane.value
            ),
            "source_scope": None if source_scope is None else source_scope.to_mapping(),
        }
    )


def _toolbox_observation(entry: RunToolCatalogEntry) -> dict[str, object]:
    return {
        "toolbox_id": entry.toolbox_id.value,
        "load_mode": entry.load_mode.value,
        "provider_state": (
            "pinned" if entry.load_mode is ToolLoadMode.PINNED else "loaded"
        ),
    }


def _entry_definition(
    registry: CapabilityRegistry,
    entry: RunToolCatalogEntry,
    lane: CallTargetLane | None,
) -> ToolDefinition:
    """One provider definition describing only the routes this entry admits."""

    return registry.tool_definition(
        entry.view.name,
        lane=lane,
        routes=tuple(mode for mode, _ in entry.routes) if entry.routes else None,
    )
