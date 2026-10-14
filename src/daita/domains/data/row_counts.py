"""Authenticate bounded count selections and continuations from the exact run.

References address existing transcript evidence. They carry no authority and
retain no mutable cursor registry or resumable execution state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import cast

from ..._json import FrozenJsonObject, canonical_json
from ...capabilities import (
    CapabilityInputError,
    CapabilityRegistry,
    OperationalEffect,
    ToolOutput,
    ToolOutputValidationError,
)
from ...catalog.capabilities import CATALOG_SCHEMA_CAPABILITY_ID
from ...catalog.models import CATALOG_SCHEMA_MAX_RESOURCE_IDS
from ...llm.models import ModelSensitivity, ToolCall, ToolResultBlock
from ...loop.models import RunInput, Transcript

COUNT_CAPABILITY_ID = "data.count_rows"


def count_reference(prefix: str, run_id: str, call_id: str) -> str:
    return (
        prefix
        + sha256(canonical_json((run_id, call_id, prefix)).encode()).hexdigest()[:16]
    )


def selection_error(message: str) -> CapabilityInputError:
    return CapabilityInputError(
        "count_selection_invalid",
        message,
        {
            "execution_disposition": "not_dispatched",
            "recovery": {
                "kind": "correct_arguments",
                "tool_name": "catalog_schema",
                "constraint": "Use an exact selection reference or next cursor from a successful tool result in this run.",
            },
        },
    )


def attach_count_selection(
    run: RunInput, call: ToolCall, output: ToolOutput
) -> ToolOutput:
    """Only identity evidence for one explicit source supplies a count selection."""
    if call.arguments.get("detail") != "identity":
        return output
    source_id = call.arguments.get("source_id")
    resources = cast(tuple[FrozenJsonObject, ...], output.data["resources"])
    if (
        not isinstance(source_id, str)
        or not resources
        or any(
            item["source_id"] != source_id or item["kind"] not in {"table", "view"}
            for item in resources
        )
    ):
        return output
    selection = cast(FrozenJsonObject, output.data["selection"])
    complete = (
        not call.arguments.get("query")
        and not call.arguments.get("resource_ids")
        and not cast(FrozenJsonObject, output.data["truncation"])["resources"]
        and not selection.get("unresolved_terms")
        and len(resources) == output.data["total_matches"]
    )
    return replace(
        output,
        data=FrozenJsonObject.from_mapping(
            {
                **output.data,
                "selection": {
                    **selection,
                    "count_selection_ref": count_reference("sel_", run.id, call.id),
                    "base_tables": sum(item["kind"] == "table" for item in resources),
                    "views": sum(item["kind"] == "view" for item in resources),
                    "complete_source_enumeration": complete,
                },
            }
        ),
    )


@dataclass(frozen=True)
class CountSelection:
    reference: str
    source_id: str
    source_revision: str
    resources: tuple[FrozenJsonObject, ...]
    complete_source_enumeration: bool
    include_views: bool
    prior_counts: tuple[FrozenJsonObject, ...]
    evidence_call_ids: tuple[str, ...]
    sensitivity: ModelSensitivity

    def arguments(self, width: int) -> FrozenJsonObject:
        offset = len(self.prior_counts)
        batch = self.resources[offset : offset + width]
        if not batch:
            raise selection_error("No countable resources remain in this selection.")
        return FrozenJsonObject.from_mapping(
            {
                "source_id": self.source_id,
                "resource_ids": tuple(item["resource_id"] for item in batch),
                "include_views": self.include_views,
                "_count_selection": {
                    "reference": self.reference,
                    "source_revision": self.source_revision,
                    "resources": self.resources,
                    "complete_source_enumeration": self.complete_source_enumeration,
                    "prior_counts": self.prior_counts,
                    "evidence_call_ids": self.evidence_call_ids,
                    "sensitivity": self.sensitivity.value,
                },
            }
        )


async def authenticate_count_selection(
    transcript: Transcript,
    run: RunInput,
    call: ToolCall,
    registry: CapabilityRegistry,
    width: int,
) -> CountSelection:
    """Resolve one immutable cursor chain; never trust model-authored aggregates."""
    if transcript.run.id != run.id or transcript.run.agent_id != run.agent_id:
        raise selection_error("Count references belong to another run or agent.")
    producers: dict[str, list[tuple[int, ToolCall]]] = {}
    results: dict[str, list[tuple[int, ToolResultBlock]]] = {}
    for index, message in enumerate(transcript.messages):
        for candidate in message.tool_calls:
            producers.setdefault(candidate.id, []).append((index, candidate))
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                results.setdefault(block.call_id, []).append((index, block))
    current = producers.get(call.id, ())
    if len(current) != 1 or current[0][1] != call:
        raise selection_error(
            "The count call is not its exact persisted current-run call."
        )
    current_index = current[0][0]
    prefix = {
        key: (producer[0][0], producer[0][1], results[key][0][0], results[key][0][1])
        for key, producer in producers.items()
        if len(producer) == 1
        and len(results.get(key, ())) == 1
        and producer[0][0] < results[key][0][0] < current_index
    }

    async def evidence(key: str, capability_id: str, before: int):
        item = prefix.get(key)
        if item is None:
            raise selection_error(
                "Count evidence is missing, duplicated, or outside this run."
            )
        producer_index, producer, result_index, block = item
        if result_index >= before or block.is_error or block.sensitivity is None:
            raise selection_error(
                "Count evidence is failed or not earlier than this call."
            )
        try:
            _, capability, owner = registry.resolve_tool_owner(producer.name)
            if (
                owner != "data"
                or capability.id != capability_id
                or capability.operational_effect is not OperationalEffect.NONE
                or block.capability_id != capability.id
                or block.executor_id != capability.executor_id
                or block.output_sha256
                != "sha256:" + sha256(canonical_json(block.output).encode()).hexdigest()
            ):
                raise ValueError("count evidence identity changed")
            await registry.validate_arguments_async(capability.id, producer.arguments)
            data = cast(FrozenJsonObject, block.output["data"])
            registry.validate_output(
                capability.id,
                ToolOutput(kind=cast(str, block.output["kind"]), data=data),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            CapabilityInputError,
            ToolOutputValidationError,
        ):
            raise selection_error(
                "Count evidence does not authenticate its current registered contract."
            ) from None
        return producer_index, producer, block, data

    cursor = call.arguments.get("cursor")
    reference = call.arguments.get("selection_ref")
    chain: list[tuple[ToolCall, ToolResultBlock, FrozenJsonObject]] = []
    before = current_index
    while cursor is not None:
        keys = [key for key in prefix if count_reference("cnt_", run.id, key) == cursor]
        if len(keys) != 1 or len(chain) >= CATALOG_SCHEMA_MAX_RESOURCE_IDS:
            raise selection_error("Unknown or invalid count continuation cursor.")
        before, producer, block, data = await evidence(
            keys[0], COUNT_CAPABILITY_ID, before
        )
        if data.get("next_cursor") != cursor:
            raise selection_error("This result did not issue a continuation cursor.")
        chain.append((producer, block, data))
        cursor = producer.arguments.get("cursor")
        reference = producer.arguments.get("selection_ref")
    keys = [key for key in prefix if count_reference("sel_", run.id, key) == reference]
    if len(keys) != 1:
        raise selection_error(
            "Unknown count selection reference; discover a selection in this run."
        )
    _, producer, block, data = await evidence(
        keys[0], CATALOG_SCHEMA_CAPABILITY_ID, before
    )
    selection = cast(FrozenJsonObject, data["selection"])
    if (
        producer.arguments.get("detail") != "identity"
        or selection.get("count_selection_ref") != reference
    ):
        raise selection_error("The catalog result did not issue this count selection.")
    source_id = producer.arguments.get("source_id")
    if not isinstance(source_id, str):
        raise selection_error("A count selection must bind one exact source.")
    source = next(
        (
            item
            for item in cast(tuple[FrozenJsonObject, ...], data["sources"])
            if item["source_id"] == source_id
        ),
        None,
    )
    if source is None or not isinstance(source.get("source_revision"), str):
        raise selection_error("Count selection has no exact source revision.")
    include_views = (
        chain[-1][0].arguments.get("include_views", False)
        if chain
        else call.arguments.get("include_views", False)
    )
    resources = tuple(
        item
        for item in cast(tuple[FrozenJsonObject, ...], data["resources"])
        if item["kind"] == "table" or (include_views and item["kind"] == "view")
    )
    if len(resources) > CATALOG_SCHEMA_MAX_RESOURCE_IDS or any(
        item["source_id"] != source_id for item in resources
    ):
        raise selection_error("Count selection exceeds its bounded catalog source.")
    if len({item["resource_id"] for item in resources}) != len(resources):
        raise selection_error("Count selection contains duplicate resources.")
    prior: list[FrozenJsonObject] = []
    calls = []
    sensitivity = block.sensitivity
    assert sensitivity is not None
    ranks = tuple(ModelSensitivity)
    for producer, block, page in reversed(chain):
        batch = resources[len(prior) : len(prior) + width]
        rows = cast(tuple[FrozenJsonObject, ...], page["counts"])
        if (
            page.get("selection_ref") != reference
            or page["source_id"] != source_id
            or page["source_revision"] != source["source_revision"]
            or tuple((item["resource_id"], item["revision"]) for item in rows)
            != tuple((item["resource_id"], item["revision"]) for item in batch)
            or not rows
            or any(
                not isinstance(item["rows"], int)
                or isinstance(item["rows"], bool)
                or item["rows"] < 0
                for item in rows
            )
            or page["total_rows"] != sum(cast(int, item["rows"]) for item in rows)
        ):
            raise selection_error(
                "Count continuation does not cover its exact bounded batch."
            )
        prior.extend(rows)
        calls.append(producer.id)
        assert block.sensitivity is not None
        sensitivity = max((sensitivity, block.sensitivity), key=ranks.index)
    return CountSelection(
        cast(str, reference),
        source_id,
        cast(str, source["source_revision"]),
        resources,
        selection.get("complete_source_enumeration") is True,
        include_views is True,
        tuple(prior),
        tuple(calls),
        sensitivity,
    )


def finalize_count_page(
    run: RunInput, call: ToolCall, arguments: FrozenJsonObject, output: ToolOutput
) -> ToolOutput:
    selection = arguments.get("_count_selection")
    if not isinstance(selection, FrozenJsonObject):
        return output
    resources = cast(tuple[FrozenJsonObject, ...], selection["resources"])
    prior = cast(tuple[FrozenJsonObject, ...], selection["prior_counts"])
    rows = cast(tuple[FrozenJsonObject, ...], output.data["counts"])
    expected = resources[len(prior) : len(prior) + len(rows)]
    if (
        not rows
        or output.data["source_revision"] != selection["source_revision"]
        or tuple((item["resource_id"], item["revision"]) for item in rows)
        != tuple((item["resource_id"], item["revision"]) for item in expected)
        or tuple(item["resource_id"] for item in rows) != arguments["resource_ids"]
        or output.data["total_rows"] != sum(cast(int, item["rows"]) for item in rows)
    ):
        raise ToolOutputValidationError(
            "Count output does not cover the bound continuation batch"
        )
    counted = len(prior) + len(rows)
    complete = counted == len(resources)
    return replace(
        output,
        data=FrozenJsonObject.from_mapping(
            {
                **output.data,
                "selection_ref": selection["reference"],
                "next_cursor": (
                    None if complete else count_reference("cnt_", run.id, call.id)
                ),
                "progress": {
                    "selected_resources": len(resources),
                    "counted_resources": counted,
                    "remaining_resources": len(resources) - counted,
                    "total_rows": sum(
                        cast(int, item["rows"]) for item in (*prior, *rows)
                    ),
                    "coverage": (
                        "complete_admitted_base_tables"
                        if complete
                        and selection["complete_source_enumeration"]
                        and not arguments["include_views"]
                        else (
                            "complete_catalog_selection"
                            if complete
                            else "partial_catalog_selection"
                        )
                    ),
                    "catalog_enumeration_complete": selection[
                        "complete_source_enumeration"
                    ],
                    "snapshot_scope": "separate_batch_snapshots",
                    "evidence_call_ids": (
                        *cast(tuple[str, ...], selection["evidence_call_ids"]),
                        call.id,
                    ),
                },
            }
        ),
    )
