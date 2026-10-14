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
from ...loop.analysis import AnalysisEvidence
from ...loop.models import RunInput, Transcript

COUNT_CAPABILITY_ID = "data.count_rows"
COUNT_OPERATION_CAPABILITY_ID = "data.count_selection"
COUNT_INSPECT_CAPABILITY_ID = "data.count_inspect"
COUNT_SELECTION_TOOL_NAME = "data_count_selection"
COUNT_PLAN_TOOL_NAME = "data_count_plan"


def count_operation_calls(
    arguments: Mapping[str, object], width: int
) -> tuple[FrozenJsonObject, ...]:
    """A finite plan over the exact admitted selection, never a wildcard scan."""
    ids = cast(tuple[str, ...], arguments["resource_ids"])
    return tuple(
        FrozenJsonObject.from_mapping(
            {
                "name": "data_count_rows",
                "arguments": {
                    "source_id": arguments["source_id"],
                    "resource_ids": ids[offset : offset + width],
                    "include_views": arguments["include_views"],
                },
            }
        )
        for offset in range(0, len(ids), width)
    )


def reduce_count_operation(
    run_id: str,
    call_id: str,
    arguments: Mapping[str, object],
    records: tuple[AnalysisEvidence, ...],
    *,
    width: int,
    observed_at: str,
    duration_ms: int,
    deadline_reached: bool,
) -> dict[str, object]:
    """Reduce authenticated child observations against an immutable denominator."""
    selection = cast(FrozenJsonObject, arguments["_count_selection"])
    expected_calls = count_operation_calls(arguments, width)
    expected = {
        r["resource_id"]: r["revision"]
        for r in cast(tuple[FrozenJsonObject, ...], selection["resources"])
    }
    counted: dict[str, int] = {}
    failed = 0
    interrupted = 0
    failures = []
    ids = []
    seen = set()
    for record in sorted(records, key=lambda r: cast(int, r.facts["sequence"])):
        facts = cast(FrozenJsonObject, record.facts)
        sequence = cast(int, facts["sequence"])
        if (
            record.run_id != run_id
            or record.kind != "child"
            or facts["parent_call_id"] != call_id
            or sequence in seen
            or not 1 <= sequence <= len(expected_calls)
            or facts["tool_name"] != "data_count_rows"
            or facts["arguments"] != expected_calls[sequence - 1]["arguments"]
            or facts["capability_id"] != COUNT_CAPABILITY_ID
            or facts["executor_id"] != "data.count_rows.executor"
        ):
            raise ToolOutputValidationError("Count operation child identity changed")
        seen.add(sequence)
        ids.append(record.evidence_id)
        batch = cast(
            tuple[str, ...], cast(FrozenJsonObject, facts["arguments"])["resource_ids"]
        )
        if facts["status"] != "succeeded":
            if facts["status"] == "failed":
                failed += len(batch)
                error = cast(Mapping[str, object], facts["result"]).get("error", {})
                code = cast(Mapping[str, object], error).get("code", "count_failed")
            else:
                interrupted += len(batch)
                code = (
                    "count_operation_deadline"
                    if deadline_reached
                    else "count_interrupted"
                )
            failures.append(
                {
                    "evidence_id": record.evidence_id,
                    "code": code,
                    "resource_count": len(batch),
                    "dispatched": facts["dispatched"],
                }
            )
            continue
        data = cast(FrozenJsonObject, cast(FrozenJsonObject, facts["result"])["data"])
        rows = cast(tuple[FrozenJsonObject, ...], data["counts"])
        if (
            data["source_id"] != arguments["source_id"]
            or data["source_revision"] != selection["source_revision"]
            or tuple(r["resource_id"] for r in rows) != batch
            or any(r["revision"] != expected[r["resource_id"]] for r in rows)
            or data["total_rows"] != sum(cast(int, r["rows"]) for r in rows)
            or any(r["resource_id"] in counted for r in rows)
        ):
            raise ToolOutputValidationError(
                "Count operation result differs from its plan"
            )
        counted.update(
            (cast(str, r["resource_id"]), cast(int, r["rows"])) for r in rows
        )
    complete = len(counted) == len(expected)
    observed = sum(counted.values())
    return {
        "operation_ref": count_reference("op_", run_id, call_id),
        "selection_ref": selection["reference"],
        "plan_sha256": "sha256:"
        + sha256(
            canonical_json(
                {
                    "source_id": arguments["source_id"],
                    "selection": selection,
                    "batch_width": width,
                }
            ).encode()
        ).hexdigest(),
        "source_id": arguments["source_id"],
        "source_name": arguments["_source_name"],
        "source_revision": selection["source_revision"],
        "status": "completed" if complete else "partial" if counted else "failed",
        "total_rows": observed if complete else None,
        "observed_total_rows": observed,
        "selected_resources": len(expected),
        "counted_resources": len(counted),
        "failed_resources": failed,
        "interrupted_resources": interrupted,
        "unexecuted_resources": len(expected) - len(counted) - failed - interrupted,
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
        "catalog_enumeration_complete": selection["complete_source_enumeration"],
        "snapshot_scope": "separate_batch_snapshots",
        "scope_limitation": "Fixed admitted catalog selection only; no complete physical-database or shared-snapshot claim. Views may overlap tables.",
        "batch_max_resources": width,
        "planned_batches": len(expected_calls),
        "evidence_call_ids": ids,
        "failures": failures,
        "deadline_reached": deadline_reached,
        "observed_at": observed_at,
        "duration_ms": duration_ms,
    }


def count_reference(prefix: str, run_id: str, call_id: str) -> str:
    return (
        prefix
        + sha256(canonical_json((run_id, call_id, prefix)).encode()).hexdigest()[:16]
    )


def count_operation_failure_details(data: Mapping[str, object]) -> dict[str, object]:
    """Project bounded authenticated child causes into an actionable graph block.

    The reducer bounds failures by its finite policy-derived batch plan. Only
    normalized codes and dispatch facts are carried forward, never source payloads.
    """
    failures = cast(tuple[Mapping[str, object], ...], data["failures"])
    codes = tuple(dict.fromkeys(cast(str, f["code"])[:128] for f in failures))
    cause_set = set(codes)
    if cause_set and cause_set <= {
        "catalog_source_stale",
        "count_binding_stale",
        "execution_contract_changed",
        "execution_contract_unavailable",
    }:
        control_kind = "source_or_contract_drift"
        constraint = "The frozen source schema or execution contract changed. Block this task; a fresh caller-admitted plan is required. Do not retry or substitute this stale selection."
    elif cause_set and cause_set <= {
        "source_scope_violation",
        "resource_read_not_allowed",
        "source_permission_state_invalid",
    }:
        control_kind = "needs_authorization"
        constraint = "The fixed selection is no longer admitted by current scope or permission. Block this task; only caller admission can restore authority."
    elif (
        data["deadline_reached"]
        or cause_set
        and cause_set
        <= {
            "count_operation_deadline",
            "tool_calls_per_run_exceeded",
            "analysis_child_budget_exhausted",
            "analysis_batch_result_budget_exhausted",
        }
    ):
        control_kind = "budget_exhausted"
        constraint = "The bounded count operation exhausted its execution allowance. Block this task without extending the allowance."
    else:
        control_kind = "needs_replan"
        constraint = "The fixed count selection could not complete its bounded reads. Block this task and report the retained failure codes; no complete total is available."
    constraint += " The framework did not automatically replay failed child reads. Preserve genuine observations; they do not establish a complete total."
    counts = {
        name: data[name]
        for name in (
            "selected_resources",
            "counted_resources",
            "failed_resources",
            "interrupted_resources",
            "unexecuted_resources",
            "observed_total_rows",
        )
    }
    child_dispatched = bool(data["counted_resources"]) or any(
        f["dispatched"] for f in failures
    )
    return {
        **counts,
        "total_rows": None,
        "coverage": data["coverage"],
        "execution_disposition": (
            "partially_executed" if child_dispatched else "not_dispatched"
        ),
        "child_execution_dispatched": child_dispatched,
        "deadline_reached": data["deadline_reached"],
        "failure_codes": codes,
        "failures": tuple({**f, "code": cast(str, f["code"])[:128]} for f in failures),
        "evidence_call_ids": data["evidence_call_ids"],
        "recovery": {
            "kind": "block_task",
            "tool_name": "task_block",
            "constraint": constraint,
            "arguments": {
                "kind": control_kind,
                "message": constraint,
                "details": {**counts, "failure_codes": codes, "total_rows": None},
            },
        },
    }


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
    *,
    selection_arguments: Mapping[str, object] | None = None,
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

    reference_arguments = (
        call.arguments if selection_arguments is None else selection_arguments
    )
    cursor = reference_arguments.get("cursor")
    reference = reference_arguments.get("selection_ref")
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
        else reference_arguments.get("include_views", False)
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
