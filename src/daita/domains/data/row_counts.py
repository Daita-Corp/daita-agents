"""Authenticate bounded count selections and continuations from the exact run.

References address existing transcript evidence. They carry no authority and
retain no mutable cursor registry or resumable execution state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import TypedDict, cast

from ..._json import FrozenJsonObject, canonical_json
from ...capabilities import (
    CapabilityInputError,
    CapabilityRegistry,
    OperationalEffect,
    ToolOutput,
    ToolOutputValidationError,
    validate_tool_schema_value,
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
COUNT_BATCH_CAPABILITY_ID = "data.count_batch"
COUNT_FINALIZE_CAPABILITY_ID = "data.finalize_count_comparison"


class CountPlanResource(TypedDict):
    source_id: str
    resource_id: str
    revision: str
    kind: str


class CountPlanSelection(TypedDict):
    source_id: str
    source_revision: str
    resources: tuple[CountPlanResource, ...]
    complete_source_enumeration: bool


class CountBatch(TypedDict):
    batch_id: str
    source_id: str
    resource_ids: tuple[str, ...]


def count_comparison_limits(
    *, batch_width: int, read_seconds: int = 30, job_seconds: int = 300
) -> Mapping[str, int]:
    from ...jobs.graph.models import MAX_TASK_RESULT_BYTES, GraphLimits

    graph = GraphLimits()
    finalizer_attempts = min(3, graph.max_attempts_per_task)
    return cast(
        Mapping[str, int],
        FrozenJsonObject.from_mapping(
            {
                "minimum_sources": 2,
                "maximum_sources": 16,
                "maximum_resources_per_source": CATALOG_SCHEMA_MAX_RESOURCE_IDS,
                "batch_width": batch_width,
                "maximum_batches": min(graph.max_direct_parents, graph.max_tasks - 1),
                "minimum_deadline_seconds": 60,
                "maximum_deadline_seconds": job_seconds,
                "read_deadline_seconds": read_seconds,
                "maximum_result_bytes": MAX_TASK_RESULT_BYTES,
                "finalizer_attempts": finalizer_attempts,
                "automatic_source_retries": 0,
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class CountComparisonPlan:
    """Required coverage, independent of the job's authorization ceiling."""

    material: FrozenJsonObject

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "material", FrozenJsonObject.from_mapping(self.material)
        )

    @property
    def selections(self) -> tuple[CountPlanSelection, ...]:
        return cast(tuple[CountPlanSelection, ...], self.material["selections"])

    @property
    def batches(self) -> tuple[CountBatch, ...]:
        return cast(tuple[CountBatch, ...], self.material["batches"])

    @property
    def limits(self) -> Mapping[str, int]:
        return cast(Mapping[str, int], self.material["limits"])

    @property
    def digest(self) -> str:
        return cast(str, self.material["plan_digest"])

    @classmethod
    def create(
        cls,
        selections: tuple[Mapping[str, object], ...],
        *,
        batch_width: int,
        read_seconds: int = 30,
        job_seconds: int = 300,
    ) -> CountComparisonPlan:
        limits = count_comparison_limits(
            batch_width=batch_width, read_seconds=read_seconds, job_seconds=job_seconds
        )
        if (
            type(batch_width) is not int
            or not 1 <= batch_width <= 4
            or not 1 <= read_seconds <= 30
            or not 60 <= job_seconds <= 300
            or not 2 <= len(selections) <= limits["maximum_sources"]
        ):
            raise ValueError("Count comparison exceeds its source or execution bounds")
        frozen = tuple(
            FrozenJsonObject.from_mapping(s)
            for s in sorted(selections, key=lambda s: str(s.get("source_id", "")))
        )
        sources: set[str] = set()
        resources: set[str] = set()
        batches: list[dict[str, object]] = []
        canonical_selections = []
        for selection in frozen:
            if set(selection) != {
                "source_id",
                "source_revision",
                "resources",
                "complete_source_enumeration",
            }:
                raise ValueError("Count comparison selection fields changed")
            source = selection["source_id"]
            revision = selection["source_revision"]
            rows = selection["resources"]
            if (
                not isinstance(source, str)
                or not source
                or source in sources
                or not isinstance(revision, str)
                or not revision
                or selection["complete_source_enumeration"] is not True
                or not isinstance(rows, tuple)
                or not 1 <= len(rows) <= CATALOG_SCHEMA_MAX_RESOURCE_IDS
            ):
                raise ValueError(
                    "Count comparison requires distinct complete nonempty sources"
                )
            sources.add(source)
            for row in rows:
                if (
                    not isinstance(row, FrozenJsonObject)
                    or set(row) != {"source_id", "resource_id", "revision", "kind"}
                    or row["source_id"] != source
                    or row["kind"] != "table"
                    or not isinstance(row["resource_id"], str)
                    or not row["resource_id"]
                    or row["resource_id"] in resources
                    or not isinstance(row["revision"], str)
                    or not row["revision"]
                ):
                    raise ValueError("Count comparison resource identity changed")
                resources.add(row["resource_id"])
            rows = tuple(
                sorted(
                    cast(tuple[FrozenJsonObject, ...], rows),
                    key=lambda r: cast(str, r["resource_id"]),
                )
            )
            canonical_selections.append({**selection, "resources": rows})
            for offset in range(0, len(rows), batch_width):
                batches.append(
                    {
                        "batch_id": f"batch:{len(batches) + 1}",
                        "source_id": source,
                        "resource_ids": tuple(
                            r["resource_id"]
                            for r in rows[offset : offset + batch_width]
                        ),
                    }
                )
        if len(batches) > limits["maximum_batches"]:
            raise ValueError(
                "Count comparison exceeds the total batch/finalizer fan-in bound"
            )
        body = {
            "kind": "count_comparison",
            "required_coverage": "complete_admitted_base_tables",
            "include_views": False,
            "selections": canonical_selections,
            "batches": batches,
            "limits": {
                **limits,
                "work_units": len(batches) + limits["finalizer_attempts"],
            },
        }
        return cls(
            FrozenJsonObject.from_mapping({**body, "plan_digest": _count_digest(body)})
        )

    @classmethod
    def from_material(cls, material: Mapping[str, object]) -> CountComparisonPlan:
        value = FrozenJsonObject.from_mapping(material)
        try:
            limits = cast(FrozenJsonObject, value["limits"])
            rebuilt = cls.create(
                cast(tuple[FrozenJsonObject, ...], value["selections"]),
                batch_width=cast(int, limits["batch_width"]),
                read_seconds=cast(int, limits["read_deadline_seconds"]),
                job_seconds=cast(int, limits["maximum_deadline_seconds"]),
            )
            if rebuilt.material != value:
                raise ValueError("Count comparison plan changed")
            return rebuilt
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Count comparison plan is malformed") from error


def _count_digest(value: object) -> str:
    return "sha256:" + sha256(canonical_json(value).encode()).hexdigest()


def validate_count_batch(
    data: Mapping[str, object],
    *,
    source_id: str,
    source_revision: str,
    resources: tuple[Mapping[str, object], ...],
) -> dict[str, int]:
    """Shared exact arithmetic and coverage validation for count observations."""
    rows = data.get("counts")
    if (
        not isinstance(rows, tuple)
        or len(rows) != len(resources)
        or data.get("source_id") != source_id
        or data.get("source_revision") != source_revision
    ):
        raise ToolOutputValidationError(
            "Count result differs from its exact source/batch"
        )
    counted: dict[str, int] = {}
    for row, expected in zip(rows, resources, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"resource_id", "revision", "rows"}
            or row["resource_id"] != expected["resource_id"]
            or row["revision"] != expected["revision"]
            or type(row["rows"]) is not int
            or row["rows"] < 0
            or row["resource_id"] in counted
        ):
            raise ToolOutputValidationError(
                "Count result has invalid or duplicate values"
            )
        counted[cast(str, row["resource_id"])] = cast(int, row["rows"])
    if type(data.get("total_rows")) is not int or data["total_rows"] != sum(
        counted.values()
    ):
        raise ToolOutputValidationError("Count batch subtotal changed")
    return counted


def reduce_count_comparison(
    plan: CountComparisonPlan,
    *,
    job_id: str,
    agent_id: str,
    batch_tasks: tuple[Mapping[str, object], ...],
    results: tuple[Mapping[str, object], ...],
    complete: bool = True,
) -> dict[str, object]:
    """Reduce only immutable accepted output envelopes; never repair evidence."""
    from ...jobs.graph.models import TaskResult
    from .capabilities import data_count_rows_capability_declarations

    plan = CountComparisonPlan.from_material(plan.material)
    output_schema = (
        data_count_rows_capability_declarations().capabilities[0].output_schema
    )
    if (
        len(batch_tasks) != len(plan.batches)
        or tuple(m.get("batch_id") for m in batch_tasks)
        != tuple(b["batch_id"] for b in plan.batches)
        or len({m.get("task_id") for m in batch_tasks}) != len(batch_tasks)
    ):
        raise ToolOutputValidationError("Count comparison task mapping changed")
    by_task = {
        m["task_id"]: (b, m) for b, m in zip(plan.batches, batch_tasks, strict=True)
    }
    selections = {s["source_id"]: s for s in plan.selections}
    counted: dict[str, dict[str, int]] = {cast(str, s): {} for s in selections}
    accepted = []
    seen: set[str] = set()
    observations = []
    for envelope in results:
        try:
            result = TaskResult.from_candidate_material(envelope)
            batch, mapping = by_task[result.task_id]
            selection = selections[batch["source_id"]]
            verification = result.verification
            validate_tool_schema_value(output_schema, result.payload)
            if (
                result.task_id in seen
                or result.job_id != job_id
                or result.agent_id != agent_id
                or result.result_id != f"result:{result.attempt_id}"
                or result.result_kind != "data.row_counts"
                or result.schema_digest != mapping["schema_digest"]
                or result.artifact_ids
                or result.effect_receipt_ids
                or result.provenance
                != FrozenJsonObject.from_mapping(
                    {
                        "authority": "authenticated_internal_capability_output",
                        "task_spec_digest": mapping["task_spec_digest"],
                        "plan_digest": plan.digest,
                        "batch_id": batch["batch_id"],
                    }
                )
                or verification.get("capability_id") != COUNT_BATCH_CAPABILITY_ID
                or verification.get("executor_id") != "data.count_rows.executor"
                or verification.get("contract_digest") != mapping["contract_digest"]
                or verification.get("output_sha256") != _count_digest(result.payload)
            ):
                raise ValueError("Count result lineage changed")
            resources = cast(tuple[FrozenJsonObject, ...], selection["resources"])
            by_id = {r["resource_id"]: r for r in resources}
            values = validate_count_batch(
                result.payload,
                source_id=cast(str, batch["source_id"]),
                source_revision=cast(str, selection["source_revision"]),
                resources=tuple(by_id[r] for r in batch["resource_ids"]),
            )
            if (
                result.payload.get("coverage") != "complete_requested_batch"
                or result.payload.get("counted_resources") != len(values)
                or result.payload.get("requested_resources") != len(values)
                or result.payload.get("snapshot_scope") != "one_source_one_statement"
            ):
                raise ValueError("Count result coverage changed")
            counted[cast(str, batch["source_id"])].update(values)
        except (KeyError, ValueError, TypeError) as error:
            raise ToolOutputValidationError(
                "Count comparison evidence is invalid"
            ) from error
        seen.add(result.task_id)
        accepted.append(
            {
                "task_id": result.task_id,
                "result_id": result.result_id,
                "result_digest": result.result_digest,
            }
        )
        observations.append(
            {
                "batch_id": batch["batch_id"],
                "source_id": batch["source_id"],
                "observed_at": result.payload["observed_at"],
                "duration_ms": result.payload["duration_ms"],
            }
        )
    if complete and seen != set(by_task):
        raise ToolOutputValidationError(
            "Count comparison requires every successful batch"
        )
    sources = []
    for selection in plan.selections:
        source = cast(str, selection["source_id"])
        resources = cast(tuple[FrozenJsonObject, ...], selection["resources"])
        values = counted[source]
        observed = sum(values.values())
        sources.append(
            {
                "source_id": source,
                "source_revision": selection["source_revision"],
                "selected_resources": len(resources),
                "counted_resources": len(values),
                "missing_resource_ids": tuple(
                    r["resource_id"]
                    for r in resources
                    if r["resource_id"] not in values
                ),
                "observed_total_rows": observed,
                "total_rows": observed if len(values) == len(resources) else None,
            }
        )
    all_complete = all(s["total_rows"] is not None for s in sources)
    maximum = max(cast(int, s["total_rows"]) for s in sources) if all_complete else None
    return {
        "plan_digest": plan.digest,
        "status": "completed" if all_complete else "partial",
        "sources": sources,
        "winner_source_ids": tuple(
            s["source_id"]
            for s in sources
            if all_complete and s["total_rows"] == maximum
        ),
        "coverage": (
            "complete_admitted_base_tables"
            if all_complete
            else "partial_admitted_base_tables"
        ),
        "snapshot_scope": "separate_batch_and_source_snapshots",
        "accepted_results": sorted(accepted, key=lambda r: r["task_id"]),
        "observations": sorted(observations, key=lambda r: cast(str, r["batch_id"])),
    }


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
        values = validate_count_batch(
            data,
            source_id=cast(str, arguments["source_id"]),
            source_revision=cast(str, selection["source_revision"]),
            resources=tuple(
                {"resource_id": rid, "revision": expected[rid]} for rid in batch
            ),
        )
        if set(values) & set(counted):
            raise ToolOutputValidationError("Count operation has duplicate coverage")
        counted.update(values)
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
