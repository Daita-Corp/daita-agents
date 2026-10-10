"""Bounded analytical evidence, separate from the exact model transcript."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

from .._json import FrozenJsonObject, canonical_json


class AnalysisTraceCapacityError(ValueError):
    """The retained trace cannot admit more work within its run allowance."""


@dataclass(frozen=True, slots=True)
class ProgrammaticReadResult:
    """Keep committed host evidence separate from the response delivered to Python."""

    response: dict[str, object]
    evidence: AnalysisEvidence


@dataclass(frozen=True, slots=True)
class AnalysisEvidence:
    run_id: str
    evidence_id: str
    kind: str
    facts: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.run_id or not self.evidence_id or len(self.evidence_id) > 256:
            raise ValueError("Analysis evidence identity is invalid")
        if self.kind not in {"generation", "cell", "child", "run"}:
            raise ValueError("Unknown analysis evidence kind")
        facts = FrozenJsonObject.from_mapping(self.facts)
        if facts.get("status") not in {
            "admitted",
            "running",
            "completed",
            "succeeded",
            "failed",
            "interrupted",
            "closed",
            "recovered_closed",
            "cleanup_failed",
        }:
            raise ValueError("Unknown analytical evidence status")
        if self.kind == "child":
            if facts["status"] not in {
                "admitted",
                "running",
                "succeeded",
                "failed",
                "interrupted",
            }:
                raise ValueError("Invalid child invocation status")
            required = {
                "status",
                "parent_call_id",
                "sequence",
                "tool_name",
                "arguments",
                "capability_id",
                "executor_id",
                "contract_digest",
                "sensitivity",
                "dispatched",
            }
            terminal = {
                "result",
                "result_digest",
                "result_sensitivity",
                "sensitivity_provenance",
                "wall_seconds",
            }
            allowed = (
                required
                | terminal
                | {
                    "interruption",
                    "completion_observed",
                    "started_at",
                    "completed_at",
                    "input_schema_digest",
                }
            )
            if not required <= set(facts) or not set(facts) <= allowed:
                raise ValueError("Invalid child invocation evidence contract")
            sequence = facts["sequence"]
            if (
                not isinstance(facts["parent_call_id"], str)
                or not isinstance(facts["tool_name"], str)
                or type(sequence) is not int
                or not 1 <= sequence <= 1_000_000
                or not isinstance(facts["arguments"], Mapping)
                or type(facts["dispatched"]) is not bool
            ):
                raise ValueError("Invalid child invocation identity or arguments")
            for name in ("parent_call_id", "tool_name"):
                if not facts[name] or len(str(facts[name])) > 256:
                    raise ValueError("Child invocation text identity exceeds its bound")
            for name in ("capability_id", "executor_id"):
                value = facts[name]
                if value is not None and (
                    not isinstance(value, str) or not value or len(value) > 256
                ):
                    raise ValueError("Invalid child invocation contract identity")
            for name in ("contract_digest", "input_schema_digest"):
                value = facts.get(name)
                if value is not None and (
                    not isinstance(value, str)
                    or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
                ):
                    raise ValueError("Invalid child invocation contract digest")
            if facts["sensitivity"] not in {
                "public",
                "internal",
                "confidential",
                "restricted",
            }:
                raise ValueError("Invalid child invocation classification")
            if facts["status"] in {"succeeded", "failed"} and not terminal <= set(
                facts
            ):
                raise ValueError(
                    "A terminal child requires its authenticated result and timing"
                )
            if facts["status"] in {"succeeded", "failed"}:
                result = facts["result"]
                wall = facts["wall_seconds"]
                if (
                    not isinstance(result, Mapping)
                    or facts["result_digest"]
                    != "sha256:" + sha256(canonical_json(result).encode()).hexdigest()
                ):
                    raise ValueError("Child invocation result digest is inconsistent")
                if (
                    isinstance(wall, bool)
                    or not isinstance(wall, (int, float))
                    or not math.isfinite(wall)
                    or wall < 0
                ):
                    raise ValueError("Invalid child invocation wall measurement")
                if facts["result_sensitivity"] not in {
                    None,
                    "public",
                    "internal",
                    "confidential",
                    "restricted",
                } or not isinstance(facts["sensitivity_provenance"], Mapping):
                    raise ValueError("Invalid child result classification evidence")
                if facts["status"] == "succeeded" and (
                    facts["dispatched"] is not True or facts["capability_id"] is None
                ):
                    raise ValueError(
                        "A successful child requires observed admitted dispatch"
                    )
        recovery = facts.get("recovery")
        if recovery is not None:
            if (
                self.kind != "generation"
                or not isinstance(recovery, Mapping)
                or set(recovery) != {"status", "pid", "usage", "cleanup"}
                or recovery["status"] != "recovered_closed"
            ):
                raise ValueError("Invalid native recovery facts")
            usage, cleanup = recovery["usage"], recovery["cleanup"]
            if not isinstance(usage, Mapping) or not isinstance(cleanup, Mapping):
                raise ValueError("Recovery requires measured closure")
            for name in ("user_cpu_seconds", "system_cpu_seconds"):
                value = usage.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("Recovery CPU measurement is unavailable")
            if usage.get("cpu_complete") is not True or not all(
                cleanup.get(name) is True
                for name in (
                    "process_reaped",
                    "scratch_deleted",
                    "descriptors_closed",
                    "child_io_settled",
                )
            ):
                raise ValueError("Recovery closure remains incomplete")
            spawned = cleanup.get("process_spawned")
            pid = recovery["pid"]
            if (
                type(spawned) is not bool
                or (spawned and (type(pid) is not int or pid <= 0))
                or (
                    not spawned
                    and (
                        recovery["pid"] is not None
                        or usage["user_cpu_seconds"] != 0
                        or usage["system_cpu_seconds"] != 0
                    )
                )
            ):
                raise ValueError(
                    "Recovery launch and consumption facts are inconsistent"
                )
        if "proof_disposed" in facts and type(facts["proof_disposed"]) is not bool:
            raise ValueError("Invalid native proof disposal fact")
        if facts.get("proof_disposed") is True:
            proved = recovery if isinstance(recovery, Mapping) else facts
            cleanup, usage = proved.get("cleanup"), proved.get("usage")
            if (
                self.kind != "generation"
                or not isinstance(cleanup, Mapping)
                or not isinstance(usage, Mapping)
                or usage.get("cpu_complete") is not True
                or not all(
                    cleanup.get(name) is True
                    for name in (
                        "process_reaped",
                        "scratch_deleted",
                        "descriptors_closed",
                        "child_io_settled",
                    )
                )
            ):
                raise ValueError("Native proof disposal requires complete closure")
        encoded_size = len(canonical_json(facts).encode())
        if self.kind == "generation" and recovery is None and encoded_size > 245_760:
            raise ValueError(
                "Generation evidence must reserve native recovery capacity"
            )
        if encoded_size > 262_144:
            raise ValueError("Analysis evidence exceeds its byte allowance")
        object.__setattr__(self, "facts", facts)


def analysis_summary(records: tuple[AnalysisEvidence, ...]) -> dict[str, object] | None:
    generations = [record.facts for record in records if record.kind == "generation"]
    if not records:
        return None
    settled = [
        recovery if isinstance((recovery := facts.get("recovery")), Mapping) else facts
        for facts in generations
    ]
    usage = [facts.get("usage") for facts in settled]
    cleanup = [facts.get("cleanup") for facts in settled]

    def total(name: str) -> float | None:
        values = [
            item.get(name) if isinstance(item, Mapping) else None for item in usage
        ]
        if any(not isinstance(value, (int, float)) for value in values):
            return None
        return sum(float(value) for value in values if isinstance(value, (int, float)))

    peaks = [
        item.get("peak_rss_bytes") if isinstance(item, Mapping) else None
        for item in usage
    ]
    children = [record for record in records if record.kind == "child"]
    run = next((record.facts for record in records if record.kind == "run"), {})
    outer_counts = run.get("outer_tool_calls")
    validation_input = sum(
        int(item.get("validation_input_bytes", 0))
        for item in usage
        if isinstance(item, Mapping)
    )
    worker_input = total("input_bytes")
    return {
        "worker_generations": len(generations),
        "user_cpu_seconds": total("user_cpu_seconds"),
        "system_cpu_seconds": total("system_cpu_seconds"),
        "broker_wait_seconds": total("broker_wait_seconds"),
        "wall_seconds": run.get("wall_seconds"),
        "analysis_wall_seconds": run.get("analysis_wall_seconds"),
        "worker_queue_wait_seconds": run.get("worker_queue_wait_seconds"),
        "tool_calls_attempted": run.get("tool_calls_attempted"),
        "outer_tool_calls": (
            dict(outer_counts) if isinstance(outer_counts, Mapping) else None
        ),
        "protocol_input_bytes": total("protocol_input_bytes"),
        "protocol_output_bytes": total("protocol_output_bytes"),
        "scratch_samples_incomplete": any(
            isinstance(item, Mapping) and item.get("scratch_samples_incomplete") is True
            for item in usage
        ),
        "captured_input_bytes": (
            None if worker_input is None else worker_input - validation_input
        ),
        "worker_input_bytes": worker_input,
        "validation_input_bytes": validation_input,
        "output_bytes": total("output_bytes"),
        "candidate_bytes": run.get("candidate_bytes"),
        "artifact_bytes_committed": run.get("artifact_bytes_committed"),
        "reservations_released": run.get("reservations_released"),
        "peak_scratch_bytes": (
            max(
                (
                    int(item.get("peak_scratch_bytes", 0))
                    for item in usage
                    if isinstance(item, Mapping)
                    and item.get("peak_scratch_bytes") is not None
                ),
                default=None,
            )
            if not any(
                isinstance(item, Mapping)
                and item.get("scratch_samples_incomplete") is True
                for item in usage
            )
            else None
        ),
        "peak_scratch_files": (
            max(
                (
                    int(item.get("peak_scratch_files", 0))
                    for item in usage
                    if isinstance(item, Mapping)
                    and item.get("peak_scratch_files") is not None
                ),
                default=None,
            )
            if not any(
                isinstance(item, Mapping)
                and item.get("scratch_samples_incomplete") is True
                for item in usage
            )
            else None
        ),
        "peak_rss_bytes": max(
            (int(value) for value in peaks if isinstance(value, int)), default=None
        ),
        "child_attempted": run.get("child_attempted", len(children)),
        "child_denied": sum(
            record.facts.get("dispatched") is False for record in children
        ),
        "child_dispatched": sum(
            record.facts.get("dispatched") is True for record in children
        ),
        "child_failed": sum(
            record.facts.get("status") in {"failed", "interrupted"}
            for record in children
        ),
        "processes_reaped": all(
            isinstance(item, Mapping) and item.get("process_reaped") is True
            for item in cleanup
        ),
        "scratch_deleted": all(
            isinstance(item, Mapping) and item.get("scratch_deleted") is True
            for item in cleanup
        ),
    }
