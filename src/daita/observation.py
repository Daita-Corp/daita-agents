"""Define bounded execution events and deliver them to observers best-effort."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

from ._json import FrozenJsonObject, FrozenJsonValue, canonical_json

_MAX_IDENTIFIER_CHARACTERS = 256
_MAX_DATA_KEY_CHARACTERS = 128
_MAX_DATA_STRING_CHARACTERS = 1_024


@dataclass(frozen=True, slots=True)
class SupervisorStatus:
    """Payload-free health of one background supervisor."""

    name: str
    state: Literal["running", "retrying", "failed", "stopped"]
    consecutive_failures: int = 0
    failure_code: str | None = None


class AgentEventKind(str, Enum):
    ADMISSION_UPDATED = "admission.updated"
    RUN_STARTED = "run.started"
    MODEL_TEXT_DELTA = "model.text_delta"
    MODEL_COMPLETED = "model.completed"
    MODEL_FAILED = "model.failed"
    TOOL_STARTED = "tool.started"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"
    TOOL_COMPLETED = "tool.completed"
    ANALYSIS_UPDATED = "analysis.updated"
    RUN_COMPLETED = "run.completed"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """One bounded, immutable, non-durable observation of execution activity."""

    kind: AgentEventKind
    occurred_at: datetime
    run_id: str
    conversation_id: str
    data: FrozenJsonObject
    run_origin: str = "user"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AgentEventKind):
            raise TypeError("event kind must be AgentEventKind")
        if (
            not isinstance(self.occurred_at, datetime)
            or self.occurred_at.tzinfo is None
            or self.occurred_at.utcoffset() is None
        ):
            raise ValueError("event occurred_at must be timezone-aware")
        _bounded_identifier(self.run_id, "event run_id")
        _bounded_identifier(self.conversation_id, "event conversation_id")
        _bounded_identifier(self.run_origin, "event run_origin")
        data = FrozenJsonObject.from_mapping(self.data)
        _validate_data(data)
        object.__setattr__(self, "data", data)


class AgentObserver(Protocol):
    """Synchronous observer that must return promptly."""

    def __call__(self, event: AgentEvent, /) -> None: ...


# The only events and fields a diagnostic file records. Each is a structural
# count, size, code or identifier; prompts, text deltas, tool arguments and
# outputs, approvals and provider bodies are never written.
_USAGE_FIELDS = frozenset(
    {
        "usage_status",
        "usage_unavailable_reason",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "cost_status",
        "cost_code",
        "cost_amount_usd",
    }
)
_MODEL_STEP_FIELDS = _USAGE_FIELDS | {
    "provider_id",
    "model_call_index",
    "duration_ms",
    "remaining_tokens_before",
    "attempt_count",
    "attempts",
}
_DIAGNOSTIC_FIELDS: Mapping[AgentEventKind, frozenset[str]] = {
    AgentEventKind.RUN_STARTED: frozenset({"agent_id"}),
    AgentEventKind.MODEL_COMPLETED: _MODEL_STEP_FIELDS
    | {"has_text", "has_tool_calls", "context_input_tokens"},
    AgentEventKind.MODEL_FAILED: _MODEL_STEP_FIELDS | {"error_code"},
    AgentEventKind.TOOL_COMPLETED: frozenset(
        {
            "call_id",
            "tool_name",
            "capability_id",
            "toolbox_id",
            "duration_ms",
            "success",
            "error_code",
            "result_bytes",
        }
    ),
    AgentEventKind.RUN_COMPLETED: frozenset(
        {
            "exit_kind",
            "reason",
            "steps",
            "duration_ms",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
            "cost_status",
            "cost_code",
            "cost_amount_usd",
        }
    ),
}
_ATTEMPT_FIELDS = _USAGE_FIELDS | {
    "provider_id",
    "candidate_attempt_number",
    "total_attempt_number",
    "duration_ms",
    "outcome",
    "retried",
}
_DIAGNOSTIC_FILE_MODE = 0o600
_DIAGNOSTIC_DIRECTORY_MODE = 0o700
_MAX_SUMMARY_FILE_BYTES = 256 * 1024 * 1024
_LARGEST_TOOL_RESULTS = 5


class PrivateDiagnosticSink:
    """Append payload-free run diagnostics to one private local JSONL file.

    The host enables it with one explicit path and attaches it where it builds
    the loop and runtime, so foreground, routine and job runs all reach
    it. It is an observer: it cannot affect execution, and its failures are
    swallowed like any observer's. Records are never stored in the agent home.
    """

    def __init__(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise TypeError("diagnostics path must be a non-empty path")
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise ValueError("diagnostics path must be absolute")
        directory = candidate.parent.resolve()
        resolved = directory / candidate.name
        repository = _enclosing_repository(directory)
        if repository is not None:
            raise ValueError(
                "diagnostics path must be outside a source repository "
                "(a parent directory contains pyproject.toml)"
            )
        if not directory.exists():
            directory.mkdir(parents=True, mode=_DIAGNOSTIC_DIRECTORY_MODE)
        descriptor = _open_private(resolved)
        os.close(descriptor)
        self.path = resolved

    def __call__(self, event: AgentEvent, /) -> None:
        record = diagnostic_record(event)
        if record is None:
            return
        line = (canonical_json(record) + "\n").encode("utf-8")
        descriptor = _open_private(self.path)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)


def diagnostic_record(event: AgentEvent) -> dict[str, object] | None:
    """The payload-free diagnostic form of one event, or None if not recorded."""

    allowed = _DIAGNOSTIC_FIELDS.get(event.kind)
    if allowed is None:
        return None
    data: dict[str, object] = {}
    for key, value in event.data.items():
        if key not in allowed:
            continue
        plain: object = (
            value.to_dict() if isinstance(value, FrozenJsonObject) else value
        )
        if key == "attempts" and isinstance(value, tuple):
            plain = [
                {
                    item_key: item_value
                    for item_key, item_value in item.to_dict().items()
                    if item_key in _ATTEMPT_FIELDS
                }
                for item in value
                if isinstance(item, FrozenJsonObject)
            ]
        elif isinstance(value, tuple):
            continue
        data[key] = plain
    return {
        "kind": event.kind.value,
        "occurred_at": event.occurred_at.isoformat(),
        "run_id": event.run_id,
        "conversation_id": event.conversation_id,
        "run_origin": event.run_origin,
        "data": data,
    }


def observe_all(*observers: AgentObserver | None) -> AgentObserver | None:
    """Deliver each event to every observer; one failing never starves another."""

    present = tuple(item for item in observers if item is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]

    def observe(event: AgentEvent, /) -> None:
        for observer in present:
            _emit_safely(observer, event)

    return observe


def summarize_diagnostics(path: str | Path) -> list[dict[str, object]]:
    """Group one diagnostic file by run: attempts, tokens and largest results.

    Unknown counts stay null and mark the run's usage incomplete; they are
    never summed as zero.
    """

    source = Path(path)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("diagnostics path must be a regular file")
        if info.st_size > _MAX_SUMMARY_FILE_BYTES:
            raise ValueError("diagnostics file exceeds the summary bound")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            lines = handle.read().splitlines()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    runs: dict[str, dict[str, object]] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or not isinstance(record.get("run_id"), str):
            continue
        data = record.get("data")
        if not isinstance(data, dict):
            continue
        run = runs.setdefault(record["run_id"], _empty_run_summary(record))
        _summarize_record(run, str(record.get("kind")), data)
    return [_finish_run_summary(run) for run in runs.values()]


def _empty_run_summary(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "run_id": record["run_id"],
        "conversation_id": record.get("conversation_id"),
        "run_origin": record.get("run_origin"),
        "exit_kind": None,
        "reason": None,
        "model_steps": [],
        "tool_results": [],
    }


def _summarize_record(
    run: dict[str, object], kind: str, data: Mapping[str, object]
) -> None:
    if kind in {
        AgentEventKind.MODEL_COMPLETED.value,
        AgentEventKind.MODEL_FAILED.value,
    }:
        attempts = data.get("attempts")
        attempt_items = attempts if isinstance(attempts, list) else None
        steps = run["model_steps"]
        assert isinstance(steps, list)
        steps.append(
            {
                "model_call_index": data.get("model_call_index"),
                "status": (
                    "completed"
                    if kind == AgentEventKind.MODEL_COMPLETED.value
                    else "failed"
                ),
                "error_code": data.get("error_code"),
                "usage_status": data.get("usage_status"),
                "input_tokens": data.get("input_tokens"),
                "cache_read_tokens": data.get("cache_read_tokens"),
                "output_tokens": data.get("output_tokens"),
                "attempts": None if attempt_items is None else len(attempt_items),
                "retries": (
                    None
                    if attempt_items is None
                    else sum(
                        1
                        for item in attempt_items
                        if isinstance(item, dict) and item.get("retried") is True
                    )
                ),
                "stalls": (
                    None
                    if attempt_items is None
                    else sum(
                        1
                        for item in attempt_items
                        if isinstance(item, dict) and item.get("outcome") == "stalled"
                    )
                ),
            }
        )
    elif kind == AgentEventKind.TOOL_COMPLETED.value:
        sizes = data.get("result_bytes")
        results = run["tool_results"]
        assert isinstance(results, list)
        results.append(
            {
                "tool_name": data.get("tool_name"),
                "call_id": data.get("call_id"),
                "success": data.get("success"),
                "stored_bytes": (
                    sizes.get("stored") if isinstance(sizes, dict) else None
                ),
                "model_bytes": sizes.get("model") if isinstance(sizes, dict) else None,
            }
        )
    elif kind == AgentEventKind.RUN_COMPLETED.value:
        run["exit_kind"] = data.get("exit_kind")
        run["reason"] = data.get("reason")


def _finish_run_summary(run: dict[str, object]) -> dict[str, object]:
    steps = run.pop("model_steps")
    results = run.pop("tool_results")
    assert isinstance(steps, list) and isinstance(results, list)
    usage_complete = all(item["usage_status"] == "reported" for item in steps)
    run.update(
        completed_model_steps=sum(item["status"] == "completed" for item in steps),
        failed_model_steps=sum(item["status"] == "failed" for item in steps),
        model_attempts=_known_sum(item["attempts"] for item in steps),
        retries=_known_sum(item["retries"] for item in steps),
        stalls=_known_sum(item["stalls"] for item in steps),
        usage_complete=usage_complete,
        input_tokens=_known_sum(item["input_tokens"] for item in steps),
        cache_read_tokens=_known_sum(item["cache_read_tokens"] for item in steps),
        output_tokens=_known_sum(item["output_tokens"] for item in steps),
        model_steps=steps,
        tool_calls=len(results),
        largest_tool_results=sorted(
            results,
            key=lambda item: (
                -_size(item["model_bytes"]),
                -_size(item["stored_bytes"]),
            ),
        )[:_LARGEST_TOOL_RESULTS],
    )
    return run


def _known_sum(values: Iterable[object]) -> int | None:
    """Sum known counts; null only when no value is known at all."""

    known = [value for value in values if type(value) is int]
    return sum(known) if known else None


def _size(value: object) -> int:
    return value if type(value) is int else -1


def _enclosing_repository(directory: Path) -> Path | None:
    for candidate in (directory, *directory.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _open_private(path: Path) -> int:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        _DIAGNOSTIC_FILE_MODE,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("diagnostics path must be a regular file")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("diagnostics file must not be readable by other users")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _emit_safely(observer: AgentObserver | None, event: AgentEvent) -> None:
    if observer is None:
        return
    try:
        observer(event)
    except Exception:
        pass


def _bounded_identifier(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_IDENTIFIER_CHARACTERS
    ):
        raise ValueError(f"{field_name} must be a bounded non-empty string")


def _validate_data(value: FrozenJsonValue, *, key: str | None = None) -> None:
    if (
        key is not None
        and key.endswith("_id")
        and not (key == "cost_rate_schedule_id" and value is None)
    ):
        _bounded_identifier(value, f"event data {key}")
    if key == "duration_ms" and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError("event duration_ms must be a non-negative integer")
    if key == "model_call_index" and (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= 1_000_000
    ):
        raise ValueError("event model_call_index must be a bounded positive integer")
    if isinstance(value, FrozenJsonObject):
        for item_key, item in value.items():
            if not item_key or len(item_key) > _MAX_DATA_KEY_CHARACTERS:
                raise ValueError("event data keys must be bounded non-empty strings")
            _validate_data(item, key=item_key)
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_data(item)
        return
    if isinstance(value, str) and len(value) > _MAX_DATA_STRING_CHARACTERS:
        raise ValueError("event data strings must be bounded")


__all__ = [
    "AgentEvent",
    "AgentEventKind",
    "AgentObserver",
    "PrivateDiagnosticSink",
    "diagnostic_record",
    "observe_all",
    "summarize_diagnostics",
]
