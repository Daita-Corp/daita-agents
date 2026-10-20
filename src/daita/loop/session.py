"""Define immutable single-use state for one admitted model-driven run."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from threading import Lock

from .._json import canonical_json
from ..capabilities import ExecutionPreference, GraphTaskBinding, RecoveryKind
from ..llm.models import ToolCall, ToolResultBlock
from .models import RunInput, RunOrigin
from .transcripts import ConversationPredecessor, RunSessionWriter


class RunCancellationToken:
    """One cancellation signal shared by admission and its admitted session."""

    __slots__ = ("_event", "_reason")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> bool:
        if not isinstance(reason, str) or not reason:
            raise ValueError("cancellation reason must be non-empty text")
        if self._event.is_set():
            return False
        self._reason = reason
        self._event.set()
        return True

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise asyncio.CancelledError(self._reason or "cancelled")


_PROGRESS_FINGERPRINT_LIMIT = 64
_PROGRESS_WARNING_LIMIT = 8
_PROGRESS_CONSTRAINT_CHARS = 300
_COUNTED_RECOVERY_KINDS = frozenset(
    kind.value for kind in RecoveryKind if kind is not RecoveryKind.TERMINAL
)
_DISPLAY_TOOL_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")


@dataclass(frozen=True, slots=True)
class _ProgressFailure:
    """One counted refusal; ``key`` groups failures a later success clears."""

    key: tuple[str, str]
    fingerprint: str
    tool_name: str
    exact_name_check: bool
    recovery: Mapping[str, str]


class RunSessionEvidence:
    """Current-run, once-set evidence; never an ambient run registry."""

    __slots__ = (
        "_lock",
        "tool_calls_attempted",
        "tool_budget_exhausted",
        "analysis_budget_exhausted",
        "run_started_monotonic",
        "programmatic_tool_calls_attempted",
        "_outer_dispatched",
        "_outer_results",
        "_progress_failures",
        "_progress_warnings",
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self.tool_calls_attempted = 0
        self.tool_budget_exhausted = False
        self.analysis_budget_exhausted = False
        self.run_started_monotonic: float | None = None
        self.programmatic_tool_calls_attempted = 0
        self._outer_dispatched: set[str] = set()
        self._outer_results: dict[str, bool] = {}
        self._progress_failures: dict[str, tuple[tuple[str, str], int]] = {}
        self._progress_warnings: dict[str, _ProgressFailure] = {}

    def observe_progress(self, call: ToolCall, result: ToolResultBlock) -> None:
        """Record one settled call as warning-only evidence.

        Nothing here decides dispatch, retries, transcript content or budgets.
        Only refusals proven ``not_dispatched`` with a correctable recovery, and
        exact-name searches that found no callable tool, are counted.
        """

        failure = _progress_failure(call, result)
        with self._lock:
            if failure is None:
                if not result.is_error:
                    self._clear_progress_keys(_progress_success_keys(call, result))
                return
            fingerprint = failure.fingerprint
            if (
                fingerprint not in self._progress_failures
                and len(self._progress_failures) >= _PROGRESS_FINGERPRINT_LIMIT
            ):
                oldest = next(iter(self._progress_failures))
                del self._progress_failures[oldest]
                self._progress_warnings.pop(oldest, None)
            _key, previous = self._progress_failures.get(fingerprint, (failure.key, 0))
            count = min(previous + 1, 2)
            self._progress_failures[fingerprint] = (failure.key, count)
            if count == 2 and fingerprint not in self._progress_warnings:
                if len(self._progress_warnings) >= _PROGRESS_WARNING_LIMIT:
                    del self._progress_warnings[next(iter(self._progress_warnings))]
                self._progress_warnings[fingerprint] = failure

    def progress_warnings(
        self, callable_tool_names: frozenset[str] = frozenset()
    ) -> tuple[str, ...]:
        """Render current warnings; a recovery tool is named only when callable."""

        with self._lock:
            failures = tuple(self._progress_warnings.values())
        return tuple(
            _render_progress_warning(failure, callable_tool_names)
            for failure in failures
        )

    def clear_progress(self) -> None:
        with self._lock:
            self._progress_failures.clear()
            self._progress_warnings.clear()

    def _clear_progress_keys(self, keys: frozenset[tuple[str, str]]) -> None:
        for fingerprint, (key, _count) in tuple(self._progress_failures.items()):
            if key in keys:
                del self._progress_failures[fingerprint]
                self._progress_warnings.pop(fingerprint, None)

    def count_tool_attempts(
        self, count: int, ceiling: int, *, programmatic: bool = False
    ) -> bool:
        with self._lock:
            self.tool_calls_attempted += count
            if programmatic:
                self.programmatic_tool_calls_attempted += count
            allowed = self.tool_calls_attempted <= ceiling
            self.tool_budget_exhausted |= not allowed
            return allowed

    def record_outer_dispatch(self, call_id: str) -> None:
        with self._lock:
            self._outer_dispatched.add(call_id)

    def record_outer_result(self, call_id: str, *, failed: bool) -> None:
        with self._lock:
            self._outer_results[call_id] = failed

    def outer_tool_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "attempted": self.tool_calls_attempted
                - self.programmatic_tool_calls_attempted,
                "dispatched": len(self._outer_dispatched),
                "failed": sum(self._outer_results.values()),
                "denied": sum(
                    failed and call_id not in self._outer_dispatched
                    for call_id, failed in self._outer_results.items()
                ),
                "unsettled": len(self._outer_dispatched - self._outer_results.keys()),
            }


def _progress_failure(
    call: ToolCall, result: ToolResultBlock
) -> _ProgressFailure | None:
    if not result.is_error:
        data = result.output.get("data")
        if (
            result.sensitivity_provenance.get("authority") != "tool_catalog_control"
            or not isinstance(data, Mapping)
            or data.get("exact_tool_available") is not False
        ):
            return None
        name = data.get("exact_tool_name")
        if not isinstance(name, str) or not name:
            return None
        return _ProgressFailure(
            key=("tool", name),
            fingerprint=_progress_digest({"exact_tool_name": name}),
            tool_name=name,
            exact_name_check=True,
            recovery={},
        )
    error = result.output.get("error")
    if not isinstance(error, Mapping):
        return None
    details = error.get("details")
    if (
        not isinstance(details, Mapping)
        or details.get("execution_disposition") != "not_dispatched"
    ):
        return None
    recovery = details.get("recovery")
    if (
        not isinstance(recovery, Mapping)
        or recovery.get("kind") not in _COUNTED_RECOVERY_KINDS
    ):
        return None
    stored_recovery = {
        field_name: value
        for field_name in ("kind", "constraint", "tool_name")
        if isinstance(value := recovery.get(field_name), str) and value
    }
    # The stored call and error only: rendered result references, timestamps
    # and other output details never prevent two identical refusals matching.
    fingerprint = _progress_digest(
        {
            "capability_id": result.capability_id,
            "tool": call.name,
            "arguments": call.arguments,
            "code": error.get("code"),
            "recovery": stored_recovery,
        }
    )
    if "constraint" in stored_recovery:
        stored_recovery["constraint"] = stored_recovery["constraint"][
            :_PROGRESS_CONSTRAINT_CHARS
        ]
    return _ProgressFailure(
        key=(
            ("capability", result.capability_id)
            if result.capability_id is not None
            else ("tool", call.name)
        ),
        fingerprint=fingerprint,
        tool_name=call.name,
        exact_name_check=False,
        recovery=stored_recovery,
    )


def _progress_success_keys(
    call: ToolCall, result: ToolResultBlock
) -> frozenset[tuple[str, str]]:
    keys = {("tool", call.name)}
    if result.capability_id is not None:
        keys.add(("capability", result.capability_id))
    data = result.output.get("data")
    if (
        result.sensitivity_provenance.get("authority") == "tool_catalog_control"
        and isinstance(data, Mapping)
        and data.get("exact_tool_available") is True
        and isinstance(name := data.get("exact_tool_name"), str)
    ):
        keys.add(("tool", name))
    return frozenset(keys)


def _progress_digest(value: object) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def _render_progress_warning(
    failure: _ProgressFailure, callable_tool_names: frozenset[str]
) -> str:
    tool = (
        f"`{failure.tool_name}`"
        if _DISPLAY_TOOL_NAME.fullmatch(failure.tool_name)
        else "an unavailable tool"
    )
    if failure.exact_name_check:
        return (
            f"Repeated failure: two exact-name searches found no callable tool "
            f"named {tool} in this run. Use a callable tool or report the blocker. "
            "This warning grants no authority or additional budget."
        )
    recovery = {
        key: value
        for key, value in failure.recovery.items()
        if key != "tool_name" or value in callable_tool_names
    }
    return (
        f"Repeated failure: two identical {tool} calls were refused before "
        "execution with the same error. Repeating the call will fail again; "
        "change the arguments or scope, follow the recovery, or report the "
        f"blocker. Recovery: {canonical_json(recovery)}. "
        "This warning grants no authority or additional budget."
    )


@dataclass(frozen=True, slots=True)
class RunSessionOptions:
    """Immutable values that vary for one run but are never persisted as authority."""

    files_only: bool = False
    explicit_learning: bool = False
    retained_skill_bindings: tuple[tuple[str, str], ...] = ()
    one_time_artifact_destinations: tuple[object, ...] = ()
    task_context: object | None = None
    task_attempt_guard: object | None = None
    execution_preference: ExecutionPreference = ExecutionPreference.AUTO

    def __post_init__(self) -> None:
        if not isinstance(self.files_only, bool) or not isinstance(
            self.explicit_learning, bool
        ):
            raise TypeError("session boolean options must be bool")
        if not isinstance(self.execution_preference, ExecutionPreference):
            raise TypeError("session execution preference is invalid")
        bindings = tuple(self.retained_skill_bindings)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or any(not isinstance(value, str) or not value for value in item)
            for item in bindings
        ):
            raise ValueError("retained skill bindings must contain text pairs")
        if len({name for name, _digest in bindings}) != len(bindings):
            raise ValueError("retained skill bindings cannot duplicate a name")
        object.__setattr__(self, "retained_skill_bindings", tuple(sorted(bindings)))
        object.__setattr__(
            self,
            "one_time_artifact_destinations",
            tuple(self.one_time_artifact_destinations),
        )
        if (self.task_context is None) != (self.task_attempt_guard is None):
            raise ValueError("task context and attempt guard must be present together")
        if self.task_context is not None:
            binding = getattr(self.task_context, "binding", None)
            guarded = getattr(self.task_attempt_guard, "binding", None)
            if not isinstance(binding, GraphTaskBinding) or guarded != binding:
                raise ValueError("task context and guard must share one exact binding")


class _SingleUse:
    __slots__ = ("_lock", "_used")

    def __init__(self) -> None:
        self._lock = Lock()
        self._used = False

    def consume(self) -> None:
        with self._lock:
            if self._used:
                raise RuntimeError("run session is single-use")
            self._used = True

    @property
    def used(self) -> bool:
        with self._lock:
            return self._used


@dataclass(frozen=True, slots=True)
class RunSession:
    """One immutable admitted invocation and its isolated mutable capabilities."""

    run: RunInput
    writer: RunSessionWriter
    absolute_deadline: float
    cancellation: RunCancellationToken
    options: RunSessionOptions = RunSessionOptions()
    admission_lease: object | None = None
    predecessor: ConversationPredecessor | None = None
    prepared: object | None = None
    budget_reservations: tuple[object, ...] = ()
    cleanup_handles: tuple[object, ...] = ()
    evidence: RunSessionEvidence = field(default_factory=RunSessionEvidence)
    _single_use: _SingleUse = field(
        default_factory=_SingleUse, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunInput):
            raise TypeError("run session requires RunInput")
        if not isinstance(self.writer, RunSessionWriter) or self.writer.run != self.run:
            raise ValueError("run session writer must belong to its exact run")
        if self.writer.predecessor != self.predecessor:
            raise ValueError("run session predecessor differs from its writer")
        if (
            not isinstance(self.absolute_deadline, (int, float))
            or isinstance(self.absolute_deadline, bool)
            or not math.isfinite(float(self.absolute_deadline))
        ):
            raise ValueError("run session deadline must be finite")
        if not isinstance(self.cancellation, RunCancellationToken):
            raise TypeError("run session cancellation token is invalid")
        if not isinstance(self.options, RunSessionOptions):
            raise TypeError("run session options are invalid")
        if not isinstance(self.evidence, RunSessionEvidence):
            raise TypeError("run session evidence is invalid")
        if self.run.origin is RunOrigin.JOB_TASK:
            scope = self.run.execution_scope
            binding = None if scope is None else scope.graph_task_binding
            if (
                binding is None
                or self.options.task_context is None
                or getattr(self.options.task_context, "binding", None) != binding
            ):
                raise ValueError(
                    "job-task sessions require one exact bound task context and guard"
                )
        elif self.options.task_context is not None:
            raise ValueError("only job-task sessions may carry task context")
        if self.admission_lease is not None and (
            getattr(self.admission_lease, "cancellation", None) is not self.cancellation
            or float(getattr(self.admission_lease, "absolute_deadline", -math.inf))
            < float(self.absolute_deadline)
        ):
            raise ValueError("run session differs from its admission lease")
        object.__setattr__(self, "absolute_deadline", float(self.absolute_deadline))
        object.__setattr__(self, "budget_reservations", tuple(self.budget_reservations))
        object.__setattr__(self, "cleanup_handles", tuple(self.cleanup_handles))

    @property
    def used(self) -> bool:
        return self._single_use.used

    def consume(self) -> None:
        self.cancellation.raise_if_cancelled()
        self._single_use.consume()

    def with_prepared(self, prepared: object) -> RunSession:
        if self.prepared is not None:
            raise RuntimeError("run session is already prepared")
        if self.used:
            raise RuntimeError("used run session cannot be prepared")
        return replace(self, prepared=prepared)


__all__ = [
    "RunCancellationToken",
    "RunSession",
    "RunSessionEvidence",
    "RunSessionOptions",
]
