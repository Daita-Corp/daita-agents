"""Define immutable single-use state for one admitted model-driven run."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from threading import Lock

from .._json import canonical_json
from ..capabilities import RESERVED_TOOL_NAMES, ExecutionPreference, GraphTaskBinding
from ..llm.models import ModelSensitivity, ToolCall, ToolResultBlock
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


class RunSessionEvidence:
    """Current-run, once-set evidence; never an ambient run registry."""

    __slots__ = (
        "_learning_mutation_call_id",
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
        self._learning_mutation_call_id: str | None = None
        self._lock = Lock()
        self.tool_calls_attempted = 0
        self.tool_budget_exhausted = False
        self.analysis_budget_exhausted = False
        self.run_started_monotonic: float | None = None
        self.programmatic_tool_calls_attempted = 0
        self._outer_dispatched: set[str] = set()
        self._outer_results: dict[str, bool] = {}
        self._progress_failures: dict[str, int] = {}
        self._progress_warnings: dict[str, str] = {}

    def observe_progress(self, call: ToolCall, result: ToolResultBlock) -> None:
        """Warning-only evidence; no dispatch, retry, transcript or budget decisions."""
        error = result.output.get("error")
        if not result.is_error:
            data = result.output.get("data")
            if (
                result.output.get("kind") != "toolbox_search_result"
                or result.sensitivity_provenance.get("authority")
                != "tool_catalog_control"
                or not isinstance(data, Mapping)
                or data.get("exact_tool_available") is not False
            ):
                return
            error = {
                "code": "exact_tool_unavailable",
                "details": {
                    "execution_disposition": "not_dispatched",
                    "recovery": {
                        "kind": "reduce_scope",
                        "constraint": "The exact tool is absent from prepared candidates; use admitted tools or report the blocker.",
                    },
                },
            }
        if not isinstance(error, Mapping):
            return
        details = error.get("details")
        if (
            not isinstance(details, Mapping)
            or details.get("execution_disposition") != "not_dispatched"
        ):
            return
        recovery = details.get("recovery")
        if not isinstance(recovery, Mapping) or recovery.get("kind") not in {
            "correct_arguments",
            "reduce_scope",
            "reinspect_binding",
            "requires_human_permission",
        }:
            return
        safe_recovery: dict[str, object] = {"kind": recovery["kind"]}
        maximum = recovery.get("max_targets")
        if isinstance(maximum, int) and not isinstance(maximum, bool):
            safe_recovery["max_targets"] = maximum
        next_tool = recovery.get("tool_name")
        if isinstance(next_tool, str) and next_tool in RESERVED_TOOL_NAMES:
            safe_recovery["tool_name"] = next_tool
        identity = result.capability_id or "discovery control/unavailable tool"
        fingerprint = sha256(
            canonical_json(
                {
                    "capability_id": result.capability_id,
                    "tool": call.name,
                    "arguments": call.arguments,
                    "outcome": result.output,
                }
            ).encode()
        ).hexdigest()
        with self._lock:
            if (
                fingerprint not in self._progress_failures
                and len(self._progress_failures) >= 64
            ):
                oldest = next(iter(self._progress_failures))
                del self._progress_failures[oldest]
                self._progress_warnings.pop(oldest, None)
            count = self._progress_failures.get(fingerprint, 0) + 1
            self._progress_failures[fingerprint] = min(count, 2)
            if count == 2 and fingerprint not in self._progress_warnings:
                if len(self._progress_warnings) >= 8:
                    del self._progress_warnings[next(iter(self._progress_warnings))]
                self._progress_warnings[fingerprint] = (
                    f"Repeated invocation failure for {identity}. "
                    + (
                        "Two exact-name searches confirmed no admitted tool. "
                        if error.get("code") == "exact_tool_unavailable"
                        else "Two identical calls were not dispatched. This does not establish that the operation is unavailable; issue a corrected call using the existing recovery guidance. "
                    )
                    + "Change the relevant arguments/scope, "
                    "use the admitted recovery path, or report the blocker. Recovery: "
                    + canonical_json(safe_recovery)
                    + ". This warning grants no authority or additional budget."
                )

    @property
    def progress_warnings(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._progress_warnings.values())

    def clear_progress(self) -> None:
        with self._lock:
            self._progress_failures.clear()
            self._progress_warnings.clear()

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

    @property
    def learning_mutation_succeeded(self) -> bool:
        with self._lock:
            return self._learning_mutation_call_id is not None

    @property
    def learning_mutation_call_id(self) -> str | None:
        with self._lock:
            return self._learning_mutation_call_id

    def record_learning_mutation(self, call_id: str) -> None:
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("learning mutation call_id must be non-empty")
        with self._lock:
            if self._learning_mutation_call_id not in {None, call_id}:
                raise RuntimeError("session recorded more than one learning mutation")
            self._learning_mutation_call_id = call_id


@dataclass(frozen=True, slots=True)
class RunSessionOptions:
    """Immutable values that vary for one run but are never persisted as authority."""

    files_only: bool = False
    explicit_learning: bool = False
    learning_candidate: object | None = None
    learning_candidate_id: str | None = None
    learning_candidate_text: str | None = None
    learning_candidate_sensitivity: ModelSensitivity | None = None
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
        selected = (
            self.learning_candidate,
            self.learning_candidate_id,
            self.learning_candidate_text,
            self.learning_candidate_sensitivity,
        )
        if any(item is not None for item in selected) and not all(
            item is not None for item in selected
        ):
            raise ValueError("learning candidate session values must be set together")
        if self.learning_candidate_id is not None and (
            not isinstance(self.learning_candidate_id, str)
            or not self.learning_candidate_id
        ):
            raise ValueError("learning candidate id must be non-empty text")
        if self.learning_candidate_text is not None and (
            not isinstance(self.learning_candidate_text, str)
            or not self.learning_candidate_text
        ):
            raise ValueError("learning candidate text must be non-empty text")
        if self.learning_candidate_sensitivity is not None and not isinstance(
            self.learning_candidate_sensitivity, ModelSensitivity
        ):
            raise TypeError("learning candidate sensitivity is invalid")
        if self.learning_candidate is not None and (
            getattr(self.learning_candidate, "id", None) != self.learning_candidate_id
            or getattr(self.learning_candidate, "sensitivity", None)
            is not self.learning_candidate_sensitivity
        ):
            raise ValueError("learning candidate session binding is inconsistent")
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
