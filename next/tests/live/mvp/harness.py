"""Shared opt-in, real-provider recorder, metrics, and redaction checks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import json
import os
from pathlib import Path
import platform
import re
from typing import Protocol

import daita
from daita.llm import ModelProfile, OpenAIProvider
from daita.loop.models import LoopBudgets, LoopExit
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import TaskStatus

REFERENCE_PROVIDER = "openai"
REFERENCE_CONTEXT_WINDOW_TOKENS = 128_000
REFERENCE_MAX_OUTPUT_TOKENS = 2_048
MVP_EVALUATOR_VERSION = "outcome-first-v2"
WAVE1_BUDGETS = LoopBudgets(
    max_turns=12,
    max_actions=24,
    max_repairs=4,
    max_identical_failures=2,
    max_observation_characters=160_000,
    max_total_tokens=120_000,
    max_wall_time_seconds=300,
    task_timeout_seconds=30,
)


@dataclass(frozen=True, slots=True)
class LiveMvpConfiguration:
    provider: str
    model: str
    credential_environment: str = "OPENAI_API_KEY"

    @property
    def provider_id(self) -> str:
        return f"{self.provider}:{self.model}"


class LiveMvpUnavailable(RuntimeError):
    def __init__(self, requirements: Sequence[str]) -> None:
        self.requirements = tuple(requirements)
        super().__init__(
            "requires explicit non-secret settings: " + ", ".join(self.requirements)
        )


def load_live_mvp_configuration(
    environ: Mapping[str, str] | None = None,
) -> LiveMvpConfiguration:
    values = os.environ if environ is None else environ
    requirements: list[str] = []
    if values.get("DAITA_RUN_LIVE_LLM") != "1":
        requirements.append("DAITA_RUN_LIVE_LLM=1")
    if values.get("DAITA_RUN_LIVE_MVP") != "1":
        requirements.append("DAITA_RUN_LIVE_MVP=1")
    provider = values.get("DAITA_LIVE_MVP_PROVIDER", "").strip().casefold()
    if provider != REFERENCE_PROVIDER:
        requirements.append("DAITA_LIVE_MVP_PROVIDER=openai")
    model = values.get("DAITA_LIVE_MVP_MODEL", "").strip()
    if not model:
        requirements.append("DAITA_LIVE_MVP_MODEL=<explicit-model>")
    if not values.get("OPENAI_API_KEY", "").strip():
        requirements.append("OPENAI_API_KEY")
    if requirements:
        raise LiveMvpUnavailable(requirements)
    return LiveMvpConfiguration(provider=provider, model=model)


def model_profile(configuration: LiveMvpConfiguration) -> ModelProfile:
    return ModelProfile(
        id=configuration.provider_id,
        context_window_tokens=REFERENCE_CONTEXT_WINDOW_TOKENS,
        max_output_tokens=REFERENCE_MAX_OUTPUT_TOKENS,
        supports_tools=True,
        supports_parallel_tools=True,
        supports_reasoning=True,
    )


class RecordingOpenAIProvider(OpenAIProvider):
    """The explicit live provider; canonical metrics persist with model calls."""

    def __init__(self, configuration: LiveMvpConfiguration) -> None:
        super().__init__(
            configuration.model,
            max_output_tokens=REFERENCE_MAX_OUTPUT_TOKENS,
        )


@dataclass(frozen=True, slots=True)
class LiveRunSummary:
    operation_id: str
    session_id: str | None
    wall_time_seconds: float
    provider_latency_ms: int
    provider_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    model_call_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    executor_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    finish_reasons: tuple[str, ...]
    error_history: tuple[str, ...]
    estimated_cost_usd: str
    route_revision: int | None
    route_fingerprint: str | None
    model_calls: int
    tool_calls: int
    actions: int
    repairs: int
    retries: int
    fallbacks: int
    input_tokens: int
    output_tokens: int
    selected_context_tokens: int
    omitted_context_tokens: int
    evidence_characters: int
    observation_characters: int
    evidence_count: int
    task_count: int
    truncated_evidence: int
    source_truncated_observations: int
    projection_truncated_observations: int
    truncated_observations: int
    truncation_history: tuple[Mapping[str, object], ...]
    failed_tasks: int
    cancelled_tasks: int
    rejected_actions: int
    skipped_actions: int
    duplicate_reads: int
    operation_status: str
    terminal_kind: str
    terminal_reason: str
    readiness_history: tuple[str, ...]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def summarize_live_run(
    result: LoopExit,
    snapshot: OperationSnapshot,
    *,
    wall_time_seconds: float,
) -> LiveRunSummary:
    input_tokens = 0
    output_tokens = 0
    tool_calls = 0
    selected_context_tokens = 0
    omitted_context_tokens = 0
    retries = 0
    fallbacks = 0
    provider_latency_ms = 0
    provider_ids: list[str] = []
    route_ids: list[str] = []
    finish_reasons: list[str] = []
    error_history: list[str] = []
    estimated_cost_usd = Decimal("0")
    for call in snapshot.model_calls:
        provider_ids.append(call.provider_id)
        if call.error_code is not None:
            error_history.append(call.error_code)
        response = call.response
        if response is not None:
            finish_reasons.append(response.finish_reason.value)
            estimated_cost_usd += response.usage.estimated_cost_usd
            if response.provider_id is not None:
                provider_ids.append(response.provider_id)
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            tool_calls += len(response.tool_calls)
        routing = (
            response.routing.to_payload()
            if response is not None and response.routing is not None
            else _failed_routing_payload(snapshot, call.id)
        )
        if routing is None:
            provider_latency_ms += max(
                0,
                int((call.updated_at - call.created_at).total_seconds() * 1_000),
            )
        else:
            route_id, primary_provider_id, selected_provider_id, attempts = (
                _routing_metrics(routing)
            )
            route_ids.append(route_id)
            provider_ids.append(primary_provider_id)
            if selected_provider_id is not None:
                provider_ids.append(selected_provider_id)
            for attempt_provider, attempt, latency_ms, error_code in attempts:
                provider_ids.append(attempt_provider)
                provider_latency_ms += latency_ms
                retries += int(attempt > 1)
                fallbacks += int(
                    attempt_provider != primary_provider_id and attempt == 1
                )
                if error_code is not None:
                    error_history.append(error_code)
        selected_context_tokens += _integer(
            call.request.context_selection.get("selected_context_tokens")
        )
        omitted_context_tokens += _integer(
            call.request.context_selection.get("omitted_context_tokens")
        )
    truncation_history = _truncation_history(snapshot)
    return LiveRunSummary(
        operation_id=result.operation_id,
        session_id=snapshot.operation.session_id,
        wall_time_seconds=wall_time_seconds,
        provider_latency_ms=provider_latency_ms,
        provider_ids=_ordered_unique(provider_ids),
        route_ids=_ordered_unique(route_ids),
        model_call_ids=tuple(call.id for call in snapshot.model_calls),
        task_ids=tuple(task.id for task in snapshot.tasks),
        evidence_ids=tuple(item.id for item in snapshot.evidence),
        event_ids=tuple(event.id for event in snapshot.events),
        executor_ids=_ordered_unique(task.executor_id for task in snapshot.tasks),
        source_ids=_ordered_unique(
            source_id
            for task in snapshot.tasks
            for source_id in task.execution_facts.validation_facts.source_ids
        ),
        finish_reasons=tuple(finish_reasons),
        error_history=_ordered_unique(
            (
                *error_history,
                *(
                    task.error_code
                    for task in snapshot.tasks
                    if task.error_code is not None
                ),
            )
        ),
        estimated_cost_usd=str(estimated_cost_usd),
        route_revision=snapshot.operation.model_route_revision,
        route_fingerprint=snapshot.operation.model_route_fingerprint,
        model_calls=len(snapshot.model_calls),
        tool_calls=tool_calls,
        actions=len(snapshot.tasks),
        repairs=snapshot.loop_state.repair_count,
        retries=retries,
        fallbacks=fallbacks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        selected_context_tokens=selected_context_tokens,
        omitted_context_tokens=omitted_context_tokens,
        evidence_characters=sum(len(repr(item.payload)) for item in snapshot.evidence),
        observation_characters=sum(
            len(repr(item.payload)) for item in snapshot.observations
        ),
        evidence_count=len(snapshot.evidence),
        task_count=len(snapshot.tasks),
        truncated_evidence=sum(
            item.payload.get("truncated") is True for item in snapshot.evidence
        ),
        source_truncated_observations=sum(
            item["source_truncated"] is True for item in truncation_history
        ),
        projection_truncated_observations=sum(
            item["projection_truncated"] is True for item in truncation_history
        ),
        truncated_observations=sum(
            item["source_truncated"] is True or item["projection_truncated"] is True
            for item in truncation_history
        ),
        truncation_history=truncation_history,
        failed_tasks=sum(task.status is TaskStatus.FAILED for task in snapshot.tasks),
        cancelled_tasks=sum(
            task.status is TaskStatus.CANCELLED for task in snapshot.tasks
        ),
        rejected_actions=sum(
            event.type == "action.rejected" for event in snapshot.events
        ),
        skipped_actions=sum(
            event.type == "action.skipped" for event in snapshot.events
        ),
        duplicate_reads=_duplicate_read_count(snapshot),
        operation_status=snapshot.operation.status.value,
        terminal_kind=result.kind.value,
        terminal_reason=result.reason,
        readiness_history=tuple(item.code for item in snapshot.readiness),
    )


class PropertyRecorder(Protocol):
    def __call__(self, name: str, value: object) -> None: ...


_SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9_.-]+")
_HARD_LAYERS = ("outcome", "safety", "evidence")
_CHECK_STATUSES = {"pass", "fail", "not_evaluated"}


def safe_junit_identifier(value: str) -> str:
    """Return one bounded identifier safe for JUnit properties and JSON keys."""

    normalized = _SAFE_IDENTIFIER.sub("-", value).strip("-.")
    return (normalized or "live-mvp-row")[:240]


def _benchmark_order(*, scenario_id: str, variant_id: str) -> int | None:
    scenario_match = re.fullmatch(r"LIVE-MVP-(0[1-4])", scenario_id)
    variant_rank = {
        "direct": 0,
        "conversational": 1,
        "answerable-ambiguous": 2,
    }.get(variant_id)
    if scenario_match is None or variant_rank is None:
        return None
    return variant_rank * 4 + int(scenario_match.group(1))


class LiveRowRecorder:
    """One failure-safe finalizer over all persisted operations in a live row."""

    schema_version = 1

    def __init__(
        self,
        *,
        row_id: str,
        scenario_id: str,
        variant_id: str,
        configuration: LiveMvpConfiguration,
        fixture_version: str,
        fixture_digest: str,
        prompt_version: str,
        sidecar_path: Path,
        record_property: PropertyRecorder,
        report_paths: Sequence[Path] = (),
    ) -> None:
        self.row_id = safe_junit_identifier(row_id)
        self.scenario_id = safe_junit_identifier(scenario_id)
        self.variant_id = safe_junit_identifier(variant_id)
        self.configuration = configuration
        self.fixture_version = fixture_version
        self.fixture_digest = fixture_digest
        self.prompt_version = prompt_version
        self.sidecar_path = sidecar_path
        self._record_property = record_property
        self._report_paths = list(report_paths)
        self._homes: list[Path] = []
        self._home_prohibited: list[str] = []
        self._report_prohibited: list[str] = []
        self._summaries: list[LiveRunSummary] = []
        self._hard_checks: list[dict[str, str]] = []
        self._diagnostic_codes: list[str] = []
        self._finalized = False

    @property
    def summaries(self) -> tuple[LiveRunSummary, ...]:
        return tuple(self._summaries)

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    @property
    def report_paths(self) -> tuple[Path, ...]:
        return (*self._report_paths, self.sidecar_path)

    @property
    def report_prohibited(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self._home_prohibited, *self._report_prohibited)))

    def register_home(self, home: Path, prohibited_values: Sequence[str]) -> None:
        self._homes.append(home)
        self._home_prohibited.extend(value for value in prohibited_values if value)

    def register_report_prohibited(self, *values: str) -> None:
        self._report_prohibited.extend(value for value in values if value)

    @contextmanager
    def hard_check(self, layer: str, code: str) -> Iterator[None]:
        """Record one blocking MVP assertion without short-circuiting later layers."""

        self._validate_check(layer=layer, code=code)
        try:
            yield
        except AssertionError:
            self.record_hard_check(layer=layer, code=code, status="fail")
        else:
            self.record_hard_check(layer=layer, code=code, status="pass")

    @contextmanager
    def diagnostic(self, code: str) -> Iterator[None]:
        """Record a nonblocking pre-cutover assertion by stable, non-secret code."""

        self._validate_code(code)
        try:
            yield
        except AssertionError:
            if code not in self._diagnostic_codes:
                self._diagnostic_codes.append(code)

    def record_hard_check(self, *, layer: str, code: str, status: str) -> None:
        self._validate_check(layer=layer, code=code)
        if status not in _CHECK_STATUSES:
            raise ValueError("live MVP check status is invalid")
        if any(
            item["layer"] == layer and item["code"] == code
            for item in self._hard_checks
        ):
            raise ValueError("live MVP hard check was already recorded")
        self._hard_checks.append({"layer": layer, "code": code, "status": status})

    def record_not_evaluated(self, *, layer: str, code: str) -> None:
        self.record_hard_check(layer=layer, code=code, status="not_evaluated")

    def assert_mvp_passed(self) -> None:
        failure_codes = self._hard_failure_codes()
        if failure_codes:
            raise AssertionError(
                "MVP hard checks failed or were not evaluated: "
                + ", ".join(failure_codes)
            )

    def _validate_check(self, *, layer: str, code: str) -> None:
        if layer not in _HARD_LAYERS:
            raise ValueError("live MVP hard-check layer is invalid")
        self._validate_code(code)

    @staticmethod
    def _validate_code(code: str) -> None:
        if not code or safe_junit_identifier(code) != code:
            raise ValueError("live MVP check code must be a stable safe identifier")

    def capture(
        self,
        result: LoopExit,
        snapshot: OperationSnapshot,
        *,
        wall_time_seconds: float,
    ) -> LiveRunSummary:
        if self._finalized:
            raise RuntimeError("cannot capture an operation after row finalization")
        if result.operation_id != snapshot.operation.id:
            raise ValueError("captured result and snapshot operations do not match")
        if any(item.operation_id == result.operation_id for item in self._summaries):
            raise ValueError("one operation cannot be captured twice")
        summary = summarize_live_run(
            result,
            snapshot,
            wall_time_seconds=wall_time_seconds,
        )
        self._summaries.append(summary)
        return summary

    def finalize(self, *, outcome: str) -> Mapping[str, object]:
        if self._finalized:
            raise RuntimeError("live row was already finalized")
        if outcome not in {"passed", "failed", "skipped", "error"}:
            raise ValueError("live row outcome is invalid")
        self._finalized = True
        row = self._row_payload(outcome)
        encoded_row = json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        prohibited = (*self._home_prohibited, *self._report_prohibited)
        if any(value in encoded_row for value in prohibited if value):
            raise ArtifactLeakError("redacted live row contains prohibited material")
        _atomic_update_sidecar(self.sidecar_path, self.row_id, row)
        self._record_properties(row)
        for home in self._homes:
            assert_artifacts_redacted(home, self._home_prohibited)
        for path in (*self._report_paths, self.sidecar_path):
            assert_paths_redacted((path,), prohibited)
        return row

    def _row_payload(self, outcome: str) -> dict[str, object]:
        summaries = tuple(self._summaries)
        layer_statuses = {layer: self._layer_status(layer) for layer in _HARD_LAYERS}
        hard_failure_codes = self._hard_failure_codes()
        mvp_status = self._mvp_status(layer_statuses)
        failure_category = (
            next(
                (
                    layer
                    for layer in _HARD_LAYERS
                    if layer_statuses[layer] in {"fail", "not_evaluated"}
                ),
                "none" if outcome == "passed" else "evaluator",
            )
            if self._hard_checks
            else ("none" if outcome == "passed" else "evaluator")
        )
        return {
            "schema_version": self.schema_version,
            "row_id": self.row_id,
            "scenario_id": self.scenario_id,
            "variant_id": self.variant_id,
            "outcome": outcome,
            "failure_category": failure_category,
            "benchmark_order": _benchmark_order(
                scenario_id=self.scenario_id,
                variant_id=self.variant_id,
            ),
            "mvp_status": mvp_status,
            "outcome_status": layer_statuses["outcome"],
            "safety_status": layer_statuses["safety"],
            "evidence_status": layer_statuses["evidence"],
            "hard_checks": [dict(item) for item in self._hard_checks],
            "hard_failure_codes": list(hard_failure_codes),
            "diagnostic_codes": list(self._diagnostic_codes),
            "provider": self.configuration.provider,
            "model": self.configuration.model,
            "interpreter": platform.python_implementation(),
            "python_version": platform.python_version(),
            "package_version": daita.__version__,
            "adapter_version": daita.__version__,
            "fixture_version": self.fixture_version,
            "fixture_digest": self.fixture_digest,
            "prompt_corpus_version": self.prompt_version,
            "evaluator_version": MVP_EVALUATOR_VERSION,
            "operation_ids": [item.operation_id for item in summaries],
            "session_ids": _ordered_unique(
                item.session_id for item in summaries if item.session_id is not None
            ),
            "route_revisions": _ordered_unique(
                str(item.route_revision)
                for item in summaries
                if item.route_revision is not None
            ),
            "route_fingerprints": _ordered_unique(
                item.route_fingerprint
                for item in summaries
                if item.route_fingerprint is not None
            ),
            "provider_ids": _ordered_unique(
                value for item in summaries for value in item.provider_ids
            ),
            "route_ids": _ordered_unique(
                value for item in summaries for value in item.route_ids
            ),
            "model_call_ids": [
                value for item in summaries for value in item.model_call_ids
            ],
            "task_ids": [value for item in summaries for value in item.task_ids],
            "evidence_ids": [
                value for item in summaries for value in item.evidence_ids
            ],
            "event_ids": [value for item in summaries for value in item.event_ids],
            "executor_ids": _ordered_unique(
                value for item in summaries for value in item.executor_ids
            ),
            "source_ids": _ordered_unique(
                value for item in summaries for value in item.source_ids
            ),
            "finish_reasons": [
                value for item in summaries for value in item.finish_reasons
            ],
            "error_history": [
                value for item in summaries for value in item.error_history
            ],
            "estimated_cost_usd": str(
                sum(
                    (Decimal(item.estimated_cost_usd) for item in summaries),
                    start=Decimal("0"),
                )
            ),
            "operation_count": len(summaries),
            "wall_time_seconds": sum(item.wall_time_seconds for item in summaries),
            "provider_latency_ms": sum(item.provider_latency_ms for item in summaries),
            "model_calls": sum(item.model_calls for item in summaries),
            "tool_calls": sum(item.tool_calls for item in summaries),
            "actions": sum(item.actions for item in summaries),
            "repairs": sum(item.repairs for item in summaries),
            "retries": sum(item.retries for item in summaries),
            "fallbacks": sum(item.fallbacks for item in summaries),
            "input_tokens": sum(item.input_tokens for item in summaries),
            "output_tokens": sum(item.output_tokens for item in summaries),
            "total_tokens": sum(item.total_tokens for item in summaries),
            "selected_context_tokens": sum(
                item.selected_context_tokens for item in summaries
            ),
            "omitted_context_tokens": sum(
                item.omitted_context_tokens for item in summaries
            ),
            "evidence_characters": sum(item.evidence_characters for item in summaries),
            "observation_characters": sum(
                item.observation_characters for item in summaries
            ),
            "evidence_count": sum(item.evidence_count for item in summaries),
            "task_count": sum(item.task_count for item in summaries),
            "truncated_evidence": sum(item.truncated_evidence for item in summaries),
            "source_truncated_observations": sum(
                item.source_truncated_observations for item in summaries
            ),
            "projection_truncated_observations": sum(
                item.projection_truncated_observations for item in summaries
            ),
            "truncated_observations": sum(
                item.truncated_observations for item in summaries
            ),
            "truncation_history": [
                dict(record) for item in summaries for record in item.truncation_history
            ],
            "failed_tasks": sum(item.failed_tasks for item in summaries),
            "cancelled_tasks": sum(item.cancelled_tasks for item in summaries),
            "rejected_actions": sum(item.rejected_actions for item in summaries),
            "skipped_actions": sum(item.skipped_actions for item in summaries),
            "duplicate_reads": sum(item.duplicate_reads for item in summaries),
            "terminal_history": [
                {
                    "operation_id": item.operation_id,
                    "operation_status": item.operation_status,
                    "terminal_kind": item.terminal_kind,
                    "terminal_reason": item.terminal_reason,
                }
                for item in summaries
            ],
            "readiness_history": [
                {
                    "operation_id": item.operation_id,
                    "codes": list(item.readiness_history),
                }
                for item in summaries
            ],
        }

    def _record_properties(self, row: Mapping[str, object]) -> None:
        scalar_names = (
            "row_id",
            "scenario_id",
            "variant_id",
            "outcome",
            "failure_category",
            "provider",
            "model",
            "interpreter",
            "python_version",
            "package_version",
            "adapter_version",
            "fixture_version",
            "fixture_digest",
            "prompt_corpus_version",
            "evaluator_version",
            "operation_count",
            "provider_latency_ms",
            "model_calls",
            "tool_calls",
            "actions",
            "repairs",
            "retries",
            "fallbacks",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "selected_context_tokens",
            "omitted_context_tokens",
            "evidence_characters",
            "observation_characters",
            "evidence_count",
            "task_count",
            "truncated_evidence",
            "source_truncated_observations",
            "projection_truncated_observations",
            "truncated_observations",
            "failed_tasks",
            "cancelled_tasks",
            "rejected_actions",
            "skipped_actions",
            "duplicate_reads",
            "estimated_cost_usd",
            "benchmark_order",
            "mvp_status",
            "outcome_status",
            "safety_status",
            "evidence_status",
        )
        for name in scalar_names:
            self._record_property(name, row[name])
        wall_time = row["wall_time_seconds"]
        assert isinstance(wall_time, (int, float)) and not isinstance(wall_time, bool)
        self._record_property("wall_time_seconds", f"{float(wall_time):.6f}")
        for name in (
            "operation_ids",
            "session_ids",
            "route_revisions",
            "route_fingerprints",
            "provider_ids",
            "route_ids",
            "model_call_ids",
            "task_ids",
            "evidence_ids",
            "event_ids",
            "executor_ids",
            "source_ids",
            "finish_reasons",
            "error_history",
            "hard_failure_codes",
            "diagnostic_codes",
        ):
            value = row[name]
            assert isinstance(value, (list, tuple))
            self._record_property(name, ",".join(str(item) for item in value))
        truncation_history = row["truncation_history"]
        assert isinstance(truncation_history, list)
        self._record_property(
            "truncation_history",
            json.dumps(
                truncation_history,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        hard_checks = row["hard_checks"]
        assert isinstance(hard_checks, list)
        self._record_property(
            "hard_checks",
            json.dumps(
                hard_checks,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    def _layer_status(self, layer: str) -> str:
        statuses = tuple(
            item["status"] for item in self._hard_checks if item["layer"] == layer
        )
        if not statuses:
            return "not_evaluated"
        if "fail" in statuses:
            return "fail"
        if "not_evaluated" in statuses:
            return "not_evaluated"
        return "pass"

    def _hard_failure_codes(self) -> tuple[str, ...]:
        recorded = tuple(
            item["code"]
            for item in self._hard_checks
            if item["status"] in {"fail", "not_evaluated"}
        )
        missing = tuple(
            f"{layer}_not_evaluated"
            for layer in _HARD_LAYERS
            if not any(item["layer"] == layer for item in self._hard_checks)
        )
        return (*recorded, *missing)

    @staticmethod
    def _mvp_status(layer_statuses: Mapping[str, str]) -> str:
        statuses = tuple(layer_statuses.values())
        if "fail" in statuses:
            return "fail"
        if "not_evaluated" in statuses:
            return "not_evaluated"
        return "pass"


def sidecar_path_for_run(
    *,
    base_temp: Path,
    junit_path: Path | None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    configured = values.get("DAITA_LIVE_MVP_JSON_SIDECAR", "").strip()
    if configured:
        return Path(configured)
    if junit_path is not None:
        return junit_path.with_suffix(".live-mvp.json")
    return base_temp / "live-mvp-results.json"


def assert_paths_redacted(
    paths: Sequence[Path], prohibited_values: Sequence[str]
) -> None:
    needles = tuple(value.encode() for value in prohibited_values if value)
    if not needles:
        return
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ArtifactLeakError("could not scan one retained report") from error
        if any(needle in payload for needle in needles):
            raise ArtifactLeakError(
                f"retained report contains prohibited configured material: {path.name}"
            )


def _atomic_update_sidecar(path: Path, row_id: str, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, object] = {"schema_version": 1, "rows": {}}
    if path.exists():
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ArtifactLeakError("live JSON sidecar is unreadable") from error
        if (
            not isinstance(decoded, dict)
            or decoded.get("schema_version") != 1
            or set(decoded) != {"rows", "schema_version"}
            or not isinstance(decoded.get("rows"), dict)
        ):
            raise ArtifactLeakError("live JSON sidecar has an unsupported schema")
        document = decoded
    rows = document["rows"]
    assert isinstance(rows, dict)
    rows[row_id] = dict(row)
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _duplicate_read_count(snapshot: OperationSnapshot) -> int:
    fingerprints: dict[tuple[str, str], int] = {}
    for task in snapshot.tasks:
        if task.execution_facts.access_mode.value != "read":
            continue
        key = (task.capability_id, repr(task.arguments))
        fingerprints[key] = fingerprints.get(key, 0) + 1
    return sum(max(0, count - 1) for count in fingerprints.values())


def _truncation_history(
    snapshot: OperationSnapshot,
) -> tuple[Mapping[str, object], ...]:
    task_call_ids = {task.id: task.call_id for task in snapshot.tasks}
    evidence_by_id = {item.id: item for item in snapshot.evidence}
    history: list[Mapping[str, object]] = []
    for observation in snapshot.observations:
        source_truncated, projection_truncated = _persisted_truncation_facts(
            observation.payload,
            observation_truncated=observation.truncated,
        )
        evidence = (
            None
            if observation.evidence_id is None
            else evidence_by_id.get(observation.evidence_id)
        )
        evidence_truncated = (
            evidence is not None and evidence.payload.get("truncated") is True
        )
        if not (evidence_truncated or source_truncated or projection_truncated):
            continue
        call_id = observation.call_id
        if call_id is None and observation.task_id is not None:
            call_id = task_call_ids.get(observation.task_id)
        history.append(
            {
                "call_id": call_id,
                "code": observation.code,
                "evidence_id": observation.evidence_id,
                "evidence_truncated": evidence_truncated,
                "operation_id": observation.operation_id,
                "projection_truncated": projection_truncated,
                "source_truncated": source_truncated,
                "task_id": observation.task_id,
                "turn_id": observation.turn_id,
            }
        )
    return tuple(history)


def _persisted_truncation_facts(
    payload: Mapping[str, object],
    *,
    observation_truncated: bool,
) -> tuple[bool, bool]:
    is_projection_envelope = payload.get("schema_version") == 2 and (
        "source_truncated" in payload or "projection_truncated" in payload
    )
    if not is_projection_envelope:
        return observation_truncated, False
    source_truncated = payload.get("source_truncated")
    projection_truncated = payload.get("projection_truncated")
    if not isinstance(source_truncated, bool) or not isinstance(
        projection_truncated,
        bool,
    ):
        raise AssertionError("persisted observation truncation facts are malformed")
    if observation_truncated != source_truncated:
        raise AssertionError("persisted observation source truncation facts disagree")
    return source_truncated, projection_truncated


def _failed_routing_payload(
    snapshot: OperationSnapshot,
    model_call_id: str,
) -> Mapping[str, object] | None:
    matches = tuple(
        event.payload.get("routing")
        for event in snapshot.events
        if event.type == "model_call.failed" and event.model_call_id == model_call_id
    )
    if len(matches) > 1:
        raise AssertionError("one model call has duplicate persisted routing facts")
    if not matches or matches[0] is None:
        return None
    routing = matches[0]
    if not isinstance(routing, Mapping):
        raise AssertionError("persisted model routing facts are malformed")
    return routing


def _routing_metrics(
    routing: Mapping[str, object],
) -> tuple[
    str,
    str,
    str | None,
    tuple[tuple[str, int, int, str | None], ...],
]:
    route_id = routing.get("route_id")
    primary = routing.get("primary_provider_id")
    selected = routing.get("selected_provider_id")
    raw_attempts = routing.get("attempts")
    if (
        not isinstance(route_id, str)
        or not route_id
        or not isinstance(primary, str)
        or not primary
        or (selected is not None and not isinstance(selected, str))
        or not isinstance(raw_attempts, tuple)
    ):
        raise AssertionError("persisted model routing facts are malformed")
    attempts: list[tuple[str, int, int, str | None]] = []
    for raw_attempt in raw_attempts:
        if not isinstance(raw_attempt, Mapping):
            raise AssertionError("persisted route attempt facts are malformed")
        provider_id = raw_attempt.get("provider_id")
        attempt = raw_attempt.get("attempt")
        latency_ms = raw_attempt.get("latency_ms")
        error_code = raw_attempt.get("error_code")
        if (
            not isinstance(provider_id, str)
            or not provider_id
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
            or not isinstance(latency_ms, int)
            or isinstance(latency_ms, bool)
            or latency_ms < 0
            or (error_code is not None and not isinstance(error_code, str))
        ):
            raise AssertionError("persisted route attempt facts are malformed")
        attempts.append((provider_id, attempt, latency_ms, error_code))
    return route_id, primary, selected, tuple(attempts)


class ArtifactLeakError(AssertionError):
    pass


def assert_artifacts_redacted(root: Path, prohibited_values: Sequence[str]) -> None:
    needles = tuple(value.encode() for value in prohibited_values if value)
    if not needles or not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ArtifactLeakError("could not scan one retained artifact") from error
        if any(needle in payload for needle in needles):
            raise ArtifactLeakError(
                f"retained artifact contains prohibited configured material: {path.name}"
            )


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = [
    "ArtifactLeakError",
    "LiveMvpConfiguration",
    "LiveMvpUnavailable",
    "LiveRunSummary",
    "LiveRowRecorder",
    "MVP_EVALUATOR_VERSION",
    "REFERENCE_CONTEXT_WINDOW_TOKENS",
    "REFERENCE_MAX_OUTPUT_TOKENS",
    "REFERENCE_PROVIDER",
    "RecordingOpenAIProvider",
    "WAVE1_BUDGETS",
    "assert_artifacts_redacted",
    "assert_paths_redacted",
    "load_live_mvp_configuration",
    "model_profile",
    "safe_junit_identifier",
    "sidecar_path_for_run",
    "summarize_live_run",
]
