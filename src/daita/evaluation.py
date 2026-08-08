"""Caller-owned deterministic learning evaluation and effectiveness reports.

This module owns no durable state. Callers retain benchmark labels and bounded
observer events, then pass them to these pure aggregation and rendering helpers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import cast

from .learning_candidates import (
    LEARNING_REVIEW_MAX_MODEL_CALLS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    LEARNING_REVIEW_MAX_WALL_TIME_SECONDS,
)
from .observation import AgentEvent, AgentEventKind

EVALUATION_MAX_CASES = 256
EVALUATION_MAX_EVENTS_PER_OUTCOME = 8_192
_LEARNING_REVIEW_MAX_DURATION_MS = int(LEARNING_REVIEW_MAX_WALL_TIME_SECONDS * 1_000)

_CASE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_CATALOG_TOOLS = frozenset(
    {"catalog_search", "catalog_schema", "catalog_inspect", "catalog_traverse"}
)
_SQL_TOOLS = frozenset({"data_query_sqlite", "data_query_postgresql"})
_LEARNING_WRITE_TOOLS = frozenset(
    {"memory_set", "semantic_save", "semantic_delete", "skill_save", "skill_delete"}
)
_REPORT_ROWS = (
    ("Answer correctness", "answer_correct"),
    ("Business-definition correctness", "business_definition_correct"),
    ("Source selection", "source_selection_correct"),
    ("Resource selection", "resource_selection_correct"),
    ("Field selection", "field_selection_correct"),
    ("Semantic-constraint adherence", "semantic_constraints_satisfied"),
    ("Irrelevant recall rate", "irrelevant_recall_rate"),
    ("Stale activations", "stale_activation_count"),
    ("Conflicting claims selected", "conflicting_claim_selection_count"),
    ("Cross-source leakage", "cross_source_leakage_count"),
    ("Relevant skill selection", "relevant_skill_selected"),
    ("Skill-induced regressions", "skill_induced_regressions"),
    ("Model calls", "model_calls"),
    ("Tool calls", "tool_calls"),
    ("Catalog discovery calls", "catalog_discovery_calls"),
    ("Failed SQL calls", "failed_sql_calls"),
    ("Corrected SQL calls", "corrected_sql_calls"),
    ("Duration (ms)", "duration_ms"),
    ("Total tokens", "total_tokens"),
    ("Estimated cost (USD)", "estimated_cost_usd"),
    ("Learning proposals", "learning_proposals"),
    ("Successful proposals", "proposal_succeeded"),
    ("Failed proposals", "proposal_failed"),
    ("Approval requests", "approvals_requested"),
    ("Approvals", "approvals"),
    ("Denials", "denials"),
    ("Approval failures", "approval_failures"),
)


class BenchmarkVariant(str, Enum):
    BASELINE = "baseline"
    LEARNED = "learned"


@dataclass(frozen=True, slots=True)
class BenchmarkJudgment:
    """Human labels and hard-safety counts for one bounded benchmark outcome."""

    answer_correct: bool
    business_definition_correct: bool | None = None
    source_selection_correct: bool | None = None
    resource_selection_correct: bool | None = None
    field_selection_correct: bool | None = None
    semantic_constraints_satisfied: bool | None = None
    recalled_annotation_count: int = 0
    irrelevant_recall_count: int = 0
    stale_activation_count: int = 0
    conflicting_claim_selection_count: int = 0
    cross_source_leakage_count: int = 0
    relevant_skill_selected: bool | None = None
    skill_induced_regression: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "answer_correct",
            "business_definition_correct",
            "source_selection_correct",
            "resource_selection_correct",
            "field_selection_correct",
            "semantic_constraints_satisfied",
            "relevant_skill_selected",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a boolean or None")
        if not isinstance(self.skill_induced_regression, bool):
            raise TypeError("skill_induced_regression must be a boolean")
        for field_name in (
            "recalled_annotation_count",
            "irrelevant_recall_count",
            "stale_activation_count",
            "conflicting_claim_selection_count",
            "cross_source_leakage_count",
        ):
            _non_negative_integer(getattr(self, field_name), field_name)
        if self.irrelevant_recall_count > self.recalled_annotation_count:
            raise ValueError(
                "irrelevant_recall_count cannot exceed recalled_annotation_count"
            )

    @property
    def hard_safety_passed(self) -> bool:
        return (
            self.stale_activation_count == 0
            and self.conflicting_claim_selection_count == 0
            and self.cross_source_leakage_count == 0
        )


@dataclass(frozen=True, slots=True)
class RunMeasurement:
    """Content-free operational counters derived from observer events."""

    model_calls: int = 0
    tool_calls: int = 0
    catalog_discovery_calls: int = 0
    failed_sql_calls: int = 0
    corrected_sql_calls: int = 0
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Decimal | None = None
    cost_complete: bool = False
    learning_proposals: int = 0
    proposal_succeeded: int = 0
    proposal_failed: int = 0
    approvals_requested: int = 0
    approvals: int = 0
    denials: int = 0
    approval_failures: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "estimated_cost_usd":
                if value is not None and (
                    not isinstance(value, Decimal) or not value.is_finite() or value < 0
                ):
                    raise ValueError(
                        "estimated_cost_usd must be a finite non-negative Decimal or None"
                    )
                continue
            if item.name == "cost_complete":
                if not isinstance(value, bool):
                    raise TypeError("cost_complete must be a boolean")
                continue
            _non_negative_integer(value, item.name)
        if self.cost_complete != (self.estimated_cost_usd is not None):
            raise ValueError(
                "cost_complete must be true exactly when estimated cost is available"
            )


@dataclass(frozen=True, slots=True)
class CandidateReviewMeasurement:
    """Caller-provided content-free measurements for one bounded review outcome."""

    proposed_candidates: int = 0
    accepted_candidates: int = 0
    rejected_candidates: int = 0
    false_positive_candidates: int = 0
    duplicate_candidates_suppressed: int = 0
    background_model_calls: int = 0
    background_total_tokens: int = 0
    background_duration_ms: int = 0
    background_estimated_cost_usd: Decimal | None = None
    background_cost_complete: bool = False
    stale_activation_count: int = 0
    conflicting_claim_selection_count: int = 0
    cross_source_leakage_count: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "background_estimated_cost_usd":
                if value is not None and (
                    not isinstance(value, Decimal) or not value.is_finite() or value < 0
                ):
                    raise ValueError(
                        "background_estimated_cost_usd must be finite and "
                        "non-negative or None"
                    )
                continue
            if item.name == "background_cost_complete":
                if not isinstance(value, bool):
                    raise TypeError("background_cost_complete must be a boolean")
                continue
            _non_negative_integer(value, item.name)
        if self.background_cost_complete != (
            self.background_estimated_cost_usd is not None
        ):
            raise ValueError(
                "background_cost_complete must agree with estimated cost availability"
            )
        if (
            self.accepted_candidates + self.rejected_candidates
            > self.proposed_candidates
        ):
            raise ValueError("candidate decisions cannot exceed proposed candidates")
        if self.false_positive_candidates > self.proposed_candidates:
            raise ValueError(
                "candidate false positives cannot exceed proposed candidates"
            )
        if self.proposed_candidates > LEARNING_REVIEW_MAX_PROPOSALS:
            raise ValueError(
                "proposed_candidates exceeds the per-review proposal bound"
            )
        if self.duplicate_candidates_suppressed > LEARNING_REVIEW_MAX_PROPOSALS:
            raise ValueError(
                "duplicate_candidates_suppressed exceeds the per-review "
                "proposal bound"
            )
        if self.background_model_calls > LEARNING_REVIEW_MAX_MODEL_CALLS:
            raise ValueError(
                "background_model_calls exceeds the per-review model-call bound"
            )
        if self.background_total_tokens > LEARNING_REVIEW_MAX_TOTAL_TOKENS:
            raise ValueError(
                "background_total_tokens exceeds the per-review token bound"
            )
        if self.background_duration_ms > _LEARNING_REVIEW_MAX_DURATION_MS:
            raise ValueError(
                "background_duration_ms exceeds the per-review wall-time bound"
            )

    @property
    def hard_safety_passed(self) -> bool:
        return (
            self.stale_activation_count == 0
            and self.conflicting_claim_selection_count == 0
            and self.cross_source_leakage_count == 0
        )


@dataclass(frozen=True, slots=True)
class CandidateReviewReport:
    """Pure aggregate over bounded caller-owned candidate review outcomes."""

    measurements: tuple[CandidateReviewMeasurement, ...]

    def __post_init__(self) -> None:
        values = tuple(self.measurements)
        if not values or len(values) > EVALUATION_MAX_CASES:
            raise ValueError(
                f"candidate review report requires 1 to {EVALUATION_MAX_CASES} "
                "measurements"
            )
        if any(not isinstance(item, CandidateReviewMeasurement) for item in values):
            raise TypeError(
                "candidate review report requires CandidateReviewMeasurement records"
            )
        object.__setattr__(self, "measurements", values)

    def to_mapping(self) -> dict[str, object]:
        totals = _candidate_review_totals(self.measurements)
        proposed = cast(int, totals["proposed_candidates"])
        accepted = cast(int, totals["accepted_candidates"])
        rejected = cast(int, totals["rejected_candidates"])
        decided = accepted + rejected
        false_positives = cast(int, totals["false_positive_candidates"])
        totals["candidate_precision"] = (
            _rate(proposed - false_positives, proposed) if proposed else None
        )
        totals["acceptance_rate"] = _rate(accepted, decided) if decided else None
        totals["rejection_rate"] = _rate(rejected, decided) if decided else None
        totals["hard_safety_passed"] = all(
            item.hard_safety_passed for item in self.measurements
        )
        return totals

    def render_markdown(self) -> str:
        data = self.to_mapping()
        rows = (
            ("Proposed candidates", "proposed_candidates"),
            ("Accepted candidates", "accepted_candidates"),
            ("Rejected candidates", "rejected_candidates"),
            ("False positives", "false_positive_candidates"),
            ("Candidate precision", "candidate_precision"),
            ("Acceptance rate", "acceptance_rate"),
            ("Rejection rate", "rejection_rate"),
            ("Duplicates suppressed", "duplicate_candidates_suppressed"),
            ("Background model calls", "background_model_calls"),
            ("Background tokens", "background_total_tokens"),
            ("Background duration (ms)", "background_duration_ms"),
            ("Background estimated cost (USD)", "background_estimated_cost_usd"),
            ("Stale activations", "stale_activation_count"),
            ("Conflicting claims selected", "conflicting_claim_selection_count"),
            ("Cross-source leakage", "cross_source_leakage_count"),
            ("Hard safety passed", "hard_safety_passed"),
        )
        lines = [
            "# Candidate Review Evaluation",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
        lines.extend(
            f"| {label} | {_summary_value(data.get(key))} |" for label, key in rows
        )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class BenchmarkOutcome:
    case_id: str
    variant: BenchmarkVariant
    judgment: BenchmarkJudgment
    measurement: RunMeasurement = field(default_factory=RunMeasurement)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_id, str)
            or _CASE_ID.fullmatch(self.case_id) is None
        ):
            raise ValueError("case_id must be a bounded lowercase portable identifier")
        if not isinstance(self.variant, BenchmarkVariant):
            raise TypeError("variant must be BenchmarkVariant")
        if not isinstance(self.judgment, BenchmarkJudgment):
            raise TypeError("judgment must be BenchmarkJudgment")
        if not isinstance(self.measurement, RunMeasurement):
            raise TypeError("measurement must be RunMeasurement")


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    case_id: str
    baseline: BenchmarkOutcome
    learned: BenchmarkOutcome

    def __post_init__(self) -> None:
        if (
            self.baseline.case_id != self.case_id
            or self.learned.case_id != self.case_id
            or self.baseline.variant is not BenchmarkVariant.BASELINE
            or self.learned.variant is not BenchmarkVariant.LEARNED
        ):
            raise ValueError(
                "comparison must contain one matching baseline/learned pair"
            )

    @property
    def verdict(self) -> str:
        if not self.learned.judgment.hard_safety_passed:
            return "unsafe"
        correctness_delta = _correctness_score(
            self.learned.judgment
        ) - _correctness_score(self.baseline.judgment)
        if correctness_delta > 0:
            return "improved_correctness"
        if correctness_delta < 0 or self.learned.judgment.skill_induced_regression:
            return "regressed"
        if self.learned.judgment.answer_correct and _has_efficiency_improvement(
            self.baseline.measurement,
            self.learned.measurement,
        ):
            return "improved_efficiency"
        return "no_measured_change"


@dataclass(frozen=True, slots=True)
class LearningEffectivenessReport:
    """One deterministic baseline-versus-learned report."""

    comparisons: tuple[BenchmarkComparison, ...]

    def __post_init__(self) -> None:
        comparisons = tuple(self.comparisons)
        if not comparisons or len(comparisons) > EVALUATION_MAX_CASES:
            raise ValueError(f"report requires 1 to {EVALUATION_MAX_CASES} comparisons")
        if any(not isinstance(item, BenchmarkComparison) for item in comparisons):
            raise TypeError("comparisons must contain BenchmarkComparison records")
        ordered = tuple(sorted(comparisons, key=lambda item: item.case_id))
        if len({item.case_id for item in ordered}) != len(ordered):
            raise ValueError("report case IDs must be unique")
        object.__setattr__(self, "comparisons", ordered)

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic machine-readable content-free measurements."""

        baseline = tuple(item.baseline for item in self.comparisons)
        learned = tuple(item.learned for item in self.comparisons)
        return {
            "case_count": len(self.comparisons),
            "baseline": _aggregate_outcomes(baseline),
            "learned": _aggregate_outcomes(learned),
            "hard_safety": {
                "required_zero": (
                    "stale_activation_count",
                    "conflicting_claim_selection_count",
                    "cross_source_leakage_count",
                ),
                "passed": all(
                    item.learned.judgment.hard_safety_passed
                    for item in self.comparisons
                ),
            },
            "cases": [
                {
                    "case_id": item.case_id,
                    "verdict": item.verdict,
                    "baseline": _outcome_mapping(item.baseline),
                    "learned": _outcome_mapping(item.learned),
                    "difference": _difference_mapping(
                        item.baseline,
                        item.learned,
                    ),
                }
                for item in self.comparisons
            ],
        }

    def render_markdown(self) -> str:
        """Render a concise human-readable effectiveness report."""

        mapping = self.to_mapping()
        baseline = _mapping(mapping["baseline"])
        learned = _mapping(mapping["learned"])
        hard_safety = _mapping(mapping["hard_safety"])
        lines = [
            "# Learning effectiveness",
            "",
            (
                f"Cases: {len(self.comparisons)} · hard safety: "
                f"{'PASS' if hard_safety['passed'] is True else 'FAIL'}"
            ),
            "",
            "| Measure | Baseline | Learned |",
            "| --- | ---: | ---: |",
            *(
                _summary_row(label, baseline, learned, key)
                for label, key in _REPORT_ROWS
            ),
            "",
            "| Case | Result | Correctness Δ | Tool-call Δ | Token Δ |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for comparison in self.comparisons:
            difference = _difference_mapping(
                comparison.baseline,
                comparison.learned,
            )
            lines.append(
                f"| {comparison.case_id} | {comparison.verdict} | "
                f"{difference['correctness_score']} | "
                f"{difference['tool_calls']} | "
                f"{difference['total_tokens']} |"
            )
        lines.extend(
            [
                "",
                (
                    "Improvement is based on labeled correctness or repeated-task "
                    "efficiency with correctness preserved; stored artifact counts "
                    "are not treated as evidence of improvement."
                ),
                "",
            ]
        )
        return "\n".join(lines)


def measure_observer_events(events: Sequence[AgentEvent]) -> RunMeasurement:
    """Aggregate bounded, content-free observer events held by the caller."""

    values = tuple(events)
    if len(values) > EVALUATION_MAX_EVENTS_PER_OUTCOME:
        raise ValueError("observer event collection exceeds the evaluation input bound")
    if any(not isinstance(event, AgentEvent) for event in values):
        raise TypeError("events must contain AgentEvent records")

    counters = {
        field.name: 0
        for field in fields(RunMeasurement)
        if field.name not in {"estimated_cost_usd", "cost_complete"}
    }
    pending_failed_sql: set[str] = set()
    completed_runs = 0
    complete_cost = Decimal("0")
    costs_available = True
    for event in values:
        data = event.data
        if event.kind is AgentEventKind.MODEL_COMPLETED:
            counters["model_calls"] += 1
        elif event.kind is AgentEventKind.TOOL_STARTED:
            counters["tool_calls"] += 1
            tool_name = _optional_text(data, "tool_name")
            if tool_name in _CATALOG_TOOLS:
                counters["catalog_discovery_calls"] += 1
            if tool_name in _LEARNING_WRITE_TOOLS:
                counters["learning_proposals"] += 1
        elif event.kind is AgentEventKind.TOOL_COMPLETED:
            tool_name = _optional_text(data, "tool_name")
            success = data.get("success")
            if tool_name in _SQL_TOOLS and success is False:
                counters["failed_sql_calls"] += 1
                pending_failed_sql.add(event.run_id)
            elif (
                tool_name in _SQL_TOOLS
                and success is True
                and event.run_id in pending_failed_sql
            ):
                counters["corrected_sql_calls"] += 1
                pending_failed_sql.remove(event.run_id)
            if tool_name in _LEARNING_WRITE_TOOLS:
                if success is True:
                    counters["proposal_succeeded"] += 1
                elif success is False:
                    counters["proposal_failed"] += 1
        elif event.kind is AgentEventKind.APPROVAL_REQUESTED:
            counters["approvals_requested"] += 1
        elif event.kind is AgentEventKind.APPROVAL_DECIDED:
            outcome = _optional_text(data, "outcome")
            if outcome == "approved":
                counters["approvals"] += 1
            elif outcome == "denied":
                counters["denials"] += 1
            elif outcome == "failed":
                counters["approval_failures"] += 1
        elif event.kind is AgentEventKind.RUN_COMPLETED:
            completed_runs += 1
            for field_name in (
                "duration_ms",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "total_tokens",
            ):
                counters[field_name] += _event_integer(data, field_name)
            if data.get("cost_status") != "complete":
                costs_available = False
                continue
            amount = data.get("cost_amount_usd")
            if not isinstance(amount, str):
                costs_available = False
                continue
            try:
                selected_cost = Decimal(amount)
            except InvalidOperation as error:
                raise ValueError("observer cost_amount_usd is invalid") from error
            if not selected_cost.is_finite() or selected_cost < 0:
                raise ValueError("observer cost_amount_usd is invalid")
            complete_cost += selected_cost

    cost_complete = completed_runs > 0 and costs_available
    return RunMeasurement(
        **counters,
        estimated_cost_usd=complete_cost if cost_complete else None,
        cost_complete=cost_complete,
    )


def build_learning_effectiveness_report(
    outcomes: Sequence[BenchmarkOutcome],
) -> LearningEffectivenessReport:
    """Pair one baseline and learned outcome per human-labeled benchmark case."""

    values = tuple(outcomes)
    if not values or len(values) > EVALUATION_MAX_CASES * 2:
        raise ValueError(
            f"outcomes require 2 to {EVALUATION_MAX_CASES * 2} bounded records"
        )
    if any(not isinstance(item, BenchmarkOutcome) for item in values):
        raise TypeError("outcomes must contain BenchmarkOutcome records")
    grouped: dict[str, dict[BenchmarkVariant, BenchmarkOutcome]] = {}
    for outcome in values:
        variants = grouped.setdefault(outcome.case_id, {})
        if outcome.variant in variants:
            raise ValueError(
                f"duplicate {outcome.variant.value} outcome for {outcome.case_id}"
            )
        variants[outcome.variant] = outcome
    comparisons: list[BenchmarkComparison] = []
    for case_id, variants in sorted(grouped.items()):
        if set(variants) != {BenchmarkVariant.BASELINE, BenchmarkVariant.LEARNED}:
            raise ValueError(f"case {case_id} requires baseline and learned outcomes")
        comparisons.append(
            BenchmarkComparison(
                case_id,
                variants[BenchmarkVariant.BASELINE],
                variants[BenchmarkVariant.LEARNED],
            )
        )
    return LearningEffectivenessReport(tuple(comparisons))


def _aggregate_outcomes(
    outcomes: tuple[BenchmarkOutcome, ...],
) -> dict[str, object]:
    judgments = tuple(item.judgment for item in outcomes)
    measurements = tuple(item.measurement for item in outcomes)
    result: dict[str, object] = {}
    for field_name in (
        "answer_correct",
        "business_definition_correct",
        "source_selection_correct",
        "resource_selection_correct",
        "field_selection_correct",
        "semantic_constraints_satisfied",
        "relevant_skill_selected",
    ):
        labels = tuple(
            value
            for judgment in judgments
            if (value := getattr(judgment, field_name)) is not None
        )
        result[field_name] = {
            "correct": sum(value is True for value in labels),
            "evaluated": len(labels),
        }
    recalled = sum(item.recalled_annotation_count for item in judgments)
    irrelevant = sum(item.irrelevant_recall_count for item in judgments)
    result["irrelevant_recall_rate"] = {
        "irrelevant": irrelevant,
        "recalled": recalled,
    }
    for field_name in (
        "stale_activation_count",
        "conflicting_claim_selection_count",
        "cross_source_leakage_count",
    ):
        result[field_name] = sum(getattr(item, field_name) for item in judgments)
    result["skill_induced_regressions"] = sum(
        item.skill_induced_regression for item in judgments
    )
    for field_name in (
        "model_calls",
        "tool_calls",
        "catalog_discovery_calls",
        "failed_sql_calls",
        "corrected_sql_calls",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
        "learning_proposals",
        "proposal_succeeded",
        "proposal_failed",
        "approvals_requested",
        "approvals",
        "denials",
        "approval_failures",
    ):
        result[field_name] = sum(getattr(item, field_name) for item in measurements)
    if measurements and all(item.cost_complete for item in measurements):
        result["estimated_cost_usd"] = format(
            sum(
                (
                    item.estimated_cost_usd
                    for item in measurements
                    if item.estimated_cost_usd is not None
                ),
                Decimal("0"),
            ),
            "f",
        )
    else:
        result["estimated_cost_usd"] = None
    return result


def _outcome_mapping(outcome: BenchmarkOutcome) -> dict[str, object]:
    judgment = outcome.judgment
    measurement = outcome.measurement
    return {
        "judgment": {
            item.name: getattr(judgment, item.name) for item in fields(judgment)
        },
        "measurement": {
            item.name: (format(value, "f") if isinstance(value, Decimal) else value)
            for item in fields(measurement)
            if (value := getattr(measurement, item.name)) is not None
        },
    }


def _difference_mapping(
    baseline: BenchmarkOutcome,
    learned: BenchmarkOutcome,
) -> dict[str, object]:
    left = baseline.measurement
    right = learned.measurement
    result: dict[str, object] = {
        "correctness_score": _correctness_score(learned.judgment)
        - _correctness_score(baseline.judgment),
        "answer_correct": int(learned.judgment.answer_correct)
        - int(baseline.judgment.answer_correct),
    }
    for field_name in (
        "model_calls",
        "tool_calls",
        "catalog_discovery_calls",
        "failed_sql_calls",
        "corrected_sql_calls",
        "duration_ms",
        "total_tokens",
        "learning_proposals",
        "proposal_succeeded",
        "proposal_failed",
        "approvals",
        "denials",
    ):
        result[field_name] = getattr(right, field_name) - getattr(left, field_name)
    result["estimated_cost_usd"] = (
        format(right.estimated_cost_usd - left.estimated_cost_usd, "f")
        if right.estimated_cost_usd is not None and left.estimated_cost_usd is not None
        else None
    )
    return result


def _correctness_score(judgment: BenchmarkJudgment) -> int:
    return sum(
        getattr(judgment, field_name) is True
        for field_name in (
            "answer_correct",
            "business_definition_correct",
            "source_selection_correct",
            "resource_selection_correct",
            "field_selection_correct",
            "semantic_constraints_satisfied",
            "relevant_skill_selected",
        )
    )


def _has_efficiency_improvement(
    baseline: RunMeasurement,
    learned: RunMeasurement,
) -> bool:
    return any(
        getattr(learned, field_name) < getattr(baseline, field_name)
        for field_name in (
            "model_calls",
            "tool_calls",
            "catalog_discovery_calls",
            "failed_sql_calls",
            "duration_ms",
            "total_tokens",
        )
    )


def _summary_row(
    label: str,
    baseline: Mapping[str, object],
    learned: Mapping[str, object],
    key: str,
) -> str:
    return (
        f"| {label} | {_summary_value(baseline[key])} | "
        f"{_summary_value(learned[key])} |"
    )


def _summary_value(value: object) -> str:
    if isinstance(value, Mapping):
        if "correct" in value and "evaluated" in value:
            return f"{value['correct']}/{value['evaluated']}"
        if "irrelevant" in value and "recalled" in value:
            return f"{value['irrelevant']}/{value['recalled']}"
    return str(value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("report mapping is invalid")
    return value


def _optional_text(data: Mapping[str, object], name: str) -> str | None:
    value = data.get(name)
    return value if isinstance(value, str) else None


def _event_integer(data: Mapping[str, object], name: str) -> int:
    value = data.get(name)
    _non_negative_integer(value, f"observer {name}")
    assert isinstance(value, int)
    return value


def _non_negative_integer(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _candidate_review_totals(
    measurements: tuple[CandidateReviewMeasurement, ...],
) -> dict[str, object]:
    integer_fields = (
        "proposed_candidates",
        "accepted_candidates",
        "rejected_candidates",
        "false_positive_candidates",
        "duplicate_candidates_suppressed",
        "background_model_calls",
        "background_total_tokens",
        "background_duration_ms",
        "stale_activation_count",
        "conflicting_claim_selection_count",
        "cross_source_leakage_count",
    )
    result: dict[str, object] = {
        field_name: sum(cast(int, getattr(item, field_name)) for item in measurements)
        for field_name in integer_fields
    }
    if all(item.background_cost_complete for item in measurements):
        result["background_estimated_cost_usd"] = format(
            sum(
                (
                    cast(Decimal, item.background_estimated_cost_usd)
                    for item in measurements
                ),
                Decimal("0"),
            ),
            "f",
        )
        result["background_cost_complete"] = True
    else:
        result["background_estimated_cost_usd"] = None
        result["background_cost_complete"] = False
    return result


def _rate(numerator: int, denominator: int) -> str:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("candidate evaluation rate inputs are invalid")
    return format(Decimal(numerator) / Decimal(denominator), ".6f")


__all__ = [
    "CandidateReviewMeasurement",
    "CandidateReviewReport",
    "BenchmarkComparison",
    "BenchmarkJudgment",
    "BenchmarkOutcome",
    "BenchmarkVariant",
    "EVALUATION_MAX_CASES",
    "EVALUATION_MAX_EVENTS_PER_OUTCOME",
    "LearningEffectivenessReport",
    "RunMeasurement",
    "build_learning_effectiveness_report",
    "measure_observer_events",
]
