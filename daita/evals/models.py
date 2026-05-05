"""Result models for Daita evals."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["passed", "failed", "warned"]
Severity = Literal["error", "warning"]


class AssertionResult(BaseModel):
    id: str
    status: Status
    code: str
    message: str
    severity: Severity = "error"
    assertion_path: str | None = None
    observed: Any = None
    expected: Any = None
    fix_hints: list[str] = Field(default_factory=list)
    related_tool_calls: list[int] = Field(default_factory=list)


class RunMetrics(BaseModel):
    latency_ms: float | None = None
    tokens_total: int | None = None
    cost: float | None = None
    iterations: int | None = None


class ToolCallEvidence(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class ExecutionSpan(BaseModel):
    kind: Literal["skill", "plugin", "tool", "workflow"]
    name: str
    operation: str | None = None
    status: str = "passed"
    latency_ms: float | None = None
    error: str | None = None
    parent_id: str | None = None
    trace_id: str | None = None


class JudgeCriterionResult(BaseModel):
    id: str
    passed: bool
    score: float
    evidence: str = ""
    required: bool = False


class JudgeResult(BaseModel):
    assertion_id: str = "judge"
    provider: str
    model: str
    score: float
    pass_score: float
    passed: bool
    reasoning: str = ""
    criteria: list[JudgeCriterionResult] = Field(default_factory=list)
    latency_ms: float | None = None
    cost: float | None = None
    tokens_total: int | None = None
    input_artifact_path: str | None = None
    output_artifact_path: str | None = None
    raw_output: Any = None
    input_payload: dict[str, Any] | None = Field(default=None, exclude=True)
    output_payload: dict[str, Any] | None = Field(default=None, exclude=True)


class RunResult(BaseModel):
    run_id: str
    status: Status
    prompt_hash: str
    answer_hash: str
    final_answer_preview: str
    final_answer: str | None = None
    tool_calls: list[ToolCallEvidence] = Field(default_factory=list)
    execution_spans: list[ExecutionSpan] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    trace_id: str | None = None
    assertions: list[AssertionResult] = Field(default_factory=list)
    judges: list[JudgeResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class StabilitySummary(BaseModel):
    answer_variants: int = 0
    tool_sequence_variants: int = 0
    cost_min: float | None = None
    cost_max: float | None = None
    latency_ms_min: float | None = None
    latency_ms_max: float | None = None
    token_min: int | None = None
    token_max: int | None = None


class CaseResult(BaseModel):
    case_id: str
    status: Status
    runs_requested: int
    runs_passed: int
    runs_failed: int
    pass_rule: str
    stability: StabilitySummary = Field(default_factory=StabilitySummary)
    runs: list[RunResult] = Field(default_factory=list)
    assertions: list[AssertionResult] = Field(default_factory=list)


class EvalFailure(BaseModel):
    case_id: str
    run_id: str | None = None
    code: str
    message: str
    severity: Severity
    assertion_path: str | None = None
    observed: Any = None
    expected: Any = None
    artifact_path: str | None = None
    fix_hints: list[str] = Field(default_factory=list)
    related_tool_calls: list[int] = Field(default_factory=list)
    related_trace_ids: list[str] = Field(default_factory=list)


class ReportSummary(BaseModel):
    cases_total: int = 0
    cases_passed: int = 0
    cases_failed: int = 0
    cases_warned: int = 0
    runs_total: int = 0
    runs_passed: int = 0
    runs_failed: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    judge_calls: int = 0
    judge_cost: float = 0.0
    judge_latency_ms: float = 0.0
    judge_tokens_total: int = 0


class EvalReport(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    suite: dict[str, Any]
    agent: dict[str, Any]
    status: Literal["passed", "failed", "warned"]
    score: float
    summary: ReportSummary
    failures: list[EvalFailure] = Field(default_factory=list)
    cases: list[CaseResult] = Field(default_factory=list)
    artifact_path: str | None = None
    baseline_comparison: dict[str, Any] | None = None
