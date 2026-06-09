"""Configuration models for runtime-native Daita eval suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

RunMode = Literal["sequential_same_agent"]
PassRule = Literal["all_runs"]
JudgeRunWhen = Literal["always", "after_deterministic_pass"]


class EvalConfigModel(BaseModel):
    """Base config model that rejects misspelled eval fields."""

    model_config = ConfigDict(extra="forbid")


class FromDbConfig(EvalConfigModel):
    """First-class ``daita.db.from_db`` target configuration."""

    source: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(EvalConfigModel):
    """Runtime-native target configuration."""

    factory: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)
    from_db: FromDbConfig | None = None
    label: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "AgentConfig":
        if bool(self.factory) == bool(self.from_db):
            raise ValueError("agent must configure exactly one of factory or from_db")
        return self


class SuiteDefaults(EvalConfigModel):
    """Defaults applied to each eval case unless overridden."""

    runs: int = Field(default=1, ge=1)
    run_mode: RunMode = "sequential_same_agent"
    pass_rule: PassRule = "all_runs"
    max_iterations: int = Field(default=20, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_cost: float | None = Field(default=None, ge=0)
    max_capability_calls: int | None = Field(default=None, ge=0)


class NumericExpectation(EvalConfigModel):
    label: str
    expected: float
    tolerance: float = Field(default=0.0, ge=0)


class AnswerExpectations(EvalConfigModel):
    equals: str | None = None
    contains: list[str] = Field(default_factory=list)
    not_contains: list[str] = Field(default_factory=list)
    regex: list[str] = Field(default_factory=list)
    numeric: list[NumericExpectation] = Field(default_factory=list)


class CapabilityExpectations(EvalConfigModel):
    required: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    required_owners: list[str] = Field(default_factory=list)
    forbidden_owners: list[str] = Field(default_factory=list)
    max_calls: int | None = Field(default=None, ge=0)


class BudgetExpectations(EvalConfigModel):
    max_tokens: int | None = Field(default=None, ge=0)
    max_cost: float | None = Field(default=None, ge=0)
    max_latency_ms: float | None = Field(default=None, ge=0)
    max_iterations: int | None = Field(default=None, ge=1)


class SQLExpectations(EvalConfigModel):
    read_only: bool = False
    require_limit: bool = False
    required_tables: list[str] = Field(default_factory=list)
    forbidden_tables: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    max_rows_returned: int | None = Field(default=None, ge=0)


class TaskExpectations(EvalConfigModel):
    """Runtime task lifecycle expectations."""

    required_statuses: list[str] = Field(default_factory=list)
    forbidden_statuses: list[str] = Field(default_factory=lambda: ["failed"])
    max_errors: int | None = Field(default=None, ge=0)


class EvidenceExpectations(EvalConfigModel):
    required_kinds: list[str] = Field(default_factory=list)
    forbidden_kinds: list[str] = Field(default_factory=list)
    required_owners: list[str] = Field(default_factory=list)
    forbidden_owners: list[str] = Field(default_factory=list)
    require_accepted: bool = True


class ResultExpectations(EvalConfigModel):
    """Expectations over runtime ``query.result`` evidence payloads."""

    required_columns: list[str] = Field(default_factory=list)
    required_rows: list[dict[str, Any]] = Field(default_factory=list)
    forbidden_rows: list[dict[str, Any]] = Field(default_factory=list)
    min_rows: int | None = Field(default=None, ge=0)
    max_rows: int | None = Field(default=None, ge=0)


class GovernanceExpectations(EvalConfigModel):
    allowed: bool | None = None
    blocked: bool | None = None
    pending_approval: bool | None = None
    required_policies: list[str] = Field(default_factory=list)
    forbidden_policies: list[str] = Field(default_factory=list)


class ApprovalExpectations(EvalConfigModel):
    required_statuses: list[str] = Field(default_factory=list)
    forbidden_statuses: list[str] = Field(default_factory=list)
    required_policies: list[str] = Field(default_factory=list)


class StabilityExpectations(EvalConfigModel):
    require_same_capabilities: bool = False
    max_answer_variants: int | None = Field(default=None, ge=1)
    max_cost_delta_pct: float | None = Field(default=None, ge=0)
    max_latency_delta_pct: float | None = Field(default=None, ge=0)
    max_token_delta_pct: float | None = Field(default=None, ge=0)


class JudgeCriterion(EvalConfigModel):
    id: str
    description: str
    required: bool = False
    weight: float = Field(default=1.0, ge=0)


class JudgeExpectations(EvalConfigModel):
    provider: str = "openai"
    model: str
    api_key: str | None = None
    pass_score: float = Field(default=0.8, ge=0, le=1)
    rubric: list[str] = Field(default_factory=list)
    criteria: list[JudgeCriterion] = Field(default_factory=list)
    require_all_criteria_pass: bool = False
    run_when: JudgeRunWhen = "after_deterministic_pass"
    include_evidence_payloads: bool = False
    max_evidence_payload_chars: int = Field(default=4000, ge=0)


class Expectations(EvalConfigModel):
    answer: AnswerExpectations = Field(default_factory=AnswerExpectations)
    capabilities: CapabilityExpectations = Field(default_factory=CapabilityExpectations)
    tasks: TaskExpectations = Field(default_factory=TaskExpectations)
    evidence: EvidenceExpectations = Field(default_factory=EvidenceExpectations)
    result: ResultExpectations = Field(default_factory=ResultExpectations)
    governance: GovernanceExpectations = Field(default_factory=GovernanceExpectations)
    approvals: ApprovalExpectations = Field(default_factory=ApprovalExpectations)
    budgets: BudgetExpectations = Field(default_factory=BudgetExpectations)
    sql: SQLExpectations = Field(default_factory=SQLExpectations)
    stability: StabilityExpectations = Field(default_factory=StabilityExpectations)
    judge: JudgeExpectations | None = None


class CaseTemplateConfig(EvalConfigModel):
    runs: int | None = Field(default=None, ge=1)
    run_mode: RunMode | None = None
    pass_rule: PassRule | None = None
    max_iterations: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    expectations: Expectations = Field(default_factory=Expectations)


class DatasetConfig(EvalConfigModel):
    path: str
    input_field: str = "prompt"
    id_field: str = "id"
    expected_field: str = "expected"
    metadata_field: str = "metadata"


class EvalCaseConfig(EvalConfigModel):
    id: str
    prompt: str
    runs: int | None = Field(default=None, ge=1)
    run_mode: RunMode | None = None
    pass_rule: PassRule | None = None
    max_iterations: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    expectations: Expectations = Field(default_factory=Expectations)


class ArtifactConfig(EvalConfigModel):
    output_dir: str = ".daita/evals/runs"
    max_chars: int = Field(default=50000, ge=100)
    include_full_answers: bool = True
    include_evidence_payloads: bool = False
    redact_patterns: list[str] = Field(default_factory=list)


class BaselineFailPolicy(EvalConfigModel):
    score_drop_gt: float | None = Field(default=None, ge=0)
    cost_increase_pct_gt: float | None = Field(default=None, ge=0)
    latency_increase_pct_gt: float | None = Field(default=None, ge=0)
    new_failures: bool = True
    capability_sequence_changed: bool = False


class BaselineConfig(EvalConfigModel):
    path: str | None = None
    fail_on: BaselineFailPolicy = Field(default_factory=BaselineFailPolicy)


class EvalSuiteConfig(EvalConfigModel):
    name: str
    version: int = 1
    agent: AgentConfig
    defaults: SuiteDefaults = Field(default_factory=SuiteDefaults)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    dataset: DatasetConfig | None = None
    case_template: CaseTemplateConfig = Field(default_factory=CaseTemplateConfig)
    baselines: BaselineConfig = Field(default_factory=BaselineConfig)
    cases: list[EvalCaseConfig] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "EvalSuiteConfig":
        path = Path(path)
        raw = path.read_text()
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            data = yaml.safe_load(raw) or {}
        return cls.model_validate(data)
