"""
Public policy objects for ``Agent.from_db()`` budget controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .intent import DbEvidenceMode, DbIntent, DbIntentKind
from . import tool_selection

BudgetPreset = Literal["auto", "full", "compact", "retrieval"]
DbCapabilityCondition = Literal[
    "needs_schema_search",
    "needs_full_schema_navigation",
    "needs_validation",
    "needs_quality",
    "needs_write",
    "needs_lineage",
    "needs_memory_write",
    "has_vector_columns",
    "requested_analyst_tool",
]


@dataclass(frozen=True)
class ConditionalCapability:
    """Capability included when a named DB-local condition is true."""

    capability: str
    when: DbCapabilityCondition


@dataclass(frozen=True)
class DbWorkflowPolicy:
    """Declarative runtime policy for a structured DB intent."""

    intent_kind: DbIntentKind
    evidence_mode: DbEvidenceMode
    required_phases: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    optional_capabilities: tuple[ConditionalCapability, ...]
    terminal_capabilities: tuple[str, ...]
    allow_catalog_final: bool
    require_executed_query: bool
    max_model_turns: int
    max_tool_calls: int
    max_repair_attempts: int
    workflow_guidance: str
    answer_guidance: str
    fast_path_capabilities: tuple[str, ...] = ()


_SCHEMA_WORKFLOW_GUIDANCE = (
    "Use catalog/schema tools to gather structural evidence. Do not plan or "
    "execute SQL for schema-only questions."
)
_SCHEMA_ANSWER_GUIDANCE = (
    "catalog/schema evidence is sufficient; explain confidence and caveats "
    "from table, column, and relationship metadata. Do not query rows unless "
    "the user asks for values, counts, samples, or calculations."
)
_QUERY_WORKFLOW_GUIDANCE = (
    "Use focused query tools. Do not repeat a plan after it returns a plan_id."
)
_QUERY_ANSWER_GUIDANCE = (
    "executed query evidence is required for data answers; include readable "
    "labels with ids when available."
)
_SCHEMA_SEARCH_OPTIONAL = (
    ConditionalCapability(
        tool_selection.CATALOG_SEARCH_CAPABILITY, "needs_schema_search"
    ),
)
_FULL_SCHEMA_NAVIGATION_OPTIONAL = (
    ConditionalCapability(
        tool_selection.CATALOG_SEARCH_CAPABILITY, "needs_full_schema_navigation"
    ),
    ConditionalCapability(
        tool_selection.CATALOG_INSPECT_CAPABILITY, "needs_full_schema_navigation"
    ),
    ConditionalCapability(
        tool_selection.CATALOG_RELATIONSHIP_PATHS_CAPABILITY,
        "needs_full_schema_navigation",
    ),
)
_COMMON_OPTIONAL_CAPABILITIES = (
    ConditionalCapability(
        tool_selection.DB_VALIDATE_SQL_CAPABILITY, "needs_validation"
    ),
    ConditionalCapability(
        tool_selection.DB_QUALITY_PROFILE_CAPABILITY, "needs_quality"
    ),
    ConditionalCapability(tool_selection.DB_WRITE_CAPABILITY, "needs_write"),
    ConditionalCapability(tool_selection.DB_LINEAGE_TRACE_CAPABILITY, "needs_lineage"),
    ConditionalCapability(
        tool_selection.DB_MEMORY_WRITE_CAPABILITY, "needs_memory_write"
    ),
    ConditionalCapability(
        tool_selection.VECTOR_SEARCH_CAPABILITY, "has_vector_columns"
    ),
    *(
        ConditionalCapability(
            f"{tool_selection.ANALYST_CAPABILITY_PREFIX}{tool_name}",
            "requested_analyst_tool",
        )
        for tool_name in sorted(tool_selection.ANALYST_TOOL_NAMES)
    ),
)


DB_WORKFLOW_POLICIES = {
    DbIntentKind.CONVERSATIONAL: DbWorkflowPolicy(
        intent_kind=DbIntentKind.CONVERSATIONAL,
        evidence_mode=DbEvidenceMode.NONE,
        required_phases=(),
        required_capabilities=(),
        optional_capabilities=_COMMON_OPTIONAL_CAPABILITIES,
        terminal_capabilities=(),
        allow_catalog_final=False,
        require_executed_query=False,
        max_model_turns=4,
        max_tool_calls=5,
        max_repair_attempts=1,
        workflow_guidance="Answer directly; no DB evidence is required by this contract.",
        answer_guidance="answer directly without DB tool evidence when appropriate.",
    ),
    DbIntentKind.SCHEMA_ONLY: DbWorkflowPolicy(
        intent_kind=DbIntentKind.SCHEMA_ONLY,
        evidence_mode=DbEvidenceMode.CATALOG,
        required_phases=(),
        required_capabilities=(
            tool_selection.CATALOG_SEARCH_CAPABILITY,
            tool_selection.CATALOG_INSPECT_CAPABILITY,
            tool_selection.CATALOG_RELATIONSHIP_PATHS_CAPABILITY,
        ),
        optional_capabilities=_COMMON_OPTIONAL_CAPABILITIES,
        terminal_capabilities=(
            tool_selection.CATALOG_SEARCH_CAPABILITY,
            tool_selection.CATALOG_INSPECT_CAPABILITY,
            tool_selection.CATALOG_RELATIONSHIP_PATHS_CAPABILITY,
        ),
        allow_catalog_final=True,
        require_executed_query=False,
        max_model_turns=2,
        max_tool_calls=5,
        max_repair_attempts=0,
        workflow_guidance=_SCHEMA_WORKFLOW_GUIDANCE,
        answer_guidance=_SCHEMA_ANSWER_GUIDANCE,
    ),
    DbIntentKind.SCHEMA_QUESTION: DbWorkflowPolicy(
        intent_kind=DbIntentKind.SCHEMA_QUESTION,
        evidence_mode=DbEvidenceMode.CATALOG,
        required_phases=(),
        required_capabilities=(
            tool_selection.CATALOG_SEARCH_CAPABILITY,
            tool_selection.CATALOG_INSPECT_CAPABILITY,
            tool_selection.CATALOG_RELATIONSHIP_PATHS_CAPABILITY,
        ),
        optional_capabilities=_COMMON_OPTIONAL_CAPABILITIES,
        terminal_capabilities=(
            tool_selection.CATALOG_SEARCH_CAPABILITY,
            tool_selection.CATALOG_INSPECT_CAPABILITY,
            tool_selection.CATALOG_RELATIONSHIP_PATHS_CAPABILITY,
        ),
        allow_catalog_final=True,
        require_executed_query=False,
        max_model_turns=2,
        max_tool_calls=5,
        max_repair_attempts=0,
        workflow_guidance=_SCHEMA_WORKFLOW_GUIDANCE,
        answer_guidance=_SCHEMA_ANSWER_GUIDANCE,
    ),
    DbIntentKind.SCHEMA_EXPLAIN: DbWorkflowPolicy(
        intent_kind=DbIntentKind.SCHEMA_EXPLAIN,
        evidence_mode=DbEvidenceMode.CATALOG,
        required_phases=(),
        required_capabilities=(
            tool_selection.CATALOG_SEARCH_CAPABILITY,
            tool_selection.CATALOG_INSPECT_CAPABILITY,
            tool_selection.CATALOG_RELATIONSHIP_PATHS_CAPABILITY,
        ),
        optional_capabilities=_COMMON_OPTIONAL_CAPABILITIES,
        terminal_capabilities=(
            tool_selection.CATALOG_SEARCH_CAPABILITY,
            tool_selection.CATALOG_INSPECT_CAPABILITY,
            tool_selection.CATALOG_RELATIONSHIP_PATHS_CAPABILITY,
        ),
        allow_catalog_final=True,
        require_executed_query=False,
        max_model_turns=2,
        max_tool_calls=5,
        max_repair_attempts=0,
        workflow_guidance=_SCHEMA_WORKFLOW_GUIDANCE,
        answer_guidance=_SCHEMA_ANSWER_GUIDANCE,
    ),
    DbIntentKind.DATA_QUERY_SIMPLE: DbWorkflowPolicy(
        intent_kind=DbIntentKind.DATA_QUERY_SIMPLE,
        evidence_mode=DbEvidenceMode.QUERY,
        required_phases=("plan", "execute"),
        required_capabilities=(
            tool_selection.DB_PLAN_CAPABILITY,
            tool_selection.DB_EXECUTE_CAPABILITY,
        ),
        optional_capabilities=(
            *_SCHEMA_SEARCH_OPTIONAL,
            *_FULL_SCHEMA_NAVIGATION_OPTIONAL,
            *_COMMON_OPTIONAL_CAPABILITIES,
        ),
        terminal_capabilities=(
            tool_selection.DB_COMPILE_AND_EXECUTE_CAPABILITY,
            tool_selection.DB_EXECUTE_CAPABILITY,
        ),
        fast_path_capabilities=(tool_selection.DB_COMPILE_AND_EXECUTE_CAPABILITY,),
        allow_catalog_final=False,
        require_executed_query=True,
        max_model_turns=3,
        max_tool_calls=3,
        max_repair_attempts=1,
        workflow_guidance=_QUERY_WORKFLOW_GUIDANCE,
        answer_guidance=_QUERY_ANSWER_GUIDANCE,
    ),
    DbIntentKind.DATA_QUERY_CATALOG_ASSISTED: DbWorkflowPolicy(
        intent_kind=DbIntentKind.DATA_QUERY_CATALOG_ASSISTED,
        evidence_mode=DbEvidenceMode.QUERY,
        required_phases=("catalog", "plan", "execute"),
        required_capabilities=(
            tool_selection.DB_PLAN_CAPABILITY,
            tool_selection.DB_EXECUTE_CAPABILITY,
            tool_selection.CATALOG_SEARCH_CAPABILITY,
            tool_selection.CATALOG_INSPECT_CAPABILITY,
            tool_selection.CATALOG_RELATIONSHIP_PATHS_CAPABILITY,
        ),
        optional_capabilities=_COMMON_OPTIONAL_CAPABILITIES,
        terminal_capabilities=(
            tool_selection.DB_COMPILE_AND_EXECUTE_CAPABILITY,
            tool_selection.DB_EXECUTE_CAPABILITY,
        ),
        allow_catalog_final=False,
        require_executed_query=True,
        max_model_turns=6,
        max_tool_calls=8,
        max_repair_attempts=2,
        workflow_guidance=(
            "Use catalog tools to resolve tables, columns, and relationships, then "
            "plan with db_plan_query and execute with db_query."
        ),
        answer_guidance=(
            "catalog evidence may guide planning, but final numeric/data answers "
            "require executed query evidence."
        ),
    ),
    DbIntentKind.MANUAL_SQL: DbWorkflowPolicy(
        intent_kind=DbIntentKind.MANUAL_SQL,
        evidence_mode=DbEvidenceMode.QUERY,
        required_phases=("execute",),
        required_capabilities=(
            tool_selection.DB_VALIDATE_SQL_CAPABILITY,
            tool_selection.DB_EXECUTE_CAPABILITY,
        ),
        optional_capabilities=_COMMON_OPTIONAL_CAPABILITIES,
        terminal_capabilities=(tool_selection.DB_EXECUTE_CAPABILITY,),
        allow_catalog_final=False,
        require_executed_query=True,
        max_model_turns=3,
        max_tool_calls=3,
        max_repair_attempts=1,
        workflow_guidance="Validate the provided SQL before execution; execute only when safe.",
        answer_guidance="validated/executed SQL evidence is required before final answers.",
    ),
    DbIntentKind.ADMIN_OR_WRITE: DbWorkflowPolicy(
        intent_kind=DbIntentKind.ADMIN_OR_WRITE,
        evidence_mode=DbEvidenceMode.QUERY,
        required_phases=("execute",),
        required_capabilities=(
            tool_selection.DB_WRITE_CAPABILITY,
            tool_selection.DB_VALIDATE_SQL_CAPABILITY,
            tool_selection.DB_EXECUTE_CAPABILITY,
        ),
        optional_capabilities=_COMMON_OPTIONAL_CAPABILITIES,
        terminal_capabilities=(
            tool_selection.DB_WRITE_CAPABILITY,
            tool_selection.DB_EXECUTE_CAPABILITY,
        ),
        allow_catalog_final=False,
        require_executed_query=True,
        max_model_turns=4,
        max_tool_calls=5,
        max_repair_attempts=1,
        workflow_guidance="Apply write/admin guardrails before any mutating action.",
        answer_guidance=_QUERY_ANSWER_GUIDANCE,
    ),
    DbIntentKind.MEMORY_ONLY: DbWorkflowPolicy(
        intent_kind=DbIntentKind.MEMORY_ONLY,
        evidence_mode=DbEvidenceMode.MEMORY,
        required_phases=(),
        required_capabilities=(tool_selection.DB_MEMORY_WRITE_CAPABILITY,),
        optional_capabilities=_COMMON_OPTIONAL_CAPABILITIES,
        terminal_capabilities=(tool_selection.DB_MEMORY_WRITE_CAPABILITY,),
        allow_catalog_final=False,
        require_executed_query=False,
        max_model_turns=3,
        max_tool_calls=3,
        max_repair_attempts=1,
        workflow_guidance="Use DB memory tools only for the requested memory operation.",
        answer_guidance="answer from the memory operation result or confirm completion.",
    ),
}


def workflow_policy_for_intent(intent: Any) -> DbWorkflowPolicy:
    """Return the runtime workflow policy for an intent-like value."""

    kind = _intent_kind(intent)
    return DB_WORKFLOW_POLICIES[kind]


@dataclass(frozen=True)
class SchemaPromptPolicy:
    """Budget controls for schema content included in the system prompt."""

    max_inline_schema_tokens: int = 2500
    max_inline_tables: int = 12
    max_inline_columns: int = 120
    compact_table_limit: int = 80
    summary_table_limit: int = 40
    max_inline_relationships: int = 12
    include_column_comments: bool = False
    include_sample_values: bool = False
    relationship_mode: str = "summary"
    preferred_strategy: BudgetPreset = "auto"

    def __post_init__(self) -> None:
        _require_positive("max_inline_schema_tokens", self.max_inline_schema_tokens)
        _require_positive("max_inline_tables", self.max_inline_tables)
        _require_positive("max_inline_columns", self.max_inline_columns)
        _require_positive("compact_table_limit", self.compact_table_limit)
        _require_positive("summary_table_limit", self.summary_table_limit)
        _require_positive("max_inline_relationships", self.max_inline_relationships)
        if self.relationship_mode not in {"summary", "full"}:
            raise ValueError("relationship_mode must be 'summary' or 'full'")
        if self.preferred_strategy not in {"auto", "full", "compact", "retrieval"}:
            raise ValueError(
                "preferred_strategy must be 'auto', 'full', 'compact', or 'retrieval'"
            )


@dataclass(frozen=True)
class PromptBuildResult:
    """Prompt text plus budget metadata used by ``from_db`` diagnostics."""

    prompt: str
    strategy: str
    estimated_tokens: int
    table_count: int
    column_count: int
    omitted_tables: list[str] = field(default_factory=list)
    budget_exceeded: bool = False


@dataclass(frozen=True)
class ToolResultPolicy:
    """Controls for compacting DB tool results before LLM synthesis turns."""

    max_result_tokens: int = 800
    max_rows_inline: int = 12
    max_cell_chars: int = 220
    summarize_large_json: bool = True
    omitted_column_patterns: list[str] = field(
        default_factory=lambda: [
            "*_context",
            "*_payload",
            "output_preview",
            "workflow_context",
            "trace_data",
        ]
    )

    def __post_init__(self) -> None:
        _require_positive("max_result_tokens", self.max_result_tokens)
        _require_positive("max_rows_inline", self.max_rows_inline)
        _require_positive("max_cell_chars", self.max_cell_chars)


def schema_prompt_policy_for_budget(budget: BudgetPreset) -> SchemaPromptPolicy:
    """Return the default schema prompt policy for a public budget preset."""

    if budget == "auto":
        return SchemaPromptPolicy()
    if budget == "full":
        return SchemaPromptPolicy(
            max_inline_schema_tokens=6000,
            max_inline_tables=30,
            max_inline_columns=300,
            summary_table_limit=80,
            max_inline_relationships=50,
            preferred_strategy="full",
        )
    if budget == "compact":
        return SchemaPromptPolicy(
            max_inline_schema_tokens=1800,
            max_inline_tables=8,
            max_inline_columns=80,
            compact_table_limit=30,
            summary_table_limit=40,
            max_inline_relationships=16,
            preferred_strategy="compact",
        )
    if budget == "retrieval":
        return SchemaPromptPolicy(
            max_inline_schema_tokens=1000,
            max_inline_tables=4,
            max_inline_columns=40,
            compact_table_limit=12,
            summary_table_limit=30,
            max_inline_relationships=8,
            preferred_strategy="retrieval",
        )
    raise ValueError("budget must be 'auto', 'full', 'compact', or 'retrieval'")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _intent_kind(intent: Any) -> DbIntentKind:
    if isinstance(intent, DbIntent):
        return intent.kind
    if isinstance(intent, DbIntentKind):
        return intent
    if isinstance(intent, str) and intent:
        try:
            return DbIntentKind(intent)
        except ValueError:
            return DbIntentKind.DATA_QUERY_SIMPLE
    return DbIntentKind.DATA_QUERY_SIMPLE
