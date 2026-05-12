"""
Public policy objects for ``Agent.from_db()`` budget controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BudgetPreset = Literal["auto", "full", "compact", "retrieval"]


@dataclass(frozen=True)
class SchemaPromptPolicy:
    """Budget controls for schema content included in the system prompt."""

    max_inline_schema_tokens: int = 2500
    max_inline_tables: int = 12
    max_inline_columns: int = 120
    compact_table_limit: int = 80
    summary_table_limit: int = 200
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

    max_result_tokens: int = 1200
    max_rows_inline: int = 20
    max_cell_chars: int = 300
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
            preferred_strategy="full",
        )
    if budget == "compact":
        return SchemaPromptPolicy(
            max_inline_schema_tokens=2000,
            max_inline_tables=8,
            max_inline_columns=80,
            compact_table_limit=60,
            preferred_strategy="compact",
        )
    if budget == "retrieval":
        return SchemaPromptPolicy(
            max_inline_schema_tokens=1200,
            max_inline_tables=4,
            max_inline_columns=40,
            compact_table_limit=20,
            summary_table_limit=120,
            preferred_strategy="retrieval",
        )
    raise ValueError("budget must be 'auto', 'full', 'compact', or 'retrieval'")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
