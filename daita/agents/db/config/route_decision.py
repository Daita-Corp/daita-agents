"""Canonical routing decisions for ``Agent.from_db()`` runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..utils import unique_preserving_order
from .intent import DbIntent, DbIntentKind
from .intent_classifier import DbPromptClassification, classify_db_prompt
from .policies import (
    ConditionalCapability,
    DbWorkflowPolicy,
    workflow_policy_for_intent,
)
from .tool_selection import (
    ANALYST_CAPABILITY_PREFIX,
    DB_AGGREGATE_READ_CAPABILITY,
    DB_COMPILE_AND_EXECUTE_CAPABILITY,
    DB_TOOL_CAPABILITIES,
    DB_ROW_READ_CAPABILITY,
    DB_SQL_EXECUTE_CAPABILITY,
    is_generic_catalog_tool,
    relational_catalog_alias,
    select_tools_for_capabilities,
    supported_required_phases,
    tool_has_any_capability,
)


@dataclass(frozen=True)
class DbFastPathDecision:
    """Deterministic fast-path route selected for a DB request."""

    eligible: bool
    strategy: str | None
    tool_name: str | None
    reason: str | None = None


@dataclass(frozen=True)
class DbRouteDecision:
    """The single DB route contract consumed by runtime layers."""

    classification: DbPromptClassification
    intent: DbIntent
    policy: DbWorkflowPolicy
    capabilities: tuple[str, ...]
    tools: tuple[str, ...]
    required_phases: tuple[str, ...]
    terminal_tools: tuple[str, ...]
    evidence_mode: str
    access_mode: str
    forbidden_capabilities: tuple[str, ...]
    allow_catalog_final: bool
    require_executed_query: bool
    max_model_turns: int
    max_tool_calls: int
    max_repair_attempts: int
    fast_path: DbFastPathDecision
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def build_db_route_decision(
    agent: Any,
    prompt: str,
    *,
    classification: DbPromptClassification | None = None,
) -> DbRouteDecision:
    """Build the canonical route decision for one ``from_db`` prompt."""

    available = _available_tool_names(agent)
    classification = classification or _classify_prompt_for_agent(agent, prompt)
    intent = classification.intent
    policy = workflow_policy_for_intent(intent)
    diagnostics: dict[str, Any] = {}
    forbidden_capabilities = _forbidden_capabilities(classification)

    if classification.strict_explicit_tool_request:
        capabilities: list[str] = []
        selected = unique_preserving_order(classification.explicit_tools)
        required_phases: tuple[str, ...] = ()
    else:
        capabilities = _capabilities_for_policy(agent, classification, policy)
        selected = select_tools_for_capabilities(available, capabilities)
        selected.extend(classification.explicit_tools)
        selected = unique_preserving_order(selected)
        required_phases = supported_required_phases(policy.required_phases, selected)

    if forbidden_capabilities:
        capabilities = [
            capability
            for capability in capabilities
            if capability not in forbidden_capabilities
        ]
        selected = [
            tool_name
            for tool_name in selected
            if not tool_has_any_capability(tool_name, forbidden_capabilities)
        ]
        diagnostics["forbidden_capabilities"] = list(forbidden_capabilities)

    tools = tuple(_canonical_tools(selected, available, prompt, diagnostics))
    terminal_tools = tuple(
        select_tools_for_capabilities(tools, policy.terminal_capabilities)
    )
    return DbRouteDecision(
        classification=classification,
        intent=intent,
        policy=policy,
        capabilities=tuple(unique_preserving_order(capabilities)),
        tools=tools,
        required_phases=tuple(required_phases),
        terminal_tools=terminal_tools,
        evidence_mode=policy.evidence_mode.value,
        access_mode=_access_mode_for_classification(classification),
        forbidden_capabilities=forbidden_capabilities,
        allow_catalog_final=policy.allow_catalog_final,
        require_executed_query=policy.require_executed_query,
        max_model_turns=policy.max_model_turns,
        max_tool_calls=policy.max_tool_calls,
        max_repair_attempts=policy.max_repair_attempts,
        fast_path=_fast_path_decision(classification, policy, tools),
        diagnostics=diagnostics,
    )


def classify_db_prompt_for_agent(agent: Any, prompt: str) -> DbPromptClassification:
    """Classify a prompt once with available-tool and catalog signals."""

    return _classify_prompt_for_agent(agent, prompt)


def _classify_prompt_for_agent(agent: Any, prompt: str) -> DbPromptClassification:
    from ..query.catalog_adapter import has_likely_catalog_match

    available = _available_tool_names(agent)
    classification = classify_db_prompt(
        prompt,
        available_tools=available,
        likely_catalog_match=False,
    )
    if classification.strict_explicit_tool_request:
        return classification

    likely_catalog_match = (
        has_likely_catalog_match(agent, str(prompt or "").lower())
        if classification.intent.needs_sql_execution
        else False
    )
    if not likely_catalog_match:
        return classification
    return classify_db_prompt(
        prompt,
        available_tools=available,
        likely_catalog_match=likely_catalog_match,
    )


def _capabilities_for_policy(
    agent: Any,
    classification: DbPromptClassification,
    policy: DbWorkflowPolicy,
) -> list[str]:
    if classification.compile_first and policy.fast_path_capabilities:
        capabilities = list(policy.fast_path_capabilities)
    else:
        capabilities = list(policy.required_capabilities)
    capabilities.extend(
        optional.capability
        for optional in policy.optional_capabilities
        if _optional_capability_applies(optional, agent, classification)
    )
    constraints = classification.access_constraints
    if constraints.allow_aggregate_only and not constraints.forbid_sql_execution:
        capabilities.append(DB_AGGREGATE_READ_CAPABILITY)
    if (
        constraints.explicit_sample_requested or constraints.explicit_rows_requested
    ) and not constraints.forbid_row_access:
        capabilities.append(DB_ROW_READ_CAPABILITY)
    return unique_preserving_order(capabilities)


def _available_tool_names(agent: Any) -> set[str]:
    registry = getattr(agent, "tool_registry", None)
    names = getattr(registry, "tool_names", None)
    if names is not None:
        return set(names)
    getter = getattr(registry, "get", None)
    if not callable(getter):
        return set()
    return {name for name in DB_TOOL_CAPABILITIES if getter(name) is not None}


def _optional_capability_applies(
    optional: ConditionalCapability,
    agent: Any,
    classification: DbPromptClassification,
) -> bool:
    when = optional.when
    if when == "needs_schema_search":
        return (
            classification.needs_schema_tools
            and classification.compile_first
            and not classification.likely_catalog_match
            and classification.intent.kind
            not in {
                DbIntentKind.SCHEMA_ONLY,
                DbIntentKind.DATA_QUERY_CATALOG_ASSISTED,
            }
        )
    if when == "needs_full_schema_navigation":
        return (
            classification.needs_schema_tools
            and classification.intent.kind
            not in {
                DbIntentKind.SCHEMA_ONLY,
                DbIntentKind.DATA_QUERY_CATALOG_ASSISTED,
            }
            and not (
                classification.compile_first and not classification.likely_catalog_match
            )
        )
    if when == "needs_validation":
        return classification.needs_validation_tool
    if when == "needs_quality":
        return classification.needs_quality_tools
    if when == "needs_write":
        return classification.needs_write_tools
    if when == "needs_lineage":
        return classification.needs_lineage_tools
    if when == "needs_memory_write":
        return (
            hasattr(agent, "_db_memory_semantics") and classification.needs_memory_tools
        )
    if when == "has_vector_columns":
        from ..query.catalog_adapter import catalog_schema_snapshot

        return _has_vector_columns(catalog_schema_snapshot(agent))
    if when == "requested_analyst_tool":
        if not optional.capability.startswith(ANALYST_CAPABILITY_PREFIX):
            return False
        tool_name = optional.capability[len(ANALYST_CAPABILITY_PREFIX) :]
        return tool_name in classification.requested_analyst_tools
    return False


def _forbidden_capabilities(
    classification: DbPromptClassification,
) -> tuple[str, ...]:
    constraints = classification.access_constraints
    forbidden: list[str] = []
    if constraints.forbid_row_access:
        forbidden.extend(
            [
                DB_ROW_READ_CAPABILITY,
                DB_SQL_EXECUTE_CAPABILITY,
                DB_COMPILE_AND_EXECUTE_CAPABILITY,
            ]
        )
    if constraints.forbid_sql_execution:
        forbidden.extend(
            [
                DB_SQL_EXECUTE_CAPABILITY,
                DB_COMPILE_AND_EXECUTE_CAPABILITY,
            ]
        )
    return tuple(unique_preserving_order(forbidden))


def _access_mode_for_classification(classification: DbPromptClassification) -> str:
    constraints = classification.access_constraints
    if constraints.schema_only_requested or constraints.forbid_sql_execution:
        return "schema_only"
    if classification.intent.is_write_or_admin:
        return "write"
    if constraints.allow_aggregate_only:
        return "aggregate_only"
    if constraints.forbid_row_access:
        return "no_row_access"
    if constraints.explicit_sample_requested or constraints.explicit_rows_requested:
        return "row_read"
    if classification.intent.needs_sql_execution:
        return "query_allowed"
    if classification.intent.kind == DbIntentKind.MEMORY_ONLY:
        return "memory"
    return "no_db_access"


def _has_vector_columns(schema: dict) -> bool:
    for table in schema.get("tables", []) or []:
        for column in table.get("columns", []) or []:
            col_type = str(column.get("type", "")).lower()
            if (
                "vector" in col_type
                or "embedding" in str(column.get("name", "")).lower()
            ):
                return True
    return False


def _canonical_tools(
    selected: list[str],
    available: set[str],
    prompt: str,
    diagnostics: dict[str, Any],
) -> list[str]:
    prompt_text = str(prompt or "").lower()
    canonical: list[str] = []
    for name in selected:
        if is_generic_catalog_tool(name) and name not in prompt_text:
            alias = relational_catalog_alias(name)
            if alias in available:
                canonical.append(alias)
            diagnostics.setdefault("hidden_generic_catalog_tools", []).append(name)
            continue
        canonical.append(name)
    return unique_preserving_order(name for name in canonical if name in available)


def _fast_path_decision(
    classification: DbPromptClassification,
    policy: DbWorkflowPolicy,
    tools: tuple[str, ...],
) -> DbFastPathDecision:
    if (
        classification.intent.kind != DbIntentKind.DATA_QUERY_SIMPLE
        or classification.query_shape != "simple_count"
        or classification.aggregation_kind != "count"
    ):
        return DbFastPathDecision(
            eligible=False,
            strategy=None,
            tool_name=None,
            reason="not_simple_count",
        )
    if DB_COMPILE_AND_EXECUTE_CAPABILITY not in policy.fast_path_capabilities:
        return DbFastPathDecision(
            eligible=False,
            strategy=None,
            tool_name=None,
            reason="policy_disallows_fast_path",
        )
    tool_name = select_tools_for_capabilities(
        tools, [DB_COMPILE_AND_EXECUTE_CAPABILITY]
    )
    if not tool_name:
        return DbFastPathDecision(
            eligible=False,
            strategy="simple_count",
            tool_name=None,
            reason="tool_unavailable",
        )
    return DbFastPathDecision(
        eligible=True,
        strategy="simple_count",
        tool_name=tool_name[0],
        reason=None,
    )
