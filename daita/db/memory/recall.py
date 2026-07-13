"""Planning-time DB semantic-memory recall policy."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .contracts import meaningful_tokens

DB_MEMORY_SEMANTIC_QUERY_INTENTS = frozenset(
    {
        "data.query",
        "data.query.catalog_assisted",
        "metric.query",
        "report.generate",
        "quality.check",
        "anomaly.investigate",
    }
)
DB_MEMORY_METADATA_RECALL_INTENTS = frozenset(
    {
        "schema.query",
        "schema.relationship_query",
    }
)
FALLBACK_SEMANTIC_RECALL_TERMS = (
    "business rule",
    "business rules",
    "calculate",
    "calculation",
    "caveat",
    "caveats",
    "definition",
    "definitions",
    "exclude",
    "excludes",
    "include",
    "includes",
    "known issue",
    "known issues",
    "meaning",
    "metric",
    "metrics",
    "remember",
    "stored rule",
    "unit",
    "units",
    "what does",
    "you said",
)
FALLBACK_DIRECT_QUERY_TERMS = (
    "average",
    "avg",
    "count",
    "find",
    "group by",
    "how many",
    "list",
    "max",
    "minimum",
    "min",
    "show",
    "sum",
    "top",
    "total",
)


def db_memory_planning_recall_decision(
    *,
    prompt: str,
    intent_kind: str,
    schema: dict[str, Any],
    memory_config: dict[str, Any],
    matched_schema_terms: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return whether planning should recall DB semantic memory."""
    if not bool(memory_config.get("enabled", False)):
        return {"recall": False, "reason": "memory_disabled"}
    if memory_config.get("recall") == "off":
        return {"recall": False, "reason": "recall_disabled"}
    if int(memory_config.get("limit") or 0) <= 0:
        return {"recall": False, "reason": "limit_zero"}
    if int(memory_config.get("char_budget") or 0) <= 0:
        return {"recall": False, "reason": "char_budget_zero"}
    if intent_kind not in (
        DB_MEMORY_SEMANTIC_QUERY_INTENTS | DB_MEMORY_METADATA_RECALL_INTENTS
    ):
        return {"recall": False, "reason": "intent_not_semantic_query"}
    if _looks_row_level(prompt):
        return {"recall": False, "reason": "row_level_or_pii_prompt"}

    text = str(prompt or "").lower()
    semantic_matches = _matched_terms(text, FALLBACK_SEMANTIC_RECALL_TERMS)
    if semantic_matches:
        return {
            "recall": True,
            "reason": "semantic_prompt",
            "matched_terms": semantic_matches,
            "query": db_memory_planning_recall_query(
                prompt,
                schema,
                intent_kind,
                matched_schema_terms=matched_schema_terms,
            ),
        }
    if _looks_direct_query(text) and _prompt_matches_schema(prompt, schema):
        return {"recall": False, "reason": "direct_schema_matched_query"}
    return {
        "recall": True,
        "reason": "semantic_fallback",
        "matched_terms": [],
        "query": db_memory_planning_recall_query(
            prompt,
            schema,
            intent_kind,
            matched_schema_terms=matched_schema_terms,
        ),
    }


def db_memory_planning_recall_query(
    prompt: str,
    schema: dict[str, Any],
    intent_kind: str,
    *,
    matched_schema_terms: Iterable[str] | None = None,
) -> str:
    """Build the bounded text used for planning-time memory recall."""
    schema_terms = _bounded_recall_terms(
        (
            matched_schema_terms
            if matched_schema_terms is not None
            else _matched_schema_terms_for_recall(prompt, schema)
        ),
        limit=24,
    )
    recall_terms = _bounded_recall_terms(
        _recall_terms_for_prompt(prompt, schema_terms, intent_kind),
        limit=32,
    )
    lines = [str(prompt or "").strip(), f"Intent: {intent_kind}"]
    if schema_terms:
        lines.append(f"Matched schema terms: {' '.join(schema_terms)}")
    if recall_terms:
        lines.append(f"Recall terms: {' '.join(recall_terms)}")
    return "\n".join(line for line in lines if line).strip()


def _matched_schema_terms_for_recall(
    prompt: str,
    schema: dict[str, Any],
) -> tuple[str, ...]:
    """Return schema terms that are actually mentioned by the prompt."""
    prompt_terms = set(meaningful_tokens(prompt))
    if not prompt_terms:
        return ()
    matches: list[str] = []
    column_tables: dict[str, list[str]] = {}
    for table in schema.get("tables", []) or []:
        table_name = str(table.get("name") or "").strip()
        if not table_name:
            continue
        if _identifier_matches_terms(table_name, prompt_terms):
            matches.append(table_name)
        for column in table.get("columns", []) or []:
            column_name = str(column.get("name") or "").strip()
            if column_name:
                column_tables.setdefault(_identifier_key(column_name), []).append(
                    table_name
                )

    for table in schema.get("tables", []) or []:
        table_name = str(table.get("name") or "").strip()
        if not table_name:
            continue
        table_matched = table_name in matches
        for column in table.get("columns", []) or []:
            column_name = str(column.get("name") or "").strip()
            if not column_name or not _identifier_matches_terms(
                column_name, prompt_terms
            ):
                continue
            owners = column_tables.get(_identifier_key(column_name), [])
            if table_matched or len(set(owners)) == 1:
                matches.append(f"{table_name}.{column_name}")
            else:
                matches.append(column_name)
    return tuple(_bounded_recall_terms(matches, limit=24))


def _recall_terms_for_prompt(
    prompt: str,
    schema_terms: Iterable[str],
    intent_kind: str,
) -> tuple[str, ...]:
    terms: list[str] = []
    prompt_terms = set(meaningful_tokens(prompt))
    for term in schema_terms:
        cleaned = str(term or "").strip()
        if not cleaned:
            continue
        terms.append(cleaned)
        if "." not in cleaned and "table" in prompt_terms:
            terms.append(f"{cleaned} table")
    if intent_kind == "schema.relationship_query":
        terms.append("relationship")
    return tuple(terms)


def _bounded_recall_terms(
    terms: Iterable[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    bounded: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        term = re.sub(r"\s+", " ", str(raw or "").strip())
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        bounded.append(term)
        if len(bounded) >= max(0, int(limit)):
            break
    return tuple(bounded)


def _identifier_matches_terms(identifier: str, prompt_terms: set[str]) -> bool:
    identifier_terms = set(meaningful_tokens(identifier))
    if identifier_terms & prompt_terms:
        return True
    return bool(_singular_terms(identifier_terms) & _singular_terms(prompt_terms))


def _identifier_key(identifier: str) -> str:
    return " ".join(meaningful_tokens(identifier))


def _singular_terms(terms: Iterable[str]) -> set[str]:
    singular: set[str] = set()
    for term in terms:
        text = str(term)
        singular.add(text)
        if len(text) > 3 and text.endswith("ies"):
            singular.add(f"{text[:-3]}y")
        elif len(text) > 3 and text.endswith("s"):
            singular.add(text[:-1])
    return singular


def _looks_row_level(prompt: str) -> bool:
    """Return True for prompts that ask for row/entity-level values."""
    text = (prompt or "").lower()
    row_value_terms = (
        "email",
        "phone",
        "address",
        "ssn",
        "social security",
        "credit card",
        "customer_id",
        "order_id",
        "user_id",
        "account_id",
    )
    row_action_terms = (
        "show",
        "list",
        "lookup",
        "look up",
        "find",
        "give me",
        "what is",
        "who is",
    )
    row_entity_terms = (" row", " record", " customer", " order", " user", " account")

    if any(term in text for term in row_value_terms):
        return True
    return any(action in text for action in row_action_terms) and any(
        entity in text for entity in row_entity_terms
    )


def _looks_direct_query(text: str) -> bool:
    return _looks_count_intent(text) or bool(
        _matched_terms(text, FALLBACK_DIRECT_QUERY_TERMS)
    )


def _looks_count_intent(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in ("count", "how many", "number of", "total"))


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def _prompt_matches_schema(prompt: str, schema: dict[str, Any]) -> bool:
    prompt_tokens = set(meaningful_tokens(prompt))
    if not prompt_tokens:
        return False
    for table in schema.get("tables", []) or []:
        names = [table.get("name")]
        names.extend(column.get("name") for column in table.get("columns", []) or [])
        if prompt_tokens & set(meaningful_tokens(" ".join(str(n) for n in names if n))):
            return True
    return False
