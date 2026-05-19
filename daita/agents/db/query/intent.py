"""Shared intent classifiers for deterministic query planning."""

from __future__ import annotations

import re

from .metadata import split_identifier

COUNT_ENTITY_TOKENS = {
    "event",
    "events",
    "execution",
    "executions",
    "item",
    "items",
    "operation",
    "operations",
    "record",
    "records",
    "row",
    "rows",
    "run",
    "runs",
    "transaction",
    "transactions",
}


def looks_like_count_intent(text: str) -> bool:
    lowered = str(text or "").lower()
    return bool(
        re.search(r"\b(count|how many|number of|row count|rows)\b", lowered)
        or re.search(r"\b(most|top|fewest|least)\b", lowered)
        or re.search(
            r"\btotal[_\s-]?"
            r"(events|executions|items|operations|records|rows|runs|transactions)\b",
            lowered,
        )
    )


def is_count_metric_name(name: str) -> bool:
    tokens = set(split_identifier(name))
    return bool(
        tokens & {"count", "number", "quantity"}
        or ("total" in tokens and bool(tokens & COUNT_ENTITY_TOKENS))
    )
