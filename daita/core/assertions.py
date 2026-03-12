"""
ItemAssertion — item-level data quality enforcement.

Used with BaseDatabasePlugin.query_checked() and any other plugin method
that returns a list of structured items (dicts). The evaluation logic lives
here, in core, so any plugin can import and use it without inheritance.

Usage::

    from daita.core.assertions import ItemAssertion

    rows = await db.query_checked(
        "SELECT * FROM transactions",
        assertions=[
            ItemAssertion(lambda r: r["amount"] > 0, "All amounts must be positive"),
            ItemAssertion(lambda r: r["customer_id"] is not None, "Every transaction needs a customer"),
        ],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ItemAssertion:
    """
    A callable assertion applied to each item in a collection.

    Args:
        check: Callable that receives a dict item and returns True if the item
               passes the assertion, False if it violates it.
        description: Human-readable description of what this assertion enforces.
                     Used in error messages and violation reports.

    Example::

        ItemAssertion(lambda r: r["amount"] > 0, "All amounts must be positive")
        ItemAssertion(lambda r: r.get("email") is not None, "Email required")
    """

    check: Callable[[Dict[str, Any]], bool]
    description: str


# Alias for readability in SQL contexts — identical to ItemAssertion
RowAssertion = ItemAssertion


def _evaluate_assertions(
    items: List[Any],
    assertions: List[ItemAssertion],
    source: Optional[str] = None,
) -> None:
    """
    Evaluate all assertions against a list of items.

    Collects every violation before raising so the caller sees all failures
    at once rather than stopping at the first.

    Args:
        items: List of dict-like items to check.
        assertions: Assertions to evaluate against each item.
        source: Optional description of where the items came from (e.g. SQL
                query string), included in error context for debugging.

    Raises:
        DataQualityError: If any assertion has one or more violations.
    """
    from daita.core.exceptions import DataQualityError

    violations = []
    for assertion in assertions:
        try:
            failing = [item for item in items if not assertion.check(item)]
        except Exception as exc:
            logger.debug(
                "ItemAssertion %r raised during evaluation: %s — skipping",
                assertion.description,
                exc,
            )
            continue

        if failing:
            violations.append(
                {
                    "description": assertion.description,
                    "violation_count": len(failing),
                    "total_items": len(items),
                    "sample": failing[:3],
                }
            )

    if violations:
        summary = "; ".join(
            f"{v['description']} ({v['violation_count']}/{v['total_items']} items)"
            for v in violations
        )
        raise DataQualityError(
            f"Data quality violations: {summary}",
            violations=violations,
            context={"source": source[:200] if source else None},
        )
