"""
Keyword-based auto-categorization and importance inference for memories.

Zero LLM cost — pure regex pattern matching. Applied as defaults when the
agent doesn't explicitly specify category or importance.
"""

import re
from typing import Optional

# Category patterns: checked in priority order, first match wins.
_CATEGORY_RULES = [
    (re.compile(r"\b(rule|must|always|never|require|mandatory)\b", re.I), "rule"),
    (re.compile(r"\b(FK|foreign key|references)\b", re.I), "schema"),
    (re.compile(r"\b(column|table|index|constraint|primary key)\b", re.I), "schema"),
    (re.compile(r"\b(error|bug|issue|incident|outage)\b", re.I), "incident"),
    (re.compile(r"\b(decision|decided|agreed|chosen|opted)\b", re.I), "decision"),
    (re.compile(r"\b(TODO|action item|follow.up)\b", re.I), "action_item"),
]

# Importance rules: applied in order, each can raise or lower the base.
_IMPORTANCE_BUMPS = [
    # Critical keywords -> at least 0.9
    (re.compile(r"\b(critical|urgent|security|vulnerability)\b", re.I), "min", 0.9),
    # Production -> at least 0.8
    (re.compile(r"\b(production|prod)\b", re.I), "min", 0.8),
    # Rules -> bump by +0.1
    (re.compile(r"\b(rule|must|always|never|require)\b", re.I), "add", 0.1),
    # Staging -> at least 0.6
    (re.compile(r"\b(staging|stg)\b", re.I), "min", 0.6),
    # Dev/test -> cap at 0.4
    (re.compile(r"\bdev\b|\btest\b|\blocal\b", re.I), "max", 0.4),
]


def infer_category(content: str) -> Optional[str]:
    """Infer a category from content using keyword patterns.

    Returns the first matching category, or None if no pattern matches.
    """
    for pattern, category in _CATEGORY_RULES:
        if pattern.search(content):
            return category
    return None


def infer_importance(content: str, base: float = 0.5) -> float:
    """Adjust importance based on content keyword patterns.

    Args:
        content: Memory content to analyze
        base: Starting importance (typically 0.5, the default)

    Returns:
        Adjusted importance clamped to [0.0, 1.0]
    """
    result = base
    for pattern, op, value in _IMPORTANCE_BUMPS:
        if pattern.search(content):
            if op == "min":
                result = max(result, value)
            elif op == "max":
                result = min(result, value)
            elif op == "add":
                result += value
    return max(0.0, min(1.0, result))
