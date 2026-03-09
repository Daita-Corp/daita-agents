"""
DSL parser: Focus string → FocusQuery.

Syntax (pipe-delimited, all clauses optional):

    <filter_expr> | SELECT field1, SUM(x) AS alias | ORDER BY field DESC | LIMIT n | GROUP BY field1, field2

The filter clause is a Python boolean expression evaluated safely against each row.
"""
from __future__ import annotations

import ast as pyast
import re
from typing import Dict, Tuple

from .ast import FocusQuery
from ..exceptions import FocusDSLError

# Matches aggregate expressions: SUM(revenue) AS total  or  COUNT(*) AS cnt
_AGG_RE = re.compile(
    r"^(SUM|COUNT|AVG|MIN|MAX)\((\*|[\w.]+)\)(?:\s+AS\s+(\w+))?$",
    re.IGNORECASE,
)

# Clause keyword prefixes (upper-cased for matching)
_SELECT_PREFIX   = "SELECT"
_ORDER_PREFIX    = "ORDER BY"
_LIMIT_PREFIX    = "LIMIT"
_GROUP_PREFIX    = "GROUP BY"


def parse(dsl: str) -> FocusQuery:
    """Parse a Focus DSL string into a FocusQuery."""
    if not dsl or not dsl.strip():
        raise FocusDSLError("Focus DSL string cannot be empty")

    query = FocusQuery()
    clauses = [c.strip() for c in dsl.split("|") if c.strip()]

    for clause in clauses:
        upper = clause.upper()
        if upper.startswith(_SELECT_PREFIX):
            _parse_select(clause, query)
        elif upper.startswith(_ORDER_PREFIX):
            _parse_order_by(clause, query)
        elif upper.startswith(_LIMIT_PREFIX):
            _parse_limit(clause, query)
        elif upper.startswith(_GROUP_PREFIX):
            _parse_group_by(clause, query)
        else:
            _parse_filter(clause, query)

    return query


# ── Clause parsers ────────────────────────────────────────────────────────────

def _parse_select(clause: str, query: FocusQuery) -> None:
    content = re.sub(r"^SELECT\s+", "", clause.strip(), flags=re.IGNORECASE).strip()
    if not content:
        raise FocusDSLError("SELECT clause cannot be empty")

    fields: list = []
    aggregates: Dict[str, str] = {}

    for part in content.split(","):
        part = part.strip()
        if not part:
            continue
        m = _AGG_RE.match(part)
        if m:
            func, field, alias = m.group(1).upper(), m.group(2), m.group(3)
            alias = alias or f"{func.lower()}_{field.replace('*', 'all')}"
            aggregates[alias] = f"{func}({field})"
        else:
            fields.append(part)

    if fields:
        query.select = fields
    if aggregates:
        query.aggregates = {**(query.aggregates or {}), **aggregates}


def _parse_order_by(clause: str, query: FocusQuery) -> None:
    content = re.sub(r"^ORDER\s+BY\s+", "", clause.strip(), flags=re.IGNORECASE).strip()
    if not content:
        raise FocusDSLError("ORDER BY clause cannot be empty")
    parts = content.split()
    query.order_by = parts[0]
    if len(parts) > 1:
        direction = parts[1].upper()
        if direction not in ("ASC", "DESC"):
            raise FocusDSLError(f"ORDER BY direction must be ASC or DESC, got '{parts[1]}'")
        query.order_dir = direction


def _parse_limit(clause: str, query: FocusQuery) -> None:
    parts = clause.strip().split()
    if len(parts) < 2:
        raise FocusDSLError(f"LIMIT clause requires an integer value")
    try:
        n = int(parts[1])
    except ValueError:
        raise FocusDSLError(f"LIMIT value must be an integer, got '{parts[1]}'")
    if n < 0:
        raise FocusDSLError(f"LIMIT value must be non-negative, got {n}")
    query.limit = n


def _parse_group_by(clause: str, query: FocusQuery) -> None:
    content = re.sub(r"^GROUP\s+BY\s+", "", clause.strip(), flags=re.IGNORECASE).strip()
    if not content:
        raise FocusDSLError("GROUP BY clause cannot be empty")
    query.group_by = [f.strip() for f in content.split(",") if f.strip()]


def _parse_filter(clause: str, query: FocusQuery) -> None:
    if query.filter_expr is not None:
        raise FocusDSLError(
            f"Multiple filter clauses found: '{query.filter_expr}' and '{clause}'. "
            "Combine them with 'and'/'or'."
        )
    try:
        tree = pyast.parse(clause, mode="eval")
    except SyntaxError as e:
        raise FocusDSLError(
            f"Invalid filter expression '{clause}': {e.msg}"
        ) from e
    query.filter_expr = clause
    query.filter_ast = tree.body
