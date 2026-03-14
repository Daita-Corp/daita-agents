"""
Universal Python evaluator — the fallback for any FocusQuery clause not handled
natively by a backend. Works on any data that can be coerced to a list of dicts.

Filter evaluation walks the Python AST directly; no eval() is used.
"""

from __future__ import annotations

import ast as pyast
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from .ast import FocusQuery, AGG_EXPR_RE as _AGG_EXPR_RE

# ── Public entry point ────────────────────────────────────────────────────────


def evaluate_remaining(data: Any, query: FocusQuery, applied: Set[str]) -> Any:
    """Apply any FocusQuery clauses not already handled natively by a backend."""
    original = data
    rows = _to_rows(data)

    if "filter" not in applied and query.filter_ast is not None:
        rows = [r for r in rows if _eval_filter(r, query.filter_ast)]

    if "group_by" not in applied and query.group_by:
        rows = _apply_group_by(rows, query)
        # group_by absorbs order/limit/select internally
        return _from_rows(rows, original)

    if "order_by" not in applied and query.order_by:
        reverse = query.order_dir == "DESC"
        rows = sorted(
            rows,
            key=lambda r: (
                _get_field(r, query.order_by) is None,
                _get_field(r, query.order_by),
            ),
            reverse=reverse,
        )

    if "limit" not in applied and query.limit is not None:
        rows = rows[: query.limit]

    if "select" not in applied and (query.select or query.aggregates):
        # If every named SELECT column is absent from every row (e.g. the LLM
        # wrote an aggregating query that dropped those columns), skip projection
        # rather than returning rows full of None values.
        select_cols = query.select or []
        if select_cols and rows:
            entirely_absent = all(
                all(_get_field(r, c) is None for r in rows) for c in select_cols
            )
        else:
            entirely_absent = False

        if not entirely_absent:
            rows = [
                _project_row(r, select_cols, query.aggregates or {}) for r in rows
            ]

    return _from_rows(rows, original)


# ── Row coercion ──────────────────────────────────────────────────────────────


def _to_rows(data: Any) -> List[Dict]:
    if isinstance(data, list):
        return [r if isinstance(r, dict) else {"_value": r} for r in data]
    if isinstance(data, dict):
        return [data]
    if hasattr(data, "to_dict"):  # pandas DataFrame
        return data.to_dict(orient="records")
    return [{"_value": data}]


def _from_rows(rows: List[Dict], original: Any) -> Any:
    if hasattr(original, "columns"):  # pandas DataFrame
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            pass
    if isinstance(original, dict):
        return rows[0] if rows else {}
    if isinstance(original, list):
        return rows
    return rows


# ── Field resolution ──────────────────────────────────────────────────────────


def _get_field(row: Dict, field: str) -> Any:
    """Resolve a field name. Supports dot notation for nested dicts."""
    if field in row:
        return row[field]
    # Dot notation traversal
    current = row
    for part in field.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


# ── Filter evaluation (AST walk — no eval) ───────────────────────────────────


def _eval_filter(row: Dict, node: pyast.expr) -> bool:
    try:
        return bool(_eval_node(row, node))
    except Exception:
        return False


def _eval_node(row: Dict, node: pyast.expr) -> Any:
    if isinstance(node, pyast.Constant):
        return node.value

    if isinstance(node, pyast.Name):
        return _get_field(row, node.id)

    if isinstance(node, pyast.Attribute):
        parts = _flatten_attr(node)
        return _get_field(row, ".".join(parts)) if parts else None

    if isinstance(node, pyast.Compare):
        left = _eval_node(row, node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(row, comparator)
            if not _apply_cmp(op, left, right):
                return False
        return True

    if isinstance(node, pyast.BoolOp):
        values = [_eval_node(row, v) for v in node.values]
        return all(values) if isinstance(node.op, pyast.And) else any(values)

    if isinstance(node, pyast.UnaryOp) and isinstance(node.op, pyast.Not):
        return not _eval_node(row, node.operand)

    if isinstance(node, (pyast.List, pyast.Tuple)):
        return [_eval_node(row, elt) for elt in node.elts]

    return None


def _flatten_attr(node: pyast.Attribute) -> Optional[List[str]]:
    """Flatten nested Attribute nodes into a list of name parts."""
    parts: List[str] = []
    current = node
    while isinstance(current, pyast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, pyast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return None


def _coerce_for_cmp(left: Any, right: Any):
    """
    Try to make left and right comparable when their types differ.

    Handles the common case of datetime/date values compared against ISO
    date strings (e.g. ``created_at >= '2023-01-01'``), which databases
    accept implicitly but Python does not.
    """
    from datetime import date, datetime

    if isinstance(left, datetime) and isinstance(right, str):
        return left, datetime.fromisoformat(right)
    if isinstance(right, datetime) and isinstance(left, str):
        return datetime.fromisoformat(left), right
    if isinstance(left, date) and not isinstance(left, datetime) and isinstance(right, str):
        return left, date.fromisoformat(right)
    if isinstance(right, date) and not isinstance(right, datetime) and isinstance(left, str):
        return date.fromisoformat(left), right
    # Generic fallback: try to cast right to left's type (e.g. int/float mixing)
    if isinstance(right, str):
        return left, type(left)(right)
    if isinstance(left, str):
        return type(right)(left), right
    return left, right


def _apply_cmp(op: pyast.cmpop, left: Any, right: Any) -> bool:
    try:
        if isinstance(op, pyast.Eq):
            return left == right
        if isinstance(op, pyast.NotEq):
            return left != right
        if isinstance(op, pyast.Lt):
            return left < right
        if isinstance(op, pyast.LtE):
            return left <= right
        if isinstance(op, pyast.Gt):
            return left > right
        if isinstance(op, pyast.GtE):
            return left >= right
        if isinstance(op, pyast.In):
            return left in (right or [])
        if isinstance(op, pyast.NotIn):
            return left not in (right or [])
    except TypeError:
        try:
            left, right = _coerce_for_cmp(left, right)
            return _apply_cmp(op, left, right)
        except (TypeError, ValueError):
            return False
    return False


# ── Projection ────────────────────────────────────────────────────────────────


def _project_row(row: Dict, fields: List[str], aggregates: Dict[str, str]) -> Dict:
    result = {}
    for field in fields:
        val = _get_field(row, field)
        out_key = field.split(".")[-1] if "." in field else field
        result[out_key] = val
    # Aggregates on individual rows — preserve any pre-computed alias values
    for alias in aggregates:
        if alias in row:
            result[alias] = row[alias]
    return result


# ── Group by + aggregates ─────────────────────────────────────────────────────

_AGG_FUNCS = {
    "SUM": lambda vals: sum(v for v in vals if v is not None),
    "COUNT": lambda vals: sum(1 for v in vals if v is not None),
    "AVG": lambda vals: (
        sum(v for v in vals if v is not None)
        / max(sum(1 for v in vals if v is not None), 1)
    ),
    "MIN": lambda vals: min((v for v in vals if v is not None), default=None),
    "MAX": lambda vals: max((v for v in vals if v is not None), default=None),
}


def _parse_agg_expr(expr: str):
    m = _AGG_EXPR_RE.match(expr.strip())
    if not m:
        raise ValueError(f"Unknown aggregate expression: '{expr}'")
    return m.group(1).upper(), m.group(2)


def _apply_group_by(rows: List[Dict], query: FocusQuery) -> List[Dict]:
    groups: Dict = defaultdict(list)
    for row in rows:
        key = tuple(_get_field(row, f) for f in query.group_by)
        groups[key].append(row)

    result = []
    for key, group_rows in groups.items():
        out = {field: val for field, val in zip(query.group_by, key)}

        if query.aggregates:
            for alias, expr in query.aggregates.items():
                func_name, field = _parse_agg_expr(expr)
                func = _AGG_FUNCS[func_name]
                vals = (
                    list(range(len(group_rows)))
                    if field == "*"
                    else [_get_field(r, field) for r in group_rows]
                )
                out[alias] = func(vals)

        result.append(out)

    if query.order_by:
        reverse = query.order_dir == "DESC"
        result = sorted(
            result,
            key=lambda r: (
                _get_field(r, query.order_by) is None,
                _get_field(r, query.order_by),
            ),
            reverse=reverse,
        )

    if query.limit is not None:
        result = result[: query.limit]

    return result
