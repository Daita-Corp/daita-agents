"""
SQL focus compiler — translates a FocusQuery into SQL clauses pushed into a
subquery wrapper, avoiding full-table fetches before Python-side filtering.

Public API
----------
compile_focus_to_sql(base_query, fq, dialect, param_offset)
    -> (modified_sql, extra_params, applied_clauses)

Supported dialects: "postgresql", "mysql", "snowflake", "sqlite"
"""

from __future__ import annotations

import ast as pyast
import re
from typing import Any, Optional

from ..ast import FocusQuery

# ── Dialect helpers ────────────────────────────────────────────────────────────

_QUOTE = {
    "postgresql": '"',
    "snowflake": '"',
    "sqlite": '"',
    "mysql": "`",
    "standard": '"',
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_AGG = {"SUM", "COUNT", "AVG", "MIN", "MAX"}
_AGG_EXPR_RE = re.compile(r"^(SUM|COUNT|AVG|MIN|MAX)\((\*|[\w]+)\)$", re.IGNORECASE)


def _quote_id(name: str, dialect: str) -> str:
    q = _QUOTE.get(dialect, '"')
    return f"{q}{name}{q}"


# ── Filter AST → SQL compiler ──────────────────────────────────────────────────


class _SQLFilterCompiler:
    """
    Walks a Python AST filter node and emits a SQL fragment.

    Returns None from compile() if any part of the tree is untranslatable,
    signalling the caller to fall back to Python-side evaluation for the filter.

    Params are collected in self.params; call _placeholder() to get the next
    positional marker in the correct dialect format.
    """

    def __init__(self, dialect: str, param_offset: int):
        self.dialect = dialect
        self.params: list[Any] = []
        self._offset = param_offset  # number of pre-existing params in the base query

    def _placeholder(self) -> str:
        # MySQL and Snowflake use positional %s; PG and SQLite use numbered markers
        if self.dialect in ("mysql", "snowflake"):
            return "%s"
        idx = self._offset + len(self.params) + 1
        return "?" if self.dialect == "sqlite" else f"${idx}"

    def compile(self, node: pyast.expr) -> Optional[str]:
        """Return SQL string for node, or None if untranslatable."""
        if isinstance(node, pyast.Constant):
            return self._const(node)

        if isinstance(node, pyast.Name):
            if not _IDENTIFIER_RE.match(node.id):
                return None
            return _quote_id(node.id, self.dialect)

        if isinstance(node, pyast.Attribute):
            # dot-notation (nested JSON) — not universally portable, skip pushdown
            return None

        if isinstance(node, pyast.Compare):
            return self._compare(node)

        if isinstance(node, pyast.BoolOp):
            return self._boolop(node)

        if isinstance(node, pyast.UnaryOp) and isinstance(node.op, pyast.Not):
            inner = self.compile(node.operand)
            return f"NOT ({inner})" if inner is not None else None

        if isinstance(node, (pyast.List, pyast.Tuple)):
            parts = []
            for elt in node.elts:
                s = self.compile(elt)
                if s is None:
                    return None
                parts.append(s)
            return f"({', '.join(parts)})"

        return None  # unknown node type

    def _const(self, node: pyast.Constant) -> str:
        if node.value is None:
            return "NULL"
        if isinstance(node.value, bool):
            return "TRUE" if node.value else "FALSE"
        ph = self._placeholder()
        self.params.append(node.value)
        return ph

    def _compare(self, node: pyast.Compare) -> Optional[str]:
        left = self.compile(node.left)
        if left is None:
            return None

        _OPS = {
            pyast.Eq: "=",
            pyast.NotEq: "!=",
            pyast.Lt: "<",
            pyast.LtE: "<=",
            pyast.Gt: ">",
            pyast.GtE: ">=",
        }

        parts = []
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type in _OPS:
                right = self.compile(comparator)
                if right is None:
                    return None
                # SQL requires IS NULL / IS NOT NULL — "col = NULL" is always FALSE
                if right == "NULL":
                    if op_type is pyast.Eq:
                        parts.append(f"{left} IS NULL")
                    elif op_type is pyast.NotEq:
                        parts.append(f"{left} IS NOT NULL")
                    else:
                        return None  # <, <=, >, >= against NULL is meaningless
                    continue
                parts.append(f"{left} {_OPS[op_type]} {right}")
            elif op_type is pyast.In:
                sql = self._in_clause(left, comparator, negated=False)
                if sql is None:
                    return None
                parts.append(sql)
            elif op_type is pyast.NotIn:
                sql = self._in_clause(left, comparator, negated=True)
                if sql is None:
                    return None
                parts.append(sql)
            else:
                return None  # unsupported operator (Is, IsNot, etc.)

        return " AND ".join(parts)

    def _in_clause(
        self, left: str, comparator: pyast.expr, negated: bool
    ) -> Optional[str]:
        if not isinstance(comparator, (pyast.List, pyast.Tuple)):
            return None
        placeholders = []
        for elt in comparator.elts:
            s = self.compile(elt)
            if s is None:
                return None
            placeholders.append(s)
        keyword = "NOT IN" if negated else "IN"
        return f"{left} {keyword} ({', '.join(placeholders)})"

    def _boolop(self, node: pyast.BoolOp) -> Optional[str]:
        op = "AND" if isinstance(node.op, pyast.And) else "OR"
        parts = []
        for v in node.values:
            s = self.compile(v)
            if s is None:
                return None
            parts.append(f"({s})")
        return f" {op} ".join(parts)


# ── Public entry point ─────────────────────────────────────────────────────────


def compile_focus_to_sql(
    base_query: str,
    fq: FocusQuery,
    dialect: str = "postgresql",
    param_offset: int = 0,
) -> tuple[str, list, set[str]]:
    """
    Wrap *base_query* in a subquery and inject as many FocusQuery clauses as
    possible as native SQL.

    Returns
    -------
    modified_sql : str
        The rewritten query ready to execute.
    extra_params : list
        Parameter values to append after any existing query params.
    applied : set[str]
        Clause names fully handled in SQL.  Pass to evaluate_remaining() so
        the Python fallback skips them.
    """
    compiler = _SQLFilterCompiler(dialect, param_offset)
    applied: set[str] = set()

    # ── WHERE (filter) ────────────────────────────────────────────────────────
    where_sql: Optional[str] = None
    if fq.filter_ast is not None:
        where_sql = compiler.compile(fq.filter_ast)
        if where_sql is None:
            # Untranslatable — reset any params already emitted by the failed attempt
            compiler.params.clear()
        else:
            applied.add("filter")

    # ── GROUP BY + aggregates ─────────────────────────────────────────────────
    group_sql: Optional[str] = None
    agg_select_parts: list[str] = []

    if fq.group_by:
        group_cols = fq.group_by
        # Validate all group-by column names are safe identifiers
        if all(_IDENTIFIER_RE.match(c) for c in group_cols):
            group_sql = ", ".join(_quote_id(c, dialect) for c in group_cols)
            # Build aggregate expressions for the outer SELECT
            if fq.aggregates:
                ok = True
                for alias, expr in fq.aggregates.items():
                    m = _AGG_EXPR_RE.match(expr.strip())
                    if not m or m.group(1).upper() not in _SAFE_AGG:
                        ok = False
                        break
                    func = m.group(1).upper()
                    col = m.group(2)
                    col_sql = "*" if col == "*" else _quote_id(col, dialect)
                    agg_select_parts.append(
                        f"{func}({col_sql}) AS {_quote_id(alias, dialect)}"
                    )
                if ok:
                    applied.update({"group_by", "aggregates"})
                else:
                    group_sql = None
                    agg_select_parts.clear()

    # ── SELECT (column projection) ────────────────────────────────────────────
    select_parts: list[str] = []
    if fq.select and not group_sql:
        # Plain projection — only if all names are safe identifiers
        if all(_IDENTIFIER_RE.match(c) for c in fq.select):
            select_parts = [_quote_id(c, dialect) for c in fq.select]
            applied.add("select")

    # ── ORDER BY ──────────────────────────────────────────────────────────────
    order_sql: Optional[str] = None
    if fq.order_by and _IDENTIFIER_RE.match(fq.order_by):
        order_sql = f"{_quote_id(fq.order_by, dialect)} {fq.order_dir}"
        applied.add("order_by")

    # ── LIMIT ─────────────────────────────────────────────────────────────────
    limit_sql: Optional[str] = None
    if fq.limit is not None:
        limit_sql = str(fq.limit)
        applied.add("limit")

    # ── Assemble outer SELECT list ────────────────────────────────────────────
    if group_sql:
        outer_select = ", ".join(
            [_quote_id(c, dialect) for c in fq.group_by] + agg_select_parts
        )
    elif select_parts:
        outer_select = ", ".join(select_parts)
    else:
        outer_select = "*"

    # ── Build final query ─────────────────────────────────────────────────────
    sql = f"SELECT {outer_select} FROM ({base_query}) _focus_q"

    if where_sql:
        sql += f" WHERE {where_sql}"
    if group_sql:
        sql += f" GROUP BY {group_sql}"
    if order_sql:
        sql += f" ORDER BY {order_sql}"
    if limit_sql:
        sql += f" LIMIT {limit_sql}"

    return sql, compiler.params, applied
