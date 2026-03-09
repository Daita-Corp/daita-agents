"""
Pandas DataFrame backend.

Native support for: filter (df.query), select (column projection),
order_by (sort_values), limit (head), group_by + aggregates (groupby.agg).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Set, Tuple

from .base import FocusBackend
from ..ast import FocusQuery

logger = logging.getLogger(__name__)

_FUNC_MAP = {"SUM": "sum", "COUNT": "count", "AVG": "mean", "MIN": "min", "MAX": "max"}
_AGG_RE = re.compile(r"^(SUM|COUNT|AVG|MIN|MAX)\((\*|[\w.]+)\)$", re.IGNORECASE)


class PandasBackend(FocusBackend):

    def supports(self, data: Any) -> bool:
        return hasattr(data, "columns") and hasattr(data, "iloc")

    def _native_apply(self, data: Any, query: FocusQuery) -> Tuple[Any, Set[str]]:
        applied: Set[str] = set()
        df = data

        # Filter via df.query() — falls back to evaluator on failure
        if query.filter_expr:
            try:
                df = df.query(query.filter_expr)
                applied.add("filter")
            except Exception as e:
                logger.debug(f"pandas df.query() failed, using evaluator fallback: {e}")

        # Group by + aggregates — handles order/limit/select internally
        if query.group_by and query.aggregates:
            try:
                df = _native_groupby(df, query)
                applied.update({"group_by", "aggregates", "select", "order_by", "limit"})
                return df, applied
            except Exception as e:
                logger.debug(f"pandas groupby failed, using evaluator fallback: {e}")

        # Column projection
        if query.select:
            available = [c for c in query.select if c in df.columns]
            if available:
                df = df[available]
                applied.add("select")

        # Sorting
        if query.order_by and query.order_by in df.columns:
            df = df.sort_values(query.order_by, ascending=(query.order_dir == "ASC"))
            applied.add("order_by")

        # Limit
        if query.limit is not None:
            df = df.head(query.limit)
            applied.add("limit")

        return df, applied


def _native_groupby(df: Any, query: FocusQuery) -> Any:
    import pandas as pd

    agg_spec: dict = {}
    count_star_alias: str | None = None

    for alias, expr in query.aggregates.items():
        m = _AGG_RE.match(expr.strip())
        if not m:
            raise ValueError(f"Cannot parse aggregate: {expr}")
        func, field = m.group(1).upper(), m.group(2)

        if field == "*":
            count_star_alias = alias
        else:
            agg_spec[alias] = pd.NamedAgg(column=field, aggfunc=_FUNC_MAP[func])

    grouped = df.groupby(query.group_by, as_index=False)

    if agg_spec:
        result = grouped.agg(**agg_spec)
    else:
        # COUNT(*) only
        result = df.groupby(query.group_by).size().reset_index(name=count_star_alias or "count")
        count_star_alias = None  # already set

    if count_star_alias:
        result[count_star_alias] = df.groupby(query.group_by).size().values

    if query.order_by and query.order_by in result.columns:
        result = result.sort_values(query.order_by, ascending=(query.order_dir == "ASC"))

    if query.limit is not None:
        result = result.head(query.limit)

    return result
