"""
FocusQuery — the intermediate representation produced by the DSL parser.
"""
from __future__ import annotations

import ast as pyast
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FocusQuery:
    """Parsed representation of a Focus DSL expression."""

    # Filter: raw string + parsed Python AST node
    filter_expr: Optional[str] = None
    filter_ast: Optional[pyast.expr] = None

    # Column projection
    select: Optional[List[str]] = None

    # Sorting
    order_by: Optional[str] = None
    order_dir: str = "ASC"

    # Row limit
    limit: Optional[int] = None

    # Aggregation
    group_by: Optional[List[str]] = None
    # Maps output alias → aggregate expression, e.g. {"total": "SUM(revenue)"}
    aggregates: Optional[Dict[str, str]] = None
