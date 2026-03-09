"""
Focus DSL — pre-filter data before it reaches the LLM to reduce token consumption.

Usage:
    from daita.core.focus import apply_focus

    # Simple column projection
    result = apply_focus(data, "SELECT name, price")

    # Row filter + projection + limit
    result = apply_focus(data, "price > 100 and status == 'active' | SELECT name, price | LIMIT 50")

    # Aggregation
    result = apply_focus(data, "region == 'EU' | GROUP BY category | SELECT category, SUM(revenue) AS total")
"""
from __future__ import annotations

from typing import Any, Optional, Union

from .ast import FocusQuery
from .parser import parse
from .registry import get_backend, register_backend
from ..exceptions import FocusDSLError

__all__ = ["apply_focus", "FocusQuery", "parse", "register_backend", "FocusDSLError"]


def apply_focus(data: Any, focus: Optional[Union[str, FocusQuery]]) -> Any:
    """
    Apply a Focus DSL expression to data, filtering it before the LLM sees it.

    Args:
        data:  Any supported data type (DataFrame, dict, list[dict], …)
        focus: A DSL string or a pre-parsed FocusQuery. Pass None to skip filtering.

    Returns:
        Filtered data of the same general type as the input.

    Raises:
        FocusDSLError: If the DSL string is malformed or an unsupported type is passed.
    """
    if focus is None or data is None:
        return data

    if isinstance(focus, str):
        query = parse(focus)
    elif isinstance(focus, FocusQuery):
        query = focus
    else:
        raise FocusDSLError(
            f"focus must be a DSL string or FocusQuery, got {type(focus).__name__}. "
            "Example: focus=\"price > 100 | SELECT name, price | LIMIT 50\""
        )

    backend = get_backend(data)
    return backend.apply(data, query)
