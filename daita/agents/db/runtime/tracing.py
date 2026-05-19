"""Tracing helpers for ``from_db`` runtime phases."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from ....core.tracing import TraceType, get_trace_manager


@asynccontextmanager
async def db_trace_span(
    owner: Any,
    operation_name: str,
    *,
    trace_type: TraceType = TraceType.AGENT_EXECUTION,
    **metadata: Any,
):
    """Create a small DB-specific child span using the existing trace manager."""

    trace_manager = getattr(owner, "trace_manager", None) or get_trace_manager()
    agent_id = getattr(owner, "agent_id", None)
    async with trace_manager.span(
        operation_name=operation_name,
        trace_type=trace_type,
        agent_id=agent_id,
        **metadata,
    ) as span_id:
        yield trace_manager, span_id
