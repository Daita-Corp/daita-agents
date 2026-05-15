"""Deterministic query planning primitives for ``Agent.from_db()``."""

from .compiler import CompiledQuery, compile_query_plan
from .intent import is_count_metric_name, looks_like_count_intent
from .ir_validator import validate_query_plan
from .ir import FieldRef, Filter, Join, Metric, OrderBy, QueryPlan
from .planner import build_query_plan
from .resolver import resolve_query_plan
from .sql_validator import validate_sql_against_schema

__all__ = [
    "CompiledQuery",
    "FieldRef",
    "Filter",
    "Join",
    "Metric",
    "OrderBy",
    "QueryPlan",
    "build_query_plan",
    "compile_query_plan",
    "is_count_metric_name",
    "looks_like_count_intent",
    "resolve_query_plan",
    "validate_query_plan",
    "validate_sql_against_schema",
]
