"""Conversation and run-history reads."""

from __future__ import annotations

from . import Route, http_routes

TABLE = (
    Route("GET", "/conversations/{conversation_id}", "conversation_exists"),
    Route("GET", "/conversations/{conversation_id}/runs", "conversation_runs"),
    Route("DELETE", "/conversations", "clear_conversations"),
    Route("GET", "/runs/{run_id}/transcript", "transcript"),
    Route("GET", "/runs/{run_id}/tool-evidence", "tool_evidence"),
    Route("GET", "/runs/{run_id}/analysis-usage", "analysis_usage"),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
