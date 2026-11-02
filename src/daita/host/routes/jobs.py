"""Background jobs: start, list, inspect, answer and cancel."""

from __future__ import annotations

from . import Route, http_routes

TABLE = (
    Route("POST", "/jobs", "start_job", work=True),
    Route("GET", "/jobs", "list_jobs"),
    Route("GET", "/jobs/{job_id}", "inspect_job"),
    Route("POST", "/jobs/{job_id}/answer", "answer_job", work=True),
    Route("POST", "/jobs/{job_id}/cancel", "cancel_job", work=True),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
