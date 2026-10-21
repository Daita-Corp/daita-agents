"""Graph jobs, their tasks and the human controls."""

from __future__ import annotations

from . import Route, http_routes

_TASK = "/jobs/{job_id}/tasks/{task_id}"
_CONTROL = _TASK + "/controls/{control_id}"
TABLE = (
    Route("GET", "/jobs", "list_jobs"),
    Route("GET", "/jobs/{job_id}", "inspect_job"),
    Route("GET", "/jobs/{job_id}/board", "job_board"),
    Route("GET", "/jobs/{job_id}/timeline", "job_timeline"),
    Route("GET", "/jobs/{job_id}/blockers", "graph_blockers"),
    Route("GET", "/jobs/{job_id}/result", "read_job_result"),
    Route("GET", "/jobs/{job_id}/dependencies", "list_job_dependencies"),
    Route("GET", "/jobs/{job_id}/artifacts", "list_task_artifacts"),
    Route("GET", "/jobs/{job_id}/tasks", "list_job_tasks"),
    Route("GET", _TASK, "inspect_job_task"),
    Route("GET", _TASK + "/attempts", "list_task_attempts"),
    Route("GET", _TASK + "/checkpoints", "list_task_checkpoints"),
    Route("GET", _TASK + "/result", "read_task_result"),
    Route("POST", "/jobs/{job_id}/cancel", "cancel_job", work=True),
    Route("POST", "/jobs/{job_id}/graph/cancel", "cancel_graph_job", work=True),
    Route("POST", _TASK + "/replace", "replace_graph_task_by_policy", work=True),
    Route("POST", _CONTROL + "/answer", "answer_task_input", work=True),
    Route("POST", _CONTROL + "/accept-review", "accept_task_review", work=True),
    Route(
        "POST", _CONTROL + "/request-changes", "request_task_review_changes", work=True
    ),
    Route("POST", _CONTROL + "/retry", "retry_task_control", work=True),
    Route("POST", _CONTROL + "/reject", "reject_task_control", work=True),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
