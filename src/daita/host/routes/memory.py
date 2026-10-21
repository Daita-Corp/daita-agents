"""Memory, the user profile, skills and semantic annotations."""

from __future__ import annotations

from . import Route, http_routes

TABLE = (
    Route("GET", "/memory", "read_memory"),
    Route("PUT", "/memory", "set_memory"),
    Route("GET", "/user-profile", "read_user_profile"),
    Route("PUT", "/user-profile", "set_user_profile"),
    Route("GET", "/skills", "list_skills"),
    Route("GET", "/skills/{name}", "read_skill"),
    Route("PUT", "/skills/{name}", "save_skill"),
    Route("DELETE", "/skills/{name}", "delete_skill"),
    Route("GET", "/semantics", "list_semantic_annotations"),
    Route("GET", "/semantics/{annotation_id}", "read_semantic_annotation"),
    Route("DELETE", "/semantics/{annotation_id}", "delete_semantic_annotation"),
)
MEMBERS = frozenset(route.member for route in TABLE)

ROUTES = http_routes(TABLE)
