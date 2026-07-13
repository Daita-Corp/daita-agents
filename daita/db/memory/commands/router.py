"""Conservative routing for explicit DB memory commands."""

from __future__ import annotations

import re

from daita.db.models import DbRequest

from .types import DbMemoryCommand


def route_db_memory_command(request: DbRequest) -> DbMemoryCommand:
    """Route a DB request into an explicit memory command shape."""
    prompt = str(request.prompt or "")
    lowered = prompt.lower()
    action = "remember"
    if re.search(r"\b(forget|delete|remove)\b", lowered):
        action = "forget"
    elif re.search(
        r"\b(list|show)\b.*\b(memory|memories|definitions|rules)\b", lowered
    ):
        action = "list"
    elif re.search(r"\b(inspect|explain|show)\b.*\b(memory|memories)\b", lowered):
        action = "inspect"
    elif re.search(r"\b(update|replace|change)\b", lowered):
        action = "update"
    elif re.search(r"\b(remember|note)\b", lowered):
        action = "remember"

    metadata_action = request.metadata.get("action") or request.constraints.get(
        "action"
    )
    if isinstance(metadata_action, str) and metadata_action.strip():
        action = metadata_action.strip().lower()

    return DbMemoryCommand(
        action=action,
        prompt=prompt,
        mode=request.mode,
        metadata=dict(request.metadata),
        constraints=dict(request.constraints),
    )
