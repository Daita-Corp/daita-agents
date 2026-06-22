"""Facade service for explicit DB memory command planning."""

from __future__ import annotations

from typing import Any

from daita.db.models import DbRequest

from .extractor import DeterministicDbMemoryIntentExtractor
from .planner import DbMemoryProposalPlanner
from .router import route_db_memory_command


class DbMemoryCommandService:
    """Route, extract, and plan explicit DB memory commands."""

    def __init__(self) -> None:
        self.extractor = DeterministicDbMemoryIntentExtractor()
        self.planner = DbMemoryProposalPlanner()

    def plan_update(
        self,
        request: DbRequest,
        *,
        source_identity: str | None,
        workspace_scope: str = "source",
        schema: dict[str, Any] | None = None,
        schema_fingerprint: str | None = None,
    ):
        command = route_db_memory_command(request)
        intent = self.extractor.extract(
            command,
            request,
            source_identity=source_identity,
            workspace_scope=workspace_scope,
        )
        return self.planner.plan(
            intent,
            schema=schema,
            source_identity=source_identity,
            schema_fingerprint=schema_fingerprint,
        )
