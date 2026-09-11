"""Shared helpers extracted from ``test_update_preview.py``."""

from __future__ import annotations

from datetime import UTC, datetime

from daita.adapters.models import SourceRegistration, source_registration_id

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
SOURCE_ID = source_registration_id(
    "agent-preview", "postgresql", "postgresql:preview-contract"
)
RESOURCE_ID = "catalog-resource:sha256:" + "2" * 64
SOURCE_REVISION = "catalog:sha256:" + "3" * 64
RESOURCE_REVISION = "sha256:" + "4" * 64


class _SourceStore:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration

    async def register_source(self, registration):
        self.registration = registration
        return registration

    async def load_source(self, agent_id: str, source_id: str):
        if (agent_id, source_id) == (
            self.registration.agent_id,
            self.registration.id,
        ):
            return self.registration
        return None

    async def list_sources(self, agent_id: str):
        return (self.registration,) if agent_id == self.registration.agent_id else ()

    async def detach_source(self, agent_id, source_id, detached_at):
        self.registration = self.registration.detach(detached_at)
        return self.registration


def _registration() -> SourceRegistration:
    return SourceRegistration.build(
        agent_id="agent-preview",
        adapter_id="postgresql",
        native_identity="postgresql:preview-contract",
        display_name="Preview PostgreSQL",
        configuration={
            "database": "warehouse",
            "host": "db.example.test",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "writer",
        },
        attached_at=NOW,
    )


def _guardrails(**changes: object) -> dict[str, object]:
    facts: dict[str, object] = {
        "relation_oid": "16384",
        "relation_kind": "r",
        "is_partition": False,
        "row_level_security": False,
        "force_row_level_security": False,
        "has_inheritance": False,
        "has_user_triggers": False,
        "has_rewrite_rules": False,
        "role_superuser": False,
        "role_bypass_rls": False,
        "role_create_database": False,
        "role_create_role": False,
        "role_replication": False,
        "can_connect": True,
        "can_use_schema": True,
        "can_select_table": True,
        "can_update_columns": True,
    }
    facts.update(changes)
    return facts
