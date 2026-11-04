"""Explicit native permission fixtures shared by boundary and composition tests."""

from collections.abc import Sequence
from dataclasses import replace

from daita.adapters.models import SourceRegistration
from daita.catalog.models import CatalogResource
from daita.storage.sqlite_records import (
    RelationalWriteScope,
    relational_write_authorization_fingerprint,
)


def update_constraints(
    columns: Sequence[str], *, max_rows: int = 10000
) -> dict[str, object]:
    return {
        "allowed_operations": ("update",),
        "allowed_insert_columns": (),
        "allowed_update_columns": tuple(columns),
        "max_rows": max_rows,
    }


def update_scope(
    source: SourceRegistration,
    resource: CatalogResource,
    columns: tuple[str, ...],
) -> RelationalWriteScope:
    scope = RelationalWriteScope(
        agent_id=source.agent_id,
        source_id=source.id,
        resource_id=resource.id,
        table=resource.native_identity,
        allowed_operations=("update",),
        allowed_insert_columns=(),
        allowed_update_columns=columns,
        max_rows=10000,
        authorization_fingerprint="sha256:" + "0" * 64,
    )
    return replace(
        scope,
        authorization_fingerprint=relational_write_authorization_fingerprint(
            source=source, scope=scope
        ),
    )
