"""Explicit native permission fixtures shared by boundary and composition tests."""

from collections.abc import Sequence
from dataclasses import replace

from daita.adapters.models import SourceRegistration
from daita.catalog.models import CatalogFacet, CatalogResource, TabularFacet
from daita.storage.sqlite_records import (
    RelationalWriteScope,
    relational_write_authorization_fingerprint,
)


def update_constraints(
    columns: Sequence[str],
    *,
    key_columns: tuple[str, ...] = ("id",),
    max_rows: int = 10000,
) -> dict[str, object]:
    return {
        "allowed_operations": ("update",),
        "allowed_insert_columns": (),
        "allowed_update_columns": tuple(columns),
        "key_columns": key_columns,
        "generated_identity_columns": (),
        "max_rows": max_rows,
    }


def update_scope(
    source: SourceRegistration,
    resource: CatalogResource,
    facet: CatalogFacet,
    columns: tuple[str, ...],
) -> RelationalWriteScope:
    keys = tuple(
        column.name
        for column in TabularFacet.from_payload(facet.payload).columns
        if column.primary_key_ordinal is not None
    )
    scope = RelationalWriteScope(
        agent_id=source.agent_id,
        source_id=source.id,
        resource_id=resource.id,
        resource_revision=resource.current_revision,
        allowed_operations=("update",),
        allowed_insert_columns=(),
        allowed_update_columns=columns,
        key_columns=keys,
        generated_identity_columns=(),
        max_rows=10000,
        authorization_fingerprint="sha256:" + "0" * 64,
    )
    return replace(
        scope,
        authorization_fingerprint=relational_write_authorization_fingerprint(
            source=source, resource=resource, facet=facet, scope=scope
        ),
    )
