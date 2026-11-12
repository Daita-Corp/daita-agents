"""Source selection, edit previews and permission inspection."""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import cast

from .._json import canonical_json
from ..adapters.models import SourceRegistration
from ..adapters.postgresql_write import RelationalWriteError
from ..adapters.registry import SourceOperation, SourceTypeRegistry
from ..catalog.models import (
    CatalogResource,
    ResourceKind,
    catalog_resource_id,
)
from ..catalog.protocols import CatalogStoreError
from ..domains.data import CatalogDataView
from ..storage.protocols import StateStore
from ..storage.sqlite_records import (
    RelationalWriteScope,
    SourcePermissionResource,
    SourcePermissionsInspection,
    SourcePermissionsPreview,
    SourcePermissionState,
    SourcePermissionSummary,
    SourceReadMode,
    SourceReadScope,
    relational_write_authorization_fingerprint,
)

_LOGGER = logging.getLogger("daita.hosting")
_SOURCE_ALIAS_SEPARATOR = re.compile(r"[^a-z0-9]+")
_UNCATALOGED_READ = "a table the catalog has not seen"


def _source_alias(value: str) -> str:
    return _SOURCE_ALIAS_SEPARATOR.sub("-", value.casefold()).strip("-")


def _resolve_source_selector(
    selector: str,
    sources: tuple[SourceRegistration, ...],
) -> SourceRegistration:
    if not isinstance(selector, str) or not (candidate := selector.strip()):
        raise ValueError("Source selector must be non-empty terminal text.")
    try:
        encoded_length = len(candidate.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError("Source selector must be non-empty terminal text.") from None
    if encoded_length > 1_024 or any(
        ord(character) < 32 or ord(character) == 127 for character in candidate
    ):
        raise ValueError("Source selector must be non-empty terminal text.")
    exact_id = tuple(source for source in sources if source.id == candidate)
    if exact_id:
        return exact_id[0]
    folded = candidate.casefold()
    matches = {
        source.id: source
        for source in sources
        if source.display_name.casefold() == folded
        or _source_alias(source.display_name) == folded
    }
    if not matches:
        raise ValueError(
            "No active source matches that name. Use /sources to list choices."
        )
    if len(matches) > 1:
        raise ValueError("That source name is ambiguous. Use /source to choose one.")
    return next(iter(matches.values()))


@dataclass(frozen=True, slots=True)
class SourceEditPreview:
    """Safe bounded facts shown immediately before a source connection edit."""

    current_source_id: str
    replacement_source_id: str
    display_name: str
    adapter_id: str
    resource_count: int
    relationship_count: int
    read_mode: SourceReadMode
    preserved_read_resource_count: int
    omitted_read_resources: tuple[str, ...]

    @property
    def identity_changed(self) -> bool:
        return self.current_source_id != self.replacement_source_id


@dataclass(frozen=True, slots=True)
class SourceEditResult:
    """Committed source connection edit plus best-effort credential cleanup."""

    source: SourceRegistration
    identity_changed: bool
    previous_credential_deleted: bool


SourceEditConfirmationHandler = Callable[[SourceEditPreview], Awaitable[bool]]


def _source_permission_state_payload(
    state: SourcePermissionState,
) -> dict[str, object]:
    return {
        "read": {
            "mode": state.read_scope.mode.value,
            "resource_ids": state.read_scope.resource_ids,
        },
        "relational_writes": tuple(
            {
                "resource_id": scope.resource_id,
                **scope.constraints(),
                "authorization_fingerprint": scope.authorization_fingerprint,
            }
            for scope in state.relational_write_scopes
        ),
    }


def _source_edit_read_scope(
    *,
    current_read_scope: SourceReadScope,
    current_resources: tuple[CatalogResource, ...],
    agent_id: str,
    source_id: str,
    replacement_resources: tuple[CatalogResource, ...],
) -> tuple[SourceReadScope, tuple[str, ...]]:
    """Carry read intent only across exact kind/native-identity matches."""

    if current_read_scope.mode is SourceReadMode.ALL:
        return (
            SourceReadScope.allow_all(
                agent_id=agent_id,
                source_id=source_id,
            ),
            (),
        )
    if current_read_scope.mode is SourceReadMode.NONE:
        return (
            SourceReadScope(
                agent_id=agent_id,
                source_id=source_id,
                mode=SourceReadMode.NONE,
            ),
            (),
        )

    current_by_id = {resource.id: resource for resource in current_resources}
    replacement_by_identity: dict[tuple[ResourceKind, str], list[CatalogResource]] = {}
    for resource in replacement_resources:
        replacement_by_identity.setdefault(
            (resource.kind, resource.native_identity),
            [],
        ).append(resource)
    preserved: list[str] = []
    omitted: list[str] = []
    for resource_id in current_read_scope.resource_ids:
        current = current_by_id.get(resource_id)
        if current is None:
            # Enabled live, so the catalog has no name for it; never show the ID.
            if _UNCATALOGED_READ not in omitted:
                omitted.append(_UNCATALOGED_READ)
            continue
        matches = replacement_by_identity.get(
            (current.kind, current.native_identity),
            [],
        )
        if len(matches) != 1:
            omitted.append(current.native_identity)
            continue
        preserved.append(matches[0].id)
    if not preserved:
        return (
            SourceReadScope(
                agent_id=agent_id,
                source_id=source_id,
                mode=SourceReadMode.NONE,
            ),
            tuple(sorted(omitted)),
        )
    return (
        SourceReadScope(
            agent_id=agent_id,
            source_id=source_id,
            mode=SourceReadMode.SELECTED,
            resource_ids=tuple(preserved),
        ),
        tuple(sorted(omitted)),
    )


def _supports_relational_writes(
    source_types: SourceTypeRegistry, adapter_id: str
) -> bool:
    return source_types.supports(adapter_id, SourceOperation.RELATIONAL_WRITE)


async def _inspect_source_permissions(
    store: StateStore,
    data_view: CatalogDataView,
    source_types: SourceTypeRegistry,
    agent_id: str,
    source_id: str,
) -> SourcePermissionsInspection:
    registration = await store.load_source(agent_id, source_id)
    if (
        registration is None
        or registration.agent_id != agent_id
        or registration.id != source_id
        or not registration.active
    ):
        raise ValueError("unknown active source for this agent")
    read_scope = await store.load_source_read_scope(
        agent_id,
        source_id,
    )
    if read_scope is None:
        raise ValueError("active source is missing its read scope")
    update_scopes = await store.list_relational_write_scopes(
        agent_id,
        source_id,
    )
    writable = _supports_relational_writes(source_types, registration.adapter_id)
    # The catalog only offers choices, so losing it must not stop a revoke.
    try:
        resources = tuple(
            resource
            for resource in await store.list_resources(agent_id, source_id)
            if resource.agent_id == agent_id and resource.source_id == source_id
        )
        schemas = (
            await data_view.resource_schemas(agent_id, source_id) if writable else ()
        )
    except (CatalogStoreError, ValueError) as error:
        _LOGGER.warning("catalog unavailable for permissions: %s", type(error).__name__)
        resources, schemas = (), ()
    schemas_by_resource_id = {schema.resource_id: schema for schema in schemas}
    # What the catalog knows only offers choices; enabling reads the table live.
    choices = tuple(
        SourcePermissionResource(
            resource_id=resource.id,
            display_name=resource.native_identity,
            resource_kind=resource.kind.value,
            eligible_assignment_columns=(
                tuple(
                    column
                    for column in schemas_by_resource_id[resource.id].updatable_columns
                    if column
                    not in schemas_by_resource_id[resource.id].identity_columns
                )
                if resource.id in schemas_by_resource_id
                and resource.kind is ResourceKind.TABLE
                else ()
            ),
            eligible_insert_columns=(
                tuple(
                    column
                    for column in schemas_by_resource_id[resource.id].columns
                    if column
                    not in schemas_by_resource_id[resource.id].identity_columns
                )
                if resource.id in schemas_by_resource_id
                and resource.kind is ResourceKind.TABLE
                else ()
            ),
        )
        for resource in resources
    )
    # A stored scope stays on offer, so it can always be removed, catalog or not.
    cataloged = {choice.resource_id for choice in choices}
    choices = (
        *choices,
        *(
            SourcePermissionResource(
                resource_id=scope.resource_id,
                display_name=scope.table,
                resource_kind=ResourceKind.TABLE.value,
                eligible_assignment_columns=scope.allowed_update_columns,
                eligible_insert_columns=scope.allowed_insert_columns,
            )
            for scope in update_scopes
            if scope.resource_id not in cataloged
        ),
    )
    return SourcePermissionsInspection(
        source_id=source_id,
        source_display_name=registration.display_name,
        adapter_id=registration.adapter_id,
        state=SourcePermissionState(read_scope, update_scopes),
        resources=choices,
        supports_relational_writes=writable,
    )


_EXPLICIT_WRITE_KEYS = frozenset(
    {
        "allowed_operations",
        "allowed_insert_columns",
        "allowed_update_columns",
        "max_rows",
    }
)
_GRANTED_WRITE_KEYS = frozenset({"allowed_operations", "max_rows"})


async def _granted_constraints(
    inspect_write_table: Callable[..., Awaitable[Mapping[str, tuple[str, ...]]]],
    agent_id: str,
    source_id: str,
    table: str,
    constraints: Mapping[str, object],
    stored: RelationalWriteScope | None,
) -> dict[str, object]:
    """Fill a scope's columns with everything the source login may write.

    The database role is the real limit, so Daita offers operations only and takes
    the columns from the live grants. Narrowing a stored scope still works with
    the source down, from the columns it already holds.
    """

    operations = tuple(cast(tuple[str, ...], constraints["allowed_operations"]))
    try:
        live = await inspect_write_table(
            agent_id=agent_id, source_id=source_id, table=table
        )
    except RelationalWriteError:
        if stored is None or not set(operations) <= set(stored.allowed_operations):
            raise
        live = {
            "update_columns": stored.allowed_update_columns,
            "insert_columns": stored.allowed_insert_columns,
            "operations": stored.allowed_operations,
        }
    refused = [item for item in operations if item not in live["operations"]]
    if refused:
        raise ValueError(
            f"The source login cannot {' or '.join(refused)} rows in {table}; "
            "grant it in the database first."
        )
    return {
        "allowed_operations": operations,
        "allowed_update_columns": (
            live["update_columns"] if "update" in operations else ()
        ),
        "allowed_insert_columns": (
            live["insert_columns"] if "insert" in operations else ()
        ),
        "max_rows": constraints["max_rows"],
    }


def _within(scope: RelationalWriteScope, stored: RelationalWriteScope | None) -> bool:
    """True when a scope grants nothing the stored one did not already grant."""

    return (
        stored is not None
        and set(scope.allowed_operations) <= set(stored.allowed_operations)
        and set(scope.allowed_insert_columns) <= set(stored.allowed_insert_columns)
        and set(scope.allowed_update_columns) <= set(stored.allowed_update_columns)
        and scope.max_rows <= stored.max_rows
    )


async def _build_source_permissions_preview(
    store: StateStore,
    source_types: SourceTypeRegistry,
    agent_id: str,
    confirmation_key: bytes,
    inspection: SourcePermissionsInspection,
    *,
    read_mode: SourceReadMode,
    read_resource_ids: tuple[str, ...],
    relational_write_scopes: Mapping[str, Mapping[str, object]],
    inspect_write_table: Callable[..., Awaitable[Mapping[str, tuple[str, ...]]]],
) -> SourcePermissionsPreview:
    if not isinstance(read_mode, SourceReadMode):
        raise TypeError("read_mode must be SourceReadMode")
    if not isinstance(read_resource_ids, tuple):
        raise TypeError("read_resource_ids must be a tuple")
    if not isinstance(relational_write_scopes, Mapping):
        raise TypeError("relational_write_scopes must be a mapping")
    current_resources = {
        resource.resource_id: resource for resource in inspection.resources
    }
    requested_read = SourceReadScope(
        agent_id=agent_id,
        source_id=inspection.source_id,
        mode=read_mode,
        resource_ids=read_resource_ids,
    )
    if relational_write_scopes and not inspection.supports_relational_writes:
        raise ValueError(
            "Relational write scopes require an active source whose type "
            "supports native writes"
        )

    registration = await store.load_source(
        agent_id,
        inspection.source_id,
    )
    if (
        registration is None
        or not registration.active
        or registration.adapter_id != inspection.adapter_id
    ):
        raise ValueError("source identity changed during permission preview")
    stored_scopes = {
        scope.table: scope for scope in inspection.state.relational_write_scopes
    }
    update_scopes: list[RelationalWriteScope] = []
    requested_update_ids: set[str] = set()
    for table, constraints in relational_write_scopes.items():
        if not isinstance(constraints, Mapping) or set(constraints) not in (
            _EXPLICIT_WRITE_KEYS,
            _GRANTED_WRITE_KEYS,
        ):
            raise ValueError(
                "write scopes require operations and a row limit, with insert/update "
                "columns given explicitly or both left to the login's grants"
            )
        if set(constraints) == _GRANTED_WRITE_KEYS:
            constraints = await _granted_constraints(
                inspect_write_table,
                agent_id,
                inspection.source_id,
                table,
                constraints,
                stored_scopes.get(table),
            )
        resource_id = catalog_resource_id(
            inspection.source_id, ResourceKind.TABLE, table
        )
        scope = RelationalWriteScope(
            agent_id=agent_id,
            source_id=inspection.source_id,
            resource_id=resource_id,
            table=table,
            allowed_operations=cast(tuple[str, ...], constraints["allowed_operations"]),
            allowed_insert_columns=cast(
                tuple[str, ...], constraints["allowed_insert_columns"]
            ),
            allowed_update_columns=cast(
                tuple[str, ...], constraints["allowed_update_columns"]
            ),
            max_rows=cast(int, constraints["max_rows"]),
            authorization_fingerprint="sha256:" + "0" * 64,
        )
        # Taking access away must work with the source down or the table gone.
        if not _within(scope, stored_scopes.get(table)):
            live = await inspect_write_table(
                agent_id=agent_id, source_id=inspection.source_id, table=table
            )
            missing = sorted(
                {*scope.allowed_insert_columns, *scope.allowed_update_columns}
                - set(live["columns"])
            )
            if missing:
                raise ValueError(f"{table} has no writable column {', '.join(missing)}")
        scope = replace(
            scope,
            authorization_fingerprint=relational_write_authorization_fingerprint(
                source=registration, scope=scope
            ),
        )
        update_scopes.append(scope)
        requested_update_ids.add(resource_id)

    if requested_read.mode is SourceReadMode.SELECTED:
        # A write's table and what is already stored need no catalog entry.
        unknown_read_ids = (
            set(requested_read.resource_ids)
            - set(current_resources)
            - requested_update_ids
            - set(inspection.state.read_scope.resource_ids)
        )
        if unknown_read_ids:
            raise ValueError(
                "selected read resources are not current resources of this source"
            )

    automatic_read_additions: set[str] = set()
    if requested_read.mode is SourceReadMode.ALL:
        final_read_scope = requested_read
    else:
        requested_ids = set(requested_read.resource_ids)
        automatic_read_additions = requested_update_ids - requested_ids
        final_ids = requested_ids | requested_update_ids
        final_read_scope = (
            SourceReadScope(
                agent_id=agent_id,
                source_id=inspection.source_id,
                mode=SourceReadMode.SELECTED,
                resource_ids=tuple(final_ids),
            )
            if final_ids
            else SourceReadScope(
                agent_id=agent_id,
                source_id=inspection.source_id,
                mode=SourceReadMode.NONE,
            )
        )
    dependent_revocations = tuple(
        scope.resource_id
        for scope in inspection.state.relational_write_scopes
        if final_read_scope.mode is not SourceReadMode.ALL
        and scope.resource_id not in final_read_scope.resource_ids
    )

    after = SourcePermissionState(
        final_read_scope,
        tuple(update_scopes),
    )
    names = {
        **{
            resource.resource_id: resource.display_name
            for resource in inspection.resources
        },
        **{
            scope.resource_id: scope.table
            for scope in (*inspection.state.relational_write_scopes, *update_scopes)
        },
    }
    summary = SourcePermissionSummary(
        source_display_name=inspection.source_display_name,
        read_mode=final_read_scope.mode,
        selected_read_resource_count=(
            len(current_resources)
            if final_read_scope.mode is SourceReadMode.ALL
            else len(final_read_scope.resource_ids)
        ),
        relational_write_table_count=len(update_scopes),
        relational_write_table_examples=tuple(
            names.get(scope.resource_id, scope.resource_id)
            for scope in sorted(
                update_scopes,
                key=lambda item: names.get(item.resource_id, item.resource_id),
            )[:5]
        ),
        automatic_read_addition_examples=tuple(
            names.get(resource_id, resource_id)
            for resource_id in sorted(
                automatic_read_additions,
                key=lambda item: names.get(item, item),
            )[:5]
        ),
        dependent_update_revocation_examples=tuple(
            names.get(resource_id, resource_id)
            for resource_id in sorted(
                dependent_revocations,
                key=lambda item: names.get(item, item),
            )[:5]
        ),
    )
    material = {
        "source_id": inspection.source_id,
        "adapter_id": inspection.adapter_id,
        "before": _source_permission_state_payload(inspection.state),
        "after": _source_permission_state_payload(after),
        "automatic_read_additions": tuple(sorted(automatic_read_additions)),
        "dependent_write_revocations": tuple(sorted(dependent_revocations)),
    }
    confirmation_fingerprint = (
        "sha256:"
        + hmac.new(
            confirmation_key,
            canonical_json(material).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )
    return SourcePermissionsPreview(
        source_id=inspection.source_id,
        before=inspection.state,
        after=after,
        automatic_read_additions=tuple(automatic_read_additions),
        dependent_write_revocations=dependent_revocations,
        summary=summary,
        confirmation_fingerprint=confirmation_fingerprint,
    )
