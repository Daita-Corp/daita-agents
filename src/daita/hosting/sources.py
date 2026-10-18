"""Source selection, edit previews, permission inspection and registration decoding."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import cast

from .._json import canonical_json
from ..adapters.models import SourceRegistration
from ..adapters.postgresql import PostgreSQLSource
from ..adapters.protocols import ResourceSource
from ..adapters.sqlite import SQLiteSource
from ..capabilities import CapabilityInputError
from ..catalog.models import CatalogResource, ResourceKind, SourceCatalogSnapshot
from ..catalog.service import CatalogService
from ..domains.data import CatalogDataView
from ..domains.data.sql import validate_relational_write_scope
from ..security import SecretProvider, SecretReference
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
from .errors import AgentHomeError

_SOURCE_ALIAS_SEPARATOR = re.compile(r"[^a-z0-9]+")


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


def _source_from_registration(
    registration: SourceRegistration,
    *,
    secret_provider: SecretProvider,
) -> ResourceSource:
    """Reconstruct one admitted source only at the composition boundary."""

    configuration = registration.configuration
    adapter_id = registration.adapter_id
    if adapter_id == "sqlite":
        _require_configuration_fields(configuration, {"path"})
        return SQLiteSource(
            path=_configuration_text(configuration, "path"),
            name=registration.display_name,
        )
    if adapter_id == "postgresql":
        required = {
            "database",
            "host",
            "port",
            "schemas",
            "ssl_mode",
            "username",
        }
        allowed = required | {"credential_ref", "public_network_only"}
        fields = set(configuration)
        if not required <= fields or not fields <= allowed:
            raise AgentHomeError("PostgreSQL source configuration is invalid")
        raw_schemas = configuration["schemas"]
        if not isinstance(raw_schemas, tuple) or any(
            not isinstance(schema, str) for schema in raw_schemas
        ):
            raise AgentHomeError("PostgreSQL source configuration is invalid")
        raw_reference = configuration.get("credential_ref")
        if raw_reference is not None and not isinstance(raw_reference, str):
            raise AgentHomeError("PostgreSQL source configuration is invalid")
        try:
            reference = (
                None if raw_reference is None else SecretReference.parse(raw_reference)
            )
            return PostgreSQLSource(
                host=_configuration_text(configuration, "host"),
                port=_configuration_integer(configuration, "port"),
                database=_configuration_text(configuration, "database"),
                username=_configuration_text(configuration, "username"),
                credential=reference,
                schemas=cast(tuple[str, ...], raw_schemas),
                ssl_mode=_configuration_text(configuration, "ssl_mode"),
                public_network_only=_configuration_boolean(
                    configuration, "public_network_only", default=False
                ),
                name=registration.display_name,
                secret_provider=secret_provider,
            )
        except (TypeError, ValueError) as error:
            raise AgentHomeError(
                "PostgreSQL source configuration is invalid"
            ) from error
    raise AgentHomeError("registered source adapter cannot be refreshed")


def _require_configuration_fields(
    configuration: Mapping[str, object],
    expected: set[str],
) -> None:
    if set(configuration) != expected:
        raise AgentHomeError("source configuration fields are invalid")


def _configuration_text(
    configuration: Mapping[str, object],
    name: str,
) -> str:
    value = configuration[name]
    if not isinstance(value, str) or not value:
        raise AgentHomeError("source configuration text is invalid")
    return value


def _configuration_integer(
    configuration: Mapping[str, object],
    name: str,
) -> int:
    value = configuration[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise AgentHomeError("source configuration integer is invalid")
    return value


def _configuration_boolean(
    configuration: Mapping[str, object],
    name: str,
    *,
    default: bool,
) -> bool:
    value = configuration.get(name, default)
    if not isinstance(value, bool):
        raise AgentHomeError("source configuration boolean is invalid")
    return value


def _source_edit_read_scope(
    *,
    current_read_scope: SourceReadScope,
    current_resources: tuple[CatalogResource, ...],
    replacement: SourceCatalogSnapshot,
) -> tuple[SourceReadScope, tuple[str, ...]]:
    """Carry read intent only across exact kind/native-identity matches."""

    if current_read_scope.mode is SourceReadMode.ALL:
        return (
            SourceReadScope.allow_all(
                agent_id=replacement.sync.agent_id,
                source_id=replacement.sync.source_id,
            ),
            (),
        )
    if current_read_scope.mode is SourceReadMode.NONE:
        return (
            SourceReadScope(
                agent_id=replacement.sync.agent_id,
                source_id=replacement.sync.source_id,
                mode=SourceReadMode.NONE,
            ),
            (),
        )

    current_by_id = {resource.id: resource for resource in current_resources}
    replacement_by_identity: dict[tuple[ResourceKind, str], list[CatalogResource]] = {}
    for resource in replacement.resources:
        replacement_by_identity.setdefault(
            (resource.kind, resource.native_identity),
            [],
        ).append(resource)
    preserved: list[str] = []
    omitted: list[str] = []
    for resource_id in current_read_scope.resource_ids:
        current = current_by_id.get(resource_id)
        if current is None:
            omitted.append(resource_id)
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
                agent_id=replacement.sync.agent_id,
                source_id=replacement.sync.source_id,
                mode=SourceReadMode.NONE,
            ),
            tuple(sorted(omitted)),
        )
    return (
        SourceReadScope(
            agent_id=replacement.sync.agent_id,
            source_id=replacement.sync.source_id,
            mode=SourceReadMode.SELECTED,
            resource_ids=tuple(preserved),
        ),
        tuple(sorted(omitted)),
    )


async def _inspect_source_permissions(
    store: StateStore,
    data_view: CatalogDataView,
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
    refs = await store.list_current_snapshot_refs(
        agent_id,
        (source_id,),
    )
    if len(refs) > 1:
        raise ValueError("source has multiple current catalog generations")
    catalog_generation = None if not refs else refs[0].sync_id
    resources = tuple(
        resource
        for resource in await store.list_resources(
            agent_id,
            source_id,
        )
        if resource.agent_id == agent_id and resource.source_id == source_id
    )
    schemas = (
        await data_view.resource_schemas(agent_id, source_id)
        if registration.adapter_id == "postgresql"
        else ()
    )
    schemas_by_resource_id = {schema.resource_id: schema for schema in schemas}
    choices: list[SourcePermissionResource] = []
    from ..domains.data.sql.relational_upsert import (
        validate_relational_upsert_scope,
    )

    for resource in resources:
        eligible_columns: tuple[str, ...] = ()
        insert_columns: tuple[str, ...] = ()
        upsert_keys: list[tuple[str, ...]] = []
        schema = schemas_by_resource_id.get(resource.id)
        if schema is not None and resource.kind is ResourceKind.TABLE:
            eligible_columns = tuple(
                column
                for column in schema.columns
                if validate_relational_write_scope(
                    source_id,
                    resource.id,
                    (column,),
                    resources=(schema,),
                ).valid
            )
            insert_columns = tuple(
                column
                for column in schema.columns
                if column not in schema.identity_columns
            )
            for keys in schema.conflict_keys:
                try:
                    validate_relational_upsert_scope(
                        schema,
                        key_columns=keys,
                        insert_columns=insert_columns,
                        update_columns=tuple(
                            column
                            for column in schema.updatable_columns
                            if column
                            not in (
                                *keys,
                                *schema.primary_key_columns,
                                *schema.identity_columns,
                            )
                        ),
                        generated_identity_columns=schema.identity_columns,
                    )
                except CapabilityInputError:
                    continue
                upsert_keys.append(keys)
        choices.append(
            SourcePermissionResource(
                resource_id=resource.id,
                display_name=resource.native_identity,
                resource_kind=resource.kind.value,
                eligible_assignment_columns=eligible_columns,
                key_columns=() if schema is None else schema.primary_key_columns,
                upsert_conflict_keys=tuple(upsert_keys),
                eligible_insert_columns=insert_columns if upsert_keys else (),
                eligible_upsert_update_columns=(
                    ()
                    if schema is None or not upsert_keys
                    else tuple(
                        column
                        for column in schema.updatable_columns
                        if column
                        not in (
                            *schema.primary_key_columns,
                            *schema.identity_columns,
                        )
                    )
                ),
                generated_identity_columns=(
                    () if schema is None or not upsert_keys else schema.identity_columns
                ),
            )
        )
    return SourcePermissionsInspection(
        source_id=source_id,
        source_display_name=registration.display_name,
        adapter_id=registration.adapter_id,
        catalog_generation=catalog_generation,
        state=SourcePermissionState(read_scope, update_scopes),
        resources=tuple(choices),
    )


async def _build_source_permissions_preview(
    store: StateStore,
    catalog_service: CatalogService,
    data_view: CatalogDataView,
    agent_id: str,
    confirmation_key: bytes,
    inspection: SourcePermissionsInspection,
    *,
    read_mode: SourceReadMode,
    read_resource_ids: tuple[str, ...],
    relational_write_scopes: Mapping[str, Mapping[str, object]],
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
    if requested_read.mode is SourceReadMode.SELECTED:
        unknown_read_ids = set(requested_read.resource_ids) - set(current_resources)
        if unknown_read_ids:
            raise ValueError(
                "selected read resources are not current resources of this source"
            )
    if relational_write_scopes and inspection.adapter_id != "postgresql":
        raise ValueError("PostgreSQL update scopes require an active PostgreSQL source")

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
    tabular_by_resource_id = {
        item.resource.id: item
        for item in await catalog_service.tabular_resources(
            agent_id,
            inspection.source_id,
        )
    }
    update_scopes: list[RelationalWriteScope] = []
    requested_update_ids: set[str] = set()
    schemas = {
        item.resource_id: item
        for item in await data_view.resource_schemas(agent_id, inspection.source_id)
    }
    from ..domains.data.sql.relational_upsert import (
        validate_relational_upsert_scope,
    )

    for resource_id, constraints in relational_write_scopes.items():
        if not isinstance(constraints, Mapping) or set(constraints) != {
            "allowed_operations",
            "allowed_insert_columns",
            "allowed_update_columns",
            "key_columns",
            "generated_identity_columns",
            "max_rows",
        }:
            raise ValueError(
                "write scopes require explicit operations, keys, insert/update columns, identities and row limit"
            )
        tabular = tabular_by_resource_id.get(resource_id)
        schema = schemas.get(resource_id)
        if (
            tabular is None
            or schema is None
            or tabular.resource.kind is not ResourceKind.TABLE
        ):
            raise ValueError("write scope requires an exact current table")
        scope = RelationalWriteScope(
            agent_id=agent_id,
            source_id=inspection.source_id,
            resource_id=resource_id,
            resource_revision=tabular.resource.current_revision,
            allowed_operations=cast(tuple[str, ...], constraints["allowed_operations"]),
            allowed_insert_columns=cast(
                tuple[str, ...], constraints["allowed_insert_columns"]
            ),
            allowed_update_columns=cast(
                tuple[str, ...], constraints["allowed_update_columns"]
            ),
            key_columns=cast(tuple[str, ...], constraints["key_columns"]),
            generated_identity_columns=cast(
                tuple[str, ...], constraints["generated_identity_columns"]
            ),
            max_rows=cast(int, constraints["max_rows"]),
            authorization_fingerprint="sha256:" + "0" * 64,
        )
        if "update" in scope.allowed_operations:
            if (
                set(scope.key_columns) != set(schema.primary_key_columns)
                or not validate_relational_write_scope(
                    inspection.source_id,
                    resource_id,
                    scope.allowed_update_columns,
                    resources=(schema,),
                ).valid
            ):
                raise ValueError(
                    "update scope requires eligible columns and the exact primary key"
                )
        if "upsert" in scope.allowed_operations:
            validate_relational_upsert_scope(
                schema,
                key_columns=scope.key_columns,
                insert_columns=scope.allowed_insert_columns,
                update_columns=scope.allowed_update_columns,
                generated_identity_columns=scope.generated_identity_columns,
            )
            if scope.max_rows > 1000:
                raise ValueError("upsert scope cannot exceed 1000 rows")
        scope = replace(
            scope,
            authorization_fingerprint=relational_write_authorization_fingerprint(
                source=registration,
                resource=tabular.resource,
                facet=tabular.facet,
                scope=scope,
            ),
        )
        update_scopes.append(scope)
        requested_update_ids.add(resource_id)

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
    effective_read_ids = (
        set(current_resources)
        if final_read_scope.mode is SourceReadMode.ALL
        else set(final_read_scope.resource_ids)
    )
    dependent_revocations = tuple(
        scope.resource_id
        for scope in inspection.state.relational_write_scopes
        if scope.resource_id not in effective_read_ids
    )

    after = SourcePermissionState(
        final_read_scope,
        tuple(update_scopes),
    )
    names = {
        resource.resource_id: resource.display_name for resource in inspection.resources
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
        "catalog_generation": inspection.catalog_generation,
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
        catalog_generation=inspection.catalog_generation,
        before=inspection.state,
        after=after,
        automatic_read_additions=tuple(automatic_read_additions),
        dependent_write_revocations=dependent_revocations,
        summary=summary,
        confirmation_fingerprint=confirmation_fingerprint,
    )
