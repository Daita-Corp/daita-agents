"""SQLite-owned durable records shared with the database-write executor."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from hashlib import sha256
from collections.abc import Iterable, Mapping

from .._json import canonical_json
from ..adapters.models import SourceRegistration
from ..catalog.models import CatalogFacet, CatalogResource, FacetKind, ResourceKind

_DATABASE_WRITE_RECEIPT_ID = re.compile(r"database-write-receipt:sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_SOURCE_ID = re.compile(r"source:sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_RESOURCE_ID = re.compile(r"catalog-resource:sha256:[0-9a-f]{64}\Z")
_DATABASE_WRITE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_SOURCE_PERMISSION_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_PERMISSION_SOURCE_ID = re.compile(r"source:sha256:[0-9a-f]{64}\Z")
_SOURCE_PERMISSION_RESOURCE_ID = re.compile(r"catalog-resource:sha256:[0-9a-f]{64}\Z")
_SOURCE_PERMISSION_MAX_RESOURCE_IDS = 10_000
_SOURCE_PERMISSION_MAX_ASSIGNMENT_COLUMNS = 32


class SourceReadMode(str, Enum):
    ALL = "all"
    SELECTED = "selected"
    NONE = "none"


class SourcePermissionStateError(RuntimeError):
    """A durable source permission record is missing, foreign, or invalid."""


def _permission_text(value: str, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty text without surrounding space")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return value


def _canonical_permission_texts(
    values: Iterable[str],
    name: str,
    *,
    maximum_items: int,
    maximum_characters: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    items = tuple(values)
    if len(items) > maximum_items:
        raise ValueError(f"{name} exceeds {maximum_items} items")
    for item in items:
        _permission_text(item, name, maximum=maximum_characters)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(items))


@dataclass(frozen=True, slots=True)
class SourceReadScope:
    """One exact fail-closed read scope for an active source."""

    agent_id: str
    source_id: str
    mode: SourceReadMode
    resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _permission_text(self.agent_id, "read scope agent_id")
        if (
            not isinstance(self.source_id, str)
            or _SOURCE_PERMISSION_SOURCE_ID.fullmatch(self.source_id) is None
        ):
            raise ValueError("read scope source_id must be a canonical source id")
        if not isinstance(self.mode, SourceReadMode):
            raise TypeError("read scope mode must be a SourceReadMode")
        resource_ids = _canonical_permission_texts(
            self.resource_ids,
            "read scope resource_ids",
            maximum_items=_SOURCE_PERMISSION_MAX_RESOURCE_IDS,
            maximum_characters=256,
        )
        if any(
            _SOURCE_PERMISSION_RESOURCE_ID.fullmatch(resource_id) is None
            for resource_id in resource_ids
        ):
            raise ValueError(
                "read scope resource_ids must be canonical catalog resource ids"
            )
        if self.mode is SourceReadMode.SELECTED and not resource_ids:
            raise ValueError("selected read scope requires resource_ids")
        if self.mode is not SourceReadMode.SELECTED and resource_ids:
            raise ValueError("only selected read scope can contain resource_ids")
        object.__setattr__(self, "resource_ids", resource_ids)

    @classmethod
    def allow_all(cls, *, agent_id: str, source_id: str) -> SourceReadScope:
        return cls(agent_id=agent_id, source_id=source_id, mode=SourceReadMode.ALL)


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateScope:
    """One exact table and assignment-column PostgreSQL authorization."""

    agent_id: str
    source_id: str
    resource_id: str
    allowed_assignment_columns: tuple[str, ...]
    authorization_fingerprint: str

    def __post_init__(self) -> None:
        _permission_text(self.agent_id, "update scope agent_id")
        if (
            not isinstance(self.source_id, str)
            or _SOURCE_PERMISSION_SOURCE_ID.fullmatch(self.source_id) is None
        ):
            raise ValueError("update scope source_id must be a canonical source id")
        if (
            not isinstance(self.resource_id, str)
            or _SOURCE_PERMISSION_RESOURCE_ID.fullmatch(self.resource_id) is None
        ):
            raise ValueError(
                "update scope resource_id must be a canonical catalog resource id"
            )
        columns = _canonical_permission_texts(
            self.allowed_assignment_columns,
            "update scope allowed_assignment_columns",
            maximum_items=_SOURCE_PERMISSION_MAX_ASSIGNMENT_COLUMNS,
            maximum_characters=256,
        )
        if not columns:
            raise ValueError("update scope allowed_assignment_columns cannot be empty")
        if (
            not isinstance(self.authorization_fingerprint, str)
            or _SOURCE_PERMISSION_HASH.fullmatch(self.authorization_fingerprint) is None
        ):
            raise ValueError(
                "update scope authorization_fingerprint must be a sha256 hash"
            )
        object.__setattr__(self, "allowed_assignment_columns", columns)


def postgresql_update_authorization_fingerprint(
    *,
    source: SourceRegistration,
    resource: CatalogResource,
    facet: CatalogFacet,
    allowed_assignment_columns: Iterable[str],
) -> str:
    """Bind only durable facts that determine one update authorization's meaning."""

    if not isinstance(source, SourceRegistration):
        raise TypeError("authorization source must be a SourceRegistration")
    if not isinstance(resource, CatalogResource):
        raise TypeError("authorization resource must be a CatalogResource")
    if not isinstance(facet, CatalogFacet):
        raise TypeError("authorization facet must be a CatalogFacet")
    if (
        source.adapter_id != "postgresql"
        or not source.active
        or resource.agent_id != source.agent_id
        or resource.source_id != source.id
        or resource.kind is not ResourceKind.TABLE
        or facet.resource_id != resource.id
        or facet.kind is not FacetKind.TABULAR
    ):
        raise ValueError(
            "authorization requires one current table from an active PostgreSQL source"
        )
    allowed = _canonical_permission_texts(
        allowed_assignment_columns,
        "authorization allowed_assignment_columns",
        maximum_items=_SOURCE_PERMISSION_MAX_ASSIGNMENT_COLUMNS,
        maximum_characters=256,
    )
    if not allowed:
        raise ValueError("authorization allowed_assignment_columns cannot be empty")
    raw_columns = facet.payload.get("columns")
    if not isinstance(raw_columns, tuple):
        raise ValueError("authorization requires exact tabular column facts")
    columns: dict[str, Mapping[str, object]] = {}
    for raw_column in raw_columns:
        if not isinstance(raw_column, Mapping):
            raise ValueError("authorization tabular column facts are invalid")
        name = raw_column.get("name")
        if not isinstance(name, str) or name in columns:
            raise ValueError("authorization tabular column identity is invalid")
        columns[name] = raw_column

    primary_key_columns: list[tuple[int, str]] = []
    for name, column in columns.items():
        ordinal = column.get("primary_key_ordinal")
        if ordinal is None:
            continue
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
            raise ValueError("authorization primary-key structure is invalid")
        primary_key_columns.append((ordinal, name))
    primary_key_columns.sort()
    if [ordinal for ordinal, _ in primary_key_columns] != list(
        range(1, len(primary_key_columns) + 1)
    ):
        raise ValueError("authorization primary-key structure is invalid")

    allowed_facts: list[dict[str, object]] = []
    primary_names = {name for _, name in primary_key_columns}
    for name in allowed:
        selected_column = columns.get(name)
        if selected_column is None:
            raise ValueError("authorization references an unknown assignment column")
        native_type = selected_column.get("native_type")
        namespace = selected_column.get("native_type_namespace")
        native_name = selected_column.get("native_type_name")
        updatable = selected_column.get("updatable")
        identity = selected_column.get("identity")
        generated = selected_column.get("generated")
        if (
            not isinstance(native_type, str)
            or not isinstance(updatable, bool)
            or not isinstance(identity, bool)
            or not isinstance(generated, bool)
            or (namespace is None) is not (native_name is None)
            or (namespace is not None and not isinstance(namespace, str))
            or (native_name is not None and not isinstance(native_name, str))
        ):
            raise ValueError("authorization assignment-column facts are invalid")
        if name in primary_names or not updatable or identity or generated:
            raise ValueError("authorization assignment column is not eligible")
        allowed_facts.append(
            {
                "generated": generated,
                "identity": identity,
                "name": name,
                "native_type": native_type,
                "native_type_name": native_name,
                "native_type_namespace": namespace,
                "updatable": updatable,
            }
        )

    material = {
        "adapter_id": source.adapter_id,
        "allowed_assignment_columns": allowed_facts,
        "primary_key": tuple(
            {"name": name, "ordinal": ordinal} for ordinal, name in primary_key_columns
        ),
        "resource_id": resource.id,
        "resource_kind": resource.kind.value,
        "source_id": source.id,
    }
    return "sha256:" + sha256(canonical_json(material).encode("utf-8")).hexdigest()


class DatabaseWriteOutcome(str, Enum):
    STARTED = "started"
    COMMITTED = "committed"
    NOT_COMMITTED = "not_committed"
    OUTCOME_UNKNOWN = "outcome_unknown"


class DatabaseWriteReceiptConflictError(RuntimeError):
    """The durable receipt identity or immutable terminal state conflicts."""


def database_write_text(value: str, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty text without surrounding space")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return value


def database_write_aware(value: datetime, name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be timezone-aware")
    return value


def database_write_receipt_id(
    *,
    agent_id: str,
    run_id: str,
    call_id: str,
    capability_id: str,
    intent_sha256: str,
) -> str:
    identity = {
        "agent_id": database_write_text(agent_id, "receipt agent_id"),
        "call_id": database_write_text(call_id, "receipt call_id"),
        "capability_id": database_write_text(
            capability_id, "receipt capability_id", maximum=128
        ),
        "intent_sha256": intent_sha256,
        "run_id": database_write_text(run_id, "receipt run_id"),
    }
    if (
        not isinstance(intent_sha256, str)
        or _DATABASE_WRITE_HASH.fullmatch(intent_sha256) is None
    ):
        raise ValueError("receipt intent_sha256 must be a sha256 hash")
    digest = sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return f"database-write-receipt:sha256:{digest}"


def validate_database_write_receipt_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or _DATABASE_WRITE_RECEIPT_ID.fullmatch(value) is None
    ):
        raise ValueError("receipt_id must be a canonical database-write receipt id")
    return value


@dataclass(frozen=True, slots=True)
class DatabaseWriteReceipt:
    """Bounded durable metadata for one exact external database-write attempt."""

    receipt_id: str
    agent_id: str
    run_id: str
    call_id: str
    capability_id: str
    source_id: str
    resource_id: str
    intent_sha256: str
    preview_fingerprint: str
    outcome: DatabaseWriteOutcome
    affected_rows: int | None
    normalized_error_code: str | None
    started_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        validate_database_write_receipt_id(self.receipt_id)
        database_write_text(self.agent_id, "receipt agent_id")
        database_write_text(self.run_id, "receipt run_id")
        database_write_text(self.call_id, "receipt call_id")
        database_write_text(self.capability_id, "receipt capability_id", maximum=128)
        if (
            not isinstance(self.source_id, str)
            or _DATABASE_WRITE_SOURCE_ID.fullmatch(self.source_id) is None
        ):
            raise ValueError("receipt source_id must be a canonical source id")
        if (
            not isinstance(self.resource_id, str)
            or _DATABASE_WRITE_RESOURCE_ID.fullmatch(self.resource_id) is None
        ):
            raise ValueError("receipt resource_id must be a canonical resource id")
        for value, name in (
            (self.intent_sha256, "intent_sha256"),
            (self.preview_fingerprint, "preview_fingerprint"),
        ):
            if (
                not isinstance(value, str)
                or _DATABASE_WRITE_HASH.fullmatch(value) is None
            ):
                raise ValueError(f"receipt {name} must be a sha256 hash")
        if not isinstance(self.outcome, DatabaseWriteOutcome):
            raise TypeError("receipt outcome must be a DatabaseWriteOutcome")
        database_write_aware(self.started_at, "receipt started_at")
        if self.completed_at is not None:
            database_write_aware(self.completed_at, "receipt completed_at")
            if self.completed_at < self.started_at:
                raise ValueError("receipt cannot complete before it starts")
        expected_id = database_write_receipt_id(
            agent_id=self.agent_id,
            run_id=self.run_id,
            call_id=self.call_id,
            capability_id=self.capability_id,
            intent_sha256=self.intent_sha256,
        )
        if self.receipt_id != expected_id:
            raise ValueError("receipt_id does not match its execution identity")
        if self.normalized_error_code is not None and (
            not isinstance(self.normalized_error_code, str)
            or _DATABASE_WRITE_ERROR_CODE.fullmatch(self.normalized_error_code) is None
        ):
            raise ValueError("receipt normalized_error_code is invalid")
        if self.affected_rows is not None and (
            not isinstance(self.affected_rows, int)
            or isinstance(self.affected_rows, bool)
        ):
            raise TypeError("receipt affected_rows must be an integer or None")
        if self.outcome is DatabaseWriteOutcome.STARTED:
            if any(
                value is not None
                for value in (
                    self.affected_rows,
                    self.normalized_error_code,
                    self.completed_at,
                )
            ):
                raise ValueError("started receipt cannot contain terminal fields")
        elif self.outcome is DatabaseWriteOutcome.COMMITTED:
            if (
                self.affected_rows != 1
                or self.normalized_error_code is not None
                or self.completed_at is None
            ):
                raise ValueError("committed receipt must record one affected row")
        elif self.outcome is DatabaseWriteOutcome.NOT_COMMITTED:
            if (
                self.affected_rows != 0
                or self.normalized_error_code is None
                or self.completed_at is None
            ):
                raise ValueError(
                    "not_committed receipt must record zero rows and an error code"
                )
        elif (
            self.affected_rows is not None
            or self.normalized_error_code != "write_outcome_unknown"
            or self.completed_at is None
        ):
            raise ValueError(
                "outcome_unknown receipt must omit affected rows and use its stable code"
            )

    @classmethod
    def start(
        cls,
        *,
        agent_id: str,
        run_id: str,
        call_id: str,
        capability_id: str,
        source_id: str,
        resource_id: str,
        intent_sha256: str,
        preview_fingerprint: str,
        started_at: datetime,
    ) -> DatabaseWriteReceipt:
        return cls(
            receipt_id=database_write_receipt_id(
                agent_id=agent_id,
                run_id=run_id,
                call_id=call_id,
                capability_id=capability_id,
                intent_sha256=intent_sha256,
            ),
            agent_id=agent_id,
            run_id=run_id,
            call_id=call_id,
            capability_id=capability_id,
            source_id=source_id,
            resource_id=resource_id,
            intent_sha256=intent_sha256,
            preview_fingerprint=preview_fingerprint,
            outcome=DatabaseWriteOutcome.STARTED,
            affected_rows=None,
            normalized_error_code=None,
            started_at=started_at,
            completed_at=None,
        )

    def finish(
        self,
        outcome: DatabaseWriteOutcome,
        *,
        completed_at: datetime,
        affected_rows: int | None,
        normalized_error_code: str | None,
    ) -> DatabaseWriteReceipt:
        if self.outcome is not DatabaseWriteOutcome.STARTED:
            raise ValueError("only a started receipt can reach a terminal outcome")
        if outcome is DatabaseWriteOutcome.STARTED:
            raise ValueError("receipt terminal outcome cannot be started")
        return replace(
            self,
            outcome=outcome,
            affected_rows=affected_rows,
            normalized_error_code=normalized_error_code,
            completed_at=completed_at,
        )

    def as_started(self) -> DatabaseWriteReceipt:
        return replace(
            self,
            outcome=DatabaseWriteOutcome.STARTED,
            affected_rows=None,
            normalized_error_code=None,
            completed_at=None,
        )


__all__ = [
    "DatabaseWriteOutcome",
    "DatabaseWriteReceipt",
    "DatabaseWriteReceiptConflictError",
    "PostgreSQLUpdateScope",
    "SourcePermissionStateError",
    "SourceReadMode",
    "SourceReadScope",
    "database_write_aware",
    "database_write_receipt_id",
    "database_write_text",
    "postgresql_update_authorization_fingerprint",
    "validate_database_write_receipt_id",
]
