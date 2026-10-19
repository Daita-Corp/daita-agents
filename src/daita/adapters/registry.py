"""Register source types: identity, reconstruction and declared operations.

Every dispatch on a source's ``adapter_id`` outside the type's own adapter
modules goes through one composed :class:`SourceTypeRegistry`. Built-in types are
always registered; callers add their own types explicitly at agent composition.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from ..config import HostCapacityLimits, PostgreSQLReadLimits
from ..security import SecretProvider
from .protocols import ResourceSource, SourceStore

if TYPE_CHECKING:
    from ..domains.data.catalog import CatalogDataView
    from ..observation import AgentObserver


class SourceOperation(str, Enum):
    """One data operation a source type implements through its own backend."""

    SQL_READ = "sql_read"
    TABULAR_EXPORT = "tabular_export"
    PROFILE = "profile"
    RELATIONAL_UPDATE = "relational_update"
    RELATIONAL_UPSERT = "relational_upsert"


# Operations whose validation and execution depend on the type's SQL dialect.
_SQL_OPERATIONS = frozenset(SourceOperation)


@dataclass(frozen=True, slots=True)
class SourceBackendContext:
    """Host-owned dependencies a source type may use to build its backends."""

    store: SourceStore
    catalog: CatalogDataView
    secret_provider: SecretProvider
    clock: Callable[[], datetime]
    host_capacity: HostCapacityLimits
    postgresql_reads: PostgreSQLReadLimits | None
    deadline_reader: Callable[[], float | None]
    observer: AgentObserver | None


def _one_read(_: HostCapacityLimits) -> int:
    return 1


def _no_backends(_: SourceBackendContext) -> Mapping[SourceOperation, object]:
    return {}


@dataclass(frozen=True, slots=True)
class SourceType:
    """One source adapter's identity, factory and declared operations."""

    adapter_id: str
    label: str
    from_registration: Callable[..., ResourceSource]
    credential_fields: tuple[str, ...] = ()
    read_capacity: Callable[[HostCapacityLimits], int] = _one_read
    sql_dialect: str | None = None
    operations: frozenset[SourceOperation] = field(default_factory=frozenset)
    create_backends: Callable[
        [SourceBackendContext], Mapping[SourceOperation, object]
    ] = _no_backends

    def __post_init__(self) -> None:
        for text, name in ((self.adapter_id, "adapter_id"), (self.label, "label")):
            if not isinstance(text, str) or not text.strip() or len(text) > 128:
                raise ValueError(f"source type {name} must be bounded non-empty text")
        for function, name in (
            (self.from_registration, "from_registration"),
            (self.read_capacity, "read_capacity"),
            (self.create_backends, "create_backends"),
        ):
            if not callable(function):
                raise TypeError(f"source type {name} must be callable")
        if not isinstance(self.credential_fields, tuple) or any(
            not isinstance(item, str) or not item for item in self.credential_fields
        ):
            raise TypeError("source type credential_fields must be a tuple of names")
        operations = frozenset(self.operations)
        if any(not isinstance(item, SourceOperation) for item in operations):
            raise TypeError("source type operations must be SourceOperation values")
        if self.sql_dialect is not None and (
            not isinstance(self.sql_dialect, str) or not self.sql_dialect
        ):
            raise ValueError("source type sql_dialect must be non-empty text or None")
        if operations & _SQL_OPERATIONS and self.sql_dialect is None:
            raise ValueError("source type SQL operations require an sql_dialect")
        object.__setattr__(self, "operations", operations)

    def backends(
        self, context: SourceBackendContext
    ) -> Mapping[SourceOperation, object]:
        """Build exactly one backend for each declared operation."""

        created = dict(self.create_backends(context))
        if frozenset(created) != self.operations:
            raise ValueError(
                f"source type {self.adapter_id} must supply one backend per operation"
            )
        return created


class UnknownSourceTypeError(ValueError):
    """A stored or requested source type is not registered in this composition."""


class SourceTypeRegistry:
    """The composed set of source types, keyed by ``adapter_id``."""

    def __init__(self, types: Iterable[SourceType] = ()) -> None:
        self._types: dict[str, SourceType] = {}
        for source_type in (*_builtin_source_types(), *types):
            self.register(source_type)

    def register(self, source_type: SourceType) -> None:
        if not isinstance(source_type, SourceType):
            raise TypeError("source type registration requires SourceType")
        if source_type.adapter_id in self._types:
            raise ValueError(
                f"source type {source_type.adapter_id} is already registered"
            )
        self._types[source_type.adapter_id] = source_type

    def get(self, adapter_id: str) -> SourceType | None:
        return self._types.get(adapter_id)

    def require(self, adapter_id: str) -> SourceType:
        source_type = self._types.get(adapter_id)
        if source_type is None:
            raise UnknownSourceTypeError(
                f"source type {adapter_id} is not registered in this agent"
            )
        return source_type

    def types(self) -> tuple[SourceType, ...]:
        return tuple(self._types.values())

    def supports(self, adapter_id: str, operation: SourceOperation) -> bool:
        source_type = self._types.get(adapter_id)
        return source_type is not None and operation in source_type.operations

    def read_capacity(self, adapter_id: str, host_capacity: HostCapacityLimits) -> int:
        source_type = self._types.get(adapter_id)
        return 1 if source_type is None else source_type.read_capacity(host_capacity)


def _sqlite_operations(
    context: SourceBackendContext,
) -> Mapping[SourceOperation, object]:
    from .sqlite_query import SQLiteQueryBackend

    backend = SQLiteQueryBackend(context.store, context.catalog)
    return {
        SourceOperation.SQL_READ: backend,
        SourceOperation.TABULAR_EXPORT: backend,
        SourceOperation.PROFILE: backend,
    }


def _postgresql_operations(
    context: SourceBackendContext,
) -> Mapping[SourceOperation, object]:
    from .postgresql_query import PostgreSQLQueryBackend
    from .postgresql_write import PostgreSQLWriteBackend

    reads = PostgreSQLQueryBackend(
        context.store,
        context.catalog,
        context.secret_provider,
        read_limits=context.postgresql_reads,
        source_capacity=context.host_capacity.postgresql_read_capacity,
        deadline_reader=context.deadline_reader,
        observer=context.observer,
    )
    writes = PostgreSQLWriteBackend(
        context.store,
        context.catalog,
        context.secret_provider,
        clock=context.clock,
    )
    return {
        SourceOperation.SQL_READ: reads,
        SourceOperation.TABULAR_EXPORT: reads,
        SourceOperation.PROFILE: reads,
        SourceOperation.RELATIONAL_UPDATE: writes,
        SourceOperation.RELATIONAL_UPSERT: writes,
    }


def _postgresql_read_capacity(host_capacity: HostCapacityLimits) -> int:
    return host_capacity.postgresql_read_capacity


def _builtin_source_types() -> tuple[SourceType, ...]:
    from .postgresql import postgresql_source_from_registration
    from .sqlite import sqlite_source_from_registration

    # Registration order fixes declared enums such as the export adapter list.
    return (
        SourceType(
            adapter_id="sqlite",
            label="SQLite",
            from_registration=sqlite_source_from_registration,
            sql_dialect="sqlite",
            operations=frozenset(
                {
                    SourceOperation.SQL_READ,
                    SourceOperation.TABULAR_EXPORT,
                    SourceOperation.PROFILE,
                }
            ),
            create_backends=_sqlite_operations,
        ),
        SourceType(
            adapter_id="postgresql",
            label="PostgreSQL",
            from_registration=postgresql_source_from_registration,
            credential_fields=("credential_ref",),
            read_capacity=_postgresql_read_capacity,
            sql_dialect="postgresql",
            operations=frozenset(SourceOperation),
            create_backends=_postgresql_operations,
        ),
    )


__all__ = [
    "SourceBackendContext",
    "SourceOperation",
    "SourceType",
    "SourceTypeRegistry",
    "UnknownSourceTypeError",
]
