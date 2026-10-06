"""Supported adapter boundary for the shared SQL state operations.

Queries use qmark parameters; dialect-specific expressions are selected explicitly.
Only backend admission owns connections, credentials and physical schema changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from types import TracebackType
from typing import Any, Protocol


class StateIntegrityError(ValueError):
    """A relational uniqueness, reference or check constraint rejected a write."""


class SQLCursor(Protocol):
    @property
    def rowcount(self) -> int: ...

    def fetchone(self) -> tuple[Any, ...] | None: ...

    def fetchall(self) -> list[tuple[Any, ...]]: ...

    def __iter__(self) -> Iterator[tuple[Any, ...]]: ...


class SQLConnection(Protocol):
    """One bounded, worker-thread connection to an already admitted logical home.

    Use qmark parameters and tuple rows. Read connections must retain a consistent
    snapshot through context exit. ``begin(write=True)`` must serialize state
    mutations within a logical home; an adapter may add isolation and fencing.
    Context exit must settle the transaction and release its connection. Dialect
    expressions come from trusted adapter code, never caller or model input.
    """

    insertion_order: str
    distinct_operator: str

    def execute(self, query: str, parameters: Sequence[Any] = ()) -> SQLCursor: ...

    def executemany(
        self, query: str, parameters: Iterable[Sequence[Any]]
    ) -> SQLCursor: ...

    def begin(self, *, write: bool = False) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...

    def conflict(self, columns: str) -> str: ...

    def caller_expression(self, column: str) -> str: ...

    def table_exists(self, table: str) -> bool: ...

    def __enter__(self) -> SQLConnection: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class SQLDatabase(Protocol):
    """Caller-owned connection source; construction performs no recovery.

    The adapter owns admission, physical schema installation, credentials and
    resource cleanup. Normalize driver failures into the existing storage error
    taxonomy, preserving ownership loss and unknown commit outcomes. Never replay
    a mutation automatically. All connection and transport waits must be bounded.
    """

    def connect(self, *, read_only: bool = False) -> SQLConnection: ...

    def normalize_error(self, error: BaseException) -> BaseException: ...


def required_row(cursor: SQLCursor) -> tuple[Any, ...]:
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("state aggregate query returned no row")
    return row
