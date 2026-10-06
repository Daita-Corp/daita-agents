"""Physical document persistence beneath the memory and skill owners."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

AdvisoryCollection = Literal["memory", "skills", "retained-skills"]


@dataclass(frozen=True, slots=True)
class AdvisoryDocument:
    """Exact canonical bytes and an opaque, non-reusable mutation identity."""

    content: bytes
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("advisory content must be bytes")
        if (
            not isinstance(self.revision, str)
            or not self.revision
            or len(self.revision) > 256
        ):
            raise ValueError("advisory revision must contain 1 to 256 characters")


class AdvisoryTransaction(Protocol):
    def get(
        self, collection: AdvisoryCollection, name: str, *, max_bytes: int
    ) -> AdvisoryDocument | None: ...

    def scan(
        self, collection: AdvisoryCollection, *, max_count: int, max_bytes: int
    ) -> Mapping[str, AdvisoryDocument]:
        """Return the complete collection; reject overflow, never truncate.

        ``max_bytes`` bounds each document, and ``max_count`` bounds the result.
        """
        ...

    def usage(self, collection: AdvisoryCollection) -> tuple[int, int]:
        """Return the exact record count and total content bytes in this snapshot."""
        ...

    def put(
        self, collection: AdvisoryCollection, name: str, content: bytes
    ) -> None: ...

    def delete(self, collection: AdvisoryCollection, name: str) -> None: ...


class AdvisoryStorage(Protocol):
    """Borrowed persistence scoped to one admitted agent by its caller.

    Transactions run in worker threads, retain a consistent read snapshot and
    serialize writes within the agent. Exit commits or rolls back and releases
    the connection. Every changed/recreated document gets a fresh revision, even
    when its bytes match an earlier version. Read bounds must be enforced before
    materializing unbounded data. All waits must be finite.

    The caller owns admission, writer fencing, schema, credentials and transport
    lifetime. Normalize failures into ``StorageError`` subclasses; preserve
    ownership loss and unknown commits, and never replay a mutation. The existing
    stores own formats, validation, preflight, limits and retained-skill identity.
    This interface supplies no migration runner or execution lease.
    """

    def transaction(
        self, *, write: bool = False
    ) -> AbstractContextManager[AdvisoryTransaction]: ...
