"""Backend-independent storage failures and graph transaction conflicts."""


class StorageError(RuntimeError):
    """Storage failure; messages may contain private driver details."""

    code = "storage_failed"


class StorageUnavailableError(StorageError):
    """Transient failure with a known non-commit; retrying the operation is safe.

    A backend must finish rollback before raising this for a mutation. Transport
    loss during COMMIT must instead raise StorageCommitUnknownError.
    """

    code = "storage_unavailable"


class StorageCommitUnknownError(StorageError):
    """A mutation may have committed; inspect durable state before proceeding."""

    code = "storage_commit_unknown"


class StorageOwnershipLostError(StorageError):
    """The agent writer is no longer current; stop work and reacquire ownership."""

    code = "storage_ownership_lost"


class GraphStoreConflictError(RuntimeError):
    """A graph CAS or idempotency precondition did not match."""


class GraphBudgetError(ValueError):
    """A graph budget reservation or settlement would violate conservation."""
