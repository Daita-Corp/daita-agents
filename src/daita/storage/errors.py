"""Backend-independent failures of durable graph transactions."""


class GraphStoreConflictError(RuntimeError):
    """A graph CAS or idempotency precondition did not match."""


class GraphBudgetError(ValueError):
    """A graph budget reservation or settlement would violate conservation."""
