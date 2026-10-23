"""Bounded delivery retention shared by every delivery producer."""

from __future__ import annotations

from ..distribution.models import MAX_DELIVERIES_PER_AGENT, DeliveryState
from .sql_connection import SQLConnection

# Oldest first. A job's record keeps its result, so its Done delivery can go
# before an unacknowledged routine result.
_RECLAIMABLE_DELIVERIES: tuple[tuple[str, tuple[object, ...]], ...] = (
    ("state = ?", (DeliveryState.ACKNOWLEDGED.value,)),
    ("subject_kind = 'job'", ()),
)


def reserve_delivery_slot(connection: SQLConnection, agent_id: str) -> bool:
    """Make room for one delivery; return False when nothing is reclaimable."""

    count = connection.execute(
        "SELECT COUNT(*) FROM deliveries WHERE agent_id = ?", (agent_id,)
    ).fetchone()
    if count is None:
        raise RuntimeError("delivery count is unavailable")
    if int(count[0]) < MAX_DELIVERIES_PER_AGENT:
        return True
    for clause, parameters in _RECLAIMABLE_DELIVERIES:
        reclaimable = connection.execute(
            "SELECT delivery_id FROM deliveries WHERE agent_id = ? AND "
            + clause
            + " ORDER BY created_at_us, delivery_id LIMIT 1",
            (agent_id, *parameters),
        ).fetchone()
        if reclaimable is None:
            continue
        deleted = connection.execute(
            "DELETE FROM deliveries WHERE agent_id = ? AND delivery_id = ?",
            (agent_id, reclaimable[0]),
        )
        if deleted.rowcount != 1:
            raise RuntimeError("reclaimable delivery changed during reclamation")
        return True
    return False
