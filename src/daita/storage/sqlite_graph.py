"""Frozen import path for released local-home migrations.

Current transitions live in sql_graph. Historical migrations retain their exact
source and checksums; this adapter supplies their SQLite connection boundary.
"""

import sqlite3

from ..jobs.graph.models import GraphAdmission, GraphInspection, GraphJob
from .errors import (
    GraphBudgetError as GraphBudgetError,
    GraphStoreConflictError as GraphStoreConflictError,
)
from .sql_graph import datetime_to_us as datetime_to_us


def admit_graph(connection: sqlite3.Connection, admission: GraphAdmission) -> GraphJob:
    from .sql_graph import admit_graph as admit
    from .sqlite import _SQLiteConnection

    return admit(_SQLiteConnection(connection), admission)


def inspect_graph(
    connection: sqlite3.Connection, agent_id: str, job_id: str
) -> GraphInspection | None:
    from .sql_graph import inspect_graph as inspect
    from .sqlite import _SQLiteConnection

    return inspect(_SQLiteConnection(connection), agent_id, job_id)
