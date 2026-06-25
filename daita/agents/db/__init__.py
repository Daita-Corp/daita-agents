"""Compatibility package for database-backed agents.

The public ``from daita.agents.db import from_db`` entry point now routes to
the operation-centric ``daita.db`` runtime.
"""

from daita.db import from_db

__all__ = ["from_db"]
