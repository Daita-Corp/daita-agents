"""Compatibility package for database-backed agents.

The public ``from daita.agents.db import from_db`` entry point now routes to
the operation-centric ``daita.db`` runtime. Legacy helper modules under this
package were removed during Phase 13 after their behavior moved to ``daita.db``.
"""

from daita.db import from_db

__all__ = ["from_db"]
