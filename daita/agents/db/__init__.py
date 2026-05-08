"""
daita.agents.db — database-backed Agent builder.

Public entry point:
    from daita.agents.db import from_db
"""

from .builder import from_db
from .findings import Finding, normalize_finding
from .memory import DBMemoryRecord, normalize_db_memory_record
from .monitors import MonitorDefinition, normalize_monitor_definition
from .presets import MODE_PRESETS
from .summary import build_db_summary

__all__ = [
    "from_db",
    "build_db_summary",
    "MODE_PRESETS",
    "Finding",
    "normalize_finding",
    "DBMemoryRecord",
    "normalize_db_memory_record",
    "MonitorDefinition",
    "normalize_monitor_definition",
]
