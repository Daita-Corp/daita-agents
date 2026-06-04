"""Canonical runtime capability IDs used by DB-facing compatibility tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DB_SCHEMA_INSPECT_CAPABILITY = "db.schema.inspect"
DB_SQL_VALIDATE_CAPABILITY = "db.sql.validate"
DB_SQL_EXECUTE_READ_CAPABILITY = "db.sql.execute_read"
DB_SQL_EXECUTE_WRITE_CAPABILITY = "db.sql.execute_write"
DB_SQL_EXPLAIN_CAPABILITY = "db.sql.explain"

CATALOG_SCHEMA_SEARCH_CAPABILITY = "catalog.schema.search"
CATALOG_ASSET_INSPECT_CAPABILITY = "catalog.asset.inspect"
CATALOG_RELATIONSHIP_PATHS_FIND_CAPABILITY = "catalog.relationship_paths.find"

MEMORY_SEMANTIC_WRITE_CAPABILITY = "memory.semantic.write"

QUALITY_PROFILE_CAPABILITY = "quality.profile"
QUALITY_ANOMALY_DETECT_CAPABILITY = "quality.anomaly.detect"
QUALITY_FRESHNESS_CHECK_CAPABILITY = "quality.freshness.check"
QUALITY_REPORT_GENERATE_CAPABILITY = "quality.report.generate"

LINEAGE_TRACE_CAPABILITY = "lineage.trace"
LINEAGE_IMPACT_ANALYZE_CAPABILITY = "lineage.impact.analyze"
LINEAGE_FLOW_REGISTER_CAPABILITY = "lineage.flow.register"
LINEAGE_PATH_FIND_CAPABILITY = "lineage.path.find"

ANALYST_CAPABILITY_PREFIX = "db.analyst."


def tool_capabilities(tool: Any) -> tuple[str, ...]:
    """Return capabilities declared by a model-visible tool adapter."""

    capability_ids = getattr(tool, "capability_ids", None)
    if capability_ids:
        return tuple(str(capability) for capability in capability_ids)
    return ()


def tool_has_any_capability(tool: Any, capabilities: Iterable[str]) -> bool:
    """Return True when a tool declares at least one requested capability."""

    requested = set(capabilities)
    return bool(requested.intersection(tool_capabilities(tool)))
