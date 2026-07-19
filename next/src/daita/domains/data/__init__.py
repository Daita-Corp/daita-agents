"""Data-domain contracts, deterministic validation, and loop projections."""

from .context import CatalogContextReader, DataContextBuilder
from .catalog import CatalogDataView
from .capabilities import (
    SQLITE_QUERY_EXECUTOR_ID,
    SQLiteQueryDeclarations,
    SQLiteQueryExecutor,
    SQLiteReadBackend,
    SQLiteReadResult,
    sqlite_query_declarations,
    sqlite_query_extension_declarations,
)
from .controller import (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_SEARCH_CAPABILITY_ID,
    SQLITE_QUERY_CAPABILITY_ID,
    SQLITE_QUERY_EVIDENCE_KIND,
    CatalogSchemaReader,
    DataDomainController,
)

from .results import BoundedResultProjection, project_result_rows
from .sql import (
    ResourceSchema,
    SqlAnalysis,
    SqlAnalysisError,
    SqlColumnReference,
    SqlTableReference,
    SqlValidationIssue,
    SqlValidationResult,
    analyze_sqlite_sql,
    normalize_sql,
    validate_sqlite_read,
)

__all__ = [
    "BoundedResultProjection",
    "CATALOG_INSPECT_CAPABILITY_ID",
    "CATALOG_SEARCH_CAPABILITY_ID",
    "CatalogContextReader",
    "CatalogDataView",
    "CatalogSchemaReader",
    "DataContextBuilder",
    "DataDomainController",
    "ResourceSchema",
    "SqlAnalysis",
    "SqlAnalysisError",
    "SqlColumnReference",
    "SqlTableReference",
    "SqlValidationIssue",
    "SqlValidationResult",
    "SQLITE_QUERY_CAPABILITY_ID",
    "SQLITE_QUERY_EVIDENCE_KIND",
    "SQLITE_QUERY_EXECUTOR_ID",
    "SQLiteQueryDeclarations",
    "SQLiteQueryExecutor",
    "SQLiteReadBackend",
    "SQLiteReadResult",
    "analyze_sqlite_sql",
    "normalize_sql",
    "project_result_rows",
    "sqlite_query_declarations",
    "sqlite_query_extension_declarations",
    "validate_sqlite_read",
]
