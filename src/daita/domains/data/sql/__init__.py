"""Export SQL analysis and read and write policy APIs for the data domain."""

from .analysis import (
    _load_sqlglot as _load_sqlglot,
    analyze_postgresql_sql,
    analyze_sqlite_sql,
)
from .contracts import (
    MAX_SQL_CHARACTERS,
    MAX_SQL_PARAMETERS,
    CheckedRead,
    ResourceSchema,
    SqlAnalysis,
    SqlAnalysisError,
    SqlTableReference,
    SqlValidationIssue,
    SqlValidationResult,
    WriteFacts,
    WritePreview,
    checked_reads,
    denied_relations,
    normalize_sql,
    sqlite_declared_type_affinity,
    sqlite_identifier_key,
    write_facts,
    write_permission_issue,
)
from .duckdb_read import (
    MAX_FILE_QUERY_SQL_CHARACTERS,
    DuckDBReadValidationError,
    ValidatedDuckDBRead,
    validate_duckdb_read,
)
from .read_validation import SQL_READ_DIALECTS, resource_id_in_scope, validate_read

__all__ = [
    "CheckedRead",
    "MAX_SQL_CHARACTERS",
    "MAX_SQL_PARAMETERS",
    "MAX_FILE_QUERY_SQL_CHARACTERS",
    "SQL_READ_DIALECTS",
    "DuckDBReadValidationError",
    "ResourceSchema",
    "SqlAnalysis",
    "SqlAnalysisError",
    "SqlTableReference",
    "SqlValidationIssue",
    "SqlValidationResult",
    "ValidatedDuckDBRead",
    "WriteFacts",
    "WritePreview",
    "analyze_postgresql_sql",
    "analyze_sqlite_sql",
    "checked_reads",
    "denied_relations",
    "normalize_sql",
    "resource_id_in_scope",
    "sqlite_declared_type_affinity",
    "sqlite_identifier_key",
    "validate_read",
    "validate_duckdb_read",
    "write_facts",
    "write_permission_issue",
]
