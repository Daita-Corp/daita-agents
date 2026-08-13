"""Pure SQL analysis and catalog-validation contracts for the data domain.

The package facade preserves the established ``daita.domains.data.sql`` import
surface while keeping analysis, read validation, and typed update validation
in cohesive internal modules.
"""

from .analysis import (
    _load_sqlglot as _load_sqlglot,
    analyze_postgresql_sql,
    analyze_sqlite_sql,
)
from .contracts import (
    MAX_SQL_CHARACTERS,
    MAX_SQL_PARAMETERS,
    ResourceSchema,
    SqlAnalysis,
    SqlAnalysisError,
    SqlColumnReference,
    SqlTableReference,
    SqlValidationIssue,
    SqlValidationResult,
    normalize_sql,
    sqlite_declared_type_affinity,
    sqlite_identifier_key,
)
from .postgresql_update import (
    POSTGRESQL_UPDATE_MAX_ASSIGNMENTS,
    POSTGRESQL_UPDATE_MAX_CANONICAL_BYTES,
    PostgreSQLUpdateCell,
    PostgreSQLUpdateCommand,
    PostgreSQLUpdateIntent,
    PostgreSQLUpdateScopeValidationResult,
    PostgreSQLUpdateStatement,
    PostgreSQLUpdateValidationResult,
    ValidatedPostgreSQLUpdate,
    ValidatedPostgreSQLUpdateScope,
    render_postgresql_update_statement,
    validate_postgresql_update_intent,
    validate_postgresql_update_scope,
)
from .read_validation import validate_postgresql_read, validate_sqlite_read

__all__ = [
    "MAX_SQL_CHARACTERS",
    "MAX_SQL_PARAMETERS",
    "POSTGRESQL_UPDATE_MAX_ASSIGNMENTS",
    "POSTGRESQL_UPDATE_MAX_CANONICAL_BYTES",
    "PostgreSQLUpdateCell",
    "PostgreSQLUpdateCommand",
    "PostgreSQLUpdateIntent",
    "PostgreSQLUpdateScopeValidationResult",
    "PostgreSQLUpdateStatement",
    "PostgreSQLUpdateValidationResult",
    "ResourceSchema",
    "SqlAnalysis",
    "SqlAnalysisError",
    "SqlColumnReference",
    "SqlTableReference",
    "SqlValidationIssue",
    "SqlValidationResult",
    "ValidatedPostgreSQLUpdate",
    "ValidatedPostgreSQLUpdateScope",
    "analyze_postgresql_sql",
    "analyze_sqlite_sql",
    "normalize_sql",
    "render_postgresql_update_statement",
    "sqlite_declared_type_affinity",
    "sqlite_identifier_key",
    "validate_postgresql_read",
    "validate_postgresql_update_intent",
    "validate_postgresql_update_scope",
    "validate_sqlite_read",
]
