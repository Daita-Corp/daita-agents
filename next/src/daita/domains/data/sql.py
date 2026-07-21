"""Provider-neutral SQL analysis and catalog-scope validation.

This module is deliberately pure: it parses and validates immutable inputs but
does not open a source, invoke an executor, or persist runtime state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Any, Literal

from ..._json import FrozenJsonObject, canonical_json

_SQLITE_INSTALL_HINT = "pip install 'daita-agents[sqlite]'"
_POSTGRESQL_INSTALL_HINT = "pip install 'daita-agents[postgresql]'"
_SqlDialect = Literal["sqlite", "postgresql"]
_MAX_ISSUES = 32
_MAX_CANDIDATES = 8
_MAX_UPDATE_IDENTIFIER_CHARACTERS = 512
_MAX_UPDATE_COLUMN_CHARACTERS = 256
_MAX_UPDATE_VALUE_CHARACTERS = 4_096
_MAX_VALIDATION_ISSUE_DETAILS_CHARACTERS = 1_536
_ASCII_IDENTIFIER_CASE_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)

# PostgreSQL functions can perform external I/O even in a read-only
# transaction.  Keep the directly callable surface deliberately smaller than
# the server catalog, and retain a non-replay-safe capability declaration until
# durable server-owned callable/operator/type provenance exists.
_POSTGRESQL_BOUNDED_FUNCTIONS = frozenset(
    {
        "ABS",
        "AVG",
        "CAST",
        "CEIL",
        "CEILING",
        "CHAR_LENGTH",
        "COALESCE",
        "CONCAT",
        "CONCAT_WS",
        "COUNT",
        "EXTRACT",
        "FLOOR",
        "GREATEST",
        "LENGTH",
        "LEAST",
        "LOWER",
        "LTRIM",
        "MAX",
        "MIN",
        "NULLIF",
        "OCTET_LENGTH",
        "REPLACE",
        "ROW_NUMBER",
        "ROUND",
        "RTRIM",
        "SIGN",
        "SUBSTR",
        "SUBSTRING",
        "SUM",
        "TRIM",
        "UPPER",
    }
)
_POSTGRESQL_NON_DISPATCH_EXPRESSIONS = frozenset(
    {
        "CAST",
        "COALESCE",
        "GREATEST",
        "LEAST",
        "NULLIF",
    }
)
_VOLATILE_CONTEXT_EXPRESSIONS = frozenset(
    {
        "CURRENT_CATALOG",
        "CURRENT_DATE",
        "CURRENT_ROLE",
        "CURRENT_SCHEMA",
        "CURRENT_TIME",
        "CURRENT_TIMESTAMP",
        "CURRENT_USER",
        "LOCALTIME",
        "LOCALTIMESTAMP",
        "SESSION_USER",
        "SYSTEM_USER",
        "USER",
    }
)
_POSTGRESQL_SAFE_DATA_TYPES = frozenset(
    {
        "ARRAY",
        "BIGINT",
        "BINARY",
        "BIT",
        "BOOLEAN",
        "BPCHAR",
        "BYTEA",
        "CHAR",
        "CIDR",
        "DATE",
        "DECIMAL",
        "DOUBLE",
        "FLOAT",
        "INET",
        "INT",
        "INTERVAL",
        "JSON",
        "JSONB",
        "MACADDR",
        "MONEY",
        "REAL",
        "SMALLINT",
        "TEXT",
        "TIME",
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "TIMETZ",
        "TINYINT",
        "UUID",
        "VARBINARY",
        "VARCHAR",
    }
)


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _identifier_key(value: str) -> str:
    return value.strip().strip('"`[]').casefold()


def sqlite_identifier_key(value: str) -> str:
    """Match SQLite identifier case without folding distinct Unicode names."""

    return value.strip().strip('"`[]').translate(_ASCII_IDENTIFIER_CASE_TRANSLATION)


def _dialect_identifier_key(
    value: str,
    *,
    dialect: _SqlDialect,
    quoted: bool = False,
) -> str:
    normalized = value.strip().strip('"`[]')
    if dialect == "postgresql" and quoted:
        return f"quoted:{normalized}"
    if dialect == "postgresql":
        return normalized.translate(_ASCII_IDENTIFIER_CASE_TRANSLATION)
    return _identifier_key(normalized)


def _short_identifier(value: str) -> str:
    return _identifier_key(value).rsplit(".", 1)[-1]


def sqlite_declared_type_affinity(declared_type: str) -> str:
    """Return SQLite's canonical affinity for one declared column type."""

    if not isinstance(declared_type, str):
        raise TypeError("declared_type must be a string")
    normalized = declared_type.strip().upper()
    if "INT" in normalized:
        return "integer"
    if any(token in normalized for token in ("CHAR", "CLOB", "TEXT")):
        return "text"
    if not normalized or "BLOB" in normalized:
        return "blob"
    if any(token in normalized for token in ("REAL", "FLOA", "DOUB")):
        return "real"
    return "numeric"


def normalize_sql(sql: str) -> str:
    """Normalize one model-proposed statement without rewriting its contents."""

    if not isinstance(sql, str):
        raise TypeError("sql must be a string")
    trimmed = sql.rstrip()
    if trimmed.endswith(";"):
        without_terminator = trimmed[:-1].rstrip()
        if not without_terminator.endswith(";"):
            return without_terminator
    return trimmed


class SqlAnalysisError(ValueError):
    """A stable, bounded parser failure safe to expose as repair context."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = _required_text(code, "analysis error code")


@dataclass(frozen=True, slots=True)
class ResourceSchema:
    """One catalog-owned tabular resource projection used for validation."""

    resource_id: str
    source_id: str
    name: str
    columns: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    revision: str | None = field(default=None, compare=False)
    source_revision: str | None = field(default=None, compare=False)
    resource_kind: str | None = field(default=None, compare=False)
    sensitivity_class: str = field(default="unknown", compare=False)
    writable: bool = field(default=False, compare=False)
    unique_key_columns: tuple[str, ...] = field(default=(), compare=False)
    column_declared_types: tuple[tuple[str, str], ...] = field(
        default=(),
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_id", _required_text(self.resource_id, "resource_id")
        )
        object.__setattr__(
            self, "source_id", _required_text(self.source_id, "source_id")
        )
        object.__setattr__(self, "name", _required_text(self.name, "resource name"))
        columns = tuple(
            _required_text(item, "resource column") for item in self.columns
        )
        if len(set(columns)) != len(columns):
            raise ValueError("resource columns must be unique")
        aliases = tuple(_required_text(item, "resource alias") for item in self.aliases)
        if self.revision is not None:
            revision = _required_text(self.revision, "resource revision")
            digest = revision.removeprefix("sha256:")
            if (
                len(revision) != 71
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("resource revision must be a sha256 hash")
        if self.source_revision is not None:
            source_revision = _required_text(
                self.source_revision,
                "resource source_revision",
            )
            if len(source_revision) > 1_024:
                raise ValueError("resource source_revision exceeds 1024 characters")
        if self.resource_kind is not None:
            resource_kind = _required_text(
                self.resource_kind,
                "resource resource_kind",
            )
            if len(resource_kind) > 128:
                raise ValueError("resource resource_kind exceeds 128 characters")
            object.__setattr__(self, "resource_kind", resource_kind.casefold())
        sensitivity_class = _required_text(
            self.sensitivity_class,
            "resource sensitivity_class",
        )
        if len(sensitivity_class) > 128:
            raise ValueError("resource sensitivity_class exceeds 128 characters")
        if not isinstance(self.writable, bool):
            raise TypeError("resource writable must be a boolean")
        unique_key_columns = tuple(
            _required_text(item, "resource unique key column")
            for item in self.unique_key_columns
        )
        if len(set(unique_key_columns)) != len(unique_key_columns):
            raise ValueError("resource unique key columns must be unique")
        if any(item not in set(columns) for item in unique_key_columns):
            raise ValueError("resource unique key columns must exist in columns")
        if isinstance(self.column_declared_types, (str, bytes)):
            raise TypeError("resource column_declared_types must be a sequence")
        raw_column_declared_types = tuple(self.column_declared_types)
        if any(
            not isinstance(item, (tuple, list)) or len(item) != 2
            for item in raw_column_declared_types
        ):
            raise ValueError(
                "resource column_declared_types must contain column/type pairs"
            )
        column_declared_types = tuple(
            (item[0], item[1]) for item in raw_column_declared_types
        )
        declared_type_by_column: dict[str, str] = {}
        for column, declared_type in column_declared_types:
            canonical_column = _required_text(
                column,
                "resource declared type column",
            )
            if canonical_column not in set(columns):
                raise ValueError("resource declared type columns must exist in columns")
            if canonical_column in declared_type_by_column:
                raise ValueError("resource declared type columns must be unique")
            if not isinstance(declared_type, str) or len(declared_type) > 256:
                raise ValueError(
                    "resource declared column types must be bounded strings"
                )
            declared_type_by_column[canonical_column] = declared_type
        canonical_declared_types = tuple(
            (column, declared_type_by_column[column])
            for column in columns
            if column in declared_type_by_column
        )
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "sensitivity_class", sensitivity_class.casefold())
        object.__setattr__(self, "unique_key_columns", unique_key_columns)
        object.__setattr__(
            self,
            "column_declared_types",
            canonical_declared_types,
        )

    @property
    def lookup_names(self) -> frozenset[str]:
        values: set[str] = set()
        for item in (self.name, *self.aliases):
            values.add(_identifier_key(item))
            values.add(_short_identifier(item))
        return frozenset(values)

    @property
    def column_keys(self) -> frozenset[str]:
        return frozenset(_identifier_key(item) for item in self.columns)


@dataclass(frozen=True, slots=True)
class SqlTableReference:
    name: str
    qualified_name: str
    alias: str | None = None
    is_cte: bool = False
    name_quoted: bool = False
    qualified_parts: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "table name"))
        object.__setattr__(
            self,
            "qualified_name",
            _required_text(self.qualified_name, "qualified table name"),
        )
        if self.alias is not None:
            object.__setattr__(self, "alias", _required_text(self.alias, "table alias"))
        if not isinstance(self.is_cte, bool):
            raise TypeError("table is_cte must be a boolean")
        if not isinstance(self.name_quoted, bool):
            raise TypeError("table name_quoted must be a boolean")
        parts = tuple(self.qualified_parts)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0].strip()
            or not isinstance(item[1], bool)
            for item in parts
        ):
            raise TypeError("table qualified_parts must contain name/quoted pairs")
        object.__setattr__(self, "qualified_parts", parts)


@dataclass(frozen=True, slots=True)
class SqlColumnReference:
    name: str
    qualifier: str | None = None
    resource_name: str | None = None
    name_quoted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "column name"))
        for field_name in ("qualifier", "resource_name"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _required_text(value, f"column {field_name}"),
                )
        if not isinstance(self.name_quoted, bool):
            raise TypeError("column name_quoted must be a boolean")


@dataclass(frozen=True, slots=True)
class SqlAnalysis:
    sql: str
    canonical_sql: str
    sql_fingerprint: str
    statement_type: str
    statement_count: int
    is_read: bool
    has_limit: bool
    mutation_types: tuple[str, ...]
    tables: tuple[SqlTableReference, ...]
    columns: tuple[SqlColumnReference, ...]
    selected_columns: tuple[str, ...]
    select_aliases: tuple[str, ...]
    positional_parameter_count: int
    parameter_ordinals: tuple[int, ...] = ()
    anonymous_parameter_count: int = 0
    invalid_parameter_count: int = 0
    function_names: tuple[str, ...] = ()
    unresolved_function_names: tuple[str, ...] = ()
    table_function_names: tuple[str, ...] = ()
    cast_type_names: tuple[str, ...] = ()
    unsafe_cast_type_names: tuple[str, ...] = ()
    explicit_operator_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sql", _required_text(self.sql, "analysis sql"))
        object.__setattr__(
            self,
            "canonical_sql",
            _required_text(self.canonical_sql, "analysis canonical_sql"),
        )
        if not self.sql_fingerprint.startswith("sha256:"):
            raise ValueError("sql_fingerprint must use sha256")
        _required_text(self.statement_type, "analysis statement_type")
        if self.statement_count < 1:
            raise ValueError("analysis statement_count must be positive")
        if self.positional_parameter_count < 0:
            raise ValueError("positional_parameter_count must be non-negative")
        if self.anonymous_parameter_count < 0:
            raise ValueError("anonymous_parameter_count must be non-negative")
        if self.invalid_parameter_count < 0:
            raise ValueError("invalid_parameter_count must be non-negative")
        ordinals = tuple(self.parameter_ordinals)
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in ordinals
        ):
            raise ValueError("parameter_ordinals must contain positive integers")
        if ordinals != tuple(sorted(set(ordinals))):
            raise ValueError("parameter_ordinals must be sorted and unique")
        for field_name in (
            "function_names",
            "unresolved_function_names",
            "table_function_names",
            "cast_type_names",
            "unsafe_cast_type_names",
            "explicit_operator_names",
        ):
            values = tuple(getattr(self, field_name))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must contain sorted unique names")
            if any(
                not isinstance(item, str) or not item or len(item) > 256
                for item in values
            ):
                raise ValueError(f"{field_name} must contain bounded strings")
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "parameter_ordinals", ordinals)


@dataclass(frozen=True, slots=True)
class SqlValidationIssue:
    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = _required_text(self.code, "validation issue code")
        message = _required_text(self.message, "validation issue message")
        if len(code) > 96 or len(message) > 320:
            raise ValueError("validation issue text must be bounded")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        details = FrozenJsonObject.from_mapping(self.details)
        if len(canonical_json(details)) > _MAX_VALIDATION_ISSUE_DETAILS_CHARACTERS:
            raise ValueError("validation issue details must be bounded")
        object.__setattr__(self, "details", details)


@dataclass(frozen=True, slots=True)
class SqlValidationResult:
    valid: bool
    source_id: str
    analysis: SqlAnalysis | None
    resource_ids: tuple[str, ...]
    resource_revisions: tuple[tuple[str, str], ...]
    source_revision: str | None
    issues: tuple[SqlValidationIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("validation valid must be a boolean")
        object.__setattr__(
            self, "source_id", _required_text(self.source_id, "validation source_id")
        )
        resource_ids = tuple(self.resource_ids)
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("validation resource_ids must be unique")
        resource_revisions = tuple(
            sorted(tuple(item) for item in self.resource_revisions)
        )
        if any(
            len(item) != 2
            or not all(isinstance(value, str) and value.strip() for value in item)
            for item in resource_revisions
        ):
            raise ValueError(
                "validation resource_revisions must contain identifier/revision pairs"
            )
        revision_ids = tuple(item[0] for item in resource_revisions)
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("validation resource_revisions must be unique")
        if resource_revisions and set(revision_ids) != set(resource_ids):
            raise ValueError(
                "validation resource_revisions must cover every resolved resource"
            )
        if self.source_revision is not None:
            source_revision = _required_text(
                self.source_revision,
                "validation source_revision",
            )
            if len(source_revision) > 1_024:
                raise ValueError("validation source_revision exceeds 1024 characters")
        if self.valid != (self.analysis is not None and not self.issues):
            raise ValueError("validation validity must agree with analysis and issues")
        object.__setattr__(self, "resource_ids", resource_ids)
        object.__setattr__(self, "resource_revisions", resource_revisions)
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


@dataclass(frozen=True, slots=True)
class _LexicalRelation:
    """One relation visible inside one sqlglot lexical scope."""

    qualifier: str
    columns: tuple[str, ...]
    lineage: tuple[str, ...]
    kind: Literal["base", "cte", "subquery", "set"]
    scope_id: str


@dataclass(frozen=True, slots=True)
class _LexicalScope:
    """The projected schema and local environment for one sqlglot scope."""

    columns: tuple[str, ...]
    lineage: tuple[str, ...]
    relations: tuple[_LexicalRelation, ...]
    scope_id: str


@dataclass(frozen=True, slots=True)
class SQLiteUpdateRecipe:
    """One catalog-grounded, single-row conditional SQLite update recipe."""

    source_id: str
    resource_id: str
    resource_revision: str
    source_revision: str
    table_name: str
    key_column: str
    target_column: str
    key_value: str
    expected_value: str
    new_value: str
    sensitivity_class: str
    recipe_fingerprint: str

    def __post_init__(self) -> None:
        for value, field_name, maximum in (
            (self.source_id, "recipe source_id", _MAX_UPDATE_IDENTIFIER_CHARACTERS),
            (
                self.resource_id,
                "recipe resource_id",
                _MAX_UPDATE_IDENTIFIER_CHARACTERS,
            ),
            (self.table_name, "recipe table_name", _MAX_UPDATE_COLUMN_CHARACTERS),
            (self.key_column, "recipe key_column", _MAX_UPDATE_COLUMN_CHARACTERS),
            (
                self.target_column,
                "recipe target_column",
                _MAX_UPDATE_COLUMN_CHARACTERS,
            ),
            (
                self.sensitivity_class,
                "recipe sensitivity_class",
                128,
            ),
        ):
            normalized = _required_text(value, field_name)
            if len(normalized) > maximum:
                raise ValueError(f"{field_name} exceeds {maximum} characters")
        for value, field_name in (
            (self.key_value, "recipe key_value"),
            (self.expected_value, "recipe expected_value"),
            (self.new_value, "recipe new_value"),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if len(value) > _MAX_UPDATE_VALUE_CHARACTERS:
                raise ValueError(
                    f"{field_name} exceeds {_MAX_UPDATE_VALUE_CHARACTERS} characters"
                )
        for value, field_name in (
            (self.resource_revision, "recipe resource_revision"),
            (self.recipe_fingerprint, "recipe recipe_fingerprint"),
        ):
            digest = value.removeprefix("sha256:") if isinstance(value, str) else ""
            if (
                not isinstance(value, str)
                or not value.startswith("sha256:")
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{field_name} must be a canonical sha256 hash")
        source_revision = _required_text(
            self.source_revision,
            "recipe source_revision",
        )
        if len(source_revision) > 1_024:
            raise ValueError("recipe source_revision exceeds 1024 characters")
        if sqlite_identifier_key(self.key_column) == sqlite_identifier_key(
            self.target_column
        ):
            raise ValueError("recipe key and target columns must be distinct")
        if self.expected_value == self.new_value:
            raise ValueError("recipe update must not be a no-op")
        expected_fingerprint = _sqlite_update_fingerprint(
            source_id=self.source_id,
            resource_id=self.resource_id,
            resource_revision=self.resource_revision,
            source_revision=self.source_revision,
            table_name=self.table_name,
            key_column=self.key_column,
            target_column=self.target_column,
            key_value=self.key_value,
            expected_value=self.expected_value,
            new_value=self.new_value,
        )
        if self.recipe_fingerprint != expected_fingerprint:
            raise ValueError("recipe_fingerprint does not match the update recipe")


@dataclass(frozen=True, slots=True)
class SQLiteUpdateValidationResult:
    valid: bool
    source_id: str
    recipe: SQLiteUpdateRecipe | None
    issues: tuple[SqlValidationIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("update validation valid must be a boolean")
        object.__setattr__(
            self,
            "source_id",
            _required_text(self.source_id, "update validation source_id"),
        )
        if self.recipe is not None and not isinstance(self.recipe, SQLiteUpdateRecipe):
            raise TypeError(
                "update validation recipe must be SQLiteUpdateRecipe or None"
            )
        issues = tuple(self.issues)
        if any(not isinstance(issue, SqlValidationIssue) for issue in issues):
            raise TypeError(
                "update validation issues must contain SqlValidationIssue records"
            )
        if self.valid != (self.recipe is not None and not issues):
            raise ValueError(
                "update validation validity must agree with recipe and issues"
            )
        object.__setattr__(self, "issues", issues)

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


def _load_sqlglot(dialect: _SqlDialect = "sqlite") -> tuple[Any, Any]:
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as error:
        if dialect == "postgresql":
            raise ImportError(
                "sqlglot is required for PostgreSQL SQL validation. "
                f"Install with: {_POSTGRESQL_INSTALL_HINT}"
            ) from error
        raise ImportError(
            "sqlglot is required for SQLite SQL validation. "
            f"Install with: {_SQLITE_INSTALL_HINT}"
        ) from error
    return sqlglot, exp


def _explain_prefix(sql: str) -> tuple[str, str] | None:
    import re

    match = re.match(
        r"(?is)^\s*(EXPLAIN(?:\s+QUERY\s+PLAN|\s+ANALYZE)?)\s+(.+)$",
        sql,
    )
    if match is None:
        return None
    return " ".join(match.group(1).upper().split()), match.group(2).strip()


def analyze_sqlite_sql(sql: str) -> SqlAnalysis:
    """Parse SQLite SQL into immutable, source-independent semantic facts."""

    return _analyze_sql(sql, dialect="sqlite")


def analyze_postgresql_sql(sql: str) -> SqlAnalysis:
    """Parse PostgreSQL SQL into immutable, source-independent semantic facts."""

    return _analyze_sql(sql, dialect="postgresql")


def _analyze_sql(sql: str, *, dialect: _SqlDialect) -> SqlAnalysis:
    display_name = "PostgreSQL" if dialect == "postgresql" else "SQLite"
    sqlglot_dialect = "postgres" if dialect == "postgresql" else "sqlite"

    normalized = normalize_sql(sql)
    if not normalized:
        raise SqlAnalysisError("empty_sql", "SQL must not be empty.")

    sqlglot, exp = _load_sqlglot(dialect)
    explain = _explain_prefix(normalized)
    parse_sql = explain[1] if explain is not None else normalized
    try:
        parsed = tuple(sqlglot.parse(parse_sql, read=sqlglot_dialect))
        expressions = tuple(item for item in parsed if item is not None)
    except Exception as error:
        sqlglot_errors = getattr(sqlglot, "errors", None)
        normalized_error_types = tuple(
            error_type
            for error_type in (
                getattr(sqlglot_errors, "ParseError", None),
                getattr(sqlglot_errors, "TokenError", None),
            )
            if isinstance(error_type, type)
        )
        if not normalized_error_types or not isinstance(error, normalized_error_types):
            raise
        raise SqlAnalysisError(
            "sql_parse_error",
            f"SQL could not be parsed for {display_name}.",
        ) from error
    if not expressions:
        raise SqlAnalysisError("empty_sql", "SQL must not be empty.")

    root = expressions[0]
    if any(bool(item.args.get("recursive", False)) for item in root.find_all(exp.With)):
        raise SqlAnalysisError(
            "recursive_cte_not_supported",
            "Recursive common-table expressions are not supported.",
        )
    try:
        cte_table_ids, column_resources = _lexical_scope_facts(
            root,
            exp,
            dialect=dialect,
        )
    except Exception as error:
        errors = getattr(sqlglot, "errors", None)
        optimize_error = getattr(errors, "OptimizeError", None)
        if optimize_error is None or not isinstance(error, optimize_error):
            raise
        raise SqlAnalysisError(
            "sql_scope_error",
            f"SQL scope could not be resolved safely for {display_name}.",
        ) from error
    tables = _table_references(
        root,
        exp,
        cte_table_ids,
        dialect=dialect,
    )
    columns = _column_references(
        root,
        exp,
        column_resources,
        dialect=dialect,
    )
    if dialect == "postgresql":
        # PostgreSQL includes inheritance/partition descendants unless ONLY is
        # present.  The catalog provenance names one exact relation, so execute
        # the exact relation represented by that provenance.
        for table in root.find_all(exp.Table):
            if id(table) not in cte_table_ids and _table_identifier_parts(table):
                table.set("only", True)
    function_names, unresolved_functions, table_functions = _function_facts(
        root,
        exp,
        dialect=dialect,
    )
    cast_types, unsafe_cast_types, explicit_operators = _expression_boundary_facts(
        root,
        exp,
        dialect=dialect,
    )
    mutation_types = _mutation_types(root, exp)
    read_roots = tuple(
        cls
        for cls in (
            getattr(exp, "Select", None),
            getattr(exp, "Union", None),
            getattr(exp, "Intersect", None),
            getattr(exp, "Except", None),
        )
        if cls is not None
    )
    is_read = isinstance(root, read_roots) and not mutation_types
    canonical_inner = "; ".join(
        item.sql(dialect=sqlglot_dialect, pretty=False) for item in expressions
    )
    if len(parsed) != len(expressions):
        # Preserve empty statements in the canonical value so repeated trailing
        # delimiters cannot share a fingerprint with one valid statement.
        canonical_inner = parse_sql
    canonical_sql = (
        f"{explain[0]} {canonical_inner}" if explain is not None else canonical_inner
    )
    fingerprint = "sha256:" + hashlib.sha256(canonical_sql.encode("utf-8")).hexdigest()
    placeholder_count = sum(
        1 for item in root.walk() if isinstance(item, exp.Placeholder)
    )
    parameter_nodes = tuple(
        item for item in root.walk() if isinstance(item, exp.Parameter)
    )
    parameter_ordinals = tuple(
        sorted(
            {
                int(item.name)
                for item in parameter_nodes
                if str(item.name).isascii()
                and str(item.name).isdecimal()
                and int(item.name) > 0
            }
        )
    )
    invalid_parameter_count = sum(
        1
        for item in parameter_nodes
        if not str(item.name).isascii()
        or not str(item.name).isdecimal()
        or int(item.name) < 1
    )
    positional_parameter_count = (
        max(parameter_ordinals, default=0)
        if dialect == "postgresql"
        else placeholder_count
    )
    selected_columns, select_aliases = _selection_facts(root, exp)
    return SqlAnalysis(
        sql=normalized,
        canonical_sql=canonical_sql,
        sql_fingerprint=fingerprint,
        statement_type=(
            "explain" if explain is not None else type(root).__name__.lower()
        ),
        statement_count=len(parsed),
        is_read=is_read,
        has_limit=root.find(exp.Limit) is not None,
        mutation_types=mutation_types,
        tables=tables,
        columns=columns,
        selected_columns=selected_columns,
        select_aliases=select_aliases,
        positional_parameter_count=positional_parameter_count,
        parameter_ordinals=parameter_ordinals,
        anonymous_parameter_count=placeholder_count,
        invalid_parameter_count=invalid_parameter_count,
        function_names=function_names,
        unresolved_function_names=unresolved_functions,
        table_function_names=table_functions,
        cast_type_names=cast_types,
        unsafe_cast_type_names=unsafe_cast_types,
        explicit_operator_names=explicit_operators,
    )


def _table_references(
    root: Any,
    exp: Any,
    cte_table_ids: set[int],
    *,
    dialect: _SqlDialect,
) -> tuple[SqlTableReference, ...]:
    references: list[SqlTableReference] = []
    seen: set[tuple[tuple[str, ...], str | None, bool]] = set()
    for table in root.find_all(exp.Table):
        part_records = tuple(
            (str(part.name), bool(part.args.get("quoted", False)))
            for part in table.parts
            if str(part.name).strip()
        )
        parts = tuple(item[0] for item in part_records)
        name = str(table.name or "").strip()
        if not name:
            continue
        qualified = ".".join(parts) or name
        alias = str(table.alias or "").strip() or None
        name_identifier = table.this
        name_quoted = bool(getattr(name_identifier, "args", {}).get("quoted", False))
        is_cte = id(table) in cte_table_ids
        # PostgreSQL can expose both ``foo`` and the distinct quoted ``"Foo"``.
        # Preserve each component's quote semantics while deduplicating so one
        # scoped table can never hide another case-distinct table reference.
        qualified_key = tuple(
            _dialect_identifier_key(name, dialect=dialect, quoted=quoted)
            for name, quoted in part_records
        )
        alias_node = table.args.get("alias")
        alias_identifier = getattr(alias_node, "this", None)
        alias_quoted = bool(getattr(alias_identifier, "args", {}).get("quoted", False))
        key = (
            qualified_key,
            (
                _dialect_identifier_key(
                    alias,
                    dialect=dialect,
                    quoted=alias_quoted,
                )
                if alias
                else None
            ),
            is_cte,
        )
        if key in seen:
            continue
        seen.add(key)
        references.append(
            SqlTableReference(
                name=name,
                qualified_name=qualified,
                alias=alias,
                is_cte=is_cte,
                name_quoted=name_quoted,
                qualified_parts=part_records,
            )
        )
    return tuple(references)


def _table_identifier_parts(table: Any) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (str(part.name), bool(part.args.get("quoted", False)))
        for part in table.parts
        if str(part.name).strip()
    )


def _lexical_scope_facts(
    root: Any,
    exp: Any,
    *,
    dialect: _SqlDialect,
) -> tuple[set[int], dict[int, str]]:
    """Resolve CTEs and qualified columns using sqlglot's lexical scopes."""

    from sqlglot.optimizer.scope import traverse_scope

    cte_table_ids: set[int] = set()
    column_resources: dict[int, str] = {}
    for scope in traverse_scope(root):
        for _, (selected, source) in scope.selected_sources.items():
            if isinstance(selected, exp.Table) and not isinstance(source, exp.Table):
                cte_table_ids.add(id(selected))
        for column in scope.columns:
            qualifier = str(column.table or "").strip()
            if not qualifier:
                continue
            resolved_source = scope.sources.get(qualifier)
            if not isinstance(resolved_source, exp.Table):
                continue
            parts = _table_identifier_parts(resolved_source)
            if parts:
                column_resources[id(column)] = ".".join(item[0] for item in parts)
    # sqlglot's scope graph intentionally omits some dialect-invalid shapes,
    # including a mutation nested in a CTE.  Mutation detection must still see
    # the CTE reference accurately, so fill only those unresolved identities
    # from the AST's lexical WITH ancestry.
    for table in root.find_all(exp.Table):
        if id(table) not in cte_table_ids and _is_visible_cte_reference(
            table,
            exp,
            dialect=dialect,
        ):
            cte_table_ids.add(id(table))
    return cte_table_ids, column_resources


def _is_visible_cte_reference(
    table: Any,
    exp: Any,
    *,
    dialect: _SqlDialect,
) -> bool:
    name = str(table.name or "").strip()
    # A CTE reference is one unqualified identifier.  ``schema.name`` always
    # denotes a base relation even when a visible CTE has the same short name.
    if not name or len(_table_identifier_parts(table)) != 1:
        return False
    identifier = table.this
    table_key = _dialect_identifier_key(
        name,
        dialect=dialect,
        quoted=bool(getattr(identifier, "args", {}).get("quoted", False)),
    )
    ancestor = table.parent
    while ancestor is not None:
        with_clause = ancestor.args.get("with_")
        if isinstance(with_clause, exp.With):
            ctes = tuple(with_clause.expressions)
            containing_index = next(
                (index for index, cte in enumerate(ctes) if _is_ancestor(cte, table)),
                None,
            )
            visible = (
                ctes
                if containing_index is None
                else ctes[
                    : containing_index
                    + int(bool(with_clause.args.get("recursive", False)))
                ]
            )
            if any(_cte_key(cte, dialect=dialect) == table_key for cte in visible):
                return True
        ancestor = ancestor.parent
    return False


def _is_ancestor(candidate: Any, node: Any) -> bool:
    current = node.parent
    while current is not None:
        if current is candidate:
            return True
        current = current.parent
    return False


def _cte_key(cte: Any, *, dialect: _SqlDialect) -> str:
    alias = cte.args.get("alias")
    identifier = getattr(alias, "this", None)
    return _dialect_identifier_key(
        str(cte.alias_or_name),
        dialect=dialect,
        quoted=bool(getattr(identifier, "args", {}).get("quoted", False)),
    )


def _function_facts(
    root: Any,
    exp: Any,
    *,
    dialect: _SqlDialect,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    functions: set[str] = set()
    unresolved: set[str] = set()
    table_functions: set[str] = set()
    anonymous_type = getattr(exp, "Anonymous", ())
    for function in root.find_all(exp.Func):
        is_anonymous = isinstance(function, anonymous_type)
        raw_name = (
            str(function.name or "") if is_anonymous else str(function.sql_name() or "")
        )
        name = (raw_name.strip() or type(function).__name__).upper()[:256]
        # sqlglot also models structural SQL (CASE branches, EXISTS, CAST, and
        # similar grammar nodes) as Func subclasses.  Parsed call tokens carry
        # source metadata; structural nodes do not.  Record actual dispatch and
        # context-sensitive keyword expressions, not every Func-shaped node.
        if (
            not is_anonymous
            and not function.meta
            and name not in _VOLATILE_CONTEXT_EXPRESSIONS
            and name not in _POSTGRESQL_NON_DISPATCH_EXPRESSIONS
        ):
            continue
        namespace = _function_namespace(function, exp)
        if namespace is not None:
            name = f"{namespace.upper()[:128]}.{name}"[:256]
        functions.add(name)
        if dialect == "postgresql":
            # The PostgreSQL executor fixes search_path to pg_catalog and
            # requires every data relation to be schema-qualified.  Therefore
            # unqualified calls resolve only against pg_catalog; an explicit
            # non-pg_catalog namespace is never admitted.
            if namespace is not None and not name.startswith("PG_CATALOG."):
                unresolved.add(name)
        elif is_anonymous:
            unresolved.add(name)
        if _is_table_function(function, exp):
            table_functions.add(name)
    return (
        tuple(sorted(functions)),
        tuple(sorted(unresolved)),
        tuple(sorted(table_functions)),
    )


def _function_namespace(function: Any, exp: Any) -> str | None:
    parent = function.parent
    if not isinstance(parent, exp.Dot) or parent.expression is not function:
        return None
    namespace = parent.this
    if not isinstance(namespace, exp.Identifier):
        return "invalid"
    value = str(namespace.name or "").strip()
    if not value:
        return "invalid"
    if bool(namespace.args.get("quoted", False)) and value != "pg_catalog":
        return f"quoted:{value}"
    return value


def _expression_boundary_facts(
    root: Any,
    exp: Any,
    *,
    dialect: _SqlDialect,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if dialect != "postgresql":
        return (), (), ()

    cast_types: set[str] = set()
    unsafe_cast_types: set[str] = set()
    for cast in root.find_all(exp.Cast):
        target = cast.args.get("to")
        if not isinstance(target, exp.DataType):
            unsafe_cast_types.add("INVALID")
            continue
        rendered = target.sql(dialect="postgres")[:256].upper()
        cast_types.add(rendered)
        data_types = (target, *tuple(target.find_all(exp.DataType)))
        for data_type in data_types:
            raw_kind = getattr(data_type.this, "value", data_type.this)
            if str(raw_kind).upper() not in _POSTGRESQL_SAFE_DATA_TYPES:
                unsafe_cast_types.add(rendered)
                break

    operator_type = getattr(exp, "Operator", None)
    explicit_operators = (
        ()
        if operator_type is None
        else tuple(
            sorted(
                {
                    str(operator.args.get("operator") or "invalid").upper()[:256]
                    for operator in root.find_all(operator_type)
                }
            )
        )
    )
    return (
        tuple(sorted(cast_types)),
        tuple(sorted(unsafe_cast_types)),
        explicit_operators,
    )


def _is_table_function(function: Any, exp: Any) -> bool:
    current = function.parent
    while current is not None and not isinstance(current, exp.Select):
        if isinstance(
            current,
            tuple(
                item
                for item in (
                    getattr(exp, "From", None),
                    getattr(exp, "Join", None),
                    getattr(exp, "Lateral", None),
                    getattr(exp, "Table", None),
                )
                if item is not None
            ),
        ):
            return True
        current = current.parent
    return False


def _column_references(
    root: Any,
    exp: Any,
    column_resources: Mapping[int, str],
    *,
    dialect: _SqlDialect,
) -> tuple[SqlColumnReference, ...]:
    references: list[SqlColumnReference] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for column in root.find_all(exp.Column):
        name = str(column.name or "").strip()
        if not name or name == "*":
            continue
        qualifier = str(column.table or "").strip() or None
        resource_name = column_resources.get(id(column))
        name_identifier = column.this
        name_quoted = bool(getattr(name_identifier, "args", {}).get("quoted", False))
        key = (
            _dialect_identifier_key(name, dialect=dialect, quoted=name_quoted),
            _dialect_identifier_key(qualifier, dialect=dialect) if qualifier else None,
            (
                _dialect_identifier_key(resource_name, dialect=dialect)
                if resource_name
                else None
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        references.append(
            SqlColumnReference(
                name=name,
                qualifier=qualifier,
                resource_name=resource_name,
                name_quoted=name_quoted,
            )
        )
    return tuple(references)


def _selection_facts(root: Any, exp: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    select = root if isinstance(root, exp.Select) else root.find(exp.Select)
    if select is None:
        return (), ()
    names: list[str] = []
    aliases: list[str] = []
    for item in select.expressions:
        alias = str(item.alias_or_name or "").strip()
        if alias and alias != "*" and alias not in names:
            names.append(alias)
        explicit_alias = str(item.alias or "").strip()
        if explicit_alias and explicit_alias not in aliases:
            aliases.append(explicit_alias)
    return tuple(names), tuple(aliases)


def _mutation_types(root: Any, exp: Any) -> tuple[str, ...]:
    classes = tuple(
        cls
        for cls in (
            getattr(exp, "Insert", None),
            getattr(exp, "Update", None),
            getattr(exp, "Delete", None),
            getattr(exp, "Merge", None),
            getattr(exp, "Create", None),
            getattr(exp, "Drop", None),
            getattr(exp, "Alter", None),
            getattr(exp, "TruncateTable", None),
            getattr(exp, "Attach", None),
            getattr(exp, "Detach", None),
            getattr(exp, "Into", None),
            # PostgreSQL row-locking clauses are attached to otherwise read-shaped
            # SELECT nodes.  Treat them as mutations so validation rejects them
            # before opening a connector, even though the backend also runs a
            # read-only transaction as defense in depth.
            getattr(exp, "Lock", None),
        )
        if cls is not None
    )
    return tuple(
        sorted(
            {
                type(item).__name__.upper()
                for item in root.walk()
                if isinstance(item, classes)
            }
        )
    )


def _semantic_identifier(
    value: str,
    *,
    dialect: _SqlDialect,
    quoted: bool = False,
) -> str:
    """Return the identifier spelling exposed by the selected SQL dialect."""

    normalized = value.strip().strip('"`[]')
    if dialect == "postgresql" and not quoted:
        return normalized.translate(_ASCII_IDENTIFIER_CASE_TRANSLATION)
    if dialect == "sqlite":
        return sqlite_identifier_key(normalized)
    return normalized


def _selected_source_qualifier(
    selected: Any,
    fallback: str,
    exp: Any,
    *,
    dialect: _SqlDialect,
) -> str:
    alias = selected.args.get("alias") if hasattr(selected, "args") else None
    identifier = getattr(alias, "this", None)
    if identifier is None and isinstance(selected, exp.Table):
        identifier = selected.this
    if identifier is None:
        parent = getattr(selected, "parent", None)
        if parent is not None and isinstance(parent, exp.Subquery):
            parent_alias = parent.args.get("alias")
            identifier = getattr(parent_alias, "this", None)
    value = str(getattr(identifier, "name", None) or fallback)
    return _semantic_identifier(
        value,
        dialect=dialect,
        quoted=bool(getattr(identifier, "args", {}).get("quoted", False)),
    )


def _projection_alias(
    expression: Any,
    *,
    dialect: _SqlDialect,
) -> str:
    alias = expression.args.get("alias") if hasattr(expression, "args") else None
    identifier = getattr(alias, "this", None)
    if identifier is None and hasattr(expression, "this"):
        candidate = expression.this
        if type(candidate).__name__ == "Identifier":
            identifier = candidate
    value = str(
        getattr(identifier, "name", None)
        or getattr(expression, "alias_or_name", None)
        or ""
    ).strip()
    if not value or value == "*":
        return ""
    return _semantic_identifier(
        value,
        dialect=dialect,
        quoted=bool(getattr(identifier, "args", {}).get("quoted", False)),
    )


def _outer_projection_columns(
    scope: Any,
    *,
    dialect: _SqlDialect,
) -> tuple[str, ...]:
    if not scope.outer_columns:
        return ()
    owner = scope.expression.parent
    alias = owner.args.get("alias") if hasattr(owner, "args") else None
    identifiers = tuple(alias.args.get("columns") or ()) if alias is not None else ()
    if len(identifiers) != len(scope.outer_columns):
        return tuple(
            _semantic_identifier(item, dialect=dialect) for item in scope.outer_columns
        )
    return tuple(
        _semantic_identifier(
            str(identifier.name),
            dialect=dialect,
            quoted=bool(identifier.args.get("quoted", False)),
        )
        for identifier in identifiers
    )


def _relation_column_matches(
    relation: _LexicalRelation,
    column: Any,
    *,
    dialect: _SqlDialect,
) -> tuple[str, ...]:
    identifier = column.this
    expected = _semantic_identifier(
        str(column.name),
        dialect=dialect,
        quoted=bool(getattr(identifier, "args", {}).get("quoted", False)),
    )
    return tuple(item for item in relation.columns if item == expected)


def _scope_projection(
    scope: Any,
    relations: tuple[_LexicalRelation, ...],
    states: Mapping[int, _LexicalScope],
    exp: Any,
    issues: list[SqlValidationIssue],
    *,
    dialect: _SqlDialect,
) -> tuple[str, ...]:
    if isinstance(
        scope.expression,
        tuple(
            item
            for item in (
                getattr(exp, "Union", None),
                getattr(exp, "Intersect", None),
                getattr(exp, "Except", None),
            )
            if item is not None
        ),
    ):
        branches = tuple(states[id(item)] for item in scope.union_scopes)
        if not branches:
            return ()
        expected = len(branches[0].columns)
        received = tuple(len(item.columns) for item in branches)
        if any(count != expected for count in received[1:]):
            issues.append(
                SqlValidationIssue(
                    "set_projection_arity_mismatch",
                    "SQL set-operation branches must expose the same column count.",
                    {"branch_column_counts": received[:_MAX_CANDIDATES]},
                )
            )
        projected = branches[0].columns
    else:
        projected_items: list[str] = []
        for expression in getattr(scope.expression, "expressions", ()):
            if isinstance(expression, exp.Star):
                for relation in relations:
                    projected_items.extend(relation.columns)
                continue
            if isinstance(expression, exp.Column) and str(expression.name) == "*":
                table_identifier = expression.args.get("table")
                qualifier = _semantic_identifier(
                    str(expression.table),
                    dialect=dialect,
                    quoted=bool(
                        getattr(table_identifier, "args", {}).get("quoted", False)
                    ),
                )
                star_relation = next(
                    (item for item in relations if item.qualifier == qualifier),
                    None,
                )
                if star_relation is not None:
                    projected_items.extend(star_relation.columns)
                else:
                    issues.append(
                        SqlValidationIssue(
                            "unknown_relation_qualifier",
                            "SQL references a relation qualifier not visible in this scope.",
                            {"qualifier": str(expression.table)},
                        )
                    )
                continue
            projected_items.append(_projection_alias(expression, dialect=dialect))
        projected = tuple(projected_items)

    outer_columns = _outer_projection_columns(scope, dialect=dialect)
    if outer_columns:
        if len(outer_columns) != len(projected):
            issues.append(
                SqlValidationIssue(
                    "derived_projection_arity_mismatch",
                    "A derived relation column list must match its projection count.",
                    {
                        "declared_column_count": len(outer_columns),
                        "projected_column_count": len(projected),
                    },
                )
            )
        else:
            projected = outer_columns
    return projected


def _direct_scope_columns(
    scope: Any,
    expression_scope_ids: frozenset[int],
    exp: Any,
) -> tuple[Any, ...]:
    """Return columns whose nearest query owner is exactly ``scope``."""

    columns: list[Any] = []
    for column in scope.expression.find_all(exp.Column):
        owner = column.parent
        while owner is not None and id(owner) not in expression_scope_ids:
            owner = owner.parent
        if owner is scope.expression:
            columns.append(column)
    return tuple(columns)


def _is_legal_output_alias(
    column: Any,
    scope: Any,
    state: _LexicalScope,
    exp: Any,
    *,
    dialect: _SqlDialect,
) -> bool:
    if str(column.table or "").strip():
        return False
    identifier = column.this
    name = _semantic_identifier(
        str(column.name),
        dialect=dialect,
        quoted=bool(getattr(identifier, "args", {}).get("quoted", False)),
    )
    explicit_aliases = {
        _projection_alias(expression, dialect=dialect)
        for expression in getattr(scope.expression, "expressions", ())
        if bool(str(getattr(expression, "alias", "") or "").strip())
    }
    if isinstance(
        scope.expression,
        tuple(
            item
            for item in (
                getattr(exp, "Union", None),
                getattr(exp, "Intersect", None),
                getattr(exp, "Except", None),
            )
            if item is not None
        ),
    ):
        explicit_aliases.update(state.columns)
    if name not in explicit_aliases:
        return False
    ancestor = column.parent
    while ancestor is not None and ancestor is not scope.expression:
        if isinstance(ancestor, (exp.Order, exp.Ordered, exp.Group)):
            return True
        if dialect == "sqlite" and isinstance(
            ancestor,
            (exp.Where, exp.Having, exp.Join),
        ):
            return True
        ancestor = ancestor.parent
    return False


def _lexical_column_issues(
    sql: str,
    *,
    resources: tuple[ResourceSchema, ...],
    allowed_resource_ids: frozenset[str] | None,
    dialect: _SqlDialect,
) -> tuple[SqlValidationIssue, ...]:
    """Validate columns against relation schemas in their lexical scopes."""

    sqlglot, exp = _load_sqlglot(dialect)
    parse_sql = (_explain_prefix(sql) or ("", sql))[1]
    root = sqlglot.parse_one(
        parse_sql,
        read="postgres" if dialect == "postgresql" else "sqlite",
    )
    from sqlglot.optimizer.scope import traverse_scope

    scopes = tuple(traverse_scope(root))
    scope_ids = {id(scope): f"scope:{index}" for index, scope in enumerate(scopes)}
    expression_scope_ids = frozenset(id(scope.expression) for scope in scopes)
    consumed_scope_ids = {
        id(source)
        for scope in scopes
        for _, source in scope.selected_sources.values()
        if not isinstance(source, exp.Table)
    }
    states: dict[int, _LexicalScope] = {}
    unresolved_qualifiers: dict[int, frozenset[str]] = {}
    issues: list[SqlValidationIssue] = []
    for scope in scopes:
        relations: list[_LexicalRelation] = []
        unresolved: set[str] = set()
        for fallback, (selected, source) in scope.selected_sources.items():
            qualifier = _selected_source_qualifier(
                selected,
                fallback,
                exp,
                dialect=dialect,
            )
            if isinstance(source, exp.Table):
                references = _table_references(
                    source,
                    exp,
                    set(),
                    dialect=dialect,
                )
                candidates = (
                    _resource_candidates(
                        references[0],
                        resources,
                        dialect=dialect,
                    )
                    if references
                    else ()
                )
                if len(candidates) != 1:
                    unresolved.add(qualifier)
                    continue
                resource = candidates[0]
                if (
                    allowed_resource_ids is not None
                    and resource.resource_id not in allowed_resource_ids
                ):
                    unresolved.add(qualifier)
                    continue
                relations.append(
                    _LexicalRelation(
                        qualifier=qualifier,
                        columns=tuple(
                            _semantic_identifier(
                                column,
                                dialect=dialect,
                                quoted=(
                                    dialect == "postgresql"
                                    and column
                                    != column.translate(
                                        _ASCII_IDENTIFIER_CASE_TRANSLATION
                                    )
                                ),
                            )
                            for column in resource.columns
                        ),
                        lineage=(resource.resource_id,),
                        kind="base",
                        scope_id=scope_ids[id(scope)],
                    )
                )
                continue
            derived = states.get(id(source))
            if derived is None:
                unresolved.add(qualifier)
                continue
            kind: Literal["cte", "subquery", "set"] = (
                "set"
                if isinstance(
                    source.expression,
                    tuple(
                        item
                        for item in (
                            getattr(exp, "Union", None),
                            getattr(exp, "Intersect", None),
                            getattr(exp, "Except", None),
                        )
                        if item is not None
                    ),
                )
                else "cte" if isinstance(selected, exp.Table) else "subquery"
            )
            relations.append(
                _LexicalRelation(
                    qualifier=qualifier,
                    columns=derived.columns,
                    lineage=derived.lineage,
                    kind=kind,
                    scope_id=scope_ids[id(scope)],
                )
            )

        relation_tuple = tuple(relations)
        projected = _scope_projection(
            scope,
            relation_tuple,
            states,
            exp,
            issues,
            dialect=dialect,
        )
        child_scopes = tuple(
            item for item in scopes if item.parent is scope and id(item) in states
        )
        if getattr(scope, "union_scopes", ()):
            lineage = tuple(
                dict.fromkeys(
                    resource_id
                    for branch in scope.union_scopes
                    for resource_id in states[id(branch)].lineage
                )
            )
        else:
            lineage = tuple(
                dict.fromkeys(
                    (
                        resource_id
                        for relation in relation_tuple
                        for resource_id in relation.lineage
                    ),
                )
            )
        lineage = tuple(
            dict.fromkeys(
                (
                    *lineage,
                    *(
                        resource_id
                        for child in child_scopes
                        for resource_id in states[id(child)].lineage
                    ),
                )
            )
        )
        states[id(scope)] = _LexicalScope(
            columns=projected,
            lineage=lineage,
            relations=relation_tuple,
            scope_id=scope_ids[id(scope)],
        )
        unresolved_qualifiers[id(scope)] = frozenset(unresolved)

    for scope in scopes:
        state = states[id(scope)]
        relation_tuple = state.relations
        for column in _direct_scope_columns(scope, expression_scope_ids, exp):
            if str(column.name) == "*":
                continue
            qualifier_text = str(column.table or "").strip()
            if qualifier_text:
                qualifier = _semantic_identifier(
                    qualifier_text,
                    dialect=dialect,
                    quoted=bool(
                        getattr(column.args.get("table"), "args", {}).get(
                            "quoted", False
                        )
                    ),
                )
                relation = next(
                    (item for item in relation_tuple if item.qualifier == qualifier),
                    None,
                )
                if relation is None:
                    ancestor = scope.parent
                    ancestor_relation = None
                    while ancestor is not None and ancestor_relation is None:
                        ancestor_relation = next(
                            (
                                item
                                for item in states[id(ancestor)].relations
                                if item.qualifier == qualifier
                            ),
                            None,
                        )
                        ancestor = ancestor.parent
                    if ancestor_relation is not None:
                        if not bool(scope.can_be_correlated):
                            issues.append(
                                SqlValidationIssue(
                                    "column_scope_escape",
                                    "SQL column escapes a non-correlated lexical scope.",
                                    {"qualifier": qualifier_text},
                                )
                            )
                            continue
                        relation = ancestor_relation
                    elif qualifier in unresolved_qualifiers[id(scope)]:
                        continue
                if relation is None:
                    issues.append(
                        SqlValidationIssue(
                            "unknown_relation_qualifier",
                            "SQL references a relation qualifier not visible in this scope.",
                            {"qualifier": qualifier_text},
                        )
                    )
                    continue
                matches = _relation_column_matches(
                    relation,
                    column,
                    dialect=dialect,
                )
            else:
                matches = tuple(
                    match
                    for relation in relation_tuple
                    for match in _relation_column_matches(
                        relation,
                        column,
                        dialect=dialect,
                    )
                )
                relation = None
                if not matches:
                    ancestor_matches: tuple[str, ...] = ()
                    ancestor = scope.parent
                    while ancestor is not None and not ancestor_matches:
                        ancestor_matches = tuple(
                            match
                            for ancestor_relation in states[id(ancestor)].relations
                            for match in _relation_column_matches(
                                ancestor_relation,
                                column,
                                dialect=dialect,
                            )
                        )
                        ancestor = ancestor.parent
                    if ancestor_matches:
                        if not bool(scope.can_be_correlated):
                            issues.append(
                                SqlValidationIssue(
                                    "column_scope_escape",
                                    "SQL column escapes a non-correlated lexical scope.",
                                    {"column": str(column.name)},
                                )
                            )
                            continue
                        matches = ancestor_matches
            if not matches:
                if unresolved_qualifiers[id(scope)]:
                    continue
                if _is_legal_output_alias(
                    column,
                    scope,
                    state,
                    exp,
                    dialect=dialect,
                ):
                    continue
                if relation is None and len(relation_tuple) == 1:
                    relation = relation_tuple[0]
                if relation is not None and relation.kind == "base":
                    missing_resource = next(
                        (
                            item
                            for item in resources
                            if item.resource_id in relation.lineage
                        ),
                        None,
                    )
                    if missing_resource is not None:
                        issues.append(
                            _missing_column_issue(
                                SqlColumnReference(
                                    name=str(column.name),
                                    qualifier=(
                                        str(column.table) if column.table else None
                                    ),
                                    name_quoted=bool(
                                        getattr(column.this, "args", {}).get(
                                            "quoted", False
                                        )
                                    ),
                                ),
                                missing_resource,
                            )
                        )
                        continue
                issues.append(
                    SqlValidationIssue(
                        (
                            "unknown_derived_column"
                            if relation is not None and relation.kind != "base"
                            else "missing_column"
                        ),
                        "SQL references a column absent from the visible relation schema.",
                        {"column": str(column.name)},
                    )
                )
            elif len(matches) > 1:
                issues.append(
                    SqlValidationIssue(
                        "ambiguous_column",
                        "SQL column is ambiguous in its lexical scope.",
                        {"column": str(column.name)},
                    )
                )

    for scope_id in consumed_scope_ids:
        consumed_state = states.get(scope_id)
        if consumed_state is not None and any(
            not column for column in consumed_state.columns
        ):
            issues.append(
                SqlValidationIssue(
                    "derived_projection_name_required",
                    "Every derived-relation output column must have a stable name.",
                )
            )
    return tuple(issues)


def validate_sqlite_read(
    sql: str,
    *,
    source_id: str,
    resources: Iterable[ResourceSchema],
    parameters: Sequence[object] = (),
    allowed_resource_ids: Iterable[str] | None = None,
) -> SqlValidationResult:
    """Validate a single SQLite read against catalog-owned source scope."""

    return _validate_sql_read(
        sql,
        source_id=source_id,
        resources=resources,
        parameters=parameters,
        allowed_resource_ids=allowed_resource_ids,
        dialect="sqlite",
    )


def validate_postgresql_read(
    sql: str,
    *,
    source_id: str,
    resources: Iterable[ResourceSchema],
    parameters: Sequence[object] = (),
    allowed_resource_ids: Iterable[str] | None = None,
) -> SqlValidationResult:
    """Validate a single PostgreSQL read against catalog-owned source scope."""

    return _validate_sql_read(
        sql,
        source_id=source_id,
        resources=resources,
        parameters=parameters,
        allowed_resource_ids=allowed_resource_ids,
        dialect="postgresql",
    )


def _validate_sql_read(
    sql: str,
    *,
    source_id: str,
    resources: Iterable[ResourceSchema],
    parameters: Sequence[object],
    allowed_resource_ids: Iterable[str] | None,
    dialect: _SqlDialect,
) -> SqlValidationResult:
    display_name = "PostgreSQL" if dialect == "postgresql" else "SQLite"

    source_id = _required_text(source_id, "source_id")
    try:
        analysis = (
            analyze_postgresql_sql(sql)
            if dialect == "postgresql"
            else analyze_sqlite_sql(sql)
        )
    except SqlAnalysisError as error:
        return SqlValidationResult(
            valid=False,
            source_id=source_id,
            analysis=None,
            resource_ids=(),
            resource_revisions=(),
            source_revision=None,
            issues=(SqlValidationIssue(error.code, str(error)),),
        )

    source_resources = tuple(
        resource for resource in resources if resource.source_id == source_id
    )
    issues: list[SqlValidationIssue] = []
    if analysis.statement_count != 1:
        issues.append(
            SqlValidationIssue(
                "multiple_statements",
                "Exactly one SQL statement is allowed.",
                {"statement_count": analysis.statement_count},
            )
        )
    if analysis.mutation_types:
        issues.append(
            SqlValidationIssue(
                "mutation_not_allowed",
                f"{display_name} data queries must be read-only.",
                {"mutation_types": list(analysis.mutation_types)},
            )
        )
    elif not analysis.is_read:
        issues.append(
            SqlValidationIssue(
                "read_statement_required",
                f"{display_name} data queries require a read statement.",
                {"statement_type": analysis.statement_type},
            )
        )
    if dialect == "postgresql" and analysis.statement_type == "explain":
        issues.append(
            SqlValidationIssue(
                "explain_not_allowed",
                (
                    "PostgreSQL EXPLAIN statements cannot use the bounded "
                    "tabular execution path."
                ),
            )
        )
    if dialect == "postgresql":
        function_names_not_admitted = {
            name
            for name in analysis.function_names
            if (name.removeprefix("PG_CATALOG.") not in _POSTGRESQL_BOUNDED_FUNCTIONS)
        }
        denied_functions = tuple(
            sorted(
                set(analysis.unresolved_function_names)
                | set(analysis.table_function_names)
                | function_names_not_admitted
            )
        )
    else:
        denied_functions = ()
    if denied_functions:
        issues.append(
            SqlValidationIssue(
                "function_not_allowed",
                (
                    f"{display_name} data queries allow only the declared "
                    "bounded function set."
                ),
                {"functions": denied_functions[:_MAX_CANDIDATES]},
            )
        )
    if dialect == "postgresql" and analysis.unsafe_cast_type_names:
        issues.append(
            SqlValidationIssue(
                "cast_type_not_allowed",
                "PostgreSQL casts must target a declared built-in data type.",
                {"types": analysis.unsafe_cast_type_names[:_MAX_CANDIDATES]},
            )
        )
    unsafe_operators = tuple(
        name
        for name in analysis.explicit_operator_names
        if not name.startswith("PG_CATALOG.")
    )
    if dialect == "postgresql" and unsafe_operators:
        issues.append(
            SqlValidationIssue(
                "operator_not_allowed",
                "Explicit PostgreSQL operators must resolve from pg_catalog.",
                {"operators": unsafe_operators[:_MAX_CANDIDATES]},
            )
        )
    if dialect == "postgresql" and analysis.anonymous_parameter_count:
        issues.append(
            SqlValidationIssue(
                "parameter_style_invalid",
                "PostgreSQL query parameters must use numbered $1 placeholders.",
                {"anonymous_placeholders": analysis.anonymous_parameter_count},
            )
        )
    expected_ordinals = tuple(range(1, len(parameters) + 1))
    if dialect == "postgresql" and (
        analysis.invalid_parameter_count
        or analysis.parameter_ordinals != expected_ordinals
    ):
        issues.append(
            SqlValidationIssue(
                "parameter_index_mismatch",
                "PostgreSQL parameter indexes must be contiguous and match the supplied values.",
                {
                    "expected": expected_ordinals,
                    "received": analysis.parameter_ordinals,
                    "invalid": analysis.invalid_parameter_count,
                },
            )
        )
    elif dialect == "sqlite" and analysis.positional_parameter_count != len(parameters):
        issues.append(
            SqlValidationIssue(
                "parameter_count_mismatch",
                "SQLite positional parameter count does not match the SQL placeholders.",
                {
                    "expected": analysis.positional_parameter_count,
                    "received": len(parameters),
                },
            )
        )
    if not source_resources:
        issues.append(
            SqlValidationIssue(
                "catalog_schema_missing",
                "No current catalog resource schema is available for the source.",
                {"source_id": source_id},
            )
        )

    allowed = (
        None
        if allowed_resource_ids is None
        else frozenset(str(item) for item in allowed_resource_ids)
    )
    resolved_ids: list[str] = []
    for table in analysis.tables:
        if table.is_cte:
            continue
        if dialect == "postgresql" and len(table.qualified_parts) < 2:
            issues.append(
                SqlValidationIssue(
                    "schema_qualification_required",
                    (
                        "PostgreSQL resources must be schema-qualified so "
                        "catalog provenance and server resolution agree."
                    ),
                    {"resource": table.qualified_name},
                )
            )
        candidates = _resource_candidates(
            table,
            source_resources,
            dialect=dialect,
        )
        if not candidates:
            issues.append(
                SqlValidationIssue(
                    "unknown_resource",
                    "SQL references a resource absent from the current catalog scope.",
                    {
                        "resource": table.qualified_name,
                        "candidates": _resource_name_candidates(
                            table.qualified_name, source_resources
                        ),
                    },
                )
            )
            continue
        if len(candidates) > 1:
            issues.append(
                SqlValidationIssue(
                    "ambiguous_resource",
                    "SQL resource reference is ambiguous in the current catalog scope.",
                    {
                        "resource": table.qualified_name,
                        "resource_ids": [item.resource_id for item in candidates][
                            :_MAX_CANDIDATES
                        ],
                    },
                )
            )
            continue
        resource = next(iter(candidates))
        if allowed is not None and resource.resource_id not in allowed:
            issues.append(
                SqlValidationIssue(
                    "resource_out_of_scope",
                    "SQL references a resource outside the allowed operation scope.",
                    {"resource_id": resource.resource_id},
                )
            )
            continue
        if dialect == "postgresql" and resource.resource_kind != "table":
            issues.append(
                SqlValidationIssue(
                    "resource_kind_not_allowed",
                    (
                        "PostgreSQL bounded reads require a cataloged base "
                        "table, not a view or unknown relation kind."
                    ),
                    {
                        "resource_id": resource.resource_id,
                        "resource_kind": resource.resource_kind or "unknown",
                    },
                )
            )
        if resource.resource_id not in resolved_ids:
            resolved_ids.append(resource.resource_id)

    resolved_resources = tuple(
        resource
        for resource in source_resources
        if resource.resource_id in set(resolved_ids)
    )
    resource_revisions = tuple(
        sorted(
            (resource.resource_id, resource.revision)
            for resource in resolved_resources
            if resource.revision is not None
        )
    )
    source_revisions = tuple(
        resource.source_revision
        for resource in resolved_resources
        if resource.source_revision is not None
    )
    if resource_revisions and len(resource_revisions) != len(resolved_resources):
        issues.append(
            SqlValidationIssue(
                "catalog_revision_scope_incomplete",
                "Catalog resource revision scope is incomplete.",
                {"resource_ids": sorted(resolved_ids)[:_MAX_CANDIDATES]},
            )
        )
        resource_revisions = ()
    if source_revisions and len(source_revisions) != len(resolved_resources):
        issues.append(
            SqlValidationIssue(
                "catalog_source_revision_scope_incomplete",
                "Catalog source revision scope is incomplete.",
                {"resource_ids": sorted(resolved_ids)[:_MAX_CANDIDATES]},
            )
        )
    unique_source_revisions = tuple(sorted(set(source_revisions)))
    if len(unique_source_revisions) > 1:
        issues.append(
            SqlValidationIssue(
                "catalog_source_revision_conflict",
                "Catalog resources do not share one current source revision.",
                {"source_revisions": unique_source_revisions[:_MAX_CANDIDATES]},
            )
        )
    issues.extend(
        _lexical_column_issues(
            analysis.sql,
            resources=source_resources,
            allowed_resource_ids=allowed,
            dialect=dialect,
        )
    )

    bounded_issues = tuple(issues[:_MAX_ISSUES])
    return SqlValidationResult(
        valid=not bounded_issues,
        source_id=source_id,
        analysis=analysis,
        resource_ids=tuple(resolved_ids),
        resource_revisions=resource_revisions,
        source_revision=(
            unique_source_revisions[0]
            if len(unique_source_revisions) == 1
            and len(source_revisions) == len(resolved_resources)
            else None
        ),
        issues=bounded_issues,
    )


def validate_sqlite_update_recipe(
    *,
    source_id: str,
    resource_id: str,
    key_column: str,
    key_value: str,
    target_column: str,
    expected_value: str,
    new_value: str,
    resources: Iterable[ResourceSchema],
    source_write_access: bool,
) -> SQLiteUpdateValidationResult:
    """Validate a semantic, conditional, single-row SQLite update recipe."""

    source_id = _required_text(source_id, "source_id")
    issues: list[SqlValidationIssue] = []
    if not isinstance(source_write_access, bool):
        raise TypeError("source_write_access must be a boolean")
    if not source_write_access:
        issues.append(
            SqlValidationIssue(
                "source_write_access_required",
                "The SQLite source was not explicitly attached with write access.",
                {"source_id": source_id},
            )
        )

    identifiers: dict[str, str] = {}
    for value, field_name, maximum in (
        (source_id, "source_id", _MAX_UPDATE_IDENTIFIER_CHARACTERS),
        (resource_id, "resource_id", _MAX_UPDATE_IDENTIFIER_CHARACTERS),
        (key_column, "key_column", _MAX_UPDATE_COLUMN_CHARACTERS),
        (target_column, "target_column", _MAX_UPDATE_COLUMN_CHARACTERS),
    ):
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value.strip()) > maximum
        ):
            issues.append(
                SqlValidationIssue(
                    "input_out_of_bounds",
                    "SQLite update identifiers must be bounded non-empty strings.",
                    {"field": field_name, "maximum_characters": maximum},
                )
            )
            continue
        identifiers[field_name] = value.strip()

    values: dict[str, str] = {}
    for value, field_name in (
        (key_value, "key_value"),
        (expected_value, "expected_value"),
        (new_value, "new_value"),
    ):
        if not isinstance(value, str) or len(value) > _MAX_UPDATE_VALUE_CHARACTERS:
            issues.append(
                SqlValidationIssue(
                    "input_out_of_bounds",
                    "SQLite update values must be bounded strings.",
                    {
                        "field": field_name,
                        "maximum_characters": _MAX_UPDATE_VALUE_CHARACTERS,
                    },
                )
            )
            continue
        values[field_name] = value

    source_resources = tuple(
        resource
        for resource in resources
        if isinstance(resource, ResourceSchema) and resource.source_id == source_id
    )
    selected_resource_id = identifiers.get("resource_id")
    matches = tuple(
        resource
        for resource in source_resources
        if resource.resource_id == selected_resource_id
    )
    resource = matches[0] if len(matches) == 1 else None
    if resource is None:
        issues.append(
            SqlValidationIssue(
                "catalog_resource_missing",
                "The update resource is absent from the current catalog scope.",
                {
                    "resource_id": selected_resource_id or "invalid",
                    "source_id": source_id,
                },
            )
        )
    else:
        if resource.resource_kind != "table" or not resource.writable:
            issues.append(
                SqlValidationIssue(
                    "resource_not_writable_table",
                    "Controlled SQLite updates require a cataloged writable table.",
                    {
                        "resource_id": resource.resource_id,
                        "resource_kind": resource.resource_kind or "unknown",
                    },
                )
            )
        if resource.revision is None or resource.source_revision is None:
            issues.append(
                SqlValidationIssue(
                    "catalog_revision_scope_incomplete",
                    "Controlled SQLite updates require current resource and source revisions.",
                    {"resource_id": resource.resource_id},
                )
            )

        column_by_key = {
            sqlite_identifier_key(column): column for column in resource.columns
        }
        canonical_key = column_by_key.get(
            sqlite_identifier_key(identifiers.get("key_column", ""))
        )
        canonical_target = column_by_key.get(
            sqlite_identifier_key(identifiers.get("target_column", ""))
        )
        if canonical_key is None:
            issues.append(
                SqlValidationIssue(
                    "unknown_key_column",
                    "The update key column is absent from the current catalog schema.",
                    {"resource_id": resource.resource_id},
                )
            )
        elif sqlite_identifier_key(canonical_key) not in {
            sqlite_identifier_key(column) for column in resource.unique_key_columns
        }:
            issues.append(
                SqlValidationIssue(
                    "key_not_unique",
                    "The update key must be one complete single-column unique or primary key.",
                    {
                        "key_column": canonical_key,
                        "resource_id": resource.resource_id,
                    },
                )
            )
        if canonical_target is None:
            issues.append(
                SqlValidationIssue(
                    "unknown_target_column",
                    "The update target column is absent from the current catalog schema.",
                    {"resource_id": resource.resource_id},
                )
            )
        else:
            declared_type_by_key = {
                sqlite_identifier_key(column): declared_type
                for column, declared_type in resource.column_declared_types
            }
            target_declared_type = declared_type_by_key.get(
                sqlite_identifier_key(canonical_target)
            )
            if target_declared_type is None:
                issues.append(
                    SqlValidationIssue(
                        "catalog_column_type_scope_incomplete",
                        "The current catalog lacks the update target declared type.",
                        {
                            "resource_id": resource.resource_id,
                            "target_column": canonical_target,
                        },
                    )
                )
            else:
                target_affinity = sqlite_declared_type_affinity(target_declared_type)
                if target_affinity != "text":
                    issues.append(
                        SqlValidationIssue(
                            "target_column_affinity_not_supported",
                            "Controlled SQLite updates currently require a TEXT-affinity target column.",
                            {
                                "resource_id": resource.resource_id,
                                "target_affinity": target_affinity,
                                "target_column": canonical_target,
                            },
                        )
                    )
        if (
            canonical_key is not None
            and canonical_target is not None
            and sqlite_identifier_key(canonical_key)
            == sqlite_identifier_key(canonical_target)
        ):
            issues.append(
                SqlValidationIssue(
                    "key_target_conflict",
                    "The update key and target columns must be distinct.",
                    {"resource_id": resource.resource_id},
                )
            )

    if (
        "expected_value" in values
        and "new_value" in values
        and values["expected_value"] == values["new_value"]
    ):
        issues.append(
            SqlValidationIssue(
                "no_op_update",
                "The expected and replacement values must be different.",
            )
        )

    bounded_issues = tuple(issues[:_MAX_ISSUES])
    if bounded_issues or resource is None:
        return SQLiteUpdateValidationResult(
            valid=False,
            source_id=source_id,
            recipe=None,
            issues=bounded_issues,
        )

    canonical_key = next(
        column
        for column in resource.columns
        if sqlite_identifier_key(column)
        == sqlite_identifier_key(identifiers["key_column"])
    )
    canonical_target = next(
        column
        for column in resource.columns
        if sqlite_identifier_key(column)
        == sqlite_identifier_key(identifiers["target_column"])
    )
    assert resource.revision is not None
    assert resource.source_revision is not None
    recipe_fingerprint = _sqlite_update_fingerprint(
        source_id=resource.source_id,
        resource_id=resource.resource_id,
        resource_revision=resource.revision,
        source_revision=resource.source_revision,
        table_name=resource.name,
        key_column=canonical_key,
        target_column=canonical_target,
        key_value=values["key_value"],
        expected_value=values["expected_value"],
        new_value=values["new_value"],
    )
    return SQLiteUpdateValidationResult(
        valid=True,
        source_id=source_id,
        recipe=SQLiteUpdateRecipe(
            source_id=resource.source_id,
            resource_id=resource.resource_id,
            resource_revision=resource.revision,
            source_revision=resource.source_revision,
            table_name=resource.name,
            key_column=canonical_key,
            target_column=canonical_target,
            key_value=values["key_value"],
            expected_value=values["expected_value"],
            new_value=values["new_value"],
            sensitivity_class=resource.sensitivity_class,
            recipe_fingerprint=recipe_fingerprint,
        ),
        issues=(),
    )


def _sqlite_update_fingerprint(
    *,
    source_id: str,
    resource_id: str,
    resource_revision: str,
    source_revision: str,
    table_name: str,
    key_column: str,
    target_column: str,
    key_value: str,
    expected_value: str,
    new_value: str,
) -> str:
    encoded = canonical_json(
        {
            "expected_value": expected_value,
            "key_column": key_column,
            "key_value": key_value,
            "new_value": new_value,
            "recipe": "data.sqlite.update",
            "resource_id": resource_id,
            "resource_revision": resource_revision,
            "schema_version": 1,
            "source_id": source_id,
            "source_revision": source_revision,
            "table_name": table_name,
            "target_column": target_column,
        }
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _resource_candidates(
    table: SqlTableReference,
    resources: tuple[ResourceSchema, ...],
    *,
    dialect: _SqlDialect,
) -> tuple[ResourceSchema, ...]:
    if dialect == "postgresql":
        return tuple(
            resource
            for resource in resources
            if _postgresql_resource_matches(table, resource)
        )
    key = _identifier_key(table.qualified_name)
    short = _short_identifier(table.qualified_name)
    return tuple(
        resource
        for resource in resources
        if key in resource.lookup_names or short in resource.lookup_names
    )


def _postgresql_resource_matches(
    table: SqlTableReference,
    resource: ResourceSchema,
) -> bool:
    expected_name = (
        table.name
        if table.name_quoted
        else table.name.translate(_ASCII_IDENTIFIER_CASE_TRANSLATION)
    )
    if resource.name != expected_name:
        return False
    if len(table.qualified_parts) < 2:
        return True
    expected_parts = tuple(
        name if quoted else name.translate(_ASCII_IDENTIFIER_CASE_TRANSLATION)
        for name, quoted in table.qualified_parts
    )
    return any(
        _postgresql_alias_parts(alias) == expected_parts for alias in resource.aliases
    )


def _postgresql_alias_parts(alias: str) -> tuple[str, ...]:
    # Attached schemas are safe single identifiers, while quoted PostgreSQL
    # resource names may themselves contain dots.  The first separator is the
    # catalog-owned schema/resource boundary.
    schema, separator, resource_name = alias.partition(".")
    return (schema, resource_name) if separator else (schema,)


def _resource_name_candidates(
    table_name: str,
    resources: tuple[ResourceSchema, ...],
) -> list[str]:
    import difflib

    names = sorted({resource.name for resource in resources})
    return difflib.get_close_matches(
        _short_identifier(table_name),
        names,
        n=_MAX_CANDIDATES,
        cutoff=0.45,
    )


def _missing_column_issue(
    column: SqlColumnReference,
    resource: ResourceSchema,
) -> SqlValidationIssue:
    import difflib

    candidates = difflib.get_close_matches(
        column.name,
        list(resource.columns),
        n=_MAX_CANDIDATES,
        cutoff=0.45,
    )
    return SqlValidationIssue(
        "missing_column",
        "SQL references a column absent from the current catalog schema.",
        {
            "column": column.name,
            "resource_id": resource.resource_id,
            "candidates": candidates,
        },
    )
