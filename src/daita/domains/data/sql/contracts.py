"""Define immutable SQL analysis, schema, reference, and validation records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from ...._json import FrozenJsonObject, canonical_json

_SqlDialect = Literal["sqlite", "postgresql"]
_MAX_VALIDATION_ISSUE_DETAILS_CHARACTERS = 1_536
MAX_SQL_CHARACTERS = 65_536
MAX_SQL_PARAMETERS = 128
_ASCII_IDENTIFIER_CASE_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
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
    primary_key_columns: tuple[str, ...] = field(default=(), compare=False)
    column_declared_types: tuple[tuple[str, str], ...] = field(
        default=(),
        compare=False,
    )
    column_nullability: tuple[tuple[str, bool], ...] = field(
        default=(),
        compare=False,
    )
    column_type_provenance: tuple[tuple[str, str, str], ...] = field(
        default=(),
        compare=False,
    )
    identity_columns: tuple[str, ...] = field(default=(), compare=False)
    generated_columns: tuple[str, ...] = field(default=(), compare=False)
    updatable_columns: tuple[str, ...] = field(default=(), compare=False)

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
        primary_key_columns = tuple(
            _required_text(item, "resource primary key column")
            for item in self.primary_key_columns
        )
        if len(set(primary_key_columns)) != len(primary_key_columns):
            raise ValueError("resource primary key columns must be unique")
        if any(item not in set(columns) for item in primary_key_columns):
            raise ValueError("resource primary key columns must exist in columns")
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
        if isinstance(self.column_nullability, (str, bytes)):
            raise TypeError("resource column_nullability must be a sequence")
        nullable_by_column: dict[str, bool] = {}
        for nullability_item in tuple(self.column_nullability):
            if (
                not isinstance(nullability_item, (tuple, list))
                or len(nullability_item) != 2
            ):
                raise ValueError(
                    "resource column_nullability must contain column/boolean pairs"
                )
            column = _required_text(nullability_item[0], "resource nullability column")
            nullable = nullability_item[1]
            if column not in set(columns):
                raise ValueError("resource nullability columns must exist in columns")
            if column in nullable_by_column:
                raise ValueError("resource nullability columns must be unique")
            if not isinstance(nullable, bool):
                raise TypeError("resource column nullability must be boolean")
            nullable_by_column[column] = nullable
        canonical_nullability = tuple(
            (column, nullable_by_column[column])
            for column in columns
            if column in nullable_by_column
        )
        if isinstance(self.column_type_provenance, (str, bytes)):
            raise TypeError("resource column_type_provenance must be a sequence")
        provenance_by_column: dict[str, tuple[str, str]] = {}
        for provenance_item in tuple(self.column_type_provenance):
            if (
                not isinstance(provenance_item, (tuple, list))
                or len(provenance_item) != 3
            ):
                raise ValueError(
                    "resource column_type_provenance must contain "
                    "column/namespace/name triples"
                )
            column = _required_text(
                provenance_item[0], "resource type provenance column"
            )
            namespace = _required_text(
                provenance_item[1], "resource type provenance namespace"
            )
            native_name = _required_text(
                provenance_item[2], "resource type provenance name"
            )
            if column not in set(columns):
                raise ValueError(
                    "resource type provenance columns must exist in columns"
                )
            if column in provenance_by_column:
                raise ValueError("resource type provenance columns must be unique")
            if len(namespace) > 128 or len(native_name) > 128:
                raise ValueError("resource type provenance must be bounded")
            provenance_by_column[column] = (namespace, native_name)
        canonical_provenance = tuple(
            (column, *provenance_by_column[column])
            for column in columns
            if column in provenance_by_column
        )
        structural_column_sets: dict[str, tuple[str, ...]] = {}
        for field_name, values in (
            ("identity_columns", self.identity_columns),
            ("generated_columns", self.generated_columns),
            ("updatable_columns", self.updatable_columns),
        ):
            if isinstance(values, (str, bytes)):
                raise TypeError(f"resource {field_name} must be a sequence")
            selected = tuple(
                _required_text(item, f"resource {field_name} item") for item in values
            )
            if len(selected) != len(set(selected)):
                raise ValueError(f"resource {field_name} must be unique")
            if any(item not in set(columns) for item in selected):
                raise ValueError(f"resource {field_name} must exist in columns")
            selected_set = set(selected)
            structural_column_sets[field_name] = tuple(
                column for column in columns if column in selected_set
            )
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "sensitivity_class", sensitivity_class.casefold())
        object.__setattr__(self, "unique_key_columns", unique_key_columns)
        object.__setattr__(self, "primary_key_columns", primary_key_columns)
        object.__setattr__(
            self,
            "column_declared_types",
            canonical_declared_types,
        )
        object.__setattr__(self, "column_nullability", canonical_nullability)
        object.__setattr__(self, "column_type_provenance", canonical_provenance)
        for field_name, values in structural_column_sets.items():
            object.__setattr__(self, field_name, values)

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
