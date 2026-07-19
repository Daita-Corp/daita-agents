"""Provider-neutral SQLite SQL analysis and catalog-scope validation.

This module is deliberately pure: it parses and validates immutable inputs but
does not open a source, invoke an executor, or persist runtime state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Any

from ..._json import FrozenJsonObject

_SQLITE_INSTALL_HINT = "pip install 'daita-agents[sqlite]'"
_MAX_ISSUES = 32
_MAX_CANDIDATES = 8


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _identifier_key(value: str) -> str:
    return value.strip().strip('"`[]').casefold()


def _short_identifier(value: str) -> str:
    return _identifier_key(value).rsplit(".", 1)[-1]


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
        if len({_identifier_key(item) for item in columns}) != len(columns):
            raise ValueError("resource columns must be unique case-insensitively")
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
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "aliases", aliases)

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


@dataclass(frozen=True, slots=True)
class SqlColumnReference:
    name: str
    qualifier: str | None = None
    resource_name: str | None = None

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
        object.__setattr__(self, "details", FrozenJsonObject.from_mapping(self.details))


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


def _load_sqlglot() -> tuple[Any, Any]:
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as error:
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

    normalized = normalize_sql(sql)
    if not normalized:
        raise SqlAnalysisError("empty_sql", "SQL must not be empty.")

    sqlglot, exp = _load_sqlglot()
    explain = _explain_prefix(normalized)
    parse_sql = explain[1] if explain is not None else normalized
    try:
        parsed = tuple(sqlglot.parse(parse_sql, read="sqlite"))
        expressions = tuple(item for item in parsed if item is not None)
    except Exception as error:
        parse_error = getattr(getattr(sqlglot, "errors", None), "ParseError", ())
        if parse_error and not isinstance(error, parse_error):
            raise
        raise SqlAnalysisError(
            "sql_parse_error",
            "SQL could not be parsed for SQLite.",
        ) from error
    if not expressions:
        raise SqlAnalysisError("empty_sql", "SQL must not be empty.")

    root = expressions[0]
    cte_names = {
        _identifier_key(str(cte.alias_or_name))
        for cte in root.find_all(exp.CTE)
        if str(cte.alias_or_name).strip()
    }
    tables = _table_references(root, exp, cte_names)
    alias_map: dict[str, str] = {}
    for table in tables:
        if table.is_cte:
            continue
        alias_map[_identifier_key(table.name)] = table.qualified_name
        alias_map[_identifier_key(table.qualified_name)] = table.qualified_name
        if table.alias:
            alias_map[_identifier_key(table.alias)] = table.qualified_name

    columns = _column_references(root, exp, alias_map, cte_names)
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
        item.sql(dialect="sqlite", pretty=False) for item in expressions
    )
    if len(parsed) != len(expressions):
        # Preserve empty statements in the canonical value so repeated trailing
        # delimiters cannot share a fingerprint with one valid statement.
        canonical_inner = parse_sql
    canonical_sql = (
        f"{explain[0]} {canonical_inner}" if explain is not None else canonical_inner
    )
    fingerprint = "sha256:" + hashlib.sha256(canonical_sql.encode("utf-8")).hexdigest()
    positional_parameter_count = sum(
        1 for item in root.walk() if isinstance(item, exp.Placeholder)
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
    )


def _table_references(
    root: Any,
    exp: Any,
    cte_names: set[str],
) -> tuple[SqlTableReference, ...]:
    references: list[SqlTableReference] = []
    seen: set[tuple[str, str | None, bool]] = set()
    for table in root.find_all(exp.Table):
        parts = tuple(str(part.name) for part in table.parts if str(part.name).strip())
        name = str(table.name or "").strip()
        if not name:
            continue
        qualified = ".".join(parts) or name
        alias = str(table.alias or "").strip() or None
        is_cte = _short_identifier(qualified) in cte_names
        key = (
            _identifier_key(qualified),
            _identifier_key(alias) if alias else None,
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
            )
        )
    return tuple(references)


def _column_references(
    root: Any,
    exp: Any,
    alias_map: Mapping[str, str],
    cte_names: set[str],
) -> tuple[SqlColumnReference, ...]:
    references: list[SqlColumnReference] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for column in root.find_all(exp.Column):
        name = str(column.name or "").strip()
        if not name or name == "*":
            continue
        qualifier = str(column.table or "").strip() or None
        resource_name = None
        if qualifier and _identifier_key(qualifier) not in cte_names:
            resource_name = alias_map.get(_identifier_key(qualifier), qualifier)
        key = (
            _identifier_key(name),
            _identifier_key(qualifier) if qualifier else None,
            _identifier_key(resource_name) if resource_name else None,
        )
        if key in seen:
            continue
        seen.add(key)
        references.append(
            SqlColumnReference(
                name=name,
                qualifier=qualifier,
                resource_name=resource_name,
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


def validate_sqlite_read(
    sql: str,
    *,
    source_id: str,
    resources: Iterable[ResourceSchema],
    parameters: Sequence[object] = (),
    allowed_resource_ids: Iterable[str] | None = None,
) -> SqlValidationResult:
    """Validate a single SQLite read against catalog-owned source scope."""

    source_id = _required_text(source_id, "source_id")
    try:
        analysis = analyze_sqlite_sql(sql)
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
                "SQLite data queries must be read-only.",
                {"mutation_types": list(analysis.mutation_types)},
            )
        )
    elif not analysis.is_read:
        issues.append(
            SqlValidationIssue(
                "read_statement_required",
                "SQLite data queries require a read statement.",
                {"statement_type": analysis.statement_type},
            )
        )
    if analysis.positional_parameter_count != len(parameters):
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
    resolved_by_table: dict[str, ResourceSchema] = {}
    resolved_ids: list[str] = []
    for table in analysis.tables:
        if table.is_cte:
            continue
        candidates = _resource_candidates(table.qualified_name, source_resources)
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
        for key in (table.qualified_name, table.name, table.alias or ""):
            if key:
                resolved_by_table[_identifier_key(key)] = resource
        if resource.resource_id not in resolved_ids:
            resolved_ids.append(resource.resource_id)

    selected_aliases = {_identifier_key(item) for item in analysis.select_aliases}
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
    for column in analysis.columns:
        if (
            _identifier_key(column.name) in selected_aliases
            and column.qualifier is None
        ):
            continue
        _validate_column(
            column,
            resolved_by_table,
            resolved_resources,
            issues,
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


def _resource_candidates(
    table_name: str,
    resources: tuple[ResourceSchema, ...],
) -> tuple[ResourceSchema, ...]:
    key = _identifier_key(table_name)
    short = _short_identifier(table_name)
    return tuple(
        resource
        for resource in resources
        if key in resource.lookup_names or short in resource.lookup_names
    )


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


def _validate_column(
    column: SqlColumnReference,
    resolved_by_table: Mapping[str, ResourceSchema],
    resolved_resources: tuple[ResourceSchema, ...],
    issues: list[SqlValidationIssue],
) -> None:
    column_key = _identifier_key(column.name)
    if column.resource_name is not None:
        resource = resolved_by_table.get(_identifier_key(column.resource_name))
        if resource is None and column.qualifier is not None:
            resource = resolved_by_table.get(_identifier_key(column.qualifier))
        if resource is None:
            return
        if column_key not in resource.column_keys:
            issues.append(_missing_column_issue(column, resource))
        return

    matches = tuple(
        resource
        for resource in resolved_resources
        if column_key in resource.column_keys
    )
    if len(resolved_resources) == 1 and not matches:
        issues.append(_missing_column_issue(column, resolved_resources[0]))
    elif len(resolved_resources) > 1 and not matches:
        issues.append(
            SqlValidationIssue(
                "missing_column",
                "SQL references a column absent from the selected resources.",
                {"column": column.name},
            )
        )
    elif len(matches) > 1:
        issues.append(
            SqlValidationIssue(
                "ambiguous_column",
                "Unqualified SQL column is ambiguous across selected resources.",
                {
                    "column": column.name,
                    "resource_ids": [item.resource_id for item in matches][
                        :_MAX_CANDIDATES
                    ],
                },
            )
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
