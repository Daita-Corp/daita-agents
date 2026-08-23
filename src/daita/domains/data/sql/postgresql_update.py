"""Validate catalog-scoped PostgreSQL update intents and render executable SQL."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

from ...._json import FrozenJsonValue, canonical_json, freeze_json, thaw_json
from ....capabilities import render_approval_arguments
from .contracts import ResourceSchema, SqlValidationIssue

POSTGRESQL_UPDATE_MAX_CANONICAL_BYTES = 64 * 1_024
_POSTGRESQL_UPDATE_CAPABILITY_ID = "data.postgresql.update_impact"
_CANONICAL_SOURCE_ID = re.compile(r"source:sha256:[0-9a-f]{64}\Z")
_CANONICAL_RESOURCE_ID = re.compile(r"catalog-resource:sha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FILTER_OPERATORS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "lte",
        "gt",
        "gte",
        "in",
        "not_in",
        "is_null",
        "is_not_null",
    }
)
_ORDERED_FILTER_OPERATORS = frozenset({"lt", "lte", "gt", "gte"})
_SET_FILTER_OPERATORS = frozenset({"in", "not_in"})
_NULL_FILTER_OPERATORS = frozenset({"is_null", "is_not_null"})
_POSTGRESQL_UPDATE_TYPES = frozenset(
    {
        "bool",
        "int2",
        "int4",
        "int8",
        "float4",
        "float8",
        "numeric",
        "text",
        "varchar",
        "bpchar",
        "uuid",
        "date",
        "timestamp",
        "timestamptz",
        "json",
        "jsonb",
    }
)


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateCell:
    """One catalog column and literal assignment value."""

    column: str
    value: FrozenJsonValue

    def __post_init__(self) -> None:
        _require_column(self.column, "update cell column")
        object.__setattr__(self, "value", freeze_json(self.value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PostgreSQLUpdateCell:
        if not isinstance(value, Mapping) or set(value) != {"column", "value"}:
            raise ValueError("update cells require exactly column and value")
        column = value.get("column")
        if not isinstance(column, str):
            raise TypeError("update cell column must be text")
        return cls(column=column, value=freeze_json(value.get("value")))

    def to_payload(self) -> dict[str, object]:
        return {"column": self.column, "value": thaw_json(self.value)}


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateFilter:
    """One typed predicate in an AND-combined structured target selection."""

    column: str
    operator: str
    value: FrozenJsonValue

    def __post_init__(self) -> None:
        _require_column(self.column, "update filter column")
        if self.operator not in _FILTER_OPERATORS:
            raise ValueError("update filter operator is unsupported")
        object.__setattr__(self, "value", freeze_json(self.value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PostgreSQLUpdateFilter:
        if not isinstance(value, Mapping) or set(value) != {
            "column",
            "operator",
            "value",
        }:
            raise ValueError(
                "update filters require exactly column, operator, and value"
            )
        column = value.get("column")
        operator = value.get("operator")
        if not isinstance(column, str) or not isinstance(operator, str):
            raise TypeError("update filter column and operator must be text")
        return cls(column, operator, freeze_json(value.get("value")))

    def to_payload(self) -> dict[str, object]:
        return {
            "column": self.column,
            "operator": self.operator,
            "value": thaw_json(self.value),
        }


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateIntent:
    """A cardinality-independent, structured PostgreSQL update proposal."""

    source_id: str
    resource_id: str
    where: tuple[PostgreSQLUpdateFilter, ...]
    assignments: tuple[PostgreSQLUpdateCell, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "update source_id"),
            (self.resource_id, "update resource_id"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        where = tuple(self.where)
        assignments = tuple(self.assignments)
        if not where or any(
            not isinstance(item, PostgreSQLUpdateFilter) for item in where
        ):
            raise TypeError("update where must contain at least one update filter")
        if not assignments or any(
            not isinstance(item, PostgreSQLUpdateCell) for item in assignments
        ):
            raise TypeError("update assignments must contain update cells")
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "assignments", assignments)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PostgreSQLUpdateIntent:
        if not isinstance(value, Mapping):
            raise TypeError("PostgreSQL update intent must be an object")
        if set(value) != {"source_id", "resource_id", "where", "assignments"}:
            raise ValueError(
                "PostgreSQL update intent requires only source_id, resource_id, "
                "where, and assignments"
            )
        source_id = value.get("source_id")
        resource_id = value.get("resource_id")
        where = value.get("where")
        assignments = value.get("assignments")
        if not isinstance(source_id, str) or not isinstance(resource_id, str):
            raise TypeError("PostgreSQL update identities must be text")
        if not isinstance(where, (tuple, list)) or not isinstance(
            assignments, (tuple, list)
        ):
            raise TypeError("PostgreSQL update filters and assignments must be arrays")
        return cls(
            source_id=source_id,
            resource_id=resource_id,
            where=tuple(PostgreSQLUpdateFilter.from_mapping(item) for item in where),
            assignments=tuple(
                PostgreSQLUpdateCell.from_mapping(item) for item in assignments
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "resource_id": self.resource_id,
            "where": tuple(item.to_payload() for item in self.where),
            "assignments": tuple(item.to_payload() for item in self.assignments),
        }


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateCommand:
    """The exact approved plan and its previewed target cardinality."""

    intent: PostgreSQLUpdateIntent
    preview_fingerprint: str
    expected_affected_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.intent, PostgreSQLUpdateIntent):
            raise TypeError("update command intent must be PostgreSQLUpdateIntent")
        _require_sha256(self.preview_fingerprint, "preview_fingerprint")
        if (
            not isinstance(self.expected_affected_rows, int)
            or isinstance(self.expected_affected_rows, bool)
            or self.expected_affected_rows < 1
        ):
            raise ValueError("expected_affected_rows must be a positive integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PostgreSQLUpdateCommand:
        if not isinstance(value, Mapping):
            raise TypeError("PostgreSQL update command must be an object")
        expected = {
            "source_id",
            "resource_id",
            "where",
            "assignments",
            "preview_fingerprint",
            "expected_affected_rows",
        }
        if set(value) != expected:
            raise ValueError(
                "PostgreSQL update command requires the exact structured update, "
                "preview_fingerprint, and expected_affected_rows"
            )
        preview_fingerprint = value.get("preview_fingerprint")
        expected_rows = value.get("expected_affected_rows")
        if not isinstance(preview_fingerprint, str):
            raise TypeError("preview_fingerprint must be text")
        return cls(
            intent=PostgreSQLUpdateIntent.from_mapping(
                {
                    "source_id": value.get("source_id"),
                    "resource_id": value.get("resource_id"),
                    "where": value.get("where"),
                    "assignments": value.get("assignments"),
                }
            ),
            preview_fingerprint=preview_fingerprint,
            expected_affected_rows=expected_rows,  # type: ignore[arg-type]
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **self.intent.to_payload(),
            "preview_fingerprint": self.preview_fingerprint,
            "expected_affected_rows": self.expected_affected_rows,
        }


@dataclass(frozen=True, slots=True)
class ValidatedPostgreSQLUpdate:
    """A catalog-bound update plan with a structured target selection."""

    source_id: str
    resource_id: str
    resource_name: str
    schema_name: str
    relation_name: str
    source_revision: str
    resource_revision: str
    primary_key_columns: tuple[str, ...]
    where: tuple[PostgreSQLUpdateFilter, ...]
    assignments: tuple[PostgreSQLUpdateCell, ...]
    column_types: tuple[tuple[str, str, str], ...]
    intent_sha256: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "validated source_id"),
            (self.resource_id, "validated resource_id"),
            (self.resource_name, "validated resource_name"),
            (self.schema_name, "validated schema_name"),
            (self.relation_name, "validated relation_name"),
            (self.source_revision, "validated source_revision"),
            (self.resource_revision, "validated resource_revision"),
        ):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"{name} must be bounded non-empty text")
        primary_key_columns = tuple(self.primary_key_columns)
        if not primary_key_columns or len(primary_key_columns) != len(
            set(primary_key_columns)
        ):
            raise ValueError("validated update requires a complete primary key")
        if not self.where or not self.assignments:
            raise ValueError("validated update requires a selection and assignments")
        column_types = tuple(self.column_types)
        if len({item[0] for item in column_types}) != len(column_types):
            raise ValueError("validated update column types must be unique")
        if any(
            len(item) != 3
            or not all(isinstance(value, str) and value for value in item)
            for item in column_types
        ):
            raise ValueError("validated update column types are invalid")
        required = {
            *primary_key_columns,
            *(item.column for item in self.where),
            *(item.column for item in self.assignments),
        }
        if {item[0] for item in column_types} != required:
            raise ValueError("validated update column types must cover the plan")
        _require_sha256(self.intent_sha256, "validated intent_sha256")
        object.__setattr__(self, "primary_key_columns", primary_key_columns)
        object.__setattr__(self, "column_types", column_types)

    def type_for(self, column: str) -> tuple[str, str]:
        return next(
            (namespace, name)
            for candidate, namespace, name in self.column_types
            if candidate == column
        )

    def intent_payload(self) -> dict[str, object]:
        return {
            "capability_id": _POSTGRESQL_UPDATE_CAPABILITY_ID,
            "source_id": self.source_id,
            "resource_id": self.resource_id,
            "where": tuple(item.to_payload() for item in self.where),
            "assignments": tuple(item.to_payload() for item in self.assignments),
        }


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateValidationResult:
    valid: bool
    validated: ValidatedPostgreSQLUpdate | None
    issues: tuple[SqlValidationIssue, ...]

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if self.valid != (self.validated is not None and not issues):
            raise ValueError("update validation state is inconsistent")
        object.__setattr__(self, "issues", issues)

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


@dataclass(frozen=True, slots=True)
class ValidatedPostgreSQLUpdateScope:
    """A value-free table and assignment-column readiness scope."""

    source_id: str
    resource_id: str
    resource_name: str
    schema_name: str
    relation_name: str
    source_revision: str
    resource_revision: str
    primary_key_columns: tuple[str, ...]
    assignment_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "readiness source_id"),
            (self.resource_id, "readiness resource_id"),
            (self.resource_name, "readiness resource_name"),
            (self.schema_name, "readiness schema_name"),
            (self.relation_name, "readiness relation_name"),
            (self.source_revision, "readiness source_revision"),
            (self.resource_revision, "readiness resource_revision"),
        ):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"{name} must be bounded non-empty text")
        primary_keys = tuple(self.primary_key_columns)
        assignments = tuple(self.assignment_columns)
        if not primary_keys or len(primary_keys) != len(set(primary_keys)):
            raise ValueError("readiness requires a complete primary key")
        if not assignments or len(assignments) != len(set(assignments)):
            raise ValueError("readiness assignment columns are invalid")
        object.__setattr__(self, "primary_key_columns", primary_keys)
        object.__setattr__(self, "assignment_columns", assignments)


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateScopeValidationResult:
    valid: bool
    validated: ValidatedPostgreSQLUpdateScope | None
    issues: tuple[SqlValidationIssue, ...]

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if self.valid != (self.validated is not None and not issues):
            raise ValueError("update readiness validation state is inconsistent")
        object.__setattr__(self, "issues", issues)

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateStatement:
    sql: str
    parameters: tuple[object, ...]
    where_sql: str
    selection_where_sql: str
    where_parameters: tuple[object, ...]
    statement_sha256: str

    def __post_init__(self) -> None:
        if not self.sql or not self.where_sql or not self.selection_where_sql:
            raise ValueError("generated update SQL must be non-empty text")
        _require_sha256(self.statement_sha256, "statement_sha256")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "where_parameters", tuple(self.where_parameters))


def validate_postgresql_update_scope(
    source_id: str,
    resource_id: str,
    assignment_columns: tuple[str, ...],
    *,
    resources: Iterable[ResourceSchema],
) -> PostgreSQLUpdateScopeValidationResult:
    """Resolve one resource/column update-readiness scope from catalog truth."""

    if not isinstance(source_id, str) or not isinstance(resource_id, str):
        raise TypeError("readiness source and resource identities must be text")
    if not isinstance(assignment_columns, tuple):
        raise TypeError("assignment_columns must be a tuple")
    resource, issue = _resolve_resource(source_id, resource_id, resources)
    if issue is not None or resource is None:
        return _invalid_scope(*(issue or ("write_resource_not_writable", "")))
    if not assignment_columns or len(assignment_columns) != len(
        set(assignment_columns)
    ):
        return _invalid_scope(
            "write_assignment_invalid",
            "Readiness assignment columns must be distinct non-empty names.",
        )
    if any(_invalid_column_name(column) for column in assignment_columns):
        return _invalid_scope(
            "write_assignment_invalid",
            "Readiness assignment columns must be distinct non-empty names.",
        )
    validation_issue = _assignment_scope_issue(resource, assignment_columns)
    if validation_issue is not None:
        return _invalid_scope(*validation_issue)
    schema_name, relation_name = _qualified_identity(resource)
    order = {column: index for index, column in enumerate(resource.columns)}
    return PostgreSQLUpdateScopeValidationResult(
        True,
        ValidatedPostgreSQLUpdateScope(
            source_id=source_id,
            resource_id=resource_id,
            resource_name=f"{schema_name}.{relation_name}",
            schema_name=schema_name,
            relation_name=relation_name,
            source_revision=resource.source_revision or "",
            resource_revision=resource.revision or "",
            primary_key_columns=resource.primary_key_columns,
            assignment_columns=tuple(sorted(assignment_columns, key=order.__getitem__)),
        ),
        (),
    )


def validate_postgresql_update_intent(
    intent: PostgreSQLUpdateIntent,
    *,
    resources: Iterable[ResourceSchema],
) -> PostgreSQLUpdateValidationResult:
    """Resolve a structured single-row or bulk update against catalog truth."""

    if not isinstance(intent, PostgreSQLUpdateIntent):
        raise TypeError("intent must be PostgreSQLUpdateIntent")
    resource, issue = _resolve_resource(intent.source_id, intent.resource_id, resources)
    if issue is not None or resource is None:
        return _invalid_update(*(issue or ("write_resource_not_writable", "")))
    assignment_names = tuple(item.column for item in intent.assignments)
    if len(assignment_names) != len(set(assignment_names)):
        return _invalid_update(
            "write_assignment_invalid", "Assignment columns cannot repeat."
        )
    assignment_issue = _assignment_scope_issue(resource, assignment_names)
    if assignment_issue is not None:
        return _invalid_update(*assignment_issue)
    type_by_column = {
        column: (namespace, name)
        for column, namespace, name in resource.column_type_provenance
    }
    nullable_by_column = dict(resource.column_nullability)
    for assignment in intent.assignments:
        issue = _literal_issue(
            assignment.value,
            assignment.column,
            type_by_column,
            nullable_by_column,
            allow_null=True,
            code="write_assignment_invalid",
        )
        if issue is not None:
            return _invalid_update(*issue)
    for predicate in intent.where:
        if predicate.column not in set(resource.columns):
            return _invalid_update(
                "write_filter_invalid",
                "The target selection references an unknown catalog column.",
            )
        provenance = type_by_column.get(predicate.column)
        if provenance is None or nullable_by_column.get(predicate.column) is None:
            return _invalid_update(
                "write_filter_invalid",
                "The catalog lacks admitted filter type or nullability provenance.",
            )
        issue = _filter_issue(predicate, provenance, nullable_by_column)
        if issue is not None:
            return _invalid_update(*issue)
    if (
        len(canonical_json(intent.to_payload()).encode("utf-8"))
        > POSTGRESQL_UPDATE_MAX_CANONICAL_BYTES
    ):
        return _invalid_update(
            "write_plan_too_large",
            "The canonical update plan exceeds 64 KiB.",
        )
    prospective_command = {
        **intent.to_payload(),
        "preview_fingerprint": "sha256:" + "0" * 64,
        "expected_affected_rows": 9_223_372_036_854_775_807,
    }
    if render_approval_arguments(prospective_command) is None:
        return _invalid_update(
            "write_plan_too_large",
            "The exact update command exceeds the approval review bound.",
        )
    schema_name, relation_name = _qualified_identity(resource)
    order = {column: index for index, column in enumerate(resource.columns)}
    assignments = tuple(sorted(intent.assignments, key=lambda item: order[item.column]))
    payload = {
        "capability_id": _POSTGRESQL_UPDATE_CAPABILITY_ID,
        "source_id": intent.source_id,
        "resource_id": intent.resource_id,
        "where": tuple(item.to_payload() for item in intent.where),
        "assignments": tuple(item.to_payload() for item in assignments),
    }
    required_columns = tuple(
        dict.fromkeys(
            (
                *resource.primary_key_columns,
                *(item.column for item in intent.where),
                *(item.column for item in assignments),
            )
        )
    )
    validated = ValidatedPostgreSQLUpdate(
        source_id=intent.source_id,
        resource_id=intent.resource_id,
        resource_name=f"{schema_name}.{relation_name}",
        schema_name=schema_name,
        relation_name=relation_name,
        source_revision=resource.source_revision or "",
        resource_revision=resource.revision or "",
        primary_key_columns=resource.primary_key_columns,
        where=intent.where,
        assignments=assignments,
        column_types=tuple(
            (column, *type_by_column[column]) for column in required_columns
        ),
        intent_sha256=_sha256_json(payload),
    )
    return PostgreSQLUpdateValidationResult(True, validated, ())


def render_postgresql_update_statement(
    validated: ValidatedPostgreSQLUpdate,
) -> PostgreSQLUpdateStatement:
    """Render a parameterized UPDATE from catalog identifiers and typed predicates."""

    if not isinstance(validated, ValidatedPostgreSQLUpdate):
        raise TypeError("validated must be ValidatedPostgreSQLUpdate")
    assignments = ", ".join(
        f"{_identifier(cell.column)} = ${index}"
        for index, cell in enumerate(validated.assignments, start=1)
    )
    where_sql, where_parameters = _render_where(
        validated.where,
        start_index=len(validated.assignments) + 1,
    )
    selection_where_sql, selection_parameters = _render_where(
        validated.where,
        start_index=1,
    )
    if selection_parameters != where_parameters:
        raise RuntimeError("selection parameter rendering changed update values")
    sql = (
        "UPDATE ONLY "
        f"{_identifier(validated.schema_name)}.{_identifier(validated.relation_name)} "
        f"SET {assignments} WHERE {where_sql}"
    )
    shape = {
        "operation": "postgresql_update",
        "schema": validated.schema_name,
        "relation": validated.relation_name,
        "primary_key": validated.primary_key_columns,
        "assignments": tuple(cell.column for cell in validated.assignments),
        "where": tuple(
            {"column": item.column, "operator": item.operator}
            for item in validated.where
        ),
    }
    assignment_parameters = tuple(
        thaw_json(cell.value) for cell in validated.assignments
    )
    return PostgreSQLUpdateStatement(
        sql=sql,
        parameters=(*assignment_parameters, *where_parameters),
        where_sql=where_sql,
        selection_where_sql=selection_where_sql,
        where_parameters=where_parameters,
        statement_sha256=_sha256_json(shape),
    )


def _resolve_resource(
    source_id: str,
    resource_id: str,
    resources: Iterable[ResourceSchema],
) -> tuple[ResourceSchema | None, tuple[str, str] | None]:
    current = tuple(resources)
    if any(not isinstance(item, ResourceSchema) for item in current):
        raise TypeError("resources must contain ResourceSchema records")
    if (
        _CANONICAL_SOURCE_ID.fullmatch(source_id) is None
        or _CANONICAL_RESOURCE_ID.fullmatch(resource_id) is None
    ):
        return None, (
            "write_resource_not_writable",
            "The update must use exact canonical source and resource identifiers.",
        )
    resource = next(
        (
            item
            for item in current
            if item.source_id == source_id and item.resource_id == resource_id
        ),
        None,
    )
    if (
        resource is None
        or resource.resource_kind != "table"
        or not resource.writable
        or resource.revision is None
        or resource.source_revision is None
    ):
        return None, (
            "write_resource_not_writable",
            "The selected resource is not a current cataloged base table.",
        )
    try:
        _qualified_identity(resource)
    except ValueError:
        return None, (
            "write_resource_not_writable",
            "The selected resource lacks exact PostgreSQL relation identity.",
        )
    if not resource.primary_key_columns:
        return None, (
            "write_primary_key_required",
            "PostgreSQL updates require a cataloged primary key for exact target-set revalidation.",
        )
    type_by_column = {
        column: (namespace, name)
        for column, namespace, name in resource.column_type_provenance
    }
    nullable_by_column = dict(resource.column_nullability)
    if any(
        nullable_by_column.get(column) is not False
        or type_by_column.get(column, (None, None))[0] != "pg_catalog"
        or type_by_column.get(column, (None, None))[1] not in _POSTGRESQL_UPDATE_TYPES
        for column in resource.primary_key_columns
    ):
        return None, (
            "write_primary_key_required",
            "The cataloged primary key lacks supported type or nullability provenance.",
        )
    return resource, None


def _assignment_scope_issue(
    resource: ResourceSchema,
    assignment_columns: Sequence[str],
) -> tuple[str, str] | None:
    if not assignment_columns or any(
        _invalid_column_name(column) for column in assignment_columns
    ):
        return (
            "write_assignment_invalid",
            "Assignments must contain distinct bounded column names.",
        )
    columns = set(resource.columns)
    forbidden = (
        set(resource.primary_key_columns)
        | set(resource.identity_columns)
        | set(resource.generated_columns)
    )
    updatable = set(resource.updatable_columns)
    if any(
        column not in columns or column in forbidden or column not in updatable
        for column in assignment_columns
    ):
        return (
            "write_assignment_invalid",
            "Assignments reference an unknown or non-updatable catalog column.",
        )
    type_by_column = {
        column: (namespace, name)
        for column, namespace, name in resource.column_type_provenance
    }
    nullable_by_column = dict(resource.column_nullability)
    if any(
        nullable_by_column.get(column) is None
        or type_by_column.get(column, (None, None))[0] != "pg_catalog"
        or type_by_column.get(column, (None, None))[1] not in _POSTGRESQL_UPDATE_TYPES
        for column in assignment_columns
    ):
        return (
            "write_assignment_invalid",
            "The catalog lacks admitted assignment type or nullability provenance.",
        )
    return None


def _filter_issue(
    predicate: PostgreSQLUpdateFilter,
    provenance: tuple[str, str],
    nullable_by_column: Mapping[str, bool],
) -> tuple[str, str] | None:
    namespace, type_name = provenance
    if namespace != "pg_catalog" or type_name not in _POSTGRESQL_UPDATE_TYPES:
        return (
            "write_filter_invalid",
            "The target selection uses an unsupported PostgreSQL type.",
        )
    if predicate.operator in _NULL_FILTER_OPERATORS:
        if predicate.value is not None:
            return (
                "write_filter_invalid",
                "Null predicates require a null value placeholder.",
            )
        return None
    if predicate.operator in _ORDERED_FILTER_OPERATORS and type_name in {
        "bool",
        "json",
        "jsonb",
    }:
        return (
            "write_filter_invalid",
            "The target selection uses an ordered operator for an unordered type.",
        )
    if type_name == "json" and predicate.operator in {
        "eq",
        "ne",
        "in",
        "not_in",
    }:
        return (
            "write_filter_invalid",
            "PostgreSQL json filters support only null predicates; use jsonb for equality selection.",
        )
    if predicate.operator in _SET_FILTER_OPERATORS:
        values = predicate.value
        if not isinstance(values, tuple) or not values:
            return (
                "write_filter_invalid",
                "Set predicates require a non-empty array of literal values.",
            )
        if len({canonical_json(value) for value in values}) != len(values):
            return (
                "write_filter_invalid",
                "Set predicate values cannot repeat.",
            )
        for value in values:
            issue = _literal_issue(
                value,
                predicate.column,
                {predicate.column: provenance},
                nullable_by_column,
                allow_null=False,
                code="write_filter_invalid",
            )
            if issue is not None:
                return issue
        return None
    return _literal_issue(
        predicate.value,
        predicate.column,
        {predicate.column: provenance},
        nullable_by_column,
        allow_null=False,
        code="write_filter_invalid",
    )


def _literal_issue(
    value: FrozenJsonValue,
    column: str,
    type_by_column: Mapping[str, tuple[str, str]],
    nullable_by_column: Mapping[str, bool],
    *,
    allow_null: bool,
    code: str,
) -> tuple[str, str] | None:
    provenance = type_by_column.get(column)
    if provenance is None or nullable_by_column.get(column) is None:
        return code, "The catalog lacks complete type or nullability provenance."
    if value is None:
        if allow_null and nullable_by_column[column] is True:
            return None
        return code, "The proposed null value is not valid for this operation."
    if not _valid_postgresql_update_value(value, *provenance):
        return (
            code,
            "The proposed literal is incompatible with the cataloged PostgreSQL type.",
        )
    return None


def _render_where(
    predicates: tuple[PostgreSQLUpdateFilter, ...],
    *,
    start_index: int,
) -> tuple[str, tuple[object, ...]]:
    clauses: list[str] = []
    parameters: list[object] = []
    for predicate in predicates:
        column = _identifier(predicate.column)
        if predicate.operator in _NULL_FILTER_OPERATORS:
            clauses.append(
                f"{column} IS {'NOT ' if predicate.operator == 'is_not_null' else ''}NULL"
            )
            continue
        if predicate.operator in _SET_FILTER_OPERATORS:
            assert isinstance(predicate.value, tuple)
            placeholders = []
            for value in predicate.value:
                placeholders.append(f"${start_index + len(parameters)}")
                parameters.append(thaw_json(value))
            keyword = "NOT IN" if predicate.operator == "not_in" else "IN"
            clauses.append(f"{column} {keyword} ({', '.join(placeholders)})")
            continue
        operator = {
            "eq": "=",
            "ne": "<>",
            "lt": "<",
            "lte": "<=",
            "gt": ">",
            "gte": ">=",
        }[predicate.operator]
        clauses.append(f"{column} {operator} ${start_index + len(parameters)}")
        parameters.append(thaw_json(predicate.value))
    return " AND ".join(f"({clause})" for clause in clauses), tuple(parameters)


def _valid_postgresql_update_value(
    value: FrozenJsonValue,
    namespace: str,
    type_name: str,
) -> bool:
    if namespace != "pg_catalog" or type_name not in _POSTGRESQL_UPDATE_TYPES:
        return False
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name in {"int2", "int4", "int8"}:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        lower, upper = {
            "int2": (-32_768, 32_767),
            "int4": (-2_147_483_648, 2_147_483_647),
            "int8": (-9_223_372_036_854_775_808, 9_223_372_036_854_775_807),
        }[type_name]
        return lower <= value <= upper
    if type_name in {"float4", "float8", "numeric"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False
        if not finite or len(canonical_json(value)) > 128:
            return False
        if type_name == "float4":
            return abs(float(value)) <= 3.4028234663852886e38
        if type_name == "float8":
            return abs(float(value)) <= 1.7976931348623157e308
        return True
    if type_name in {"text", "varchar", "bpchar"}:
        return isinstance(value, str) and "\x00" not in value
    if type_name == "uuid":
        if not isinstance(value, str):
            return False
        try:
            UUID(value)
        except (ValueError, AttributeError):
            return False
        return True
    if type_name == "date":
        return _valid_iso_date(value)
    if type_name in {"timestamp", "timestamptz"}:
        return _valid_iso_datetime(value, require_timezone=type_name == "timestamptz")
    if type_name in {"json", "jsonb"}:
        return len(canonical_json(value).encode("utf-8")) <= 64 * 1_024
    return False


def _valid_iso_date(value: FrozenJsonValue) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return parsed.isoformat() == value


def _valid_iso_datetime(
    value: FrozenJsonValue,
    *,
    require_timezone: bool,
) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    aware = parsed.tzinfo is not None and parsed.utcoffset() is not None
    return aware if require_timezone else not aware


def _qualified_identity(resource: ResourceSchema) -> tuple[str, str]:
    qualified = next(
        (
            alias.partition(".")
            for alias in resource.aliases
            if "." in alias and "\x00" not in alias
        ),
        None,
    )
    if (
        qualified is None
        or not qualified[0]
        or not qualified[1]
        or not qualified[2]
        or qualified[2] != resource.name
        or len(qualified[0]) > 256
        or len(qualified[2]) > 256
    ):
        raise ValueError("resource lacks a qualified PostgreSQL identity")
    return qualified[0], qualified[2]


def _invalid_update(code: str, message: str) -> PostgreSQLUpdateValidationResult:
    return PostgreSQLUpdateValidationResult(
        False, None, (SqlValidationIssue(code, message),)
    )


def _invalid_scope(code: str, message: str) -> PostgreSQLUpdateScopeValidationResult:
    return PostgreSQLUpdateScopeValidationResult(
        False, None, (SqlValidationIssue(code, message),)
    )


def _invalid_column_name(value: object) -> bool:
    return (
        not isinstance(value, str) or not value or len(value) > 256 or "\x00" in value
    )


def _require_column(value: str, name: str) -> None:
    if _invalid_column_name(value):
        raise ValueError(f"{name} must be bounded non-empty text")


def _identifier(value: str) -> str:
    _require_column(value, "PostgreSQL identifier")
    return '"' + value.replace('"', '""') + '"'


def _sha256_json(value: object) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 hash")


__all__ = [
    "POSTGRESQL_UPDATE_MAX_CANONICAL_BYTES",
    "PostgreSQLUpdateCell",
    "PostgreSQLUpdateCommand",
    "PostgreSQLUpdateFilter",
    "PostgreSQLUpdateIntent",
    "PostgreSQLUpdateScopeValidationResult",
    "PostgreSQLUpdateStatement",
    "PostgreSQLUpdateValidationResult",
    "ValidatedPostgreSQLUpdate",
    "ValidatedPostgreSQLUpdateScope",
    "render_postgresql_update_statement",
    "validate_postgresql_update_intent",
    "validate_postgresql_update_scope",
]
