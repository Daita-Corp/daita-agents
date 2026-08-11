"""Typed PostgreSQL single-row update intent validation and rendering."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

from ...._json import (
    FrozenJsonObject,
    FrozenJsonValue,
    canonical_json,
    freeze_json,
    thaw_json,
)
from .contracts import ResourceSchema, SqlValidationIssue

POSTGRESQL_UPDATE_MAX_ASSIGNMENTS = 32
POSTGRESQL_UPDATE_MAX_CANONICAL_BYTES = 64 * 1_024
_POSTGRESQL_UPDATE_MAX_VALUE_DEPTH = 32
_POSTGRESQL_UPDATE_CAPABILITY_ID = "data.postgresql.update_impact"
_CANONICAL_SOURCE_ID = re.compile(r"source:sha256:[0-9a-f]{64}\Z")
_CANONICAL_RESOURCE_ID = re.compile(r"catalog-resource:sha256:[0-9a-f]{64}\Z")
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
    """One exact catalog-column/literal pair in a typed update intent."""

    column: str
    value: FrozenJsonValue

    def __post_init__(self) -> None:
        if (
            not isinstance(self.column, str)
            or not self.column
            or len(self.column) > 256
            or "\x00" in self.column
        ):
            raise ValueError("update cell column must be bounded non-empty text")
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
class PostgreSQLUpdateIntent:
    """Model-proposed structured intent before catalog-owned validation."""

    source_id: str
    resource_id: str
    match: tuple[PostgreSQLUpdateCell, ...]
    assignments: tuple[PostgreSQLUpdateCell, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "update source_id"),
            (self.resource_id, "update resource_id"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        match = tuple(self.match)
        assignments = tuple(self.assignments)
        if any(not isinstance(item, PostgreSQLUpdateCell) for item in match):
            raise TypeError("update match must contain PostgreSQLUpdateCell records")
        if any(not isinstance(item, PostgreSQLUpdateCell) for item in assignments):
            raise TypeError(
                "update assignments must contain PostgreSQLUpdateCell records"
            )
        object.__setattr__(self, "match", match)
        object.__setattr__(self, "assignments", assignments)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PostgreSQLUpdateIntent:
        if not isinstance(value, Mapping):
            raise TypeError("PostgreSQL update intent must be an object")
        if set(value) != {"source_id", "resource_id", "match", "assignments"}:
            raise ValueError(
                "PostgreSQL update intent requires only source_id, resource_id, "
                "match, and assignments"
            )
        source_id = value.get("source_id")
        resource_id = value.get("resource_id")
        match = value.get("match")
        assignments = value.get("assignments")
        if not isinstance(source_id, str) or not isinstance(resource_id, str):
            raise TypeError("PostgreSQL update identities must be text")
        if not isinstance(match, (tuple, list)) or not isinstance(
            assignments, (tuple, list)
        ):
            raise TypeError("PostgreSQL update cells must be arrays")
        return cls(
            source_id=source_id,
            resource_id=resource_id,
            match=tuple(PostgreSQLUpdateCell.from_mapping(item) for item in match),
            assignments=tuple(
                PostgreSQLUpdateCell.from_mapping(item) for item in assignments
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "resource_id": self.resource_id,
            "match": tuple(item.to_payload() for item in self.match),
            "assignments": tuple(item.to_payload() for item in self.assignments),
        }


@dataclass(frozen=True, slots=True)
class PostgreSQLUpdateCommand:
    """Exact approved update invocation, including preview and row bound."""

    intent: PostgreSQLUpdateIntent
    preview_fingerprint: str
    max_affected_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.intent, PostgreSQLUpdateIntent):
            raise TypeError("update command intent must be PostgreSQLUpdateIntent")
        _require_sha256(self.preview_fingerprint, "preview_fingerprint")
        if (
            not isinstance(self.max_affected_rows, int)
            or isinstance(self.max_affected_rows, bool)
            or self.max_affected_rows != 1
        ):
            raise ValueError("max_affected_rows must be exactly one")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PostgreSQLUpdateCommand:
        if not isinstance(value, Mapping):
            raise TypeError("PostgreSQL update command must be an object")
        expected = {
            "source_id",
            "resource_id",
            "match",
            "assignments",
            "preview_fingerprint",
            "max_affected_rows",
        }
        if set(value) != expected:
            raise ValueError(
                "PostgreSQL update command requires the exact typed update, "
                "preview_fingerprint, and max_affected_rows"
            )
        preview_fingerprint = value.get("preview_fingerprint")
        max_affected_rows = value.get("max_affected_rows")
        if not isinstance(preview_fingerprint, str):
            raise TypeError("preview_fingerprint must be text")
        return cls(
            intent=PostgreSQLUpdateIntent.from_mapping(
                {
                    "source_id": value.get("source_id"),
                    "resource_id": value.get("resource_id"),
                    "match": value.get("match"),
                    "assignments": value.get("assignments"),
                }
            ),
            preview_fingerprint=preview_fingerprint,
            max_affected_rows=max_affected_rows,  # type: ignore[arg-type]
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **self.intent.to_payload(),
            "preview_fingerprint": self.preview_fingerprint,
            "max_affected_rows": self.max_affected_rows,
        }


@dataclass(frozen=True, slots=True)
class ValidatedPostgreSQLUpdate:
    """Catalog-bound, canonical one-row PostgreSQL update proposal."""

    source_id: str
    resource_id: str
    resource_name: str
    schema_name: str
    relation_name: str
    source_revision: str
    resource_revision: str
    match: tuple[PostgreSQLUpdateCell, ...]
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
        if len(self.schema_name) > 256 or len(self.relation_name) > 256:
            raise ValueError("validated PostgreSQL identifiers must be bounded")
        if len(self.match) != 1 or not self.assignments:
            raise ValueError("validated update must contain one match and assignments")
        column_types = tuple(self.column_types)
        expected_columns = tuple(
            item.column for item in (*self.match, *self.assignments)
        )
        if tuple(item[0] for item in column_types) != expected_columns:
            raise ValueError("validated update column types must preserve cell order")
        if any(
            len(item) != 3
            or not all(isinstance(value, str) and value for value in item)
            for item in column_types
        ):
            raise ValueError("validated update column types are invalid")
        _require_sha256(self.intent_sha256, "validated intent_sha256")
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
            "match": tuple(item.to_payload() for item in self.match),
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
    """Catalog-bound resource and assignment columns for readiness checks."""

    source_id: str
    resource_id: str
    resource_name: str
    schema_name: str
    relation_name: str
    source_revision: str
    resource_revision: str
    primary_key_column: str
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
            (self.primary_key_column, "readiness primary_key_column"),
        ):
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"{name} must be bounded non-empty text")
        assignment_columns = tuple(self.assignment_columns)
        if (
            not 1 <= len(assignment_columns) <= POSTGRESQL_UPDATE_MAX_ASSIGNMENTS
            or len(assignment_columns) != len(set(assignment_columns))
            or any(
                not isinstance(column, str)
                or not column
                or len(column) > 256
                or "\x00" in column
                for column in assignment_columns
            )
        ):
            raise ValueError("readiness assignment columns are invalid")
        object.__setattr__(self, "assignment_columns", assignment_columns)


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
    statement_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.sql, str) or not self.sql:
            raise ValueError("generated update SQL must be non-empty text")
        _require_sha256(self.statement_sha256, "statement_sha256")
        object.__setattr__(self, "parameters", tuple(self.parameters))


def validate_postgresql_update_scope(
    source_id: str,
    resource_id: str,
    assignment_columns: tuple[str, ...],
    *,
    resources: Iterable[ResourceSchema],
) -> PostgreSQLUpdateScopeValidationResult:
    """Resolve a value-free update-readiness scope from current catalog truth."""

    if not isinstance(source_id, str) or not isinstance(resource_id, str):
        raise TypeError("readiness source and resource identities must be text")
    if not isinstance(assignment_columns, tuple):
        raise TypeError("assignment_columns must be a tuple")
    current = tuple(resources)
    if any(not isinstance(item, ResourceSchema) for item in current):
        raise TypeError("resources must contain ResourceSchema records")
    if (
        _CANONICAL_SOURCE_ID.fullmatch(source_id) is None
        or _CANONICAL_RESOURCE_ID.fullmatch(resource_id) is None
    ):
        return _invalid_postgresql_update_scope(
            "write_resource_not_writable",
            "Readiness requires exact canonical source and resource identifiers.",
        )
    if (
        not 1 <= len(assignment_columns) <= POSTGRESQL_UPDATE_MAX_ASSIGNMENTS
        or len(assignment_columns) != len(set(assignment_columns))
        or any(
            not isinstance(column, str)
            or not column
            or len(column) > 256
            or "\x00" in column
            for column in assignment_columns
        )
    ):
        return _invalid_postgresql_update_scope(
            "write_assignment_invalid",
            "Readiness assignment columns must be one through 32 distinct names.",
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
        return _invalid_postgresql_update_scope(
            "write_resource_not_writable",
            "The selected resource is not a current cataloged base table.",
        )
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
        return _invalid_postgresql_update_scope(
            "write_resource_not_writable",
            "The selected resource lacks exact PostgreSQL relation identity.",
        )
    if len(resource.primary_key_columns) != 1:
        return _invalid_postgresql_update_scope(
            "write_primary_key_required",
            "PostgreSQL update readiness requires one single-column primary key.",
        )
    primary_key = resource.primary_key_columns[0]
    columns = set(resource.columns)
    forbidden = set(resource.identity_columns) | set(resource.generated_columns)
    updatable = set(resource.updatable_columns)
    if any(
        column not in columns
        or column == primary_key
        or column in forbidden
        or column not in updatable
        for column in assignment_columns
    ):
        return _invalid_postgresql_update_scope(
            "write_assignment_invalid",
            "Readiness references an unknown or non-updatable catalog column.",
        )
    type_by_column = {
        column: (namespace, name)
        for column, namespace, name in resource.column_type_provenance
    }
    nullable_by_column = dict(resource.column_nullability)
    required_columns = (primary_key, *assignment_columns)
    if nullable_by_column.get(primary_key) is not False or any(
        nullable_by_column.get(column) is None
        or type_by_column.get(column, (None, None))[0] != "pg_catalog"
        or type_by_column.get(column, (None, None))[1] not in _POSTGRESQL_UPDATE_TYPES
        for column in required_columns
    ):
        return _invalid_postgresql_update_scope(
            "write_assignment_invalid",
            "The catalog lacks admitted update type or nullability provenance.",
        )
    order = {column: index for index, column in enumerate(resource.columns)}
    ordered_assignments = tuple(sorted(assignment_columns, key=order.__getitem__))
    return PostgreSQLUpdateScopeValidationResult(
        True,
        ValidatedPostgreSQLUpdateScope(
            source_id=source_id,
            resource_id=resource_id,
            resource_name=f"{qualified[0]}.{qualified[2]}",
            schema_name=qualified[0],
            relation_name=qualified[2],
            source_revision=resource.source_revision,
            resource_revision=resource.revision,
            primary_key_column=primary_key,
            assignment_columns=ordered_assignments,
        ),
        (),
    )


def validate_postgresql_update_intent(
    intent: PostgreSQLUpdateIntent,
    *,
    resources: Iterable[ResourceSchema],
) -> PostgreSQLUpdateValidationResult:
    """Resolve one literal update proposal against current catalog truth."""

    if not isinstance(intent, PostgreSQLUpdateIntent):
        raise TypeError("intent must be PostgreSQLUpdateIntent")
    current = tuple(resources)
    if any(not isinstance(item, ResourceSchema) for item in current):
        raise TypeError("resources must contain ResourceSchema records")
    if (
        _CANONICAL_SOURCE_ID.fullmatch(intent.source_id) is None
        or _CANONICAL_RESOURCE_ID.fullmatch(intent.resource_id) is None
    ):
        return _invalid_postgresql_update(
            "write_resource_not_writable",
            "The update must use exact canonical source and resource identifiers.",
        )
    resource = next(
        (
            item
            for item in current
            if item.source_id == intent.source_id
            and item.resource_id == intent.resource_id
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
        return _invalid_postgresql_update(
            "write_resource_not_writable",
            "The selected resource is not a current cataloged base table.",
        )
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
        return _invalid_postgresql_update(
            "write_resource_not_writable",
            "The selected resource lacks exact PostgreSQL relation identity.",
        )
    if len(resource.primary_key_columns) != 1:
        return _invalid_postgresql_update(
            "write_primary_key_required",
            "PostgreSQL update preview requires one single-column primary key.",
        )
    primary_key = resource.primary_key_columns[0]
    if (
        len(intent.match) != 1
        or intent.match[0].column != primary_key
        or intent.match[0].value is None
    ):
        return _invalid_postgresql_update(
            "write_match_invalid",
            "The match must exactly name the complete ordered primary key.",
        )
    if not 1 <= len(intent.assignments) <= POSTGRESQL_UPDATE_MAX_ASSIGNMENTS:
        return _invalid_postgresql_update(
            "write_assignment_invalid",
            "Assignments must contain from one through 32 literal cells.",
        )
    assignment_names = tuple(item.column for item in intent.assignments)
    if len(assignment_names) != len(set(assignment_names)):
        return _invalid_postgresql_update(
            "write_assignment_invalid",
            "Assignment columns cannot repeat.",
        )
    columns = set(resource.columns)
    forbidden = set(resource.identity_columns) | set(resource.generated_columns)
    updatable = set(resource.updatable_columns)
    if any(
        column not in columns
        or column == primary_key
        or column in forbidden
        or column not in updatable
        for column in assignment_names
    ):
        return _invalid_postgresql_update(
            "write_assignment_invalid",
            "Assignments reference an unknown or non-updatable catalog column.",
        )
    type_by_column = {
        column: (namespace, name)
        for column, namespace, name in resource.column_type_provenance
    }
    nullable_by_column = dict(resource.column_nullability)
    if nullable_by_column.get(primary_key) is not False:
        return _invalid_postgresql_update(
            "write_primary_key_required",
            "The cataloged primary key must be explicitly non-null.",
        )
    cells = (*intent.match, *intent.assignments)
    for cell in cells:
        type_provenance = type_by_column.get(cell.column)
        if type_provenance is None or nullable_by_column.get(cell.column) is None:
            code = (
                "write_match_invalid"
                if cell.column == primary_key
                else "write_assignment_invalid"
            )
            return _invalid_postgresql_update(
                code,
                "The catalog lacks complete type or nullability provenance.",
            )
        if cell.value is None:
            if (
                cell.column == primary_key
                or nullable_by_column[cell.column] is not True
            ):
                code = (
                    "write_match_invalid"
                    if cell.column == primary_key
                    else "write_assignment_invalid"
                )
                return _invalid_postgresql_update(
                    code,
                    "The proposed null value violates current catalog nullability.",
                )
            continue
        if not _valid_postgresql_update_value(cell.value, *type_provenance):
            code = (
                "write_match_invalid"
                if cell.column == primary_key
                else "write_assignment_invalid"
            )
            return _invalid_postgresql_update(
                code,
                "The proposed literal is incompatible with the cataloged PostgreSQL type.",
            )
    if (
        len(canonical_json(intent.to_payload()).encode("utf-8"))
        > POSTGRESQL_UPDATE_MAX_CANONICAL_BYTES
    ):
        return _invalid_postgresql_update(
            "write_assignment_invalid",
            "The canonical update intent exceeds 64 KiB.",
        )
    order = {column: index for index, column in enumerate(resource.columns)}
    assignments = tuple(sorted(intent.assignments, key=lambda item: order[item.column]))
    match = intent.match
    intent_payload = {
        "capability_id": _POSTGRESQL_UPDATE_CAPABILITY_ID,
        "source_id": intent.source_id,
        "resource_id": intent.resource_id,
        "match": tuple(item.to_payload() for item in match),
        "assignments": tuple(item.to_payload() for item in assignments),
    }
    intent_sha256 = _sha256_json(intent_payload)
    ordered_cells = (*match, *assignments)
    validated = ValidatedPostgreSQLUpdate(
        source_id=intent.source_id,
        resource_id=intent.resource_id,
        resource_name=f"{qualified[0]}.{qualified[2]}",
        schema_name=qualified[0],
        relation_name=qualified[2],
        source_revision=resource.source_revision,
        resource_revision=resource.revision,
        match=match,
        assignments=assignments,
        column_types=tuple(
            (cell.column, *type_by_column[cell.column]) for cell in ordered_cells
        ),
        intent_sha256=intent_sha256,
    )
    return PostgreSQLUpdateValidationResult(True, validated, ())


def render_postgresql_update_statement(
    validated: ValidatedPostgreSQLUpdate,
) -> PostgreSQLUpdateStatement:
    """Render the fixed update shape from catalog-owned identifiers only."""

    if not isinstance(validated, ValidatedPostgreSQLUpdate):
        raise TypeError("validated must be ValidatedPostgreSQLUpdate")
    assignments = ", ".join(
        f"{_postgresql_update_identifier(cell.column)} = ${index}"
        for index, cell in enumerate(validated.assignments, start=1)
    )
    match_index = len(validated.assignments) + 1
    primary_key = validated.match[0]
    sql = (
        "UPDATE ONLY "
        f"{_postgresql_update_identifier(validated.schema_name)}."
        f"{_postgresql_update_identifier(validated.relation_name)} "
        f"SET {assignments} WHERE "
        f"{_postgresql_update_identifier(primary_key.column)} = ${match_index} "
        "RETURNING "
        + ", ".join(
            _postgresql_update_identifier(cell.column)
            for cell in (primary_key, *validated.assignments)
        )
    )
    shape = {
        "operation": "postgresql_update_one",
        "schema": validated.schema_name,
        "relation": validated.relation_name,
        "assignments": tuple(cell.column for cell in validated.assignments),
        "match": tuple(cell.column for cell in validated.match),
        "parameter_order": (
            *(cell.column for cell in validated.assignments),
            *(cell.column for cell in validated.match),
        ),
        "returning": (
            primary_key.column,
            *(cell.column for cell in validated.assignments),
        ),
    }
    return PostgreSQLUpdateStatement(
        sql=sql,
        parameters=tuple(
            thaw_json(cell.value) for cell in (*validated.assignments, *validated.match)
        ),
        statement_sha256=_sha256_json(shape),
    )


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
            return str(UUID(value)) == value.casefold()
        except ValueError:
            return False
    if type_name == "date":
        if not isinstance(value, str):
            return False
        try:
            return date.fromisoformat(value).isoformat() == value
        except ValueError:
            return False
    if type_name in {"timestamp", "timestamptz"}:
        if not isinstance(value, str):
            return False
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return False
        return (parsed.tzinfo is None) is (type_name == "timestamp")
    if type_name in {"json", "jsonb"}:
        return _json_value_depth(value) <= _POSTGRESQL_UPDATE_MAX_VALUE_DEPTH
    return False


def _json_value_depth(value: FrozenJsonValue, *, current: int = 0) -> int:
    if isinstance(value, FrozenJsonObject):
        return max(
            (
                current,
                *(
                    _json_value_depth(item, current=current + 1)
                    for item in value.values()
                ),
            )
        )
    if isinstance(value, tuple):
        return max(
            (current, *(_json_value_depth(item, current=current + 1) for item in value))
        )
    return current


def _invalid_postgresql_update(
    code: str,
    message: str,
) -> PostgreSQLUpdateValidationResult:
    return PostgreSQLUpdateValidationResult(
        False,
        None,
        (SqlValidationIssue(code, message),),
    )


def _invalid_postgresql_update_scope(
    code: str,
    message: str,
) -> PostgreSQLUpdateScopeValidationResult:
    return PostgreSQLUpdateScopeValidationResult(
        False,
        None,
        (SqlValidationIssue(code, message),),
    )


def _postgresql_update_identifier(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 256:
        raise ValueError("PostgreSQL identifier must be bounded catalog text")
    return '"' + value.replace('"', '""') + '"'


def _sha256_json(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be a canonical sha256 hash")
