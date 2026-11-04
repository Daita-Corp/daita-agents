"""PostgreSQL's write language: early refusals, Daita's RETURNING, the preview SELECT.

The parser never has the last word: the source reports what the exact text really
targets. The derived ``SELECT`` is the claim the card shows: its ``ctid`` and ``xmin``
tie preview to apply, and it runs read-only so no trigger fires.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._json import canonical_json
from ..catalog.models import ResourceKind, catalog_resource_id
from ..domains.data.sql.analysis import (
    _load_sqlglot,
    analyze_postgresql_write,
    stash_escape_strings,
)
from ..domains.data.sql.contracts import (
    SqlAnalysis,
    SqlAnalysisError,
    SqlValidationIssue,
    WriteFacts,
)
from ..domains.data.sql.read_validation import (
    check_postgresql_relation,
    fold_postgresql,
    function_issues,
    postgresql_function_lookups,
    resource_id_in_scope,
)

WRITE_MAX_BYTES = 64 * 1_024
WRITE_MAX_VALUES_ROWS = 1_000
SAMPLE_CELL_CHARACTERS = 1_024
COLUMN_DEFAULT = "(column default)"
COMPUTED_AT_WRITE = "(computed when written)"
# Parser keywords, not pg_proc entries, so the function lookup cannot place them.
_TIME_KEYWORDS = frozenset(
    {
        "CURRENT_DATE",
        "CURRENT_TIME",
        "CURRENT_TIMESTAMP",
        "LOCALTIME",
        "LOCALTIMESTAMP",
    }
)
_TEXT = "pg_catalog.text"


class _Refused(Exception):
    def __init__(self, code: str, message: str, details: Mapping[str, object] = {}):
        super().__init__(message)
        self.issue = SqlValidationIssue(code, message, details)


def _cannot_preview(message: str) -> _Refused:
    return _Refused(
        "write_preview_failed", f"This write cannot be previewed. {message}"
    )


@dataclass(frozen=True, slots=True)
class PostgreSQLStatement:
    """What the backend needs to run one write; only this adapter reads it."""

    canonical_sql: str
    parameters: tuple[object, ...]
    matches_all: bool
    uses_time: bool
    function_lookups: tuple[str, ...]
    attribute_lookups: tuple[str, ...]
    written_calls: tuple[str, ...]
    unqualified_relations: tuple[str, ...]

    @classmethod
    def of(cls, facts: WriteFacts) -> PostgreSQLStatement:
        return cls(**facts.statement)  # type: ignore[arg-type]


def describe_write(
    sql: str,
    *,
    source_id: str,
    parameters: Sequence[object] = (),
    allowed_resource_ids: frozenset[str] | None = None,
    schemas: tuple[str, ...] = (),
) -> WriteFacts | tuple[SqlValidationIssue, ...]:
    """Parse for early errors, before any source I/O, and report the write's facts."""

    try:
        analysis, root = analyze_postgresql_write(sql)
    except SqlAnalysisError as error:
        return (SqlValidationIssue(error.code, str(error)),)
    _, exp = _load_sqlglot("postgresql")
    try:
        shape = _statement_shape(root, analysis, exp)
        full_name, resource_id = _target_identity(shape.table, source_id, schemas, exp)
    except _Refused as error:
        return (error.issue,)

    issues = list(function_issues(analysis, "postgresql"))
    size = len(
        canonical_json(
            {"sql": analysis.canonical_sql, "parameters": parameters}
        ).encode("utf-8")
    )
    if size > WRITE_MAX_BYTES:
        issues.append(
            SqlValidationIssue(
                "write_too_large",
                "A write's SQL and parameters must fit in 64 KiB.",
                {"utf8_bytes": size},
            )
        )
    selected: dict[str, str] = {}
    unqualified: list[str] = []
    for table in analysis.tables:
        if table.is_cte:
            continue
        check = check_postgresql_relation(
            table, source_id, allowed_resource_ids, schemas
        )
        if isinstance(check, SqlValidationIssue):
            issues.append(check)
        elif check is None:
            unqualified.append(fold_postgresql(table.name, table.name_quoted))
        else:
            selected.setdefault(check[0], check[1])
    # A CTE can share the target's name, and it is never the target.
    if resource_id_in_scope(source_id, full_name, allowed_resource_ids) is None:
        issues.append(
            SqlValidationIssue(
                "resource_out_of_scope",
                f"{full_name} is not available for reading in this run.",
                {"relation": full_name},
            )
        )
    if issues:
        return tuple(issues)
    selected.setdefault(resource_id, full_name)
    statement = PostgreSQLStatement(
        canonical_sql=analysis.canonical_sql,
        parameters=tuple(parameters),
        matches_all=shape.matches_all,
        uses_time=bool(_TIME_KEYWORDS.intersection(analysis.function_names)),
        function_lookups=postgresql_function_lookups(analysis),
        attribute_lookups=analysis.attribute_names,
        written_calls=analysis.written_call_names,
        unqualified_relations=tuple(dict.fromkeys(unqualified)),
    )
    return WriteFacts(
        table=full_name,
        resource_id=resource_id,
        operation=shape.operation,
        insert_columns=shape.insert_columns,
        update_columns=shape.update_columns,
        statement={name: getattr(statement, name) for name in statement.__slots__},
        resource_ids=tuple(selected),
        resource_names=tuple(selected.items()),
        allowed_resource_ids=(
            None
            if allowed_resource_ids is None
            else tuple(sorted(allowed_resource_ids))
        ),
    )


class _Shape:
    """What the policy and the derived SELECT read from one parsed statement."""

    def __init__(self, operation: str, root: Any, table: Any) -> None:
        self.operation = operation
        self.root = root
        self.table = table
        self.insert_columns: tuple[str, ...] = ()
        self.update_columns: tuple[str, ...] = ()
        # (column as written, value expression) for every assignment.
        self.assignments: tuple[tuple[Any, Any], ...] = ()
        self.insert_identifiers: tuple[Any, ...] = ()
        self.conflict: Any = None
        self.matches_all = False


def _statement_shape(root: Any, analysis: SqlAnalysis, exp: Any) -> _Shape:
    if analysis.statement_count != 1:
        raise _Refused(
            "multiple_statements",
            "Exactly one SQL statement is allowed.",
            {"statement_count": analysis.statement_count},
        )
    operations = {exp.Insert: "insert", exp.Update: "update", exp.Delete: "delete"}
    operation = operations.get(type(root))
    if operation is None or analysis.statement_type == "explain":
        raise _Refused(
            "write_statement_required",
            "data_write runs one INSERT, UPDATE or DELETE. DDL, TRUNCATE, COPY and "
            "MERGE are refused; use data_query to read.",
            {"statement_type": analysis.statement_type},
        )
    mutations = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Merge,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.TruncateTable,
        exp.Into,
        exp.Lock,
    )
    nested = sorted(
        {
            type(node).__name__.upper()
            for node in root.walk()
            if node is not root and isinstance(node, mutations)
        }
    )
    if nested:
        raise _Refused(
            "nested_write_not_allowed",
            "A write is one statement: data-modifying CTEs, SELECT INTO and locking "
            "clauses are refused.",
            {"found": nested},
        )
    table = root.this.this if isinstance(root.this, exp.Schema) else root.this
    if not isinstance(table, exp.Table) or not isinstance(table.this, exp.Identifier):
        raise _Refused("write_target_invalid", "The write target must be one table.")

    shape = _Shape(operation, root, table)
    if operation == "insert":
        _read_insert(shape, exp)
    else:
        where = root.args.get("where")
        shape.matches_all = (
            where is None or where.find(exp.Column, exp.Subquery, exp.Parameter) is None
        )
        if operation == "update":
            shape.assignments = _assignments(root.expressions, exp)
            shape.update_columns = _distinct_columns(shape.assignments)
    return shape


def _read_insert(shape: _Shape, exp: Any) -> None:
    root, table = shape.root, shape.table
    alias = table.args.get("alias")
    identifiers = (
        tuple(root.this.expressions)
        if isinstance(root.this, exp.Schema)
        else tuple(alias.args.get("columns") or ()) if alias is not None else ()
    )
    if not identifiers:
        raise _Refused(
            "insert_columns_required", "INSERT must list the columns it writes."
        )
    if root.expression is None:
        raise _cannot_preview("INSERT here needs VALUES or a SELECT.")
    shape.insert_identifiers = identifiers
    shape.insert_columns = tuple(
        fold_postgresql(item.name, bool(item.args.get("quoted")))
        for item in identifiers
    )
    if isinstance(root.expression, exp.Values):
        rows = tuple(root.expression.expressions)
        if len(rows) > WRITE_MAX_VALUES_ROWS:
            raise _Refused(
                "write_too_large",
                f"A batch can insert at most {WRITE_MAX_VALUES_ROWS} rows.",
                {"rows": len(rows)},
            )
    conflict = root.args.get("conflict")
    if conflict is None:
        return
    keys = tuple(conflict.args.get("conflict_keys") or ())
    if (
        conflict.args.get("constraint") is not None
        or conflict.args.get("index_predicate") is not None
        or not keys
        or any(not isinstance(key.this, exp.Column) for key in keys)
    ):
        raise _cannot_preview(
            "ON CONFLICT must name plain conflict columns, as in ON CONFLICT (sku), "
            "so the preview can split inserts from updates."
        )
    shape.conflict = conflict
    if str(conflict.args["action"]).upper().endswith("UPDATE"):
        shape.assignments = _assignments(conflict.expressions, exp)
        shape.update_columns = _distinct_columns(shape.assignments)


def _assignments(items: Sequence[Any], exp: Any) -> tuple[tuple[Any, Any], ...]:
    """Each assigned column with its value; a tuple assignment is read pairwise."""

    pairs: list[tuple[Any, Any]] = []
    for item in items:
        target, value = item.this, item.expression
        if isinstance(target, exp.Tuple):
            if not isinstance(value, exp.Tuple) or len(value.expressions) != len(
                target.expressions
            ):
                raise _cannot_preview(
                    "Assign each column on its own, or as (a, b) = (1, 2)."
                )
            pairs.extend(zip(target.expressions, value.expressions, strict=True))
        else:
            pairs.append((target, value))
    for target, _value in pairs:
        if not isinstance(target, exp.Column) or str(target.table or ""):
            raise _cannot_preview("SET must assign whole columns by name.")
    return tuple(pairs)


def _distinct_columns(pairs: Sequence[tuple[Any, Any]]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            fold_postgresql(target.name, bool(target.this.args.get("quoted")))
            for target, _value in pairs
        )
    )


def _target_identity(
    table: Any, source_id: str, schemas: tuple[str, ...], exp: Any
) -> tuple[str, str]:
    parts = tuple(
        fold_postgresql(part.name, bool(part.args.get("quoted")))
        for part in table.parts
    )
    if len(parts) == 1:
        if len(schemas) != 1:
            raise _Refused(
                "unqualified_relation",
                f"{parts[0]} is unqualified and this source has several schemas; "
                f"qualify it with one of {', '.join(schemas)}.",
                {"schemas": schemas},
            )
        parts = (schemas[0], parts[0])
    if len(parts) != 2:
        raise _Refused("write_target_invalid", "The write target must be one table.")
    if parts[0] not in schemas:
        raise _Refused(
            "schema_not_configured",
            f"{'.'.join(parts)} is in schema {parts[0]}, which is not attached to "
            "this source. Add the schema to the source to write it.",
            {"relation": ".".join(parts), "schema": parts[0]},
        )
    full_name = ".".join(parts)
    return full_name, catalog_resource_id(source_id, ResourceKind.TABLE, full_name)


def _parsed(sql: str) -> tuple[SqlAnalysis, _Shape, Any]:
    """Text that already passed ``describe_write``, parsed for building from."""

    analysis, root = analyze_postgresql_write(sql)
    _, exp = _load_sqlglot("postgresql")
    return analysis, _statement_shape(root, analysis, exp), exp


def verified_statement(
    sql: str, keys: Sequence[Sequence[str]]
) -> tuple[str, tuple[str, ...] | None]:
    """The exact text plus Daita's RETURNING, and the key that identifies its rows.

    RETURNING sits on its own line so a trailing comment cannot swallow it. Without a
    usable key the statement reports only what happened to each row. A key stops being
    usable when the statement assigns it, since the returned value is the new one, or
    when an INSERT does not list it, since the preview cannot know a generated value.
    """

    analysis, shape, exp = _parsed(sql)
    key = next(
        (
            tuple(columns)
            for columns in keys
            if not set(columns) & set(shape.update_columns)
            and (
                shape.operation != "insert" or set(columns) <= set(shape.insert_columns)
            )
        ),
        None,
    )
    name = _target_name(shape.table)
    identity = (
        f"CAST({name}.ctid AS {_TEXT})"
        if shape.operation == "delete"
        else _key_cell(name, key, exp)
    )
    # Only an INSERT can ask: xmax means nothing on the rows of a plain UPDATE.
    inserted = f"({name}.xmax = 0)" if shape.operation == "insert" else "FALSE"
    return (
        f"{analysis.canonical_sql}\nRETURNING {identity} AS __daita_key, "
        f"{inserted} AS __daita_inserted",
        key,
    )


def derived_select(
    sql: str, types: Mapping[str, str], key: Sequence[str] | None
) -> tuple[str, str | None]:
    """The preview SELECT and, where rows already exist, the one that locks them.

    Column types are only known live, so this runs after the source described the
    statement. It is a claim: a wrong one fails preview or fails verification. It is
    rendered from the tree, so literals the tree would change stay out of it.
    """

    protected, stashed = stash_escape_strings(sql)
    _, shape, exp = _parsed(protected)
    scan, lock = _derived_select(shape, types, key, exp)
    for placeholder, literal in stashed.items():
        scan = scan.replace(placeholder, literal)
        lock = None if lock is None else lock.replace(placeholder, literal)
    return scan, lock


def _derived_select(
    shape: _Shape, types: Mapping[str, str], key: Sequence[str] | None, exp: Any
) -> tuple[str, str | None]:
    root, table = shape.root, shape.table
    alias = table.args.get("alias")
    name = _target_name(table)
    relation = ".".join(part.sql(dialect="postgres") for part in table.parts)
    if alias is not None and alias.this:
        relation += " AS " + name

    def cast(expression: str, column: str) -> str:
        return f"CAST({expression} AS {types[column]})"

    row = (
        f"CAST({name}.ctid AS {_TEXT}) AS __daita_row",
        f"CAST({name}.xmin AS {_TEXT}) AS __daita_version",
    )
    no_row = (
        f"CAST(NULL AS {_TEXT}) AS __daita_row",
        f"CAST(NULL AS {_TEXT}) AS __daita_version",
    )
    update_cells, update_bound = _assignment_cells(shape, name, types, exp)
    with_sql = ""
    if root.args.get("with_") is not None:
        with_sql = root.args["with_"].sql(dialect="postgres", comments=False) + " "

    if shape.operation == "update":
        from_ = root.args.get("from_")
        sources = relation + (
            ""
            if from_ is None
            else ", " + from_.sql(dialect="postgres").removeprefix("FROM ")
        )
        select = _select(
            with_sql,
            (
                *row,
                "'update' AS __daita_action",
                f"{_key_cell(name, key, exp)} AS __daita_key",
                _bound(update_bound),
                *update_cells,
            ),
            sources,
            _where(root),
            f"{name}.ctid",
        )
        return select, f"{select} FOR UPDATE OF {name}"
    if shape.operation == "delete":
        sources = ", ".join(
            (
                relation,
                *(
                    item.sql(dialect="postgres", comments=False)
                    for item in root.args.get("using") or ()
                ),
            )
        )
        select = _select(
            with_sql,
            (
                *row,
                "'delete' AS __daita_action",
                f"CAST({name}.ctid AS {_TEXT}) AS __daita_key",
                _bound(()),
                f"{_cell(f'({name}.*)')} AS __daita_whole_row",
            ),
            sources,
            _where(root),
            f"{name}.ctid",
        )
        return select, f"{select} FOR UPDATE OF {name}"

    # INSERT: the incoming rows become a CTE named as ON CONFLICT names them, so
    # EXCLUDED.x in an assignment means the same thing in the derived SELECT.
    identifiers = tuple(
        item.sql(dialect="postgres") for item in shape.insert_identifiers
    )
    source = root.expression
    with_prefix = with_sql.rstrip() + ", " if with_sql else "WITH "
    incoming = ", ".join(identifiers)
    if isinstance(source, exp.Values):
        rows = ", ".join(
            "("
            + ", ".join(
                cast(value.sql(dialect="postgres", comments=False), column)
                for value, column in zip(
                    row_node.expressions, shape.insert_columns, strict=True
                )
            )
            + ")"
            for row_node in source.expressions
        )
        cte = f"{with_prefix}excluded({incoming}) AS (VALUES {rows}) "
    else:
        casted = ", ".join(
            f"{cast(item, column)} AS {item}"
            for item, column in zip(identifiers, shape.insert_columns, strict=True)
        )
        cte = (
            f"{with_prefix}__daita_source({incoming}) AS "
            f"({source.sql(dialect='postgres', comments=False)}), "
            f"excluded({incoming}) AS (SELECT {casted} FROM __daita_source) "
        )
    value_cells = tuple(
        f"{_cell(f'excluded.{item}')} AS __daita_value_{index}"
        for index, item in enumerate(identifiers)
    )
    value_bound = tuple(f"excluded.{item}" for item in identifiers)
    order = ", ".join(f"__daita_value_{i}" for i in range(len(identifiers)))
    if shape.conflict is None:
        select = _select(
            cte,
            (
                *no_row,
                "'insert' AS __daita_action",
                f"{_key_cell('excluded', key, exp)} AS __daita_key",
                _bound(value_bound),
                *value_cells,
            ),
            "excluded",
            "",
            order,
        )
        return select, None

    keys = " AND ".join(
        f"{name}.{item.this.sql(dialect='postgres')} = "
        f"excluded.{item.this.sql(dialect='postgres')}"
        for item in shape.conflict.args["conflict_keys"]
    )
    condition = shape.conflict.args.get("where")
    applies = (
        "FALSE"
        if not shape.update_columns
        else (
            "TRUE"
            if condition is None
            else condition.this.sql(dialect="postgres", comments=False)
        )
    )
    action = (
        f"CASE WHEN {name}.ctid IS NULL THEN 'insert' "
        f"WHEN {applies} THEN 'update' ELSE 'skip' END AS __daita_action"
    )
    expected_key = (
        f"CASE WHEN {name}.ctid IS NULL THEN {_key_cell('excluded', key, exp)} "
        f"ELSE {_key_cell(name, key, exp)} END AS __daita_key"
    )
    select_list = (
        *row,
        action,
        expected_key,
        _bound((*value_bound, *update_bound)),
        *value_cells,
        *update_cells,
    )
    joined = f"excluded {{}} JOIN {relation} ON {keys}"
    select = _select(
        cte, select_list, joined.format("LEFT"), "", f"__daita_row, {order}"
    )
    if not shape.update_columns:
        return select, None
    lock = _select(
        cte, select_list, joined.format("INNER"), "", f"__daita_row, {order}"
    )
    return select, f"{lock} FOR UPDATE OF {name}"


def _target_name(table: Any) -> str:
    alias = table.args.get("alias")
    return (alias.this if alias is not None and alias.this else table.this).sql(
        dialect="postgres"
    )


def _key_cell(name: str, key: Sequence[str] | None, exp: Any) -> str:
    if key is None:
        return f"CAST(NULL AS {_TEXT})"
    cells = ", ".join(
        f"CAST({name}.{exp.to_identifier(column, quoted=True).sql(dialect='postgres')} "
        f"AS {_TEXT})"
        for column in key
    )
    return f"CAST(pg_catalog.array_to_json(ARRAY[{cells}]) AS {_TEXT})"


def _where(root: Any) -> str:
    where = root.args.get("where")
    return "" if where is None else where.sql(dialect="postgres", comments=False)


def _select(
    prefix: str, items: Sequence[str], sources: str, where: str, order: str
) -> str:
    return (
        f"/* daita:postgresql.write_scan */ {prefix}SELECT {', '.join(items)} "
        f"FROM {sources}{' ' + where if where else ''} ORDER BY {order}"
    )


def _assignment_cells(
    shape: _Shape, name: str, types: Mapping[str, str], exp: Any
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cells: list[str] = []
    bound: list[str] = []
    for index, (target, value) in enumerate(shape.assignments):
        column = fold_postgresql(target.name, bool(target.this.args.get("quoted")))
        written = target.this.sql(dialect="postgres")
        cells.append(f"{_cell(f'{name}.{written}')} AS __daita_before_{index}")
        if _is_default(value, exp):
            cells.append(f"'{COLUMN_DEFAULT}' AS __daita_after_{index}")
            continue
        expression = (
            f"CAST({value.sql(dialect='postgres', comments=False)} "
            f"AS {types[column]})"
        )
        cells.append(f"{_cell(expression)} AS __daita_after_{index}")
        bound.append(expression)
    return tuple(cells), tuple(bound)


def _cell(expression: str) -> str:
    return f"pg_catalog.left(CAST({expression} AS {_TEXT}), {SAMPLE_CELL_CHARACTERS})"


def _bound(expressions: Sequence[str]) -> str:
    if not expressions:
        return f"CAST('' AS {_TEXT}) AS __daita_bound"
    array = "ARRAY[" + ", ".join(f"CAST({e} AS {_TEXT})" for e in expressions) + "]"
    return (
        "pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(CAST("
        f"pg_catalog.array_to_json({array}) AS {_TEXT}), 'UTF8')), 'hex') "
        "AS __daita_bound"
    )


def _is_default(value: Any, exp: Any) -> bool:
    return (
        isinstance(value, exp.Column)
        and not str(value.table or "")
        and not value.this.args.get("quoted")
        and value.name.upper() == "DEFAULT"
    )


__all__ = [
    "COMPUTED_AT_WRITE",
    "PostgreSQLStatement",
    "derived_select",
    "describe_write",
    "verified_statement",
]
