"""Provider-neutral SQLite and PostgreSQL SQL semantic analysis."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ...._installation import repair_guidance
from .contracts import (
    MAX_SQL_CHARACTERS,
    SqlAnalysis,
    SqlAnalysisError,
    SqlColumnReference,
    SqlTableReference,
    SqlValidationIssue,
    _ASCII_IDENTIFIER_CASE_TRANSLATION,
    _SqlDialect,
    _dialect_identifier_key,
    normalize_sql,
    sqlite_identifier_key,
)

_MAX_CANDIDATES = 8
_SQL_CALL_PREFIX = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*\(",
)
_STRUCTURAL_CALL_EXPRESSIONS = frozenset({"EXISTS"})
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


def _load_sqlglot(dialect: _SqlDialect = "sqlite") -> tuple[Any, Any]:
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as error:
        if dialect == "postgresql":
            raise ImportError(
                "Daita's PostgreSQL SQL validation dependency is unavailable. "
                f"{repair_guidance()}"
            ) from error
        raise ImportError(
            "Daita's SQLite SQL validation dependency is unavailable. "
            f"{repair_guidance()}"
        ) from error
    return sqlglot, exp


def _explain_prefix(sql: str) -> tuple[str, str] | None:
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

    if isinstance(sql, str) and len(sql) > MAX_SQL_CHARACTERS:
        raise SqlAnalysisError(
            "sql_too_large",
            f"SQL must contain at most {MAX_SQL_CHARACTERS} characters.",
        )
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
        raw_name = _rendered_callable_name(
            function,
            dialect=dialect,
            is_anonymous=is_anonymous,
        )
        if raw_name is None:
            continue
        name = (raw_name.strip() or type(function).__name__).upper()[:256]
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


def _rendered_callable_name(
    function: Any,
    *,
    dialect: _SqlDialect,
    is_anonymous: bool,
) -> str | None:
    """Return the callable identity emitted by the selected SQL generator."""

    if is_anonymous:
        name = str(function.name or "").strip()
        return name or type(function).__name__
    rendered = str(
        function.sql(
            dialect="postgres" if dialect == "postgresql" else "sqlite",
            pretty=False,
        )
    ).strip()
    match = _SQL_CALL_PREFIX.match(rendered)
    if match is not None:
        name = match.group(1).upper()
        if name in _STRUCTURAL_CALL_EXPRESSIONS:
            return None
        return name
    sql_name = str(function.sql_name() or "").strip().upper()
    if sql_name in _VOLATILE_CONTEXT_EXPRESSIONS:
        return sql_name
    # sqlglot models infix operators, CASE branches, ARRAY constructors, and
    # other structural grammar as Func subclasses.  If the selected dialect
    # does not render the node as a call or context keyword, it has no callable
    # identity for the function boundary.
    return None


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
    current = function
    while (parent := current.parent) is not None:
        if isinstance(parent, exp.Dot) and parent.expression is current:
            current = parent
            continue
        if isinstance(parent, exp.Table) and parent.this is current:
            current = parent
            continue
        if isinstance(parent, exp.Alias):
            if parent.this is not current:
                return False
            current = parent
            continue
        if isinstance(parent, exp.Lateral):
            return parent.this is current
        if isinstance(parent, exp.From):
            return parent.this is current or current in parent.expressions
        if isinstance(parent, exp.Join):
            return parent.this is current
        return False
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
