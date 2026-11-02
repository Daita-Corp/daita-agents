"""Parse SQLite and PostgreSQL statements into provider-neutral analysis records."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from ...._installation import repair_guidance
from .contracts import (
    MAX_SQL_CHARACTERS,
    SqlAnalysis,
    SqlAnalysisError,
    SqlTableReference,
    _dialect_identifier_key,
    _SqlDialect,
    normalize_sql,
)

_LOGGER = logging.getLogger("daita.data")
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


def _load_sqlglot(dialect: _SqlDialect = "sqlite") -> tuple[Any, Any]:
    try:
        import sqlglot
        from sqlglot import exp

        # Its warnings echo whole statements, which can hold literal values.
        logging.getLogger("sqlglot").setLevel(logging.ERROR)
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
        raise _parse_refusal(error, parse_sql, dialect, display_name) from error
    if not expressions:
        raise SqlAnalysisError("empty_sql", "SQL must not be empty.")

    root = expressions[0]
    try:
        cte_table_ids = _cte_table_ids(root, exp, dialect=dialect)
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
    function_names, unresolved_functions, table_functions = _function_facts(
        root,
        exp,
        dialect=dialect,
    )
    explicit_operators = _explicit_operators(root, exp, dialect=dialect)
    attribute_names = _attribute_names(root, exp, dialect=dialect)
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
    return SqlAnalysis(
        sql=normalized,
        canonical_sql=canonical_sql,
        sql_fingerprint=fingerprint,
        statement_type=(
            "explain" if explain is not None else type(root).__name__.lower()
        ),
        statement_count=len(parsed),
        is_read=is_read,
        mutation_types=mutation_types,
        tables=tables,
        positional_parameter_count=positional_parameter_count,
        parameter_ordinals=parameter_ordinals,
        anonymous_parameter_count=placeholder_count,
        invalid_parameter_count=invalid_parameter_count,
        function_names=function_names,
        unresolved_function_names=unresolved_functions,
        table_function_names=table_functions,
        explicit_operator_names=explicit_operators,
        attribute_names=attribute_names,
    )


def _parse_refusal(
    error: Exception, sql: str, dialect: _SqlDialect, display_name: str
) -> SqlAnalysisError:
    """Refuse unparseable SQL with its position; it is never run raw."""

    first: dict[str, object] = next(iter(getattr(error, "errors", None) or ()), {})
    line, column = first.get("line"), first.get("col")
    # shortcut: logs the position and fingerprint, not SQL text that can hold
    # literal values; the statement itself stays in the run's transcript.
    _LOGGER.info(
        "sql_parse_refused dialect=%s line=%s column=%s fingerprint=sha256:%s",
        dialect,
        line,
        column,
        hashlib.sha256(sql.encode("utf-8")).hexdigest(),
    )
    position = "" if line is None else f" at line {line}, column {column}"
    return SqlAnalysisError(
        "sql_parse_error",
        f"SQL could not be parsed for {display_name}{position}. Rewrite the "
        "statement with standard syntax; SQL that cannot be parsed is never run.",
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
        # A relation-position call is a function fact, not a relation name.
        if not isinstance(table.this, exp.Identifier):
            continue
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


def _cte_table_ids(
    root: Any,
    exp: Any,
    *,
    dialect: _SqlDialect,
) -> set[int]:
    """Resolve which table nodes name CTEs, using sqlglot's lexical scopes."""

    from sqlglot.optimizer.scope import traverse_scope

    cte_table_ids: set[int] = set()
    for scope in traverse_scope(root):
        for _, (selected, source) in scope.selected_sources.items():
            if isinstance(selected, exp.Table) and not isinstance(source, exp.Table):
                cte_table_ids.add(id(selected))
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
    return cte_table_ids


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
            # An explicit namespace other than pg_catalog is never admitted, and
            # canonical SQL upper-cases a quoted name, so it could run another
            # function than the one the lookup checked.
            if (namespace is not None and not name.startswith("PG_CATALOG.")) or (
                isinstance(function.this, exp.Identifier)
                and bool(function.this.args.get("quoted", False))
            ):
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
    if isinstance(parent, exp.Table) and parent.this is function:
        namespace = parent.args.get("db")
        if namespace is None:
            return None
        if parent.args.get("catalog") is not None:
            return "invalid"
    elif isinstance(parent, exp.Dot) and parent.expression is function:
        namespace = parent.this
    else:
        return None
    if not isinstance(namespace, exp.Identifier):
        return "invalid"
    value = str(namespace.name or "").strip()
    if not value:
        return "invalid"
    if bool(namespace.args.get("quoted", False)) and value != "pg_catalog":
        return f"quoted:{value}"
    return value


def _explicit_operators(
    root: Any,
    exp: Any,
    *,
    dialect: _SqlDialect,
) -> tuple[str, ...]:
    operator_type = getattr(exp, "Operator", None)
    if dialect != "postgresql" or operator_type is None:
        return ()
    return tuple(
        sorted(
            {
                str(operator.args.get("operator") or "invalid").upper()[:256]
                for operator in root.find_all(operator_type)
            }
        )
    )


def _attribute_names(
    root: Any,
    exp: Any,
    *,
    dialect: _SqlDialect,
) -> tuple[str, ...]:
    """Names read as ``x.name`` or ``(x).name``, in server spelling.

    PostgreSQL resolves either form to ``name(x)`` when no such column exists,
    so what parses as a column can be a function call.
    """

    if dialect != "postgresql":
        return ()
    identifiers = (
        *(
            column.this
            for column in root.find_all(exp.Column)
            if str(column.table or "").strip()
        ),
        *(dot.expression for dot in root.find_all(exp.Dot)),
    )
    return tuple(
        sorted(
            {
                (
                    str(item.name)
                    if item.args.get("quoted", False)
                    else str(item.name).lower()
                )[:256]
                for item in identifiers
                if isinstance(item, exp.Identifier) and str(item.name).strip()
            }
        )
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
