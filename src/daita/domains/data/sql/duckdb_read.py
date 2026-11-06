"""Validate the narrow DuckDB read contract of the private query worker.

A folder-source ``data_query`` exposes one relation, ``data``; a combined
``data_query`` exposes its input aliases.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from ...._installation import repair_guidance

MAX_FILE_QUERY_SQL_CHARACTERS = 8 * 1_024

_ALLOWED_FUNCTIONS = frozenset(
    {
        "ABS",
        "AVG",
        # CASE parses to CASE with IF branches.
        "CASE",
        "CAST",
        "CEIL",
        "CEILING",
        "COALESCE",
        "CONCAT",
        "COUNT",
        "DATE_TRUNC",
        "DENSE_RANK",
        "EXTRACT",
        "FIRST_VALUE",
        "FLOOR",
        "GREATEST",
        "IF",
        "LAG",
        "LAST_VALUE",
        "LEAD",
        "LEAST",
        "LENGTH",
        "LOWER",
        "MAX",
        "MIN",
        "NTILE",
        "NULLIF",
        "PERCENT_RANK",
        "RANK",
        "REPLACE",
        "ROUND",
        "ROW_NUMBER",
        "SIGN",
        "STDDEV_POP",
        "STDDEV_SAMP",
        "SUBSTRING",
        "SUM",
        # strftime and date_trunc parse to these names.
        "TIME_TO_STR",
        "TIMESTAMP_TRUNC",
        "TRIM",
        "TRY_CAST",
        "UPPER",
        "VAR_POP",
        "VAR_SAMP",
    }
)
_MAX_JOIN_CHECKS = 16
_MAX_JOIN_LABEL = 128
_SCHEME_OR_ABSOLUTE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*://|/|\\\\)")


class DuckDBReadValidationError(RuntimeError):
    """One bounded rejection of model-authored local-file SQL."""

    def __init__(self, reason: str, message: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("DuckDB validation reason must be non-empty text")
        if not isinstance(message, str) or not message:
            raise ValueError("DuckDB validation message must be non-empty text")
        self.reason = reason
        self.message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ValidatedDuckDBRead:
    canonical_sql: str
    sql_fingerprint: str
    functions: tuple[str, ...]


def validate_duckdb_read(
    sql: str, relations: tuple[str, ...] = ("data",)
) -> ValidatedDuckDBRead:
    """Accept one canonical SELECT (or UNION) reading every code-owned relation."""

    names = frozenset(item.casefold() for item in relations)
    if not relations or len(names) != len(relations):
        raise ValueError("DuckDB read relations must be unique non-empty names")

    if (
        not isinstance(sql, str)
        or not sql.strip()
        or len(sql) > MAX_FILE_QUERY_SQL_CHARACTERS
    ):
        raise DuckDBReadValidationError(
            "sql_size",
            "SQL must be bounded non-empty text of at most 8 KiB.",
        )
    sqlglot, exp, traverse_scope = _load_sqlglot()
    try:
        parsed = tuple(sqlglot.parse(sql, read="duckdb"))
    except Exception as error:
        errors = getattr(sqlglot, "errors", None)
        parse_errors = tuple(
            item
            for item in (
                getattr(errors, "ParseError", None),
                getattr(errors, "TokenError", None),
            )
            if isinstance(item, type)
        )
        if not parse_errors or not isinstance(error, parse_errors):
            raise
        raise DuckDBReadValidationError(
            "parse_error",
            "SQL could not be parsed safely.",
        ) from error
    if len(parsed) != 1 or parsed[0] is None:
        raise DuckDBReadValidationError(
            "statement_count",
            "The query requires exactly one SELECT statement.",
        )
    root = cast(Any, parsed[0])
    if not isinstance(root, (exp.Select, exp.Union)):
        raise DuckDBReadValidationError(
            "select_only",
            "The query accepts only one SELECT statement.",
        )
    expression = cast(Any, root)
    forbidden_nodes = tuple(
        item
        for item in (
            getattr(exp, name, None)
            for name in (
                "Alter",
                "Analyze",
                "Attach",
                "Cache",
                "Command",
                "Copy",
                "Create",
                "Delete",
                "Detach",
                "Drop",
                "Execute",
                "Export",
                "Grant",
                "Insert",
                "Install",
                "Into",
                "LoadData",
                "Merge",
                "Pragma",
                "Set",
                "Transaction",
                "TruncateTable",
                "Update",
                "Use",
            )
        )
        if isinstance(item, type)
    )
    if forbidden_nodes and any(
        isinstance(node, forbidden_nodes) for node in expression.walk()
    ):
        raise DuckDBReadValidationError(
            "operation_forbidden",
            "SQL contains a prohibited operation.",
        )
    if any(
        isinstance(node, (exp.Values, exp.Placeholder, exp.Parameter))
        for node in expression.walk()
    ):
        raise DuckDBReadValidationError(
            "source_forbidden",
            "SQL cannot introduce values, parameters, or another source.",
        )
    for with_clause in expression.find_all(exp.With):
        if bool(with_clause.args.get("recursive", False)):
            raise DuckDBReadValidationError(
                "recursive_cte",
                "Recursive queries are not supported.",
            )
    for cte in expression.find_all(exp.CTE):
        if str(cte.alias_or_name).casefold() in names:
            raise DuckDBReadValidationError(
                "data_shadowed",
                "A code-owned relation cannot be shadowed.",
            )

    references = dict.fromkeys(names, 0)
    try:
        scopes = tuple(traverse_scope(expression))
    except Exception as error:
        errors = getattr(sqlglot, "errors", None)
        optimize_error = getattr(errors, "OptimizeError", None)
        if not isinstance(optimize_error, type) or not isinstance(
            error, optimize_error
        ):
            raise
        raise DuckDBReadValidationError(
            "scope_error",
            "SQL scope could not be resolved safely.",
        ) from error
    for scope in scopes:
        if isinstance(scope.expression, exp.Union):
            # A UNION selects nothing itself; each side is its own scope.
            continue
        if not scope.selected_sources:
            raise DuckDBReadValidationError(
                "source_forbidden",
                "Every relation must derive from a code-owned relation.",
            )
        for _node, source in scope.selected_sources.values():
            if isinstance(source, exp.Table):
                if not isinstance(source.this, exp.Identifier):
                    raise DuckDBReadValidationError(
                        "table_function",
                        "SQL cannot invoke a table or filesystem function.",
                    )
                if source.catalog or source.db or source.name.casefold() not in names:
                    raise DuckDBReadValidationError(
                        "relation_forbidden",
                        "SQL can reference only the relations "
                        + ", ".join(relations)
                        + ".",
                    )
                references[source.name.casefold()] += 1
            elif not hasattr(source, "expression"):
                raise DuckDBReadValidationError(
                    "source_unknown",
                    "SQL contains an unsupported relation source.",
                )
    if any(count < 1 for count in references.values()):
        raise DuckDBReadValidationError(
            "data_reference_count",
            "SQL must reference each of " + ", ".join(relations) + ".",
        )

    # sqlglot models AND/OR as functions; they are boolean operators.
    functions = tuple(
        sorted(
            {
                _function_name(node)
                for node in expression.find_all(exp.Func)
                if not isinstance(node, exp.Connector)
            }
        )
    )
    if any(name not in _ALLOWED_FUNCTIONS for name in functions):
        raise DuckDBReadValidationError(
            "function_forbidden",
            "SQL contains a function outside the bounded read contract.",
        )
    for literal in expression.find_all(exp.Literal):
        if literal.is_string and _SCHEME_OR_ABSOLUTE.match(str(literal.this)):
            raise DuckDBReadValidationError(
                "external_literal",
                "SQL cannot contain an external URL or absolute path.",
            )

    canonical = expression.sql(dialect="duckdb", pretty=False)
    try:
        reparsed = tuple(sqlglot.parse(canonical, read="duckdb"))
    except Exception as error:  # pragma: no cover - sqlglot invariant
        raise DuckDBReadValidationError(
            "canonicalization",
            "SQL could not be canonicalized safely.",
        ) from error
    if (
        len(reparsed) != 1
        or reparsed[0] is None
        or cast(Any, reparsed[0]).sql(dialect="duckdb", pretty=False) != canonical
    ):
        raise DuckDBReadValidationError(
            "canonicalization",
            "SQL could not be canonicalized safely.",
        )
    return ValidatedDuckDBRead(
        canonical_sql=canonical,
        sql_fingerprint=(
            "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        ),
        functions=functions,
    )


@dataclass(frozen=True, slots=True)
class EqualityJoin:
    """One equality join with code-built diagnostic SQL (all read-only)."""

    left: str
    right: str
    left_key_sql: str
    right_key_sql: str
    left_unmatched_sql: str
    right_unmatched_sql: str
    left_rows_sql: str
    right_rows_sql: str
    left_near_sql: str
    right_near_sql: str


def equality_joins(
    canonical_sql: str, columns: Mapping[str, frozenset[str]]
) -> tuple[EqualityJoin, ...]:
    """Find each column-to-column equality join of a validated SELECT.

    ``columns`` maps each input relation to its column names. Anything that is
    not a plain equality between two different from-items is skipped; this
    never raises, so a diagnostic gap cannot affect a query result.
    """

    try:
        return _equality_joins(canonical_sql, columns)
    except Exception:  # noqa: BLE001 - diagnostics must not fail a valid query
        return ()


def _conjuncts(node: Any, exp: Any) -> Iterator[Any]:
    if isinstance(node, exp.Paren):
        yield from _conjuncts(node.this, exp)
    elif isinstance(node, exp.And):
        yield from _conjuncts(node.this, exp)
        yield from _conjuncts(node.expression, exp)
    elif node is not None:
        yield node


def _side_item(
    side: Any,
    items: list[Any],
    names: list[str],
    known: Mapping[str, set[str]],
    exp: Any,
) -> int | None:
    if side.find(exp.Subquery, exp.Select) is not None:
        return None
    owners: set[int] = set()
    for column in side.find_all(exp.Column):
        if column.table:
            matches = [
                i
                for i, n in enumerate(names)
                if n.casefold() == column.table.casefold()
            ]
        else:
            # Unqualified names resolve only when every from-item is a
            # known input relation and exactly one has the column.
            if any(
                not isinstance(i, exp.Table) or i.name.casefold() not in known
                for i in items
            ):
                return None
            matches = [
                i
                for i, item in enumerate(items)
                if column.name.casefold() in known[item.name.casefold()]
            ]
        if len(matches) != 1:
            return None
        owners.add(matches[0])
    return next(iter(owners)) if len(owners) == 1 else None


def select_equalities(
    root: Any, exp: Any, known: Mapping[str, set[str]]
) -> Iterator[tuple[Any, list[Any], list[str], Any, int, int]]:
    """Yield each equality between two different from-items of every SELECT.

    Each item is (select, from-items, their aliases, the EQ node, left item
    index, right item index). ``known`` maps lower-cased relation names to
    lower-cased column names; an empty map resolves only qualified columns.
    """

    for select in root.find_all(exp.Select):
        source = select.args.get("from_") or select.args.get("from")
        joins = select.args.get("joins") or []
        if source is None:
            continue
        items = [source.this, *(j.this for j in joins)]
        names = [i.alias_or_name for i in items]
        if not all(names):
            continue
        candidates: list[Any] = []
        for position, join in enumerate(joins, start=1):
            candidates.extend(_conjuncts(join.args.get("on"), exp))
            for using in join.args.get("using") or []:
                name = using.name.casefold()
                holders = [
                    i
                    for i in range(position + 1)
                    if isinstance(items[i], exp.Table)
                    and name in known.get(items[i].name.casefold(), ())
                ]
                if not known and position == 1:
                    # Without column facts, a two-item USING join is unambiguous.
                    holders = [0, 1]
                if position in holders and len(holders) == 2:
                    other = next(i for i in holders if i != position)
                    candidates.append(
                        exp.EQ(
                            this=exp.column(using.name, names[other]),
                            expression=exp.column(using.name, names[position]),
                        )
                    )
        candidates.extend(
            _conjuncts(
                select.args["where"].this if select.args.get("where") else None, exp
            )
        )
        for eq in candidates:
            if not isinstance(eq, exp.EQ):
                continue
            li = _side_item(eq.this, items, names, known, exp)
            ri = _side_item(eq.expression, items, names, known, exp)
            if li is not None and ri is not None and li != ri:
                yield select, items, names, eq, li, ri


def _equality_joins(
    canonical_sql: str, columns: Mapping[str, frozenset[str]]
) -> tuple[EqualityJoin, ...]:
    sqlglot, exp, _ = _load_sqlglot()
    known = {
        name.casefold(): {c.casefold() for c in cs} for name, cs in columns.items()
    }
    root = sqlglot.parse_one(canonical_sql, read="duckdb")
    with_clause = root.args.get("with_") or root.args.get("with")
    prefix = with_clause.sql(dialect="duckdb") + " " if with_clause else ""
    found: dict[tuple[str, str], EqualityJoin] = {}

    for _select, items, names, eq, li, ri in select_equalities(root, exp, known):
        left, right = eq.this, eq.expression
        label = (left.sql(dialect="duckdb"), right.sql(dialect="duckdb"))
        sides = []
        for side, index in ((left, li), (right, ri)):
            side = side.copy()
            for column in side.find_all(exp.Column):
                column.set("table", exp.to_identifier(names[index]))
            sides.append(side.sql(dialect="duckdb"))
        lexpr, rexpr = sides
        litem, ritem = (
            items[li].sql(dialect="duckdb"),
            items[ri].sql(dialect="duckdb"),
        )
        condition = f"{lexpr} = {rexpr}"
        near = (
            f"lower(trim(CAST({lexpr} AS VARCHAR))) = "
            f"lower(trim(CAST({rexpr} AS VARCHAR)))"
        )
        join = EqualityJoin(
            left=label[0][:_MAX_JOIN_LABEL],
            right=label[1][:_MAX_JOIN_LABEL],
            left_key_sql=f"{prefix}SELECT {lexpr} AS k FROM {litem}",
            right_key_sql=f"{prefix}SELECT {rexpr} AS k FROM {ritem}",
            left_unmatched_sql=(
                f"{prefix}SELECT count(*) FROM {litem} WHERE NOT EXISTS "
                f"(SELECT 1 FROM {ritem} WHERE {condition})"
            ),
            right_unmatched_sql=(
                f"{prefix}SELECT count(*) FROM {ritem} WHERE NOT EXISTS "
                f"(SELECT 1 FROM {litem} WHERE {condition})"
            ),
            left_rows_sql=f"{prefix}SELECT count(*) FROM {litem}",
            right_rows_sql=f"{prefix}SELECT count(*) FROM {ritem}",
            # Unmatched rows that would match if case and padding were ignored.
            left_near_sql=(
                f"{prefix}SELECT count(*) FROM {litem} WHERE NOT EXISTS "
                f"(SELECT 1 FROM {ritem} WHERE {condition}) AND EXISTS "
                f"(SELECT 1 FROM {ritem} WHERE {near})"
            ),
            right_near_sql=(
                f"{prefix}SELECT count(*) FROM {ritem} WHERE NOT EXISTS "
                f"(SELECT 1 FROM {litem} WHERE {condition}) AND EXISTS "
                f"(SELECT 1 FROM {litem} WHERE {near})"
            ),
        )
        found.setdefault((join.left_unmatched_sql, join.right_unmatched_sql), join)
    return tuple(found.values())[:_MAX_JOIN_CHECKS]


def _load_sqlglot() -> tuple[Any, Any, Any]:
    try:
        import sqlglot
        from sqlglot import exp
        from sqlglot.optimizer.scope import traverse_scope
    except ImportError as error:
        raise ImportError(
            "Daita's local file SQL validation dependency is unavailable. "
            f"{repair_guidance()}"
        ) from error
    return sqlglot, exp, traverse_scope


def _function_name(node: Any) -> str:
    name = node.sql_name()
    if name == "ANONYMOUS":
        name = getattr(node, "name", "")
    return str(name).upper()


__all__ = [
    "MAX_FILE_QUERY_SQL_CHARACTERS",
    "DuckDBReadValidationError",
    "EqualityJoin",
    "ValidatedDuckDBRead",
    "equality_joins",
    "validate_duckdb_read",
]
