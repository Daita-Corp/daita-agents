"""Validate the narrow DuckDB read contract used by local ``file_query``."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, cast

from ...._installation import repair_guidance

MAX_FILE_QUERY_SQL_CHARACTERS = 8 * 1_024

_ALLOWED_FUNCTIONS = frozenset(
    {
        "ABS",
        "AVG",
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
        "TRIM",
        "UPPER",
        "VAR_POP",
        "VAR_SAMP",
    }
)
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


def validate_duckdb_read(sql: str) -> ValidatedDuckDBRead:
    """Accept one canonical read over exactly one base relation named ``data``."""

    if (
        not isinstance(sql, str)
        or not sql.strip()
        or len(sql) > MAX_FILE_QUERY_SQL_CHARACTERS
    ):
        raise DuckDBReadValidationError(
            "sql_size",
            "File query SQL must be bounded non-empty text.",
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
            "File query SQL could not be parsed safely.",
        ) from error
    if len(parsed) != 1 or parsed[0] is None:
        raise DuckDBReadValidationError(
            "statement_count",
            "File query requires exactly one SELECT statement.",
        )
    root = cast(Any, parsed[0])
    if not isinstance(root, exp.Select):
        raise DuckDBReadValidationError(
            "select_only",
            "File query accepts only one SELECT statement.",
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
            "File query SQL contains a prohibited operation.",
        )
    if any(
        isinstance(node, (exp.Values, exp.Placeholder, exp.Parameter))
        for node in expression.walk()
    ):
        raise DuckDBReadValidationError(
            "source_forbidden",
            "File query SQL cannot introduce values, parameters, or another source.",
        )
    for with_clause in expression.find_all(exp.With):
        if bool(with_clause.args.get("recursive", False)):
            raise DuckDBReadValidationError(
                "recursive_cte",
                "Recursive file queries are not supported.",
            )
    for cte in expression.find_all(exp.CTE):
        if str(cte.alias_or_name).casefold() == "data":
            raise DuckDBReadValidationError(
                "data_shadowed",
                "The code-owned data relation cannot be shadowed.",
            )

    data_references = 0
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
            "File query SQL scope could not be resolved safely.",
        ) from error
    for scope in scopes:
        if not scope.selected_sources:
            raise DuckDBReadValidationError(
                "source_forbidden",
                "Every file-query relation must derive from data.",
            )
        for _node, source in scope.selected_sources.values():
            if isinstance(source, exp.Table):
                if not isinstance(source.this, exp.Identifier):
                    raise DuckDBReadValidationError(
                        "table_function",
                        "File query SQL cannot invoke a table or filesystem function.",
                    )
                if source.catalog or source.db or source.name.casefold() != "data":
                    raise DuckDBReadValidationError(
                        "relation_forbidden",
                        "File query SQL can reference only the relation data.",
                    )
                data_references += 1
            elif not hasattr(source, "expression"):
                raise DuckDBReadValidationError(
                    "source_unknown",
                    "File query SQL contains an unsupported relation source.",
                )
    if data_references != 1:
        raise DuckDBReadValidationError(
            "data_reference_count",
            "File query SQL must reference the relation data exactly once.",
        )

    functions = tuple(
        sorted({_function_name(node) for node in expression.find_all(exp.Func)})
    )
    if any(name not in _ALLOWED_FUNCTIONS for name in functions):
        raise DuckDBReadValidationError(
            "function_forbidden",
            "File query SQL contains a function outside the bounded read contract.",
        )
    for literal in expression.find_all(exp.Literal):
        if literal.is_string and _SCHEME_OR_ABSOLUTE.match(str(literal.this)):
            raise DuckDBReadValidationError(
                "external_literal",
                "File query SQL cannot contain an external URL or absolute path.",
            )

    canonical = expression.sql(dialect="duckdb", pretty=False)
    try:
        reparsed = tuple(sqlglot.parse(canonical, read="duckdb"))
    except Exception as error:  # pragma: no cover - sqlglot invariant
        raise DuckDBReadValidationError(
            "canonicalization",
            "File query SQL could not be canonicalized safely.",
        ) from error
    if (
        len(reparsed) != 1
        or reparsed[0] is None
        or cast(Any, reparsed[0]).sql(dialect="duckdb", pretty=False) != canonical
    ):
        raise DuckDBReadValidationError(
            "canonicalization",
            "File query SQL could not be canonicalized safely.",
        )
    return ValidatedDuckDBRead(
        canonical_sql=canonical,
        sql_fingerprint=(
            "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        ),
        functions=functions,
    )


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
    "ValidatedDuckDBRead",
    "validate_duckdb_read",
]
