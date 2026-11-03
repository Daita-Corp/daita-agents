"""One read policy: parse once, then check statement, functions and name scope.

The source resolves names and reports what does not exist. Scope needs no catalog:
a name is folded and hashed forward to the ID stored permissions list.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ....catalog.models import ResourceKind, catalog_resource_id
from .analysis import analyze_postgresql_sql, analyze_sqlite_sql
from .contracts import (
    _ASCII_IDENTIFIER_CASE_TRANSLATION,
    MAX_SQL_PARAMETERS,
    SqlAnalysis,
    SqlAnalysisError,
    SqlTableReference,
    SqlValidationIssue,
    SqlValidationResult,
    _required_text,
    _SqlDialect,
    sqlite_identifier_key,
)

SQL_READ_DIALECTS = frozenset({"sqlite", "postgresql", "duckdb"})
_MAX_ISSUES = 32
_MAX_CANDIDATES = 8
_POSTGRESQL_SYSTEM_SCHEMAS = frozenset({"pg_catalog", "information_schema", "pg_toast"})
_SQLITE_SYSTEM_TABLES = frozenset(
    {"sqlite_schema", "sqlite_master", "sqlite_temp_schema", "sqlite_temp_master"}
)
# These run SQL or read whole tables from arguments the parser cannot see, and
# several are STABLE, so a volatility check alone would admit them.
_POSTGRESQL_REFUSED_FUNCTIONS = re.compile(
    r"^(?:.*_to_xml|.*_to_xmlschema|.*_to_xml_and_xmlschema|cursor_to_.*"
    r"|ts_stat|ts_rewrite)$"
)
# Parser keywords, not pg_proc entries: no schema can shadow them.
_POSTGRESQL_GRAMMAR_FORMS = frozenset(
    {
        "CAST",
        "COALESCE",
        "COLLATION",
        "CURRENT_CATALOG",
        "CURRENT_DATE",
        "CURRENT_ROLE",
        "CURRENT_SCHEMA",
        "CURRENT_TIME",
        "CURRENT_TIMESTAMP",
        "CURRENT_USER",
        "EXTRACT",
        "GREATEST",
        "GROUPING",
        "LEAST",
        "LOCALTIME",
        "LOCALTIMESTAMP",
        "NULLIF",
        "OVERLAY",
        "POSITION",
        "ROW",
        "SESSION_USER",
        "SUBSTRING",
        "SYSTEM_USER",
        "TREAT",
        "TRIM",
        "USER",
        "XMLCONCAT",
        "XMLELEMENT",
        "XMLEXISTS",
        "XMLFOREST",
        "XMLPARSE",
        "XMLPI",
        "XMLROOT",
        "XMLSERIALIZE",
        "XMLTABLE",
    }
)


def validate_read(
    sql: str,
    *,
    dialect: _SqlDialect,
    source_id: str,
    parameters: Sequence[object] = (),
    allowed_resource_ids: frozenset[str] | None = None,
    schemas: tuple[str, ...] = (),
) -> SqlValidationResult:
    """Check one SELECT before any source I/O, using only the SQL and the run's scope.

    ``allowed_resource_ids`` of None means the run may read the source in full;
    ``schemas`` are the source's configured schemas.
    """

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
            issues=(SqlValidationIssue(error.code, str(error)),),
        )

    issues: list[SqlValidationIssue] = []
    if len(parameters) > MAX_SQL_PARAMETERS:
        issues.append(
            SqlValidationIssue(
                "parameter_count_exceeded",
                f"SQL reads accept at most {MAX_SQL_PARAMETERS} bound parameters.",
                {"received": len(parameters)},
            )
        )
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
                "PostgreSQL EXPLAIN statements cannot use the bounded "
                "tabular execution path.",
            )
        )
    issues.extend(_function_issues(analysis, dialect))
    issues.extend(_parameter_issues(analysis, parameters, dialect))

    selected: dict[str, str] = {}
    unqualified: list[str] = []
    for table in analysis.tables:
        if table.is_cte:
            continue
        check = (
            _check_postgresql_relation(table, source_id, allowed_resource_ids, schemas)
            if dialect == "postgresql"
            else _check_sqlite_relation(table, source_id, allowed_resource_ids)
        )
        if isinstance(check, SqlValidationIssue):
            issues.append(check)
        elif check is None:
            unqualified.append(_fold_postgresql(table.name, table.name_quoted))
        else:
            selected.setdefault(check[0], check[1])

    bounded_issues = tuple(issues[:_MAX_ISSUES])
    return SqlValidationResult(
        valid=not bounded_issues,
        source_id=source_id,
        analysis=analysis,
        resource_ids=tuple(selected),
        issues=bounded_issues,
        resource_names=tuple(selected.items()),
        function_lookups=(
            tuple(
                name
                for name in analysis.function_names
                if name.removeprefix("PG_CATALOG.") not in _POSTGRESQL_GRAMMAR_FORMS
            )
            if dialect == "postgresql"
            else ()
        ),
        attribute_lookups=analysis.attribute_names,
        unqualified_relations=tuple(dict.fromkeys(unqualified)),
    )


def resource_id_in_scope(
    source_id: str,
    native_name: str,
    allowed: frozenset[str] | None,
) -> str | None:
    """Hash a folded name forward: its ID if readable, with no catalog lookup."""

    table_id = catalog_resource_id(source_id, ResourceKind.TABLE, native_name)
    if allowed is None or table_id in allowed:
        return table_id
    view_id = catalog_resource_id(source_id, ResourceKind.VIEW, native_name)
    return view_id if view_id in allowed else None


def _check_postgresql_relation(
    table: SqlTableReference,
    source_id: str,
    allowed: frozenset[str] | None,
    schemas: tuple[str, ...],
) -> tuple[str, str] | SqlValidationIssue | None:
    """A relation's (ID, full name), the issue refusing it, or None when unqualified."""

    parts = tuple(
        _fold_postgresql(name, quoted) for name, quoted in table.qualified_parts
    )
    if len(parts) == 1:
        # pg_catalog is searched first, so an unqualified pg_ name is a system
        # catalog even when a configured schema has a table of that name.
        if parts[0].startswith("pg_"):
            parts = ("pg_catalog", parts[0])
        elif len(schemas) == 1:
            parts = (schemas[0], parts[0])
        else:
            return None
    if len(parts) != 2:
        return _out_of_scope(table)
    schema, name = parts
    full_name = f"{schema}.{name}"
    if schema in _POSTGRESQL_SYSTEM_SCHEMAS:
        if allowed is not None:
            return _introspection_refused(table)
    elif schema not in schemas:
        return SqlValidationIssue(
            "schema_not_configured",
            f"{table.qualified_name} is in schema {schema}, which is not attached "
            "to this source. Add the schema to the source to read it.",
            {"relation": table.qualified_name, "schema": schema},
        )
    resource_id = resource_id_in_scope(source_id, full_name, allowed)
    return _out_of_scope(table) if resource_id is None else (resource_id, full_name)


def _check_sqlite_relation(
    table: SqlTableReference,
    source_id: str,
    allowed: frozenset[str] | None,
) -> tuple[str, str] | SqlValidationIssue:
    parts = tuple(sqlite_identifier_key(name) for name, _ in table.qualified_parts)
    if len(parts) > 2 or (len(parts) == 2 and parts[0] != "main"):
        return _out_of_scope(table)
    name = parts[-1]
    # A pragma table reads connection state, such as file paths, not source data.
    if name.startswith("pragma_"):
        return _out_of_scope(table)
    if name in _SQLITE_SYSTEM_TABLES and allowed is not None:
        return _introspection_refused(table)
    full_name = f"main.{name}"
    resource_id = resource_id_in_scope(source_id, full_name, allowed)
    return _out_of_scope(table) if resource_id is None else (resource_id, full_name)


def _fold_postgresql(name: str, quoted: bool) -> str:
    return name if quoted else name.translate(_ASCII_IDENTIFIER_CASE_TRANSLATION)


def _out_of_scope(table: SqlTableReference) -> SqlValidationIssue:
    return SqlValidationIssue(
        "resource_out_of_scope",
        f"{table.qualified_name} is not available for reading in this run.",
        {"relation": table.qualified_name},
    )


def _introspection_refused(table: SqlTableReference) -> SqlValidationIssue:
    # System catalogs would show names outside this run's scope.
    return SqlValidationIssue(
        "resource_out_of_scope",
        f"{table.qualified_name} is not available in this run, which can read only "
        "some tables of this source. Use catalog_search or catalog_describe.",
        {"relation": table.qualified_name},
    )


def _function_issues(
    analysis: SqlAnalysis, dialect: _SqlDialect
) -> tuple[SqlValidationIssue, ...]:
    if dialect == "sqlite":
        # The authorizer owns SQLite's scalar functions; a relation-position
        # call such as pragma_table_info reads outside the name scope.
        refused = analysis.table_function_names
    else:
        refused = tuple(
            sorted(
                {
                    *analysis.unresolved_function_names,
                    *(
                        name
                        for name in analysis.function_names
                        if _POSTGRESQL_REFUSED_FUNCTIONS.match(
                            name.removeprefix("PG_CATALOG.").lower()
                        )
                    ),
                }
            )
        )
    issues = []
    if refused:
        issues.append(
            SqlValidationIssue(
                "function_not_allowed",
                f"Reads cannot call {', '.join(refused[:_MAX_CANDIDATES])}.",
                {"functions": refused[:_MAX_CANDIDATES]},
            )
        )
    operators = tuple(
        name
        for name in analysis.explicit_operator_names
        if not name.startswith("PG_CATALOG.")
    )
    if operators:
        issues.append(
            SqlValidationIssue(
                "operator_not_allowed",
                "Explicit PostgreSQL operators must resolve from pg_catalog.",
                {"operators": operators[:_MAX_CANDIDATES]},
            )
        )
    return tuple(issues)


def _parameter_issues(
    analysis: SqlAnalysis, parameters: Sequence[object], dialect: _SqlDialect
) -> tuple[SqlValidationIssue, ...]:
    if dialect == "sqlite":
        if analysis.positional_parameter_count == len(parameters):
            return ()
        return (
            SqlValidationIssue(
                "parameter_count_mismatch",
                "SQLite positional parameter count does not match the SQL "
                "placeholders.",
                {
                    "expected": analysis.positional_parameter_count,
                    "received": len(parameters),
                },
            ),
        )
    issues = []
    if analysis.anonymous_parameter_count:
        issues.append(
            SqlValidationIssue(
                "parameter_style_invalid",
                "PostgreSQL query parameters must use numbered $1 placeholders.",
                {"anonymous_placeholders": analysis.anonymous_parameter_count},
            )
        )
    expected = tuple(range(1, len(parameters) + 1))
    if analysis.invalid_parameter_count or analysis.parameter_ordinals != expected:
        issues.append(
            SqlValidationIssue(
                "parameter_index_mismatch",
                "PostgreSQL parameter indexes must be contiguous and match the "
                "supplied values.",
                {
                    "expected": expected,
                    "received": analysis.parameter_ordinals,
                    "invalid": analysis.invalid_parameter_count,
                },
            )
        )
    return tuple(issues)
