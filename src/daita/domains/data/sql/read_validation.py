"""One read policy: parse once, then check statement, functions and name scope.

The source resolves names and reports what does not exist; nothing here compares
a read with a stored description of the source.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .analysis import analyze_postgresql_sql, analyze_sqlite_sql
from .contracts import (
    _ASCII_IDENTIFIER_CASE_TRANSLATION,
    MAX_SQL_PARAMETERS,
    ResourceSchema,
    SqlAnalysis,
    SqlAnalysisError,
    SqlTableReference,
    SqlValidationIssue,
    SqlValidationResult,
    _identifier_key,
    _required_text,
    _short_identifier,
    _SqlDialect,
)

SQL_READ_DIALECTS = frozenset({"sqlite", "postgresql", "duckdb"})
_MAX_ISSUES = 32
_MAX_CANDIDATES = 8
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
    resources: Iterable[ResourceSchema],
    parameters: Sequence[object] = (),
    allowed_resource_ids: Iterable[str] | None = None,
    schemas: tuple[str, ...] = (),
) -> SqlValidationResult:
    """Check one SELECT before any source I/O; ``schemas`` are configured, in order."""

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
            resource_revisions=(),
            source_revision=None,
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

    allowed = None if allowed_resource_ids is None else frozenset(allowed_resource_ids)
    source_resources = tuple(item for item in resources if item.source_id == source_id)
    selected: list[ResourceSchema] = []
    for table in analysis.tables:
        if table.is_cte:
            continue
        candidates = _resource_candidates(table, source_resources, dialect)
        visible = tuple(
            item
            for item in candidates
            if allowed is None or item.resource_id in allowed
        )
        if (
            dialect == "postgresql"
            and len(table.qualified_parts) < 2
            and len(schemas) > 1
        ):
            issues.append(
                SqlValidationIssue(
                    "unqualified_relation",
                    f"{table.qualified_name} is unqualified and this source has "
                    "several schemas; qualify it with one of the candidates.",
                    {
                        "relation": table.qualified_name,
                        "candidates": [_full_name(item) for item in visible][
                            :_MAX_CANDIDATES
                        ],
                        "schemas": list(schemas),
                    },
                )
            )
        elif len(visible) != 1:
            issues.append(
                SqlValidationIssue(
                    "resource_out_of_scope",
                    f"{table.qualified_name} is not available for reading in this run.",
                    {"relation": table.qualified_name},
                )
            )
        elif visible[0] not in selected:
            selected.append(visible[0])

    if not any(not table.is_cte for table in analysis.tables):
        issues.append(
            SqlValidationIssue(
                "resource_scope_empty",
                f"{display_name} data queries must read a table.",
            )
        )
    bounded_issues = tuple(issues[:_MAX_ISSUES])
    return SqlValidationResult(
        valid=not bounded_issues,
        source_id=source_id,
        analysis=analysis,
        resource_ids=tuple(item.resource_id for item in selected),
        resource_revisions=tuple(
            (item.resource_id, item.revision)
            for item in selected
            if item.revision is not None
        ),
        source_revision=next(
            (item.source_revision for item in selected if item.source_revision), None
        ),
        issues=bounded_issues,
        resource_names=tuple((item.resource_id, item.name) for item in selected),
        sensitivity_classes=tuple(item.sensitivity_class for item in selected),
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


def qualified_relation(schema: ResourceSchema, dialect: str | None) -> str | None:
    """Quote one catalog relation in a dialect, or None without an exact name."""

    def quote(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    if dialect == "sqlite":
        return quote(schema.name)
    if dialect != "postgresql":
        return None
    identity = next(
        (
            (alias[: -len(suffix)], schema.name)
            for alias in schema.aliases
            for suffix in ("." + schema.name,)
            if alias.endswith(suffix) and len(alias) > len(suffix)
        ),
        None,
    )
    return None if identity is None else ".".join(quote(part) for part in identity)


def _full_name(resource: ResourceSchema) -> str:
    return next(
        (item for item in resource.aliases if item.endswith("." + resource.name)),
        resource.name,
    )


def _resource_candidates(
    table: SqlTableReference,
    resources: tuple[ResourceSchema, ...],
    dialect: _SqlDialect,
) -> tuple[ResourceSchema, ...]:
    if dialect == "postgresql":
        return tuple(
            resource
            for resource in resources
            if _postgresql_resource_matches(table, resource)
        )
    key = _identifier_key(table.qualified_name)
    short = _short_identifier(table.qualified_name)
    return tuple(
        resource
        for resource in resources
        if key in resource.lookup_names or short in resource.lookup_names
    )


def _postgresql_resource_matches(
    table: SqlTableReference,
    resource: ResourceSchema,
) -> bool:
    expected_name = (
        table.name
        if table.name_quoted
        else table.name.translate(_ASCII_IDENTIFIER_CASE_TRANSLATION)
    )
    if resource.name != expected_name:
        return False
    if len(table.qualified_parts) < 2:
        # pg_catalog is searched first, so an unqualified pg_ name is a system
        # catalog even when a configured schema has a table of that name.
        return not expected_name.startswith("pg_")
    expected_parts = tuple(
        name if quoted else name.translate(_ASCII_IDENTIFIER_CASE_TRANSLATION)
        for name, quoted in table.qualified_parts
    )
    return any(
        _postgresql_alias_parts(alias) == expected_parts for alias in resource.aliases
    )


def _postgresql_alias_parts(alias: str) -> tuple[str, ...]:
    # Attached schemas are safe single identifiers, while quoted PostgreSQL
    # resource names may themselves contain dots.  The first separator is the
    # catalog-owned schema/resource boundary.
    schema, separator, resource_name = alias.partition(".")
    return (schema, resource_name) if separator else (schema,)
