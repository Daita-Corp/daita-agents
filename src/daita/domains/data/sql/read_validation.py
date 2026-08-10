"""Catalog-scoped validation for bounded SQLite and PostgreSQL reads."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from .analysis import (
    _MAX_CANDIDATES,
    _LexicalRelation,
    _LexicalScope,
    _direct_scope_columns,
    _explain_prefix,
    _is_legal_output_alias,
    _load_sqlglot,
    _relation_column_matches,
    _scope_projection,
    _selected_source_qualifier,
    _semantic_identifier,
    _table_references,
    analyze_postgresql_sql,
    analyze_sqlite_sql,
)
from .contracts import (
    MAX_SQL_PARAMETERS,
    ResourceSchema,
    SqlAnalysisError,
    SqlColumnReference,
    SqlTableReference,
    SqlValidationIssue,
    SqlValidationResult,
    _ASCII_IDENTIFIER_CASE_TRANSLATION,
    _SqlDialect,
    _identifier_key,
    _required_text,
    _short_identifier,
)

_MAX_ISSUES = 32
# PostgreSQL functions can perform external I/O even in a read-only
# transaction.  Keep the directly callable surface deliberately smaller than
# the server catalog, and retain a non-replay-safe capability declaration until
# durable server-owned callable/operator/type provenance exists.
_POSTGRESQL_BOUNDED_FUNCTIONS = frozenset(
    {
        "ABS",
        "AGE",
        "ARRAY_AGG",
        "AVG",
        "BOOL_AND",
        "BOOL_OR",
        "CAST",
        "CEIL",
        "CEILING",
        "CHAR_LENGTH",
        "COALESCE",
        "CONCAT",
        "CONCAT_WS",
        "CORR",
        "COUNT",
        "COVAR_POP",
        "COVAR_SAMP",
        "CUME_DIST",
        "DATE",
        "DATE_BIN",
        "DATE_TRUNC",
        "DECODE",
        "DENSE_RANK",
        "EVERY",
        "EXTRACT",
        "FIRST_VALUE",
        "FLOOR",
        "GREATEST",
        "JSON_AGG",
        "JSONB_AGG",
        "LAG",
        "LAST_VALUE",
        "LEAD",
        "LENGTH",
        "LEAST",
        "LOWER",
        "LTRIM",
        "MAX",
        "MIN",
        "NTH_VALUE",
        "NTILE",
        "NULLIF",
        "OCTET_LENGTH",
        "PERCENTILE_CONT",
        "PERCENTILE_DISC",
        "PERCENT_RANK",
        "RANK",
        "REPLACE",
        "ROW_NUMBER",
        "ROUND",
        "RTRIM",
        "SIGN",
        "SPLIT_PART",
        "STDDEV",
        "STDDEV_POP",
        "STDDEV_SAMP",
        "STRING_AGG",
        "SUBSTR",
        "SUBSTRING",
        "SUM",
        "TIMEZONE",
        "TO_CHAR",
        "TRIM",
        "UPPER",
        "VARIANCE",
        "VAR_POP",
        "VAR_SAMP",
    }
)


def _lexical_column_issues(
    sql: str,
    *,
    resources: tuple[ResourceSchema, ...],
    allowed_resource_ids: frozenset[str] | None,
    dialect: _SqlDialect,
) -> tuple[SqlValidationIssue, ...]:
    """Validate columns against relation schemas in their lexical scopes."""

    sqlglot, exp = _load_sqlglot(dialect)
    parse_sql = (_explain_prefix(sql) or ("", sql))[1]
    root = sqlglot.parse_one(
        parse_sql,
        read="postgres" if dialect == "postgresql" else "sqlite",
    )
    from sqlglot.optimizer.scope import traverse_scope

    scopes = tuple(traverse_scope(root))
    scope_ids = {id(scope): f"scope:{index}" for index, scope in enumerate(scopes)}
    expression_scope_ids = frozenset(id(scope.expression) for scope in scopes)
    consumed_scope_ids = {
        id(source)
        for scope in scopes
        for _, source in scope.selected_sources.values()
        if not isinstance(source, exp.Table)
    }
    states: dict[int, _LexicalScope] = {}
    unresolved_qualifiers: dict[int, frozenset[str]] = {}
    issues: list[SqlValidationIssue] = []
    for scope in scopes:
        relations: list[_LexicalRelation] = []
        unresolved: set[str] = set()
        for fallback, (selected, source) in scope.selected_sources.items():
            qualifier = _selected_source_qualifier(
                selected,
                fallback,
                exp,
                dialect=dialect,
            )
            if isinstance(source, exp.Table):
                references = _table_references(
                    source,
                    exp,
                    set(),
                    dialect=dialect,
                )
                candidates = (
                    _resource_candidates(
                        references[0],
                        resources,
                        dialect=dialect,
                    )
                    if references
                    else ()
                )
                if len(candidates) != 1:
                    unresolved.add(qualifier)
                    continue
                resource = candidates[0]
                if (
                    allowed_resource_ids is not None
                    and resource.resource_id not in allowed_resource_ids
                ):
                    unresolved.add(qualifier)
                    continue
                relations.append(
                    _LexicalRelation(
                        qualifier=qualifier,
                        columns=tuple(
                            _semantic_identifier(
                                column,
                                dialect=dialect,
                                quoted=(
                                    dialect == "postgresql"
                                    and column
                                    != column.translate(
                                        _ASCII_IDENTIFIER_CASE_TRANSLATION
                                    )
                                ),
                            )
                            for column in resource.columns
                        ),
                        lineage=(resource.resource_id,),
                        kind="base",
                        scope_id=scope_ids[id(scope)],
                    )
                )
                continue
            derived = states.get(id(source))
            if derived is None:
                unresolved.add(qualifier)
                continue
            kind: Literal["cte", "subquery", "set"] = (
                "set"
                if isinstance(
                    source.expression,
                    tuple(
                        item
                        for item in (
                            getattr(exp, "Union", None),
                            getattr(exp, "Intersect", None),
                            getattr(exp, "Except", None),
                        )
                        if item is not None
                    ),
                )
                else "cte" if isinstance(selected, exp.Table) else "subquery"
            )
            relations.append(
                _LexicalRelation(
                    qualifier=qualifier,
                    columns=derived.columns,
                    lineage=derived.lineage,
                    kind=kind,
                    scope_id=scope_ids[id(scope)],
                )
            )

        relation_tuple = tuple(relations)
        projected = _scope_projection(
            scope,
            relation_tuple,
            states,
            exp,
            issues,
            dialect=dialect,
        )
        child_scopes = tuple(
            item for item in scopes if item.parent is scope and id(item) in states
        )
        if getattr(scope, "union_scopes", ()):
            lineage = tuple(
                dict.fromkeys(
                    resource_id
                    for branch in scope.union_scopes
                    for resource_id in states[id(branch)].lineage
                )
            )
        else:
            lineage = tuple(
                dict.fromkeys(
                    (
                        resource_id
                        for relation in relation_tuple
                        for resource_id in relation.lineage
                    ),
                )
            )
        lineage = tuple(
            dict.fromkeys(
                (
                    *lineage,
                    *(
                        resource_id
                        for child in child_scopes
                        for resource_id in states[id(child)].lineage
                    ),
                )
            )
        )
        states[id(scope)] = _LexicalScope(
            columns=projected,
            lineage=lineage,
            relations=relation_tuple,
            scope_id=scope_ids[id(scope)],
        )
        unresolved_qualifiers[id(scope)] = frozenset(unresolved)

    for scope in scopes:
        state = states[id(scope)]
        relation_tuple = state.relations
        for column in _direct_scope_columns(scope, expression_scope_ids, exp):
            if str(column.name) == "*":
                continue
            qualifier_text = str(column.table or "").strip()
            if qualifier_text:
                qualifier = _semantic_identifier(
                    qualifier_text,
                    dialect=dialect,
                    quoted=bool(
                        getattr(column.args.get("table"), "args", {}).get(
                            "quoted", False
                        )
                    ),
                )
                relation = next(
                    (item for item in relation_tuple if item.qualifier == qualifier),
                    None,
                )
                if relation is None:
                    ancestor = scope.parent
                    ancestor_relation = None
                    while ancestor is not None and ancestor_relation is None:
                        ancestor_relation = next(
                            (
                                item
                                for item in states[id(ancestor)].relations
                                if item.qualifier == qualifier
                            ),
                            None,
                        )
                        ancestor = ancestor.parent
                    if ancestor_relation is not None:
                        if not bool(scope.can_be_correlated):
                            issues.append(
                                SqlValidationIssue(
                                    "column_scope_escape",
                                    "SQL column escapes a non-correlated lexical scope.",
                                    {"qualifier": qualifier_text},
                                )
                            )
                            continue
                        relation = ancestor_relation
                    elif qualifier in unresolved_qualifiers[id(scope)]:
                        continue
                if relation is None:
                    issues.append(
                        SqlValidationIssue(
                            "unknown_relation_qualifier",
                            "SQL references a relation qualifier not visible in this scope.",
                            {"qualifier": qualifier_text},
                        )
                    )
                    continue
                matches = _relation_column_matches(
                    relation,
                    column,
                    dialect=dialect,
                )
            else:
                matches = tuple(
                    match
                    for relation in relation_tuple
                    for match in _relation_column_matches(
                        relation,
                        column,
                        dialect=dialect,
                    )
                )
                relation = None
                if not matches:
                    if _is_legal_output_alias(
                        column,
                        scope,
                        state,
                        exp,
                        dialect=dialect,
                    ):
                        continue
                    ancestor_matches: tuple[str, ...] = ()
                    ancestor = scope.parent
                    while ancestor is not None and not ancestor_matches:
                        ancestor_matches = tuple(
                            match
                            for ancestor_relation in states[id(ancestor)].relations
                            for match in _relation_column_matches(
                                ancestor_relation,
                                column,
                                dialect=dialect,
                            )
                        )
                        ancestor = ancestor.parent
                    if ancestor_matches:
                        if not bool(scope.can_be_correlated):
                            issues.append(
                                SqlValidationIssue(
                                    "column_scope_escape",
                                    "SQL column escapes a non-correlated lexical scope.",
                                    {"column": str(column.name)},
                                )
                            )
                            continue
                        matches = ancestor_matches
            if not matches:
                if unresolved_qualifiers[id(scope)]:
                    continue
                if relation is None and len(relation_tuple) == 1:
                    relation = relation_tuple[0]
                if relation is not None and relation.kind == "base":
                    missing_resource = next(
                        (
                            item
                            for item in resources
                            if item.resource_id in relation.lineage
                        ),
                        None,
                    )
                    if missing_resource is not None:
                        issues.append(
                            _missing_column_issue(
                                SqlColumnReference(
                                    name=str(column.name),
                                    qualifier=(
                                        str(column.table) if column.table else None
                                    ),
                                    name_quoted=bool(
                                        getattr(column.this, "args", {}).get(
                                            "quoted", False
                                        )
                                    ),
                                ),
                                missing_resource,
                            )
                        )
                        continue
                issues.append(
                    SqlValidationIssue(
                        (
                            "unknown_derived_column"
                            if relation is not None and relation.kind != "base"
                            else "missing_column"
                        ),
                        "SQL references a column absent from the visible relation schema.",
                        {"column": str(column.name)},
                    )
                )
            elif len(matches) > 1:
                issues.append(
                    SqlValidationIssue(
                        "ambiguous_column",
                        "SQL column is ambiguous in its lexical scope.",
                        {"column": str(column.name)},
                    )
                )

    for scope_id in consumed_scope_ids:
        consumed_state = states.get(scope_id)
        if consumed_state is not None and any(
            not column for column in consumed_state.columns
        ):
            issues.append(
                SqlValidationIssue(
                    "derived_projection_name_required",
                    "Every derived-relation output column must have a stable name.",
                )
            )
    return tuple(issues)


def validate_sqlite_read(
    sql: str,
    *,
    source_id: str,
    resources: Iterable[ResourceSchema],
    parameters: Sequence[object] = (),
    allowed_resource_ids: Iterable[str] | None = None,
) -> SqlValidationResult:
    """Validate a single SQLite read against catalog-owned source scope."""

    return _validate_sql_read(
        sql,
        source_id=source_id,
        resources=resources,
        parameters=parameters,
        allowed_resource_ids=allowed_resource_ids,
        dialect="sqlite",
    )


def validate_postgresql_read(
    sql: str,
    *,
    source_id: str,
    resources: Iterable[ResourceSchema],
    parameters: Sequence[object] = (),
    allowed_resource_ids: Iterable[str] | None = None,
) -> SqlValidationResult:
    """Validate a single PostgreSQL read against catalog-owned source scope."""

    return _validate_sql_read(
        sql,
        source_id=source_id,
        resources=resources,
        parameters=parameters,
        allowed_resource_ids=allowed_resource_ids,
        dialect="postgresql",
    )


def _validate_sql_read(
    sql: str,
    *,
    source_id: str,
    resources: Iterable[ResourceSchema],
    parameters: Sequence[object],
    allowed_resource_ids: Iterable[str] | None,
    dialect: _SqlDialect,
) -> SqlValidationResult:
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

    source_resources = tuple(
        resource for resource in resources if resource.source_id == source_id
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
                (
                    "PostgreSQL EXPLAIN statements cannot use the bounded "
                    "tabular execution path."
                ),
            )
        )
    if dialect == "postgresql":
        function_names_not_admitted = {
            name
            for name in analysis.function_names
            if (name.removeprefix("PG_CATALOG.") not in _POSTGRESQL_BOUNDED_FUNCTIONS)
        }
        denied_functions = tuple(
            sorted(
                set(analysis.unresolved_function_names)
                | set(analysis.table_function_names)
                | function_names_not_admitted
            )
        )
    else:
        denied_functions = ()
    if denied_functions:
        issues.append(
            SqlValidationIssue(
                "function_not_allowed",
                (
                    f"{display_name} data queries allow only the declared "
                    "bounded function set."
                ),
                {"functions": denied_functions[:_MAX_CANDIDATES]},
            )
        )
    if dialect == "postgresql" and analysis.unsafe_cast_type_names:
        issues.append(
            SqlValidationIssue(
                "cast_type_not_allowed",
                "PostgreSQL casts must target a declared built-in data type.",
                {"types": analysis.unsafe_cast_type_names[:_MAX_CANDIDATES]},
            )
        )
    unsafe_operators = tuple(
        name
        for name in analysis.explicit_operator_names
        if not name.startswith("PG_CATALOG.")
    )
    if dialect == "postgresql" and unsafe_operators:
        issues.append(
            SqlValidationIssue(
                "operator_not_allowed",
                "Explicit PostgreSQL operators must resolve from pg_catalog.",
                {"operators": unsafe_operators[:_MAX_CANDIDATES]},
            )
        )
    if dialect == "postgresql" and analysis.anonymous_parameter_count:
        issues.append(
            SqlValidationIssue(
                "parameter_style_invalid",
                "PostgreSQL query parameters must use numbered $1 placeholders.",
                {"anonymous_placeholders": analysis.anonymous_parameter_count},
            )
        )
    expected_ordinals = tuple(range(1, len(parameters) + 1))
    if dialect == "postgresql" and (
        analysis.invalid_parameter_count
        or analysis.parameter_ordinals != expected_ordinals
    ):
        issues.append(
            SqlValidationIssue(
                "parameter_index_mismatch",
                "PostgreSQL parameter indexes must be contiguous and match the supplied values.",
                {
                    "expected": expected_ordinals,
                    "received": analysis.parameter_ordinals,
                    "invalid": analysis.invalid_parameter_count,
                },
            )
        )
    elif dialect == "sqlite" and analysis.positional_parameter_count != len(parameters):
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
    resolved_ids: list[str] = []
    for table in analysis.tables:
        if table.is_cte:
            continue
        if dialect == "postgresql" and len(table.qualified_parts) < 2:
            issues.append(
                SqlValidationIssue(
                    "schema_qualification_required",
                    (
                        "PostgreSQL resources must be schema-qualified so "
                        "catalog provenance and server resolution agree."
                    ),
                    {"resource": table.qualified_name},
                )
            )
        candidates = _resource_candidates(
            table,
            source_resources,
            dialect=dialect,
        )
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
        if dialect == "postgresql" and resource.resource_kind != "table":
            issues.append(
                SqlValidationIssue(
                    "resource_kind_not_allowed",
                    (
                        "PostgreSQL bounded reads require a cataloged base "
                        "table, not a view or unknown relation kind."
                    ),
                    {
                        "resource_id": resource.resource_id,
                        "resource_kind": resource.resource_kind or "unknown",
                    },
                )
            )
        if resource.resource_id not in resolved_ids:
            resolved_ids.append(resource.resource_id)

    if not any(not table.is_cte for table in analysis.tables):
        issues.append(
            SqlValidationIssue(
                "resource_scope_empty",
                f"{display_name} data queries must reference a cataloged resource.",
            )
        )

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
    issues.extend(
        _lexical_column_issues(
            analysis.sql,
            resources=source_resources,
            allowed_resource_ids=allowed,
            dialect=dialect,
        )
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
    table: SqlTableReference,
    resources: tuple[ResourceSchema, ...],
    *,
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
        return True
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
