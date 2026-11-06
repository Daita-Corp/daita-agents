"""Advisory checks on a successful read: fan-out, repeated keys, empty results.

Pure analysis of SQL text, key facts and returned rows. Every public function
returns an empty result when unsure and never raises, so a check can never
change a correct read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .duckdb_read import _conjuncts, _load_sqlglot, select_equalities

_MAX_VALUE_CHARACTERS = 40
_PROBE_VALUES = 10


def fan_out_warnings(
    sql: str, dialect: str, keys: Mapping[str, Sequence[Sequence[str]]]
) -> tuple[str, ...]:
    """Warn when SUM, AVG or COUNT(column) aggregates rows a 1:N join repeats.

    ``keys`` maps a table's catalog name to its unique column sets. A table with
    no listed keys is unknown (it may be a view), never a reason to warn.
    """

    try:
        return _fan_out(sql, dialect, keys)
    except Exception:  # noqa: BLE001 - advice must not fail a valid read
        return ()


def duplicate_key_warnings(
    sql: str,
    dialect: str,
    keys: Mapping[str, Sequence[Sequence[str]]],
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Warn when a joined table's primary key repeats in plain (non-aggregate) rows."""

    try:
        return _duplicate_keys(sql, dialect, keys, columns, rows)
    except Exception:  # noqa: BLE001
        return ()


def empty_result_probe(sql: str, dialect: str) -> tuple[str, str] | None:
    """One bounded statement that explains an empty result, as (kind, sql).

    A literal range filter on a column asks for the column's min and max
    ("range"); a literal equality or IN asks for its distinct values ("values").
    """

    try:
        return _probe(sql, dialect)
    except Exception:  # noqa: BLE001
        return None


def empty_result_warning(
    kind: str, derived_sql: str, rows: Sequence[Mapping[str, object]]
) -> str | None:
    """Phrase the probe's answer; None when it says nothing useful."""

    try:
        sqlglot, exp, _ = _load_sqlglot()
        derived = sqlglot.parse_one(derived_sql, read="duckdb")
        column = derived.find(exp.Column)
        table = derived.find(exp.Table)
        label = f"{table.sql()}.{column.name}"
        if kind == "range":
            low, high = (_text(rows[0][key]) for key in ("lowest", "highest"))
            if low == "NULL" or high == "NULL":
                return f"No rows matched. {label} has no values in the table."
            return f"No rows matched. {label} spans {low} to {high} in the table."
        values = [_text(next(iter(row.values()))) for row in rows]
        if not values:
            return f"No rows matched. {label} has no values in the table."
        more = ", ..." if len(values) >= _PROBE_VALUES else ""
        return f"No rows matched. {label} has: {', '.join(values)}{more}"
    except Exception:  # noqa: BLE001
        return None


def _text(value: object) -> str:
    text = "NULL" if value is None else str(value)
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    if len(text) > _MAX_VALUE_CHARACTERS:
        text = text[: _MAX_VALUE_CHARACTERS - 3] + "..."
    return text


def _glot(dialect: str) -> str:
    return "postgres" if dialect.startswith("postgres") else dialect


def _parse(sql: str, dialect: str) -> tuple[Any, Any]:
    sqlglot, exp, _ = _load_sqlglot()
    return sqlglot.parse_one(sql, read=_glot(dialect)), exp


def _table_name(table: Any) -> str:
    return ".".join(part for part in (table.db, table.name) if part)


def _lookup_keys(
    name: str, keys: Mapping[str, Sequence[Sequence[str]]]
) -> list[set[str]] | None:
    folded = {k.casefold(): v for k, v in keys.items()}
    found = folded.get(name.casefold())
    if found is None and "." not in name:
        bare = [v for k, v in folded.items() if k.rsplit(".", 1)[-1] == name.casefold()]
        found = bare[0] if len(bare) == 1 else None
    if not found:
        return None
    return [{c.casefold() for c in key} for key in found]


def _owner(column: Any, names: list[str]) -> int | None:
    if not column.table:
        return None
    matches = [
        i for i, n in enumerate(names) if n.casefold() == column.table.casefold()
    ]
    return matches[0] if len(matches) == 1 else None


def _base_table(item: Any, ctes: Mapping[str, Any], exp: Any) -> Any:
    """The base table behind a plain pass-through CTE or subquery, else the item."""

    inner = item.this if isinstance(item, exp.Subquery) else None
    if isinstance(item, exp.Table) and not item.db:
        inner = ctes.get(item.name.casefold())
    if (
        not isinstance(inner, exp.Select)
        or inner.args.get("joins")
        or inner.args.get("group")
        or inner.args.get("distinct")
        or inner.args.get("limit")
        or inner.find(exp.AggFunc, exp.Alias, exp.Window)
    ):
        return item
    source = inner.args.get("from_") or inner.args.get("from")
    table = source.this if source is not None else None
    return table if isinstance(table, exp.Table) and table.name else item


def _fan_out(
    sql: str, dialect: str, keys: Mapping[str, Sequence[Sequence[str]]]
) -> tuple[str, ...]:
    root, exp = _parse(sql, dialect)
    ctes = {c.alias_or_name.casefold(): c.this for c in root.find_all(exp.CTE)}
    groups: dict[int, tuple[Any, list[Any], list[str], dict[Any, Any]]] = {}
    for select, items, names, eq, li, ri in select_equalities(root, exp, {}):
        items = [_base_table(item, ctes, exp) for item in items]
        edges = groups.setdefault(id(select), (select, items, names, {}))[3]
        pair = edges.setdefault(frozenset((li, ri)), {li: set(), ri: set()})
        for index, side in ((li, eq.this), (ri, eq.expression)):
            pair[index].update(c.name.casefold() for c in side.find_all(exp.Column))
    warnings: list[str] = []
    for select, items, names, edges in groups.values():
        for agg in select.find_all(exp.Sum, exp.Avg, exp.Count):
            if agg.find_ancestor(exp.Select) is not select:
                continue
            if isinstance(agg.this, (exp.Distinct, exp.Star)):
                continue
            owners = {_owner(c, names) for c in agg.find_all(exp.Column)}
            if len(owners) != 1 or None in owners:
                continue
            start = owners.pop()
            assert start is not None
            hit = _multiplier(start, items, edges, keys)
            if hit is not None:
                far, columns = hit
                near = _table_name(items[start])
                warnings.append(
                    f"{agg.sql(dialect=_glot(dialect)).lower()} repeats each {near} row "
                    f"once per matching {far} row ({', '.join(sorted(columns))} "
                    "is not unique there), so the total may be inflated. "
                    "Aggregate before joining or use EXISTS."
                )
    return tuple(dict.fromkeys(warnings))


def _multiplier(
    start: int,
    items: list[Any],
    edges: dict[Any, Any],
    keys: Mapping[str, Sequence[Sequence[str]]],
) -> tuple[str, set[str]] | None:
    seen = {start}
    queue = [start]
    while queue:
        here = queue.pop(0)
        for pair, columns in edges.items():
            if here not in pair:
                continue
            (far,) = pair - {here}
            if far in seen:
                continue
            seen.add(far)
            queue.append(far)
            item = items[far]
            if not hasattr(item, "name") or item.key != "table":
                continue
            known = _lookup_keys(_table_name(item), keys)
            if known is not None and not any(key <= columns[far] for key in known):
                return _table_name(item), columns[far]
    return None


def _duplicate_keys(
    sql: str,
    dialect: str,
    keys: Mapping[str, Sequence[Sequence[str]]],
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    root, exp = _parse(sql, dialect)
    if not isinstance(root, exp.Select) or len(rows) < 2:
        return ()
    if (
        not root.args.get("joins")
        or root.args.get("group")
        or root.args.get("distinct")
        or root.find(exp.AggFunc)
    ):
        return ()
    items = [root.args["from_" if "from_" in root.args else "from"].this]
    items += [j.this for j in root.args["joins"]]
    names = [i.alias_or_name for i in items]
    warnings: list[str] = []
    for projection in root.expressions:
        column = (
            projection.unalias() if isinstance(projection, exp.Alias) else projection
        )
        if not isinstance(column, exp.Column):
            continue
        owner = _owner(column, names)
        if owner is None or items[owner].key != "table":
            continue
        known = _lookup_keys(_table_name(items[owner]), keys)
        if known is None or known[0] != {column.name.casefold()}:
            continue
        output = projection.alias_or_name
        if columns.count(output) != 1:
            continue
        values = [repr(row[output]) for row in rows if row.get(output) is not None]
        repeated = len(values) - len(set(values))
        if repeated:
            warnings.append(
                f"{output} is the primary key of {_table_name(items[owner])} but "
                f"{repeated} of {len(rows)} returned rows repeat it, so the join "
                "returns that table's rows more than once."
            )
    return tuple(warnings)


def _is_literal(node: Any, exp: Any) -> bool:
    if isinstance(node, (exp.Cast, exp.Neg, exp.Paren)):
        return _is_literal(node.this, exp)
    return isinstance(node, exp.Literal)


def _filter(node: Any, exp: Any) -> tuple[Any, str] | None:
    """The (column, kind) of a literal comparison; kind is range or values."""

    if isinstance(node, (exp.LT, exp.LTE, exp.GT, exp.GTE, exp.EQ)):
        kind = "values" if isinstance(node, exp.EQ) else "range"
        for column, other in (
            (node.this, node.expression),
            (node.expression, node.this),
        ):
            if isinstance(column, exp.Column) and _is_literal(other, exp):
                return column, kind
    elif isinstance(node, exp.Between):
        bounds = (node.args.get("low"), node.args.get("high"))
        if isinstance(node.this, exp.Column) and all(
            _is_literal(b, exp) for b in bounds
        ):
            return node.this, "range"
    elif isinstance(node, exp.In) and node.expressions and not node.args.get("query"):
        if isinstance(node.this, exp.Column) and all(
            _is_literal(e, exp) for e in node.expressions
        ):
            return node.this, "values"
    return None


def _probe(sql: str, dialect: str) -> tuple[str, str] | None:
    root, exp = _parse(sql, dialect)
    where = root.args.get("where") if isinstance(root, exp.Select) else None
    source = root.args.get("from_") or root.args.get("from")
    if (
        where is None
        or source is None
        or root.args.get("with_")
        or root.args.get("with")
    ):
        return None
    items = [source.this, *(j.this for j in root.args.get("joins") or [])]
    names = [i.alias_or_name for i in items]
    found: dict[str, tuple[Any, Any]] = {}
    for node in _conjuncts(where.this, exp):
        hit = _filter(node, exp)
        if hit is None:
            continue
        column, kind = hit
        index = (
            _owner(column, names) if column.table else (0 if len(items) == 1 else None)
        )
        if index is not None and items[index].key == "table":
            found.setdefault(kind, (column, items[index]))
    for kind in ("range", "values"):
        if kind in found:
            column, table = found[kind]
            table = table.copy()
            table.set("alias", None)
            col = exp.Column(this=column.this.copy())
            if kind == "range":
                query = exp.select(
                    exp.alias_(exp.Min(this=col), "lowest"),
                    exp.alias_(exp.Max(this=col.copy()), "highest"),
                )
            else:
                query = exp.select(col).distinct().order_by("1").limit(_PROBE_VALUES)
            return kind, query.from_(table).sql(dialect=_glot(dialect))
    return None
