"""
Compact schema and data-health summary for agents created by ``from_db()``.
"""

from typing import Any, Dict, List

from .schema import is_numeric_type

_TIMESTAMP_HINTS = (
    "created_at",
    "updated_at",
    "modified_at",
    "event_time",
    "event_at",
    "timestamp",
    "date",
)
_MONEY_HINTS = ("amount", "price", "total", "revenue", "cost", "fee", "balance")
_QUANTITY_HINTS = ("qty", "quantity", "count", "units", "volume")
_ENTITY_HINTS = ("customer", "user", "account", "product", "employee", "vendor")
_FACT_HINTS = ("order", "transaction", "event", "payment", "invoice", "session")


def build_db_summary(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact, JSON-serializable DB summary without querying the DB."""
    tables = schema.get("tables", []) or []
    foreign_keys = schema.get("foreign_keys", []) or []

    important_tables = _important_tables(tables)
    timestamp_columns = _matching_columns(tables, _TIMESTAMP_HINTS)
    money_columns = _matching_columns(tables, _MONEY_HINTS)
    quantity_columns = _matching_columns(tables, _QUANTITY_HINTS)
    fact_tables = _classify_tables(tables, _FACT_HINTS, require_numeric=True)
    entity_tables = _classify_tables(tables, _ENTITY_HINTS, require_pk=True)
    suspicious = _suspicious_signals(tables, foreign_keys, timestamp_columns)
    candidate_metrics = _candidate_metrics(money_columns, quantity_columns)

    return {
        "table_count": schema.get("table_count", len(tables)),
        "column_count": sum(len(t.get("columns", []) or []) for t in tables),
        "relationship_count": len(foreign_keys),
        "important_tables": important_tables[:10],
        "fact_tables": fact_tables[:10],
        "entity_tables": entity_tables[:10],
        "timestamp_columns": timestamp_columns[:20],
        "money_columns": money_columns[:20],
        "quantity_columns": quantity_columns[:20],
        "candidate_metrics": candidate_metrics[:12],
        "signals": suspicious[:20],
        "suggested_questions": suggested_questions(
            important_tables=important_tables,
            fact_tables=fact_tables,
            entity_tables=entity_tables,
            candidate_metrics=candidate_metrics,
            timestamp_columns=timestamp_columns,
        ),
    }


def suggested_questions(
    *,
    important_tables: List[str],
    fact_tables: List[str],
    entity_tables: List[str],
    candidate_metrics: List[Dict[str, Any]],
    timestamp_columns: List[Dict[str, str]],
) -> List[str]:
    """Generate starter questions from compact schema signals."""
    questions: List[str] = []
    if candidate_metrics:
        metric = candidate_metrics[0]
        questions.append(f"How has {metric['name']} trended over time?")
        if entity_tables:
            questions.append(
                f"Which {entity_tables[0]} records contribute most to {metric['name']}?"
            )
    if fact_tables:
        questions.append(f"What changed recently in {fact_tables[0]}?")
        questions.append(f"Are there anomalies in {fact_tables[0]}?")
    if timestamp_columns:
        first = timestamp_columns[0]
        questions.append(
            f"Is {first['table']}.{first['column']} fresh compared with expectations?"
        )
    if important_tables:
        questions.append(
            f"What are the most important patterns in {important_tables[0]}?"
        )
    return _dedupe(questions)[:6]


def _important_tables(tables: List[Dict[str, Any]]) -> List[str]:
    scored = []
    for table in tables:
        name = table.get("name", "")
        row_count = table.get("row_count")
        score = len(table.get("columns", []) or [])
        if isinstance(row_count, int):
            score += min(row_count, 1_000_000) / 100_000
        lowered = name.lower()
        if any(h in lowered for h in _FACT_HINTS + _ENTITY_HINTS):
            score += 5
        scored.append((score, name))
    return [name for _, name in sorted(scored, reverse=True) if name]


def _matching_columns(
    tables: List[Dict[str, Any]], hints: tuple[str, ...]
) -> List[Dict[str, str]]:
    matches = []
    for table in tables:
        table_name = table.get("name", "")
        for col in table.get("columns", []) or []:
            name = col.get("name", "")
            lowered = name.lower()
            if any(h in lowered for h in hints):
                matches.append(
                    {"table": table_name, "column": name, "type": col.get("type", "")}
                )
    return matches


def _classify_tables(
    tables: List[Dict[str, Any]],
    hints: tuple[str, ...],
    *,
    require_numeric: bool = False,
    require_pk: bool = False,
) -> List[str]:
    out = []
    for table in tables:
        name = table.get("name", "")
        columns = table.get("columns", []) or []
        lowered = name.lower()
        if not any(h in lowered for h in hints):
            continue
        if require_numeric and not any(
            is_numeric_type(c.get("type", "")) for c in columns
        ):
            continue
        if require_pk and not any(c.get("is_primary_key") for c in columns):
            continue
        out.append(name)
    return out


def _suspicious_signals(
    tables: List[Dict[str, Any]],
    foreign_keys: List[Dict[str, Any]],
    timestamp_columns: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    timestamp_tables = {c["table"] for c in timestamp_columns}
    has_relationships = bool(foreign_keys)
    signals: List[Dict[str, Any]] = []
    for table in tables:
        name = table.get("name", "")
        row_count = table.get("row_count")
        columns = table.get("columns", []) or []
        if row_count == 0:
            signals.append({"type": "empty_table", "table": name})
        if name not in timestamp_tables and any(h in name.lower() for h in _FACT_HINTS):
            signals.append({"type": "missing_timestamp", "table": name})
        if not has_relationships and _has_fk_like_column(columns):
            signals.append({"type": "missing_relationships", "table": name})
        for col in columns:
            if col.get("nullable") is False:
                continue
            if col.get("is_primary_key"):
                signals.append(
                    {
                        "type": "nullable_primary_key",
                        "table": name,
                        "column": col.get("name", ""),
                    }
                )
    return signals


def _has_fk_like_column(columns: List[Dict[str, Any]]) -> bool:
    for col in columns:
        name = (col.get("name") or "").lower()
        if name.endswith("_id") and not col.get("is_primary_key"):
            return True
    return False


def _candidate_metrics(
    money_columns: List[Dict[str, str]], quantity_columns: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    metrics = []
    for col in money_columns:
        metrics.append(
            {
                "name": f"{col['table']}_{col['column']}_sum",
                "type": "sum",
                "table": col["table"],
                "column": col["column"],
                "semantic_type": "money",
            }
        )
    for col in quantity_columns:
        metrics.append(
            {
                "name": f"{col['table']}_{col['column']}_sum",
                "type": "sum",
                "table": col["table"],
                "column": col["column"],
                "semantic_type": "quantity",
            }
        )
    return metrics


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
