"""Relational evidence projection from CatalogPlugin into SQL planner shapes.

This module does not discover schemas, search local schema dicts, or scan graph
state directly. It adapts generic catalog evidence into the SQL-native evidence
contract consumed by ``from_db`` planning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils import unique_preserving_order

MAX_EVIDENCE_ITEMS = 8


async def collect_query_evidence(
    prompt: str,
    intent_args: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    run_state: Any = None,
    catalog: Any = None,
    store_id: Optional[str] = None,
    graph_backend: Any = None,
) -> Dict[str, Any]:
    """Collect compact relational evidence for SQL planning."""
    run_evidence = _run_state_evidence(run_state)
    if evidence_join_paths(run_evidence):
        return run_evidence

    if catalog is None or not store_id:
        return run_evidence if _has_evidence(run_evidence) else _empty_evidence()

    text = _search_text(prompt, intent_args)
    payload = catalog.collect_evidence(
        store_id,
        text,
        intent_args,
        asset_types=["table", "view", "collection"],
        limit=MAX_EVIDENCE_ITEMS,
    )
    return _merge_evidence(run_evidence, _catalog_evidence(payload))


def evidence_table_names(evidence: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(evidence, dict):
        return []
    return unique_preserving_order(
        str(item.get("table") or item.get("name") or "").strip()
        for item in evidence.get("tables", []) or []
        if isinstance(item, dict)
        and str(item.get("table") or item.get("name") or "").strip()
    )


def evidence_join_paths(evidence: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(evidence, dict):
        return []
    return [item for item in evidence.get("joins", []) or [] if isinstance(item, dict)]


def compact_evidence_summary(
    evidence: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(evidence, dict) or not evidence:
        return None
    sources = evidence.get("sources") or []
    if not sources:
        return None
    tables = evidence_table_names(evidence)
    joins = evidence_join_paths(evidence)
    columns = [
        item for item in evidence.get("columns", []) or [] if isinstance(item, dict)
    ][:MAX_EVIDENCE_ITEMS]
    return {
        "sources": sources,
        "table_count": len(tables),
        "column_count": len(columns),
        "join_count": len(joins),
        "confidence": evidence.get("confidence", 0.0),
        "tables": tables[:MAX_EVIDENCE_ITEMS],
    }


def _catalog_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
    tables = []
    columns = []
    for asset in payload.get("assets", []) or []:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or asset.get("asset_ref") or "").strip()
        if not name:
            continue
        tables.append(
            {
                "table": name,
                "confidence": min(0.9, float(asset.get("score") or 0) / 10),
                "source": "catalog",
                "provenance": "catalog_search",
            }
        )
        for field in asset.get("matched_fields") or []:
            field_name = str(field.get("name") or "").strip()
            if field_name:
                columns.append(
                    {
                        "table": name,
                        "column": field_name,
                        "confidence": min(0.85, float(field.get("score") or 0) / 10),
                        "source": "catalog",
                        "provenance": "catalog_search",
                    }
                )

    joins = [
        {**path, "source": "catalog", "provenance": "catalog_relationship_paths"}
        for path in payload.get("relationships", []) or []
        if isinstance(path, dict)
    ]
    return {
        "tables": _dedupe_evidence(tables, key="table")[:MAX_EVIDENCE_ITEMS],
        "columns": _dedupe_evidence(columns, key=("table", "column"))[
            :MAX_EVIDENCE_ITEMS
        ],
        "joins": joins[:MAX_EVIDENCE_ITEMS],
        "sources": ["catalog"],
        "confidence": _evidence_confidence(tables + columns + joins),
    }


def _empty_evidence() -> Dict[str, Any]:
    return {
        "tables": [],
        "columns": [],
        "joins": [],
        "sources": [],
        "confidence": 0.0,
    }


def _run_state_evidence(run_state: Any) -> Dict[str, Any]:
    if run_state is None:
        return _empty_evidence()
    getter = getattr(run_state, "catalog_planner_evidence", None)
    if not callable(getter):
        return _empty_evidence()
    evidence = getter()
    return evidence if isinstance(evidence, dict) else _empty_evidence()


def _has_evidence(evidence: Dict[str, Any]) -> bool:
    return bool(evidence_join_paths(evidence) or evidence_table_names(evidence))


def _merge_evidence(
    primary: Dict[str, Any], fallback: Dict[str, Any]
) -> Dict[str, Any]:
    if not _has_evidence(primary):
        return fallback
    tables = _dedupe_evidence(
        list(primary.get("tables") or []) + list(fallback.get("tables") or []),
        key="table",
    )
    columns = _dedupe_evidence(
        list(primary.get("columns") or []) + list(fallback.get("columns") or []),
        key=("table", "column"),
    )
    joins = _dedupe_joins(
        list(primary.get("joins") or []) + list(fallback.get("joins") or [])
    )
    sources = unique_preserving_order(
        list(primary.get("sources") or []) + list(fallback.get("sources") or [])
    )
    return {
        "tables": tables[:MAX_EVIDENCE_ITEMS],
        "columns": columns[:MAX_EVIDENCE_ITEMS],
        "joins": joins[:MAX_EVIDENCE_ITEMS],
        "sources": sources,
        "confidence": _evidence_confidence(tables + columns + joins),
    }


def _dedupe_joins(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Any, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (
            tuple(str(value).lower() for value in item.get("from_tables") or []),
            tuple(str(value).lower() for value in item.get("to_tables") or []),
            bool(item.get("reachable")),
        )
        if key in best and _score(item) <= _score(best[key]):
            continue
        best[key] = item
    return sorted(best.values(), key=lambda item: (-_score(item), str(item)))


def _search_text(prompt: str, intent_args: Dict[str, Any]) -> str:
    parts = [prompt]
    for key in (
        "goal",
        "required_fields",
        "candidate_tables",
        "filters",
        "aggregations",
        "grouping",
    ):
        value = intent_args.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(part for part in parts if part).strip()


def _dedupe_evidence(items: List[Dict[str, Any]], *, key: Any) -> List[Dict[str, Any]]:
    best: Dict[Any, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if isinstance(key, tuple):
            item_key = tuple(str(item.get(part) or "").lower() for part in key)
        else:
            item_key = str(item.get(key) or "").lower()
        if not item_key or item_key in best and _score(item) <= _score(best[item_key]):
            continue
        best[item_key] = item
    return sorted(best.values(), key=lambda item: (-_score(item), str(item)))


def _evidence_confidence(items: List[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return round(min(1.0, max(_score(item) for item in items)), 3)


def _score(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
