"""
Compact DB tool results before they are appended to LLM conversation context.
"""

from __future__ import annotations

import fnmatch
import json
from typing import Any, Dict, List, Tuple

from ..config.policies import ToolResultPolicy


def compact_tool_result_for_context(
    tool_name: str,
    result: Any,
    *,
    policy: ToolResultPolicy | None = None,
) -> Any:
    policy = policy or ToolResultPolicy()
    if not _is_db_tool_name(tool_name):
        return result
    if tool_name == "db_find_join_path":
        return _compact_join_path_result(result, policy)
    compacted = _compact_value(result, policy=policy)
    if _estimate_tokens(compacted) <= policy.max_result_tokens:
        return compacted
    return _token_trim(compacted, policy)


def _is_db_tool_name(tool_name: str) -> bool:
    return (
        tool_name.startswith("db_")
        or tool_name.startswith("postgres_")
        or tool_name.startswith("mysql_")
        or tool_name.startswith("sqlite_")
        or tool_name.startswith("snowflake_")
        or tool_name.startswith("mongodb_")
    )


def _compact_value(value: Any, *, policy: ToolResultPolicy) -> Any:
    if isinstance(value, dict):
        if "rows" in value and isinstance(value.get("rows"), list):
            return _compact_rows_payload(value, "rows", policy)
        if "documents" in value and isinstance(value.get("documents"), list):
            return _compact_rows_payload(value, "documents", policy)
        if "results" in value and isinstance(value.get("results"), list):
            return _compact_rows_payload(value, "results", policy)
        return {key: _compact_value(val, policy=policy) for key, val in value.items()}
    if isinstance(value, list):
        rows, omitted = _compact_rows(value, policy)
        payload = {
            "returned_rows": len(rows),
            "rows_preview": rows,
            "truncated": len(value) > len(rows) or bool(omitted),
        }
        if omitted:
            payload["omitted_columns"] = sorted(omitted)
        return payload
    if isinstance(value, str):
        return _compact_scalar(value, policy)
    return value


def _compact_join_path_result(result: Any, policy: ToolResultPolicy) -> Any:
    if not isinstance(result, dict):
        return result

    paths = result.get("paths")
    if not isinstance(paths, list):
        return _compact_value(result, policy=policy)

    compact_paths = []
    for path in paths[:5]:
        if not isinstance(path, dict):
            compact_paths.append(path)
            continue
        joins = path.get("joins")
        compact_paths.append(
            {
                key: path.get(key)
                for key in (
                    "tables",
                    "joins",
                    "predicate",
                    "join_predicate",
                    "confidence",
                    "warnings",
                    "hop_count",
                )
                if path.get(key) is not None
            }
        )
        if isinstance(joins, list):
            compact_paths[-1]["joins"] = [
                {
                    join_key: join.get(join_key)
                    for join_key in (
                        "left_table",
                        "left_column",
                        "right_table",
                        "right_column",
                        "predicate",
                    )
                    if isinstance(join, dict) and join.get(join_key) is not None
                }
                for join in joins
            ]

    compacted = {
        key: result.get(key)
        for key in ("success", "from_tables", "to_tables", "path_count", "warnings")
        if result.get(key) is not None
    }
    compacted["paths"] = compact_paths
    compacted["truncated"] = len(paths) > len(compact_paths)
    return compacted


def _compact_rows_payload(
    payload: Dict[str, Any], row_key: str, policy: ToolResultPolicy
) -> Dict[str, Any]:
    rows = payload.get(row_key) or []
    compact_rows, omitted_columns = _compact_rows(rows, policy)
    out = {
        key: _compact_value(val, policy=policy)
        for key, val in payload.items()
        if key != row_key
    }
    columns = _columns_from_rows(compact_rows)
    out["row_count"] = payload.get("total_rows", len(rows))
    out["returned_rows"] = len(compact_rows)
    out["columns"] = columns
    out["rows_preview"] = compact_rows
    out["truncated"] = bool(payload.get("truncated")) or len(rows) > len(compact_rows)
    if omitted_columns:
        out["omitted_columns"] = sorted(omitted_columns)
        out["truncated"] = True
    return out


def _compact_rows(
    rows: List[Any], policy: ToolResultPolicy
) -> Tuple[List[Any], set[str]]:
    omitted_columns: set[str] = set()
    compacted = []
    for row in rows[: policy.max_rows_inline]:
        if isinstance(row, dict):
            compacted_row = {}
            for key, value in row.items():
                if _should_omit_column(str(key), policy):
                    omitted_columns.add(str(key))
                    continue
                compacted_row[key] = _compact_cell(value, policy)
            compacted.append(compacted_row)
        else:
            compacted.append(_compact_cell(row, policy))
    return compacted, omitted_columns


def _compact_cell(value: Any, policy: ToolResultPolicy) -> Any:
    if isinstance(value, str):
        return _compact_scalar(value, policy)
    if isinstance(value, (dict, list)) and policy.summarize_large_json:
        serialized = _safe_json(value)
        if len(serialized) > policy.max_cell_chars:
            if isinstance(value, dict):
                return {
                    "type": "dict",
                    "keys": sorted(str(key) for key in value.keys())[:20],
                    "truncated": True,
                }
            return {"type": "list", "length": len(value), "truncated": True}
    return value


def _compact_scalar(value: str, policy: ToolResultPolicy) -> str:
    if len(value) <= policy.max_cell_chars:
        return value
    return value[: policy.max_cell_chars - 3].rstrip() + "..."


def _should_omit_column(column: str, policy: ToolResultPolicy) -> bool:
    return any(
        fnmatch.fnmatch(column.lower(), pattern.lower())
        for pattern in policy.omitted_column_patterns
    )


def _columns_from_rows(rows: List[Any]) -> List[str]:
    columns: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            if key not in columns:
                columns.append(str(key))
    return columns


def _token_trim(value: Any, policy: ToolResultPolicy) -> Any:
    if not isinstance(value, dict) or "rows_preview" not in value:
        return {
            "summary": _compact_scalar(_safe_json(value), policy),
            "truncated": True,
        }
    rows = value.get("rows_preview") or []
    trimmed = dict(value)
    while rows and _estimate_tokens(trimmed) > policy.max_result_tokens:
        rows = rows[:-1]
        trimmed["rows_preview"] = rows
        trimmed["returned_rows"] = len(rows)
        trimmed["truncated"] = True
    if _estimate_tokens(trimmed) > policy.max_result_tokens:
        trimmed["rows_preview"] = []
        trimmed["returned_rows"] = 0
        trimmed["truncated"] = True
    return trimmed


def _estimate_tokens(value: Any) -> int:
    return max(1, (len(_safe_json(value)) + 3) // 4)


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        return str(value)
